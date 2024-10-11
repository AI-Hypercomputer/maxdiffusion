"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
import functools
import math
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset
import jax
import jax.numpy as jnp
from transformers import CLIPTokenizer

from maxdiffusion import multihost_dataloading

AUTOTUNE = tf.data.experimental.AUTOTUNE

def vae_apply(images, sample_rng, vae, vae_params):
  vae_outputs = vae.apply(
    {"params" : vae_params}, images,
      deterministic=True, method=vae.encode
  )
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor

  return latents

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

# TODO - https://github.com/google/array_record/blob/main/beam/examples/example_gcs_conversion.py
def make_laion400m_train_iterator(
    config,
    mesh,
    global_batch_size,
):
  """Iterator for Laion dataset.
  To see how to prepare this dataset, look at
  maxdiffusion/pedagogical_examples/to_tfrecords.py
  """
  feature_description = {
    "moments" : tf.io.FixedLenFeature([], tf.string),
    "clip_embeddings" : tf.io.FixedLenFeature([], tf.string)
  }

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, feature_description)

  def prepare_sample(features):
    moments = tf.io.parse_tensor(tnp.asarray(features["moments"]), out_type=tf.float32)
    captions = tf.io.parse_tensor(tnp.asarray(features["clip_embeddings"]), out_type=tf.float32)
    return (moments, captions)
  
  def tokenize(moments, captions, tokenizer):
    captions = captions.numpy().decode("utf-8")
    input_ids = tokenizer(captions,
      max_length=tokenizer.model_max_length,
      padding="max_length",
      truncation=True
    )["input_ids"]
    return (moments, input_ids)

  def create_dict(moments, input_ids):
    return {"moments" : moments, "clip_embeddings" : input_ids}

  tokenizer = CLIPTokenizer.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="tokenizer"
  )

  partial_tokenize = functools.partial(tokenize, tokenizer=tokenizer)

  num_thread=32
  train_ds = (
    tf.data.Dataset.list_files(os.path.join(config.train_data_dir,"*"), shuffle=True, seed=config.seed)
      .shard(num_shards = jax.process_count(), index = jax.process_index())
      .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
      .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
      .map(prepare_sample, num_parallel_calls=AUTOTUNE)
      .map(create_dict, num_parallel_calls=AUTOTUNE)
      .shuffle(4 * global_batch_size // // jax.process_count() , seed=config.seed)
      .batch(global_batch_size // jax.process_count(), drop_remainder=False)
      .repeat(-1)
      .prefetch(AUTOTUNE)
  )

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

#def make_coco_eval_iterator(
#    config,
#    mesh,
#    global_batch_size,
#  ):
#  blah blah.

def make_pokemon_train_iterator(
    config,
    mesh,
    global_batch_size,
    pipeline,
    params,
    init_rng):
  train_ds = load_dataset("lambdalabs/pokemon-blip-captions",split="train")

  captions_column = config.caption_column
  image_column = config.image_column
  image_resolution = config.resolution
  cache_latents_text_encoder_outputs = config.cache_latents_text_encoder_outputs

  def tokenize_captions(examples):
    captions = list(examples[captions_column])
    text_inputs = pipeline.tokenizer(captions,
                                     max_length=pipeline.tokenizer.model_max_length,
                                     padding="max_length",
                                     truncation=True
                                    )

    if cache_latents_text_encoder_outputs:
      p_encode = jax.jit(functools.partial(encode,
                                           text_encoder=pipeline.text_encoder,
                                           text_encoder_params=params["text_encoder"]))
      encoder_hidden_states = p_encode(np.stack(text_inputs.input_ids))
      examples["input_ids"] = encoder_hidden_states
    else:
      examples["input_ids"] = text_inputs.input_ids
    return examples

  train_ds = train_ds.map(
    function=tokenize_captions,
    batched=True,
    remove_columns=[captions_column],
    num_proc=1 if cache_latents_text_encoder_outputs else 4,
    desc="Running tokenizer on train dataset"
  )
  # need to do it before load_as_tf_dataset
  # since raw images are different sizes
  # will break from_tensor_slices
  def transform_images(examples, rng, global_batch_size):
    images = list(examples[image_column])
    images = [np.asarray(image) for image in images]
    tensor_list = []
    for image in images:
      image = tf.image.resize(image, [image_resolution, image_resolution], method="bilinear", antialias=True)
      image = image / 255.0
      image = (image - 0.5) / 0.5
      image = tf.transpose(image, perm=[2,0,1])
      tensor_list.append(image)
    if cache_latents_text_encoder_outputs:
      tensor_list = np.stack(tensor_list)
      p_vae_apply = jax.jit(functools.partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
      ds_length = tensor_list.shape[0]
      iters = ds_length // global_batch_size
      latents_list = []
      for i in range(0, iters * global_batch_size, global_batch_size):
        sample_rng, rng = jax.random.split(rng)
        latents = p_vae_apply(tensor_list[i:i+global_batch_size], sample_rng)
        latents_list.append(latents)

      latents_list = np.stack(latents_list)
      b1, b2, c, l1, l2 = latents_list.shape
      latents_list = np.reshape(latents_list, (b1*b2,c, l1, l2))

      # TODO (Juan Acevedo): do last iteration, its required for the Pyarrow dataset
      # to not break due to items being fewer than expected. Is there a better way?
      sample_rng, rng = jax.random.split(rng)
      latents = p_vae_apply(tensor_list[i+global_batch_size:], sample_rng)

      examples["pixel_values"] = np.append(latents_list, latents, axis=0)
    else:
      examples["pixel_values"] = tf.stack(tensor_list)

    return examples

  p_transform_images = functools.partial(transform_images, rng=init_rng, global_batch_size=global_batch_size)

  train_ds = train_ds.map(
    function=p_transform_images,
    batched=True,
    remove_columns=[image_column],
    num_proc=1 if cache_latents_text_encoder_outputs else config.transform_images_num_proc,
    desc="Transforming images"
  )

  # taken from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/examples/tensorflow/contrastive-image-text/run_clip.py#L225
  def load_as_tf_dataset(dataset, batch_size, shuffle):
    dataset = dataset.with_format("tensorflow")[:]
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if shuffle:
      tf_dataset = tf_dataset.shuffle(len(tf_dataset), seed=config.seed)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    repeats = math.ceil((config.max_train_steps * batch_size) / len(tf_dataset))
    tf_dataset = tf_dataset.repeat(repeats)

    return tf_dataset

  train_ds = load_as_tf_dataset(
    train_ds, global_batch_size, True
  )
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

def get_shaped_batch(config, pipeline):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""
  vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
  total_train_batch_size = config.per_device_batch_size * jax.device_count()
  batch_image_shape = (total_train_batch_size, 
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor, 8)
  #bs, encoder_input, seq_length
  batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings, pipeline.text_encoder.config.hidden_size)
  shaped_batch = {}
  shaped_batch["moments"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
  shaped_batch["clip_embeddings"] = jax.ShapeDtypeStruct(batch_ids_shape, jnp.float32)
  return shaped_batch