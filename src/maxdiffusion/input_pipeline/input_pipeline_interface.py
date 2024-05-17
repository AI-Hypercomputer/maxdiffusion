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
import math
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset
import jax
import jax.numpy as jnp

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
    "latents" : tf.io.FixedLenFeature([], tf.string),
    "hidden_states" : tf.io.FixedLenFeature([], tf.string)
  }

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, feature_description)

  def prepare_sample(features):
    latents = tf.io.parse_tensor(tnp.asarray(features["latents"]), out_type=tf.float32)
    hidden_states = tf.io.parse_tensor(tnp.asarray(features["hidden_states"]), out_type=tf.float32)
    return {"pixel_values" : latents, "input_ids" : hidden_states}

  filenames = tf.io.gfile.glob(os.path.join(config.train_data_dir,"*"))
  train_ds = (
    tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
      .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
      .map(prepare_sample, num_parallel_calls=AUTOTUNE)
      .shuffle(global_batch_size * 10)
      .batch(global_batch_size, drop_remainder=True)
      .prefetch(AUTOTUNE)
      .repeat(100000000)
  )

  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

def make_pokemon_train_iterator(
    config,
    mesh,
    global_batch_size,
    tokenize_fn,
    image_transforms_fn):
  train_ds = load_dataset(config.dataset_name,split="train")

  captions_column = config.caption_column
  image_column = config.image_column
  cache_latents_text_encoder_outputs = config.cache_latents_text_encoder_outputs

  train_ds = train_ds.map(
    function=tokenize_fn,
    batched=True,
    remove_columns=[captions_column],
    num_proc=1 if cache_latents_text_encoder_outputs else 4,
    desc="Running tokenizer on train dataset",
  )
  # need to do it before load_as_tf_dataset
  # since raw images are different sizes
  # will break from_tensor_slices
  train_ds = train_ds.map(
    function=image_transforms_fn,
    batched=True,
    remove_columns=[image_column],
    num_proc=1 if cache_latents_text_encoder_outputs else config.transform_images_num_proc,
    desc="Transforming images",
  )

  # taken from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/examples/tensorflow/contrastive-image-text/run_clip.py#L225
  def load_as_tf_dataset(dataset, batch_size, shuffle):
    dataset = dataset.with_format("tensorflow")[:]
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if shuffle:
      tf_dataset = tf_dataset.shuffle(len(tf_dataset))
    tf_dataset = tf_dataset.batch(batch_size // jax.process_count(), drop_remainder=True)
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
