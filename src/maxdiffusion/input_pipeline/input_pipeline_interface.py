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
from functools import partial
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset, load_from_disk, Dataset
import jax
import jax.numpy as jnp

from maxdiffusion import multihost_dataloading
from maxdiffusion.maxdiffusion_utils import tokenize_captions, transform_images, vae_apply
from maxdiffusion.dreambooth.dreambooth_constants import (
  INSTANCE_IMAGES,
  INSTANCE_IMAGE_LATENTS,
  INSTANCE_PROMPT_IDS,
  INSTANCE_PROMPT_INPUT_IDS,
  CLASS_IMAGES,
  CLASS_IMAGE_LATENTS,
  CLASS_PROMPT_IDS,
  CLASS_PROMPT_INPUT_IDS
)
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

# taken from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/examples/tensorflow/contrastive-image-text/run_clip.py#L225
def load_as_tf_dataset(dataset, batch_size, shuffle, config):
  dataset = dataset.with_format("tensorflow")[:]
  tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

  if shuffle:
    tf_dataset = tf_dataset.shuffle(len(tf_dataset))
  tf_dataset = tf_dataset.batch(batch_size // jax.process_count(), drop_remainder=True)
  tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  tf_dataset = tf_dataset.repeat(-1)

  return tf_dataset

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
      .batch(global_batch_size // jax.process_count(), drop_remainder=True)
      .prefetch(AUTOTUNE)
      .repeat(-1)
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
  dataset_save_location = config.dataset_save_location
  if os.path.isdir(dataset_save_location):
    train_ds = load_from_disk(dataset_save_location)
  else:
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
    train_ds.save_to_disk(dataset_save_location)
    train_ds.cleanup_cache_files()

  train_ds = load_as_tf_dataset(
    train_ds, global_batch_size, True, config
  )
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

def make_dreambooth_train_iterator(
  config,
  mesh,
  global_batch_size,
  tokenizer,
  vae,
  vae_params
):
  """Creates a dreambooth training iterator for sd1.x,sd2.x"""

  instance_images = []
  instance_prompt_ids = []
  class_images = []
  class_prompt_ids = []
  
  # load class data
  class_image_paths = tf.io.gfile.glob(f"{config.class_data_dir}/*")
  for image_path in class_image_paths:
    class_images.append(Image.open(image_path).convert("RGB"))
  class_prompt_ids.extend([config.class_prompt] * len(class_images))

  # load instance data. Since we use prior preservation, we need to match
  # the number of instance images we're using.
  instance_image_paths = tf.io.gfile.glob(f"{config.instance_data_dir}/*")
  for index in range(len(class_images)):
    instance_image = instance_image_paths[index % len(instance_image_paths)]
    instance_images.append(Image.open(instance_image).convert("RGB"))
  instance_prompt_ids.extend([config.instance_prompt] * len(instance_images))
  
  instance_dataset_dict = {
    INSTANCE_IMAGES : instance_images,
    INSTANCE_PROMPT_IDS : instance_prompt_ids,
  }

  class_dataset_dict = {
    CLASS_IMAGES : class_images,
    CLASS_PROMPT_IDS : class_prompt_ids
  }

  instance_train_ds = Dataset.from_dict(instance_dataset_dict)
  class_train_ds = Dataset.from_dict(class_dataset_dict)

  tokenize_fn = partial(tokenize_captions,
                        caption_column=INSTANCE_PROMPT_IDS,
                        tokenizer=tokenizer,
                        input_ids_key=INSTANCE_PROMPT_INPUT_IDS)
  instance_train_ds = instance_train_ds.map(
    function=tokenize_fn,
    batched=True,
    remove_columns=[INSTANCE_PROMPT_IDS],
    num_proc=1,
    desc="Running tokenizer on instance dataset",
  )
  rng = jax.random.key(config.seed)
  p_vae_apply = jax.jit(partial(vae_apply, vae=vae, vae_params=vae_params))
  transform_images_fn = partial(transform_images,
                                image_column=INSTANCE_IMAGES,
                                image_resolution=config.resolution,
                                rng=rng,
                                global_batch_size=global_batch_size,
                                pixel_ids_key=INSTANCE_IMAGE_LATENTS,
                                p_vae_apply=p_vae_apply)
  instance_train_ds = instance_train_ds.map(
    function=transform_images_fn,
    batched=True,
    remove_columns=[INSTANCE_IMAGES],
    num_proc=1,
    desc="Running vae on instance dataset"
  )

  tokenize_fn = partial(tokenize_captions,
                        caption_column=CLASS_PROMPT_IDS,
                        tokenizer=tokenizer,
                        input_ids_key=CLASS_PROMPT_INPUT_IDS)
  class_train_ds = class_train_ds.map(
    function=tokenize_fn,
    batched=True,
    remove_columns=[CLASS_PROMPT_IDS],
    num_proc=1,
    desc="Running tokenizer on class dataset",
  )
  transform_images_fn = partial(transform_images,
                                image_column=CLASS_IMAGES,
                                image_resolution=config.resolution,
                                rng=rng,
                                global_batch_size=global_batch_size,
                                pixel_ids_key=CLASS_IMAGE_LATENTS,
                                p_vae_apply=p_vae_apply)
  class_train_ds = class_train_ds.map(
    function=transform_images_fn,
    batched=True,
    remove_columns=[CLASS_IMAGES],
    num_proc=1,
    desc="Running vae on instance dataset"
  )

  instance_train_ds = load_as_tf_dataset(
    instance_train_ds, global_batch_size, True, config
  )
  class_train_ds = load_as_tf_dataset(
    class_train_ds, global_batch_size, True, config
  )
  train_ds = tf.data.Dataset.zip((instance_train_ds, class_train_ds))
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter