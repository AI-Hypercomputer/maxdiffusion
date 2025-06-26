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
from functools import partial
import tensorflow as tf
from datasets import load_from_disk, Dataset
import jax

from maxdiffusion.input_pipeline import _hf_data_processing
from maxdiffusion.input_pipeline import _grain_data_processing
from maxdiffusion.input_pipeline import _tfds_data_processing
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
    CLASS_PROMPT_INPUT_IDS,
    INSTANCE_DATASET_NAME,
    CLASS_DATASET_NAME,
)
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def make_data_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    tokenize_fn=None,
    image_transforms_fn=None,
    feature_description=None,
    prepare_sample_fn=None
):
  """Make data iterator for SD1, 2, XL, dataset_types in (hf, tf, tfrecord)"""
  
  if config.dataset_type == "hf" or config.dataset_type == "tf":
    if tokenize_fn is None or image_transforms_fn is None:
      raise ValueError(f"dataset type {config.dataset_type} needs to pass a tokenize_fn and image_transforms_fn")
  
  if config.dataset_type == "tfrecord" and config.cache_latents_text_encoder_outputs and feature_description is None or prepare_sample_fn is None:
    raise ValueError(f"dataset type {config.dataset_type} needs to pass a feature_description dictionary and prepare_sample_fn function when cache_latents_text_encoder_outputs is True.")

  if config.dataset_type == "hf":
    return _hf_data_processing.make_hf_streaming_iterator(
        config,
        dataloading_host_index,
        dataloading_host_count,
        mesh,
        global_batch_size,
        tokenize_fn=tokenize_fn,
        image_transforms_fn=image_transforms_fn,
    )
  elif config.dataset_type == "grain":
    return _grain_data_processing.make_grain_iterator(
        config,
        dataloading_host_index,
        dataloading_host_count,
        mesh,
        global_batch_size,
    )
  elif config.dataset_type == "tf":
    return _tfds_data_processing.make_tf_iterator(
        config,
        dataloading_host_index,
        dataloading_host_count,
        mesh,
        global_batch_size,
        tokenize_fn=tokenize_fn,
        image_transforms_fn=image_transforms_fn,
    )
  elif config.dataset_type == "tfrecord":
    return _tfds_data_processing.make_tfrecord_iterator(
        config,
        dataloading_host_index,
        dataloading_host_count,
        mesh,
        global_batch_size,
        feature_description,
        prepare_sample_fn
    )
  else:
    assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be in (tf, tfrecord, hf, grain)"


def make_dreambooth_train_iterator(config, mesh, global_batch_size, tokenizer, vae, vae_params):
  """Creates a dreambooth training iterator for sd1.x,sd2.x"""

  instance_images = []
  instance_prompt_ids = []
  class_images = []
  class_prompt_ids = []

  instance_dataset_full_path = os.path.join(config.dataset_save_location, INSTANCE_DATASET_NAME)
  class_dataset_full_path = os.path.join(config.dataset_save_location, CLASS_DATASET_NAME)

  if config.cache_dreambooth_dataset and os.path.isdir(config.dataset_save_location):
    instance_train_ds = load_from_disk(instance_dataset_full_path)
    class_train_ds = load_from_disk(class_dataset_full_path)
  else:
    # load class data
    class_image_paths = tf.io.gfile.glob(os.path.join(config.class_data_dir, "*"))
    for image_path in class_image_paths:
      class_images.append(Image.open(image_path).convert("RGB"))
    class_prompt_ids.extend([config.class_prompt] * len(class_images))

    # load instance data. Since we use prior preservation, we need to match
    # the number of instance images we're using.
    instance_image_paths = tf.io.gfile.glob(os.path.join(config.instance_data_dir, "*"))
    for index in range(len(class_images)):
      instance_image = instance_image_paths[index % len(instance_image_paths)]
      instance_images.append(Image.open(instance_image).convert("RGB"))
    instance_prompt_ids.extend([config.instance_prompt] * len(instance_images))

    instance_dataset_dict = {
        INSTANCE_IMAGES: instance_images,
        INSTANCE_PROMPT_IDS: instance_prompt_ids,
    }

    class_dataset_dict = {CLASS_IMAGES: class_images, CLASS_PROMPT_IDS: class_prompt_ids}

    instance_train_ds = Dataset.from_dict(instance_dataset_dict)
    class_train_ds = Dataset.from_dict(class_dataset_dict)

    tokenize_fn = partial(
        tokenize_captions, caption_column=INSTANCE_PROMPT_IDS, tokenizer=tokenizer, input_ids_key=INSTANCE_PROMPT_INPUT_IDS
    )
    instance_train_ds = instance_train_ds.map(
        function=tokenize_fn,
        batched=True,
        remove_columns=[INSTANCE_PROMPT_IDS],
        num_proc=1,
        desc="Running tokenizer on instance dataset",
    )
    rng = jax.random.key(config.seed)
    p_vae_apply = jax.jit(partial(vae_apply, vae=vae, vae_params=vae_params))
    transform_images_fn = partial(
        transform_images,
        image_column=INSTANCE_IMAGES,
        image_resolution=config.resolution,
        rng=rng,
        global_batch_size=global_batch_size,
        pixel_ids_key=INSTANCE_IMAGE_LATENTS,
        p_vae_apply=p_vae_apply,
    )
    instance_train_ds = instance_train_ds.map(
        function=transform_images_fn,
        batched=True,
        remove_columns=[INSTANCE_IMAGES],
        num_proc=1,
        desc="Running vae on instance dataset",
    )

    tokenize_fn = partial(
        tokenize_captions, caption_column=CLASS_PROMPT_IDS, tokenizer=tokenizer, input_ids_key=CLASS_PROMPT_INPUT_IDS
    )
    class_train_ds = class_train_ds.map(
        function=tokenize_fn,
        batched=True,
        remove_columns=[CLASS_PROMPT_IDS],
        num_proc=1,
        desc="Running tokenizer on class dataset",
    )
    transform_images_fn = partial(
        transform_images,
        image_column=CLASS_IMAGES,
        image_resolution=config.resolution,
        rng=rng,
        global_batch_size=global_batch_size,
        pixel_ids_key=CLASS_IMAGE_LATENTS,
        p_vae_apply=p_vae_apply,
    )
    class_train_ds = class_train_ds.map(
        function=transform_images_fn,
        batched=True,
        remove_columns=[CLASS_IMAGES],
        num_proc=1,
        desc="Running vae on instance dataset",
    )

    if config.cache_dreambooth_dataset:
      instance_train_ds.save_to_disk(instance_dataset_full_path)
      class_train_ds.save_to_disk(class_dataset_full_path)

  instance_train_ds = _tfds_data_processing.load_as_tf_dataset(
      instance_train_ds, global_batch_size, True, jax.process_count()
  )
  class_train_ds = _tfds_data_processing.load_as_tf_dataset(class_train_ds, global_batch_size, True, jax.process_count())
  train_ds = tf.data.Dataset.zip((instance_train_ds, class_train_ds))
  train_ds = train_ds.shard(num_shards=jax.process_count(), index=jax.process_index())

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  return train_iter
