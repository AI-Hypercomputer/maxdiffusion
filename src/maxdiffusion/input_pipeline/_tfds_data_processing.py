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
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset, load_from_disk

from maxdiffusion import multihost_dataloading

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_as_tf_dataset(dataset, global_batch_size, shuffle, dataloading_host_count):
  dataset = dataset.with_format("tensorflow")[:]
  tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

  if shuffle:
    tf_dataset = tf_dataset.shuffle(len(tf_dataset))
  tf_dataset = tf_dataset.batch(global_batch_size // dataloading_host_count, drop_remainder=True)
  tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  tf_dataset = tf_dataset.repeat(-1)

  return tf_dataset


def make_tf_iterator(
    config, dataloading_host_index, dataloading_host_count, mesh, global_batch_size, tokenize_fn, image_transforms_fn
):

  if config.cache_latents_text_encoder_outputs and os.path.isdir(config.dataset_save_location):
    train_ds = load_from_disk(config.dataset_save_location)
  else:
    train_ds = load_dataset(config.dataset_name, split=config.train_split)
    train_ds = train_ds.select_columns([config.caption_column, config.image_column])
    train_ds = train_ds.map(
        function=tokenize_fn,
        batched=True,
        remove_columns=[config.caption_column],
        num_proc=1 if config.cache_latents_text_encoder_outputs else config.tokenize_captions_num_proc,
        desc="Running tokenizer on train dataset",
    )
    # need to do it before load_as_tf_dataset
    # since raw images are different sizes
    # will break from_tensor_slices
    train_ds = train_ds.map(
        function=image_transforms_fn,
        batched=True,
        remove_columns=[config.image_column],
        num_proc=1 if config.cache_latents_text_encoder_outputs else config.transform_images_num_proc,
        desc="Transforming images",
    )
    if config.cache_latents_text_encoder_outputs:
      train_ds.save_to_disk(config.dataset_save_location)
      train_ds.cleanup_cache_files()

  train_ds = load_as_tf_dataset(train_ds, global_batch_size, True, dataloading_host_count)
  train_ds = train_ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  return train_iter


# TODO - https://github.com/google/array_record/blob/main/beam/examples/example_gcs_conversion.py
def make_tfrecord_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
):
  """Iterator for TFRecord format. For Laion dataset,
  check out preparation script
  maxdiffusion/pedagogical_examples/to_tfrecords.py
  """
  feature_description = {
      "moments": tf.io.FixedLenFeature([], tf.string),
      "clip_embeddings": tf.io.FixedLenFeature([], tf.string),
  }

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, feature_description)

  def prepare_sample(features):
    moments = tf.io.parse_tensor(tnp.asarray(features["moments"]), out_type=tf.float32)
    clip_embeddings = tf.io.parse_tensor(tnp.asarray(features["clip_embeddings"]), out_type=tf.float32)
    return {"pixel_values": moments, "input_ids": clip_embeddings}

  filenames = tf.io.gfile.glob(os.path.join(config.train_data_dir, "*"))
  train_ds = (
      tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
      .shard(num_shards=dataloading_host_count, index=dataloading_host_index)
      .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
      .map(prepare_sample, num_parallel_calls=AUTOTUNE)
      .shuffle(global_batch_size * 10)
      .batch(global_batch_size // dataloading_host_count, drop_remainder=True)
      .repeat(-1)
      .prefetch(AUTOTUNE)
  )

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  return train_iter
