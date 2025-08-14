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
import jax
from maxdiffusion import multihost_dataloading

AUTOTUNE = tf.data.AUTOTUNE


def load_as_tf_dataset(dataset, global_batch_size, shuffle, dataloading_host_count):
  dataset = dataset.with_format("tensorflow")[:]
  tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

  if shuffle:
    tf_dataset = tf_dataset.shuffle(len(tf_dataset))
  tf_dataset = tf_dataset.batch(global_batch_size // dataloading_host_count, drop_remainder=True)
  tf_dataset = tf_dataset.prefetch(AUTOTUNE)
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
      # Only process 0 should attempt to clean up cache files
      if jax.process_index() == 0:
        try:
          train_ds.cleanup_cache_files()
        except FileNotFoundError:
          # Ignore FileNotFoundError as files may have been cleaned up by another process
          pass
  train_ds = load_as_tf_dataset(train_ds, global_batch_size, True, dataloading_host_count)
  train_ds = train_ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  return train_iter

# TODO - https://github.com/google/array_record/blob/main/beam/examples/example_gcs_conversion.py
def _make_tfrecord_iterator(
    config, dataloading_host_index, dataloading_host_count, mesh, global_batch_size, feature_description_fn, prepare_sample_fn, dataset_path, is_training: bool
):
  # set load_tfrecord_cached to True in config to use pre-processed tfrecord dataset.
  # pedagogical_examples/dataset_tf_cache_to_tfrecord.py to convert tf preprocessed dataset to tfrecord.
  # Dataset cache in github runner test doesn't contain all the features since its shared, Use the default tfrecord iterator.
  # if is_training is True, loads the training dataset. If False, loads the evaluation dataset.

  # checks that the dataset path is valid. In case of gcs, the existance of the dir is not checked.
  is_dataset_dir_valid = "gs://" in config.dataset_save_location or os.path.isdir(config.dataset_save_location)

  # Determine whether to use the "cached" dataset, which requires externally
  # provided parsing functions, or the default one with its internal parsing logic.
  make_cached_tfrecord_iterator = (
    config.cache_latents_text_encoder_outputs
    and is_dataset_dir_valid
    and "load_tfrecord_cached" in config.get_keys()
    and config.load_tfrecord_cached
  )

  feature_description = {
      "moments": tf.io.FixedLenFeature([], tf.string),
      "clip_embeddings": tf.io.FixedLenFeature([], tf.string),
  }

  used_feature_description = feature_description_fn if make_cached_tfrecord_iterator else feature_description

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, used_feature_description)

  def prepare_sample(features):
    moments = tf.io.parse_tensor(tnp.asarray(features["moments"]), out_type=tf.float32)
    clip_embeddings = tf.io.parse_tensor(tnp.asarray(features["clip_embeddings"]), out_type=tf.float32)
    return {"pixel_values": moments, "input_ids": clip_embeddings}

  filenames = tf.io.gfile.glob(os.path.join(dataset_path, "*"))
  ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

  # --- PADDING LOGIC FOR EVALUATION ---
  if not is_training:
    num_eval_samples = 0
    for _ in ds:
        num_eval_samples += 1

    remainder = num_eval_samples % global_batch_size
    if remainder != 0:
        num_to_pad = global_batch_size - remainder
        # Create a dataset of padding samples from the beginning
        padding_ds = ds.take(num_to_pad)
        # Add the padding samples to the end
        ds = ds.concatenate(padding_ds)
        print(f"Padded evaluation dataset with {num_to_pad} samples.")

  used_prepare_sample = prepare_sample_fn if make_cached_tfrecord_iterator else prepare_sample
  ds = (
    ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
    .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(used_prepare_sample, num_parallel_calls=AUTOTUNE)
  )
  if is_training:
    ds = (
      ds.shuffle(global_batch_size * 10)
      .batch(global_batch_size // dataloading_host_count, drop_remainder=True)
      .repeat(-1)
      .prefetch(AUTOTUNE)
    )
  # For Evaluation
  else:
    ds = (
      ds.batch(global_batch_size // dataloading_host_count, drop_remainder=False)
      .prefetch(AUTOTUNE)
    )

  iter = multihost_dataloading.MultiHostDataLoadIterator(ds, mesh)
  return iter

def make_tfrecord_iterator(
    config, dataloading_host_index, dataloading_host_count, mesh, global_batch_size, feature_description, prepare_sample_fn, is_training
):
  """Iterator for TFRecord format. For Laion dataset,
  check out preparation script
  maxdiffusion/pedagogical_examples/to_tfrecords.py
  """
  # Currently only support evaluation on tfrecord. To avoid influencing previous reference, judge whether is training dataset.
  # TODO: refactor to support evaluation on all dataset format.
  dataset_path = config.train_data_dir if is_training else config.eval_data_dir
  return _make_tfrecord_iterator(config, dataloading_host_index, dataloading_host_count, mesh, global_batch_size, feature_description, prepare_sample_fn, dataset_path, is_training)