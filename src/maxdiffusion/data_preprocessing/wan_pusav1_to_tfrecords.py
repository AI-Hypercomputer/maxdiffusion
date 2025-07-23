"""
 Copyright 2025 Google LLC

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

"""
Prepare tfrecords with latents and text embeddings preprocessed.
1. Download the dataset
"""

import os
import functools
from absl import app
from typing import Sequence, Union, List
from datasets import load_dataset
import csv
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxdiffusion import pyconfig, max_utils
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.video_processor import VideoProcessor

import torch
import tensorflow as tf


def image_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
  """Returns a list of float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(latent, hidden_states):
  latent = tf.io.serialize_tensor(latent)
  hidden_states = tf.io.serialize_tensor(hidden_states)
  feature = {
      "latents": bytes_feature(latent),
      "encoder_hidden_states": bytes_feature(hidden_states),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()

def generate_dataset(config):

  tfrecords_dir = config.tfrecords_dir
  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)

  tf_rec_num = 0
  no_records_per_shard = config.no_records_per_shard
  global_record_count = 0
  writer = tf.io.TFRecordWriter(
      tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard))
  )
  shard_record_count = 0

  # Load dataset
  metadata_path = os.path.join(config.train_data_dir, "metadata.csv")
  with open(metadata_path, 'r', newline='') as file:
    # Create a csv.reader object
    csv_reader = csv.reader(file)
    next(csv_reader)

    # If your CSV has a header row, you can skip it
    # next(csv_reader, None)

    # Iterate over each row in the CSV file
    for row in csv_reader:
      video_name = row[0]
      pth_path = os.path.join(config.train_data_dir,"train", f"{video_name}.tensors.pth")
      loaded_state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
      prompt_embeds = loaded_state_dict["prompt_emb"]["context"]
      latent = loaded_state_dict["latents"]
      # Format we want(4, 16, 1, 64, 64)
      latent = jnp.array(latent.float().numpy(), dtype=config.weights_dtype)
      prompt_embeds = jnp.array(prompt_embeds.float().numpy(), dtype=config.weights_dtype)
      writer.write(create_example(latent, prompt_embeds))
      shard_record_count += 1
      global_record_count += 1

      if shard_record_count >= no_records_per_shard:
        writer.close()
        tf_rec_num += 1
        writer = tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard))
        )
        shard_record_count = 0

def run(config):
  generate_dataset(config)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
