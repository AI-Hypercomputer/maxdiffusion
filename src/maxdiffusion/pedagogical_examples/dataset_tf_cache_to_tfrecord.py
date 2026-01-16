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
import os
import argparse
import tensorflow as tf
from datasets import load_from_disk
import numpy as np


def _bytes_feature(value):
  """Returns a bytes_list from a serialized tensor."""
  if not isinstance(value, tf.Tensor):
    value = tf.constant(value)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def create_4_feature_example(record):
  """Creates a tf.train.Example proto with all 4 pre-computed features."""
  pixel_values = tf.io.serialize_tensor(record["pixel_values"])
  input_ids = tf.io.serialize_tensor(record["input_ids"])
  prompt_embeds = tf.io.serialize_tensor(record["prompt_embeds"])
  text_embeds = tf.io.serialize_tensor(record["text_embeds"])

  feature = {
      "pixel_values": _bytes_feature(pixel_values),
      "input_ids": _bytes_feature(input_ids),
      "prompt_embeds": _bytes_feature(prompt_embeds),
      "text_embeds": _bytes_feature(text_embeds),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def run(args):
  """Main processing function."""
  # Load the cached dataset from the location specified in the arguments
  print(f"Loading processed dataset from disk: {args.dataset_save_location}")
  processed_ds = load_from_disk(args.dataset_save_location)
  print("Dataset loaded successfully.")

  # Get sharding and output directory from the arguments
  tfrecords_dir = args.tfrecords_dir
  num_shards = args.data_num_shards
  os.makedirs(tfrecords_dir, exist_ok=True)

  writers = [
      tf.io.TFRecordWriter(os.path.join(tfrecords_dir, f"shard-{i:05d}-of-{num_shards:05d}.tfrecord"))
      for i in range(num_shards)
  ]

  print(f"Writing {len(processed_ds)} records into {num_shards} TFRecord shards...")

  for i, record in enumerate(processed_ds):
    # Create a new record with explicit casting for float types
    casted_record = {
        "pixel_values": np.float32(record["pixel_values"]),
        "input_ids": record["input_ids"],  # This is already integer type
        "prompt_embeds": np.float32(record["prompt_embeds"]),
        "text_embeds": np.float32(record["text_embeds"]),
    }

    writer_index = i % num_shards
    tf_example = create_4_feature_example(casted_record)
    writers[writer_index].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()

  print("TFRecord conversion complete.")


def main():
  """Parses command-line arguments and runs the conversion."""
  parser = argparse.ArgumentParser(description="Convert a cached Hugging Face dataset to sharded TFRecords.")
  parser.add_argument(
      "--dataset_save_location",
      type=str,
      required=False,
      default="/tmp/pokemon-gpt4-captions_xl",
      help="Path to the cached dataset created by the training pipeline.",
  )
  parser.add_argument(
      "--tfrecords_dir",
      type=str,
      required=False,
      default="/tmp/cached_pokemon_tfrecords_sharded",
      help="Output directory to save the sharded TFRecord files.",
  )
  parser.add_argument(
      "--data_num_shards", type=int, default=128, help="Number of shards to split the TFRecord dataset into."
  )

  args = parser.parse_args()
  run(args)


if __name__ == "__main__":
  main()
