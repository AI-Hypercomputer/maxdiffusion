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

"""
Example file of how to prepare tfrecords with latents and hidden_states preprocessed.
1. Download the dataset as instructed here https://github.com/mlcommons/training/tree/master/stable_diffusion#the-datasets
2. Create an attachable disk and mount it onto the machine. https://cloud.google.com/tpu/docs/setup-persistent-disk
3. Run this file:

python read_dataset.py \
  maxdiffusion/src/maxdiffusion/configs/base_2_base.yml \
  attention=dot_product \
  laion_dataset_file_pattern=gs://jfacevedo-maxdiffusion/datasets/laion400m/webdataset-latents-filtered/*.tar \
  tfrecords_dir=/data/tfrecords
"""

import os
import glob
import functools
from absl import app
from typing import Sequence

import tensorflow as tf
import tarfile
import tensorflow_datasets as tfds
import numpy as np
import jax

from maxdiffusion import (
  FlaxStableDiffusionPipeline,
  pyconfig,
  max_utils
)

dl_manager = tfds.download.DownloadManager(download_dir="/tmp")
tmp_dataset = "dataset"

def delete_files(path):
  files = glob.glob(path+"/*")
  for f in files:
      os.remove(f)

def tokenize_captions(caption, pipeline, p_encode):
   text_inputs = pipeline.tokenizer([caption],
                                    max_length=pipeline.tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True)
   hidden_states = p_encode(np.stack(text_inputs.input_ids))
   return hidden_states

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

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
        "hidden_states": bytes_feature(hidden_states),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def generate_dataset(config, pipeline, p_encode):

  tfrecords_dir=config.tfrecords_dir

  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

  tf_rec_num = 0
  filenames = tf.io.gfile.glob(config.data_files_pattern)
  for filename in filenames:
    tmp_file = dl_manager.download(filename)
    file = tarfile.open(tmp_file)
    file.extractall("dataset", filter='data')
    extracted_filenames = tf.io.gfile.glob("dataset/*.npy")
    with tf.io.TFRecordWriter(
       tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, len(extracted_filenames))
    ) as writer:
      for latent_file in extracted_filenames:
        latent = np.load(latent_file)
        caption_file = latent_file.split(".")[0] + ".txt"
        with open(caption_file, "r") as f:
          caption = f.read()
        hidden_states = np.array(tokenize_captions(caption, pipeline, p_encode))
        example = create_example(latent, hidden_states)
        writer.write(example)
      tf_rec_num+=1
      delete_files("dataset")
      os.remove(tmp_file)

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

def run(config):

  weight_dtype = max_utils.get_dtype(config)
  pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
    safety_checker=None, feature_extractor=None,
    split_head_dim=config.split_head_dim, from_pt=config.from_pt,
    attention_kernel=config.attention, flash_block_sizes=None,
    mesh=None
  )

  p_encode = jax.jit(functools.partial(encode,
                                           text_encoder=pipeline.text_encoder,
                                           text_encoder_params=params["text_encoder"]))

  generate_dataset(config, pipeline, p_encode)

def main(argv: Sequence[str]) -> None:
   pyconfig.initialize(argv)
   run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
