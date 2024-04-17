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
2. Create a persistent disk and attach to VM as read-write.
3. Create 2 directories inside the persistent disk to store the extracted files and created tfrecords.
3. Run this file:
python src/maxdiffusion/pedagogical_examples/to_tfrecords.py \
  src/maxdiffusion/configs/base_2_base.yml attention=dot_product \
  data_files_pattern=/mnt/disks/laion400-disk/webdataset-moments-filtered-tars/*.tar \
  extracted_files_dir=/mnt/disks/laion400-disk/raw-data-extracted \
  tfrecords_dir=/mnt/disks/laion400-disk/laion400m_moments-tfrec \
  run_name=test no_records_per_shard=2048 base_output_directory=gs://jfacevedo-maxdiffusion/training_results/
"""

import os
import glob
import functools
from absl import app
from typing import Sequence
import time
import tarfile
import tensorflow as tf
from torchvision import transforms
import tensorflow_datasets as tfds
import numpy as np
import jax
import jax.numpy as jnp
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None

from tqdm import tqdm

from maxdiffusion import (
  FlaxStableDiffusionPipeline,
  pyconfig,
  max_utils
)

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
        "moments": bytes_feature(latent),
        "caption": bytes_feature(hidden_states),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def generate_dataset(config):
  tfrecords_dir=config.tfrecords_dir
  extracted_files_dir=config.extracted_files_dir

  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

  tf_rec_num = 0
  filenames = tf.io.gfile.glob(config.data_files_pattern)
  no_records_per_shard = config.no_records_per_shard
  global_record_count = 0
  shard_record_count = 0
  writer = tf.io.TFRecordWriter(
    tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard)))
  for i in tqdm(range(len(filenames))):
    filename = filenames[i]
    extract_to_folder = filename.split("/")[-1].split(".")[0]
    extract_to_folder = os.path.join(extracted_files_dir,extract_to_folder)
    os.makedirs(extract_to_folder, exist_ok=True)
    start = time.time()
    file = tarfile.open(filename)
    file.extractall(extract_to_folder, filter='data')
    extracted_filenames = tf.io.gfile.glob(f"{extract_to_folder}/*.npy")
    for moments_file in extracted_filenames:
      moments = np.load(moments_file)
      moments = np.squeeze(moments)
      # Moments generated with Pytorch which is Channel first.
      # CWH -> WHC
      moments = moments.transpose((1,2,0))
      caption_file = moments_file.split(".")[0] + ".txt"
      with open(caption_file, "r") as f:
        caption = f.read()
      example = create_example(moments, caption)
      writer.write(example)
      shard_record_count+=1
      global_record_count+=1
      if shard_record_count >= no_records_per_shard:
        writer.close()
        shard_record_count = 0
        tf_rec_num+=1
        writer = tf.io.TFRecordWriter(
          tfrecords_dir + "/file_%.8i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard)))
    print("one record time: ", (time.time() - start))

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

def vae_apply(images, sample_rng, vae, vae_params):
  vae_outputs = vae.apply(
    {"params" : vae_params},
    images,deterministic=True,
    method=vae.encode
  )
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor
  return latents

def img_to_latents(img, p_vae_apply, sample_rng):
  img = TRANSFORMS(img)
  img = np.expand_dims(np.array(img), axis=0)
  latents = p_vae_apply(img, sample_rng)
  latents = jnp.squeeze(latents)
  return latents

def run(config):

  generate_dataset(config)

def main(argv: Sequence[str]) -> None:
   pyconfig.initialize(argv)
   run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
