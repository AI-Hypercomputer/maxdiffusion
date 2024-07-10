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
  data_files_pattern=/mnt/disks/laion400-disk/webdataset-filtered-images/*.tar \
  extracted_files_dir=/tmp/raw-data-extracted \
  tfrecords_dir=/mnt/disks/laion400-disk/tf_records_processed_512_encoder_state \
  run_name=test no_records_per_shard=12720 base_output_directory=/tmp/output > result_512_encode.txt
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

TRANSFORMS = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Resize(size=512, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=512),
    transforms.Normalize([0.5], [0.5])
  ]
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

def create_example(moments, hidden_states):
    moments = tf.io.serialize_tensor(moments)
    hidden_states = tf.io.serialize_tensor(hidden_states)
    feature = {
        "moments": bytes_feature(moments),
        "clip_embeddings": bytes_feature(hidden_states),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def tokenize_captions(caption, pipeline, p_encode):
   text_inputs = pipeline.tokenizer([caption],
                                    max_length=pipeline.tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True)
   hidden_states = p_encode(np.stack(text_inputs.input_ids))
   hidden_states = jnp.squeeze(hidden_states).astype(jnp.bfloat16)
   return hidden_states

def vae_apply(images, vae, vae_params):
  moments = vae.apply(
    {"params" : vae_params},
    images,deterministic=True,
    method=vae.moments
  )
  return moments

def img_to_moments(img, p_vae_apply):
  img = TRANSFORMS(img)
  img = np.expand_dims(np.array(img), axis=0)
  moments = p_vae_apply(img)
  moments = jnp.squeeze(moments).astype(jnp.bfloat16)
  return moments

def generate_dataset(config):
  tfrecords_dir=config.tfrecords_dir
  extracted_files_dir=config.extracted_files_dir

  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

  weight_dtype = max_utils.get_dtype(config)

  pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
      config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
      safety_checker=None,
      feature_extractor=None,
      split_head_dim=config.split_head_dim,
      norm_num_groups=config.norm_num_groups,
      from_pt=config.from_pt,
      attention_kernel=config.attention,
  )

  p_encode = jax.jit(functools.partial(encode,
                                        text_encoder=pipeline.text_encoder,
                                        text_encoder_params=params["text_encoder"]))

  p_vae_apply = jax.jit(functools.partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))

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
    extracted_filenames = tf.io.gfile.glob(f"{extract_to_folder}/*.jpg")
    for image_file in extracted_filenames:
      img = Image.open(image_file).convert("RGB")
      moments = img_to_moments(img, p_vae_apply)
      caption_file = image_file.split(".")[0] + ".txt"
      json_file = image_file.split(".")[0] + ".json"
      with open(caption_file, "r") as f:
        caption = f.read()

      embedding = np.array(tokenize_captions(caption, pipeline, p_encode))
      example = create_example(moments, embedding)
      writer.write(example)
      shard_record_count+=1
      global_record_count+=1
      os.remove(image_file)
      os.remove(caption_file)
      os.remove(json_file)
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

def run(config):
  generate_dataset(config)

def main(argv: Sequence[str]) -> None:
   pyconfig.initialize(argv)
   run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
