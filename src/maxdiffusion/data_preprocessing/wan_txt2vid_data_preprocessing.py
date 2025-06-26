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
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxdiffusion import pyconfig, max_utils
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.video_processor import VideoProcessor

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


def text_encode(pipeline, prompt: Union[str, List[str]]):
  encoder_hidden_states = pipeline._get_t5_prompt_embeds(prompt)
  encoder_hidden_states = encoder_hidden_states.detach().numpy()
  return encoder_hidden_states

def vae_encode(video, rng, vae, vae_cache):
  latent = vae.encode(video, feat_cache=vae_cache)
  latent = latent.latent_dist.sample(rng)
  return latent
  
def generate_dataset(config, pipeline):

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

  # create mesh
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  rng = jax.random.key(config.seed)

  vae_scale_factor_spatial = 2 ** len(pipeline.vae.temperal_downsample)
  video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial) 
  
  # jit vae fun.
  p_vae_encode = jax.jit(functools.partial(vae_encode, vae=pipeline.vae, vae_cache=pipeline.vae_cache))
  
  # Load dataset
  ds = load_dataset(config.dataset_name, split='train')
  ds = ds.shuffle(seed=config.seed)
  ds = ds.select_columns([config.caption_column, config.image_column])
  batch_size = 10
  for i in range(0, len(ds), batch_size):
    rng, new_rng = jax.random.split(rng)
    text = ds[i:i+batch_size]['text']
    videos = ds[i:i+batch_size]['image']
    
    videos = [video_processor.preprocess_video([video], height=config.height, width=config.width) for video in videos]
    video = jnp.array(np.squeeze(np.array(videos), axis=1), dtype=config.weights_dtype)
    with mesh:
      latents = p_vae_encode(video=video, rng=new_rng)
      latents = jnp.transpose(latents, (0, 4, 1, 2, 3))
    encoder_hidden_states = text_encode(pipeline, text)
    for latent, encoder_hidden_state in zip(latents, encoder_hidden_states):
      writer.write(create_example(latent, encoder_hidden_state))
      shard_record_count += 1
      global_record_count += 1

    if shard_record_count >= no_records_per_shard:
      writer.close()
      tf_rec_num +=1
      writer = tf.io.TFRecordWriter(
          tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard))
      )
      shard_record_count = 0



def run(config):
  pipeline = WanPipeline.from_pretrained(config, load_transformer=False)
  # Don't need the transformer for preprocessing.
  generate_dataset(config, pipeline)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

if __name__ == "__main__":
  app.run(main)