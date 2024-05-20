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
import pathlib
import shutil
import unittest
from absl.testing import absltest

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from ..import pyconfig
from ..import max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_laion400m_train_iterator,
  make_pokemon_train_iterator
)

from skimage.metrics import structural_similarity as ssim
from PIL import Image

from maxdiffusion import (
  FlaxStableDiffusionPipeline,
  FlaxStableDiffusionXLPipeline
)
from maxdiffusion.models import FlaxAutoencoderKL
from maxdiffusion.models.train import (
  encode,
  tokenize_captions
)
from maxdiffusion.train_sdxl import (
   encode_xl,
   tokenize_captions_xl
)
from maxdiffusion.maxdiffusion_utils import vae_apply, transform_images

HOME_DIR = pathlib.Path.home()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = str(HOME_DIR / ".cache" / "huggingface" / "datasets")

def cleanup(output_dir):
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

class InputPipelineInterface(unittest.TestCase):
  """Test Unet sharding"""
  def setUp(self):
    InputPipelineInterface.dummy_data = {}

  def test_make_pokemon_iterator_cache(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "cache_latents_text_encoder_outputs=True",
      "dataset_name=diffusers/pokemon-gpt4-captions"])
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    weight_dtype = max_utils.get_dtype(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt
    )
    rng = jax.random.PRNGKey(config.seed)
    p_encode = None
    p_vae_apply = None
    if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(partial(encode,text_encoder=pipeline.text_encoder,text_encoder_params=params["text_encoder"]))
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
    tokenize_fn = partial(tokenize_captions,caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode)
    image_transforms_fn = partial(transform_images,
                                  image_column=config.image_column,
                                  image_resolution=config.resolution,
                                  rng=rng,
                                  global_batch_size=global_batch_size,
                                  p_vae_apply=p_vae_apply)

    train_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      global_batch_size,
      tokenize_fn,
      image_transforms_fn
    )
    data = train_iterator()
    device_count = jax.device_count()

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    encoder_hidden_states = data["input_ids"]

    assert encoder_hidden_states.shape == (device_count,77, 1024)
    assert data["pixel_values"].shape == (device_count,
                                          pipeline.unet.config.in_channels,
                                          config.resolution // vae_scale_factor,
                                          config.resolution // vae_scale_factor)


  def test_make_pokemon_iterator_no_cache(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "cache_latents_text_encoder_outputs=False","tokenize_captions_num_proc=1","transform_images_num_proc=1",
      "dataset_name=diffusers/pokemon-gpt4-captions"])
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    weight_dtype = max_utils.get_dtype(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt
    )
    rng = jax.random.PRNGKey(config.seed)
    p_encode = None
    p_vae_apply = None
    if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(partial(encode,text_encoder=pipeline.text_encoder,text_encoder_params=params["text_encoder"]))
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
    tokenize_fn = partial(tokenize_captions,caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode)
    image_transforms_fn = partial(transform_images,
                                  image_column=config.image_column,
                                  image_resolution=config.resolution,
                                  rng=rng,
                                  global_batch_size=global_batch_size,
                                  p_vae_apply=p_vae_apply)

    train_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      global_batch_size,
      tokenize_fn,
      image_transforms_fn
    )
    data = train_iterator()
    device_count = jax.device_count()

    encoder_hidden_states = data["input_ids"]
    assert encoder_hidden_states.shape == (device_count,77)
    assert data["pixel_values"].shape == (device_count,
                                          3,
                                          config.resolution,
                                          config.resolution)


  def test_make_pokemon_iterator_sdxl_cache(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
        "cache_latents_text_encoder_outputs=True","per_device_batch_size=1",
        "dataset_name=diffusers/pokemon-gpt4-captions"])
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    weight_dtype = max_utils.get_dtype(config)
    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt
    )
    rng = jax.random.PRNGKey(config.seed)
    p_encode = None
    p_vae_apply = None
    if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(partial(encode_xl,
                                   text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
                                   text_encoder_params=[params["text_encoder"], params["text_encoder_2"]]))
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
    tokenize_fn = partial(tokenize_captions_xl,
                          caption_column=config.caption_column,
                          tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
                          p_encode=p_encode)
    image_transforms_fn = partial(transform_images,
                                  image_column=config.image_column,
                                  image_resolution=config.resolution,
                                  rng=rng,
                                  global_batch_size=global_batch_size,
                                  p_vae_apply=p_vae_apply)

    train_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      global_batch_size,
      tokenize_fn,
      image_transforms_fn
    )
    data = train_iterator()
    device_count = jax.device_count()

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

    prompt_embeds = data["prompt_embeds"]
    text_embeds = data["text_embeds"]
    assert prompt_embeds.shape == (device_count,77, 2048)
    assert text_embeds.shape == (device_count, 1280)
    assert data["pixel_values"].shape == (device_count,
                                          pipeline.unet.config.in_channels,
                                          config.resolution // vae_scale_factor,
                                          config.resolution // vae_scale_factor)


  def test_make_laion_iterator(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "cache_latents_text_encoder_outputs=True",
      "train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/processed/laion400m_tfrec"])
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    weight_dtype = max_utils.get_dtype(config)

    pipeline, _ = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt
    )

    train_iterator = make_laion400m_train_iterator(
      config,
      mesh,
      global_batch_size,
    )
    data = train_iterator()
    device_count = jax.device_count()

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    encoder_hidden_states = data["input_ids"]

    # TODO - laion dataset was prepared with an extra dim.
    # need to preprocess the dataset with dim removed.
    if len(encoder_hidden_states.shape) == 4:
        encoder_hidden_states = jnp.squeeze(encoder_hidden_states)

    assert encoder_hidden_states.shape == (device_count,77, 1024)
    assert data["pixel_values"].shape == (device_count,
                                          pipeline.unet.config.in_channels,
                                          config.resolution // vae_scale_factor,
                                          config.resolution // vae_scale_factor)

  def test_tfrecord(self):
    """Validate latents match a deterministic output image"""

    image_feature_description = {
      "latents": tf.io.FixedLenFeature([], tf.string),
      "hidden_states": tf.io.FixedLenFeature([], tf.string)
    }
    def _parse_image_function(example_proto):
      # Parse the input tf.train.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, image_feature_description)

    @tf.function()
    def prepare_sample(features):
      latents = tf.io.parse_tensor(tnp.asarray(features["latents"]), out_type=tf.float32)
      hidden_states = tf.io.parse_tensor(tnp.asarray(features["hidden_states"]), out_type=tf.float32)
      return {"pixel_values" : latents, "input_ids" : hidden_states}


    raw_image_dataset = tf.data.TFRecordDataset('gs://maxdiffusion-github-runner-test-assets/tfrecords/file_00-1000.tfrec')
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function).map(prepare_sample).batch(4)

    iterator = iter(parsed_image_dataset)
    image_features = iterator.get_next()
    latents = image_features["pixel_values"]
    vae, params = FlaxAutoencoderKL.from_pretrained(
      "stabilityai/stable-diffusion-2-base",
      subfolder="vae",
      from_pt=True,
    )
    latents = latents.numpy()
    latents = 1 / vae.config.scaling_factor * latents

    image = vae.apply(
        {"params" : params},
        latents,
        method=vae.decode
    ).sample

    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    image = np.array(image)
    test_image = image[0]
    test_image = (test_image * 255).round().astype("uint8")

    img_url = os.path.join(THIS_DIR,'images','latent_test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )

    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

if __name__ == '__main__':
  absltest.main()
