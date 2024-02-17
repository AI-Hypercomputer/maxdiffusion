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
import unittest
from absl.testing import absltest

import jax
from jax.sharding import Mesh

from ..import pyconfig
from ..import max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import make_pokemon_train_iterator
from maxdiffusion import FlaxStableDiffusionPipeline

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class InputPipelineInterface(unittest.TestCase):
  """Test Unet sharding"""
  def setUp(self):
    InputPipelineInterface.dummy_data = {}

  def test_make_pokemon_train_iterator(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","resolution=768",
      "cache_latents_text_encoder_outputs=False","transform_images_num_proc=4"])
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()

    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    rng = jax.random.PRNGKey(config.seed)
    weight_dtype = max_utils.get_dtype(config)

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None
    )

    train_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      global_batch_size,
      pipeline,
      params,
      rng
    )

    data = train_iterator()
    device_count = jax.device_count()
    assert data["input_ids"].shape == (device_count,77)
    assert data["pixel_values"].shape == (device_count, 3, config.resolution, config.resolution)

  def test_make_pokemon_train_iterator_w_latents_caching(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","resolution=768",
      "cache_latents_text_encoder_outputs=True","transform_images_num_proc=4"])
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()

    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    rng = jax.random.PRNGKey(config.seed)
    weight_dtype = max_utils.get_dtype(config)

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None
    )

    train_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      global_batch_size,
      pipeline,
      params,
      rng
    )

    data = train_iterator()
    device_count = jax.device_count()

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

    assert data["input_ids"].shape == (device_count,77, 1024)
    assert data["pixel_values"].shape == (device_count,
                                          pipeline.unet.config.in_channels,
                                          config.resolution // vae_scale_factor,
                                          config.resolution // vae_scale_factor)

if __name__ == '__main__':
  absltest.main()
