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
import unittest
import pytest
from absl.testing import absltest

import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from maxdiffusion import FlaxAutoencoderKL
from skimage.metrics import structural_similarity as ssim

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class VaeTest(unittest.TestCase):
  """Test Vae"""

  def setUp(self):
    VaeTest.dummy_data = {}

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_flux_vae(self):
    img_url = os.path.join(THIS_DIR, "images", "test_hyper_sdxl.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    img_min = np.min(base_image)
    img_max = np.max(base_image)
    image = (base_image - img_min) / (img_max - img_min)
    image = 2.0 * image - 1.0
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))  # (1, 3, 1024, 1024), BCWH
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="vae", from_pt=True, use_safetensors=True, dtype="bfloat16"
    )

    encoded_image = vae.apply({"params": vae_params}, image, deterministic=True, method=vae.encode)
    latents = encoded_image[0].sample(jax.random.key(0))
    latents = jnp.transpose(latents, (0, 3, 1, 2))

    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

    assert latents.shape == (1, 16, 128, 128)

    # decode back
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.apply({"params": vae_params}, latents, deterministic=True, method=vae.decode).sample[0]
    image = np.array(image)
    image = (image * 0.5 + 0.5).clip(0, 1)
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(image * 255)
    ssim_compare = ssim(base_image, image, multichannel=True, channel_axis=-1, data_range=255)
    assert ssim_compare >= 0.90


if __name__ == "__main__":
  absltest.main()
