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

import numpy as np

from .. import pyconfig
from absl.testing import absltest
from maxdiffusion.generate_flux import run as generate_flux
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from google.cloud import storage

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

JAX_CACHE_DIR = "gs://maxdiffusion-github-runner-test-assets/cache_dir"


def download_blob(gcs_file, local_file):
  gcs_dir_arr = gcs_file.replace("gs://", "").split("/")
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_dir_arr[0])
  blob = bucket.blob("/".join(gcs_dir_arr[1:]))
  blob.download_to_filename(local_file)


class GenerateFlux(unittest.TestCase):
  """Smoke test."""

  def setUp(self):
    GenerateFlux.dummy_data = {}

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_flux_dev(self):
    img_url = os.path.join(THIS_DIR, "images", "test_flux_dev.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_flux_dev.yml"),
            "run_name=flux_test",
            "output_dir=/tmp/",
            "jax_cache_dir=/tmp/cache_dir",
            'prompt="A cute corgi lives in a house made out of sushi, anime"',
        ],
        unittest=True,
    )

    images = generate_flux(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.80

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_flux_dev_lora(self):
    img_url = os.path.join(THIS_DIR, "images", "test_flux_dev_lora.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    gcs_lora_path = "gs://maxdiffusion-github-runner-test-assets/flux/lora/anime_lora.safetensors"
    local_path = "/tmp/anime_lora.safetensors"
    download_blob(gcs_lora_path, local_path)

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_flux_dev.yml"),
            "run_name=flux_test",
            "output_dir=/tmp/",
            "jax_cache_dir=/tmp/cache_dir",
            'prompt="A cute corgi lives in a house made out of sushi, anime"',
            'lora_config={"lora_model_name_or_path" : ["/tmp/anime_lora.safetensors"], "weight_name" : ["anime_lora.safetensors"], "adapter_name" : ["anime"], "scale": [0.8], "from_pt": ["true"]}',
        ],
        unittest=True,
    )

    images = generate_flux(pyconfig.config)
    test_image = np.array(images[1]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.80


if __name__ == "__main__":
  absltest.main()
