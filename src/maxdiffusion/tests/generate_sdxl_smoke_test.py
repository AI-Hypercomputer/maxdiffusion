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
import pytest

import numpy as np

from .. import pyconfig
from absl.testing import absltest
from maxdiffusion.generate_sdxl import run as generate_run_xl
from PIL import Image
from skimage.metrics import structural_similarity as ssim

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

JAX_CACHE_DIR = "gs://maxdiffusion-github-runner-test-assets/cache_dir"


class Generate(unittest.TestCase):
  """Smoke test."""

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_hyper_sdxl_lora(self):
    img_url = os.path.join(THIS_DIR, "images", "test_hyper_sdxl.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "output_dir=gs://maxdiffusion-github-runner-test-assets",
            "run_name=test-hypersdxl-lora",
            "num_inference_steps=2",
            "per_device_batch_size=1",
            "do_classifier_free_guidance=False",
            'diffusion_scheduler_config={"_class_name" : "FlaxDDIMScheduler", "timestep_spacing" : "trailing"}',
            'lora_config={"lora_model_name_or_path" : ["ByteDance/Hyper-SD"], "weight_name" : ["Hyper-SDXL-2steps-lora.safetensors"], "adapter_name" : ["hyper-sdxl"], "scale": [0.7], "from_pt": ["true"]}',
            f"jax_cache_dir={JAX_CACHE_DIR}",
        ],
        unittest=True,
    )
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.80

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sdxl_config(self):
    img_url = os.path.join(THIS_DIR, "images", "test_sdxl.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "resolution=1024",
            "prompt=A magical castle in the middle of a forest, artistic drawing",
            "negative_prompt=purple, red",
            "guidance_scale=9",
            "num_inference_steps=20",
            "seed=47",
            "per_device_batch_size=1",
            "run_name=sdxl-inference-test",
            "split_head_dim=False",
            f"jax_cache_dir={JAX_CACHE_DIR}",
        ],
        unittest=True,
    )
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.80

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sdxl_from_gcs(self):
    """Verify load weights from gcs."""
    img_url = os.path.join(THIS_DIR, "images", "test_sdxl.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=main",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "resolution=1024",
            "prompt=A magical castle in the middle of a forest, artistic drawing",
            "negative_prompt=purple, red",
            "guidance_scale=9",
            "num_inference_steps=20",
            "seed=47",
            "per_device_batch_size=1",
            "run_name=sdxl-inference-test",
            "split_head_dim=False",
            f"jax_cache_dir={JAX_CACHE_DIR}",
        ],
        unittest=True,
    )
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.80

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_controlnet_sdxl(self):
    from maxdiffusion.controlnet.generate_controlnet_sdxl_replicated import run as generate_run_sdxl_controlnet

    img_url = os.path.join(THIS_DIR, "images", "cnet_test_sdxl.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "activations_dtype=bfloat16",
            "weights_dtype=bfloat16",
            f"jax_cache_dir={JAX_CACHE_DIR}",
        ],
        unittest=True,
    )
    images = generate_run_sdxl_controlnet(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.70

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sdxl_lightning(self):
    img_url = os.path.join(THIS_DIR, "images", "test_lightning.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl_lightning.yml"),
            "run_name=sdxl-lightning-test",
            f"jax_cache_dir={JAX_CACHE_DIR}",
        ],
        unittest=True,
    )
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.70


if __name__ == "__main__":
  absltest.main()
