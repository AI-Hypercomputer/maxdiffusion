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

""" Smoke test """
import os
import pytest
import pathlib
import shutil
import unittest
from maxdiffusion import max_utils
from maxdiffusion.train import train as train_orbax_main
from maxdiffusion.train_sdxl import train as train_sdxl
from maxdiffusion.dreambooth.train_dreambooth import train as train_orbax_dreambooth
from ..import pyconfig
from maxdiffusion.generate import run as generate_run
from absl.testing import absltest
from google.cloud import storage
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

from maxdiffusion.train_utils import (
    validate_train_config,
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

HOME_DIR = pathlib.Path.home()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def cleanup(output_dir):
  shutil.rmtree(output_dir)

def delete_blobs(gcs_dir):
  gcs_dir_arr = gcs_dir.replace("gs://","").split("/")
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_dir_arr[0])
  folder = "/".join(gcs_dir_arr[1:])
  blobs = bucket.list_blobs(prefix=folder)
  for blob in blobs:
    blob.delete()

class Train(unittest.TestCase):
  """Smoke test."""
  def setUp(self):
    Train.dummy_data = {}

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sdxl_config(self):
    output_dir="gs://maxdiffusion-github-runner-test-assets"
    run_name="sdxl_train_smoke_test"
    cache_dir="gs://maxdiffusion-github-runner-test-assets/cache_dir"

    delete_blobs(os.path.join(output_dir,run_name))

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
      "revision=refs/pr/95","activations_dtype=bfloat16","weights_dtype=bfloat16",f"run_name={run_name}",
      "max_train_steps=21","dataset_name=diffusers/pokemon-gpt4-captions",
      "resolution=1024","per_device_batch_size=1","snr_gamma=5.0",
      "per_device_batch_size=1",
      'timestep_bias={"strategy" : "later", "multiplier" : 2.0, "portion" : 0.25}',
      f"output_dir={output_dir}",
      f"jax_cache_dir={cache_dir}"], unittest=True)

    train_sdxl(pyconfig.config)

    delete_blobs(os.path.join(output_dir,run_name))

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_dreambooth_orbax(self):
    num_class_images=100
    output_dir="gs://maxdiffusion-github-runner-test-assets"
    run_name="dreambooth_orbax_smoke_test"
    cache_dir="gs://maxdiffusion-github-runner-test-assets/cache_dir"
    instance_class_gcs_dir="gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/instance_class"
    class_class_gcs_dir="gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/class_class"
    local_dir="/tmp/"

    delete_blobs(os.path.join(output_dir,run_name))

    instance_class_local_dir = max_utils.download_blobs(instance_class_gcs_dir, local_dir)
    class_class_local_dir = max_utils.download_blobs(class_class_gcs_dir, local_dir)

    pyconfig.initialize([None, os.path.join(THIS_DIR,'..','configs','base15.yml'),
      f"instance_data_dir={instance_class_local_dir}",
      f"class_data_dir={class_class_local_dir}","instance_prompt=a photo of ohwx dog",
      "class_prompt=photo of a dog","max_train_steps=150",f"jax_cache_dir={cache_dir}",
      "class_prompt=a photo of a dog", "activations_dtype=bfloat16",
      "train_text_encoder=True","text_encoder_learning_rate=4e-6",
      "cache_latents_text_encoder_outputs=False",
      "weights_dtype=float32","per_device_batch_size=1","enable_profiler=False","precision=DEFAULT",
      "cache_dreambooth_dataset=False","learning_rate=4e-6",f"output_dir={output_dir}",
      f"num_class_images={num_class_images}",f"run_name={run_name}",
      "prompt=a photo of ohwx dog", "seed=47"], unittest=True)

    config = pyconfig.config
    validate_train_config(config)
    train_orbax_dreambooth(config)

    cleanup(instance_class_local_dir)
    cleanup(class_class_local_dir)
    delete_blobs(os.path.join(output_dir,run_name))

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sd15_orbax(self):
    output_dir="gs://maxdiffusion-github-runner-test-assets"
    run_name="sd15_orbax_smoke_test"
    cache_dir="gs://maxdiffusion-github-runner-test-assets/cache_dir"

    delete_blobs(os.path.join(output_dir,run_name))

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      f"run_name={run_name}", "checkpoint_every=256",
      "max_train_steps=21","per_device_batch_size=8",
      f"output_dir={output_dir}", "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47",f"jax_cache_dir={cache_dir}"], unittest=True)

    config = pyconfig.config
    validate_train_config(config)
    train_orbax_main(config)

    img_url = os.path.join(THIS_DIR,'images','test_sd15.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

    delete_blobs(os.path.join(output_dir,run_name))

if __name__ == '__main__':
  absltest.main()
