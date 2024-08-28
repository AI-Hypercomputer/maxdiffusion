import os
import unittest
import pytest
import numpy as np

from ..import pyconfig
from absl.testing import absltest
from maxdiffusion.generate import run as generate_run
from maxdiffusion.controlnet.generate_controlnet_replicated import run as generate_run_controlnet
from PIL import Image
from skimage.metrics import structural_similarity as ssim

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Generate(unittest.TestCase):
  """Smoke test."""

  def setUp(self):
    super().setUp()
    Generate.dummy_data = {}

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sd15_config(self):
    img_url = os.path.join(THIS_DIR,'images','test_gen_sd15.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      "seed=47","output_dir=gs://maxdiffusion-github-runner-test-assets",
      "run_name=gen-test-15-config"],unittest=True)
    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_sd_2_base_from_gcs(self):
    img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "seed=47", "from_pt=False",
      "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-2-base",
      "output_dir=gs://maxdiffusion-github-runner-test-assets","run_name=gen-test-sd2-base-config"],unittest=True)
    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_controlnet(self):
    img_url = os.path.join(THIS_DIR,'images','cnet_test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      "prompt=best quality, extremely detailed","activations_dtype=bfloat16","weights_dtype=bfloat16",
      "negative_prompt=monochrome, lowres, bad anatomy, worst quality, low quality",
      "num_inference_steps=50","seed=0","split_head_dim=False"],unittest=True)

    images = generate_run_controlnet(pyconfig.config)
    test_image = np.array(images[1]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

if __name__ == '__main__':
  absltest.main()
