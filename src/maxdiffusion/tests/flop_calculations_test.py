import os
import unittest

import jax.numpy as jnp
from ..import pyconfig
from absl.testing import absltest
from maxdiffusion import FlaxUNet2DConditionModel, FlaxAutoencoderKL
from ..import flop_calc_utils

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class FlopCalculation(unittest.TestCase):
  def test_sd2_base(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml')])

    config = pyconfig.config

    vae_config = FlaxAutoencoderKL.load_config(
      config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae"
    )
    vae_scale_factor = 2 ** (len(vae_config['block_out_channels']) - 1)

    unet_config = FlaxUNet2DConditionModel.load_config(
      config.pretrained_model_name_or_path, revision=config.revision, subfolder="unet"
    )
    input_shape = (config.per_device_batch_size,
                   unet_config['in_channels'],
                   config.resolution / vae_scale_factor,
                   config.resolution / vae_scale_factor)

    print(flop_calc_utils.calculate_unet_flops(config, input_shape, unet_config))
    breakpoint()

if __name__ == '__main__':
  absltest.main()