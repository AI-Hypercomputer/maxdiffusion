import os
import unittest

import jax
from ..import pyconfig
from ..import max_utils
from ..models.train import calculate_training_tflops
from absl.testing import absltest
from maxdiffusion import FlaxStableDiffusionPipeline


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class FlopCalculation(unittest.TestCase):
  def test_sd2_base(self):
    scale_factor=0.67
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
                         "per_device_batch_size=1"])
    config = pyconfig.config
    weight_dtype = max_utils.get_dtype(config)

    config = pyconfig.config
    weight_dtype = max_utils.get_dtype(config)

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt,
        split_head_dim=config.split_head_dim
    )
    unet_param_count = sum(x.size for x in jax.tree_util.tree_leaves(params["unet"]))

    vae_scale_factor = 2 ** (len(pipeline.vae.config['block_out_channels']) - 1)
    embedding_dim = config.resolution // vae_scale_factor
    calculated_tflops = scale_factor * embedding_dim**2 * config.per_device_batch_size * unet_param_count / 10**12
    training_tflops = calculate_training_tflops(pipeline, params["unet"], config)

    # 5 percent error tolerance
    assert abs(1 -(training_tflops/calculated_tflops)) * 100 < 5

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
                         "per_device_batch_size=2"])

    calculated_tflops = scale_factor * embedding_dim**2 * config.per_device_batch_size * unet_param_count / 10**12
    training_tflops = calculate_training_tflops(pipeline, params["unet"], config)

    assert abs(1 -(training_tflops/calculated_tflops)) * 100 < 5

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
                         "per_device_batch_size=4"])

    calculated_tflops = scale_factor * embedding_dim**2 * config.per_device_batch_size * unet_param_count / 10**12
    training_tflops = calculate_training_tflops(pipeline, params["unet"], config)

    assert abs(1 -(training_tflops/calculated_tflops)) * 100 < 5

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
                         "per_device_batch_size=8"])

    calculated_tflops = scale_factor * embedding_dim**2 * config.per_device_batch_size * unet_param_count / 10**12
    training_tflops = calculate_training_tflops(pipeline, params["unet"], config)

    assert abs(1 -(training_tflops/calculated_tflops)) * 100 < 5

if __name__ == '__main__':
  absltest.main()

