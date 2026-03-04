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

from jax.sharding import Mesh

from .. import pyconfig
from maxdiffusion.max_utils import (
    create_device_mesh,
    get_flash_block_sizes,
)
from maxdiffusion import (FlaxStableDiffusionXLPipeline, FlaxDDIMScheduler, FlaxDDPMScheduler, maxdiffusion_utils)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class MaxDiffusionUtilsTest(unittest.TestCase):
  """Test maxdiffusion_utils.py functions"""

  def setUp(self):
    MaxDiffusionUtilsTest.dummy_data = {}

  def test_create_scheduler(self):
    """Test create scheduler with different schedulers"""
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config

    # Setup Mesh
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    flash_block_sizes = get_flash_block_sizes(config)

    pipeline, _ = FlaxStableDiffusionXLPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        revision=config.revision,
        dtype=config.activations_dtype,
        split_head_dim=config.split_head_dim,
        norm_num_groups=config.norm_num_groups,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        text_encoder_2=None,
        unet=None,
    )
    scheduler_config = pipeline.scheduler.config

    assert scheduler_config["prediction_type"] == "epsilon"
    assert not scheduler_config.get("rescale_zero_terminal_snr", False)
    assert scheduler_config["timestep_spacing"] == "leading"

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"

    # Test class name override without Flax Name.
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"_class_name" : "DDIMScheduler", "prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config
    scheduler_config = scheduler.config

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"
    assert type(scheduler) is FlaxDDIMScheduler

    # Test class name override with Flax Name.
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"_class_name" : "FlaxDDPMScheduler", "prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config
    scheduler_config = scheduler.config

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"
    assert type(scheduler) is FlaxDDPMScheduler
