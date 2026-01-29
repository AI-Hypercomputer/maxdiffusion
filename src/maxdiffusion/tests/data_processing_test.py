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
import pytest
import functools
import jax
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from .. import pyconfig
from ..max_utils import (
    create_device_mesh,
)
import numpy as np
import unittest
from ..data_preprocessing.wan_txt2vid_data_preprocessing import vae_encode
from ..checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from ..utils import load_video
from ..video_processor import VideoProcessor
import flax

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_T = 2

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

flax.config.update("flax_always_shard_variable", False)


class DataProcessingTest(unittest.TestCase):

  def setUp(self):
    DataProcessingTest.dummy_data = {}
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    self.config = config
    devices_array = create_device_mesh(config)
    self.mesh = Mesh(devices_array, config.mesh_axes)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_wan_vae_encode_normalization(self):
    """Test wan vae encode function normalization"""
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    checkpoint_loader = WanCheckpointer2_1(config=config)
    pipeline, _, _ = checkpoint_loader.load_checkpoint()

    vae_scale_factor_spatial = 2 ** len(pipeline.vae.temperal_downsample)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)

    video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
    video = load_video(video_path)
    videos = [video_processor.preprocess_video([video], height=config.height, width=config.width)]
    videos = jnp.array(np.squeeze(np.array(videos), axis=1), dtype=config.weights_dtype)
    p_vae_encode = jax.jit(functools.partial(vae_encode, vae=pipeline.vae, vae_cache=pipeline.vae_cache))

    rng = jax.random.key(config.seed)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      latents = p_vae_encode(videos, rng=rng)
    # 1. Verify Channel Count (Wan 2.1 requires 16)
    self.assertEqual(latents.shape[1], 16, f"Expected 16 channels, got {latents.shape[1]}")

    # 2. Verify Global Stats
    # We expect mean near 0 and variance near 1.
    # We use a threshold (e.g., 0.15) since this is just one video.
    global_mean = jnp.mean(latents)
    global_var = jnp.var(latents)

    self.assertLess(abs(global_mean), 0.2, f"Global mean {global_mean} is too far from 0")
    self.assertAlmostEqual(global_var, 1.0, delta=0.2, msg=f"Global variance {global_var} is too far from 1.0")

    # 3. Verify Channel-wise Range
    # Ensure no channel is completely "dead" or "exploding"
    channel_vars = jnp.var(latents, axis=(0, 2, 3, 4))
    self.assertTrue(jnp.all(channel_vars > 0.1), "One or more channels have near-zero variance")
    self.assertTrue(jnp.all(channel_vars < 5.0), "One or more channels have exploding variance")
