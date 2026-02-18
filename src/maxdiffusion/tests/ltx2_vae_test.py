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

import sys
import os
import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
import numpy as np
import unittest
from absl.testing import absltest

# Add maxdiffusion/src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import (
    LTX2VideoCausalConv3d,
    LTXVideoDownsampler3d,
    LTXVideoUpsampler3d,
    LTX2VideoResnetBlock3d,
    LTX2VideoEncoder3d,
    LTX2VideoDecoder3d,
    LTX2VideoAutoencoderKL,
    LTX2DiagonalGaussianDistribution
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class LTX2VaeTest(unittest.TestCase):

    def setUp(self):
        pyconfig.initialize(
            [
                None,
                os.path.join(THIS_DIR, "..", "configs", "ltx2_video.yml"),
            ],
            unittest=True,
        )
        self.config = pyconfig.config
        devices_array = create_device_mesh(self.config)
        self.mesh = Mesh(devices_array, self.config.mesh_axes)

    def test_ltx2_causal_conv3d(self):
        """Tests the causal padding constraint of LTX2VideoCausalConv3d."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        in_channels = 16
        out_channels = 32
        # (B, T, H, W, C)
        dummy_input = jnp.ones((2, 5, 16, 16, in_channels))
        
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            conv = LTX2VideoCausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                rngs=rngs,
                mesh=self.mesh
            )
            
            # Causal forward
            out_causal = conv(dummy_input, causal=True)
            # The spatial/temporal output should perfectly match input shapes (before stride changes)
            # because LTX-2 Conv internally pads to maintain dimension size
            self.assertEqual(out_causal.shape, (2, 5, 16, 16, out_channels))
            
            # Non-Causal forward
            out_non_causal = conv(dummy_input, causal=False)
            self.assertEqual(out_non_causal.shape, (2, 5, 16, 16, out_channels))

    def test_ltx2_video_downsampler3d(self):
        """Tests pooling and reshaping alignment in LTXVideoDownsampler3d."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        in_channels = 32
        out_channels = 64
        
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            downsampler = LTXVideoDownsampler3d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=(1, 2, 2), # Compress spatial, keep time identical
                rngs=rngs,
                mesh=self.mesh
            )
            
            # (B, T, H, W, C) -> T should remain 5, HW should halve from 16 to 8
            dummy_input = jnp.ones((1, 5, 16, 16, in_channels))
            out = downsampler(dummy_input, causal=True)
            
            self.assertEqual(out.shape, (1, 5, 8, 8, out_channels))

    def test_ltx2_video_upsampler3d(self):
        """Tests the tile duplication and causal shift of LTXVideoUpsampler3d."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        in_channels = 64
        out_channels = 32
        
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            upsampler = LTXVideoUpsampler3d(
                in_channels=in_channels,
                stride=(1, 2, 2), # Upscale spatial by 2, keep time identical
                rngs=rngs,
                mesh=self.mesh
            )
            
            # (B, T, H, W, C) -> T should remain 3, HW should double from 8 to 16
            dummy_input = jnp.ones((1, 3, 8, 8, in_channels))
            out = upsampler(dummy_input, causal=True)
            
            self.assertEqual(out.shape, (1, 3, 16, 16, in_channels))

    def test_ltx2_diagonal_gaussian_distribution(self):
        """Tests that the custom 129-channel distribution splits and reconstructs successfully."""
        B, T, H, W = 2, 4, 8, 8
        latent_channels = 128
        parameters_channels = 129 # 128 mean + 1 logvar
        
        # Mock moments tensor
        parameters = jnp.zeros((B, T, H, W, parameters_channels))
        parameters = parameters.at[..., :128].set(0.5) # Set mean to 0.5
        parameters = parameters.at[..., 128:].set(1.0) # Set logvar to 1.0
        
        dist = LTX2DiagonalGaussianDistribution(parameters, cls_latent_channels=latent_channels)
        
        # Verify splits
        self.assertEqual(dist.mean.shape, (B, T, H, W, 128))
        self.assertEqual(dist.logvar.shape, (B, T, H, W, 1))
        
        # Logvar mathematically broadcasts to variance
        self.assertEqual(dist.var.shape, (B, T, H, W, 128))
        
        # Sampling should return matching shapes
        key = jax.random.PRNGKey(0)
        sample = dist.sample(key=key)
        self.assertEqual(sample.shape, (B, T, H, W, 128))

    def test_ltx2_full_vae_encode_decode(self):
        """Tests a full, mini forward pass through the LTX-2 VAE hierarchy."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)
        
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            vae = LTX2VideoAutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=8,
                block_out_channels=(16, 32),
                decoder_block_out_channels=(16, 32),
                layers_per_block=(2, 2),
                decoder_layers_per_block=(2, 2, 2),
                patch_size=2,
                patch_size_t=1,
                rngs=rngs,
                mesh=self.mesh
            )
            
            B, T, H, W, C = 1, 3, 16, 16, 3
            dummy_video = jnp.ones((B, T, H, W, C))
            
            # Encode
            # Uses slice optimization based on `use_slicing`
            encoded_dist = vae.encode(dummy_video, return_dict=False)[0]
            latents = encoded_dist.sample(key=key)
            
            # Validate Downsampling Math:
            # Spatial halves twice: 16 -> 8 -> 4, then unpatchified by patch_size=2 : 4 -> 2
            # 16 / (2 downblocks * 2 patch) = 16 / 4 = 4
            self.assertEqual(latents.shape, (B, T, 4, 4, 8))
            
            # Decode
            decoded = vae.decode(latents, return_dict=False)[0]
            
            # Validate output matches original dimensions
            self.assertEqual(decoded.shape, (B, T, H, W, C))

if __name__ == "__main__":
    absltest.main()
