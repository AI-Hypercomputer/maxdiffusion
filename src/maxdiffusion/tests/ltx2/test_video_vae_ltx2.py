"""
Copyright 2026 Google LLC

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
import functools
import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh
import unittest
from absl.testing import absltest
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.utils import load_video
from maxdiffusion.video_processor import VideoProcessor
from maxdiffusion.models.ltx2.ltx2_utils import load_vae_weights
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import (
    LTX2VideoCausalConv3d,
    LTX2VideoDownBlock3D,
    LTX2VideoUpBlock3d,
    LTX2VideoAutoencoderKL,
)

from maxdiffusion.models.vae_flax import FlaxDiagonalGaussianDistribution

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class LTX2VaeTest(unittest.TestCase):

  def setUp(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "..", "configs", "ltx2_video.yml"),
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
          in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, rngs=rngs, mesh=self.mesh
      )

      # Causal forward
      out_causal = conv(dummy_input, causal=True)
      self.assertEqual(out_causal.shape, (2, 5, 16, 16, out_channels))

      # Non-Causal forward
      out_non_causal = conv(dummy_input, causal=False)
      self.assertEqual(out_non_causal.shape, (2, 5, 16, 16, out_channels))

  def test_ltx2_video_downblock3d(self):
    """Tests pooling and reshaping alignment in LTX2VideoDownBlock3D."""
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)

    in_channels = 32
    out_channels = 64

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      downsampler = LTX2VideoDownBlock3D(
          in_channels=in_channels,
          out_channels=out_channels,
          num_layers=1,
          resnet_eps=1e-6,
          spatio_temporal_scale=True,
          downsample_type="spatial",
          rngs=rngs,
          mesh=self.mesh,
      )

      # (B, T, H, W, C) -> T should remain 5, HW should halve from 16 to 8
      dummy_input = jnp.ones((1, 5, 16, 16, in_channels))
      out = downsampler(dummy_input, causal=True)

      self.assertEqual(out.shape, (1, 5, 8, 8, out_channels))

  def test_ltx2_video_upblock3d(self):
    """Tests the tile duplication and causal shift of LTX2VideoUpBlock3d."""
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)

    in_channels = 64
    out_channels = 32

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      upsampler = LTX2VideoUpBlock3d(
          in_channels=in_channels,
          out_channels=out_channels,
          num_layers=1,
          resnet_eps=1e-6,
          spatio_temporal_scale=True,
          upsample_residual=False,
          upscale_factor=1,
          rngs=rngs,
          mesh=self.mesh,
      )

      dummy_input = jnp.ones((1, 3, 8, 8, in_channels))
      out = upsampler(dummy_input, causal=True)

      self.assertEqual(out.shape, (1, 5, 16, 16, out_channels))

  def test_ltx2_diagonal_gaussian_distribution(self):
    """Tests that the custom distribution splits and reconstructs successfully."""
    B, T, H, W = 2, 4, 8, 8
    parameters_channels = 256  # 128 mean + 128 logvar

    # Mock moments tensor
    parameters = jnp.zeros((B, T, H, W, parameters_channels))
    parameters = parameters.at[..., :128].set(0.5)  # Set mean to 0.5
    parameters = parameters.at[..., 128:].set(1.0)  # Set logvar to 1.0

    dist = FlaxDiagonalGaussianDistribution(parameters)

    # Verify splits
    self.assertEqual(dist.mean.shape, (B, T, H, W, 128))
    self.assertEqual(dist.logvar.shape, (B, T, H, W, 128))

    # Logvar mathematically broadcasts to variance during sampling
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
          mesh=self.mesh,
      )

      B, T, H, W, C = 1, 9, 16, 16, 3
      dummy_video = jnp.ones((B, T, H, W, C))

      # Encode
      encoded_dist = vae.encode(dummy_video, return_dict=False)[0]
      latents = encoded_dist.sample(key=key)

      self.assertEqual(latents.shape, (B, 5, 4, 4, 8))

      # Decode
      decoded = vae.decode(latents, return_dict=False)[0]
      self.assertEqual(decoded.shape, (B, 17, 32, 32, C))

  def test_ltx2_tiled_encode_decode(self):
    """Tests the spatial tiled encode/decode logic for large resolutions."""
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
          spatio_temporal_scaling=(True, True),
          decoder_spatio_temporal_scaling=(True, True),
          downsample_type=("spatial", "spatial"),
          upsample_factor=(2, 2),
          upsample_residual=(True, True),
          rngs=rngs,
          mesh=self.mesh,
      )
      # Tiling boundaries natively
      # Spatial compression = patch_size(2) * 2**2 = 8
      vae.tile_sample_min_height = 24
      vae.tile_sample_min_width = 24
      vae.tile_sample_stride_height = 16
      vae.tile_sample_stride_width = 16
      vae.tile_latent_min_height = 3
      vae.tile_latent_min_width = 3
      vae.tile_latent_stride_height = 2
      vae.tile_latent_stride_width = 2
      vae.enable_tiling()

      # Test encode with tiling
      B, T, H, W, C = 1, 9, 32, 32, 3
      dummy_video = jnp.ones((B, T, H, W, C))

      encoded_dist = vae.encode(dummy_video, return_dict=False)[0]
      latents = encoded_dist.sample(key=key)

      # Spatial downsample factor is 2 * 2**2 = 8.
      # So 32 -> 4 (overlapping 4x4 effectively)
      self.assertEqual(latents.shape, (B, 9, 4, 4, 8))

      # Test decode with tiling
      decoded = vae.decode(latents, return_dict=False)[0]
      self.assertEqual(decoded.shape, (B, 33, 32, 32, C))

  def test_ltx2_temporal_tiled_encode_decode(self):
    """Tests the temporal tiled encode/decode logic (framewise decoding/encoding)."""
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
          spatio_temporal_scaling=(True, True),
          decoder_spatio_temporal_scaling=(True, True),
          downsample_type=("temporal", "temporal"),
          upsample_factor=(2, 2),
          upsample_residual=(True, True),
          rngs=rngs,
          mesh=self.mesh,
      )
      # Temporal compression natively = 1 * 2**2 = 4
      # Temporal boundaries natively
      # The total temporal stride down is `4` (2 * 2**1 blocks) based on `decoder_spatio_temporal_scaling`.
      vae.tile_sample_min_num_frames = 16
      vae.tile_sample_stride_num_frames = 8
      vae.use_framewise_decoding = True

      # Test 2 chunks: length = stride * chunks + overlap
      B, T, H, W, C = 1, 25, 16, 16, 3
      dummy_video = jnp.ones((B, T, H, W, C))

      encoded_dist = vae.encode(dummy_video, return_dict=False)[0]
      latents = encoded_dist.sample(key=key)

      self.assertEqual(latents.shape, (B, 7, 8, 8, 8))

      decoded = vae.decode(latents, return_dict=False)[0]
      self.assertEqual(decoded.shape, (B, 25, 64, 64, C))

  def test_load_checkpoint(self):
    def vae_encode(video, vae, key):
      latent = vae.encode(video, return_dict=False)[0]
      latent = latent.sample(key)
      return latent

    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "..", "configs", "ltx2_video.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      vae = LTX2VideoAutoencoderKL(
          rngs=rngs,
          in_channels=3,
          out_channels=3,
          latent_channels=128,
          block_out_channels=(256, 512, 1024, 2048),
          decoder_block_out_channels=(256, 512, 1024),
          layers_per_block=(4, 6, 6, 2, 2),
          decoder_layers_per_block=(5, 5, 5, 5),
          spatio_temporal_scaling=(True, True, True, True),
          decoder_spatio_temporal_scaling=(True, True, True),
          decoder_inject_noise=(False, False, False, False),
          downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
          upsample_residual=(True, True, True),
          upsample_factor=(2, 2, 2),
          mesh=mesh,
      )

    video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
    video = load_video(video_path)

    vae_scale_factor_spatial = 32
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    width, height = video[0].size
    video = video_processor.preprocess_video(video, height=height, width=width)
    original_video = jnp.array(np.array(video), dtype=jnp.bfloat16)

    video_input = jnp.transpose(original_video, (0, 2, 3, 4, 1))

    graphdef, state = nnx.split(vae)
    eval_shapes = state.to_pure_dict()
    pretrained_model_name_or_path = "Lightricks/LTX-2"
    loaded_weights = load_vae_weights(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        eval_shapes=eval_shapes,
        device="cpu",
        hf_download=True,
    )

    filtered_eval_shapes = {}
    flat_eval_shapes = flatten_dict(eval_shapes)
    flat_loaded = flatten_dict(loaded_weights)
    for k, v in flat_eval_shapes.items():
      k_str = [str(x) for x in k]
      if "dropout" in k_str or "rngs" in k_str:
        filtered_eval_shapes[k] = v
      else:
        filtered_eval_shapes[k] = flat_loaded[k]

    new_state = unflatten_dict(filtered_eval_shapes)

    def cast_to_bf16(x):
      if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.bfloat16)
      return x

    params = jax.tree_util.tree_map(cast_to_bf16, new_state)
    vae = nnx.merge(graphdef, params)

    p_vae_encode = functools.partial(vae_encode, vae=vae, key=key)
    original_video_shape = original_video.shape
    latent = p_vae_encode(video_input)

    jitted_decode = functools.partial(vae.decode, return_dict=False)
    video_out = jitted_decode(latent)[0]
    video_out = jnp.transpose(video_out, (0, 4, 1, 2, 3))
    self.assertEqual(video_out.shape, original_video_shape)

    original_video = torch.from_numpy(np.array(original_video.astype(jnp.float32))).to(dtype=torch.bfloat16)
    video_out = torch.from_numpy(np.array(video_out.astype(jnp.float32))).to(dtype=torch.bfloat16)
    video_out = video_processor.postprocess_video(video_out, output_type="np")
    original_video = video_processor.postprocess_video(original_video, output_type="np")
    ssim_compare = ssim(video_out[0], original_video[0], multichannel=True, channel_axis=-1, data_range=255)
    self.assertGreaterEqual(ssim_compare, 0.998)


if __name__ == "__main__":
  absltest.main()
