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

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import unittest
from absl.testing import absltest
from skimage.metrics import structural_similarity as ssim
from ..models.wan.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanUpsample,
    AutoencoderKLWan,
    WanMidBlock,
    WanResidualBlock,
    WanRMS_norm,
    WanResample,
    ZeroPaddedConv2D,
    WanAttentionBlock,
    AutoencoderKLWanCache,
)
from ..models.wan.wan_utils import load_wan_vae
from ..utils import load_video
from ..video_processor import VideoProcessor

CACHE_T = 2


class TorchWanRMS_norm(nn.Module):
  r"""
  A custom RMS normalization layer.

  Args:
      dim (int): The number of dimensions to normalize over.
      channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
          Default is True.
      images (bool, optional): Whether the input represents image data. Default is True.
      bias (bool, optional): Whether to include a learnable bias term. Default is False.
  """

  def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
    super().__init__()
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)

    self.channel_first = channel_first
    self.scale = dim**0.5
    self.gamma = nn.Parameter(torch.ones(shape))
    self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

  def forward(self, x):
    return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class TorchWanResample(nn.Module):
  r"""
  A custom resampling module for 2D and 3D data.

  Args:
      dim (int): The number of input/output channels.
      mode (str): The resampling mode. Must be one of:
          - 'none': No resampling (identity operation).
          - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
          - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
          - 'downsample2d': 2D downsampling with zero-padding and convolution.
          - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
  """

  def __init__(self, dim: int, mode: str) -> None:
    super().__init__()
    self.dim = dim
    self.mode = mode

    # layers
    if mode == "upsample2d":
      self.resample = nn.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
      )
    elif mode == "upsample3d":
      raise Exception("downsample3d not supported")

    elif mode == "downsample2d":
      self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
    elif mode == "downsample3d":
      raise Exception("downsample3d not supported")
    else:
      self.resample = nn.Identity()

  def forward(self, x, feat_cache=None, feat_idx=[0]):
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = "Rep"
          feat_idx[0] += 1
        else:
          cache_x = x[:, :, -CACHE_T:, :, :].clone()
          if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
          if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
            cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
          if feat_cache[idx] == "Rep":
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache[idx] = cache_x
          feat_idx[0] += 1

          x = x.reshape(b, 2, c, t, h, w)
          x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
          x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = x.clone()
          feat_idx[0] += 1
        else:
          cache_x = x[:, :, -1:, :, :].clone()
          x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
          feat_cache[idx] = cache_x
          feat_idx[0] += 1
    return x


class WanVaeTest(unittest.TestCase):

  def setUp(self):
    WanVaeTest.dummy_data = {}

  def test_wanrms_norm(self):
    """Test against the Pytorch implementation"""

    # --- Test Case 1: images == True ---
    dim = 96
    input_shape = (1, 96, 1, 480, 720)

    model = TorchWanRMS_norm(dim)
    input = torch.ones(input_shape)
    torch_output = model(input)
    torch_output_np = torch_output.detach().numpy()

    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    wanrms_norm = WanRMS_norm(dim=dim, rngs=rngs)
    dummy_input = jnp.ones(input_shape)
    output = wanrms_norm(dummy_input)
    output_np = np.array(output)
    assert np.allclose(output_np, torch_output_np) is True

    # --- Test Case 2: images == False ---
    model = TorchWanRMS_norm(dim, images=False)
    input = torch.ones(input_shape)
    torch_output = model(input)
    torch_output_np = torch_output.detach().numpy()

    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    wanrms_norm = WanRMS_norm(dim=dim, rngs=rngs, images=False)
    dummy_input = jnp.ones(input_shape)
    output = wanrms_norm(dummy_input)
    output_np = np.array(output)
    assert np.allclose(output_np, torch_output_np) is True

  def test_zero_padded_conv(self):

    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    dim = 96
    kernel_size = 3
    stride = (2, 2)
    resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, kernel_size, stride=stride))
    input_shape = (1, 96, 480, 720)
    input = torch.ones(input_shape)
    output_torch = resample(input)
    assert output_torch.shape == (1, 96, 240, 360)

    model = ZeroPaddedConv2D(dim=dim, rngs=rngs, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    dummy_input = jnp.ones(input_shape)
    dummy_input = jnp.transpose(dummy_input, (0, 2, 3, 1))
    output = model(dummy_input)
    output = jnp.transpose(output, (0, 3, 1, 2))
    assert output.shape == (1, 96, 240, 360)

  def test_wan_upsample(self):
    batch_size = 1
    in_depth, in_height, in_width = 10, 32, 32
    in_channels = 3

    dummy_input = jnp.ones((batch_size * in_depth, in_height, in_width, in_channels))

    upsample = WanUpsample(scale_factor=(2.0, 2.0))

    # --- Test Case 1: depth > 1 ---
    output = upsample(dummy_input)
    assert output.shape == (10, 64, 64, 3)

  def test_wan_resample(self):
    # TODO - needs to test all modes - upsample2d, upsample3d, downsample2d, downsample3d and identity
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    # --- Test Case 1: downsample2d ---
    batch = 1
    dim = 96
    t = 1
    h = 480
    w = 720
    mode = "downsample2d"
    input_shape = (batch, dim, t, h, w)
    dummy_input = torch.ones(input_shape)
    torch_wan_resample = TorchWanResample(dim=dim, mode=mode)
    torch_output = torch_wan_resample(dummy_input)
    assert torch_output.shape == (batch, dim, t, h // 2, w // 2)

    wan_resample = WanResample(dim, mode=mode, rngs=rngs)
    # channels is always last here
    input_shape = (batch, t, h, w, dim)
    dummy_input = jnp.ones(input_shape)
    output = wan_resample(dummy_input)
    assert output.shape == (batch, t, h // 2, w // 2, dim)

  def test_3d_conv(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch_size = 1
    in_depth, in_height, in_width = 10, 32, 32
    in_channels = 3
    out_channels = 16
    kernel_d, kernel_h, kernel_w = 3, 3, 3  # Kernel size (Depth, Height, Width)
    padding_d, padding_h, padding_w = 1, 1, 1  # Base padding (Depth, Height, Width)

    # Create dummy input data
    dummy_input = jnp.ones((batch_size, in_depth, in_height, in_width, in_channels))

    # Create dummy cache data (from a previous step)
    cache_depth = 2 * padding_d
    dummy_cache = jnp.zeros((batch_size, cache_depth, in_height, in_width, in_channels))

    # Instantiate the module
    causal_conv_layer = WanCausalConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_d, kernel_h, kernel_w),
        padding=(padding_d, padding_h, padding_w),
        rngs=rngs,  # Pass rngs for initialization
    )

    # --- Test Case 1: No Cache ---
    output_no_cache = causal_conv_layer(dummy_input)
    assert output_no_cache.shape == (1, 10, 32, 32, 16)

    # --- Test Case 2: With Cache ---
    output_with_cache = causal_conv_layer(dummy_input, cache_x=dummy_cache)
    assert output_with_cache.shape == (1, 10, 32, 32, 16)

    # --- Test Case 3: With Cache larger than padding ---
    larger_cache_depth = 4  # Larger than needed padding (2*padding_d = 2)
    dummy_larger_cache = jnp.zeros((batch_size, larger_cache_depth, in_height, in_width, in_channels))
    output_with_larger_cache = causal_conv_layer(dummy_input, cache_x=dummy_larger_cache)
    assert output_with_larger_cache.shape == (1, 10, 32, 32, 16)

  def test_wan_residual(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    # --- Test Case 1: same in/out dim ---
    in_dim = out_dim = 96
    batch = 1
    t = 1
    height = 480
    width = 720
    dim = 96
    input_shape = (batch, t, height, width, dim)
    expected_output_shape = (batch, t, height, width, dim)

    wan_residual_block = WanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        rngs=rngs,
    )
    dummy_input = jnp.ones(input_shape)
    dummy_output = wan_residual_block(dummy_input)
    assert dummy_output.shape == expected_output_shape

    # --- Test Case 1: different in/out dim ---
    in_dim = 96
    out_dim = 196
    expected_output_shape = (batch, t, height, width, out_dim)

    wan_residual_block = WanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        rngs=rngs,
    )
    dummy_input = jnp.ones(input_shape)
    dummy_output = wan_residual_block(dummy_input)
    assert dummy_output.shape == expected_output_shape

  def test_wan_attention(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    dim = 384
    batch = 1
    t = 1
    height = 60
    width = 90
    input_shape = (batch, t, height, width, dim)
    wan_attention = WanAttentionBlock(dim=dim, rngs=rngs)
    dummy_input = jnp.ones(input_shape)
    output = wan_attention(dummy_input)
    assert output.shape == input_shape

  def test_wan_midblock(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch = 1
    t = 1
    dim = 384
    height = 60
    width = 90
    input_shape = (batch, t, height, width, dim)
    wan_midblock = WanMidBlock(dim=dim, rngs=rngs)
    dummy_input = jnp.ones(input_shape)
    output = wan_midblock(dummy_input)
    assert output.shape == input_shape

  def test_wan_decode(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales = []
    temperal_downsample = [False, True, True]
    wan_vae = AutoencoderKLWan(
        rngs=rngs,
        base_dim=dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
    )
    vae_cache = AutoencoderKLWanCache(wan_vae)
    batch = 1
    t = 13
    channels = 16
    height = 60
    width = 90
    input_shape = (batch, t, height, width, channels)
    input = jnp.ones(input_shape)

    latents_mean = jnp.array(wan_vae.latents_mean).reshape(1, 1, 1, 1, wan_vae.z_dim)
    latents_std = 1.0 / jnp.array(wan_vae.latents_std).reshape(1, 1, 1, 1, wan_vae.z_dim)
    input = input / latents_std + latents_mean
    dummy_output = wan_vae.decode(input, feat_cache=vae_cache)
    assert dummy_output.sample.shape == (batch, 49, 480, 720, 3)

  def test_wan_encode(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales = []
    temperal_downsample = [False, True, True]
    wan_vae = AutoencoderKLWan(
        rngs=rngs,
        base_dim=dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
    )
    vae_cache = AutoencoderKLWanCache(wan_vae)
    batch = 1
    channels = 3
    t = 49
    height = 480
    width = 720
    input_shape = (batch, channels, t, height, width)
    input = jnp.ones(input_shape)
    output = wan_vae.encode(input, feat_cache=vae_cache)
    assert output.latent_dist.sample(key).shape == (1, 13, 60, 90, 16)

  def test_load_checkpoint(self):
    def vae_encode(video, wan_vae, vae_cache, key):
      latent = wan_vae.encode(video, feat_cache=vae_cache)
      latent = latent.latent_dist.sample(key)
      return latent

    pretrained_model_name_or_path = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    wan_vae = AutoencoderKLWan.from_config(pretrained_model_name_or_path, subfolder="vae", rngs=rngs)
    vae_cache = AutoencoderKLWanCache(wan_vae)
    video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
    video = load_video(video_path)

    vae_scale_factor_spatial = 2 ** len(wan_vae.temperal_downsample)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    width, height = video[0].size
    video = video_processor.preprocess_video(video, height=height, width=width)  # .to(dtype=jnp.float32)
    original_video = jnp.array(np.array(video), dtype=jnp.bfloat16)

    graphdef, state = nnx.split(wan_vae)
    params = state.to_pure_dict()
    # This replaces random params with the model.
    params = load_wan_vae(pretrained_model_name_or_path, params, "cpu")
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    wan_vae = nnx.merge(graphdef, params)

    p_vae_encode = jax.jit(functools.partial(vae_encode, wan_vae=wan_vae, vae_cache=vae_cache, key=key))
    original_video_shape = original_video.shape
    latent = p_vae_encode(original_video)

    jitted_decode = jax.jit(functools.partial(wan_vae.decode, feat_cache=vae_cache, return_dict=False))
    video = jitted_decode(latent)[0]
    video = jnp.transpose(video, (0, 4, 1, 2, 3))
    assert video.shape == original_video_shape

    original_video = torch.from_numpy(np.array(original_video.astype(jnp.float32))).to(dtype=torch.bfloat16)
    video = torch.from_numpy(np.array(video)).to(dtype=torch.bfloat16)
    video = video_processor.postprocess_video(video, output_type="np")
    original_video = video_processor.postprocess_video(original_video, output_type="np")
    ssim_compare = ssim(video[0], original_video[0], multichannel=True, channel_axis=-1, data_range=255)
    assert ssim_compare >= 0.9999


if __name__ == "__main__":
  absltest.main()
