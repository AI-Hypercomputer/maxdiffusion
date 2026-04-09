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

import os
import json
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

RATIONAL_RESAMPLER_SCALE_MAPPING = {
    0.75: (3, 4),
    1.5: (3, 2),
    2.0: (2, 1),
    4.0: (4, 1),
}


class ResBlock(nnx.Module):

  def __init__(self, in_channels: int, channels: int, mid_channels: Optional[int] = None, dims: int = 3, *, rngs: nnx.Rngs):
    self.channels = channels
    self.mid_channels = mid_channels if mid_channels is not None else channels
    self.dims = dims

    kernel_size = (3,) * self.dims
    padding = ((1, 1),) * self.dims

    self.conv1 = nnx.Conv(in_channels, self.mid_channels, kernel_size=kernel_size, padding=padding, rngs=rngs)
    self.norm1 = nnx.GroupNorm(num_groups=32, num_features=self.mid_channels, epsilon=1e-5, rngs=rngs)

    self.conv2 = nnx.Conv(self.mid_channels, self.channels, kernel_size=kernel_size, padding=padding, rngs=rngs)
    self.norm2 = nnx.GroupNorm(num_groups=32, num_features=self.channels, epsilon=1e-5, rngs=rngs)

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    residual = hidden_states

    hidden_states = self.conv1(hidden_states)
    hidden_states = self.norm1(hidden_states)
    hidden_states = jax.nn.silu(hidden_states)

    hidden_states = self.conv2(hidden_states)
    hidden_states = self.norm2(hidden_states)

    hidden_states = jax.nn.silu(hidden_states + residual)

    return hidden_states


class PixelShuffleND(nnx.Module):

  def __init__(self, dims: int, upscale_factors: Tuple[int, ...] = (2, 2, 2)):
    self.dims = dims
    self.upscale_factors = upscale_factors

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.dims == 3:
      p1, p2, p3 = self.upscale_factors[:3]
      b, d, h, w, c_p = x.shape
      c = c_p // (p1 * p2 * p3)
      x = jnp.reshape(x, (b, d, h, w, c, p1, p2, p3))
      x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
      x = jnp.reshape(x, (b, d * p1, h * p2, w * p3, c))
      return x
    elif self.dims == 2:
      p1, p2 = self.upscale_factors[:2]
      b, h, w, c_p = x.shape
      c = c_p // (p1 * p2)
      x = jnp.reshape(x, (b, h, w, c, p1, p2))
      x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
      x = jnp.reshape(x, (b, h * p1, w * p2, c))
      return x
    elif self.dims == 1:
      p1 = self.upscale_factors[0]
      b, f, h, w, c_p = x.shape
      c = c_p // p1
      x = jnp.reshape(x, (b, f, h, w, c, p1))
      x = jnp.transpose(x, (0, 1, 5, 2, 3, 4))
      x = jnp.reshape(x, (b, f * p1, h, w, c))
      return x


class BlurDownsample(nnx.Module):

  def __init__(self, dims: int, stride: int, kernel_size: int = 5):
    self.dims = dims
    self.stride = stride
    self.kernel_size = kernel_size

    if self.dims not in (2, 3):
      raise ValueError(f"`dims` must be either 2 or 3 but is {self.dims}")
    if self.kernel_size < 3 or self.kernel_size % 2 != 1:
      raise ValueError(f"`kernel_size` must be an odd number >= 3 but is {self.kernel_size}")

    k = jnp.array([math.comb(self.kernel_size - 1, i) for i in range(self.kernel_size)], dtype=jnp.float32)
    k2d = jnp.outer(k, k)
    k2d = k2d / jnp.sum(k2d)
    self.kernel = jnp.reshape(k2d, (self.kernel_size, self.kernel_size, 1, 1))

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.stride == 1:
      return x

    pad = self.kernel_size // 2
    c = x.shape[-1]
    kernel_broadcast = jnp.tile(self.kernel, (1, 1, 1, c))

    if self.dims == 2:
      x = jax.lax.conv_general_dilated(
          lhs=x,
          rhs=kernel_broadcast,
          window_strides=(self.stride, self.stride),
          padding=((pad, pad), (pad, pad)),
          feature_group_count=c,
          dimension_numbers=("NHWC", "HWIO", "NHWC"),
      )
    else:
      b, f, h, w, _ = x.shape
      x = jnp.reshape(x, (b * f, h, w, c))

      x = jax.lax.conv_general_dilated(
          lhs=x,
          rhs=kernel_broadcast,
          window_strides=(self.stride, self.stride),
          padding=((pad, pad), (pad, pad)),
          feature_group_count=c,
          dimension_numbers=("NHWC", "HWIO", "NHWC"),
      )

      h2, w2 = x.shape[1], x.shape[2]
      x = jnp.reshape(x, (b, f, h2, w2, c))

    return x


class SpatialRationalResampler(nnx.Module):

  def __init__(self, in_channels: int, mid_channels: int = 1024, scale: float = 2.0, *, rngs: nnx.Rngs):
    self.mid_channels = mid_channels
    self.scale = scale

    if self.scale not in RATIONAL_RESAMPLER_SCALE_MAPPING:
      raise ValueError(f"scale {self.scale} not supported.")
    num, den = RATIONAL_RESAMPLER_SCALE_MAPPING[self.scale]

    self.conv = nnx.Conv(
        in_channels, (num**2) * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs
    )
    self.pixel_shuffle = PixelShuffleND(dims=2, upscale_factors=(num, num))
    self.blur = BlurDownsample(dims=2, stride=den)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.conv(x)
    x = self.pixel_shuffle(x)
    x = self.blur(x)
    return x


class LTX2LatentUpsamplerModel(nnx.Module):

  def __init__(
      self,
      in_channels: int = 128,
      mid_channels: int = 1024,
      num_blocks_per_stage: int = 4,
      dims: int = 3,
      spatial_upsample: bool = True,
      temporal_upsample: bool = False,
      rational_spatial_scale: Optional[float] = 2.0,
      *,
      rngs: nnx.Rngs,
  ):
    self.in_channels = in_channels
    self.mid_channels = mid_channels
    self.num_blocks_per_stage = num_blocks_per_stage
    self.dims = dims
    self.spatial_upsample = spatial_upsample
    self.temporal_upsample = temporal_upsample
    self.rational_spatial_scale = rational_spatial_scale

    if self.dims == 2:
      self.initial_conv = nnx.Conv(
          self.in_channels, self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs
      )
      self.initial_norm = nnx.GroupNorm(epsilon=1e-5, num_groups=32, num_features=self.mid_channels, rngs=rngs)

      for i in range(self.num_blocks_per_stage):
        setattr(self, f"res_blocks_{i}", ResBlock(self.mid_channels, self.mid_channels, dims=2, rngs=rngs))

      if self.spatial_upsample:
        if self.rational_spatial_scale is not None:
          self.upsampler = SpatialRationalResampler(
              self.mid_channels, self.mid_channels, self.rational_spatial_scale, rngs=rngs
          )
        else:
          self.upsampler_0 = nnx.Conv(
              self.mid_channels, 4 * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs
          )
          self.pixel_shuffle = PixelShuffleND(dims=2)

      for i in range(self.num_blocks_per_stage):
        setattr(self, f"post_upsample_res_blocks_{i}", ResBlock(self.mid_channels, self.mid_channels, dims=2, rngs=rngs))

      self.final_conv = nnx.Conv(
          self.mid_channels, self.in_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs
      )

    else:
      self.initial_conv = nnx.Conv(
          self.in_channels, self.mid_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), rngs=rngs
      )
      self.initial_norm = nnx.GroupNorm(epsilon=1e-5, num_groups=32, num_features=self.mid_channels, rngs=rngs)

      for i in range(self.num_blocks_per_stage):
        setattr(self, f"res_blocks_{i}", ResBlock(self.mid_channels, self.mid_channels, dims=3, rngs=rngs))

      if self.spatial_upsample and self.temporal_upsample:
        self.upsampler_0 = nnx.Conv(
            self.mid_channels, 8 * self.mid_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), rngs=rngs
        )
        self.pixel_shuffle = PixelShuffleND(dims=3)
      elif self.temporal_upsample:
        self.upsampler_0 = nnx.Conv(
            self.mid_channels, 2 * self.mid_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), rngs=rngs
        )
        self.pixel_shuffle = PixelShuffleND(dims=1)
      elif self.spatial_upsample:
        if self.rational_spatial_scale is not None:
          self.upsampler = SpatialRationalResampler(
              self.mid_channels, self.mid_channels, self.rational_spatial_scale, rngs=rngs
          )
        else:
          self.upsampler_0 = nnx.Conv(
              self.mid_channels, 4 * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs
          )
          self.pixel_shuffle = PixelShuffleND(dims=2)

      for i in range(self.num_blocks_per_stage):
        setattr(self, f"post_upsample_res_blocks_{i}", ResBlock(self.mid_channels, self.mid_channels, dims=3, rngs=rngs))

      self.final_conv = nnx.Conv(
          self.mid_channels, self.in_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), rngs=rngs
      )

  @classmethod
  def load_config(cls, pretrained_model_name_or_path: str, subfolder: str = "", **kwargs):
    try:
      if os.path.isdir(pretrained_model_name_or_path):
        config_file = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
      else:
        config_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json", subfolder=subfolder)

      with open(config_file, "r") as f:
        config_dict = json.load(f)

      config_dict.update(kwargs)
      return config_dict

    except (OSError, json.JSONDecodeError, EntryNotFoundError, HfHubHTTPError) as e:
      print(f"Warning: Could not load upsampler config.json (using defaults). Reason: {e}")
      return kwargs

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    b, f, h, w, c = hidden_states.shape

    if self.dims == 2:
      hidden_states = jnp.reshape(hidden_states, (b * f, h, w, c))

      hidden_states = self.initial_conv(hidden_states)
      hidden_states = self.initial_norm(hidden_states)
      hidden_states = jax.nn.silu(hidden_states)

      for i in range(self.num_blocks_per_stage):
        block = getattr(self, f"res_blocks_{i}")
        hidden_states = block(hidden_states)

      if self.spatial_upsample:
        if self.rational_spatial_scale is not None:
          hidden_states = self.upsampler(hidden_states)
        else:
          hidden_states = self.upsampler_0(hidden_states)
          hidden_states = self.pixel_shuffle(hidden_states)

      for i in range(self.num_blocks_per_stage):
        block = getattr(self, f"post_upsample_res_blocks_{i}")
        hidden_states = block(hidden_states)

      hidden_states = self.final_conv(hidden_states)

      h2, w2 = hidden_states.shape[1], hidden_states.shape[2]
      hidden_states = jnp.reshape(hidden_states, (b, f, h2, w2, self.in_channels))

    else:
      hidden_states = self.initial_conv(hidden_states)
      hidden_states = self.initial_norm(hidden_states)
      hidden_states = jax.nn.silu(hidden_states)

      for i in range(self.num_blocks_per_stage):
        block = getattr(self, f"res_blocks_{i}")
        hidden_states = block(hidden_states)

      if self.spatial_upsample and self.temporal_upsample:
        hidden_states = self.upsampler_0(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = hidden_states[:, 1:, :, :, :]
      elif self.temporal_upsample:
        hidden_states = self.upsampler_0(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = hidden_states[:, 1:, :, :, :]
      elif self.spatial_upsample:
        hidden_states = jnp.reshape(hidden_states, (b * f, h, w, self.mid_channels))
        if self.rational_spatial_scale is not None:
          hidden_states = self.upsampler(hidden_states)
        else:
          hidden_states = self.upsampler_0(hidden_states)
          hidden_states = self.pixel_shuffle(hidden_states)
        h2, w2 = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = jnp.reshape(hidden_states, (b, f, h2, w2, self.mid_channels))

      for i in range(self.num_blocks_per_stage):
        block = getattr(self, f"post_upsample_res_blocks_{i}")
        hidden_states = block(hidden_states)

      hidden_states = self.final_conv(hidden_states)

    return hidden_states
