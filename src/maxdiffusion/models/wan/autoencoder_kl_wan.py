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

from typing import Tuple, List, Sequence, Union, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from ...configuration_utils import ConfigMixin
from ..modeling_flax_utils import FlaxModelMixin
from ... import common_types
from ..vae_flax import (FlaxAutoencoderKLOutput, FlaxDiagonalGaussianDistribution, FlaxDecoderOutput)

BlockSizes = common_types.BlockSizes

CACHE_T = 2

_ACTIVATIONS = {"swish": jax.nn.silu, "silu": jax.nn.silu, "relu": jax.nn.relu, "gelu": jax.nn.gelu, "mish": jax.nn.mish}


def get_activation(name: str):
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


# Helper to ensure kernel_size, stride, padding are tuples of 3 integers
def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  """Canonicalizes a value to a tuple of integers."""
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank} integers. Got {x}")


class WanCausalConv3d(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,  # rngs are required for initializing parameters,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      padding: Union[int, Tuple[int, int, int]] = 0,
      use_bias: bool = True,
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    padding_tuple = _canonicalize_tuple(padding, 3, "padding")  # (D, H, W) padding amounts

    self._causal_padding = (
        (0, 0),  # Batch dimension - no padding
        (2 * padding_tuple[0], 0),  # Depth dimension - causal padding (pad only before)
        (padding_tuple[1], padding_tuple[1]),  # Height dimension - symmetric padding
        (padding_tuple[2], padding_tuple[2]),  # Width dimension - symmetric padding
        (0, 0),  # Channel dimension - no padding
    )

    # Store the amount of padding needed *before* the depth dimension for caching logic
    self._depth_padding_before = self._causal_padding[1][0]  # 2 * padding_tuple[0]

    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=self.kernel_size,
        strides=self.stride,
        use_bias=use_bias,
        padding="VALID",  # Handle padding manually
        rngs=rngs,
    )

  def __call__(self, x: jax.Array, cache_x: Optional[jax.Array] = None, idx=-1) -> jax.Array:
    current_padding = list(self._causal_padding)  # Mutable copy
    padding_needed = self._depth_padding_before

    if cache_x is not None and padding_needed > 0:
      # Ensure cache has same spatial/channel dims, potentially different depth
      assert cache_x.shape[0] == x.shape[0] and cache_x.shape[2:] == x.shape[2:], "Cache spatial/channel dims mismatch"
      cache_len = cache_x.shape[1]
      x = jnp.concatenate([cache_x, x], axis=1)  # Concat along depth (D)

      padding_needed -= cache_len
      if padding_needed < 0:
        # Cache longer than needed padding, trim from start
        x = x[:, -padding_needed:, ...]
        current_padding[1] = (0, 0)  # No explicit padding needed now
      else:
        # Update depth padding needed
        current_padding[1] = (padding_needed, 0)

    # Apply padding if any dimension requires it
    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(x, padding_to_apply, mode="constant", constant_values=0.0)
    else:
      x_padded = x
    out = self.conv(x_padded)
    return out


class WanRMS_norm(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      channel_first: bool = True,
      images: bool = True,
      eps: float = 1e-6,
      use_bias: bool = False,
  ):
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)
    self.eps = eps
    self.channel_first = channel_first
    self.scale = dim**0.5
    # Initialize gamma as parameter
    self.gamma = nnx.Param(jnp.ones(shape))
    if use_bias:
      self.bias = nnx.Param(jnp.zeros(shape))
    else:
      self.bias = 0

  def __call__(self, x: jax.Array) -> jax.Array:
    normalized = jnp.linalg.norm(x, ord=2, axis=(1 if self.channel_first else -1), keepdims=True)
    normalized = x / jnp.maximum(normalized, self.eps)
    normalized = normalized * self.scale * self.gamma
    if self.bias:
      return normalized + self.bias.value
    return normalized


class WanUpsample(nnx.Module):

  def __init__(self, scale_factor: Tuple[float, float], method: str = "nearest"):
    # scale_factor for (H, W)
    # JAX resize works on spatial dims, H, W assumming (N, D, H, W, C) or (N, H, W, C)
    self.scale_factor = scale_factor
    self.method = method

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    in_shape = x.shape
    assert len(in_shape) == 4, "This module only takes tensors with shape of 4."
    n, h, w, c = in_shape
    target_h = int(h * self.scale_factor[0])
    target_w = int(w * self.scale_factor[1])
    out = jax.image.resize(x.astype(jnp.float32), (n, target_h, target_w, c), method=self.method)
    return out.astype(input_dtype)


class Identity(nnx.Module):

  def __call__(self, x):
    return x


class ZeroPaddedConv2D(nnx.Module):
  """
  Module for adding padding before conv.
  Currently it does not add any padding.
  """

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
  ):
    self.conv = nnx.Conv(dim, dim, kernel_size=kernel_size, strides=stride, use_bias=True, rngs=rngs)

  def __call__(self, x):
    return self.conv(x)


class WanResample(nnx.Module):

  def __init__(
      self,
      dim: int,
      mode: str,
      rngs: nnx.Rngs,
  ):
    self.dim = dim
    self.mode = mode
    self.time_conv = None

    if mode == "upsample2d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
          ),
      )
    elif mode == "upsample3d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
          ),
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim * 2,
          kernel_size=(3, 1, 1),
          padding=(1, 0, 0),
      )
    elif mode == "downsample2d":
      self.resample = ZeroPaddedConv2D(dim=dim, rngs=rngs, kernel_size=(3, 3), stride=(2, 2))
    elif mode == "downsample3d":
      self.resample = ZeroPaddedConv2D(dim=dim, rngs=rngs, kernel_size=(3, 3), stride=(2, 2))
      self.time_conv = WanCausalConv3d(
          rngs=rngs, in_channels=dim, out_channels=dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
      )
    else:
      self.resample = Identity()

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]) -> jax.Array:
    # Input x: (N, D, H, W, C), assume C = self.dim
    b, t, h, w, c = x.shape
    assert c == self.dim

    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = "Rep"
          feat_idx[0] += 1
        else:
          cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
            # cache last frame of last two chunk
            cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
            cache_x = jnp.concatenate([jnp.zeros(cache_x.shape), cache_x], axis=1)
          if feat_cache[idx] == "Rep":
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache[idx] = cache_x
          feat_idx[0] += 1
          x = x.reshape(b, t, h, w, 2, c)
          x = jnp.stack([x[:, :, :, :, 0, :], x[:, :, :, :, 1, :]], axis=1)
          x = x.reshape(b, t * 2, h, w, c)
    t = x.shape[1]
    x = x.reshape(b * t, h, w, c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(b, t, h_new, w_new, c_new)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = jnp.copy(x)
          feat_idx[0] += 1
        else:
          cache_x = jnp.copy(x[:, -1:, :, :, :])
          x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
          feat_cache[idx] = cache_x
          feat_idx[0] += 1

    return x


class WanResidualBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
  ):
    self.nonlinearity = get_activation(non_linearity)

    # layers
    self.norm1 = WanRMS_norm(dim=in_dim, rngs=rngs, images=False, channel_first=False)
    self.conv1 = WanCausalConv3d(rngs=rngs, in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1)
    self.norm2 = WanRMS_norm(dim=out_dim, rngs=rngs, images=False, channel_first=False)
    self.conv2 = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1)
    self.conv_shortcut = (
        WanCausalConv3d(rngs=rngs, in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        if in_dim != out_dim
        else Identity()
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    # Apply shortcut connection
    h = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv1(x, feat_cache[idx], idx)
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)
    idx = feat_idx[0]

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv2(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv2(x)
    x = x + h
    return x


class WanAttentionBlock(nnx.Module):

  def __init__(self, dim: int, rngs: nnx.Rngs):
    self.dim = dim
    self.norm = WanRMS_norm(rngs=rngs, dim=dim, channel_first=False)
    self.to_qkv = nnx.Conv(in_features=dim, out_features=dim * 3, kernel_size=(1, 1), rngs=rngs)
    self.proj = nnx.Conv(in_features=dim, out_features=dim, kernel_size=(1, 1), rngs=rngs)

  def __call__(self, x: jax.Array):

    identity = x
    batch_size, time, height, width, channels = x.shape

    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)

    qkv = self.to_qkv(x)  # Output: (N*D, H, W, C * 3)
    # qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
    qkv = qkv.reshape(batch_size * time, 1, -1, channels * 3)
    qkv = jnp.transpose(qkv, (0, 1, 3, 2))
    # q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = jnp.split(qkv, 3, axis=-2)
    q = jnp.transpose(q, (0, 1, 3, 2))
    k = jnp.transpose(k, (0, 1, 3, 2))
    v = jnp.transpose(v, (0, 1, 3, 2))
    x = jax.nn.dot_product_attention(q, k, v)
    x = jnp.squeeze(x, 1).reshape(batch_size * time, height, width, channels)

    # output projection
    x = self.proj(x)
    # Reshape back
    x = x.reshape(batch_size, time, height, width, channels)

    return x + identity


class WanMidBlock(nnx.Module):

  def __init__(self, dim: int, rngs: nnx.Rngs, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
    self.dim = dim
    resnets = [WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs, dropout=dropout, non_linearity=non_linearity)]
    attentions = []
    for _ in range(num_layers):
      attentions.append(WanAttentionBlock(dim=dim, rngs=rngs))
      resnets.append(WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs, dropout=dropout, non_linearity=non_linearity))
    self.attentions = attentions
    self.resnets = resnets

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    x = self.resnets[0](x, feat_cache, feat_idx)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x = attn(x)
      x = resnet(x, feat_cache, feat_idx)
    return x


class WanUpBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      num_res_blocks: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      upsample_mode: Optional[str] = None,
      non_linearity: str = "silu",
  ):
    # Create layers list
    resnets = []
    # Add residual blocks and attention if needed
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
      resnets.append(
          WanResidualBlock(in_dim=current_dim, out_dim=out_dim, dropout=dropout, non_linearity=non_linearity, rngs=rngs)
      )
      current_dim = out_dim
    self.resnets = resnets

    # Add upsampling layer if needed.
    self.upsamplers = None
    if upsample_mode is not None:
      self.upsamplers = [WanResample(dim=out_dim, mode=upsample_mode, rngs=rngs)]

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    for resnet in self.resnets:
      if feat_cache is not None:
        x = resnet(x, feat_cache, feat_idx)
      else:
        x = resnet(x)

    if self.upsamplers is not None:
      if feat_cache is not None:
        x = self.upsamplers[0](x, feat_cache, feat_idx)
      else:
        x = self.upsamplers[0](x)
    return x


class WanEncoder3d(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      attn_scales=[],
      temperal_downsample=[True, True, False],
      dropout=0.0,
      non_linearity: str = "silu",
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_downsample = temperal_downsample
    self.nonlinearity = get_activation(non_linearity)

    # dimensions
    dims = [dim * u for u in [1] + dim_mult]
    scale = 1.0

    # init block
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=3,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
    )

    # downsample blocks
    self.down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      for _ in range(num_res_blocks):
        self.down_blocks.append(WanResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout, rngs=rngs))
        if scale in attn_scales:
          self.down_blocks.append(WanAttentionBlock(dim=out_dim, rngs=rngs))
        in_dim = out_dim

      # downsample block
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.down_blocks.append(WanResample(out_dim, mode=mode, rngs=rngs))
        scale /= 2.0

    # middle_blocks
    self.mid_block = WanMidBlock(
        dim=out_dim,
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
    )

    # output blocks
    self.norm_out = WanRMS_norm(out_dim, channel_first=False, images=False, rngs=rngs)
    self.conv_out = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=z_dim, kernel_size=3, padding=1)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_in(x)
    for layer in self.down_blocks:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)

    x = self.mid_block(x, feat_cache, feat_idx)

    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_out(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_out(x)
    return x


class WanDecoder3d(nnx.Module):
  r"""
  A 3D decoder module.
  Args:
    dim (int): The base number of channels in the first layer.
    z_dim (int): The dimensionality of the latent space.
    dim_mult (list of int): Multipliers for the number of channels in each block.
    num_res_blocks (int): Number of residual blocks in each block.
    attn_scales (list of float): Scales at which to apply attention mechanisms.
    temperal_upsample (list of bool): Whether to upsample temporally in each block.
    dropout (float): Dropout rate for the dropout layers.
    non_linearity (str): Type of non-linearity to use.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult: List[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales=List[float],
      temperal_upsample=[False, True, True],
      dropout=0.0,
      non_linearity: str = "silu",
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_upsample = temperal_upsample

    self.nonlinearity = get_activation(non_linearity)

    # dimensions
    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)

    # init block
    self.conv_in = WanCausalConv3d(rngs=rngs, in_channels=z_dim, out_channels=dims[0], kernel_size=3, padding=1)

    # middle_blocks
    self.mid_block = WanMidBlock(dim=dims[0], rngs=rngs, dropout=dropout, non_linearity=non_linearity, num_layers=1)

    # upsample blocks
    self.up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      if i > 0:
        in_dim = in_dim // 2

      # Determine if we need upsampling
      upsample_mode = None
      if i != len(dim_mult) - 1:
        upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
      # Create and add the upsampling block
      up_block = WanUpBlock(
          in_dim=in_dim,
          out_dim=out_dim,
          num_res_blocks=num_res_blocks,
          dropout=dropout,
          upsample_mode=upsample_mode,
          non_linearity=non_linearity,
          rngs=rngs,
      )
      self.up_blocks.append(up_block)

      # Update scale for next iteration
      if upsample_mode is not None:
        scale *= 2.0

    # output blocks
    self.norm_out = WanRMS_norm(dim=out_dim, images=False, rngs=rngs, channel_first=False)
    self.conv_out = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=3, kernel_size=3, padding=1)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_in(x)

    ## middle
    x = self.mid_block(x, feat_cache, feat_idx)
    ## upsamples
    for up_block in self.up_blocks:
      x = up_block(x, feat_cache, feat_idx)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_out(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_out(x)
    return x


class AutoencoderKLWanCache:

  def __init__(self, module):
    self.module = module
    self.clear_cache()

  def clear_cache(self):
    """Resets cache dictionaries and indices"""

    def _count_conv3d(module):
      count = 0
      node_types = nnx.graph.iter_graph([module])
      for _, value in node_types:
        if isinstance(value, WanCausalConv3d):
          count += 1
      return count

    self._conv_num = _count_conv3d(self.module.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num
    # cache encode
    self._enc_conv_num = _count_conv3d(self.module.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num


class AutoencoderKLWan(nnx.Module, FlaxModelMixin, ConfigMixin):

  def __init__(
      self,
      rngs: nnx.Rngs,
      base_dim: int = 96,
      z_dim: int = 16,
      dim_mult: Tuple[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_downsample: List[bool] = [False, True, True],
      dropout: float = 0.0,
      latents_mean: List[float] = [
          -0.7571,
          -0.7089,
          -0.9113,
          0.1075,
          -0.1745,
          0.9653,
          -0.1517,
          1.5508,
          0.4134,
          -0.0715,
          0.5517,
          -0.3632,
          -0.1922,
          -0.9497,
          0.2503,
          -0.2921,
      ],
      latents_std: List[float] = [
          2.8184,
          1.4541,
          2.3275,
          2.6558,
          1.2196,
          1.7708,
          2.6052,
          2.0743,
          3.2687,
          2.1526,
          2.8652,
          1.5579,
          1.6382,
          1.1253,
          2.8251,
          1.9160,
      ],
  ):
    self.z_dim = z_dim
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]
    self.latents_mean = latents_mean
    self.latents_std = latents_std

    self.encoder = WanEncoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim * 2,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        dropout=dropout,
    )
    self.quant_conv = WanCausalConv3d(rngs=rngs, in_channels=z_dim * 2, out_channels=z_dim * 2, kernel_size=1)
    self.post_quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=z_dim,
        kernel_size=1,
    )

    self.decoder = WanDecoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=self.temporal_upsample,
        dropout=dropout,
    )

  def _encode(self, x: jax.Array, feat_cache: AutoencoderKLWanCache):
    feat_cache.clear_cache()
    if x.shape[-1] != 3:
      # reshape channel last for JAX
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert x.shape[-1] == 3, f"Expected input shape (N, D, H, W, 3), got {x.shape}"

    t = x.shape[1]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
      feat_cache._enc_conv_idx = [0]
      if i == 0:
        out = self.encoder(x[:, :1, :, :, :], feat_cache=feat_cache._enc_feat_map, feat_idx=feat_cache._enc_conv_idx)
      else:
        out_ = self.encoder(
            x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :],
            feat_cache=feat_cache._enc_feat_map,
            feat_idx=feat_cache._enc_conv_idx,
        )
        out = jnp.concatenate([out, out_], axis=1)
    enc = self.quant_conv(out)
    mu, logvar = enc[:, :, :, :, : self.z_dim], enc[:, :, :, :, self.z_dim :]
    enc = jnp.concatenate([mu, logvar], axis=-1)
    feat_cache.clear_cache()
    return enc

  def encode(
      self, x: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxAutoencoderKLOutput, Tuple[FlaxDiagonalGaussianDistribution]]:
    """Encode video into latent distribution."""
    h = self._encode(x, feat_cache)
    posterior = FlaxDiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior,)
    return FlaxAutoencoderKLOutput(latent_dist=posterior)

  def _decode(
      self, z: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    feat_cache.clear_cache()
    iter_ = z.shape[1]
    x = self.post_quant_conv(z)
    for i in range(iter_):
      feat_cache._conv_idx = [0]
      if i == 0:
        out = self.decoder(x[:, i : i + 1, :, :, :], feat_cache=feat_cache._feat_map, feat_idx=feat_cache._conv_idx)
      else:
        out_ = self.decoder(x[:, i : i + 1, :, :, :], feat_cache=feat_cache._feat_map, feat_idx=feat_cache._conv_idx)

        # This is to bypass an issue where frame[1] should be frame[2] and vise versa.
        # Ideally shouldn't need to do this however, can't find where the frame is going out of sync.
        # Most likely due to an incorrect reshaping in the decoder.
        fm1, fm2, fm3, fm4 = out_[:, 0, :, :, :], out_[:, 1, :, :, :], out_[:, 2, :, :, :], out_[:, 3, :, :, :]
        if len(fm1.shape) == 4:
          fm1 = jnp.expand_dims(fm1, axis=0)
          fm2 = jnp.expand_dims(fm2, axis=0)
          fm3 = jnp.expand_dims(fm3, axis=0)
          fm4 = jnp.expand_dims(fm4, axis=0)

        out = jnp.concatenate([out, fm1, fm3, fm2, fm4], axis=1)
    out = jnp.clip(out, min=-1.0, max=1.0)
    feat_cache.clear_cache()
    if not return_dict:
      return (out,)

    return FlaxDecoderOutput(sample=out)

  def decode(
      self, z: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    if z.shape[-1] != self.z_dim:
      # reshape channel last for JAX
      z = jnp.transpose(z, (0, 2, 3, 4, 1))
      assert z.shape[-1] == self.z_dim, f"Expected input shape (N, D, H, W, {self.z_dim}, got {z.shape}"
    decoded = self._decode(z, feat_cache).sample
    if not return_dict:
      return (decoded,)
    return FlaxDecoderOutput(sample=decoded)
