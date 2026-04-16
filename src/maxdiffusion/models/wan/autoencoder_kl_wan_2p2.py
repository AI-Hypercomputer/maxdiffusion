"""Copyright 2025 Google LLC

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

from typing import Any, List, Optional, Sequence, Tuple, Union

import flax
from flax import nnx
import jax
from jax import tree_util
import jax.numpy as jnp
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache, WanCausalConv3d  # pylint: disable=g-importing-member

from ... import common_types
from ...configuration_utils import ConfigMixin
from ..modeling_flax_utils import FlaxModelMixin, get_activation
from ..vae_flax import (
    FlaxAutoencoderKLOutput,
    FlaxDecoderOutput,
    FlaxDiagonalGaussianDistribution,
    WanDiagonalGaussianDistribution,
)

BlockSizes = common_types.BlockSizes

CACHE_T = 2
try:
  flax.config.update("flax_always_shard_variable", False)
except LookupError:
  pass


def _update_cache(cache, idx, value):
  if cache is None:
    return None
  return cache[:idx] + (value,) + cache[idx + 1 :]


# Helper to ensure kernel_size, stride, padding are tuples of 3 integers
def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  """Canonicalizes a value to a tuple of integers."""
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank}" f" integers. Got {x}")


class RepSentinel:

  def __eq__(self, other):
    return isinstance(other, RepSentinel)


tree_util.register_pytree_node(RepSentinel, lambda x: ((), None), lambda _, __: RepSentinel())


class WanPatchify(nnx.Module):

  def __init__(self, patch_size: int = 1):
    self.patch_size = patch_size

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.patch_size == 1:
      return x
    if x.ndim == 5:
      # [N, D, H, W, C] -> [N, D, H/q, W/r, C*q*r]
      b, t, h, w, c = x.shape
      q = r = self.patch_size
      x = x.reshape(b, t, h // q, q, w // r, r, c)
      x = x.transpose(0, 1, 2, 4, 6, 5, 3)
      x = x.reshape(b, t, h // q, w // r, c * q * r)
    return x


class WanUnpatchify(nnx.Module):

  def __init__(self, patch_size: int = 1):
    self.patch_size = patch_size

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.patch_size == 1:
      return x
    if x.ndim == 5:
      # [N, D, H/q, W/r, C*q*r] -> [N, D, H, W, C]
      b, t, h, w, c_total = x.shape
      q = r = self.patch_size
      c = c_total // (q * r)
      x = x.reshape(b, t, h, w, c, r, q)
      x = x.transpose(0, 1, 2, 6, 3, 5, 4)
      x = x.reshape(b, t, h * q, w * r, c)
    return x


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
    # JAX resize works on spatial dims, H, W assuming (N, D, H, W, C) or (N, H, W, C)
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
  """Module for adding padding before conv.

  Currently it does not add any padding.
  """

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    # Padding to match PyTorch's nn.ZeroPad2d((0, 1, 0, 1))
    self.conv = nnx.Conv(
        dim,
        dim,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=True,
        padding=[(0, 1), (0, 1)],
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, None)),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x):
    return self.conv(x)


class WanResample(nnx.Module):

  def __init__(
      self,
      dim: int,
      mode: str,
      rngs: nnx.Rngs,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    self.mode = mode
    self.time_conv = nnx.data(None)

    if mode == "upsample2d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(
                  nnx.initializers.xavier_uniform(),
                  (None, None, None, "conv_out"),
              ),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
    elif mode == "upsample3d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(
                  nnx.initializers.xavier_uniform(),
                  (None, None, None, "conv_out"),
              ),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim * 2,
          kernel_size=(3, 1, 1),
          padding=(1, 0, 0),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample2d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample3d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim,
          kernel_size=(3, 1, 1),
          stride=(2, 1, 1),
          padding=(0, 0, 0),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      self.resample = Identity()

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    # Input x: (N, D, H, W, C), assume C = self.dim
    b, t, h, w, c = x.shape
    assert c == self.dim

    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, RepSentinel())
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and not isinstance(feat_cache[idx], RepSentinel):
            # cache last frame of last two chunk
            cache_x = jnp.concatenate(
                [
                    jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1),
                    cache_x,
                ],
                axis=1,
            )
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and isinstance(feat_cache[idx], RepSentinel):
            cache_x = jnp.concatenate([jnp.zeros(cache_x.shape), cache_x], axis=1)
          if isinstance(feat_cache[idx], RepSentinel):
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1
          x = x.reshape(b, t, h, w, 2, c)
          x = x.transpose(0, 1, 4, 2, 3, 5)
          x = x.reshape(b, t * 2, h, w, c)
    t = x.shape[1]
    x = x.reshape(b * t, h, w, c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(b, t, h_new, w_new, c_new)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, jnp.copy(x))
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -1:, :, :, :])
          x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1

    return x, feat_cache, feat_idx


class WanResidualBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.nonlinearity = get_activation(non_linearity)

    # layers
    self.norm1 = WanRMS_norm(dim=in_dim, rngs=rngs, images=False, channel_first=False)
    self.conv1 = WanCausalConv3d(
        rngs=rngs,
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm2 = WanRMS_norm(dim=out_dim, rngs=rngs, images=False, channel_first=False)
    self.conv2 = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.conv_shortcut = (
        WanCausalConv3d(
            rngs=rngs,
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        if in_dim != out_dim
        else Identity()
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    # Apply shortcut connection
    h = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv1(x, feat_cache[idx], idx)
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv2(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv2(x)
    x = x + h
    return x, feat_cache, feat_idx


class WanAttentionBlock(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    self.norm = WanRMS_norm(rngs=rngs, dim=dim, channel_first=False)
    self.to_qkv = nnx.Conv(
        in_features=dim,
        out_features=dim * 3,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )
    self.proj = nnx.Conv(
        in_features=dim,
        out_features=dim,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "conv_in", None)),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    identity = x
    batch_size, time, height, width, channels = x.shape

    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)

    qkv = self.to_qkv(x)  # Output: (N*D, H, W, C * 3)
    # qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
    qkv = qkv.reshape(batch_size * time, 1, -1, channels * 3)
    qkv = jnp.transpose(qkv, (0, 1, 3, 2))
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

    return x + identity, feat_cache, feat_idx


class WanMidBlock(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      num_layers: int = 1,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    resnets = [
        WanResidualBlock(
            in_dim=dim,
            out_dim=dim,
            rngs=rngs,
            dropout=dropout,
            non_linearity=non_linearity,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
    ]
    attentions = []
    for _ in range(num_layers):
      attentions.append(
          WanAttentionBlock(
              dim=dim,
              rngs=rngs,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      resnets.append(
          WanResidualBlock(
              in_dim=dim,
              out_dim=dim,
              rngs=rngs,
              dropout=dropout,
              non_linearity=non_linearity,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
    self.attentions = nnx.data(attentions)
    self.resnets = nnx.data(resnets)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    x, feat_cache, feat_idx = self.resnets[0](x, feat_cache, feat_idx)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x, feat_cache, feat_idx = attn(x, feat_cache, feat_idx)
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
    return x, feat_cache, feat_idx


class WanAvgDown3D(nnx.Module):
  """Average downsampling 3d."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      factor_t: int,
      factor_s: int,
  ):
    self.factor_t = factor_t
    self.factor_s = factor_s
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.factor = self.factor_t * self.factor_s * self.factor_s
    self.group_size = in_channels * self.factor // out_channels

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0) -> Tuple[jax.Array, Any, int]:
    if self.factor_t > 1 or self.factor_s > 1:
      n, d, h, w, c = x.shape
      pad_d = (self.factor_t - d % self.factor_t) % self.factor_t
      # pad_h = (self.factor_s - h % self.factor_s) % self.factor_s
      # pad_w = (self.factor_s - w % self.factor_s) % self.factor_s
      pad_h = 0
      pad_w = 0
      if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = jnp.pad(x, ((0, 0), (pad_d, 0), (pad_h, 0), (pad_w, 0), (0, 0)))
      n, d, h, w, c = x.shape
      x = x.reshape(
          n,
          d // self.factor_t,
          self.factor_t,
          h // self.factor_s,
          self.factor_s,
          w // self.factor_s,
          self.factor_s,
          c,
      )
      # Permute to (N, D, H, W, C, f_t, f_s, f_s)
      x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
      x = x.reshape(
          n,
          d // self.factor_t,
          h // self.factor_s,
          w // self.factor_s,
          c * self.factor_t * self.factor_s * self.factor_s,
      )
      x = x.reshape(
          n,
          d // self.factor_t,
          h // self.factor_s,
          w // self.factor_s,
          self.out_channels,
          self.group_size,
      )
      x = x.mean(axis=-1)
    return x, feat_cache, feat_idx


class WanDupUp3D(nnx.Module):
  """Duplicate upsampling 3d."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      factor_t: int,
      factor_s: int,
  ):
    self.factor_t = factor_t
    self.factor_s = factor_s
    self.factor = factor_t * factor_s * factor_s
    self.out_channels = out_channels
    self.repeats = out_channels * self.factor // in_channels

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0, first_chunk: bool = False) -> Tuple[jax.Array, Any, int]:
    # Duplicate channels to match the expected total channels for upsampling.
    # x: (N, D, H, W, in_channels) -> (N, D, H, W, in_channels * self.repeats)
    x = jnp.repeat(x, repeats=self.repeats, axis=4)
    # x: (N, D, H, W, C)
    n, d, h, w, c_total = x.shape
    c = c_total // self.factor
    jax.debug.print(
        "DEBUG DupUp: c_total={ct}, factor={f}, c={c}, d={d}",
        ct=c_total,
        f=self.factor,
        c=c,
        d=d,
    )
    # Interleave logic: (N, D, H, W, C_out, f_t, f_s, f_s)
    x = x.reshape(n, d, h, w, c, self.factor_t, self.factor_s, self.factor_s)
    # Permute to (N, D, f_t, H, f_s, W, f_s, C_out)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    # Reshape: (N, D*f_t, H*f_s, W*f_s, C_out)
    x = x.reshape(
        n,
        d * self.factor_t,
        h * self.factor_s,
        w * self.factor_s,
        c,
    )
    if first_chunk:
      x = x[:, self.factor_t - 1 :, :, :, :]
    return x, feat_cache, feat_idx


class WanDownBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      num_res_blocks: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      downsample_mode: Optional[str] = None,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      down_flag: bool = False,
  ):
    # Create layers list
    resnets = []
    # Add residual blocks and attention if needed
    current_dim = in_dim

    self.avg_shortcut = WanAvgDown3D(
        in_dim,
        out_dim,
        factor_t=2 if downsample_mode == "downsample3d" else 1,
        factor_s=2 if down_flag else 1,
    )

    for _ in range(num_res_blocks):
      resnets.append(
          WanResidualBlock(
              in_dim=current_dim,
              out_dim=out_dim,
              dropout=dropout,
              non_linearity=non_linearity,
              rngs=rngs,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      current_dim = out_dim

    if downsample_mode is not None:
      resnets.append(
          WanResample(
              dim=out_dim,
              mode=downsample_mode,
              rngs=rngs,
              mesh=mesh,
              weights_dtype=weights_dtype,
              dtype=dtype,
              precision=precision,
          )
      )
    self.resnets = nnx.data(resnets)

  def __call__(
      self,
      x: jax.Array,
      feat_cache=None,
      feat_idx=0,
      return_shortcut: bool = False,
  ):
    x_main = x
    for resnet in self.resnets:
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)

    x_shortcut = None
    if self.avg_shortcut is not None:
      x_shortcut, feat_cache, feat_idx = self.avg_shortcut(x_main, feat_cache, feat_idx)
      x = x + x_shortcut

    if return_shortcut:
      return x, feat_cache, feat_idx, x_shortcut
    return x, feat_cache, feat_idx


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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      up_flag: bool = False,
  ):
    # Create layers list
    resnets = []
    # Add residual blocks and attention if needed
    current_dim = in_dim

    if up_flag:
      self.avg_shortcut = WanDupUp3D(
          in_dim,
          out_dim,
          factor_t=2 if upsample_mode == "upsample3d" else 1,
          factor_s=2 if up_flag else 1,
      )
    else:
      self.avg_shortcut = None

    for _ in range(num_res_blocks + 1):
      resnets.append(
          WanResidualBlock(
              in_dim=current_dim,
              out_dim=out_dim,
              dropout=dropout,
              non_linearity=non_linearity,
              rngs=rngs,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      current_dim = out_dim

    if upsample_mode is not None:
      resnets.append(
          WanResample(
              dim=out_dim,
              mode=upsample_mode,
              rngs=rngs,
              mesh=mesh,
              weights_dtype=weights_dtype,
              dtype=dtype,
              precision=precision,
          )
      )
    self.resnets = nnx.data(resnets)

  def __call__(
      self,
      x: jax.Array,
      feat_cache=None,
      feat_idx=0,
      first_chunk: bool = False,
      return_shortcut: bool = False,
  ):
    x_main = x
    for resnet in self.resnets:
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)

    x_shortcut = None
    if self.avg_shortcut is not None:
      x_shortcut, feat_cache, feat_idx = self.avg_shortcut(x_main, feat_cache, feat_idx, first_chunk)
      x = x + x_shortcut

    if return_shortcut:
      return x, feat_cache, feat_idx, x_shortcut
    return x, feat_cache, feat_idx


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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      in_channels: int = 3,
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
        in_channels=in_channels,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # downsample blocks
    self.down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      if i != len(dim_mult) - 1:
        downsample_mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
      else:
        downsample_mode = None
      self.down_blocks.append(
          WanDownBlock(
              in_dim=in_dim,
              out_dim=out_dim,
              num_res_blocks=num_res_blocks,
              dropout=dropout,
              downsample_mode=downsample_mode,
              non_linearity=non_linearity,
              rngs=rngs,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
              down_flag=i != len(dim_mult) - 1,
          )
      )

      # downsample block
      if i != len(dim_mult) - 1:
        scale /= 2.0
    self.down_blocks = nnx.data(self.down_blocks)

    # middle_blocks
    self.mid_block = WanMidBlock(
        dim=out_dim,
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # output blocks
    self.norm_out = WanRMS_norm(out_dim, channel_first=False, images=False, rngs=rngs)
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=z_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  @nnx.jit(static_argnames="feat_idx")
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)
    for layer in self.down_blocks:
      x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)

    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)

    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of last two chunk
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_out(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_out(x)
    return x, feat_cache, jnp.array(feat_idx, dtype=jnp.int32)


class WanDecoder3d(nnx.Module):
  r"""A 3D decoder module.

  Args:
    dim (int): The base number of channels in the first layer.
    z_dim (int): The dimensionality of the latent space.
    dim_mult (list of int): Multipliers for the number of channels in each
      block.
    num_res_blocks (int): Number of residual blocks in each block.
    attn_scales (list of float): Scales at which to apply attention mechanisms.
    temperal_upsample (list of bool): Whether to upsample temporally in each
      block.
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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      out_channels: int = 3,
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
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # middle_blocks
    self.mid_block = WanMidBlock(
        dim=dims[0],
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # upsample blocks
    self.up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
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
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          up_flag=i != len(dim_mult) - 1,
      )
      self.up_blocks.append(up_block)

      # Update scale for next iteration
      if upsample_mode is not None:
        scale *= 2.0
    self.up_blocks = nnx.data(self.up_blocks)

    # output blocks
    self.norm_out = WanRMS_norm(dim=out_dim, images=False, rngs=rngs, channel_first=False)
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  @nnx.jit(static_argnames=("feat_idx", "first_chunk"))
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0, first_chunk: bool = False):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)

    ## middle
    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)
    ## upsamples
    for up_block in self.up_blocks:
      x, feat_cache, feat_idx = up_block(x, feat_cache, feat_idx, first_chunk)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_out(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_out(x)
    return x, feat_cache, jnp.array(feat_idx, dtype=jnp.int32)


class AutoencoderKLWan2p2(nnx.Module, FlaxModelMixin, ConfigMixin):

  def __init__(
      self,
      rngs: nnx.Rngs,
      base_dim: int = 160,
      base_dec_dim: int = 256,
      z_dim: int = 48,
      dim_mult: Tuple[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_downsample: List[bool] = [False, True, True],
      dropout: float = 0.0,
      latents_mean: List[float] = [
          -0.2289,
          -0.0052,
          -0.1323,
          -0.2339,
          -0.2799,
          0.0174,
          0.1838,
          0.1557,
          -0.1382,
          0.0542,
          0.2813,
          0.0891,
          0.1570,
          -0.0098,
          0.0375,
          -0.1825,
          -0.2246,
          -0.1207,
          -0.0698,
          0.5109,
          0.2665,
          -0.2108,
          -0.2158,
          0.2502,
          -0.2055,
          -0.0322,
          0.1109,
          0.1567,
          -0.0729,
          0.0899,
          -0.2799,
          -0.1230,
          -0.0313,
          -0.1649,
          0.0117,
          0.0723,
          -0.2839,
          -0.2083,
          -0.0520,
          0.3748,
          0.0152,
          0.1957,
          0.1433,
          -0.2944,
          0.3573,
          -0.0548,
          -0.1681,
          -0.0667,
      ],
      latents_std: List[float] = [
          0.4765,
          1.0364,
          0.4514,
          1.1677,
          0.5313,
          0.4990,
          0.4818,
          0.5013,
          0.8158,
          1.0344,
          0.5894,
          1.0901,
          0.6885,
          0.6165,
          0.8454,
          0.4978,
          0.5759,
          0.3523,
          0.7135,
          0.6804,
          0.5833,
          1.4146,
          0.8986,
          0.5659,
          0.7069,
          0.5338,
          0.4889,
          0.4917,
          0.4069,
          0.4999,
          0.6866,
          0.4093,
          0.5709,
          0.6065,
          0.6415,
          0.4944,
          0.5726,
          1.2042,
          0.5458,
          1.6887,
          0.3971,
          1.0600,
          0.3943,
          0.5537,
          0.5444,
          0.4089,
          0.7468,
          0.7744,
      ],
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.z_dim = z_dim
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]
    self.latents_mean = latents_mean
    self.latents_std = latents_std

    self.patch_size = 2
    self.patchify = WanPatchify(patch_size=self.patch_size)
    self.unpatchify = WanUnpatchify(patch_size=self.patch_size)

    self.encoder = WanEncoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim * 2,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        dropout=dropout,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        in_channels=3 * self.patch_size**2,
    )
    self.quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim * 2,
        out_channels=z_dim * 2,
        kernel_size=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    self.post_quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=z_dim,
        kernel_size=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    self.decoder = WanDecoder3d(
        rngs=rngs,
        dim=base_dec_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=self.temporal_upsample,
        dropout=dropout,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        out_channels=3 * self.patch_size**2,
    )

  def _encode(self, x: jax.Array, feat_cache: AutoencoderKLWanCache):
    feat_cache.init_cache()
    # [N, C, D, H, W]

    if x.shape[-1] != 3:
      # reshape channel last for JAX
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert x.shape[-1] == 3, f"Expected input shape (N, D, H, W, 3), got {x.shape}"

    x = self.patchify(x)

    t = x.shape[1]
    iter_ = 1 + (t - 1) // 4
    enc_feat_map = feat_cache._enc_feat_map

    for i in range(iter_):
      enc_conv_idx = 0
      if i == 0:
        out, enc_feat_map, enc_conv_idx = self.encoder(x[:, :1, :, :, :], feat_cache=enc_feat_map, feat_idx=enc_conv_idx)
      else:
        out_, enc_feat_map, enc_conv_idx = self.encoder(
            x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :],
            feat_cache=enc_feat_map,
            feat_idx=enc_conv_idx,
        )
        out = jnp.concatenate([out, out_], axis=1)

    # Update back to the wrapper object if needed, but for result we use local vars
    feat_cache._enc_feat_map = enc_feat_map

    enc = self.quant_conv(out)
    mu, logvar = enc[:, :, :, :, : self.z_dim], enc[:, :, :, :, self.z_dim :]
    enc = jnp.concatenate([mu, logvar], axis=-1)
    feat_cache.init_cache()
    return enc

  def encode(
      self,
      x: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[FlaxAutoencoderKLOutput, Tuple[FlaxDiagonalGaussianDistribution]]:
    """Encode video into latent distribution."""
    h = self._encode(x, feat_cache)
    posterior = WanDiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior,)
    return FlaxAutoencoderKLOutput(latent_dist=posterior)

  def _decode(
      self,
      z: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    feat_cache.init_cache()
    iter_ = z.shape[1]
    x = self.post_quant_conv(z)

    dec_feat_map = feat_cache._feat_map

    for i in range(iter_):
      conv_idx = 0
      if i == 0:
        out, dec_feat_map, conv_idx = self.decoder(
            x[:, i : i + 1, :, :, :],
            feat_cache=dec_feat_map,
            feat_idx=conv_idx,
            first_chunk=True,
        )
      else:
        out_, dec_feat_map, conv_idx = self.decoder(x[:, i : i + 1, :, :, :], feat_cache=dec_feat_map, feat_idx=conv_idx)
        out = jnp.concatenate([out, out_], axis=1)

    feat_cache._feat_map = dec_feat_map

    out = self.unpatchify(out)
    out = jnp.clip(out, min=-1.0, max=1.0)
    feat_cache.init_cache()
    if not return_dict:
      return (out,)

    return FlaxDecoderOutput(sample=out)

  def decode(
      self,
      z: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    if z.shape[-1] != self.z_dim:
      # reshape channel last for JAX
      z = jnp.transpose(z, (0, 2, 3, 4, 1))
      assert z.shape[-1] == self.z_dim, f"Expected input shape (N, D, H, W, {self.z_dim}, got {z.shape}"
    decoded = self._decode(z, feat_cache).sample
    if not return_dict:
      return (decoded,)
    return FlaxDecoderOutput(sample=decoded)
