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

from typing import Tuple, List, Sequence, Union, Optional, Dict, Any

import flax
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
from ...configuration_utils import ConfigMixin
from ..modeling_flax_utils import FlaxModelMixin, get_activation
from ... import common_types
from ..vae_flax import (FlaxAutoencoderKLOutput, FlaxDiagonalGaussianDistribution, FlaxDecoderOutput)

BlockSizes = common_types.BlockSizes

CACHE_T = 2
flax.config.update("flax_always_shard_variable", False)

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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    padding_tuple = _canonicalize_tuple(padding, 3, "padding")  # (D, H, W) padding amounts
    self.mesh = mesh

    self._causal_padding = (
        (0, 0),  # Batch dimension - no padding
        (2 * padding_tuple[0], 0),  # Depth dimension - causal padding (pad only before)
        (padding_tuple[1], padding_tuple[1]),  # Height dimension - symmetric padding
        (padding_tuple[2], padding_tuple[2]),  # Width dimension - symmetric padding
        (0, 0),  # Channel dimension - no padding
    )

    # Store the amount of padding needed *before* the depth dimension for caching logic
    self._depth_padding_before = self._causal_padding[1][0]  # 2 * padding_tuple[0]

    # Set sharding dynamically based on out_channels.
    num_fsdp_axis_devices = 1
    if mesh is not None and "fsdp" in mesh.axis_names:
        num_fsdp_axis_devices = mesh.shape["fsdp"]

    kernel_sharding = (None, None, None, None, None)
    if num_fsdp_axis_devices > 1 and out_channels % num_fsdp_axis_devices == 0:
        kernel_sharding = (None, None, None, None, "conv_out")

    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=self.kernel_size,
        strides=self.stride,
        use_bias=use_bias,
        padding="VALID",  # Handle padding manually
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), kernel_sharding),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def initialize_cache(self, batch_size, height, width, dtype):
    cache = jnp.zeros(
        (batch_size, CACHE_T, height, width, self.conv.in_features), dtype=dtype
    )

    # OPTIMIZATION: Spatial Partitioning on Initialization
    if self.mesh is not None and "fsdp" in self.mesh.axis_names:
        num_fsdp_devices = self.mesh.shape["fsdp"]
        # Axis 2 is Height
        shard_axis = "fsdp" if (height % num_fsdp_devices == 0) else None

        # If height isn't divisible, try width (Axis 3)
        shard_width_axis = None
        if shard_axis is None and width % num_fsdp_devices == 0:
            shard_width_axis = "fsdp"

        cache = jax.lax.with_sharding_constraint(
            cache, PartitionSpec("data", None, shard_axis, shard_width_axis, None)
        )
    return cache

  def __call__(
      self, x: jax.Array, cache_x: Optional[jax.Array] = None
  ) -> Tuple[jax.Array, jax.Array]:
      # OPTIMIZATION: Spatial Partitioning during execution
    if self.mesh is not None and "fsdp" in self.mesh.axis_names:
      height = x.shape[2]
      width = x.shape[3]
      num_fsdp_devices = self.mesh.shape["fsdp"]

      shard_axis = "fsdp" if (height % num_fsdp_devices == 0) else None
      shard_width_axis = None
      if shard_axis is None and width % num_fsdp_devices == 0:
          shard_width_axis = "fsdp"

      x = jax.lax.with_sharding_constraint(
          x, PartitionSpec("data", None, shard_axis, shard_width_axis, None)
      )

    current_padding = list(self._causal_padding)

    if cache_x is not None:
      x_concat = jnp.concatenate([cache_x, x], axis=1)
      new_cache = x_concat[:, -CACHE_T:, ...]

      padding_needed = self._depth_padding_before - cache_x.shape[1]
      if padding_needed < 0:
          x_input = x_concat[:, -padding_needed:, ...]
          current_padding[1] = (0, 0)
      else:
          x_input = x_concat
          current_padding[1] = (padding_needed, 0)
    else:
      new_cache = x[:, -CACHE_T:, ...]
      x_input = x

    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(
          x_input, padding_to_apply, mode="constant", constant_values=0.0
      )
    else:
      x_padded = x_input

    out = self.conv(x_padded)
    return out, new_cache


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
  def __call__(self, x, cache=None):
    return x, cache


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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.conv = nnx.Conv(
        dim,
        dim,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, None)),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x, cache=None):
    return self.conv(x), cache


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
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
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
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
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

  def initialize_cache(self, batch_size, height, width, dtype):
    cache = {}
    if hasattr(self, "time_conv") and self.time_conv is not None:
        h_curr, w_curr = height, width
        cache["time_conv"] = self.time_conv.initialize_cache(
            batch_size, h_curr, w_curr, dtype
        )
    return cache

  def __call__(
      self, x: jax.Array, cache: Dict[str, Any] = None
  ) -> Tuple[jax.Array, Dict[str, Any]]:
    if cache is None:
        cache = {}
    new_cache = {}

    if self.mode == "upsample2d":
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.resample(x)
        h_new, w_new, c_new = x.shape[1:]
        x = x.reshape(b, t, h_new, w_new, c_new)

    elif self.mode == "upsample3d":
        x, tc_cache = self.time_conv(x, cache.get("time_conv"))
        new_cache["time_conv"] = tc_cache

        b, t, h, w, c = x.shape
        x = x.reshape(b, t, h, w, 2, c // 2)
        x = jnp.stack([x[:, :, :, :, 0, :], x[:, :, :, :, 1, :]], axis=1)
        x = x.reshape(b, t * 2, h, w, c // 2)

        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.resample(x)
        h_new, w_new, c_new = x.shape[1:]
        x = x.reshape(b, t, h_new, w_new, c_new)

    elif self.mode == "downsample2d":
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x, _ = self.resample(x, None)
        h_new, w_new, c_new = x.shape[1:]
        x = x.reshape(b, t, h_new, w_new, c_new)

    elif self.mode == "downsample3d":
        x, tc_cache = self.time_conv(x, cache.get("time_conv"))
        new_cache["time_conv"] = tc_cache

        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x, _ = self.resample(x, None)
        h_new, w_new, c_new = x.shape[1:]
        x = x.reshape(b, t, h_new, w_new, c_new)

    else:
        if hasattr(self, "resample"):
            if isinstance(self.resample, Identity):
                x, _ = self.resample(x, None)
            else:
                x = self.resample(x)

    return x, new_cache


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

  def initialize_cache(self, batch_size, height, width, dtype):
    cache = {
        "conv1": self.conv1.initialize_cache(batch_size, height, width, dtype),
        "conv2": self.conv2.initialize_cache(batch_size, height, width, dtype),
    }
    if isinstance(self.conv_shortcut, WanCausalConv3d):
        cache["shortcut"] = self.conv_shortcut.initialize_cache(
            batch_size, height, width, dtype
        )
    else:
        cache["shortcut"] = None
    return cache

  def __call__(self, x: jax.Array, cache: Dict[str, Any] = None):
    if cache is None:
        cache = {}
    new_cache = {}

    h, sc_cache = self.conv_shortcut(x, cache.get("shortcut"))
    new_cache["shortcut"] = sc_cache

    x = self.norm1(x)
    x = self.nonlinearity(x)

    x, c1 = self.conv1(x, cache.get("conv1"))
    new_cache["conv1"] = c1

    x = self.norm2(x)
    x = self.nonlinearity(x)

    x, c2 = self.conv2(x, cache.get("conv2"))
    new_cache["conv2"] = c2

    x = x + h
    return x, new_cache


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

  def __call__(self, x: jax.Array):

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

    return x + identity


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
    self.resnets = nnx.List(
        [
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
    )
    self.attentions = nnx.List([])
    for _ in range(num_layers):
        self.attentions.append(
            WanAttentionBlock(
                dim=dim,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )
        )
        self.resnets.append(
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

  def initialize_cache(self, batch_size, height, width, dtype):
    cache = {"resnets": []}
    for resnet in self.resnets:
        cache["resnets"].append(
            resnet.initialize_cache(batch_size, height, width, dtype)
        )
    return cache

  def __call__(self, x: jax.Array, cache: Dict[str, Any] = None):
    if cache is None:
        cache = {}
    new_cache = {"resnets": []}

    x, c = self.resnets[0](x, cache.get("resnets", [None])[0])
    new_cache["resnets"].append(c)

    for i, (attn, resnet) in enumerate(zip(self.attentions, self.resnets[1:])):
        if attn is not None:
            x = attn(x)
        x, c = resnet(x, cache.get("resnets", [None] * len(self.resnets))[i + 1])
        new_cache["resnets"].append(c)

    return x, new_cache


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
  ):
    self.resnets = nnx.List([])
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
        self.resnets.append(
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

    self.upsamplers = nnx.List([])
    if upsample_mode is not None:
        self.upsamplers.append(
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

  def initialize_cache(self, batch_size, height, width, dtype):
    cache = {"resnets": [], "upsamplers": []}
    for resnet in self.resnets:
        cache["resnets"].append(
            resnet.initialize_cache(batch_size, height, width, dtype)
        )

    h_curr, w_curr = height, width
    if self.upsamplers:
        cache["upsamplers"].append(
            self.upsamplers[0].initialize_cache(batch_size, h_curr, w_curr, dtype)
        )
    return cache

  def __call__(self, x: jax.Array, cache: Dict[str, Any] = None):
    if cache is None:
        cache = {}
    new_cache = {"resnets": [], "upsamplers": []}

    for i, resnet in enumerate(self.resnets):
        x, c = resnet(x, cache.get("resnets", [None] * len(self.resnets))[i])
        new_cache["resnets"].append(c)

    if self.upsamplers:
        x, c = self.upsamplers[0](x, cache.get("upsamplers", [None])[0])
        new_cache["upsamplers"].append(c)
    return x, new_cache


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
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # downsample blocks
    self.down_blocks = nnx.List([])
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      for _ in range(num_res_blocks):
        self.down_blocks.append(
            WanResidualBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )
        )
        if scale in attn_scales:
          self.down_blocks.append(
              WanAttentionBlock(
                  dim=out_dim, rngs=rngs, mesh=mesh, dtype=dtype, weights_dtype=weights_dtype, precision=precision
              )
          )
        in_dim = out_dim

      # downsample block
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.down_blocks.append(
            WanResample(
                out_dim, mode=mode, rngs=rngs, mesh=mesh, dtype=dtype, weights_dtype=weights_dtype, precision=precision
            )
        )
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

  def init_cache(self, batch_size, height, width, dtype):
    cache = {}
    cache["conv_in"] = self.conv_in.initialize_cache(
        batch_size, height, width, dtype
    )
    cache["down_blocks"] = []

    h_curr, w_curr = height, width
    for layer in self.down_blocks:
        if isinstance(layer, WanResidualBlock):
            cache["down_blocks"].append(
                layer.initialize_cache(batch_size, h_curr, w_curr, dtype)
            )
        elif isinstance(layer, WanResample):
            cache["down_blocks"].append(
                layer.initialize_cache(batch_size, h_curr, w_curr, dtype)
            )
            if layer.mode == "downsample2d" or layer.mode == "downsample3d":
                h_curr, w_curr = h_curr // 2, w_curr // 2
        else:
            cache["down_blocks"].append(None)

    cache["mid_block"] = self.mid_block.initialize_cache(
        batch_size, h_curr, w_curr, dtype
    )
    cache["conv_out"] = self.conv_out.initialize_cache(
        batch_size, h_curr, w_curr, dtype
    )
    return cache

  def __call__(self, x: jax.Array, cache: Dict[str, Any] = None):
    if cache is None:
        cache = {}
    new_cache = {}

    x, c = self.conv_in(x, cache.get("conv_in"))
    new_cache["conv_in"] = c

    new_cache["down_blocks"] = []
    current_down_caches = cache.get("down_blocks", [None] * len(self.down_blocks))

    for i, layer in enumerate(self.down_blocks):
        if isinstance(layer, (WanResidualBlock, WanResample)):
            x, c = layer(x, current_down_caches[i])
            new_cache["down_blocks"].append(c)
        else:
            x = layer(x)
            new_cache["down_blocks"].append(None)

    x, c = self.mid_block(x, cache.get("mid_block"))
    new_cache["mid_block"] = c

    x = self.norm_out(x)
    x = self.nonlinearity(x)

    x, c = self.conv_out(x, cache.get("conv_out"))
    new_cache["conv_out"] = c

    return x, new_cache


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
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
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
    self.up_blocks = nnx.List([])
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
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
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
        out_channels=3,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  def init_cache(self, batch_size, height, width, dtype):
    cache = {}
    cache["conv_in"] = self.conv_in.initialize_cache(
        batch_size, height, width, dtype
    )
    cache["mid_block"] = self.mid_block.initialize_cache(
        batch_size, height, width, dtype
    )
    cache["up_blocks"] = []

    h_curr, w_curr = height, width
    for block in self.up_blocks:
        cache["up_blocks"].append(
            block.initialize_cache(batch_size, h_curr, w_curr, dtype)
        )
        if block.upsamplers:
            h_curr, w_curr = h_curr * 2, w_curr * 2

    cache["conv_out"] = self.conv_out.initialize_cache(
        batch_size, h_curr, w_curr, dtype
    )
    return cache

  def __call__(self, x: jax.Array, cache: Dict[str, Any] = None):
    if cache is None:
        cache = {}
    new_cache = {}

    x, c = self.conv_in(x, cache.get("conv_in"))
    new_cache["conv_in"] = c

    x, c = self.mid_block(x, cache.get("mid_block"))
    new_cache["mid_block"] = c

    new_cache["up_blocks"] = []
    current_up_caches = cache.get("up_blocks", [None] * len(self.up_blocks))
    for i, up_block in enumerate(self.up_blocks):
        x, c = up_block(x, current_up_caches[i])
        new_cache["up_blocks"].append(c)

    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x, c = self.conv_out(x, cache.get("conv_out"))
    new_cache["conv_out"] = c

    return x, new_cache

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
        dim=base_dim,
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
    )

  @nnx.jit
  def encode(
      self, x: jax.Array, return_dict: bool = True
  ) -> Union[FlaxAutoencoderKLOutput, Tuple[FlaxDiagonalGaussianDistribution]]:
      if x.shape[-1] != 3:
          x = jnp.transpose(x, (0, 2, 3, 4, 1))
      
      b, t, h, w, c = x.shape
      
      # --- STEP 1: Process First Frame (With Padding Hack) ---
      x_first = x[:, :1, ...] # (B, 1, ...)
      
      # We PAD this single frame to T=4 so it survives the strides.
      # We will take only the first result.
      x_first_padded = jnp.concatenate([x_first] * 4, axis=1) # (B, 4, ...)
      
      # Initialize Cache
      init_cache = self.encoder.init_cache(b, h, w, x.dtype)
      
      # Run Encoder on padded first frame
      # We discard the cache update here because this is a "Fake" run to get the latent
      # BUT wait, we need the cache state for the next frames.
      # This is tricky. If we pad [0, 0, 0, 0], the cache will be filled with Frame 0's history.
      # This is actually correct for a static image or start of video.
      
      enc_first_padded, cache_after_first = self.encoder(x_first_padded, init_cache)
      
      # Take only the first frame of the output
      enc_first = enc_first_padded[:, :1, ...]

      # --- STEP 2: Process Rest of Frames (Chunks of 4) ---
      x_rest = x[:, 1:, ...]
      t_rest = t - 1
      
      # Pad remainder to be divisible by 4
      pad_len = (4 - (t_rest % 4)) % 4
      if pad_len > 0:
          last = x_rest[:, -1:, ...]
          padding = jnp.repeat(last, pad_len, axis=1)
          x_rest_padded = jnp.concatenate([x_rest, padding], axis=1)
      else:
          x_rest_padded = x_rest
          
      num_chunks = x_rest_padded.shape[1] // 4
      x_chunks = x_rest_padded.reshape(b, num_chunks, 4, h, w, c)
      x_chunks = jnp.transpose(x_chunks, (1, 0, 2, 3, 4, 5))

      # Scan Function
      def scan_fn(carry, input_chunk):
          out_chunk, new_carry = self.encoder(input_chunk, carry)
          return new_carry, out_chunk

      final_cache, enc_rest_chunks = jax.lax.scan(scan_fn, cache_after_first, x_chunks)

      # Flatten Rest
      enc_rest_chunks = jnp.swapaxes(enc_rest_chunks, 0, 1)
      b_out, n_chunks, t_chunk, h_out, w_out, c_out = enc_rest_chunks.shape
      enc_rest = enc_rest_chunks.reshape(b_out, n_chunks * t_chunk, h_out, w_out, c_out)
      
      # Slice off padding from result if needed
      # We padded input by 'pad_len'. Output is downsampled by 4 (likely).
      # Actually, since we chunked by 4 and got 1 output, the mapping is 1-to-1 chunk-to-latent.
      # If we added 1 chunk of padding, we remove 1 frame of output.
      if pad_len > 0:
          # We padded inputs. Does that mean we generated extra latents?
          # If t_rest=5. Pad to 8. Chunks=2. Output=2 latents.
          # Real latents needed: ceil(5/4) = 2.
          # So actually, we don't need to slice! The ceiling behavior is what we want.
          pass

      # Concatenate
      encoded = jnp.concatenate([enc_first, enc_rest], axis=1)

      # Quantize
      enc, _ = self.quant_conv(encoded)
      mu, logvar = enc[:, :, :, :, : self.z_dim], enc[:, :, :, :, self.z_dim :]
      h_latents = jnp.concatenate([mu, logvar], axis=-1)

      posterior = FlaxDiagonalGaussianDistribution(h_latents)
      if not return_dict:
          return (posterior,)
      return FlaxAutoencoderKLOutput(latent_dist=posterior)

  @nnx.jit
  def decode(
      self, z: jax.Array, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    if z.shape[-1] != self.z_dim:
        z = jnp.transpose(z, (0, 2, 3, 4, 1))

    x, _ = self.post_quant_conv(z)
    x_scan = jnp.swapaxes(x, 0, 1)

    b, t, h, w, c = x.shape
    init_cache = self.decoder.init_cache(b, h, w, x.dtype)

    def scan_fn(carry, input_slice):
        # Expand Time dimension for Conv3d
        input_slice = jnp.expand_dims(input_slice, 1)
        # OPTIMIZATION: Force bfloat16 accumulation within the scan
        # to save memory on the massive output buffer
        out_slice, new_carry = self.decoder(input_slice, carry)
        # out_slice = out_slice.astype(jnp.bfloat16)
        return new_carry, out_slice

    final_cache, decoded_frames = jax.lax.scan(scan_fn, init_cache, x_scan)

    # decoded_frames shape: (T_lat, B, 4, H, W, C)
    # We need to flatten T_lat and 4.
    # Transpose to (B, T_lat, 4, H, W, C)
    decoded = jnp.transpose(decoded_frames, (1, 0, 2, 3, 4, 5))

    # Reshape to (B, T_lat*4, H, W, C)
    b, t_lat, t_sub, h, w, c = decoded.shape
    decoded = decoded.reshape(b, t_lat * t_sub, h, w, c)

    out = jnp.clip(decoded, min=-1.0, max=1.0)

    if not return_dict:
        return (out,)
    return FlaxDecoderOutput(sample=out)