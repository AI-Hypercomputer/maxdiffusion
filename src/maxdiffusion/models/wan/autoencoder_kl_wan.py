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
from ...configuration_utils import ConfigMixin, flax_register_to_config
from ..modeling_flax_utils import FlaxModelMixin
from ... import common_types

BlockSizes = common_types.BlockSizes

_ACTIVATIONS = {
  "swish": jax.nn.silu,  
  "silu": jax.nn.silu,
  "relu": jax.nn.relu,
  "gelu": jax.nn.gelu,
  "mish": jax.nn.mish
}

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
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int, int]],
    *, # Mark subsequent arguments as keyword-only
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    use_bias: bool = True,
    rngs: nnx.Rngs, # rngs are required for initializing parameters,
    flash_min_seq_length: int = 4096,
    flash_block_sizes: BlockSizes = None,
    mesh: jax.sharding.Mesh = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
    attention: str = "dot_product",
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, 'kernel_size')
    self.stride = _canonicalize_tuple(stride, 3, 'stride')
    padding_tuple = _canonicalize_tuple(padding, 3, 'padding') # (D, H, W) padding amounts

    self._causal_padding = (
      (0, 0), # Batch dimension - no padding
      (2 * padding_tuple[0], 0), # Depth dimension - causal padding (pad only before)
      (padding_tuple[1], padding_tuple[1]), # Height dimension - symmetric padding
      (padding_tuple[2], padding_tuple[2]), # Width dimension - symmetric padding
      (0, 0) # Channel dimension - no padding
    )

    # Store the amount of padding needed *before* the depth dimension for caching logoic
    self._depth_padding_before = self._causal_padding[1][0] # 2 * padding_tuple[0]

    self.conv = nnx.Conv(
      in_features=in_channels,
      out_features=out_channels,
      kernel_size=self.kernel_size,
      strides=self.stride,
      use_bias=use_bias,
      padding='VALID', # Handle padding manually
      rngs=rngs
    )
  
  def __call__(self, x: jax.Array, cache_x: Optional[jax.Array] = None) -> jax.Array:
    current_padding = list(self._causal_padding) # Mutable copy
    padding_needed = self._depth_padding_before

    if cache_x is not None and padding_needed > 0:
      # Ensure cache has same spatial/channel dims, potentially different depth
      assert cache_x.shape[0] == x.shape[0] and \
             cache_x.shape[2:] == x.shape[2:], "Cache spatial/channel dims mismatch"

      cache_len = cache_x.shape[1]
      x = jnp.concatenate([cache_x, x], axis=1) # Concat along depth (D)

      padding_needed -= cache_len
      if padding_needed < 0:
        # Cache longer than needed padding, trim from start
        x = x[:, -padding_needed:, ...]
        current_padding[1] = (0, 0) # No explicit padding needed now
      else:
        # Update depth padding needed
        current_padding[1] = (padding_needed, 0)
    
    # Apply padding if any dimension requires it
    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(x, padding_to_apply, mode='constant', constant_values=0.0)
    else:
      x_padded = x
    
    out = self.conv(x_padded)
    return out

class WanRMS_norm(nnx.Module):
  def __init__(
    self,
    dim: int,
    *,
    eps: float = 1e-6,
    use_bias: bool = False,
    rngs: nnx.Rngs
  ):
    self.eps = eps
    shape = (dim,)
    self.scale = dim ** 0.5
    # Initialize gamma as parameter
    self.gamma = nnx.Param(jax.random.ones(rngs.params(), shape))
    if use_bias:
      self.bias = nnx.Param(jax.random.zeros(rngs.params(), shape))
    else:
      self.bias = None
  
  def __call__(self, x: jax.Array) -> jax.Array:
    # Expects input channels in the last dimension
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    inv_std = jax.lax.rsqrt(variance + self.eps)
    normalized = x * inv_std * self.gamma.value * self.scale
    if self.bias:
      return normalized + self.bias.value
    return normalized


class WanEncoder3d(nnx.Module):
  pass

class WanCausalConv3d(nnx.Module):
  pass

class WanDecoder3d(nnx.Module):
  pass

class AutoencoderKLWan(nnx.Module, FlaxModelMixin, ConfigMixin):
  def __init__(
    self,
    base_dim: int = 96,
    z_dim: int = 16,
    dim_mult: Tuple[int] = [1,2,4,4],
    num_res_blocks: int = 2,
    attn_scales: List[float] = [],
    temporal_downsample: List[bool] = [False, True, True],
    dropout: float = 0.0,
    latents_mean: List[float] = [
      -0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517, 1.5508,
      0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921,
    ],
    latents_std: List[float] = [
      2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
      3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160,
    ],
  ):
    self.z_dim = z_dim
    self.temporal_downsample = temporal_downsample
    self.temporal_upsample = temporal_downsample[::-1]

    self.encoder = WanEncoder3d(z_dim * 2, z_dim * 2, 1)
    self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

    self.decoder = WanDecoder3d(
      base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temporal_upsample, dropout
    )