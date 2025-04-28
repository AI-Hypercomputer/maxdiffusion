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
from ..vae_flax import FlaxAutoencoderKLOutput, FlaxDiagonalGaussianDistribution

BlockSizes = common_types.BlockSizes

CACHE_T = 2

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
    rngs: nnx.Rngs, # rngs are required for initializing parameters,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    use_bias: bool = True,
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
    rngs: nnx.Rngs,
    channel_first: bool = True,
    images: bool = True,
    eps: float = 1e-6,
    use_bias: bool = False,
  ):
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim, )
    self.eps = eps
    self.channel_first = channel_first
    self.scale = dim ** 0.5
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
  def __init__(self, scale_factor: Tuple[float, float], method: str = 'nearest'):
    # scale_factor for (H, W)
    # JAX resize works on spatial dims, H, W assumming (N, D, H, W, C) or (N, H, W, C)
    self.scale_factor = scale_factor
    self.method = method
  
  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    in_shape = x.shape
    is_3d = len(in_shape) == 5
    n, d, h, w, c = in_shape if is_3d else(in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3])

    target_h = int(h * self.scale_factor[0])
    target_w = int(w * self.scale_factor[1])

    # jax.image.resize expects (..., H, W, C)
    if is_3d:
      x_reshaped = x.reshape(n * d, h, w, c)
      out_reshaped = jax.image.resize(x_reshaped.astype(jnp.float32),
                                      (n * d, target_h, target_w, c),
                                      method=self.method)
      out = out_reshaped.reshape(n, d, target_h, target_w, c)
    else: # Asumming (N, H, W, C)
      out = jax.image.resize(x.astype(jnp.float32),
                             (n, target_h, target_w, c),
                             method=self.method)
    
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
    flash_min_seq_length: int = 4096,
    flash_block_sizes: BlockSizes = None,
    mesh: jax.sharding.Mesh = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
    attention: str = "dot_product",
  ):
    kernel_size = _canonicalize_tuple(kernel_size, 3, 'kernel_size')
    stride = _canonicalize_tuple(stride, 3, 'stride')
    self.conv = nnx.Conv(
      dim,
      dim,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=True,
      rngs=rngs
    )
  
  def __call__(self, x):
    return self.conv(x)


class WanResample(nnx.Module):
  def __init__(
    self,
    dim: int,
    mode: str,
    rngs: nnx.Rngs,
    flash_min_seq_length: int = 4096,
    flash_block_sizes: BlockSizes = None,
    mesh: jax.sharding.Mesh = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
    attention: str = "dot_product",
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
          kernel_size=(1, 3, 3),
          padding='SAME',
          use_bias=True,
          rngs=rngs,
        )
      )
    elif mode == "upsample3d":
      self.resample = nnx.Sequential(
        WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
        nnx.Conv(
          dim,
          dim // 2,
          kernel_size=(1, 3, 3),
          padding='SAME',
          use_bias=True,
          rngs=rngs,
        )
      )
      self.time_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=dim,
        out_channels=dim * 2,
        kernel_size=(3, 1, 1),
        padding=(1, 0, 0),
      )
    elif mode == "downsample2d":
      # TODO - do I need to transpose?
      self.resample = ZeroPaddedConv2D(
        dim=dim,
        rngs=rngs,
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2)
      )
    elif mode == "downsample3d":
      # TODO - do I need to transpose?
      self.resample = ZeroPaddedConv2D(
        dim=dim,
        rngs=rngs,
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2)
      )
      self.time_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels = dim,
        out_channels = dim,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding= (0, 0, 0)
      )
    else:
      self.resample = Identity()

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]) -> jax.Array:
    # Input x: (N, D, H, W, C), assume C = self.dim
    n, d, h, w, c = x.shape
    assert c == self.dim

    x = x.reshape(n*d,h,w,c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(n, d, h_new, w_new, c_new)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = jnp.copy(x)
          feat_idx[0] +=1
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
    self.conv1 = WanCausalConv3d(
      rngs=rngs,
      in_channels=in_dim,
      out_channels=out_dim,
      kernel_size=3,
      padding=1
    )
    self.norm2 = WanRMS_norm(dim=out_dim, rngs=rngs, images=False, channel_first=False)
    self.dropout = nnx.Dropout(dropout, rngs=rngs)
    self.conv2 = WanCausalConv3d(
      rngs=rngs,
      in_channels=out_dim,
      out_channels=out_dim,
      kernel_size=3,
      padding=1
    )
    self.conv_shortcut = WanCausalConv3d(
      rngs=rngs,
      in_channels=in_dim,
      out_channels=out_dim,
      kernel_size=1
    ) if in_dim != out_dim else Identity()


  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    # Apply shortcut connection
    h = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)
    x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.dropout(x)
    x = self.conv2(x)

    return x + h

class WanAttentionBlock(nnx.Module):
  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs
  ):
    self.dim = dim
    self.norm = WanRMS_norm(rngs=rngs, dim=dim, channel_first=False)
    self.to_qkv = nnx.Conv(
      in_features=dim,
      out_features=dim * 3,
      kernel_size=1,
      rngs=rngs
    )
    self.proj = nnx.Conv(
      in_features=dim,
      out_features=dim,
      kernel_size=1,
      rngs=rngs
    )
  
  def __call__(self, x: jax.Array):
    batch_size, time, height, width, channels = x.shape
    identity = x
    
    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)

    qkv = self.to_qkv(x) # Output: (N*D, H, W, C * 3)

    qkv = qkv.reshape(batch_size*time, 1, channels * 3, -1)
    qkv = jnp.transpose(qkv, (0, 1, 3, 2))
    q, k, v = jnp.split(qkv, 3, axis=-1)

    x = jax.nn.dot_product_attention(q, k, v)
    x = jnp.squeeze(x, 1).reshape(batch_size * time, height, width, channels)

    #output projection
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
    num_layers: int = 1
  ):
    self.dim = dim
    resnets = [WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs,dropout=dropout, non_linearity=non_linearity)]
    attentions = []
    for _ in range(num_layers):
      attentions.append(WanAttentionBlock(dim=dim, rngs=rngs))
      resnets.append(WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs,dropout=dropout, non_linearity=non_linearity))
    self.attentions = attentions
    self.resnets = resnets
  
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    x = self.resnets[0](x)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x = attn(x)
      x = resnet(x)
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
    non_linearity: str = "silu"
  ):
    # Create layers list
    self.resnets = []
    # Add residual blocks and attention if needed
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
      self.resnets.append(WanResidualBlock(in_dim=current_dim, out_dim=out_dim, dropout=dropout, non_linearity=non_linearity, rngs=rngs))
      current_dim = out_dim
    
    # Add upsampling layer if needed.
    self.upsamplers = None
    if upsample_mode is not None:
      self.upsamplers = WanResample(dim=out_dim, mode=upsample_mode, rngs=rngs)
  
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    return x

class WanEncoder3d(nnx.Module):
  def __init__(
    self,
    rngs: nnx.Rngs,
    dim: int =128,
    z_dim: int = 4,
    dim_mult = [1, 2, 4, 4],
    num_res_blocks = 2,
    attn_scales = [],
    temperal_downsample = [True, True, False],
    dropout = 0.0,
    non_linearity: str = 'silu',
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
    self.norm_out = WanRMS_norm(
      out_dim,
      channel_first=False,
      images=False,
      rngs=rngs
    )
    self.conv_out = WanCausalConv3d(
      rngs=rngs,
      in_channels=out_dim,
      out_channels=z_dim,
      kernel_size=3,
      padding=1
    )
  
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] +=1
    else:
      x = self.conv_in(x)
    # (1, 1, 480, 720, 96)
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
      feat_idx[0] +=1
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
    z_dim: int = 128,
    dim_mult: List[int] = [1, 2, 4, 4],
    num_res_blocks: int = 2,
    attn_scales = List[float],
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
    self.conv_in = WanCausalConv3d(
      rngs=rngs,
      in_channels=z_dim,
      out_channels=dims[0],
      kernel_size=3,
      padding=1
    )

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
      
      # Crete and add the upsampling block
      up_block = WanUpBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        upsample_mode=upsample_mode,
        non_linearity=non_linearity,
        rngs=rngs
      )
      self.up_blocks.append(up_block)

      # Update scale for next iteration
      if upsample_mode is not None:
        scale *=2.0
    
    # output blocks
    self.norm_out = nnx.RMSNorm(num_features=out_dim, )
    self.norm_out = WanRMS_norm(dim=out_dim, images=False, rngs=rngs)
    self.conv_out = WanCausalConv3d(
      rngs=rngs,
      in_channels=out_dim,
      out_channels=3,
      kernel_size=3,
      padding=1
    )
  
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=[0]):
    x = self.conv_in(x)
    return x

class AutoencoderKLWan(nnx.Module, FlaxModelMixin, ConfigMixin):
  def __init__(
    self,
    rngs: nnx.Rngs,
    base_dim: int = 96,
    z_dim: int = 16,
    dim_mult: Tuple[int] = [1,2,4,4],
    num_res_blocks: int = 2,
    attn_scales: List[float] = [],
    temperal_downsample: List[bool] = [False, True, True],
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
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]

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
    self.quant_conv = WanCausalConv3d(
      rngs=rngs,
      in_channels=z_dim * 2,
      out_channels=z_dim * 2,
      kernel_size=1
    )
    self.post_quant_conv = WanCausalConv3d(
      rngs=rngs,
      in_channels=z_dim,
      out_channels=z_dim,
      kernel_size=1,
    )

    # self.decoder = WanDecoder3d(
    #   rngs=rngs,
    #   dim=base_dim,
    #   z_dim=z_dim,
    #   dim_mult=dim_mult,
    #   num_res_blocks=num_res_blocks,
    #   attn_scales=attn_scales,
    #   temperal_upsample=self.temporal_upsample,
    #   dropout=dropout
    # )
    self.clear_cache()
  
  def clear_cache(self):
    """ Resets cache dictionaries and indices"""
    def _count_conv3d(module):
      count = 0
      node_types = nnx.graph.iter_graph([module])
      for path, value in node_types:
        if isinstance(value, WanCausalConv3d):
          count +=1
      return count

    # self._conv_num = _count_conv3d(self.decoder)
    # self._conv_idx = [0]
    # self._feat_map = [None] * self._conv_num
    # cache encode
    self._enc_conv_num = _count_conv3d(self.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num

  def _encode(self, x: jax.Array):
    self.clear_cache()
    if x.shape[-1] != 3:
      # reshape channel last for JAX
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert x.shape[-1] == 3, f"Expected input shape (N, D, H, W, 3), got {x.shape}"
    
    #self.clear_cache()

    t = x.shape[1]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
      self._enc_conv_idx = [0]
      if i == 0:
        out = self.encoder(
          x[:, :1, :, :, :],
          feat_cache=self._enc_feat_map,
          feat_idx=self._enc_conv_idx
        )
      else:
        out_ = self.encoder(
          x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :],
          feat_cache=self._enc_feat_map,
          feat_idx=self._enc_conv_idx
        )
        out = jnp.concatenate([out, out_], axis=1)
    enc = self.quant_conv(out)
    mu, logvar = enc[:, :, :, :, : self.z_dim], enc[:, :, :, :, self.z_dim :]
    enc = jnp.concatenate([mu, logvar], axis=-1)
    self.clear_cache()
    # return enc
    return enc

  def encode(self, x: jax.Array, return_dict: bool = True) -> Union[FlaxAutoencoderKLOutput, Tuple[FlaxDiagonalGaussianDistribution]]:
    """ Encode video into latent distribution."""
    h = self._encode(x)
    posterior = FlaxDiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior, )
    return FlaxAutoencoderKLOutput(latent_dist=posterior)
    