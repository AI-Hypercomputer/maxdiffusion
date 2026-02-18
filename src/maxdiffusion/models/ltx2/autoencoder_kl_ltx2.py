# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_flax_utils import FlaxModelMixin, get_activation
from ... import common_types
from ..vae_flax import FlaxDiagonalGaussianDistribution, FlaxAutoencoderKLOutput, FlaxDecoderOutput
from ..embeddings_flax import NNXPixArtAlphaCombinedTimestepSizeEmbeddings

def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  """Canonicalizes a value to a tuple of integers."""
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank} integers. Got {x}")


class PerChannelRMSNorm(nnx.Module):
  """
  Per-pixel (per-location) RMS normalization layer.
  
  For each element along the chosen dimension, this layer normalizes the tensor by the root-mean-square of its values
  across that dimension.
  """
  def __init__(self, channel_dim: int = -1, eps: float = 1e-8, rngs: Optional[nnx.Rngs] = None, dtype: jnp.dtype = jnp.float32):
    self.eps = eps
    self.channel_dim = channel_dim

  def __call__(self, x: jax.Array, channel_dim: Optional[int] = None) -> jax.Array:
    channel_dim = channel_dim if channel_dim is not None else self.channel_dim
    # Compute mean of squared values along channel dimension.
    mean_sq = jnp.mean(jnp.square(x), axis=channel_dim, keepdims=True)
    rms = jnp.sqrt(mean_sq + self.eps)
    return x / rms


class LTX2VideoCausalConv3d(nnx.Module):
  """
  3D Causal Convolution with replication padding for time dimension.
  """
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, int, int]] = 3,
      stride: Union[int, Tuple[int, int, int]] = 1,
      dilation: Union[int, Tuple[int, int, int]] = 1,
      groups: int = 1,
      spatial_padding_mode: str = "constant", # 'constant' or 'reflect'
      rngs: nnx.Rngs = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    if isinstance(dilation, int):
        self.dilation = (dilation, 1, 1)
    else:
        self.dilation = _canonicalize_tuple(dilation, 3, "dilation")
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    self.spatial_padding_mode = spatial_padding_mode
    
    self.time_kernel_size = self.kernel_size[0]
    self.height_pad = self.kernel_size[1] // 2
    self.width_pad = self.kernel_size[2] // 2
    
    self.pad_time_causal = self.time_kernel_size - 1
    self.pad_time_non_causal = (self.time_kernel_size - 1) // 2
    
    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=self.kernel_size,
        strides=self.stride,
        kernel_dilation=self.dilation,
        feature_group_count=groups,
        use_bias=True,
        padding="VALID",
        rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, hidden_states: jax.Array, causal: bool = True) -> jax.Array:
    
    # 1. Temporal Padding (Replication)
    if causal:
        if self.pad_time_causal > 0:
            first_frame = hidden_states[:, :1, ...]
            pad_left = jnp.repeat(first_frame, self.pad_time_causal, axis=1)
            hidden_states = jnp.concatenate([pad_left, hidden_states], axis=1)
    else:
        if self.pad_time_non_causal > 0:
            first_frame = hidden_states[:, :1, ...]
            last_frame = hidden_states[:, -1:, ...]
            pad_left = jnp.repeat(first_frame, self.pad_time_non_causal, axis=1)
            pad_right = jnp.repeat(last_frame, self.pad_time_non_causal, axis=1)
            hidden_states = jnp.concatenate([pad_left, hidden_states, pad_right], axis=1)

    # 2. Spatial Padding
    pad_h = self.height_pad
    pad_w = self.width_pad
    
    if pad_h > 0 or pad_w > 0:
        padding_config = (
            (0, 0), # B
            (0, 0), # T
            (pad_h, pad_h), # H
            (pad_w, pad_w), # W
            (0, 0)  # C
        )
        
        mode = self.spatial_padding_mode
        if mode == "zeros":
            mode = "constant"
        elif mode == "replicate":
            mode = "edge"
        hidden_states = jnp.pad(hidden_states, padding_config, mode=mode)
    
    # 3. Conv
    hidden_states = self.conv(hidden_states)
    

    return hidden_states


class LTX2VideoResnetBlock3d(nnx.Module):
  """
  A 3D ResNet block used in the LTX 2.0 audiovisual model.
  """
  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      dropout: float = 0.0,
      eps: float = 1e-6,
      elementwise_affine: bool = False,
      non_linearity: str = "swish",
      inject_noise: bool = False,
      timestep_conditioning: bool = False,
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    out_channels = out_channels or in_channels
    self.nonlinearity = get_activation(non_linearity)

    self.norm1 = PerChannelRMSNorm()
    self.conv1 = LTX2VideoCausalConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    self.norm2 = PerChannelRMSNorm()
    self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    self.conv2 = LTX2VideoCausalConv3d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    if in_channels != out_channels:
        self.norm3 = nnx.LayerNorm(in_channels, epsilon=eps, use_scale=True, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.conv_shortcut = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            use_bias=True,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision
        )
    else:
        self.norm3 = None
        self.conv_shortcut = None

    if inject_noise:
        self.per_channel_scale1 = nnx.Param(jnp.zeros((in_channels,), dtype=dtype))
        self.per_channel_scale2 = nnx.Param(jnp.zeros((in_channels,), dtype=dtype))
    else:
        self.per_channel_scale1 = None
        self.per_channel_scale2 = None

    if timestep_conditioning:
        self.scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params(), (4, in_channels)) / (in_channels ** 0.5)
        )
    else:
        self.scale_shift_table = None

  def __call__(
      self,
      hidden_states: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = True,
      deterministic: bool = True
  ) -> jax.Array:
    inputs = hidden_states

    hidden_states = self.norm1(hidden_states)

    if self.scale_shift_table is not None:
        B, C = inputs.shape[0], inputs.shape[-1]
        temb = temb.reshape(B, 4, C)
        
        temb = temb + self.scale_shift_table.value[None, :, :]
        
        shift_1 = temb[:, 0, :]
        scale_1 = temb[:, 1, :]
        shift_2 = temb[:, 2, :]
        scale_2 = temb[:, 3, :]
        
        hidden_states = hidden_states * (1 + scale_1[:, None, None, None, :]) + shift_1[:, None, None, None, :]

    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.conv1(hidden_states, causal=causal)

    if self.per_channel_scale1 is not None and key is not None and not deterministic:
        key, subkey = jax.random.split(key)
        H, W = hidden_states.shape[2], hidden_states.shape[3]
        spatial_noise = jax.random.normal(subkey, (H, W), dtype=hidden_states.dtype)
        noise_scaled = spatial_noise[..., None] * self.per_channel_scale1.value
        hidden_states = hidden_states + noise_scaled[None, None, ...]

    hidden_states = self.norm2(hidden_states)

    if self.scale_shift_table is not None:
         hidden_states = hidden_states * (1 + scale_2[:, None, None, None, :]) + shift_2[:, None, None, None, :]

    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    hidden_states = self.conv2(hidden_states, causal=causal)

    if self.per_channel_scale2 is not None and key is not None and not deterministic:
        key, subkey = jax.random.split(key)
        H, W = hidden_states.shape[2], hidden_states.shape[3]
        spatial_noise = jax.random.normal(subkey, (H, W), dtype=hidden_states.dtype)
        noise_scaled = spatial_noise[..., None] * self.per_channel_scale2.value
        hidden_states = hidden_states + noise_scaled[None, None, ...]

    if self.norm3 is not None:
        inputs = self.norm3(inputs)

    if self.conv_shortcut is not None:
        inputs = self.conv_shortcut(inputs)

    hidden_states = hidden_states + inputs

    return hidden_states


class LTXVideoDownsampler3d(nnx.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      stride: Union[int, Tuple[int, int, int]] = 1,
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    self.group_size = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // out_channels
    
    conv_out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])
    
    self.conv = LTX2VideoCausalConv3d(
        in_channels=in_channels,
        out_channels=conv_out_channels,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

  def __call__(self, hidden_states: jax.Array, causal: bool = True) -> jax.Array: 
    # 1. Residual Path
    t_stride = self.stride[0]
    if t_stride > 1:
        pad = hidden_states[:, :t_stride-1, ...]
        padded_states = jnp.concatenate([pad, hidden_states], axis=1)
    else:
        padded_states = hidden_states
        
    B, T_pad, H, W, C = padded_states.shape
    new_T = T_pad // self.stride[0]
    new_H = H // self.stride[1]
    new_W = W // self.stride[2]
    
    residual = padded_states.reshape(B, new_T, self.stride[0], new_H, self.stride[1], new_W, self.stride[2], C)
    
    # transpose: (B, new_T, stride_T, new_H, stride_H, new_W, stride_W, C) -> (B, new_T, new_H, new_W, C, stride_T, stride_H, stride_W)
    # This matches PyTorch channel grouping
    residual = residual.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    
    # Flatten last 4 dims: C*stride_T*stride_H*stride_W
    residual = residual.reshape(B, new_T, new_H, new_W, -1)
    
    # Now reshape to (..., out_channels_target, group_size)
    residual = residual.reshape(B, new_T, new_H, new_W, -1, self.group_size)
    
    # Mean over group_size
    residual = jnp.mean(residual, axis=-1)
    
    # 2. Conv Path
    conv_out = self.conv(padded_states, causal=causal)
    C_conv = conv_out.shape[-1]
    
    conv_out = conv_out.reshape(B, new_T, self.stride[0], new_H, self.stride[1], new_W, self.stride[2], C_conv)
    # Transpose identically to match the residual flat feature order
    conv_out = conv_out.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    conv_out = conv_out.reshape(B, new_T, new_H, new_W, -1)
    
    # 3. Add Residual
    return conv_out + residual


class LTXVideoUpsampler3d(nnx.Module):
  def __init__(
      self,
      in_channels: int,
      stride: Union[int, Tuple[int, int, int]] = 1,
      residual: bool = False,
      upscale_factor: int = 1,
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    self.residual = residual
    self.upscale_factor = upscale_factor
    
    self.out_channels_conv = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // upscale_factor
    
    self.conv = LTX2VideoCausalConv3d(
        in_channels=in_channels,
        out_channels=self.out_channels_conv,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

  def __call__(self, hidden_states: jax.Array, causal: bool = True) -> jax.Array:
    B, T, H, W, C = hidden_states.shape
    s0, s1, s2 = self.stride

    # Residual path
    residual = None
    if self.residual:
        C_out = C // (s0 * s1 * s2)
        residual = hidden_states.reshape(B, T, H, W, C_out, s0, s1, s2)
        residual = residual.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        residual = residual.reshape(B, T*s0, H*s1, W*s2, C_out)
        
        repeats = (s0 * s1 * s2) // self.upscale_factor
        if repeats > 1:
            residual = jnp.tile(residual, (1, 1, 1, 1, repeats))
            
        if s0 > 1:
            residual = residual[:, s0-1:, ...]

    # Conv path
    hidden_states = self.conv(hidden_states, causal=causal)
    
    # PixelShuffle
    hidden_states = hidden_states.reshape(B, T, H, W, -1, s0, s1, s2)
    hidden_states = hidden_states.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    hidden_states = hidden_states.reshape(B, T*s0, H*s1, W*s2, -1)
    
    if s0 > 1:
        hidden_states = hidden_states[:, s0-1:, ...]

    if self.residual:
        hidden_states = hidden_states + residual
        

    return hidden_states


class LTX2VideoDownBlock3D(nnx.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      num_layers: int = 1,
      dropout: float = 0.0,
      resnet_eps: float = 1e-6,
      resnet_act_fn: str = "swish",
      spatio_temporal_scale: bool = True,
      downsample_type: str = "conv",
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    out_channels = out_channels or in_channels
    

    self.resnets = nnx.List([
        LTX2VideoResnetBlock3d(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=dropout,
            eps=resnet_eps,
            non_linearity=resnet_act_fn,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision
        )
        for _ in range(num_layers)
    ])
    
    self.downsamplers = nnx.List([])
    if spatio_temporal_scale:
        if downsample_type == "conv":
            self.downsamplers.append(
                LTX2VideoCausalConv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                    precision=precision
                )
            )
        elif downsample_type == "spatial":
            self.downsamplers.append(
                LTXVideoDownsampler3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                    precision=precision
                )
            )
        elif downsample_type == "temporal":
             self.downsamplers.append(
                LTXVideoDownsampler3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                    precision=precision
                )
            )
        elif downsample_type == "spatiotemporal":
             self.downsamplers.append(
                LTXVideoDownsampler3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                    precision=precision
                )
            )

  def __call__(
      self,
      hidden_states: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = True,
      deterministic: bool = True
  ) -> jax.Array:
    for resnet in self.resnets:
        subkey = None
        if key is not None:
             key, subkey = jax.random.split(key)
        hidden_states = resnet(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states, causal=causal)
        

    return hidden_states


class LTX2VideoMidBlock3d(nnx.Module):
  def __init__(
      self,
      in_channels: int,
      num_layers: int = 1,
      dropout: float = 0.0,
      resnet_eps: float = 1e-6,
      resnet_act_fn: str = "swish",
      inject_noise: bool = False,
      timestep_conditioning: bool = False,
      spatial_padding_mode: str = "zeros",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    if timestep_conditioning:
      self.time_embedder = nnx.data(NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
          rngs=rngs,
          embedding_dim=in_channels * 4,
          size_emb_dim=0,
          use_additional_conditions=False,
          dtype=dtype,
          weights_dtype=weights_dtype
      ))
    else:
      self.time_embedder = None

    self.resnets = nnx.List([
        LTX2VideoResnetBlock3d(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=dropout,
            eps=resnet_eps,
            non_linearity=resnet_act_fn,
            inject_noise=inject_noise,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision
        )
        for _ in range(num_layers)
    ])

  def __call__(
      self,
      hidden_states: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = True,
      deterministic: bool = True
  ) -> jax.Array:
    if self.time_embedder is not None:
         temb = self.time_embedder(timestep=temb.flatten(), hidden_dtype=hidden_states.dtype)
         temb = temb.reshape(temb.shape[0], 1, 1, 1, -1)

    for resnet in self.resnets:
        subkey = None
        if key is not None:
            key, subkey = jax.random.split(key)

        hidden_states = resnet(
            hidden_states,
            temb=temb,
            key=subkey,
            causal=causal,
            deterministic=deterministic
        )


    return hidden_states


class LTX2VideoUpBlock3d(nnx.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      num_layers: int = 1,
      dropout: float = 0.0,
      resnet_eps: float = 1e-6,
      resnet_act_fn: str = "swish",
      spatio_temporal_scale: bool = True,
      inject_noise: bool = False,
      timestep_conditioning: bool = False,
      upsample_residual: bool = False,
      upscale_factor: int = 1,
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    out_channels = out_channels or in_channels
    
    self.time_embedder = None
    if timestep_conditioning:
        self.time_embedder = nnx.data(NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
            rngs=rngs,
            embedding_dim=in_channels * 4,
            size_emb_dim=0,
            use_additional_conditions=False,
            dtype=dtype,
            weights_dtype=weights_dtype
        ))

    self.conv_in = None
    if in_channels != out_channels:
        self.conv_in = nnx.data(LTX2VideoResnetBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            eps=resnet_eps,
            non_linearity=resnet_act_fn,
            inject_noise=inject_noise,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision
        ))

    self.upsamplers = nnx.List([])
    if spatio_temporal_scale:
        self.upsamplers.append(
            LTXVideoUpsampler3d(
                in_channels=out_channels * upscale_factor, # Wait, reference passes `out_channels * upscale_factor`
                stride=(2, 2, 2),
                residual=upsample_residual,
                upscale_factor=upscale_factor,
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision
            )
        )
    
    self.resnets = nnx.List([
        LTX2VideoResnetBlock3d(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout=dropout,
            eps=resnet_eps,
            non_linearity=resnet_act_fn,
            inject_noise=inject_noise,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision
        )
        for _ in range(num_layers)
    ])

  def __call__(
      self,
      hidden_states: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = True,
      deterministic: bool = True
  ) -> jax.Array:
    if self.conv_in is not None:
        subkey = None
        if key is not None:
            key, subkey = jax.random.split(key)
        hidden_states = self.conv_in(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    if self.time_embedder is not None:
        temb = self.time_embedder(timestep=temb.flatten(), hidden_dtype=hidden_states.dtype)
        temb = temb.reshape(temb.shape[0], 1, 1, 1, -1)

    for upsampler in self.upsamplers:
        hidden_states = upsampler(hidden_states, causal=causal)
    
    for resnet in self.resnets:
        subkey = None
        if key is not None:
            key, subkey = jax.random.split(key)
        
        hidden_states = resnet(
            hidden_states,
            temb=temb,
            key=subkey,
            causal=causal,
            deterministic=deterministic
        )


    return hidden_states


class LTX2VideoEncoder3d(nnx.Module):
  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 128,
      block_out_channels: Tuple[int, ...] = (256, 512, 1024, 2048),
      down_block_types: Tuple[str, ...] = (
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
      ),
      spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, True),
      layers_per_block: Tuple[int, ...] = (4, 6, 6, 2, 2),
      downsample_type: Tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
      patch_size: int = 4,
      patch_size_t: int = 1,
      resnet_norm_eps: float = 1e-6,
      is_causal: bool = True,
      spatial_padding_mode: str = "constant",
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.in_channels = in_channels * patch_size**2
    self.is_causal = is_causal

    output_channel = out_channels

    self.conv_in = LTX2VideoCausalConv3d(
        in_channels=self.in_channels,
        out_channels=output_channel,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    # down blocks
    num_block_out_channels = len(block_out_channels)
    self.down_blocks = nnx.List([
        LTX2VideoDownBlock3D(
            in_channels=output_channel if i == 0 else block_out_channels[i-1],
            out_channels=block_out_channels[i],
            num_layers=layers_per_block[i],
            resnet_eps=resnet_norm_eps,
            spatio_temporal_scale=spatio_temporal_scaling[i],
            downsample_type=downsample_type[i],
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision
        ) for i in range(num_block_out_channels)
    ])

    # Update output_channel for mid block
    output_channel = block_out_channels[-1]

    # mid block
    self.mid_block = LTX2VideoMidBlock3d(
        in_channels=output_channel,
        num_layers=layers_per_block[-1],
        resnet_eps=resnet_norm_eps,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    # out
    self.norm_out = PerChannelRMSNorm()
    self.conv_act = nnx.silu
    
    self.conv_out = LTX2VideoCausalConv3d(
        in_channels=output_channel,
        out_channels=out_channels + 1,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

  # Using static_argnames for boolean flags that affect control flow or shapes
  @nnx.jit(static_argnames=("causal", "deterministic"))
  def __call__(
      self,
      sample: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = True,
      deterministic: bool = True
  ) -> jax.Array:
    # JAX: (B, T, H, W, C)
    B, T, H, W, C = sample.shape
    p = self.patch_size
    p_t = self.patch_size_t
    
    hidden_states = sample.reshape(B, T//p_t, p_t, H//p, p, W//p, p, C)
    hidden_states = hidden_states.transpose(0, 1, 3, 5, 7, 2, 6, 4)
    hidden_states = hidden_states.reshape(B, T//p_t, H//p, W//p, -1)
    
    num_blocks = len(self.down_blocks) + 1
    keys = None
    if key is not None:
        keys = jax.random.split(key, num_blocks)
    
    hidden_states = self.conv_in(hidden_states, causal=causal)

    for i, down_block in enumerate(self.down_blocks):
        subkey = keys[i] if keys is not None else None
        hidden_states = down_block(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    subkey = keys[-1] if keys is not None else None
    hidden_states = self.mid_block(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    hidden_states = self.norm_out(hidden_states)
    hidden_states = self.conv_act(hidden_states)

    hidden_states = self.conv_out(hidden_states, causal=causal)


    return hidden_states


class LTX2VideoDecoder3d(nnx.Module):
  def __init__(
      self,
      in_channels: int = 128,
      out_channels: int = 3,
      block_out_channels: Tuple[int, ...] = (256, 512, 1024),
      spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True),
      layers_per_block: Tuple[int, ...] = (5, 5, 5, 5),
      patch_size: int = 4,
      patch_size_t: int = 1,
      resnet_norm_eps: float = 1e-6,
      is_causal: bool = False,
      inject_noise: Tuple[bool, ...] = (False, False, False, False),
      timestep_conditioning: bool = False,
      upsample_residual: Tuple[bool, ...] = (True, True, True),
      upsample_factor: Tuple[int, ...] = (2, 2, 2),
      spatial_padding_mode: str = "reflect", 
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.out_channels = out_channels * patch_size**2
    self.is_causal = is_causal

    block_out_channels = tuple(reversed(block_out_channels))
    spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
    layers_per_block = tuple(reversed(layers_per_block))
    inject_noise = tuple(reversed(inject_noise))
    upsample_residual = tuple(reversed(upsample_residual))
    upsample_factor = tuple(reversed(upsample_factor))
    output_channel = block_out_channels[0]

    self.conv_in = LTX2VideoCausalConv3d(
        in_channels=in_channels,
        out_channels=output_channel,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    self.mid_block = LTX2VideoMidBlock3d(
        in_channels=output_channel,
        num_layers=layers_per_block[0],
        resnet_eps=resnet_norm_eps,
        inject_noise=inject_noise[0],
        timestep_conditioning=timestep_conditioning,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    # up blocks
    num_block_out_channels = len(block_out_channels)
    self.up_blocks = nnx.List([])
    
    for i in range(num_block_out_channels):
        input_channel = output_channel // upsample_factor[i]
        output_channel = block_out_channels[i] // upsample_factor[i]
        
        self.up_blocks.append(
            LTX2VideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=decoder_layers_per_block[i],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision
            )
        )

    # out
    self.norm_out = PerChannelRMSNorm()
    self.conv_act = nnx.silu
    
    self.conv_out = LTX2VideoCausalConv3d(
        in_channels=output_channel,
        out_channels=self.out_channels,
        kernel_size=3,
        stride=1,
        spatial_padding_mode=spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    # timestep embedding
    self.scale_shift_table = None
    self.timestep_scale_multiplier = None
    if timestep_conditioning:
      self.timestep_scale_multiplier = nnx.Param(jnp.array(1000.0, dtype=jnp.float32))
      self.time_embedder = nnx.data(NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
          rngs=rngs,
          embedding_dim=in_channels * 4,
          size_emb_dim=0,
          use_additional_conditions=False,
          dtype=dtype,
          weights_dtype=weights_dtype
      ))
    else:
      self.timestep_scale_multiplier = None
      self.time_embedder = None

  @nnx.jit(static_argnames=("causal", "deterministic"))
  def __call__(
      self,
      sample: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: bool = False,
      deterministic: bool = True
  ) -> jax.Array:
    if self.timestep_scale_multiplier is not None and temb is not None:
        temb = temb * self.timestep_scale_multiplier.value
        
    hidden_states = self.conv_in(sample, causal=causal)
    
    subkey = None
    if key is not None:
        key, subkey = jax.random.split(key)
    
    hidden_states = self.mid_block(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    for up_block in self.up_blocks:
        subkey = None
        if key is not None:
            key, subkey = jax.random.split(key)
        hidden_states = up_block(hidden_states, temb=temb, key=subkey, causal=causal, deterministic=deterministic)

    hidden_states = self.norm_out(hidden_states)
    
    hidden_states = self.conv_act(hidden_states)
    hidden_states = self.conv_out(hidden_states, causal=causal)

    
    # Unpatchify
    B, T, H, W, C = hidden_states.shape
    p = self.patch_size
    p_t = self.patch_size_t
    
    # (B, T, H, W, C) -> (B, T*p_t, H*p, W*p, C/(p_t*p*p))
    C_out_final = C // (p_t * p * p)
    
    hidden_states = hidden_states.reshape(B, T, H, W, C_out_final, p_t, p, p)
    hidden_states = hidden_states.transpose(0, 1, 5, 2, 7, 3, 6, 4)
    hidden_states = hidden_states.reshape(B, T * p_t, H * p, W * p, C_out_final)
    

    return hidden_states



class LTX2DiagonalGaussianDistribution(nnx.Module):
    def __init__(self, parameters: jax.Array, cls_latent_channels: int = 128, deterministic: bool = False):
        self.parameters = parameters
        # Split into mean and logvar
        self.mean, self.logvar = jnp.split(parameters, [cls_latent_channels], axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(
                self.mean, dtype=self.parameters.dtype
            )

    def sample(self, key: jax.Array) -> jax.Array:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = jax.random.normal(key, self.mean.shape, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "LTX2DiagonalGaussianDistribution" = None) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0])
        else:
            if other is None:
                return 0.5 * jnp.sum(
                    jnp.power(self.mean, 2) + self.var - 1.0 - self.logvar,
                    axis=[1, 2, 3],
                )
            else:
                return 0.5 * jnp.sum(
                    jnp.power(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    axis=[1, 2, 3],
                )

    def nll(self, sample: jax.Array, dims: Tuple[int, ...] = (1, 2, 3)) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0])
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.power(sample - self.mean, 2) / self.var,
            axis=dims,
        )

    def mode(self) -> jax.Array:
        return self.mean


class LTX2VideoAutoencoderKL(nnx.Module, ConfigMixin):
  _supports_gradient_checkpointing = True
  config_name = "config.json"

  @register_to_config
  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 3,
      latent_channels: int = 128,
      block_out_channels: Tuple[int, ...] = (256, 512, 1024, 2048),
      down_block_types: Tuple[str, ...] = (
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
          "LTX2VideoDownBlock3D",
      ),
      decoder_block_out_channels: Tuple[int, ...] = (256, 512, 1024),
      layers_per_block: Tuple[int, ...] = (4, 6, 6, 2, 2),
      decoder_layers_per_block: Tuple[int, ...] = (5, 5, 5, 5),
      spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, True),
      decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True),
      decoder_inject_noise: Tuple[bool, ...] = (False, False, False, False),
      downsample_type: Tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
      upsample_residual: Tuple[bool, ...] = (True, True, True),
      upsample_factor: Tuple[int, ...] = (2, 2, 2),
      timestep_conditioning: bool = False,
      patch_size: int = 4,
      patch_size_t: int = 1,
      resnet_norm_eps: float = 1e-6,
      scaling_factor: float = 1.0,
      encoder_causal: bool = True,
      decoder_causal: bool = True,
      encoder_spatial_padding_mode: str = "zeros",
      decoder_spatial_padding_mode: str = "reflect",
      spatial_compression_ratio: Optional[int] = None,
      temporal_compression_ratio: Optional[int] = None,
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.encoder = LTX2VideoEncoder3d(
        in_channels=in_channels,
        out_channels=latent_channels,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        spatio_temporal_scaling=spatio_temporal_scaling,
        down_block_types=down_block_types,
        downsample_type=downsample_type,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        resnet_norm_eps=resnet_norm_eps,
        is_causal=encoder_causal,
        spatial_padding_mode=encoder_spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )

    self.decoder = LTX2VideoDecoder3d(
        in_channels=latent_channels,
        out_channels=out_channels,
        block_out_channels=decoder_block_out_channels,
        layers_per_block=decoder_layers_per_block,
        spatio_temporal_scaling=decoder_spatio_temporal_scaling,
        upsample_factor=upsample_factor,
        upsample_residual=upsample_residual,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        resnet_norm_eps=resnet_norm_eps,
        is_causal=decoder_causal,
        inject_noise=decoder_inject_noise,
        timestep_conditioning=timestep_conditioning,
        spatial_padding_mode=decoder_spatial_padding_mode,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
    )
    
    self.scaling_factor = scaling_factor
    self.latents_mean = jnp.zeros((latent_channels,), dtype=dtype)
    self.latents_std = jnp.ones((latent_channels,), dtype=dtype)
    self.encoder_causal = encoder_causal
    self.decoder_causal = decoder_causal
    
    self.spatial_compression_ratio = (
        patch_size * 2 ** sum(spatio_temporal_scaling)
        if spatial_compression_ratio is None
        else spatial_compression_ratio
    )
    self.temporal_compression_ratio = (
        patch_size_t * 2 ** sum(spatio_temporal_scaling)
        if temporal_compression_ratio is None
        else temporal_compression_ratio
    )

    # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
    # to perform decoding of a single video latent at a time.
    self.use_slicing = False

    # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
    # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
    # intermediate tiles together, the memory requirement can be lowered.
    self.use_tiling = False

    # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
    # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
    self.use_framewise_encoding = False
    self.use_framewise_decoding = False

    # This can be configured based on the amount of GPU memory available.
    # `16` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
    # Setting it to higher values results in higher memory usage.
    self.num_sample_frames_batch_size = 16
    self.num_latent_frames_batch_size = 2

    # The minimal tile height and width for spatial tiling to be used
    self.tile_sample_min_height = 512
    self.tile_sample_min_width = 512
    self.tile_sample_min_num_frames = 16

    # The minimal distance between two spatial tiles
    self.tile_sample_stride_height = 448
    self.tile_sample_stride_width = 448
    self.tile_sample_stride_num_frames = 8

  def enable_tiling(
      self,
      tile_sample_min_height: Optional[int] = None,
      tile_sample_min_width: Optional[int] = None,
      tile_sample_min_num_frames: Optional[int] = None,
      tile_sample_stride_height: Optional[float] = None,
      tile_sample_stride_width: Optional[float] = None,
      tile_sample_stride_num_frames: Optional[float] = None,
  ) -> None:
    self.use_tiling = True
    self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
    self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
    self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
    self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
    self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
    self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames


  def blend_v(self, a: jax.Array, b: jax.Array, blend_extent: int) -> jax.Array:
      blend_extent = min(a.shape[2], b.shape[2], blend_extent)
      for y in range(blend_extent):
          val = a[:, :, -blend_extent + y, :, :] * (1 - y / blend_extent) + b[:, :, y, :, :] * (
              y / blend_extent
          )
          b = b.at[:, :, y, :, :].set(val)
      return b

  def blend_h(self, a: jax.Array, b: jax.Array, blend_extent: int) -> jax.Array:
      blend_extent = min(a.shape[3], b.shape[3], blend_extent)
      for x in range(blend_extent):
          val = a[:, :, :, -blend_extent + x, :] * (1 - x / blend_extent) + b[:, :, :, x, :] * (
              x / blend_extent
          )
          b = b.at[:, :, :, x, :].set(val)
      return b

  def blend_t(self, a: jax.Array, b: jax.Array, blend_extent: int) -> jax.Array:
      blend_extent = min(a.shape[1], b.shape[1], blend_extent)
      for x in range(blend_extent):
          val = a[:, -blend_extent + x, :, :, :] * (1 - x / blend_extent) + b[:, x, :, :, :] * (
              x / blend_extent
          )
          b = b.at[:, x, :, :, :].set(val)
      return b

  def tiled_encode(self, x: jax.Array, key: Optional[jax.Array] = None, causal: Optional[bool] = None) -> jax.Array:
      B, T, H, W, C = x.shape
      latent_height = H // self.spatial_compression_ratio
      latent_width = W // self.spatial_compression_ratio
      
      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
      tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
      
      blend_height = tile_latent_min_height - tile_latent_stride_height
      blend_width = tile_latent_min_width - tile_latent_stride_width
      
      rows = []
      keys_i = None
      if key is not None:
          # Estimate number of tiles to split key
          num_h_tiles = (H + self.tile_sample_stride_height - 1) // self.tile_sample_stride_height
          num_w_tiles = (W + self.tile_sample_stride_width - 1) // self.tile_sample_stride_width
          keys_i = jax.random.split(key, num_h_tiles)

      row_idx = 0
      for i in range(0, H, self.tile_sample_stride_height):
          row = []
          key_i = keys_i[row_idx] if keys_i is not None else None
          keys_j = None
          if key_i is not None:
             num_w_tiles = (W + self.tile_sample_stride_width - 1) // self.tile_sample_stride_width
             keys_j = jax.random.split(key_i, num_w_tiles)

          col_idx = 0
          for j in range(0, W, self.tile_sample_stride_width):
              tile = x[:, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width, :]
              subkey = keys_j[col_idx] if keys_j is not None else None
              latent_tile = self.encoder(tile, key=subkey, causal=causal)
              row.append(latent_tile)
              col_idx += 1
          rows.append(row)
          row_idx += 1
          
      result_rows = []
      for i, row in enumerate(rows):
          result_row = []
          for j, tile in enumerate(row):
              if i > 0:
                  tile = self.blend_v(rows[i - 1][j], tile, blend_height)
              if j > 0:
                  tile = self.blend_h(row[j - 1], tile, blend_width)
              
              result_row.append(tile[:, :, :tile_latent_stride_height, :tile_latent_stride_width, :])
          
          result_rows.append(jnp.concatenate(result_row, axis=3))
      
      enc = jnp.concatenate(result_rows, axis=2)
      enc = enc[:, :, :latent_height, :latent_width, :]
      return enc

  def tiled_decode(
      self, z: jax.Array, temb: Optional[jax.Array] = None, key: Optional[jax.Array] = None, causal: Optional[bool] = None, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
      B, T, H, W, C = z.shape
      sample_height = H * self.spatial_compression_ratio
      sample_width = W * self.spatial_compression_ratio
      
      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
      tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
      
      blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
      blend_width = self.tile_sample_min_width - self.tile_sample_stride_width
      
      rows = []
      keys_i = None
      if key is not None:
           num_h_tiles = (H + tile_latent_stride_height - 1) // tile_latent_stride_height
           keys_i = jax.random.split(key, num_h_tiles)

      row_idx = 0
      for i in range(0, H, tile_latent_stride_height):
          row = []
          key_i = keys_i[row_idx] if keys_i is not None else None
          keys_j = None
          if key_i is not None:
             num_w_tiles = (W + tile_latent_stride_width - 1) // tile_latent_stride_width
             keys_j = jax.random.split(key_i, num_w_tiles)

          col_idx = 0
          for j in range(0, W, tile_latent_stride_width):
              tile = z[:, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width, :]
              subkey = keys_j[col_idx] if keys_j is not None else None
              decoded_tile = self.decoder(tile, temb=temb, key=subkey, causal=causal)
              row.append(decoded_tile)
              col_idx += 1
          rows.append(row)
          row_idx += 1
          
      result_rows = []
      for i, row in enumerate(rows):
          result_row = []
          for j, tile in enumerate(row):
              if i > 0:
                  tile = self.blend_v(rows[i - 1][j], tile, blend_height)
              if j > 0:
                  tile = self.blend_h(row[j - 1], tile, blend_width)
              
              result_row.append(tile[:, :, :self.tile_sample_stride_height, :self.tile_sample_stride_width, :])
          result_rows.append(jnp.concatenate(result_row, axis=3))
          
      dec = jnp.concatenate(result_rows, axis=2)
      dec = dec[:, :, :sample_height, :sample_width, :]
      
      if not return_dict:
          return (dec,)
      return FlaxDecoderOutput(sample=dec)

  def _temporal_tiled_encode(self, x: jax.Array, key: Optional[jax.Array] = None, causal: Optional[bool] = None) -> jax.Array:
      B, T, H, W, C = x.shape
      latent_num_frames = (T - 1) // self.temporal_compression_ratio + 1
      
      tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
      tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
      blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames
      
      keys_i = None
      if key is not None:
          num_blocks = (T + self.tile_sample_stride_num_frames - 1) // self.tile_sample_stride_num_frames
          keys_i = jax.random.split(key, num_blocks)
      
      row = []
      row_idx = 0
      for i in range(0, T, self.tile_sample_stride_num_frames):
          tile = x[:, i : i + self.tile_sample_min_num_frames + 1, :, :, :]
          subkey = keys_i[row_idx] if keys_i is not None else None
          if self.use_tiling and (H > self.tile_sample_min_height or W > self.tile_sample_min_width):
              tile = self.tiled_encode(tile, key=subkey, causal=causal)
          else:
              tile = self.encoder(tile, key=subkey, causal=causal)
          
          if i > 0:
              tile = tile[:, 1:, :, :, :]
          row.append(tile)
          row_idx += 1
          
      result_row = []
      for i, tile in enumerate(row):
          if i > 0:
              tile = self.blend_t(row[i - 1], tile, blend_num_frames)
              result_row.append(tile[:, :tile_latent_stride_num_frames, :, :, :])
          else:
              result_row.append(tile[:, :tile_latent_stride_num_frames + 1, :, :, :])
              
      enc = jnp.concatenate(result_row, axis=1)
      enc = enc[:, :latent_num_frames, :, :, :]
      return enc

  def _temporal_tiled_decode(
      self, z: jax.Array, temb: Optional[jax.Array] = None, key: Optional[jax.Array] = None, causal: Optional[bool] = None, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
      B, T, H, W, C = z.shape
      num_sample_frames = (T - 1) * self.temporal_compression_ratio + 1
      
      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
      tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
      blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
      
      keys_i = None
      if key is not None:
          num_blocks = (T + tile_latent_stride_num_frames - 1) // tile_latent_stride_num_frames
          keys_i = jax.random.split(key, num_blocks)
      
      row = []
      row_idx = 0
      for i in range(0, T, tile_latent_stride_num_frames):
          tile = z[:, i : i + tile_latent_min_num_frames + 1, :, :, :]
          subkey = keys_i[row_idx] if keys_i is not None else None
          if self.use_tiling and (tile.shape[2] > tile_latent_min_height or tile.shape[3] > tile_latent_min_width):
              decoded = self.tiled_decode(tile, temb, key=subkey, causal=causal, return_dict=True).sample
          else:
              decoded = self.decoder(tile, temb=temb, key=subkey, causal=causal)
          
          if i > 0:
              decoded = decoded[:, :-1, :, :, :]
          row.append(decoded)
          row_idx += 1
          
      result_row = []
      for i, tile in enumerate(row):
          if i > 0:
              tile = self.blend_t(row[i - 1], tile, blend_num_frames)
              tile = tile[:, :self.tile_sample_stride_num_frames, :, :, :]
              result_row.append(tile)
          else:
              result_row.append(tile[:, :self.tile_sample_stride_num_frames + 1, :, :, :])
              
      dec = jnp.concatenate(result_row, axis=1)
      dec = dec[:, :num_sample_frames, :, :, :]
      
      if not return_dict:
          return (dec,)
      return FlaxDecoderOutput(sample=dec)

  def _encode(self, x: jax.Array, key: Optional[jax.Array] = None, causal: Optional[bool] = None) -> jax.Array:
      B, T, H, W, C = x.shape
      if self.use_framewise_decoding and T > self.tile_sample_min_num_frames:
          return self._temporal_tiled_encode(x, key=key, causal=causal)
      
      if self.use_tiling and (W > self.tile_sample_min_width or H > self.tile_sample_min_height):
          return self.tiled_encode(x, key=key, causal=causal)
      
      enc = self.encoder(x, key=key, causal=causal)
      return enc

  def _decode(
      self,
      z: jax.Array,
      temb: Optional[jax.Array] = None,
      key: Optional[jax.Array] = None,
      causal: Optional[bool] = None,
      return_dict: bool = True,
  ) -> Union[FlaxDecoderOutput, jax.Array]:
      B, T, H, W, C = z.shape
      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
      
      if self.use_framewise_decoding and T > tile_latent_min_num_frames:
          return self._temporal_tiled_decode(z, temb, key=key, causal=causal, return_dict=return_dict)
      
      if self.use_tiling and (W > tile_latent_min_width or H > tile_latent_min_height):
          return self.tiled_decode(z, temb, key=key, causal=causal, return_dict=return_dict)
      
      dec = self.decoder(z, temb, key=key, causal=causal)
      
      if not return_dict:
          return (dec,)
      return FlaxDecoderOutput(sample=dec)

  def encode(
      self,
      sample: jax.Array,
      temb: Optional[jax.Array] = None,
      return_dict: bool = True,
      key: Optional[jax.Array] = None,
      causal: Optional[bool] = None,
  ) -> Union[FlaxAutoencoderKLOutput, Tuple[jax.Array]]:
    
    causal = self.encoder_causal if causal is None else causal
    


    if self.use_slicing and sample.shape[0] > 1:
        if key is not None:
             keys_slice = jax.random.split(key, sample.shape[0])

        encoded_slices = [] 
        for i in range(sample.shape[0]):
             subkey = keys_slice[i] if keys_slice is not None else None
             encoded_slices.append(self._encode(sample[i:i+1], key=subkey, causal=causal))
        moments = jnp.concatenate(encoded_slices, axis=0)
    else:
        moments = self._encode(sample, key=key, causal=causal)
        
    posterior = LTX2DiagonalGaussianDistribution(moments, cls_latent_channels=self.latent_channels)

    if not return_dict:
      return (posterior,)
    return FlaxAutoencoderKLOutput(latent_dist=posterior)

  def decode(
      self,
      latents: jax.Array,
      temb: Optional[jax.Array] = None,
      return_dict: bool = True,
      generator: Optional[jax.Array] = None, # generator acts as key
      causal: Optional[bool] = None,
  ) -> Union[FlaxDecoderOutput, Tuple[jax.Array]]:
    
    causal = self.decoder_causal if causal is None else causal
    key = generator
    
    keys_slice = None
    if self.use_slicing and latents.shape[0] > 1:
        if key is not None:
             keys_slice = jax.random.split(key, latents.shape[0])
        decoded_slices = []
        for i in range(latents.shape[0]):
            z_slice = latents[i:i+1]
            t_slice = temb[i:i+1] if temb is not None else None
            subkey = keys_slice[i] if keys_slice is not None else None
            res = self._decode(z_slice, t_slice, key=subkey, causal=causal, return_dict=True)
            decoded_slices.append(res.sample)
        
        dec = jnp.concatenate(decoded_slices, axis=0)
    else:
        dec = self._decode(latents, temb, key=key, causal=causal, return_dict=True).sample

    if not return_dict:
      return (dec,)
    return FlaxDecoderOutput(sample=dec)

  def __call__(
      self,
      sample: jax.Array,
      temb: Optional[jax.Array] = None,
      sample_posterior: bool = False,
      return_dict: bool = True,
      generator: Optional[jax.Array] = None,
      encoder_causal: Optional[bool] = None,
      decoder_causal: Optional[bool] = None,
  ) -> Union[FlaxDecoderOutput, Tuple[jax.Array]]:
    
    encoder_causal = self.encoder_causal if encoder_causal is None else encoder_causal
    decoder_causal = self.decoder_causal if decoder_causal is None else decoder_causal

    key = generator 
    key_encode, key_sample, key_decode = None, None, None
    if key is not None:
         key_encode, key_sample, key_decode = jax.random.split(key, 3)

    posterior = self.encode(sample, temb=temb, return_dict=True, key=key_encode, causal=encoder_causal).latent_dist
    
    if sample_posterior:
      z = posterior.sample(key=key_sample)
    else:
      z = posterior.mode()
    
    dec = self.decode(z, temb=temb, return_dict=True, generator=key_decode, causal=decoder_causal)

    if not return_dict:
      return (dec.sample,)
    return dec
