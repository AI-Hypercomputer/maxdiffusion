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

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from ... import common_types

Array = common_types.Array
DType = common_types.DType


class ResBlock(nnx.Module):
  """
  Residual Block for the LTX-2 Vocoder.
  """

  def __init__(
      self,
      channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      dilations: Sequence[int] = (1, 3, 5),
      leaky_relu_negative_slope: float = 0.1,
      *,
      rngs: nnx.Rngs,
      dtype: DType = jnp.float32,
  ):
    self.dilations = dilations
    self.negative_slope = leaky_relu_negative_slope

    self.convs1 = nnx.List(
        [
            nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(kernel_size,),
                strides=(stride,),
                kernel_dilation=(dilation,),
                padding="SAME",
                rngs=rngs,
                dtype=dtype,
            )
            for dilation in dilations
        ]
    )

    self.convs2 = nnx.List(
        [
            nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(kernel_size,),
                strides=(stride,),
                kernel_dilation=(1,),
                padding="SAME",
                rngs=rngs,
                dtype=dtype,
            )
            for _ in range(len(dilations))
        ]
    )

  def __call__(self, x: Array) -> Array:
    for conv1, conv2 in zip(self.convs1, self.convs2):
      xt = jax.nn.leaky_relu(x, negative_slope=self.negative_slope)
      xt = conv1(xt)
      xt = jax.nn.leaky_relu(xt, negative_slope=self.negative_slope)
      xt = conv2(xt)
      x = x + xt
    return x


class LTX2Vocoder(nnx.Module):
  """
  LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
  """

  def __init__(
      self,
      in_channels: int = 128,
      hidden_channels: int = 1024,
      out_channels: int = 2,
      upsample_kernel_sizes: Sequence[int] = (16, 15, 8, 4, 4),
      upsample_factors: Sequence[int] = (6, 5, 2, 2, 2),
      resnet_kernel_sizes: Sequence[int] = (3, 7, 11),
      resnet_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
      leaky_relu_negative_slope: float = 0.1,
      # output_sampling_rate is unused in model structure but kept for config compat
      output_sampling_rate: int = 24000,
      *,
      rngs: nnx.Rngs,
      dtype: DType = jnp.float32,
  ):
    self.num_upsample_layers = len(upsample_kernel_sizes)
    self.resnets_per_upsample = len(resnet_kernel_sizes)
    self.out_channels = out_channels
    self.total_upsample_factor = math.prod(upsample_factors)
    self.negative_slope = leaky_relu_negative_slope
    self.dtype = dtype

    if self.num_upsample_layers != len(upsample_factors):
      raise ValueError(
          f"`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
          f" {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
      )

    if self.resnets_per_upsample != len(resnet_dilations):
      raise ValueError(
          f"`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
          f" {self.resnets_per_upsample} and {len(resnet_dilations)}, respectively."
      )

    # PyTorch Conv1d expects (Batch, Channels, Length), we use (Batch, Length, Channels)
    # So in_channels/out_channels args are standard, but data layout is transposed in __call__
    self.conv_in = nnx.Conv(
        in_features=in_channels,
        out_features=hidden_channels,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        rngs=rngs,
        dtype=self.dtype,
    )

    self.upsamplers = nnx.List()
    self.resnets = nnx.List()
    input_channels = hidden_channels

    for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
      output_channels = input_channels // 2

      # ConvTranspose with padding='SAME' matches PyTorch's specific padding logic
      # for these standard HiFi-GAN upsampling configurations.
      self.upsamplers.append(
          nnx.ConvTranspose(
              in_features=input_channels,
              out_features=output_channels,
              kernel_size=(kernel_size,),
              strides=(stride,),
              padding="SAME",
              rngs=rngs,
              dtype=self.dtype,
          )
      )

      for res_kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
        self.resnets.append(
            ResBlock(
                channels=output_channels,
                kernel_size=res_kernel_size,
                dilations=dilations,
                leaky_relu_negative_slope=leaky_relu_negative_slope,
                rngs=rngs,
                dtype=self.dtype,
            )
        )
      input_channels = output_channels

    self.conv_out = nnx.Conv(
        in_features=input_channels,
        out_features=out_channels,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        rngs=rngs,
        dtype=self.dtype,
    )

  def __call__(self, hidden_states: Array, time_last: bool = False) -> Array:
    """
    Forward pass of the vocoder.

    Args:
        hidden_states: Input Mel spectrogram tensor.
            Shape: `(B, C, T, F)` or `(B, C, F, T)`
        time_last: Legacy flag for input layout.

    Returns:
        Audio waveform: `(B, OutChannels, AudioLength)`
    """
    # Ensure layout: (Batch, Channels, MelBins, Time)
    if not time_last:
      hidden_states = jnp.transpose(hidden_states, (0, 1, 3, 2))

    # Flatten Channels and MelBins -> (Batch, Features, Time)
    batch, channels, mel_bins, time = hidden_states.shape
    hidden_states = hidden_states.reshape(batch, channels * mel_bins, time)

    # Transpose to (Batch, Time, Features) for Flax NWC Convolutions
    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    hidden_states = self.conv_in(hidden_states)

    for i in range(self.num_upsample_layers):
      hidden_states = jax.nn.leaky_relu(hidden_states, negative_slope=self.negative_slope)
      hidden_states = self.upsamplers[i](hidden_states)

      # Accumulate ResNet outputs (Memory Optimization)
      start = i * self.resnets_per_upsample
      end = (i + 1) * self.resnets_per_upsample

      res_sum = 0.0
      for j in range(start, end):
        res_sum = res_sum + self.resnets[j](hidden_states)

      # Average the outputs (matches PyTorch mean(stack))
      hidden_states = res_sum / self.resnets_per_upsample

    # Final Post-Processing
    # Note: using 0.01 slope here specifically (matches Diffusers implementation quirk)
    hidden_states = jax.nn.leaky_relu(hidden_states, negative_slope=0.01)
    hidden_states = self.conv_out(hidden_states)
    hidden_states = jnp.tanh(hidden_states)

    # Transpose back to (Batch, Channels, Time) to match PyTorch/Diffusers output format
    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    return hidden_states
