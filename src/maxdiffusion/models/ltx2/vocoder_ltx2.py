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
from typing import Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from ... import common_types
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.modeling_flax_utils import FlaxModelMixin

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
      act_fn: str = "leaky_relu",
      leaky_relu_negative_slope: float = 0.1,
      antialias: bool = False,
      antialias_ratio: int = 2,
      antialias_kernel_size: int = 12,
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
    self.acts1 = nnx.List()
    for _ in range(len(self.convs1)):
      if act_fn == "snakebeta":
        act = SnakeBeta(channels, use_beta=True, rngs=rngs)
      elif act_fn == "snake":
        act = SnakeBeta(channels, use_beta=False, rngs=rngs)
      else:
        act = lambda x: jax.nn.leaky_relu(x, negative_slope=leaky_relu_negative_slope)
        
      if antialias:
        act = AntiAliasAct1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
      self.acts1.append(act)

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
    self.acts2 = nnx.List()
    for _ in range(len(self.convs2)):
      if act_fn == "snakebeta":
        act = SnakeBeta(channels, use_beta=True, rngs=rngs)
      elif act_fn == "snake":
        act = SnakeBeta(channels, use_beta=False, rngs=rngs)
      else:
        act = lambda x: jax.nn.leaky_relu(x, negative_slope=leaky_relu_negative_slope)
        
      if antialias:
        act = AntiAliasAct1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
      self.acts2.append(act)

  def __call__(self, x: Array) -> Array:
    for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
      xt = act1(x)
      xt = conv1(xt)
      xt = act2(xt)
      xt = conv2(xt)
      x = x + xt
    return x


class LTX2Vocoder(nnx.Module, FlaxModelMixin, ConfigMixin):
  """
  LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
  """

  @register_to_config
  def __init__(
      self,
      in_channels: int = 128,
      hidden_channels: int = 1024,
      out_channels: int = 2,
      upsample_kernel_sizes: Sequence[int] = (16, 15, 8, 4, 4),
      upsample_factors: Sequence[int] = (6, 5, 2, 2, 2),
      resnet_kernel_sizes: Sequence[int] = (3, 7, 11),
      resnet_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
      act_fn: str = "leaky_relu",
      leaky_relu_negative_slope: float = 0.1,
      antialias: bool = False,
      antialias_ratio: int = 2,
      antialias_kernel_size: int = 12,
      final_act_fn: Optional[str] = None,
      final_bias: bool = False,
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
    self.act_fn = act_fn
    self.final_act_fn = final_act_fn
    self.final_bias = final_bias
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
                act_fn=act_fn,
                leaky_relu_negative_slope=leaky_relu_negative_slope,
                antialias=antialias,
                antialias_ratio=antialias_ratio,
                antialias_kernel_size=antialias_kernel_size,
                rngs=rngs,
                dtype=self.dtype,
            )
        )
      input_channels = output_channels

    if act_fn == "snakebeta" or act_fn == "snake":
      # Always use antialiasing
      act_out = SnakeBeta(channels=output_channels, use_beta=True, rngs=rngs)
      self.act_out = AntiAliasAct1d(act_out, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
    elif act_fn == "leaky_relu":
      # NOTE: does NOT use self.negative_slope, following the original code
      self.act_out = lambda x: jax.nn.leaky_relu(x)

    self.conv_out = nnx.Conv(
        in_features=input_channels,
        out_features=out_channels,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        use_bias=self.final_bias,
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
      if self.act_fn == "leaky_relu":
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
    hidden_states = self.act_out(hidden_states)
    hidden_states = self.conv_out(hidden_states)
    if self.final_act_fn == "tanh":
      hidden_states = jnp.tanh(hidden_states)
    elif self.final_act_fn == "clamp":
      hidden_states = jnp.clip(hidden_states, -1, 1)

    # Transpose back to (Batch, Channels, Time) to match PyTorch/Diffusers output format
    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    return hidden_states


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Array:
  """Creates a Kaiser sinc kernel for low-pass filtering in JAX."""
  delta_f = 4 * half_width
  half_size = kernel_size // 2
  amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
  
  if amplitude > 50.0:
    beta = 0.1102 * (amplitude - 8.7)
  elif amplitude >= 21.0:
    beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
  else:
    beta = 0.0

  # JAX equivalent of torch.kaiser_window
  window = jnp.kaiser(kernel_size, beta)

  even = kernel_size % 2 == 0
  time = jnp.arange(-half_size, half_size) + 0.5 if even else jnp.arange(kernel_size) - half_size

  if cutoff == 0.0:
    filter_out = jnp.zeros_like(time)
  else:
    time = 2 * cutoff * time
    sinc = jnp.where(
        time == 0,
        jnp.ones_like(time),
        jnp.sin(math.pi * time) / (math.pi * time),
    )
    filter_out = 2 * cutoff * window * sinc
    filter_out = filter_out / filter_out.sum()
    
  return filter_out


class DownSample1d(nnx.Module):
  """1D low-pass filter for antialias downsampling in JAX."""

  def __init__(
      self,
      ratio: int = 2,
      kernel_size: Optional[int] = None,
      use_padding: bool = True,
      padding_mode: str = "replicate",
  ):
    self.ratio = ratio
    self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
    self.pad_left = self.kernel_size // 2 + (self.kernel_size % 2) - 1
    self.pad_right = self.kernel_size // 2
    self.use_padding = use_padding
    self.padding_mode = padding_mode

    cutoff = 0.5 / ratio
    half_width = 0.6 / ratio
    self.filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)

  def __call__(self, x: Array) -> Array:
    # x expected shape: [batch_size, num_channels, hidden_dim]
    # JAX Conv1d expects [batch, spatial, in_channels]
    x = jnp.transpose(x, (0, 2, 1))
    num_channels = x.shape[-1]
    
    if self.use_padding:
      mode = "constant" if self.padding_mode == "constant" else "edge"
      x = jnp.pad(x, ((0, 0), (self.pad_left, self.pad_right), (0, 0)), mode=mode)

    filter_expanded = jnp.expand_dims(self.filter, axis=(1, 2))
    filter_expanded = jnp.tile(filter_expanded, (1, 1, num_channels))
    
    x_filtered = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=filter_expanded,
        window_strides=(self.ratio,),
        padding="VALID",
        feature_group_count=num_channels,
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    
    return jnp.transpose(x_filtered, (0, 2, 1))


class UpSample1d(nnx.Module):
  def __init__(
      self,
      ratio: int = 2,
      kernel_size: Optional[int] = None,
      window_type: str = "kaiser",
      padding_mode: str = "replicate",
  ):
    self.ratio = ratio
    self.padding_mode = padding_mode

    if window_type == "hann":
      rolloff = 0.99
      lowpass_filter_width = 6
      width = math.ceil(lowpass_filter_width / rolloff)
      self.kernel_size = 2 * width * ratio + 1
      self.pad = width
      self.pad_left = 2 * width * ratio
      self.pad_right = self.kernel_size - ratio

      time_axis = (jnp.arange(self.kernel_size) / ratio - width) * rolloff
      time_clamped = jnp.clip(time_axis, -lowpass_filter_width, lowpass_filter_width)
      window = jnp.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
      sinc_filter = (jnp.sinc(time_axis) * window * rolloff / ratio).reshape(1, 1, -1)
    else:
      self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
      self.pad = self.kernel_size // ratio - 1
      self.pad_left = self.pad * self.ratio + (self.kernel_size - self.ratio) // 2
      self.pad_right = self.pad * self.ratio + (self.kernel_size - self.ratio + 1) // 2

      sinc_filter = kaiser_sinc_filter1d(
          cutoff=0.5 / ratio,
          half_width=0.6 / ratio,
          kernel_size=self.kernel_size,
      )

    self.filter = sinc_filter

  def __call__(self, x: Array) -> Array:
    x = jnp.transpose(x, (0, 2, 1))
    num_channels = x.shape[-1]
    
    mode = "constant" if self.padding_mode == "constant" else "edge"
    x = jnp.pad(x, ((0, 0), (self.pad, self.pad), (0, 0)), mode=mode)
    
    filter_expanded = jnp.expand_dims(self.filter, axis=(1, 2))
    filter_expanded = jnp.tile(filter_expanded, (1, 1, num_channels))
    
    x_upsampled = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=filter_expanded,
        window_strides=(1,),
        padding="VALID",
        lhs_dilation=(self.ratio,),
        feature_group_count=num_channels,
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    
    out = x_upsampled[:, self.pad_left : -self.pad_right, :]
    return jnp.transpose(out, (0, 2, 1))


class AntiAliasAct1d(nnx.Module):
  def __init__(
      self,
      act_fn: nnx.Module,
      ratio: int = 2,
      kernel_size: int = 12,
  ):
    self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
    self.act = act_fn
    self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

  def __call__(self, x: Array) -> Array:
    x = self.upsample(x)
    x = self.act(x)
    x = self.downsample(x)
    return x


class SnakeBeta(nnx.Module):
  def __init__(
      self,
      channels: int,
      alpha: float = 1.0,
      eps: float = 1e-9,
      trainable_params: bool = True,
      logscale: bool = True,
      use_beta: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.eps = eps
    self.logscale = logscale
    self.use_beta = use_beta

    init_val = jnp.zeros((channels,)) if self.logscale else jnp.ones((channels,)) * alpha
    
    if trainable_params:
      self.alpha = nnx.Param(init_val)
      if use_beta:
        self.beta = nnx.Param(init_val)
    else:
      self.alpha = nnx.data(init_val)
      if use_beta:
        self.beta = nnx.data(init_val)

  def __call__(self, hidden_states: Array, channel_dim: int = 1) -> Array:
    broadcast_shape = [1] * hidden_states.ndim
    broadcast_shape[channel_dim] = -1
    
    alpha = self.alpha.value.reshape(broadcast_shape)
    if self.use_beta:
      beta = self.beta.value.reshape(broadcast_shape)

    if self.logscale:
      alpha = jnp.exp(alpha)
      if self.use_beta:
        beta = jnp.exp(beta)

    amplitude = beta if self.use_beta else alpha
    hidden_states = hidden_states + (1.0 / (amplitude + self.eps)) * jnp.sin(hidden_states * alpha) ** 2
    return hidden_states


class CausalSTFT(nnx.Module):
  def __init__(self, filter_length: int = 512, hop_length: int = 80, window_length: int = 512, *, rngs: nnx.Rngs):
    self.hop_length = hop_length
    self.window_length = window_length
    n_freqs = filter_length // 2 + 1

    self.forward_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2)))
    self.inverse_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2)))

  def __call__(self, waveform: Array) -> Tuple[Array, Array]:
    if waveform.ndim == 2:
      waveform = jnp.expand_dims(waveform, 1)
      
    left_pad = max(0, self.window_length - self.hop_length)
    waveform = jnp.pad(waveform, ((0, 0), (0, 0), (left_pad, 0)), mode="constant")
    
    waveform = jnp.transpose(waveform, (0, 2, 1))
    
    spec = jax.lax.conv_general_dilated(
        lhs=waveform,
        rhs=self.forward_basis.value,
        window_strides=(self.hop_length,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    
    spec = jnp.transpose(spec, (0, 2, 1))
    n_freqs = spec.shape[1] // 2
    real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
    magnitude = jnp.sqrt(real**2 + imag**2)
    phase = jnp.arctan2(imag, real)
    return magnitude, phase


class MelSTFT(nnx.Module):
  def __init__(
      self,
      filter_length: int = 512,
      hop_length: int = 80,
      window_length: int = 512,
      num_mel_channels: int = 64,
      *,
      rngs: nnx.Rngs,
  ):
    self.stft_fn = CausalSTFT(filter_length, hop_length, window_length, rngs=rngs)
    num_freqs = filter_length // 2 + 1
    self.mel_basis = nnx.Param(jnp.zeros((num_mel_channels, num_freqs)))

  def __call__(self, waveform: Array) -> Tuple[Array, Array, Array, Array]:
    magnitude, phase = self.stft_fn(waveform)
    energy = jnp.linalg.norm(magnitude, axis=1)
    mel = jnp.matmul(self.mel_basis.value, magnitude)
    log_mel = jnp.log(jnp.clip(mel, min=1e-5))
    return log_mel, magnitude, phase, energy


class LTX2VocoderWithBWE(nnx.Module, FlaxModelMixin, ConfigMixin):
  """
  LTX-2.X vocoder with bandwidth extension (BWE) upsampling.
  """

  @register_to_config
  def __init__(
      self,
      in_channels: int = 128,
      hidden_channels: int = 1536,
      out_channels: int = 2,
      upsample_kernel_sizes: list[int] = [11, 4, 4, 4, 4, 4],
      upsample_factors: list[int] = [5, 2, 2, 2, 2, 2],
      resnet_kernel_sizes: list[int] = [3, 7, 11],
      resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      act_fn: str = "snakebeta",
      leaky_relu_negative_slope: float = 0.1,
      antialias: bool = True,
      antialias_ratio: int = 2,
      antialias_kernel_size: int = 12,
      final_act_fn: str | None = None,
      final_bias: bool = False,
      bwe_in_channels: int = 128,
      bwe_hidden_channels: int = 512,
      bwe_out_channels: int = 2,
      bwe_upsample_kernel_sizes: list[int] = [12, 11, 4, 4, 4],
      bwe_upsample_factors: list[int] = [6, 5, 2, 2, 2],
      bwe_resnet_kernel_sizes: list[int] = [3, 7, 11],
      bwe_resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      bwe_act_fn: str = "snakebeta",
      bwe_leaky_relu_negative_slope: float = 0.1,
      bwe_antialias: bool = True,
      bwe_antialias_ratio: int = 2,
      bwe_antialias_kernel_size: int = 12,
      bwe_final_act_fn: str | None = None,
      bwe_final_bias: bool = False,
      filter_length: int = 512,
      hop_length: int = 80,
      window_length: int = 512,
      num_mel_channels: int = 64,
      input_sampling_rate: int = 16000,
      output_sampling_rate: int = 48000,
      *,
      rngs: nnx.Rngs,
      dtype: DType = jnp.float32,
  ):
    self.vocoder = LTX2Vocoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        upsample_kernel_sizes=upsample_kernel_sizes,
        upsample_factors=upsample_factors,
        resnet_kernel_sizes=resnet_kernel_sizes,
        resnet_dilations=resnet_dilations,
        act_fn=act_fn,
        leaky_relu_negative_slope=leaky_relu_negative_slope,
        antialias=antialias,
        antialias_ratio=antialias_ratio,
        antialias_kernel_size=antialias_kernel_size,
        final_act_fn=final_act_fn,
        final_bias=final_bias,
        output_sampling_rate=input_sampling_rate,
        rngs=rngs,
        dtype=dtype,
    )
    
    self.bwe_generator = LTX2Vocoder(
        in_channels=bwe_in_channels,
        hidden_channels=bwe_hidden_channels,
        out_channels=bwe_out_channels,
        upsample_kernel_sizes=bwe_upsample_kernel_sizes,
        upsample_factors=bwe_upsample_factors,
        resnet_kernel_sizes=bwe_resnet_kernel_sizes,
        resnet_dilations=bwe_resnet_dilations,
        act_fn=bwe_act_fn,
        leaky_relu_negative_slope=bwe_leaky_relu_negative_slope,
        antialias=bwe_antialias,
        antialias_ratio=bwe_antialias_ratio,
        antialias_kernel_size=bwe_antialias_kernel_size,
        final_act_fn=bwe_final_act_fn,
        final_bias=bwe_final_bias,
        output_sampling_rate=output_sampling_rate,
        rngs=rngs,
        dtype=dtype,
    )

    self.mel_stft = MelSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        window_length=window_length,
        num_mel_channels=num_mel_channels,
        rngs=rngs,
    )

    self.resampler = UpSample1d(
        ratio=output_sampling_rate // input_sampling_rate,
        window_type="hann",
    )

  def __call__(self, mel_spec: Array) -> Array:
    x = self.vocoder(mel_spec)
    
    batch_size, num_channels, num_samples = x.shape

    hop_length = getattr(self.config, "hop_length", 80)
    remainder = num_samples % hop_length
    if remainder != 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, hop_length - remainder)), mode="constant")

    x_flat = x.reshape(-1, x.shape[-1])
    mel, _, _, _ = self.mel_stft(x_flat)
    mel = mel.reshape(batch_size, num_channels, -1)

    mel_for_bwe = mel.transpose(0, 1, 3, 2)
    
    
    residual = self.bwe_generator(mel_for_bwe)
    

    skip = self.resampler(x)
    waveform = jnp.clip(residual + skip, -1, 1)
    
    input_sampling_rate = getattr(self.config, "input_sampling_rate", 16000)
    output_sampling_rate = getattr(self.config, "output_sampling_rate", 48000)
    output_samples = num_samples * output_sampling_rate // input_sampling_rate
    
    waveform = waveform[..., :output_samples]
    return waveform
