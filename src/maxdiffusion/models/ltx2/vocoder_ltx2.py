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
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from flax import nnx
from ... import common_types
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.modeling_flax_utils import FlaxModelMixin

Array = common_types.Array
DType = common_types.DType

def kaiser_window(n: int, beta: float) -> Array:
  """Computes the Kaiser window."""
  alpha = (n - 1) / 2.0
  time = jnp.arange(n)
  term = beta * jnp.sqrt(1 - ((time - alpha) / alpha) ** 2)
  return jss.i0(term) / jss.i0(beta)

def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Array:
  """Creates a Kaiser sinc kernel for low-pass filtering."""
  delta_f = 4 * half_width
  half_size = kernel_size // 2
  amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
  if amplitude > 50.0:
    beta = 0.1102 * (amplitude - 8.7)
  elif amplitude >= 21.0:
    beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
  else:
    beta = 0.0

  window = kaiser_window(kernel_size, beta)

  even = kernel_size % 2 == 0
  time = jnp.arange(-half_size, half_size) + 0.5 if even else jnp.arange(kernel_size) - half_size

  if cutoff == 0.0:
    filter = jnp.zeros_like(time)
  else:
    time = 2 * cutoff * time
    sinc = jnp.where(
        time == 0,
        jnp.ones_like(time),
        jnp.sin(math.pi * time) / math.pi / time,
    )
    filter = 2 * cutoff * window * sinc
    filter = filter / filter.sum()
  return filter


class DownSample1d(nnx.Module):
  """1D low-pass filter for antialias downsampling."""

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
    low_pass_filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)
    self.filter = jnp.expand_dims(low_pass_filter, axis=(1, 2))

  def __call__(self, x: Array) -> Array:
    num_channels = x.shape[-1]
    if self.use_padding:
      x = jnp.pad(x, ((0, 0), (self.pad_left, self.pad_right), (0, 0)), mode='edge')
      
    filter_expanded = jnp.repeat(self.filter, num_channels, axis=2)
    filter_expanded = filter_expanded.astype(x.dtype)
    
    x_filtered = jax.lax.conv_general_dilated(
        x,
        filter_expanded,
        window_strides=(self.ratio,),
        padding=((0, 0),),
        dimension_numbers=('NLC', 'LIO', 'NLC'),
        feature_group_count=num_channels,
    )
    return x_filtered


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
      sinc_filter = jnp.sinc(time_axis) * window * rolloff / ratio
      self.filter = sinc_filter.reshape(-1, 1, 1)
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
      self.filter = sinc_filter.reshape(-1, 1, 1)

  def __call__(self, x: Array) -> Array:
    num_channels = x.shape[-1]
    x = jnp.pad(x, ((0, 0), (self.pad, self.pad), (0, 0)), mode='edge')
    
    filter_expanded = jnp.repeat(self.filter, num_channels, axis=2)
    filter_expanded = filter_expanded.astype(x.dtype)
    
    x_upsampled = jax.lax.conv_general_dilated(
        x,
        filter_expanded,
        window_strides=(1,),
        padding=((0, 0),),
        lhs_dilation=(self.ratio,),
        dimension_numbers=('NLC', 'LIO', 'NLC'),
        feature_group_count=num_channels,
    )
    
    x_upsampled = x_upsampled * self.ratio
    return x_upsampled[:, self.pad_left : -self.pad_right, :]


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

    if logscale:
      self.alpha = nnx.Param(jnp.zeros((channels,)))
      if use_beta:
        self.beta = nnx.Param(jnp.zeros((channels,)))
    else:
      self.alpha = nnx.Param(jnp.ones((channels,)) * alpha)
      if use_beta:
        self.beta = nnx.Param(jnp.ones((channels,)) * alpha)

    self.trainable_params = trainable_params

  def __call__(self, hidden_states: Array) -> Array:
    alpha = self.alpha.value
    if self.use_beta:
      beta = self.beta.value

    if self.logscale:
      alpha = jnp.exp(alpha)
      if self.use_beta:
        beta = jnp.exp(beta)

    amplitude = beta if self.use_beta else alpha
    alpha = jnp.expand_dims(alpha, axis=0)
    amplitude = jnp.expand_dims(amplitude, axis=0)
    
    hidden_states = hidden_states + (1.0 / (amplitude + self.eps)) * jnp.sin(hidden_states * alpha) ** 2
    return hidden_states


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
      
      if xt.shape[1] < x.shape[1]:
        xt = jnp.pad(xt, ((0, 0), (0, x.shape[1] - xt.shape[1]), (0, 0)), mode='edge')
      elif xt.shape[1] > x.shape[1]:
        xt = xt[:, :x.shape[1], :]
        
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
      final_act_fn: Optional[str] = "tanh",
      final_bias: bool = True,
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
    self.act_fn = act_fn
    self.final_act_fn = final_act_fn

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
      act_out = SnakeBeta(channels=output_channels, use_beta=True, rngs=rngs)
      self.act_out = AntiAliasAct1d(act_out, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
    elif act_fn == "leaky_relu":
      self.act_out = lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)

    self.conv_out = nnx.Conv(
        in_features=output_channels,
        out_features=out_channels,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        use_bias=final_bias,
        rngs=rngs,
        dtype=self.dtype,
    )

  def __call__(self, hidden_states: Array, time_last: bool = False) -> Array:
    if not time_last:
      hidden_states = jnp.transpose(hidden_states, (0, 1, 3, 2))

    batch, channels, mel_bins, time = hidden_states.shape
    hidden_states = hidden_states.reshape(batch, channels * mel_bins, time)
    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    hidden_states = self.conv_in(hidden_states)

    for i in range(self.num_upsample_layers):
      if self.act_fn == "leaky_relu":
        hidden_states = jax.nn.leaky_relu(hidden_states, negative_slope=self.negative_slope)
      hidden_states = self.upsamplers[i](hidden_states)

      start = i * self.resnets_per_upsample
      end = (i + 1) * self.resnets_per_upsample

      res_sum = 0.0
      for j in range(start, end):
        res_sum = res_sum + self.resnets[j](hidden_states)

      hidden_states = res_sum / self.resnets_per_upsample

    hidden_states = self.act_out(hidden_states)
    hidden_states = self.conv_out(hidden_states)
    
    if self.final_act_fn == "tanh":
      hidden_states = jnp.tanh(hidden_states)
    elif self.final_act_fn == "clamp":
      hidden_states = jnp.clip(hidden_states, -1, 1)

    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))
    return hidden_states


class CausalSTFT(nnx.Module):
  def __init__(self, filter_length: int = 512, hop_length: int = 80, window_length: int = 512, *, rngs: nnx.Rngs):
    self.hop_length = hop_length
    self.window_length = window_length
    n_freqs = filter_length // 2 + 1

    self.forward_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2)))
    self.inverse_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2)))

  def __call__(self, waveform: Array) -> tuple[Array, Array]:
    if waveform.ndim == 2:
      waveform = waveform[..., None]

    left_pad = max(0, self.window_length - self.hop_length)
    waveform = jnp.pad(waveform, ((0, 0), (left_pad, 0), (0, 0)))
    waveform = waveform.astype(self.forward_basis.value.dtype)

    spec = jax.lax.conv_general_dilated(
        waveform,
        self.forward_basis.value,
        window_strides=(self.hop_length,),
        padding=((0, 0),),
        dimension_numbers=('NLC', 'LIO', 'NLC'),
    )

    n_freqs = spec.shape[-1] // 2
    real = spec[..., :n_freqs]
    imag = spec[..., n_freqs:]

    magnitude = jnp.sqrt(real**2 + imag**2)
    phase = jnp.atan2(imag, real)

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

  def __call__(self, waveform: Array) -> tuple[Array, Array, Array, Array]:
    magnitude, phase = self.stft_fn(waveform)
    energy = jnp.linalg.norm(magnitude, axis=-1)
    mel = jnp.matmul(magnitude, self.mel_basis.value.T)
    log_mel = jnp.log(jnp.clip(mel, min=1e-5))
    return log_mel, magnitude, phase, energy


class LTX2VocoderWithBWE(nnx.Module, FlaxModelMixin, ConfigMixin):
  """LTX-2.X vocoder with bandwidth extension (BWE) upsampling."""

  @register_to_config
  def __init__(
      self,
      in_channels: int = 128,
      hidden_channels: int = 1536,
      out_channels: int = 2,
      upsample_kernel_sizes: Sequence[int] = (11, 4, 4, 4, 4, 4),
      upsample_factors: Sequence[int] = (5, 2, 2, 2, 2, 2),
      resnet_kernel_sizes: Sequence[int] = (3, 7, 11),
      resnet_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
      act_fn: str = "snakebeta",
      leaky_relu_negative_slope: float = 0.1,
      antialias: bool = True,
      antialias_ratio: int = 2,
      antialias_kernel_size: int = 12,
      final_act_fn: Optional[str] = None,
      final_bias: bool = False,
      bwe_in_channels: int = 128,
      bwe_hidden_channels: int = 512,
      bwe_out_channels: int = 2,
      bwe_upsample_kernel_sizes: Sequence[int] = (12, 11, 4, 4, 4),
      bwe_upsample_factors: Sequence[int] = (6, 5, 2, 2, 2),
      bwe_resnet_kernel_sizes: Sequence[int] = (3, 7, 11),
      bwe_resnet_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
      bwe_act_fn: str = "snakebeta",
      bwe_leaky_relu_negative_slope: float = 0.1,
      bwe_antialias: bool = True,
      bwe_antialias_ratio: int = 2,
      bwe_antialias_kernel_size: int = 12,
      bwe_final_act_fn: Optional[str] = None,
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
    self.hop_length = hop_length
    self.output_sampling_rate = output_sampling_rate
    self.input_sampling_rate = input_sampling_rate

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
    print(f"=== BWE Vocoder Debug ===")
    print(f"Input mel_spec - shape: {mel_spec.shape}, min: {mel_spec.min()}, max: {mel_spec.max()}")
    
    x = self.vocoder(mel_spec)
    print(f"Base vocoder output (x) - shape: {x.shape}, min: {x.min()}, max: {x.max()}")
    
    x = jnp.transpose(x, (0, 2, 1))
    batch_size, num_samples, num_channels = x.shape
    print(f"Transposed x - shape: {x.shape}")

    remainder = num_samples % self.hop_length
    if remainder != 0:
      x = jnp.pad(x, ((0, 0), (0, self.hop_length - remainder), (0, 0)))
      print(f"Padded x - shape: {x.shape}")

    x_flattened = x.transpose(0, 2, 1).reshape(-1, x.shape[1], 1)
    print(f"x_flattened - shape: {x_flattened.shape}")
    
    log_mel, _, _, _ = self.mel_stft(x_flattened)
    print(f"MelSTFT output (log_mel) before reshape - shape: {log_mel.shape}, min: {log_mel.min()}, max: {log_mel.max()}")
    
    log_mel = log_mel.reshape(batch_size, num_channels, -1, log_mel.shape[-1])
    print(f"Reshaped log_mel - shape: {log_mel.shape}")
    
    residual = self.bwe_generator(log_mel, time_last=False)
    print(f"BWE generator output (residual) - shape: {residual.shape}, min: {residual.min()}, max: {residual.max()}")
    
    skip = self.resampler(x)
    print(f"Resampler output (skip) - shape: {skip.shape}, min: {skip.min()}, max: {skip.max()}")
    
    residual = jnp.transpose(residual, (0, 2, 1))
    
    if residual.shape[1] < skip.shape[1]:
      residual = jnp.pad(residual, ((0, 0), (0, skip.shape[1] - residual.shape[1]), (0, 0)), mode='edge')
    elif residual.shape[1] > skip.shape[1]:
      residual = residual[:, :skip.shape[1], :]
    print(f"Matched residual - shape: {residual.shape}")
      
    raw_waveform = residual + skip
    print(f"Raw waveform (residual + skip) - min: {raw_waveform.min()}, max: {raw_waveform.max()}")
    
    waveform = jnp.clip(raw_waveform, -1, 1)
    
    output_samples = num_samples * self.output_sampling_rate // self.input_sampling_rate
    waveform = waveform[:, :output_samples, :]
    waveform = jnp.transpose(waveform, (0, 2, 1))
    print(f"Final waveform - shape: {waveform.shape}, min: {waveform.min()}, max: {waveform.max()}")
    print(f"=========================")
    
    return waveform
