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
from typing import Tuple, Union, Sequence, Optional
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import scipy.signal.windows
from maxdiffusion import common_types
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.modeling_flax_utils import FlaxModelMixin

Array = common_types.Array
DType = common_types.DType

# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

class Snake(nnx.Module):
  def __init__(
      self,
      in_features: int,
      alpha: float = 1.0,
      alpha_trainable: bool = True,
      alpha_logscale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.alpha_logscale = alpha_logscale
    if alpha_logscale:
      self.alpha = nnx.Param(jnp.zeros((in_features,), dtype=jnp.float32))
    else:
      self.alpha = nnx.Param(jnp.ones((in_features,), dtype=jnp.float32) * alpha)
    
    self.eps = 1e-9

  def __call__(self, x: Array) -> Array:
    alpha = self.alpha.value
    if self.alpha_logscale:
      alpha = jnp.exp(alpha)
    
    return x + (1.0 / (alpha + self.eps)) * jnp.sin(x * alpha)**2


class SnakeBeta(nnx.Module):
  def __init__(
      self,
      in_features: int,
      alpha: float = 1.0,
      alpha_trainable: bool = True,
      alpha_logscale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.alpha_logscale = alpha_logscale
    if alpha_logscale:
      self.alpha = nnx.Param(jnp.zeros((in_features,), dtype=jnp.float32))
      self.beta = nnx.Param(jnp.zeros((in_features,), dtype=jnp.float32))
    else:
      self.alpha = nnx.Param(jnp.ones((in_features,), dtype=jnp.float32) * alpha)
      self.beta = nnx.Param(jnp.ones((in_features,), dtype=jnp.float32) * alpha)
    
    self.eps = 1e-9

  def __call__(self, x: Array) -> Array:
    alpha = self.alpha.value
    beta = self.beta.value
    if self.alpha_logscale:
      alpha = jnp.exp(alpha)
      beta = jnp.exp(beta)
    
    return x + (1.0 / (beta + self.eps)) * jnp.sin(x * alpha)**2

# ---------------------------------------------------------------------------
# Anti-aliased resampling helpers
# ---------------------------------------------------------------------------

def _sinc(x: np.ndarray) -> np.ndarray:
  return np.where(
      x == 0,
      1.0,
      np.sin(np.pi * x) / (np.pi * x),
  )

def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> np.ndarray:
  even = kernel_size % 2 == 0
  half_size = kernel_size // 2
  delta_f = 4 * half_width
  amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
  if amplitude > 50.0:
    beta = 0.1102 * (amplitude - 8.7)
  elif amplitude >= 21.0:
    beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
  else:
    beta = 0.0
  
  window = scipy.signal.windows.kaiser(kernel_size, beta=beta, sym=not even)
  
  time = np.arange(-half_size, half_size) + 0.5 if even else np.arange(kernel_size) - half_size
  if cutoff == 0:
    filter_ = np.zeros_like(time)
  else:
    filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
  
  return filter_

class LowPassFilter1d(nnx.Module):
  def __init__(
      self,
      channels: int,
      cutoff: float = 0.5,
      half_width: float = 0.6,
      stride: int = 1,
      padding: bool = True,
      kernel_size: int = 12,
      *,
      rngs: nnx.Rngs,
  ):
    self.kernel_size = kernel_size
    self.even = kernel_size % 2 == 0
    self.pad_left = kernel_size // 2 - int(self.even)
    self.pad_right = kernel_size // 2
    self.stride = stride
    self.padding = padding
    
    filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
    self.filter_weight = jnp.tile(filt[:, None, None], (1, 1, channels))
    self.channels = channels
    
  def __call__(self, x: Array) -> Array:
    b, t, c = x.shape
    
    if self.padding:
      padding_config = ((0, 0), (self.pad_left, self.pad_right), (0, 0))
      x = jnp.pad(x, padding_config, mode="edge")
    
    x_transposed = jnp.transpose(x, (0, 2, 1)) # (B, C, T)
    w_transposed = jnp.transpose(self.filter_weight, (2, 1, 0)) # (C, 1, K)
    
    out = jax.lax.conv_general_dilated(
        lhs=x_transposed,
        rhs=w_transposed,
        window_strides=(self.stride,),
        padding="VALID",
        feature_group_count=self.channels,
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2), # N, C, W
            rhs_spec=(0, 1, 2), # O, I, W
            out_spec=(0, 1, 2), # N, C, W
        )
    )
    
    return jnp.transpose(out, (0, 2, 1))


class UpSample1d(nnx.Module):
  def __init__(
      self,
      channels: int,
      ratio: int = 2,
      kernel_size: Optional[int] = None,
      window_type: str = "kaiser",
      *,
      rngs: nnx.Rngs,
  ):
    self.ratio = ratio
    self.stride = ratio
    self.channels = channels
    
    if window_type == "hann":
      rolloff = 0.99
      lowpass_filter_width = 6
      width = math.ceil(lowpass_filter_width / rolloff)
      self.kernel_size = 2 * width * ratio + 1
      self.pad = width
      self.pad_left = 2 * width * ratio
      self.pad_right = self.kernel_size - ratio
      
      time_axis = (np.arange(self.kernel_size) / ratio - width) * rolloff
      time_clamped = np.clip(time_axis, -lowpass_filter_width, lowpass_filter_width)
      window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
      sinc_filter = (_sinc(time_axis) * window * rolloff / ratio)
      
    else:
      self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
      self.pad = self.kernel_size // ratio - 1
      self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
      self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
      
      sinc_filter = kaiser_sinc_filter1d(
          cutoff=0.5 / ratio,
          half_width=0.6 / ratio,
          kernel_size=self.kernel_size,
      )
      
    self.filter_weight = jnp.tile(sinc_filter[:, None, None], (1, 1, channels))

  def __call__(self, x: Array) -> Array:
    b, t, c = x.shape
    
    padding_config = ((0, 0), (self.pad, self.pad), (0, 0))
    x = jnp.pad(x, padding_config, mode="edge")
    
    x_transposed = jnp.transpose(x, (0, 2, 1)) # (B, C, T)
    w_transposed = jnp.transpose(self.filter_weight, (2, 1, 0)) # (C, 1, K)
    w_flipped = w_transposed[..., ::-1]
    
    out = jax.lax.conv_general_dilated(
        lhs=x_transposed,
        rhs=w_flipped,
        window_strides=(1,),
        padding="VALID",
        lhs_dilation=(self.ratio,),
        feature_group_count=self.channels,
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2), # N, C, W
            rhs_spec=(0, 1, 2), # O, I, W
            out_spec=(0, 1, 2), # N, C, W
        )
    )
    
    out = out * self.ratio
    out = out[..., self.pad_left : -self.pad_right]
    
    return jnp.transpose(out, (0, 2, 1))


class DownSample1d(nnx.Module):
  def __init__(self, channels: int, ratio: int = 2, kernel_size: Optional[int] = None, *, rngs: nnx.Rngs):
    self.ratio = ratio
    self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
    self.lowpass = LowPassFilter1d(
        channels=channels,
        cutoff=0.5 / ratio,
        half_width=0.6 / ratio,
        stride=ratio,
        kernel_size=self.kernel_size,
        rngs=rngs,
    )

  def __call__(self, x: Array) -> Array:
    return self.lowpass(x)


class Activation1d(nnx.Module):
  def __init__(
      self,
      channels: int,
      activation: nnx.Module,
      up_ratio: int = 2,
      down_ratio: int = 2,
      up_kernel_size: int = 12,
      down_kernel_size: int = 12,
      *,
      rngs: nnx.Rngs,
  ):
    self.act = activation
    self.upsample = UpSample1d(channels, up_ratio, up_kernel_size, rngs=rngs)
    self.downsample = DownSample1d(channels, down_ratio, down_kernel_size, rngs=rngs)

  def __call__(self, x: Array) -> Array:
    x = self.upsample(x)
    x = self.act(x)
    return self.downsample(x)

# ---------------------------------------------------------------------------
# AMP Blocks and Vocoder
# ---------------------------------------------------------------------------

class AMPBlock1(nnx.Module):
  def __init__(
      self,
      channels: int,
      kernel_size: int = 3,
      dilation: Sequence[int] = (1, 3, 5),
      activation: str = "snake",
      *,
      rngs: nnx.Rngs,
  ):
    act_cls = SnakeBeta if activation == "snakebeta" else Snake
    
    self.convs1 = nnx.List(
        [
            nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(kernel_size,),
                strides=(1,),
                kernel_dilation=(dilation[0],),
                padding="SAME",
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(kernel_size,),
                strides=(1,),
                kernel_dilation=(dilation[1],),
                padding="SAME",
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(kernel_size,),
                strides=(1,),
                kernel_dilation=(dilation[2],),
                padding="SAME",
                rngs=rngs,
            ),
        ]
    )

    self.convs2 = nnx.List(
        [
            nnx.Conv(channels, channels, (kernel_size,), (1,), padding="SAME", rngs=rngs),
            nnx.Conv(channels, channels, (kernel_size,), (1,), padding="SAME", rngs=rngs),
            nnx.Conv(channels, channels, (kernel_size,), (1,), padding="SAME", rngs=rngs),
        ]
    )

    self.acts1 = nnx.List([Activation1d(channels, act_cls(channels, rngs=rngs), rngs=rngs) for _ in range(len(self.convs1))])
    self.acts2 = nnx.List([Activation1d(channels, act_cls(channels, rngs=rngs), rngs=rngs) for _ in range(len(self.convs2))])

  def __call__(self, x: Array) -> Array:
    for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
      xt = a1(x)
      xt = c1(xt)
      xt = a2(xt)
      xt = c2(xt)
      x = x + xt
    return x


class Vocoder(nnx.Module, FlaxModelMixin, ConfigMixin):
  @register_to_config
  def __init__(
      self,
      resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
      upsample_rates: Sequence[int] = (6, 5, 2, 2, 2),
      upsample_kernel_sizes: Sequence[int] = (16, 15, 8, 4, 4),
      resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
      upsample_initial_channel: int = 1024,
      resblock: str = "AMP1",
      output_sampling_rate: int = 24000,
      activation: str = "snakebeta",
      use_tanh_at_final: bool = True,
      apply_final_activation: bool = True,
      use_bias_at_final: bool = True,
      *,
      rngs: nnx.Rngs,
      dtype: DType = jnp.float32,
  ):
    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.use_tanh_at_final = use_tanh_at_final
    self.apply_final_activation = apply_final_activation
    self.dtype = dtype

    self.conv_in = nnx.Conv(
        in_features=128,
        out_features=upsample_initial_channel,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        rngs=rngs,
        dtype=self.dtype,
    )

    self.upsamplers = nnx.List()
    for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      self.upsamplers.append(
          nnx.ConvTranspose(
              in_features=upsample_initial_channel // (2**i),
              out_features=upsample_initial_channel // (2 ** (i + 1)),
              kernel_size=(kernel_size,),
              strides=(stride,),
              padding="SAME",
              rngs=rngs,
              dtype=self.dtype,
          )
      )

    self.resnets = nnx.List()
    for i in range(len(upsample_rates)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
        self.resnets.append(AMPBlock1(ch, kernel_size, dilations, activation=activation, rngs=rngs))

    final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
    self.act_out = Activation1d(final_channels, SnakeBeta(final_channels, rngs=rngs), rngs=rngs)

    self.conv_out = nnx.Conv(
        in_features=final_channels,
        out_features=2,
        kernel_size=(7,),
        strides=(1,),
        padding="SAME",
        use_bias=use_bias_at_final,
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

    for i in range(self.num_upsamples):
      hidden_states = self.upsamplers[i](hidden_states)
      
      start = i * self.num_kernels
      end = (i + 1) * self.num_kernels

      res_sum = 0.0
      for j in range(start, end):
        res_sum = res_sum + self.resnets[j](hidden_states)

      hidden_states = res_sum / self.num_kernels

    hidden_states = self.act_out(hidden_states)
    hidden_states = self.conv_out(hidden_states)

    if self.apply_final_activation:
      if self.use_tanh_at_final:
        hidden_states = jnp.tanh(hidden_states)
      else:
        hidden_states = jnp.clip(hidden_states, -1.0, 1.0)

    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    return hidden_states

# ---------------------------------------------------------------------------
# STFT and Mel Spectrogram Modules
# ---------------------------------------------------------------------------

class _STFTFn(nnx.Module):
  def __init__(self, filter_length: int, hop_length: int, win_length: int, *, rngs: nnx.Rngs):
    self.hop_length = hop_length
    self.win_length = win_length
    n_freqs = filter_length // 2 + 1
    
    # These will be loaded from checkpoint, shape (n_freqs * 2, 1, filter_length) in PyTorch
    # For Flax NWC layout, we need (filter_length, 1, n_freqs * 2)
    self.forward_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2), dtype=jnp.float32))
    self.inverse_basis = nnx.Param(jnp.zeros((filter_length, 1, n_freqs * 2), dtype=jnp.float32))

  def __call__(self, y: Array) -> Tuple[Array, Array]:
    # y shape: (B, T) or (B, 1, T)
    if y.ndim == 2:
      y = y[:, :, None]  # (B, T, 1)
    
    left_pad = max(0, self.win_length - self.hop_length)
    
    # Causal padding (left only)
    padding_config = ((0, 0), (left_pad, 0), (0, 0))
    y = jnp.pad(y, padding_config, mode="edge")
    
    # Transpose to (B, 1, T) for standard conv or use dimension spec
    # Let's use lax.conv_general_dilated
    y_transposed = jnp.transpose(y, (0, 2, 1)) # (B, 1, T)
    w_transposed = jnp.transpose(self.forward_basis.value, (2, 1, 0)) # (O, I, K)
    
    spec = jax.lax.conv_general_dilated(
        lhs=y_transposed,
        rhs=w_transposed,
        window_strides=(self.hop_length,),
        padding="VALID",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2), # N, C, W
            rhs_spec=(0, 1, 2), # O, I, W
            out_spec=(0, 1, 2), # N, C, W
        )
    )
    
    # spec shape is (B, n_freqs * 2, T_frames)
    n_freqs = spec.shape[1] // 2
    real = spec[:, :n_freqs, :]
    imag = spec[:, n_freqs:, :]
    
    magnitude = jnp.sqrt(real**2 + imag**2)
    phase = jnp.arctan2(imag, real)
    
    return magnitude, phase


class MelSTFT(nnx.Module):
  def __init__(self, filter_length: int, hop_length: int, win_length: int, n_mel_channels: int, *, rngs: nnx.Rngs):
    self.stft_fn = _STFTFn(filter_length, hop_length, win_length, rngs=rngs)
    n_freqs = filter_length // 2 + 1
    
    # Loaded from checkpoint, shape (n_mel_channels, n_freqs)
    self.mel_basis = nnx.Param(jnp.zeros((n_mel_channels, n_freqs), dtype=jnp.float32))

  def mel_spectrogram(self, y: Array) -> Tuple[Array, Array, Array, Array]:
    magnitude, phase = self.stft_fn(y)
    
    # magnitude shape: (B, n_freqs, T_frames)
    # mel_basis shape: (n_mel_channels, n_freqs)
    # we want (B, n_mel_channels, T_frames)
    
    # Use jnp.tensordot or einsum
    mel = jnp.einsum("mf,bft->bmt", self.mel_basis.value, magnitude)
    
    energy = jnp.linalg.norm(magnitude, axis=1)
    log_mel = jnp.log(jnp.clip(mel, a_min=1e-5))
    
    return log_mel, magnitude, phase, energy

# ---------------------------------------------------------------------------
# Vocoder With BWE
# ---------------------------------------------------------------------------

class LTX2VocoderWithBWE(nnx.Module, FlaxModelMixin, ConfigMixin):
  @register_to_config
  def __init__(
      self,
      vocoder: Vocoder,
      bwe_generator: Vocoder,
      mel_stft: MelSTFT,
      input_sampling_rate: int,
      output_sampling_rate: int,
      hop_length: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.vocoder = vocoder
    self.bwe_generator = bwe_generator
    self.mel_stft = mel_stft
    self.input_sampling_rate = input_sampling_rate
    self.output_sampling_rate = output_sampling_rate
    self.hop_length = hop_length
    
    ratio = output_sampling_rate // input_sampling_rate
    # Resampler uses 2 channels (stereo) as in reference
    self.resampler = UpSample1d(channels=2, ratio=ratio, window_type="hann", rngs=rngs)

  def _compute_mel(self, audio: Array) -> Array:
    batch, n_channels, length = audio.shape
    flat = audio.reshape(batch * n_channels, length)
    mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
    return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

  def __call__(self, mel_spec: Array) -> Array:
    # Ensure run in fp32
    mel_spec = jax.lax.convert_element_type(mel_spec, jnp.float32)
    
    x = self.vocoder(mel_spec)
    _, _, length_low_rate = x.shape
    output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

    remainder = length_low_rate % self.hop_length
    if remainder != 0:
      padding_config = ((0, 0), (0, 0), (0, self.hop_length - remainder))
      x = jnp.pad(x, padding_config, mode="edge")

    mel = self._compute_mel(x)
    mel_for_bwe = jnp.transpose(mel, (0, 1, 3, 2)) # (B, C, T, F)
    
    residual = self.bwe_generator(mel_for_bwe)
    skip = self.resampler(x)
    
    # Transpose x to (B, T, C) for resampler?
    # UpSample1d expects (B, T, C).
    # x is (B, C, T). Let's transpose!
    x_transposed = jnp.transpose(x, (0, 2, 1))
    skip = self.resampler(x_transposed)
    skip = jnp.transpose(skip, (0, 2, 1)) # Back to (B, C, T)
    
    out = residual + skip
    out = jnp.clip(out, -1.0, 1.0)
    
    return out[..., :output_length]
