# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import flax.linen as nn
import jax.numpy as jnp
from typing import List, Union
import jax


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
  """Returns the positional encoding (same as Tensor2Tensor).

  Args:
      timesteps: a 1-D Tensor of N indices, one per batch element.
      These may be fractional.
      embedding_dim: The number of output channels.
      min_timescale: The smallest time unit (should probably be 0.0).
      max_timescale: The largest time unit.
  Returns:
      a Tensor of timing signals [N, num_channels]
  """
  assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
  assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
  num_timescales = float(embedding_dim // 2)
  log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
  inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

  # scale embeddings
  scaled_time = scale * emb

  if flip_sin_to_cos:
    signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
  else:
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
  signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
  return signal


class FlaxTimestepEmbedding(nn.Module):
  r"""
  Time step Embedding Module. Learns embeddings for input time steps.

  Args:
      time_embed_dim (`int`, *optional*, defaults to `32`):
              Time step embedding dimension
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
              Parameters `dtype`
  """

  time_embed_dim: int = 32
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, temb):
    temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, name="linear_1")(temb)
    temb = nn.silu(temb)
    temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, name="linear_2")(temb)
    return temb


class FlaxTimesteps(nn.Module):
  r"""
  Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

  Args:
      dim (`int`, *optional*, defaults to `32`):
              Time step embedding dimension
  """

  dim: int = 32
  flip_sin_to_cos: bool = False
  freq_shift: float = 1.0
  scale: int = 1

  @nn.compact
  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(
        timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
    )


def get_1d_rotary_pos_embed(
    dim: int, pos: Union[jnp.array, int], theta: float = 10000.0, linear_factor=1.0, ntk_factor=1.0, freqs_dtype=jnp.float32
):
  """
  Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
  """
  assert dim % 2 == 0

  if isinstance(pos, int):
    pos = jnp.arange(pos)

  theta = theta * ntk_factor
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor
  freqs = jnp.outer(pos, freqs)
  freqs_cos = jnp.cos(freqs)
  freqs_sin = jnp.sin(freqs)
  out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)

  return out


class PixArtAlphaTextProjection(nn.Module):
  """
  Projects caption embeddings. Also handles dropout for classifier-free guidance.
  Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
  """

  hidden_size: int
  out_features: int = None
  act_fn: str = "gelu_tanh"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, caption):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_1",
    )(caption)
    if self.act_fn == "gelu_tanh":
      act_1 = nn.gelu
    elif self.act_fn == "silu":
      act_1 = nn.swish
    else:
      raise ValueError(f"Unknown activation function: {self.act_fn}")
    hidden_states = act_1(hidden_states)

    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_2",
    )(hidden_states)
    return hidden_states


class FluxPosEmbed(nn.Module):
  theta: int
  axes_dim: List[int]
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, ids):
    n_axes = ids.shape[-1]
    out_freqs = []
    pos = ids.astype(self.dtype)
    freqs_dtype = self.dtype
    for i in range(n_axes):
      out = get_1d_rotary_pos_embed(self.axes_dim[i], pos[..., i], freqs_dtype=freqs_dtype)
      out_freqs.append(out)

    out_freqs = jnp.concatenate(out_freqs, axis=1)
    return out_freqs


class CombinedTimestepTextProjEmbeddings(nn.Module):
  embedding_dim: int
  pooled_projection_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, timestep, pooled_projection):
    timesteps_proj = timestep
    timestep_emb = FlaxTimestepEmbedding(
        time_embed_dim=self.embedding_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
    )(timesteps_proj)
    pooled_projections = PixArtAlphaTextProjection(
        self.embedding_dim,
        act_fn="silu",
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )(pooled_projection)

    conditioning = timestep_emb + pooled_projections
    return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
  embedding_dim: int
  pooled_projection_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, timestep, guidance, pooled_projection):
    timesteps_proj = timestep
    timestep_emb = FlaxTimestepEmbedding(
        time_embed_dim=self.embedding_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
    )(timesteps_proj.astype(pooled_projection.dtype))

    guidance_proj = guidance
    guidance_emb = FlaxTimestepEmbedding(
        time_embed_dim=self.embedding_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
    )(guidance_proj.astype(pooled_projection.dtype))

    time_guidance_emb = timestep_emb + guidance_emb

    pooled_projections = PixArtAlphaTextProjection(
        self.embedding_dim, act_fn="silu", dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
    )(pooled_projection)
    conditioning = time_guidance_emb + pooled_projections

    return conditioning
