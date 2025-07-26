# Copyright 2025 Lightricks Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Lightricks/LTX-Video/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the Torch version available at:
# https://github.com/Lightricks/LTX-Video/tree/main
from typing import Dict, Optional, Tuple

import jax
import jax.nn
import jax.numpy as jnp
from flax import linen as nn

from maxdiffusion.models.ltx_video.transformers.activations import get_activation
from maxdiffusion.models.ltx_video.linear import DenseGeneral


def get_timestep_embedding_multidim(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> jnp.ndarray:
  """
  Computes sinusoidal timestep embeddings while preserving the original dimensions.
  No reshaping to 1D is performed at any stage.

  Args:
      timesteps (jnp.ndarray): A Tensor of arbitrary shape containing timestep values.
      embedding_dim (int): The dimension of the output.
      flip_sin_to_cos (bool): Whether the embedding order should be `cos, sin` (if True)
                              or `sin, cos` (if False).
      downscale_freq_shift (float): Controls the delta between frequencies between dimensions.
      scale (float): Scaling factor applied to the embeddings.
      max_period (int): Controls the maximum frequency of the embeddings.

  Returns:
      jnp.ndarray: A Tensor of shape (*timesteps.shape, embedding_dim) with positional embeddings.
  """
  half_dim = embedding_dim // 2
  exponent = -jnp.log(max_period) * jnp.arange(half_dim, dtype=jnp.float32)
  exponent = exponent / (half_dim - downscale_freq_shift)
  shape = (1,) * timesteps.ndim + (half_dim,)  # (1, 1, ..., 1, half_dim)
  emb = jnp.exp(exponent).reshape(*shape)  # Expand to match timesteps' shape
  emb = nn.with_logical_constraint(emb, ("activation_batch", "activation_norm_length", "activation_embed"))
  # Broadcasting to match shape (*timesteps.shape, half_dim)
  emb = timesteps[..., None] * emb
  emb = scale * emb
  # Shape (*timesteps.shape, embedding_dim)
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
  if flip_sin_to_cos:
    emb = jnp.concatenate([emb[..., half_dim:], emb[..., :half_dim]], axis=-1)

  return emb


class TimestepEmbedding(nn.Module):
  in_channels: int
  time_embed_dim: int
  act_fn: str = "silu"
  out_dim: Optional[int] = None
  sample_proj_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    """Initialize layers efficiently"""
    self.linear_1 = DenseGeneral(
        self.time_embed_dim,
        use_bias=self.sample_proj_bias,
        kernel_axes=(None, "mlp"),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="linear_1",
    )

    self.act = get_activation(self.act_fn)
    time_embed_dim_out = self.out_dim if self.out_dim is not None else self.time_embed_dim
    self.linear_2 = DenseGeneral(
        time_embed_dim_out,
        use_bias=self.sample_proj_bias,
        kernel_axes=("embed", "mlp"),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="linear_2",
    )

  def __call__(self, sample, condition=None):
    sample = nn.with_logical_constraint(sample, ("activation_batch", "activation_norm_length", "activation_embed"))
    sample = self.linear_1(sample)
    sample = self.act(sample)
    sample = self.linear_2(sample)
    return sample


class Timesteps(nn.Module):
  num_channels: int
  flip_sin_to_cos: bool
  downscale_freq_shift: float
  scale: int = 1

  def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
    t_emb = get_timestep_embedding_multidim(
        timesteps,
        self.num_channels,
        flip_sin_to_cos=self.flip_sin_to_cos,
        downscale_freq_shift=self.downscale_freq_shift,
        scale=self.scale,
    )
    return t_emb


class AlphaCombinedTimestepSizeEmbeddings(nn.Module):

  embedding_dim: int
  size_emb_dim: int
  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    """Initialize sub-modules."""
    self.outdim = self.size_emb_dim
    self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    self.timestep_embedder = TimestepEmbedding(
        in_channels=256,
        time_embed_dim=self.embedding_dim,
        name="timestep_embedder",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
    )

  def __call__(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_dtype))
    return timesteps_emb


class AdaLayerNormSingle(nn.Module):
  r"""
  Norm layer adaptive layer norm single (adaLN-single).

  As proposed in: https://arxiv.org/abs/2310.00426; Section 2.3.

  Parameters:
      embedding_dim (`int`): The size of each embedding vector.
  """

  embedding_dim: int
  embedding_coefficient: int = 6
  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    self.emb = AlphaCombinedTimestepSizeEmbeddings(
        self.embedding_dim,
        size_emb_dim=self.embedding_dim // 3,
        name="emb",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
    )

    self.silu = jax.nn.silu
    self.linear = DenseGeneral(
        self.embedding_coefficient * self.embedding_dim,
        use_bias=True,
        kernel_axes=("mlp", "embed"),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="linear",
    )

  def __call__(
      self,
      timestep: jnp.ndarray,
      added_cond_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
      batch_size: Optional[int] = None,
      hidden_dtype: Optional[jnp.dtype] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute AdaLayerNorm-Single modulation.

    Returns:
        Tuple:
            - Processed embedding after SiLU + linear transformation.
            - Original embedded timestep.
    """
    embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
    return self.linear(self.silu(embedded_timestep)), embedded_timestep
