# Copyright 2026 Google LLC / The HuggingFace Team. All rights reserved.
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

"""DotProductAttentionStrategy for standard un-sharded or memory-efficient dot-product attention."""

from typing import Any, Optional
import jax
import jax.numpy as jnp
from maxdiffusion.models import attention_utils


class DotProductAttentionStrategy:
  """Orchestrates standard dot-product attention or memory-efficient chunked attention."""

  def __init__(
      self,
      scale: float,
      split_head_dim: bool = True,
      float32_qk_product: bool = True,
      use_memory_efficient_attention: bool = False,
  ):
    self.scale = scale
    self.split_head_dim = split_head_dim
    self.float32_qk_product = float32_qk_product
    self.use_memory_efficient_attention = use_memory_efficient_attention

  def pre_all_to_all_hook(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      context: Optional[dict[str, Any]] = None,
  ) -> Optional[tuple[jax.Array, jax.Array]]:
    """No pre-all-to-all Cauchy-Schwarz bounds needed for dot product attention."""
    return None

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      q_seq_len: int = 0,
      kv_seq_len: int = 0,
      attention_mask: Optional[jax.Array] = None,
      dtype: Optional[jnp.dtype] = None,
      **kwargs,
  ) -> jax.Array:
    """Computes scaled dot-product attention."""
    out_dtype = dtype or query.dtype

    if self.use_memory_efficient_attention and attention_mask is None:
      query_states = query.transpose(1, 0, 2)
      key_states = key.transpose(1, 0, 2)
      value_states = value.transpose(1, 0, 2)

      flatten_latent_dim = query_states.shape[-3]
      if flatten_latent_dim % 64 == 0:
        query_chunk_size = int(flatten_latent_dim / 64)
      elif flatten_latent_dim % 16 == 0:
        query_chunk_size = int(flatten_latent_dim / 16)
      elif flatten_latent_dim % 4 == 0:
        query_chunk_size = int(flatten_latent_dim / 4)
      else:
        query_chunk_size = int(flatten_latent_dim)

      hidden_states = attention_utils.jax_memory_efficient_attention(
          query_states,
          key_states,
          value_states,
          query_chunk_size=query_chunk_size,
          key_chunk_size=4096 * 4,
      )
      return hidden_states.transpose(1, 0, 2)

    q_in = query.astype(jnp.float32) if self.float32_qk_product else query
    k_in = key.astype(jnp.float32) if self.float32_qk_product else key

    if self.split_head_dim:
      attention_scores = jnp.einsum("b t n h, b f n h -> b n f t", k_in, q_in)
    else:
      attention_scores = jnp.einsum("b i d, b j d -> b i j", q_in, k_in)

    attention_scores = attention_scores * self.scale
    if attention_mask is not None:
      attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)

    attention_probs = jax.nn.softmax(attention_scores, axis=-1 if self.split_head_dim else 2)
    if attention_mask is not None:
      has_valid_key = jnp.any(attention_mask == 0, axis=-1, keepdims=True)
      attention_probs = jnp.where(has_valid_key, attention_probs, 0)

    attention_probs = attention_probs.astype(out_dtype)

    if self.split_head_dim:
      hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value)
      b = hidden_states.shape[0]
      heads = hidden_states.shape[2]
      dim_head = hidden_states.shape[3]
      hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    else:
      hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value)

    return hidden_states
