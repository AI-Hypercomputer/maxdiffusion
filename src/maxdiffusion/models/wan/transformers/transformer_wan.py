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

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import nnx
from .... import common_types, max_logging
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin
from ...embeddings_flax import get_1d_rotary_pos_embed

BlockSizes = common_types.BlockSizes

class WanRotaryPosEmbed(nnx.Module):
  def __init__(
    self,
    attention_head_dim: int,
    patch_size: Tuple[int, int, int],
    max_seq_len: int,
    theta: float = 10000.0
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len

    h_dim = w_dim = 2 * (attention_head_dim // 6)
    t_dim = attention_head_dim - h_dim - w_dim

    freqs = []
    for dim in [t_dim, h_dim, w_dim]:
      freq = get_1d_rotary_pos_embed(
        dim,
        self.max_seq_len,
        theta,
        freqs_dtype=jnp.float64,
        use_real=False
      )
      freqs.append(freq)
    self.freqs = jnp.concatenate(freqs, axis=1)
  
  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    sizes = [
        self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
        self.attention_head_dim // 6,
        self.attention_head_dim // 6,
    ]
    cumulative_sizes = jnp.cumsum(jnp.array(sizes))
    split_indices = cumulative_sizes[:-1]
    freqs_split = jnp.split(self.freqs, split_indices, axis=1)
    
    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))
    return freqs_final


class WanTransformer3DModel(nnx.Module, FlaxModelMixin, ConfigMixin):
  def __init__(
    self,
    rngs: nnx.Rngs,
    patch_size: Tuple[int] = (1, 2, 2),
    num_attention_heads: int = 40,
    attention_head_dim: int = 128,
    in_channels: int = 16,
    out_channels: int = 16,
    text_dim: int = 4096,
    freq_dim: int = 256,
    ffn_dim: int = 13824,
    num_layers: int = 40,
    cross_attn_norm: bool = True,
    qk_norm: Optional[str] = "rms_norm_across_heads",
    eps: float = 1e-6,
    image_dim: Optional[int] = None,
    added_kv_proj_dim: Optional[int] = None,
    rope_max_seq_len: int = 1024,
    pos_embed_seq_len: Optional[int] = None,
    flash_min_seq_length: int = 4096,
    flash_block_sizes: BlockSizes = None,
    mesh: jax.sharding.Mesh = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
    attention: str = "dot_product",
  ):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels

    #1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(
      attention_head_dim,
      patch_size,
      rope_max_seq_len
    )


class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):

  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size=(1, 2, 2),
      text_len=512,
      in_dim=16,
      dim=2048,
      ffn_dim=8192,
      freq_dim=256,
      text_dim=4096,
      out_dim=16,
      num_heads=16,
      num_layers=32,
      window_size=(-1, -1),
      qk_norm=True,
      cross_attn_norm=True,
      eps=1e-6,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
  ):
    self.path_embedding = nnx.Conv(
        in_dim,
        dim,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("batch",)),
        rngs=rngs,
    )

  def __call__(self, x):
    x = self.path_embedding(x)
    return x