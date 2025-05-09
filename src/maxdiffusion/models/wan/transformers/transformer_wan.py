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

from typing import Tuple, Optional, Dict, Union, Any
import jax
import jax.numpy as jnp
from flax import nnx
from .... import common_types, max_logging
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin, register_to_config
from ...embeddings_flax import get_1d_rotary_pos_embed, NNXFlaxTimesteps, NNXTimestepEmbedding

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


class WanTimeTextImageEmbedding(nnx.Module):
  def __init__(
    self,
    rngs: nnx.Rngs,
    dim: int,
    time_freq_dim: int,
    time_proj_dim: int,
    text_embed_dim: int,
    image_embed_dim: Optional[int] = None,
    pos_embed_seq_len: Optional[int] = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(
      dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0
    )
    self.time_embedder = NNXTimestepEmbedding(
      rngs=rngs, in_channels=time_freq_dim, time_embed_dim=dim,
      dtype=dtype, weights_dtype=weights_dtype, precision=precision
    )
  
  def __call__(
    self,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    encoder_hidden_states_image: Optional[jax.Array] = None
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    breakpoint()



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
  
  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
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
      added_kn_proj_dim: Optional[int] = None,
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
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("batch",)),
    )

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    self.condition_embedder = WanTimeTextImageEmbedding(
      rngs=rngs,
      dim=inner_dim,
      time_freq_dim=freq_dim,
      time_proj_dim=inner_dim * 6,
      text_embed_dim=text_dim,
      image_embed_dim=image_dim,
      pos_embed_seq_len=pos_embed_seq_len
    )

  def __call__(
    self,
    hidden_states: jax.Array,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    encoder_hidden_states_image: Optional[jax.Array] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Union[jax.Array, Dict[str, jax.Array]]:
    batch_size, num_frames, height, width, num_channels = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w


    rotary_emb = self.rope(hidden_states)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = jax.lax.collapse(hidden_states, 1, -1)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
      timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    #hidden_states = 
    # Torch shape: ([1, 5120, 21, 45, 80])
    # Jax shape: (1, 21, 45, 80, 5120) so channels is 5120


    return hidden_states