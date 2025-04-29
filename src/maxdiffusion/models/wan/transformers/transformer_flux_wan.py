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

from typing import Tuple, Dict, Optional, Any, Union
import jax
import math
import jax.numpy as jnp
from chex import Array
import flax.linen as nn

from ...attention_flax import FlaxFeedForward, Fla
from ...embeddings_flax import (get_1d_rotary_pos_embed, FlaxTimesteps, FlaxTimestepEmbedding, PixArtAlphaTextProjection)

from ....configuration_utils import ConfigMixin, flax_register_to_config
from ...modeling_flax_utils import FlaxModelMixin


class WanRotaryPosEmbed(nn.Module):
  attention_head_dim: int
  patch_size: Tuple[int, int, int]
  theta: float = 10000.0
  max_seq_len: int

  @nn.compact
  def __call__(self, hidden_states: Array) -> Array:
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    h_dim = w_dim = 2 * (self.attention_head_dim // 6)
    t_dim = self.attention_head_dim - h_dim - w_dim

    freqs = []
    for dim in [t_dim, h_dim, w_dim]:
      freq = get_1d_rotary_pos_embed(dim, self.max_seq_length, self.theta, freqs_dtype=jnp.float64)
      freqs.append(freq)
    self.freqs = jnp.concatenate(freqs, dim=1)

    sizes = [
        self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
        self.attention_head_dim // 6,
        self.attention_head_dim // 6,
    ]
    cumulative_sizes = jnp.cumsum(jnp.array(sizes))
    split_indices = cumulative_sizes[:-1]
    freqs_split = jnp.split(freqs, split_indices, axis=1)

    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))

    return freqs_final


class WanImageEmbeddings(nn.Module):
  out_features: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, encoder_hidden_states_image: Array) -> Array:
    hidden_states = nn.LayerNorm(
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )(encoder_hidden_states_image)
    hidden_states = FlaxFeedForward(
        self.out_features, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
    )(hidden_states)
    hidden_states = nn.LayerNorm(
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )(hidden_states)
    return hidden_states


class WanTimeTextImageEmbeddings(nn.Module):
  dim: int
  time_freq_dim: int
  time_proj_dim: int
  text_embed_dim: int
  image_embed_dim: Optional[int] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, timestep: Array, encoder_hidden_states: Array, encoder_hidden_states_image: Array) -> Array:

    timestep = FlaxTimesteps(
        dim=self.time_freq_dim,
        flip_sin_to_cos=True,
        freq_shift=0,
    )(timestep)
    temb = FlaxTimestepEmbedding(time_embed_dim=self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype)(timestep)
    timestep_proj = nn.Dense(
        self.time_proj_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )(nn.silu(temb))
    encoder_hidden_states = PixArtAlphaTextProjection(
        hidden_size=self.dim,
        act_fn="gelu_tanh",
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )(encoder_hidden_states)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states_image = WanImageEmbeddings(
          out_features=self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
      )(encoder_hidden_states_image)

    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTransformerBlock(nn.Module):
  dim: int
  ffn_dim: int
  num_heads: int
  qk_norm: str = "rms_norm_across_heads"
  cross_attn_norm: bool = False
  eps: float = 1e-6
  added_kv_proj_dim: Optional[int] = None

  @nn.compact
  def __call__(self, hidden_states: Array, encoder_hidden_states: Array, temb: Array, rotary_emb: Array):

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
        (scale_shift_table + temb.astype(jnp.float32)), 6, axis=1
    )

    # 1. Self-attention
    norm_hidden_states = (
        nn.LayerNorm(
            epsilon=self.eps,
            use_bias=False,
            use_scale=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(hidden_states.astype(jnp.float32))
        * (1 + scale_msa)
        + shift_msa
    ).astype(hidden_states.dtype)
    attn_output = FlaxWanAttention(
        query_dim=self.dim,
        heads=self.num_heads,
        dim_head=self.dim // self.num_heads,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        attention_kernel=self.attention_kernel,
        mesh=self.mesh,
        flash_block_sizes=self.flash_block_sizes,
    )


class WanTransformer3dModel(nn.Module, FlaxModelMixin, ConfigMixin):
  r"""
  A Transformer model for video-like data used in the Wan model.

  Args:
      patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
          3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
      num_attention_heads (`int`, defaults to `40`):
          Fixed length for text embeddings.
      attention_head_dim (`int`, defaults to `128`):
          The number of channels in each head.
      in_channels (`int`, defaults to `16`):
          The number of channels in the input.
      out_channels (`int`, defaults to `16`):
          The number of channels in the output.
      text_dim (`int`, defaults to `512`):
          Input dimension for text embeddings.
      freq_dim (`int`, defaults to `256`):
          Dimension for sinusoidal time embeddings.
      ffn_dim (`int`, defaults to `13824`):
          Intermediate dimension in feed-forward network.
      num_layers (`int`, defaults to `40`):
          The number of layers of transformer blocks to use.
      window_size (`Tuple[int]`, defaults to `(-1, -1)`):
          Window size for local attention (-1 indicates global attention).
      cross_attn_norm (`bool`, defaults to `True`):
          Enable cross-attention normalization.
      qk_norm (`bool`, defaults to `True`):
          Enable query/key normalization.
      eps (`float`, defaults to `1e-6`):
          Epsilon value for normalization layers.
      add_img_emb (`bool`, defaults to `False`):
          Whether to use img_emb.
      added_kv_proj_dim (`int`, *optional*, defaults to `None`):
          The number of channels to use for the added key and value projections. If `None`, no projection is used.
  """

  patch_size: Tuple[int] = (1, 2, 2)
  num_attention_heads: int = 40
  attention_head_dim: int = 128
  in_channels: int = 16
  out_channels: int = 16
  text_dim: int = 4096
  freq_dim: int = 256
  ffn_dim: int = 13824
  num_layers: int = 40
  cross_attn_norm: bool = True
  qk_norm: Optional[str] = "rms_norm_across_heads"
  eps: float = 1e-6
  image_dim: Optional[int] = None
  added_kv_proj_dim: Optional[int] = None
  rope_max_seq_len: int = 1024
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  attention: str = "dot_product"

  @nn.compact
  def __call__(
      self,
      hidden_states: Array,
      timestep: Array,
      encoder_hidden_states: Array,
      encoder_hidden_states_image: Optional[Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Union[Array, Dict[str, Array]]:

    inner_dim = self.num_attention_heads * self.attention_head_dim
    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 1. Patch & position embedding
    rotary_emb = WanRotaryPosEmbed(
        attention_head_dim=self.attention_head_dim, patch_size=self.patch_size, max_seq_len=self.rope_max_seq_len
    )(hidden_states)
    hidden_states = nn.Conv(
        features=inner_dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )(hidden_states)
    flattened_shape = (batch_size, num_channels, -1)  # TODO is his num_channels or frames?
    flattened = hidden_states.reshape(flattened_shape)
    transposed = jnp.transpose(flattened, (0, 2, 1))

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = WanTimeTextImageEmbeddings(
        dim=inner_dim,
        time_freq_dim=self.freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=self.text_dim,
        image_embed_dim=self.image_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )(timestep, encoder_hidden_states, encoder_hidden_states_image)
