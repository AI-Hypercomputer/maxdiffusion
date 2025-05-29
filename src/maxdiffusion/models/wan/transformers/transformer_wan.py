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
import math
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from .... import common_types
from ...modeling_flax_utils import FlaxModelMixin, get_activation
from ....configuration_utils import ConfigMixin, register_to_config
from ...embeddings_flax import (
  get_1d_rotary_pos_embed,
  NNXFlaxTimesteps,
  NNXTimestepEmbedding,
  NNXPixArtAlphaTextProjection
)
from ...normalization_flax import FP32LayerNorm
from ...attention_flax import FlaxWanAttention

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
    freqs = jnp.concatenate(freqs, axis=1)

    sizes = [
        self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
        self.attention_head_dim // 6,
        self.attention_head_dim // 6,
    ]
    cumulative_sizes = jnp.cumsum(jnp.array(sizes))
    split_indices = cumulative_sizes[:-1]
    self.freqs_split = jnp.split(freqs, split_indices, axis=1)
  
  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_f = jnp.expand_dims(jnp.expand_dims(self.freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, self.freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(self.freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, self.freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(self.freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, self.freqs_split[2].shape[-1]))

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
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(
      rngs=rngs,
      in_features=dim,
      out_features=time_proj_dim,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", "mlp",)),
      bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
      rngs=rngs,
      in_features=text_embed_dim,
      hidden_size=dim,
      act_fn="gelu_tanh",
    )
  
  def __call__(
    self,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    encoder_hidden_states_image: Optional[jax.Array] = None
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    
    timestep_proj = self.time_proj(self.act_fn(temb))

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
      raise NotImplementedError("currently img2vid is not supported")
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class ApproximateGELU(nnx.Module):
  r"""
  The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).
  """
  def __init__(
    self,
    rngs: nnx.Rngs,
    dim_in: int,
    dim_out: int,
    bias: bool,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
  ):
    self.proj = nnx.Linear(
      rngs=rngs,
      in_features=dim_in,
      out_features=dim_out,
      use_bias=bias,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", "mlp",)),
      bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
  
  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.proj(x)
    return nnx.gelu(x)
  

class WanFeedForward(nnx.Module):
  def __init__(
    self,
    rngs: nnx.Rngs,
    dim: int,
    dim_out: Optional[int] = None,
    mult: int = 4,
    dropout: float = 0.0,
    activation_fn: str = "geglu",
    final_dropout: bool = False,
    inner_dim: int = None,
    bias: bool = True,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
    
    self.act_fn = None
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(
        rngs=rngs,
        dim_in=dim,
        dim_out=inner_dim,
        bias=bias,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision
      )
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")

    self.proj_out = nnx.Linear(
      rngs=rngs,
      in_features=inner_dim,
      out_features=dim_out,
      use_bias=bias,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("mlp", "embed",)),
      bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
    )
  
  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    hidden_states = self.act_fn(hidden_states)
    return self.proj_out(hidden_states)



class WanTransformerBlock(nnx.Module):
  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      # In torch, this is none, so it can be ignored.  
      # added_kv_proj_dim: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",

  ):
    
    # 1. Self-attention
    self.norm1 = FP32LayerNorm(
      rngs=rngs,
      dim=dim,
      eps=eps,
      elementwise_affine=False
    )
    self.attn1 = FlaxWanAttention(
      rngs=rngs,
      query_dim=dim,
      heads=num_heads,
      dim_head= dim // num_heads,
      qk_norm=qk_norm,
      eps=eps,
      flash_min_seq_length=flash_min_seq_length,
      flash_block_sizes=flash_block_sizes,
      mesh=mesh,
      dtype=dtype,
      weights_dtype=weights_dtype,
      precision=precision,
      attention_kernel=attention
    )

    # 1. Cross-attention
    self.attn2 = FlaxWanAttention(
      rngs=rngs,
      query_dim=dim,
      heads=num_heads,
      dim_head= dim // num_heads,
      qk_norm=qk_norm,
      eps=eps,
      flash_min_seq_length=flash_min_seq_length,
      flash_block_sizes=flash_block_sizes,
      mesh=mesh,
      dtype=dtype,
      weights_dtype=weights_dtype,
      precision=precision,
      attention_kernel=attention
    )
    assert cross_attn_norm == True
    self.norm2 = FP32LayerNorm(
      rngs=rngs,
      dim=dim,
      eps=eps,
      elementwise_affine=True
    )

    # 3. Feed-forward
    self.ffn = WanFeedForward(
      rngs=rngs,
      dim=dim,
      inner_dim=ffn_dim,
      activation_fn="gelu-approximate",
      dtype=dtype,
      weights_dtype=weights_dtype,
      precision=precision
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    
    key = rngs.params()
    self.scale_shift_table = nnx.Param(jax.random.normal(key, (1, 6, dim)) / dim**0.5)
  
  def __call__(
    self,
    hidden_states: jax.Array,
    encoder_hidden_states: jax.Array,
    temb: jax.Array,
    rotary_emb: jax.Array
    ):
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
      (self.scale_shift_table + temb.astype(jnp.float32)), 6, axis=1
    )
    

    # 1. Self-attention
    norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(hidden_states.dtype)
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
    hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32))
    attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(hidden_states.dtype)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(hidden_states.dtype)
    return hidden_states
  

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

    # 3. Transformer blocks
    blocks = []
    for _ in range(num_layers):
      block = WanTransformerBlock(
        rngs=rngs,
        dim=inner_dim,
        ffn_dim=ffn_dim,
        num_heads=num_attention_heads,
        qk_norm=qk_norm,
        cross_attn_norm=cross_attn_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention=attention
      )
      blocks.append(block)
    self.blocks = blocks

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
      rngs=rngs,
      in_features=inner_dim,
      out_features=out_channels * math.prod(patch_size),
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", "mlp",)),
      bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5)

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
    timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_hidden_states_image is not None:
      raise NotImplementedError("img2vid is not yet implemented.")
    for block in self.blocks:
      hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)

    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    hidden_states = self.proj_out(hidden_states)

    # TODO - can this reshape happen in a single command?
    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
    hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, num_channels)
    # jax.debug.print("FINAL hidden_states min: {x}", x=hidden_states.min())
    # jax.debug.print("FINAL hidden_states max: {x}", x=hidden_states.max())
    return hidden_states