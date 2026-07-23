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

from typing import Optional, Tuple, Any, Dict
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn

from maxdiffusion.models.ltx2.attention_ltx2 import LTX2Attention, LTX2RotaryPosEmbed
from maxdiffusion.models.attention_flax import NNXSimpleFeedForward
from maxdiffusion.models.embeddings_flax import NNXPixArtAlphaCombinedTimestepSizeEmbeddings, NNXPixArtAlphaTextProjection
from maxdiffusion.models.gradient_checkpoint import GradientCheckpointType
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.common_types import BlockSizes
from .logical_sharding_ltx2 import get_sharding_specs, LTX2DiTShardingSpecs
from flax import struct


@struct.dataclass
class LTX2BlockContext:
  hidden_states: jax.Array
  audio_hidden_states: jax.Array
  encoder_hidden_states: jax.Array
  audio_encoder_hidden_states: jax.Array
  temb: jax.Array
  temb_audio: jax.Array
  temb_ca_scale_shift: jax.Array
  temb_ca_audio_scale_shift: jax.Array
  temb_ca_gate: jax.Array
  temb_ca_audio_gate: jax.Array
  temb_prompt: Optional[jax.Array] = None
  temb_prompt_audio: Optional[jax.Array] = None
  modality_mask: Optional[jax.Array] = None
  video_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None
  audio_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None
  ca_video_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None
  ca_audio_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None
  encoder_attention_mask: Optional[jax.Array] = None
  audio_encoder_attention_mask: Optional[jax.Array] = None
  a2v_cross_attention_mask: Optional[jax.Array] = None
  v2a_cross_attention_mask: Optional[jax.Array] = None
  perturbation_mask: Optional[jax.Array] = None


class LTX2AdaLayerNormSingle(nnx.Module):
  """
  Adaptive Layer Normalization (AdaLN) single modulation module for LTX-Video/LTX-2.

  This module is a key component of LTX-2.3, deriving scale and shift parameter modulation
  tensors directly from the continuous noise level (sigma / timestep) through a combination of
  sinusoidal/combined embeddings and feedforward linear projection.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      embedding_dim: int,
      num_mod_params: int = 6,
      use_additional_conditions: bool = False,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      sharding_specs: Optional[LTX2DiTShardingSpecs] = None,
  ):
    self.num_mod_params = num_mod_params

    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "ltx2_dit")
    self.use_additional_conditions = use_additional_conditions
    self.emb = NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
        rngs=rngs,
        embedding_dim=embedding_dim,
        size_emb_dim=embedding_dim // 3,
        use_additional_conditions=use_additional_conditions,
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=sharding_specs,
    )
    self.silu = nnx.silu
    self.linear = nnx.Linear(
        rngs=rngs,
        in_features=embedding_dim,
        out_features=num_mod_params * embedding_dim,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.zeros, sharding_specs.adaln_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, sharding_specs.adaln_bias),
    )

  def __call__(
      self,
      timestep: jax.Array,
      added_cond_kwargs: Optional[Dict[str, jax.Array]] = None,
      batch_size: Optional[int] = None,  # Unused in JAX path usually inferred
      hidden_dtype: Optional[jnp.dtype] = None,
  ) -> Tuple[jax.Array, jax.Array]:
    resolution = None
    aspect_ratio = None
    if self.use_additional_conditions:
      if added_cond_kwargs is None:
        raise ValueError("added_cond_kwargs must be provided when use_additional_conditions is True")
      resolution = added_cond_kwargs.get("resolution", None)
      aspect_ratio = added_cond_kwargs.get("aspect_ratio", None)

    embedded_timestep = self.emb(timestep, resolution=resolution, aspect_ratio=aspect_ratio, hidden_dtype=hidden_dtype)
    return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTX2VideoTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      cross_attention_dim: int,
      audio_dim: int,
      audio_num_attention_heads: int,
      audio_attention_head_dim: int,
      audio_cross_attention_dim: int,
      activation_fn: str = "gelu",
      attention_bias: bool = True,
      attention_out_bias: bool = True,
      norm_elementwise_affine: bool = False,
      norm_eps: float = 1e-6,
      rope_type: str = "interleaved",
      gated_attn: bool = False,
      cross_attn_mod: bool = False,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      mesh: jax.sharding.Mesh = None,
      remat_policy: str = "None",
      precision: jax.lax.Precision = None,
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      attention_kernel: str = "flash",
      a2v_attention_kernel: str = "flash",
      v2a_attention_kernel: str = "dot_product",
      flash_block_sizes: BlockSizes = None,
      flash_min_seq_length: int = 4096,
      sharding_specs: Optional[LTX2DiTShardingSpecs] = None,
      perturbed_attn: bool = False,
      ulysses_shards: int = -1,
      ulysses_attention_chunks: int = 1,
  ):
    self.dim = dim
    self.norm_eps = norm_eps
    self.norm_elementwise_affine = norm_elementwise_affine
    self.attention_kernel = attention_kernel
    self.perturbed_attn = perturbed_attn

    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "ltx2_dit")
    self.sharding_specs = sharding_specs

    # 1. Self-Attention (video and audio)
    self.norm1 = nnx.RMSNorm(
        self.dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.attn1 = LTX2Attention(
        rngs=rngs,
        query_dim=self.dim,
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=self.attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    self.audio_norm1 = nnx.RMSNorm(
        audio_dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.audio_attn1 = LTX2Attention(
        rngs=rngs,
        query_dim=audio_dim,
        heads=audio_num_attention_heads,
        dim_head=audio_attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=self.attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    # 2. Prompt Cross-Attention
    self.norm2 = nnx.RMSNorm(
        self.dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.attn2 = LTX2Attention(
        rngs=rngs,
        query_dim=dim,
        context_dim=cross_attention_dim,
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=self.attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    self.audio_norm2 = nnx.RMSNorm(
        audio_dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.audio_attn2 = LTX2Attention(
        rngs=rngs,
        query_dim=audio_dim,
        context_dim=audio_cross_attention_dim,
        heads=audio_num_attention_heads,
        dim_head=audio_attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=self.attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
    self.audio_to_video_norm = nnx.RMSNorm(
        dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.audio_to_video_attn = LTX2Attention(
        rngs=rngs,
        query_dim=dim,
        context_dim=audio_dim,
        heads=audio_num_attention_heads,
        dim_head=audio_attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=a2v_attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    self.video_to_audio_norm = nnx.RMSNorm(
        audio_dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.video_to_audio_attn = LTX2Attention(
        rngs=rngs,
        query_dim=audio_dim,
        context_dim=dim,
        heads=audio_num_attention_heads,
        dim_head=audio_attention_head_dim,
        dropout=0.0,
        bias=attention_bias,
        out_bias=attention_out_bias,
        eps=norm_eps,
        dtype=dtype,
        mesh=mesh,
        attention_kernel=v2a_attention_kernel,
        rope_type=rope_type,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        sharding_specs=self.sharding_specs,
        gated_attn=gated_attn,
        ulysses_shards=ulysses_shards,
        ulysses_attention_chunks=ulysses_attention_chunks,
    )

    # 4. Feed Forward
    self.norm3 = nnx.RMSNorm(
        dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.ff = NNXSimpleFeedForward(
        rngs=rngs,
        dim=dim,
        dim_out=dim,
        activation_fn=activation_fn,
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=self.sharding_specs,
    )

    self.audio_norm3 = nnx.RMSNorm(
        audio_dim,
        epsilon=self.norm_eps,
        use_scale=self.norm_elementwise_affine,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale),
    )
    self.audio_ff = NNXSimpleFeedForward(
        rngs=rngs,
        dim=audio_dim,
        dim_out=audio_dim,
        activation_fn=activation_fn,
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=self.sharding_specs,
    )

    key = rngs.params()
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    self.cross_attn_mod = cross_attn_mod
    table_size = 9 if cross_attn_mod else 6
    table_sharding = self.sharding_specs.scale_shift_table

    self.scale_shift_table = nnx.Param(
        nnx.with_partitioning(
            lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype) / jnp.sqrt(self.dim), table_sharding
        )(k1, (table_size, self.dim))
    )

    if self.cross_attn_mod:
      self.prompt_scale_shift_table = nnx.Param(
          nnx.with_partitioning(
              lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype) / jnp.sqrt(self.dim), table_sharding
          )(k5, (2, self.dim))
      )

    self.audio_scale_shift_table = nnx.Param(
        nnx.with_partitioning(
            lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype) / jnp.sqrt(audio_dim), table_sharding
        )(k2, (table_size, audio_dim))
    )

    self.video_a2v_cross_attn_scale_shift_table = nnx.Param(
        nnx.with_partitioning(lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype), table_sharding)(
            k3, (5, self.dim)
        )
    )

    self.audio_a2v_cross_attn_scale_shift_table = nnx.Param(
        nnx.with_partitioning(lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype), table_sharding)(
            k4, (5, audio_dim)
        )
    )

    if self.cross_attn_mod:
      self.audio_prompt_scale_shift_table = nnx.Param(
          nnx.with_partitioning(
              lambda key, shape: jax.random.normal(key, shape, dtype=weights_dtype) / jnp.sqrt(audio_dim), table_sharding
          )(k6, (2, audio_dim))
      )

  def __call__(
      self,
      ctx: "LTX2BlockContext",
  ) -> Tuple[jax.Array, jax.Array]:
    """
    Forward pass of the LTX2 video/audio transformer block.

    This block handles complex multi-modal attention including:
      - Video Self-Attention (video -> video)
      - Audio Self-Attention (audio -> audio)
      - Video Cross-Attention (video -> text caption)
      - Audio Cross-Attention (audio -> text caption)
      - Video-to-Audio Cross-Attention
      - Audio-to-Video Cross-Attention

    Args:
      ctx: An `LTX2BlockContext` object containing all hidden states, timestep
           embeddings, attention masks, rotary embeddings, and modulation
           parameters needed for this layer's forward pass.

    Returns:
      A tuple of `(output_hidden_states, output_audio_hidden_states)`.
    """
    hidden_states = ctx.hidden_states
    audio_hidden_states = ctx.audio_hidden_states
    encoder_hidden_states = ctx.encoder_hidden_states
    audio_encoder_hidden_states = ctx.audio_encoder_hidden_states
    temb = ctx.temb
    temb_audio = ctx.temb_audio
    temb_ca_scale_shift = ctx.temb_ca_scale_shift
    temb_ca_audio_scale_shift = ctx.temb_ca_audio_scale_shift
    temb_ca_gate = ctx.temb_ca_gate
    temb_ca_audio_gate = ctx.temb_ca_audio_gate
    temb_prompt = ctx.temb_prompt
    temb_prompt_audio = ctx.temb_prompt_audio
    modality_mask = ctx.modality_mask
    video_rotary_emb = ctx.video_rotary_emb
    audio_rotary_emb = ctx.audio_rotary_emb
    ca_video_rotary_emb = ctx.ca_video_rotary_emb
    ca_audio_rotary_emb = ctx.ca_audio_rotary_emb
    encoder_attention_mask = ctx.encoder_attention_mask
    audio_encoder_attention_mask = ctx.audio_encoder_attention_mask
    a2v_cross_attention_mask = ctx.a2v_cross_attention_mask
    v2a_cross_attention_mask = ctx.v2a_cross_attention_mask
    perturbation_mask = ctx.perturbation_mask
    batch_size = hidden_states.shape[0]

    axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_embed"))
    hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
    axis_names_audio = nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed"))
    audio_hidden_states = jax.lax.with_sharding_constraint(audio_hidden_states, axis_names_audio)

    if encoder_hidden_states is not None:
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)
    if audio_encoder_hidden_states is not None:
      audio_encoder_hidden_states = jax.lax.with_sharding_constraint(audio_encoder_hidden_states, axis_names_audio)

    # 1. Video and Audio Self-Attention
    norm_hidden_states = self.norm1(hidden_states)

    # Calculate Video AdaLN values
    num_ada_params = self.scale_shift_table.shape[0]
    # table shape: (6, dim) -> (1, 1, 6, dim)
    scale_shift_table_reshaped = jnp.expand_dims(self.scale_shift_table, axis=(0, 1))
    # temb shape: (batch, temb_dim) -> (batch, 1, 6, dim)
    temb_reshaped = temb.reshape(batch_size, 1, num_ada_params, -1)
    ada_values = scale_shift_table_reshaped + temb_reshaped

    # Diffusers Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    shift_msa = ada_values[:, :, 0, :]
    scale_msa = ada_values[:, :, 1, :]
    gate_msa = ada_values[:, :, 2, :]
    shift_mlp = ada_values[:, :, 3, :]
    scale_mlp = ada_values[:, :, 4, :]
    gate_mlp = ada_values[:, :, 5, :]

    if getattr(self, "cross_attn_mod", False):
      shift_q = ada_values[:, :, 6, :]
      scale_q = ada_values[:, :, 7, :]
      gate_q = ada_values[:, :, 8, :]

    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    with jax.named_scope("Video Self-Attention"):
      attn_hidden_states = self.attn1(
          hidden_states=norm_hidden_states,
          encoder_hidden_states=None,
          rotary_emb=video_rotary_emb,
          perturbation_mask=perturbation_mask,
      )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    # Calculate Audio AdaLN values
    norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)

    num_audio_ada_params = self.audio_scale_shift_table.shape[0]
    audio_scale_shift_table_reshaped = jnp.expand_dims(self.audio_scale_shift_table, axis=(0, 1))
    temb_audio_reshaped = temb_audio.reshape(batch_size, 1, num_audio_ada_params, -1)
    audio_ada_values = audio_scale_shift_table_reshaped + temb_audio_reshaped

    audio_shift_msa = audio_ada_values[:, :, 0, :]
    audio_scale_msa = audio_ada_values[:, :, 1, :]
    audio_gate_msa = audio_ada_values[:, :, 2, :]
    audio_shift_mlp = audio_ada_values[:, :, 3, :]
    audio_scale_mlp = audio_ada_values[:, :, 4, :]
    audio_gate_mlp = audio_ada_values[:, :, 5, :]

    if getattr(self, "cross_attn_mod", False):
      audio_shift_q = audio_ada_values[:, :, 6, :]
      audio_scale_q = audio_ada_values[:, :, 7, :]
      audio_gate_q = audio_ada_values[:, :, 8, :]

    norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa

    with jax.named_scope("Audio Self-Attention"):
      attn_audio_hidden_states = self.audio_attn1(
          hidden_states=norm_audio_hidden_states,
          encoder_hidden_states=None,
          rotary_emb=audio_rotary_emb,
          perturbation_mask=perturbation_mask,
      )
    audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

    # 2. Video and Audio Cross-Attention with the text embeddings
    norm_hidden_states = self.norm2(hidden_states)
    if getattr(self, "cross_attn_mod", False):
      norm_hidden_states = norm_hidden_states * (1 + scale_q) + shift_q

    if getattr(self, "cross_attn_mod", False) and temb_prompt is not None:
      prompt_table_reshaped = jnp.expand_dims(self.prompt_scale_shift_table, axis=(0, 1))
      temb_prompt_reshaped = temb_prompt.reshape(batch_size, 1, 2, -1)
      prompt_ada_values = prompt_table_reshaped + temb_prompt_reshaped
      shift_text_kv = prompt_ada_values[:, :, 0, :]
      scale_text_kv = prompt_ada_values[:, :, 1, :]
      encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv

    attn_hidden_states = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    if getattr(self, "cross_attn_mod", False):
      attn_hidden_states = attn_hidden_states * gate_q
    hidden_states = hidden_states + attn_hidden_states

    norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
    if getattr(self, "cross_attn_mod", False):
      norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_q) + audio_shift_q

    if getattr(self, "cross_attn_mod", False) and temb_prompt_audio is not None:
      audio_prompt_table_reshaped = jnp.expand_dims(self.audio_prompt_scale_shift_table, axis=(0, 1))
      temb_prompt_audio_reshaped = temb_prompt_audio.reshape(batch_size, 1, 2, -1)
      audio_prompt_ada_values = audio_prompt_table_reshaped + temb_prompt_audio_reshaped
      audio_shift_text_kv = audio_prompt_ada_values[:, :, 0, :]
      audio_scale_text_kv = audio_prompt_ada_values[:, :, 1, :]
      audio_encoder_hidden_states = audio_encoder_hidden_states * (1 + audio_scale_text_kv) + audio_shift_text_kv

    attn_audio_hidden_states = self.audio_attn2(
        norm_audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        rotary_emb=None,
        attention_mask=audio_encoder_attention_mask,
    )
    if getattr(self, "cross_attn_mod", False):
      attn_audio_hidden_states = attn_audio_hidden_states * audio_gate_q
    audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

    # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
    norm_hidden_states = self.audio_to_video_norm(hidden_states)
    norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)

    # Calculate Cross-Attention Modulation values
    # Video
    video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
    video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]

    video_ca_scale_shift_table = jnp.expand_dims(video_per_layer_ca_scale_shift, axis=(0, 1)) + temb_ca_scale_shift.reshape(
        batch_size, 1, 4, -1
    )

    video_a2v_ca_scale = video_ca_scale_shift_table[:, :, 0, :]
    video_a2v_ca_shift = video_ca_scale_shift_table[:, :, 1, :]
    video_v2a_ca_scale = video_ca_scale_shift_table[:, :, 2, :]
    video_v2a_ca_shift = video_ca_scale_shift_table[:, :, 3, :]

    a2v_gate = (jnp.expand_dims(video_per_layer_ca_gate, axis=(0, 1)) + temb_ca_gate.reshape(batch_size, 1, 1, -1))[
        :, :, 0, :
    ]

    # Audio
    audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
    audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]

    audio_ca_scale_shift_table = jnp.expand_dims(
        audio_per_layer_ca_scale_shift, axis=(0, 1)
    ) + temb_ca_audio_scale_shift.reshape(batch_size, 1, 4, -1)

    audio_a2v_ca_scale = audio_ca_scale_shift_table[:, :, 0, :]
    audio_a2v_ca_shift = audio_ca_scale_shift_table[:, :, 1, :]
    audio_v2a_ca_scale = audio_ca_scale_shift_table[:, :, 2, :]
    audio_v2a_ca_shift = audio_ca_scale_shift_table[:, :, 3, :]

    v2a_gate = (jnp.expand_dims(audio_per_layer_ca_gate, axis=(0, 1)) + temb_ca_audio_gate.reshape(batch_size, 1, 1, -1))[
        :, :, 0, :
    ]

    # Audio-to-Video Cross Attention: Q: Video; K,V: Audio
    mod_norm_hidden_states = norm_hidden_states * (1 + video_a2v_ca_scale) + video_a2v_ca_shift
    mod_norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_a2v_ca_scale) + audio_a2v_ca_shift

    with jax.named_scope("Audio-to-Video Cross-Attention"):
      a2v_attn_hidden_states = self.audio_to_video_attn(
          mod_norm_hidden_states,
          encoder_hidden_states=mod_norm_audio_hidden_states,
          rotary_emb=ca_video_rotary_emb,
          k_rotary_emb=ca_audio_rotary_emb,
          attention_mask=a2v_cross_attention_mask,
      )
    if modality_mask is not None:
      a2v_attn_hidden_states = a2v_attn_hidden_states * modality_mask
    hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

    # Video-to-Audio Cross Attention: Q: Audio; K,V: Video
    mod_norm_hidden_states_v2a = norm_hidden_states * (1 + video_v2a_ca_scale) + video_v2a_ca_shift
    mod_norm_audio_hidden_states_v2a = norm_audio_hidden_states * (1 + audio_v2a_ca_scale) + audio_v2a_ca_shift

    with jax.named_scope("Video-to-Audio Cross-Attention"):
      v2a_attn_hidden_states = self.video_to_audio_attn(
          mod_norm_audio_hidden_states_v2a,
          encoder_hidden_states=mod_norm_hidden_states_v2a,
          rotary_emb=ca_audio_rotary_emb,
          k_rotary_emb=ca_video_rotary_emb,
          attention_mask=v2a_cross_attention_mask,
      )
    if modality_mask is not None:
      v2a_attn_hidden_states = v2a_attn_hidden_states * modality_mask
    audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states

    # 4. Feedforward
    norm_hidden_states = self.norm3(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    ff_output = self.ff(norm_hidden_states)
    hidden_states = hidden_states + ff_output * gate_mlp

    norm_audio_hidden_states = self.audio_norm3(audio_hidden_states)
    norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_mlp) + audio_shift_mlp
    audio_ff_output = self.audio_ff(norm_audio_hidden_states)
    audio_hidden_states = audio_hidden_states + audio_ff_output * audio_gate_mlp

    return hidden_states, audio_hidden_states


class LTX2VideoTransformer3DModel(nnx.Module, ConfigMixin):
  config_name = "config.json"

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int = 128,  # Video Arguments
      out_channels: Optional[int] = 128,
      patch_size: int = 1,
      patch_size_t: int = 1,
      num_attention_heads: int = 32,
      attention_head_dim: int = 128,
      cross_attention_dim: int = 4096,
      vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
      pos_embed_max_pos: int = 20,
      base_height: int = 2048,
      base_width: int = 2048,
      audio_in_channels: int = 128,  # Audio Arguments
      audio_out_channels: Optional[int] = 128,
      audio_patch_size: int = 1,
      audio_patch_size_t: int = 1,
      audio_num_attention_heads: int = 32,
      audio_attention_head_dim: int = 64,
      audio_cross_attention_dim: int = 2048,
      audio_scale_factor: int = 4,
      audio_pos_embed_max_pos: int = 20,
      audio_sampling_rate: int = 16000,
      audio_hop_length: int = 160,
      num_layers: int = 48,  # Shared arguments
      activation_fn: str = "gelu",
      norm_elementwise_affine: bool = False,
      norm_eps: float = 1e-6,
      caption_channels: int = 3840,
      audio_caption_channels: int = None,
      attention_bias: bool = True,
      attention_out_bias: bool = True,
      rope_theta: float = 10000.0,
      rope_double_precision: bool = True,
      causal_offset: int = 1,
      timestep_scale_multiplier: int = 1000,
      cross_attn_timestep_scale_multiplier: int = 1000,
      rope_type: str = "interleaved",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      mesh: jax.sharding.Mesh = None,
      remat_policy: str = "None",
      precision: jax.lax.Precision = None,
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      scan_layers: bool = True,
      attention_kernel: str = "flash",
      a2v_attention_kernel: str = "flash",
      v2a_attention_kernel: str = "dot_product",
      qk_norm: str = "rms_norm_across_heads",
      flash_block_sizes: BlockSizes = None,
      flash_min_seq_length: int = 4096,
      sharding_specs: Optional[LTX2DiTShardingSpecs] = None,
      gated_attn: bool = False,
      cross_attn_mod: bool = False,
      use_prompt_embeddings: bool = True,
      perturbed_attn: bool = False,
      spatio_temporal_guidance_blocks: Tuple[int, ...] = (),
      ulysses_shards: int = -1,
      ulysses_attention_chunks: int = 1,
      **kwargs,
  ):
    self.spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    self.cross_attention_dim = cross_attention_dim
    self.vae_scale_factors = vae_scale_factors
    self.pos_embed_max_pos = pos_embed_max_pos
    self.base_height = base_height
    self.base_width = base_width
    self.audio_in_channels = audio_in_channels
    self.audio_out_channels = audio_out_channels
    self.audio_patch_size = audio_patch_size
    self.audio_patch_size_t = audio_patch_size_t
    self.audio_num_attention_heads = audio_num_attention_heads
    self.audio_attention_head_dim = audio_attention_head_dim
    self.audio_cross_attention_dim = audio_cross_attention_dim
    self.audio_scale_factor = audio_scale_factor
    self.audio_pos_embed_max_pos = audio_pos_embed_max_pos
    self.audio_sampling_rate = audio_sampling_rate
    self.audio_hop_length = audio_hop_length
    self.num_layers = num_layers
    self.activation_fn = activation_fn
    self.norm_elementwise_affine = norm_elementwise_affine
    self.norm_eps = norm_eps
    self.caption_channels = caption_channels
    self.audio_caption_channels = audio_caption_channels or caption_channels
    self.attention_bias = attention_bias
    self.attention_out_bias = attention_out_bias
    self.rope_theta = rope_theta
    self.rope_double_precision = rope_double_precision
    self.use_prompt_embeddings = use_prompt_embeddings
    self.causal_offset = causal_offset
    self.timestep_scale_multiplier = timestep_scale_multiplier
    self.cross_attn_timestep_scale_multiplier = cross_attn_timestep_scale_multiplier
    self.rope_type = rope_type
    self.dtype = dtype
    self.weights_dtype = weights_dtype
    self.mesh = mesh
    self.remat_policy = remat_policy
    self.precision = precision
    self.names_which_can_be_saved = names_which_can_be_saved
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.scan_layers = scan_layers
    self.attention_kernel = attention_kernel
    self.gated_attn = gated_attn
    self.cross_attn_mod = cross_attn_mod
    self.perturbed_attn = perturbed_attn
    self.a2v_attention_kernel = a2v_attention_kernel
    self.v2a_attention_kernel = v2a_attention_kernel
    self.flash_min_seq_length = flash_min_seq_length

    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "ltx2_dit")
    self.sharding_specs = sharding_specs

    _out_channels = self.out_channels or self.in_channels
    _audio_out_channels = self.audio_out_channels or self.audio_in_channels
    inner_dim = self.num_attention_heads * self.attention_head_dim
    audio_inner_dim = self.audio_num_attention_heads * self.audio_attention_head_dim

    # 1. Patchification input projections
    self.proj_in = nnx.Linear(
        self.in_channels,
        inner_dim,
        rngs=rngs,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), self.sharding_specs.embed_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, self.sharding_specs.embed_bias),
    )
    self.audio_proj_in = nnx.Linear(
        self.audio_in_channels,
        audio_inner_dim,
        rngs=rngs,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), self.sharding_specs.embed_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, self.sharding_specs.embed_bias),
    )

    if self.use_prompt_embeddings:
      self.caption_projection = NNXPixArtAlphaTextProjection(
          rngs=rngs,
          in_features=self.caption_channels,
          hidden_size=inner_dim,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          sharding_specs=self.sharding_specs,
      )
      self.audio_caption_projection = NNXPixArtAlphaTextProjection(
          rngs=rngs,
          in_features=self.caption_channels,
          hidden_size=audio_inner_dim,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          sharding_specs=self.sharding_specs,
      )
    else:
      self.caption_projection = None
      self.audio_caption_projection = None

    if self.cross_attn_mod:
      self.prompt_adaln = LTX2AdaLayerNormSingle(
          rngs=rngs,
          embedding_dim=inner_dim,
          num_mod_params=2,
          use_additional_conditions=False,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          sharding_specs=self.sharding_specs,
      )
      self.audio_prompt_adaln = LTX2AdaLayerNormSingle(
          rngs=rngs,
          embedding_dim=audio_inner_dim,
          num_mod_params=2,
          use_additional_conditions=False,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          sharding_specs=self.sharding_specs,
      )
    # 3. Timestep Modulation Params and Embedding
    num_mod_params = 9 if self.cross_attn_mod else 6
    self.time_embed = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=inner_dim,
        num_mod_params=num_mod_params,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.audio_time_embed = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=audio_inner_dim,
        num_mod_params=num_mod_params,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=inner_dim,
        num_mod_params=4,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=audio_inner_dim,
        num_mod_params=4,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=inner_dim,
        num_mod_params=1,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
        rngs=rngs,
        embedding_dim=audio_inner_dim,
        num_mod_params=1,
        use_additional_conditions=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        sharding_specs=self.sharding_specs,
    )

    # 3. Output Layer Scale/Shift Modulation parameters
    param_rng = rngs.params()
    audio_param_rng = rngs.params()
    table_sharding = self.sharding_specs.scale_shift_table
    self.scale_shift_table = nnx.Param(
        nnx.with_partitioning(
            lambda key, shape: jax.random.normal(key, shape, dtype=self.weights_dtype) / jnp.sqrt(inner_dim), table_sharding
        )(param_rng, (2, inner_dim))
    )
    self.audio_scale_shift_table = nnx.Param(
        nnx.with_partitioning(
            lambda key, shape: jax.random.normal(key, shape, dtype=self.weights_dtype) / jnp.sqrt(audio_inner_dim),
            table_sharding,
        )(audio_param_rng, (2, audio_inner_dim))
    )

    # 4. Rotary Positional Embeddings (RoPE)
    self.rope = LTX2RotaryPosEmbed(
        dim=inner_dim,
        patch_size=self.patch_size,
        patch_size_t=self.patch_size_t,
        base_num_frames=self.pos_embed_max_pos,
        base_height=self.base_height,
        base_width=self.base_width,
        scale_factors=self.vae_scale_factors,
        theta=self.rope_theta,
        causal_offset=self.causal_offset,
        modality="video",
        double_precision=self.rope_double_precision,
        rope_type=self.rope_type,
        num_attention_heads=self.num_attention_heads,
    )
    self.audio_rope = LTX2RotaryPosEmbed(
        dim=audio_inner_dim,
        patch_size=self.audio_patch_size,
        patch_size_t=self.audio_patch_size_t,
        base_num_frames=self.audio_pos_embed_max_pos,
        sampling_rate=self.audio_sampling_rate,
        hop_length=self.audio_hop_length,
        scale_factors=[self.audio_scale_factor],
        theta=self.rope_theta,
        causal_offset=self.causal_offset,
        modality="audio",
        double_precision=self.rope_double_precision,
        rope_type=self.rope_type,
        num_attention_heads=self.audio_num_attention_heads,
    )

    cross_attn_pos_embed_max_pos = max(self.pos_embed_max_pos, self.audio_pos_embed_max_pos)
    self.cross_attn_rope = LTX2RotaryPosEmbed(
        dim=audio_cross_attention_dim,
        patch_size=self.patch_size,
        patch_size_t=self.patch_size_t,
        base_num_frames=cross_attn_pos_embed_max_pos,
        base_height=self.base_height,
        base_width=self.base_width,
        theta=self.rope_theta,
        causal_offset=self.causal_offset,
        modality="video",
        double_precision=self.rope_double_precision,
        rope_type=self.rope_type,
        num_attention_heads=self.num_attention_heads,
    )
    self.cross_attn_audio_rope = LTX2RotaryPosEmbed(
        dim=audio_cross_attention_dim,
        patch_size=self.audio_patch_size,
        patch_size_t=self.audio_patch_size_t,
        base_num_frames=cross_attn_pos_embed_max_pos,
        sampling_rate=self.audio_sampling_rate,
        hop_length=self.audio_hop_length,
        theta=self.rope_theta,
        causal_offset=self.causal_offset,
        modality="audio",
        double_precision=self.rope_double_precision,
        rope_type=self.rope_type,
        num_attention_heads=self.audio_num_attention_heads,
    )

    # 5. Transformer Blocks
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.vmap(in_axes=0, out_axes=0, axis_size=self.num_layers, transform_metadata={nnx.PARTITION_NAME: "layers"})
    def init_block(rngs):
      return LTX2VideoTransformerBlock(
          rngs=rngs,
          sharding_specs=self.sharding_specs,
          dim=inner_dim,
          num_attention_heads=self.num_attention_heads,
          attention_head_dim=self.attention_head_dim,
          cross_attention_dim=inner_dim,
          audio_dim=audio_inner_dim,
          audio_num_attention_heads=self.audio_num_attention_heads,
          audio_attention_head_dim=self.audio_attention_head_dim,
          audio_cross_attention_dim=audio_inner_dim,
          activation_fn=self.activation_fn,
          attention_bias=self.attention_bias,
          attention_out_bias=self.attention_out_bias,
          norm_elementwise_affine=self.norm_elementwise_affine,
          norm_eps=self.norm_eps,
          rope_type=self.rope_type,
          gated_attn=self.gated_attn,
          cross_attn_mod=self.cross_attn_mod,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          mesh=self.mesh,
          remat_policy=self.remat_policy,
          precision=self.precision,
          names_which_can_be_saved=self.names_which_can_be_saved,
          names_which_can_be_offloaded=self.names_which_can_be_offloaded,
          attention_kernel=self.attention_kernel,
          a2v_attention_kernel=self.a2v_attention_kernel,
          v2a_attention_kernel=self.v2a_attention_kernel,
          flash_block_sizes=flash_block_sizes,
          flash_min_seq_length=self.flash_min_seq_length,
          perturbed_attn=self.perturbed_attn,
          ulysses_shards=ulysses_shards,
          ulysses_attention_chunks=ulysses_attention_chunks,
      )

    if self.scan_layers:
      self.transformer_blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(self.num_layers):
        block = LTX2VideoTransformerBlock(
            rngs=rngs,
            sharding_specs=self.sharding_specs,
            dim=inner_dim,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            cross_attention_dim=inner_dim,
            audio_dim=audio_inner_dim,
            audio_num_attention_heads=self.audio_num_attention_heads,
            audio_attention_head_dim=self.audio_attention_head_dim,
            audio_cross_attention_dim=audio_inner_dim,
            activation_fn=self.activation_fn,
            attention_bias=self.attention_bias,
            attention_out_bias=self.attention_out_bias,
            norm_elementwise_affine=self.norm_elementwise_affine,
            norm_eps=self.norm_eps,
            rope_type=self.rope_type,
            gated_attn=self.gated_attn,
            cross_attn_mod=self.cross_attn_mod,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            mesh=self.mesh,
            remat_policy=self.remat_policy,
            precision=self.precision,
            names_which_can_be_saved=self.names_which_can_be_saved,
            names_which_can_be_offloaded=self.names_which_can_be_offloaded,
            attention_kernel=self.attention_kernel,
            a2v_attention_kernel=self.a2v_attention_kernel,
            v2a_attention_kernel=self.v2a_attention_kernel,
            flash_block_sizes=flash_block_sizes,
            flash_min_seq_length=self.flash_min_seq_length,
            perturbed_attn=self.perturbed_attn,
            ulysses_shards=ulysses_shards,
            ulysses_attention_chunks=ulysses_attention_chunks,
        )
        blocks.append(block)
      self.transformer_blocks = nnx.List(blocks)

    # 6. Output layers
    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.norm_out = nnx.LayerNorm(
        inner_dim, epsilon=1e-6, use_scale=False, use_bias=False, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32
    )
    self.proj_out = nnx.Linear(
        inner_dim,
        _out_channels,
        rngs=rngs,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), self.sharding_specs.out_embed_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, self.sharding_specs.out_embed_bias),
    )

    self.audio_norm_out = nnx.LayerNorm(
        audio_inner_dim, epsilon=1e-6, use_scale=False, use_bias=False, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32
    )
    self.audio_proj_out = nnx.Linear(
        audio_inner_dim,
        _audio_out_channels,
        rngs=rngs,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), self.sharding_specs.out_embed_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, self.sharding_specs.out_embed_bias),
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      audio_hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      audio_encoder_hidden_states: jax.Array,
      timestep: jax.Array,
      audio_timestep: Optional[jax.Array] = None,
      sigma: Optional[jax.Array] = None,
      audio_sigma: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      audio_encoder_attention_mask: Optional[jax.Array] = None,
      num_frames: Optional[int] = None,
      height: Optional[int] = None,
      width: Optional[int] = None,
      fps: float = 24.0,
      audio_num_frames: Optional[int] = None,
      video_coords: Optional[jax.Array] = None,
      audio_coords: Optional[jax.Array] = None,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      use_cross_timestep: bool = False,
      modality_mask: Optional[jax.Array] = None,
      return_dict: bool = True,
      perturbation_mask: Optional[jax.Array] = None,
  ) -> Any:
    """
    Forward pass for the full LTX2 Video/Audio Diffusion Transformer.

    Args:
      hidden_states: Video latent patches of shape `(batch, seq_len, in_channels)`.
      audio_hidden_states: Audio latent patches of shape `(batch, audio_seq_len, audio_in_channels)`.
      encoder_hidden_states: Text embeddings for video generation.
      audio_encoder_hidden_states: Text embeddings for audio generation.
      timestep: Timestep array for video diffusion.
      audio_timestep: Optional timestep array for audio diffusion. If None, uses `timestep`.
      sigma: Optional noise scale for video (for flow matching).
      audio_sigma: Optional noise scale for audio.
      encoder_attention_mask: Mask for video text embeddings.
      audio_encoder_attention_mask: Mask for audio text embeddings.
      num_frames: Number of video frames.
      height: Height of the video frames.
      width: Width of the video frames.
      fps: Frames per second.
      audio_num_frames: Number of audio frames.
      video_coords: Optional pre-computed 3D coordinates for video RoPE.
      audio_coords: Optional pre-computed 1D coordinates for audio RoPE.
      attention_kwargs: Additional kwargs for the attention mechanisms.
      use_cross_timestep: Whether to use a cross-modal timestep interaction.
      modality_mask: Mask indicating which modality to drop/keep.
      return_dict: If True, returns a dictionary. Otherwise, returns a tuple.
      perturbation_mask: Optional mask for perturbing attention.

    Returns:
      Output dict containing `sample` (video) and `audio_sample` (audio).
    """
    # Determine timestep for audio.
    audio_timestep = audio_timestep if audio_timestep is not None else timestep

    a2v_cross_attention_mask = None
    v2a_cross_attention_mask = None
    if attention_kwargs is not None:
      a2v_cross_attention_mask = attention_kwargs.get("a2v_cross_attention_mask", None)
      v2a_cross_attention_mask = attention_kwargs.get("v2a_cross_attention_mask", None)

    if self.attention_kernel == "dot_product":
      if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.astype(self.dtype)) * -10000.0
        encoder_attention_mask = jnp.expand_dims(encoder_attention_mask, axis=1)

      if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
        audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.astype(self.dtype)) * -10000.0
        audio_encoder_attention_mask = jnp.expand_dims(audio_encoder_attention_mask, axis=1)

      if a2v_cross_attention_mask is not None and a2v_cross_attention_mask.ndim == 2:
        a2v_cross_attention_mask = (1 - a2v_cross_attention_mask.astype(self.dtype)) * -10000.0
        a2v_cross_attention_mask = jnp.expand_dims(a2v_cross_attention_mask, axis=1)

      if v2a_cross_attention_mask is not None and v2a_cross_attention_mask.ndim == 2:
        v2a_cross_attention_mask = (1 - v2a_cross_attention_mask.astype(self.dtype)) * -10000.0
        v2a_cross_attention_mask = jnp.expand_dims(v2a_cross_attention_mask, axis=1)

    batch_size = hidden_states.shape[0]

    # 1. Prepare RoPE positional embeddings
    with jax.named_scope("RoPE Preparation"):
      if video_coords is None:
        video_coords = self.rope.prepare_video_coords(batch_size, num_frames, height, width, fps=fps)
      if audio_coords is None:
        audio_coords = self.audio_rope.prepare_audio_coords(batch_size, audio_num_frames)

      video_rotary_emb = self.rope(video_coords)
      audio_rotary_emb = self.audio_rope(audio_coords)

      video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :])
      audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(audio_coords[:, 0:1, :])

    # 2. Patchify input projections
    with jax.named_scope("Input Projection"):
      hidden_states = self.proj_in(hidden_states)
      audio_hidden_states = self.audio_proj_in(audio_hidden_states)

    # 3. Prepare timestep embeddings and modulation parameters
    with jax.named_scope("Timestep and Caption Projection"):
      timestep_cross_attn_gate_scale_factor = self.cross_attn_timestep_scale_multiplier / self.timestep_scale_multiplier

      temb, embedded_timestep = self.time_embed(
          timestep.flatten(),
          hidden_dtype=hidden_states.dtype,
      )
      temb = temb.reshape(batch_size, -1, temb.shape[-1])
      embedded_timestep = embedded_timestep.reshape(batch_size, -1, embedded_timestep.shape[-1])

      temb_audio, audio_embedded_timestep = self.audio_time_embed(
          audio_timestep.flatten(),
          hidden_dtype=audio_hidden_states.dtype,
      )
      temb_audio = temb_audio.reshape(batch_size, -1, temb_audio.shape[-1])
      audio_embedded_timestep = audio_embedded_timestep.reshape(batch_size, -1, audio_embedded_timestep.shape[-1])

      if self.cross_attn_mod and sigma is not None:
        audio_sigma = audio_sigma if audio_sigma is not None else sigma
        temb_prompt, _ = self.prompt_adaln(
            sigma.flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        temb_prompt_audio, _ = self.audio_prompt_adaln(
            audio_sigma.flatten(),
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_prompt = temb_prompt.reshape(batch_size, -1, temb_prompt.shape[-1])
        temb_prompt_audio = temb_prompt_audio.reshape(batch_size, -1, temb_prompt_audio.shape[-1])
      else:
        temb_prompt = None
        temb_prompt_audio = None

      if use_cross_timestep:
        if sigma is None or audio_sigma is None:
          raise ValueError("sigma and audio_sigma must be provided when use_cross_timestep is True")
        video_ca_timestep = audio_sigma.flatten()
        audio_ca_timestep = sigma.flatten()
      else:
        video_ca_timestep = timestep.flatten()
        audio_ca_timestep = audio_timestep.flatten() if audio_timestep is not None else timestep.flatten()

      video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
          video_ca_timestep,
          hidden_dtype=hidden_states.dtype,
      )
      video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
          video_ca_timestep * timestep_cross_attn_gate_scale_factor,
          hidden_dtype=hidden_states.dtype,
      )
      video_cross_attn_scale_shift = video_cross_attn_scale_shift.reshape(
          batch_size, -1, video_cross_attn_scale_shift.shape[-1]
      )
      video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.reshape(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])

      audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
          audio_ca_timestep,
          hidden_dtype=audio_hidden_states.dtype,
      )
      audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
          audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
          hidden_dtype=audio_hidden_states.dtype,
      )
      audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.reshape(
          batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
      )
      audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.reshape(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])

      if self.use_prompt_embeddings and self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, hidden_states.shape[-1])
        audio_encoder_hidden_states = audio_encoder_hidden_states.reshape(batch_size, -1, audio_hidden_states.shape[-1])
    # 5. Run transformer blocks
    with jax.named_scope("Transformer Blocks"):
      base_context = LTX2BlockContext(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          temb=temb,
          temb_audio=temb_audio,
          temb_ca_scale_shift=video_cross_attn_scale_shift,
          temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
          temb_ca_gate=video_cross_attn_a2v_gate,
          temb_ca_audio_gate=audio_cross_attn_v2a_gate,
          temb_prompt=temb_prompt,
          temb_prompt_audio=temb_prompt_audio,
          video_rotary_emb=video_rotary_emb,
          audio_rotary_emb=audio_rotary_emb,
          ca_video_rotary_emb=video_cross_attn_rotary_emb,
          ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
          encoder_attention_mask=encoder_attention_mask,
          audio_encoder_attention_mask=audio_encoder_attention_mask,
          a2v_cross_attention_mask=a2v_cross_attention_mask,
          v2a_cross_attention_mask=v2a_cross_attention_mask,
          modality_mask=modality_mask,
      )

      def apply_block(block, context: LTX2BlockContext, mask):
        orig_perturbation_mask = context.perturbation_mask
        context = context.replace(perturbation_mask=mask)
        with jax.named_scope("Transformer Layer"):
          hidden_states_out, audio_hidden_states_out = block(context)
        context = context.replace(
            hidden_states=hidden_states_out.astype(context.hidden_states.dtype),
            audio_hidden_states=audio_hidden_states_out.astype(context.audio_hidden_states.dtype),
            perturbation_mask=orig_perturbation_mask,
        )
        return context

      if perturbation_mask is None:

        def scan_fn_ltx2(carry, block):
          context, rngs_carry = carry
          context = apply_block(block, context, None)
          return (context, rngs_carry), None

        if self.scan_layers:
          rematted_scan_fn = self.gradient_checkpoint.apply(
              scan_fn_ltx2,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          carry = (base_context, nnx.Rngs(0))
          (final_context, _), _ = nnx.scan(
              rematted_scan_fn,
              length=self.num_layers,
              in_axes=(nnx.Carry, 0),
              out_axes=(nnx.Carry, 0),
              transform_metadata={nnx.PARTITION_NAME: "layers"},
          )(carry, self.transformer_blocks)
          hidden_states = final_context.hidden_states
          audio_hidden_states = final_context.audio_hidden_states
        else:
          current_context = base_context
          for block in self.transformer_blocks:
            current_context = apply_block(block, current_context, None)
          hidden_states = current_context.hidden_states
          audio_hidden_states = current_context.audio_hidden_states
      else:
        masks = jnp.ones((self.num_layers, batch_size, 1, 1), dtype=self.dtype)
        for i in self.spatio_temporal_guidance_blocks:
          if i < self.num_layers:
            masks = masks.at[i].set(perturbation_mask)
        perturbation_mask_per_layer = masks

        def scan_fn_ltx23(carry, block_and_mask):
          block, mask = block_and_mask
          context, rngs_carry = carry
          context = apply_block(block, context, mask)
          return (context, rngs_carry), None

        if self.scan_layers:
          rematted_scan_fn = self.gradient_checkpoint.apply(
              scan_fn_ltx23,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          carry = (base_context, nnx.Rngs(0))
          (final_context, _), _ = nnx.scan(
              rematted_scan_fn,
              length=self.num_layers,
              in_axes=(nnx.Carry, 0),
              out_axes=(nnx.Carry, 0),
              transform_metadata={nnx.PARTITION_NAME: "layers"},
          )(carry, (self.transformer_blocks, perturbation_mask_per_layer))
          hidden_states = final_context.hidden_states
          audio_hidden_states = final_context.audio_hidden_states
        else:
          current_context = base_context
          for i, block in enumerate(self.transformer_blocks):
            mask = perturbation_mask_per_layer[i] if perturbation_mask_per_layer is not None else None
            current_context = apply_block(block, current_context, mask)
          hidden_states = current_context.hidden_states
          audio_hidden_states = current_context.audio_hidden_states

    # 6. Output layers
    with jax.named_scope("Output Projection & Norm"):
      scale_shift_values = jnp.expand_dims(self.scale_shift_table, axis=(0, 1)) + jnp.expand_dims(embedded_timestep, axis=2)
      shift = scale_shift_values[:, :, 0, :]
      scale = scale_shift_values[:, :, 1, :]

      hidden_states = self.norm_out(hidden_states)
      hidden_states = hidden_states * (1 + scale) + shift
      output = self.proj_out(hidden_states)

      audio_scale_shift_values = jnp.expand_dims(self.audio_scale_shift_table, axis=(0, 1)) + jnp.expand_dims(
          audio_embedded_timestep, axis=2
      )
      audio_shift = audio_scale_shift_values[:, :, 0, :]
      audio_scale = audio_scale_shift_values[:, :, 1, :]

      audio_hidden_states = self.audio_norm_out(audio_hidden_states)
      audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
      audio_output = self.audio_proj_out(audio_hidden_states)

    if not return_dict:
      return (output, audio_output)
    return {"sample": output, "audio_sample": audio_output}
