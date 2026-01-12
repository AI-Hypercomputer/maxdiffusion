"""Copyright 2025 Google LLC

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
from typing import Any, Dict, Optional, Tuple

from flax import nnx
import flax.linen as nn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from .... import common_types
from ....configuration_utils import register_to_config
from ...attention_flax import FlaxWanAttention
from ...gradient_checkpoint import GradientCheckpointType
from ...normalization_flax import FP32LayerNorm
from .transformer_wan import WanFeedForward, WanModel, WanRotaryPosEmbed, WanTimeTextImageEmbedding, WanTransformerBlock


BlockSizes = common_types.BlockSizes


class WanVACETransformerBlock(nnx.Module):
  """Attention block for VACE.

  Processes the conditioning signals and produces latent codes that can be
  summed to the main branch of WAN.

  Based on
  https://github.com/huggingface/diffusers/blob/be3c2a0667493022f17d756ca3dba631d28dfb40/src/diffusers/models/transformers/transformer_wan_vace.py#L41C7-L41C30
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      *,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes | None = None,
      mesh: jax.sharding.Mesh | None = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision | None = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      apply_input_projection: bool = False,
      apply_output_projection: bool = False,
  ):
    """Sets up the model.

    Args:
      rngs: Random number generator.
      dim: Internal dimension of the block.
      ffn_dim: Dimension of the feed-forward network.
      num_heads: Number of attention heads.
      qk_norm: Whether to apply RMSNorm to the query and key vectors.
      cross_attn_norm: Whether to apply layer normalization before
        cross-attention (only True supported).
      eps: Epsilon value for normalization.
      flash_min_seq_length: Minimum sequence length for flash attention.
      flash_block_sizes: Block sizes for flash attention.
      mesh: Sharding topology.
      dtype: Data type for the computation.
      weights_dtype: Data type for parameter initializers (see param_dtype in
        nnx.Linear).
      precision: Precision for the computation.
      attention: Type of attention to use.
      dropout: Dropout rate.
      apply_input_projection: Whether to apply a linear projection to the
        inputs.
      apply_output_projection: Whether to apply an output projection before
        outputting the result.
    """

    self.apply_input_projection = apply_input_projection
    self.apply_output_projection = apply_output_projection

    # 1. Input projection
    self.proj_in = nnx.data([None])
    if apply_input_projection:
      self.proj_in = nnx.Linear(
          rngs=rngs,
          in_features=dim,
          out_features=dim,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.with_partitioning(
              nnx.initializers.xavier_uniform(), ("embed", None)
          ),
      )

    # 2. Self-attention
    self.norm1 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=False
    )
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        residual_checkpoint_name="self_attn",
    )

    # 3. Cross-attention
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        residual_checkpoint_name="cross_attn",
    )
    assert cross_attn_norm is True, "cross_attn_norm must be True"
    self.norm2 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=True
    )

    # 4. Feed-forward
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        dropout=dropout,
    )

    self.norm3 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=False
    )

    # 5. Output projection
    self.proj_out = nnx.data([None])
    if apply_output_projection:
      self.proj_out = nnx.Linear(
          rngs=rngs,
          in_features=dim,
          out_features=dim,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.with_partitioning(
              nnx.initializers.xavier_uniform(), ("embed", None)
          ),
      )

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

  def __call__(
      self,
      *,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      control_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs | None = None,
  ) -> Tuple[jax.Array, jax.Array]:
    if self.apply_input_projection:
      control_hidden_states = self.proj_in(control_hidden_states)
      control_hidden_states = control_hidden_states + hidden_states

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        jnp.split(
            (self.adaln_scale_shift_table + temb.astype(jnp.float32)), 6, axis=1
        )
    )

    control_hidden_states = jax.lax.with_sharding_constraint(
        control_hidden_states,
        PartitionSpec("data", "fsdp", "tensor"),
    )
    control_hidden_states = checkpoint_name(
        control_hidden_states, "control_hidden_states"
    )
    encoder_hidden_states = jax.lax.with_sharding_constraint(
        encoder_hidden_states,
        PartitionSpec("data", "fsdp", None),
    )

    # 1. Self-attention
    with jax.named_scope("attn1"):
      norm_hidden_states = (
          self.norm1(control_hidden_states.astype(jnp.float32))
          * (1 + scale_msa)
          + shift_msa
      ).astype(control_hidden_states.dtype)
      attn_output = self.attn1(
          hidden_states=norm_hidden_states,
          encoder_hidden_states=norm_hidden_states,
          rotary_emb=rotary_emb,
          deterministic=deterministic,
          rngs=rngs,
      )
      control_hidden_states = (
          control_hidden_states.astype(jnp.float32) + attn_output * gate_msa
      ).astype(control_hidden_states.dtype)

    # 2. Cross-attention
    with jax.named_scope("attn2"):
      norm_hidden_states = self.norm2(
          control_hidden_states.astype(jnp.float32)
      ).astype(control_hidden_states.dtype)
      attn_output = self.attn2(
          hidden_states=norm_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          deterministic=deterministic,
          rngs=rngs,
      )
      control_hidden_states = control_hidden_states + attn_output

    # 3. Feed-forward
    with jax.named_scope("ffn"):
      norm_hidden_states = (
          self.norm3(control_hidden_states.astype(jnp.float32))
          * (1 + c_scale_msa)
          + c_shift_msa
      ).astype(control_hidden_states.dtype)
      ff_output = self.ffn(
          norm_hidden_states, deterministic=deterministic, rngs=rngs
      )
      control_hidden_states = (
          control_hidden_states.astype(jnp.float32)
          + ff_output.astype(jnp.float32) * c_gate_msa
      ).astype(control_hidden_states.dtype)
      conditioning_states = None
      if self.apply_output_projection:
        conditioning_states = self.proj_out(control_hidden_states)

    return conditioning_states, control_hidden_states


class WanVACEModel(WanModel):
  """Extension of Wan to include VACE conditioning."""

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      vace_layers: list[int],
      vace_in_channels: int,
      model_type="t2v",
      patch_size: Tuple[int, ...] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
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
      remat_policy: str = "None",
      names_which_can_be_saved: list[str] = [],
      names_which_can_be_offloaded: list[str] = [],
      scan_layers: bool = True,
  ):
    """Initializes the VACE model.

    All arguments are similar to WanModel with the exception of:
      vace_layers: Indices of the layers at which the VACE conditioning is
      injected.
      vace_in_channels: Number of channels in the VACE conditioning.
    """
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers
    self.scan_layers = scan_layers

    # 1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(
        attention_head_dim, patch_size, rope_max_seq_len
    )
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    # 2. Condition embeddings
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
    )

    self.gradient_checkpoint = GradientCheckpointType.from_str(
        remat_policy
    )
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved

    # 3. Transformer blocks

    if scan_layers:
      raise NotImplementedError("scan_layers is not supported yet")
    else:
      blocks = nnx.List([])
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
            attention=attention,
            dropout=dropout,
        )
        blocks.append(block)
      self.blocks = blocks

    if scan_layers:
      raise NotImplementedError("scan_layers is not supported yet")
    else:
      vace_blocks = nnx.List([])

      for vace_block_id in self.config.vace_layers:
        vace_block = WanVACETransformerBlock(
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
            attention=attention,
            dropout=dropout,
            apply_input_projection=vace_block_id == 0,
            apply_output_projection=True,
        )
        vace_blocks.append(vace_block)
      self.vace_blocks = vace_blocks

    self.vace_patch_embedding = nnx.Conv(
        rngs=rngs,
        in_features=vace_in_channels,
        out_features=inner_dim,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    self.norm_out = FP32LayerNorm(
        rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False
    )
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), ("embed", None)
        ),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), (None, None, "embed")
        ),
    )

  @jax.named_scope("WanVACEModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      control_hidden_states: jax.Array,
      control_hidden_states_scale: Optional[jax.Array] = None,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    hidden_states = nn.with_logical_constraint(
        hidden_states, ("batch", None, None, None, None)
    )
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    if control_hidden_states_scale is None:
      control_hidden_states_scale = jnp.ones_like(
          control_hidden_states, shape=(len(self.config.vace_layers),)
      )
    if control_hidden_states_scale.shape[0] != len(self.config.vace_layers):
      raise ValueError(
          "Length of `control_hidden_states_scale`"
          f" {len(control_hidden_states_scale)} should be equal to"
          f" {len(self.config.vace_layers)}."
      )

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    control_hidden_states = jnp.transpose(
        control_hidden_states, (0, 2, 3, 4, 1)
    )
    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = jax.lax.collapse(hidden_states, 1, -1)

    control_hidden_states = self.vace_patch_embedding(control_hidden_states)
    control_hidden_states = jax.lax.collapse(control_hidden_states, 1, -1)
    control_hidden_states_padding = jnp.zeros((
        batch_size,
        control_hidden_states.shape[1],
        hidden_states.shape[2] - control_hidden_states.shape[2],
    ))

    control_hidden_states = jnp.concatenate(
        [control_hidden_states, control_hidden_states_padding], axis=2
    )

    # Condition embedder is a FC layer.
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(  # We will need to mask out the text embedding.
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
    )
    timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_hidden_states_image is not None:
      raise NotImplementedError("img2vid is not yet implemented.")

    if self.scan_layers:
      raise NotImplementedError("scan_layers is not supported yet")
    else:
      # Prepare VACE hints
      control_hidden_states_list = nnx.List([])
      for i, vace_block in enumerate(self.vace_blocks):
        def layer_forward(hidden_states, control_hidden_states):
          return vace_block(
              hidden_states=hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              control_hidden_states=control_hidden_states,
              temb=timestep_proj,
              rotary_emb=rotary_emb,
              deterministic=deterministic,
              rngs=rngs,
          )

        rematted_layer_forward = self.gradient_checkpoint.apply(
            layer_forward,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        conditioning_states, control_hidden_states = rematted_layer_forward(
            hidden_states, control_hidden_states
        )
        control_hidden_states_list.append(
            (conditioning_states, control_hidden_states_scale[i])
        )

      control_hidden_states_list = control_hidden_states_list[::-1]

      for i, block in enumerate(self.blocks):

        def layer_forward_vace(hidden_states):
          return block(
              hidden_states,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs,
          )

        rematted_layer_forward = self.gradient_checkpoint.apply(
            layer_forward_vace,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        hidden_states = rematted_layer_forward(hidden_states)
        if i in self.config.vace_layers:
          control_hint, scale = control_hidden_states_list.pop()
          hidden_states = hidden_states + control_hint * scale

    # 6. Output norm, projection & unpatchify
    shift, scale = jnp.split(
        self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1
    )

    hidden_states = (
        self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift
    ).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)  # Linear layer.

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = jax.lax.collapse(hidden_states, 6, None)
    hidden_states = jax.lax.collapse(hidden_states, 4, 6)
    hidden_states = jax.lax.collapse(hidden_states, 2, 4)
    return hidden_states
