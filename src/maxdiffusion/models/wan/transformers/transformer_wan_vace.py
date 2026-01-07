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

from typing import Tuple

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from .... import common_types
from ...attention_flax import FlaxWanAttention
from ...normalization_flax import FP32LayerNorm
from .transformer_wan import WanFeedForward

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
