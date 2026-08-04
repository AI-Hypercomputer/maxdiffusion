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

import contextlib
import functools
import math
from typing import Optional, Tuple, Any, Dict
import flax.linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from einops import rearrange
from .. import common_types
from maxdiffusion.tpu_utils import get_tpu_type, TpuType
from maxdiffusion.models.feedforward_flax import NNXSimpleFeedForward  # noqa: F401

from . import quantizations

LOG2E = math.log2(math.e)

Array = common_types.Array
Mesh = common_types.Mesh
DType = common_types.DType
BlockSizes = common_types.BlockSizes


AxisNames = common_types.AxisNames
CONTEXT = common_types.CONTEXT
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED
Quant = quantizations.AqtQuantization

SELF_ATTN_HEAD = common_types.SELF_ATTN_HEAD
SELF_ATTN_Q_LENGTH = common_types.SELF_ATTN_Q_LENGTH
SELF_ATTN_KV_LENGTH = common_types.SELF_ATTN_KV_LENGTH
CROSS_ATTN_HEAD = common_types.CROSS_ATTN_HEAD
CROSS_ATTN_Q_LENGTH = common_types.CROSS_ATTN_Q_LENGTH
CROSS_ATTN_KV_LENGTH = common_types.CROSS_ATTN_KV_LENGTH


# TODO(v2.0): Emit DeprecationWarning on these legacy symbol re-exports and purge in next major release.
# Import and re-export all utility functions, KERNEL_REGISTRY, and wrappers for 100% backwards compatibility
from maxdiffusion.models.attention_utils import (
    _build_padding_segment_ids,  # noqa: F401
    _coerce_tokamax_block_sizes,  # noqa: F401
    _extract_custom_block_sizes,  # noqa: F401
    _max_row_norm_per_head,  # noqa: F401
    _pad_data_for_flash,  # noqa: F401
    _prepare_attention_mask_for_shard_map,  # noqa: F401
    _reshape_batch_dim_to_heads,  # noqa: F401
    _reshape_heads_to_batch_dim,  # noqa: F401
    _reshape_heads_to_head_dim,  # noqa: F401
    _run_chunked_ulysses_attention,  # noqa: F401
    _select_flash_block_sizes,  # noqa: F401
    _ulysses_head_chunk_ranges,  # noqa: F401
    _unflatten_heads,  # noqa: F401
    AttentionBlockSizes,  # noqa: F401
    INTERNAL_RING_AXIS,  # noqa: F401
    INTERNAL_ULYSSES_AXIS,  # noqa: F401
    jax_memory_efficient_attention,  # noqa: F401
)
from maxdiffusion.models.attention_dispatch import (
    _apply_attention,  # noqa: F401
    _apply_attention_dot,  # noqa: F401
    _tpu_flash_attention,  # noqa: F401
    _ulysses_attention,  # noqa: F401
    _ulysses_ring_attention,  # noqa: F401
    _ulysses_ring_custom_attention,  # noqa: F401
    KERNEL_REGISTRY,  # noqa: F401
    register_kernel,  # noqa: F401
)
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel  # noqa: F401
from maxdiffusion.kernels.splash_attention import ring_attention_kernel as tokamax_ring_attention_kernel  # noqa: F401

BlockSizes = AttentionBlockSizes


def apply_rope(xq: Array, xk: Array, freqs_cis: Any) -> tuple[Array, Array]:
  if isinstance(freqs_cis, (tuple, list)):
    cos, sin = freqs_cis
    if cos.ndim == 2:
      seq_len = cos.shape[0]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
      else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3 and cos.shape[0] == 1:
      seq_len = cos.shape[1]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
      else:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]

    def _rotate(x):
      x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
      x_real = x_reshaped[..., 0]
      x_imag = x_reshaped[..., 1]
      return jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)

    xq_out = xq * cos + _rotate(xq) * sin
    xk_out = xk * cos + _rotate(xk) * sin
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

  xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

  return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)


class NNXAttentionOp(nnx.Module):

  def __init__(
      self,
      mesh: Mesh,
      attention_kernel: str,
      scale: float,
      heads: int,
      dim_head: int,
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = True,
      float32_qk_product: bool = True,
      axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV),
      axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV),
      # Uses splash attention on cross attention.
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      dtype: DType = jnp.float32,
      quant: Quant = None,
      mask_padding_tokens: bool = True,
      residual_checkpoint_name: str | None = None,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
      ulysses_shards: int = -1,
      ulysses_attention_chunks: int = 1,
  ):
    self.dpa_layer = None
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.ulysses_shards = ulysses_shards
    self.ulysses_attention_chunks = ulysses_attention_chunks
    if attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      jax.config.update("jax_use_shardy_partitioner", False)

      dpa_layer = DotProductAttention(
          head_dim=dim_head,
          num_attention_heads=heads,
          num_gqa_groups=heads,
          attn_mask_type="no_mask",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
          attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
          # attention_dropout=self.dropout_rate,
          dropout_rng_name="aqt",
          dtype=dtype,
          qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
          scale_factor=scale,
          transpose_batch_sequence=False,
      )
      variables = {}
      self.dpa_layer = functools.partial(dpa_layer.apply, variables)

    self.mesh = mesh
    self.scale = scale
    self.heads = heads
    self.dim_head = dim_head
    self.attention_kernel = attention_kernel
    self.use_memory_efficient_attention = use_memory_efficient_attention
    self.split_head_dim = split_head_dim
    self.float32_qk_product = float32_qk_product
    self.axis_names_q = axis_names_q
    self.axis_names_kv = axis_names_kv
    self.flash_min_seq_length = flash_min_seq_length
    self.flash_block_sizes = flash_block_sizes
    self.dtype = dtype
    self.quant = quant
    self.mask_padding_tokens = mask_padding_tokens
    self.residual_checkpoint_name = residual_checkpoint_name

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      attention_mask: Array = None,
  ):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        dpa_layer=self.dpa_layer,
        mask_padding_tokens=self.mask_padding_tokens,
        residual_checkpoint_name=self.residual_checkpoint_name,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp if hasattr(self, "use_base2_exp") else False,
        use_experimental_scheduler=self.use_experimental_scheduler if hasattr(self, "use_experimental_scheduler") else False,
        ulysses_shards=(self.ulysses_shards if hasattr(self, "ulysses_shards") else -1),
        ulysses_attention_chunks=(self.ulysses_attention_chunks if hasattr(self, "ulysses_attention_chunks") else 1),
    )


class AttentionOp(nn.Module):
  mesh: Mesh
  attention_kernel: str
  scale: float
  heads: int
  dim_head: int
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  float32_qk_product: bool = True
  axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV)
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  dtype: DType = jnp.float32
  quant: Quant = None
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False
  ulysses_shards: int = -1
  ulysses_attention_chunks: int = 1

  def setup(self):
    self.dpa_layer = None
    if self.attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      jax.config.update("jax_use_shardy_partitioner", False)

      dpa_layer = DotProductAttention(
          head_dim=self.dim_head,
          num_attention_heads=self.heads,
          num_gqa_groups=self.heads,
          attn_mask_type="no_mask",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
          attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
          # attention_dropout=self.dropout_rate,
          dropout_rng_name="aqt",
          dtype=self.dtype,
          # float32_logits=self.float32_logits,
          qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
          scale_factor=self.scale,
          transpose_batch_sequence=False,
      )
      variables = {}
      self.dpa_layer = functools.partial(dpa_layer.apply, variables)

  def apply_attention(self, query: Array, key: Array, value: Array, attention_mask: Array = None):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        dpa_layer=self.dpa_layer,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
        ulysses_shards=self.ulysses_shards,
        ulysses_attention_chunks=self.ulysses_attention_chunks,
    )


class FlaxWanAttention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      query_dim: int,
      cross_attention_dim: Optional[int] = None,
      heads: int = 8,
      dim_head: int = 64,
      dropout: float = 0.0,
      eps: float = 1e-6,
      qk_norm: str = "rms_norm_across_heads",
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = False,
      attention_kernel: str = "flash",
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      key_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      value_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      out_axis_names: AxisNames = (BATCH, LENGTH, EMBED),
      precision: jax.lax.Precision = None,
      qkv_bias: bool = False,
      quant: Quant = None,
      is_self_attention: bool = True,
      mask_padding_tokens: bool = True,
      residual_checkpoint_name: str | None = None,
      enable_jax_named_scopes: bool = False,
      added_kv_proj_dim: Optional[int] = None,  # New for I2V
      image_seq_len: Optional[int] = None,  # New for I2V
      attention_config: Optional[dict] = None,
  ):
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        "ulysses_attention_chunks": 1,
        **(attention_config or {}),
    }

    if attention_kernel in {"flash", "cudnn_flash_te"} and mesh is None:
      raise ValueError(f"The flash attention kernel requires a value for mesh, but mesh is {self.mesh}")
    self.dim_head = dim_head
    self.heads = heads
    self.inner_dim = dim_head * heads
    scale = dim_head**-0.5
    self.qk_norm = qk_norm
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.out_axis_names = out_axis_names
    self.enable_jax_named_scopes = enable_jax_named_scopes

    cross_attention_remapped_to_flash = not is_self_attention and attention_kernel in (
        "tokamax_ring",
        "tokamax_ring_custom",
        "ulysses_ring",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
        "ulysses_ring_custom_bidir",
        "ulysses_custom",
        "ulysses_custom_fixed_m",
    )
    cross_attention_uses_local_kv = not is_self_attention and (
        cross_attention_remapped_to_flash or attention_kernel in ("flash", "tokamax_flash", "cudnn_flash_te")
    )
    if is_self_attention:
      axis_names_q = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV)
    else:
      axis_names_q = (BATCH, CROSS_ATTN_HEAD, CROSS_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (
          BATCH,
          CROSS_ATTN_HEAD,
          None if cross_attention_uses_local_kv else CROSS_ATTN_KV_LENGTH,
          D_KV,
      )
    if cross_attention_remapped_to_flash:
      attention_kernel = "tokamax_flash"
    elif attention_kernel in ("tokamax_ring", "tokamax_ring_custom", "ulysses_ring") and not is_self_attention:
      attention_kernel = "tokamax_flash"  # do not use ring attention for cross attention
    elif (
        attention_kernel in ("ulysses_ring_custom", "ulysses_ring_custom_bidir", "ulysses_ring_custom_fixed_m")
        and not is_self_attention
    ):
      attention_kernel = "ulysses_custom"  # plain ulysses (no ring) for cross attention
    self.added_kv_proj_dim = added_kv_proj_dim  # New for I2V
    self.image_seq_len = image_seq_len  # New for I2V
    tpu_type = get_tpu_type()
    self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128

    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=scale,
        heads=heads,
        dim_head=dim_head,
        use_memory_efficient_attention=use_memory_efficient_attention,
        split_head_dim=split_head_dim,
        float32_qk_product=False,
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        quant=quant,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name=residual_checkpoint_name,
        use_base2_exp=attention_config["use_base2_exp"],
        use_experimental_scheduler=attention_config["use_experimental_scheduler"],
        ulysses_shards=attention_config["ulysses_shards"],
        ulysses_attention_chunks=attention_config["ulysses_attention_chunks"],
    )
    # None axes corresponds to the stacked weights across all blocks
    # because of the use of nnx.vmap and nnx.scan.
    # Dims are [num_blocks, embed, heads]
    kernel_axes = ("embed", "heads")
    qkv_init_kernel = nnx.with_partitioning(nnx.initializers.lecun_normal(), kernel_axes)

    self.query = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.key = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.value = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.proj_attn = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("embed",),
        ),
    )

    self.drop_out = nnx.Dropout(dropout, deterministic=False)

    self.norm_q = nnx.data(None)
    self.norm_k = nnx.data(None)
    if qk_norm is not None:
      self.norm_q = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

      self.norm_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

    # New layers for I2V image conditioning
    self.add_k_proj = nnx.data(None)
    self.add_v_proj = nnx.data(None)
    self.norm_added_k = nnx.data(None)
    if self.added_kv_proj_dim is not None:
      self.add_k_proj = nnx.Linear(
          self.added_kv_proj_dim,
          self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          bias_init=nnx.with_partitioning(
              nnx.initializers.zeros,
              ("embed",),
          ),
      )
      self.add_v_proj = nnx.Linear(
          self.added_kv_proj_dim,
          self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          bias_init=nnx.with_partitioning(
              nnx.initializers.zeros,
              ("embed",),
          ),
      )
      self.norm_added_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          param_dtype=weights_dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
      )

  def _apply_rope(self, xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # 1. Extract cos and sin, keeping them in native bfloat16
    cos = jnp.real(freqs_cis).astype(xq.dtype)
    sin = jnp.imag(freqs_cis).astype(xq.dtype)

    # 2. Reshape the last dimension into pairs
    xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)

    # 3. Unbind the pairs
    xq_0, xq_1 = xq_reshaped[..., 0], xq_reshaped[..., 1]
    xk_0, xk_1 = xk_reshaped[..., 0], xk_reshaped[..., 1]

    # 4. Pure real arithmetic (XLA will fuse these instantly into FMA instructions)
    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos

    xk_out_0 = xk_0 * cos - xk_1 * sin
    xk_out_1 = xk_0 * sin + xk_1 * cos

    # 5. Stack and reshape back to original
    xq_out = jnp.stack([xq_out_0, xq_out_1], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([xk_out_0, xk_out_1], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array = None,
      rotary_emb: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
  ) -> jax.Array:
    axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
    hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
    encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)
    dtype = hidden_states.dtype
    is_self_attention = encoder_hidden_states is None
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    is_i2v_cross_attention = self.added_kv_proj_dim is not None and not is_self_attention

    # For T2V self-attention and cross-attention, we skip passing the mask
    # to avoid overhead, as it should be all 1s for unpadded sequences.
    if not is_i2v_cross_attention:
      encoder_attention_mask = None

    if not is_i2v_cross_attention:
      with jax.named_scope("query_proj"):
        query_proj = self.query(hidden_states)

      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj = self.norm_q(query_proj)

      if not is_self_attention and cached_kv is not None and "text" in cached_kv:
        key_proj, value_proj = cached_kv["text"]
      else:
        with jax.named_scope("key_proj"):
          key_proj = self.key(encoder_hidden_states)
        with jax.named_scope("value_proj"):
          value_proj = self.value(encoder_hidden_states)

        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj = self.norm_k(key_proj)

      if rotary_emb is not None:
        with self.conditional_named_scope("attn_rope"):
          query_proj = _unflatten_heads(query_proj, self.heads)
          key_proj = _unflatten_heads(key_proj, self.heads)
          value_proj = _unflatten_heads(value_proj, self.heads)
          # output of _unflatten_heads Batch, heads, seq_len, head_dim
          query_proj, key_proj = self._apply_rope(query_proj, key_proj, rotary_emb)

      query_proj = checkpoint_name(query_proj, "query_proj")
      key_proj = checkpoint_name(key_proj, "key_proj")
      value_proj = checkpoint_name(value_proj, "value_proj")

      with jax.named_scope("apply_attention"):
        attn_output = self.attention_op.apply_attention(
            query_proj,
            key_proj,
            value_proj,
            attention_mask=encoder_attention_mask,
        )

    else:
      # NEW PATH for I2V CROSS-ATTENTION
      with self.conditional_named_scope("proj_query"):
        query_proj_raw = self.query(hidden_states)

      # Image embeddings are padded to multiples of 128 (v5p and below) or 256 (v6e and above) for TPU flash attention
      # Calculate the padded length to correctly split image and text embeddings
      if self.added_kv_proj_dim is not None:
        alignment = self.alignment
        if self.image_seq_len is not None:
          image_seq_len_actual = self.image_seq_len
        else:
          image_seq_len_actual = 257
        padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment  # 257 -> 384
        encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
        encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]

        # Use the passed encoder_attention_mask (created in embeddings_flax.py) if using Flash Attention
        # It contains the image mask: [1]*257 + [0]*127 for 257 real image tokens padded to 384
        if encoder_attention_mask is not None:
          encoder_attention_mask_img = encoder_attention_mask[:, :padded_img_len]
        else:
          # Fallback: no mask means treat all as valid (for dot product attention)
          encoder_attention_mask_img = None
      else:
        # If no image_seq_len is specified, treat all as text
        encoder_hidden_states_img = None
        encoder_hidden_states_text = encoder_hidden_states
        encoder_attention_mask_img = None

      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj_text = self.norm_q(query_proj_raw)
      else:
        query_proj_text = query_proj_raw

      # Text K/V
      if cached_kv is not None and "text" in cached_kv:
        key_proj_text, value_proj_text = cached_kv["text"]
      else:
        with self.conditional_named_scope("proj_key"):
          key_proj_text = self.key(encoder_hidden_states_text)
        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj_text = self.norm_k(key_proj_text)
        with self.conditional_named_scope("proj_value"):
          value_proj_text = self.value(encoder_hidden_states_text)

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        if cached_kv is not None and "image" in cached_kv:
          key_proj_img, value_proj_img = cached_kv["image"]
        else:
          with self.conditional_named_scope("add_proj_k"):
            key_proj_img = self.add_k_proj(encoder_hidden_states_img)
          with self.conditional_named_scope("norm_add_k"):
            key_proj_img = self.norm_added_k(key_proj_img)
          with self.conditional_named_scope("add_proj_v"):
            value_proj_img = self.add_v_proj(encoder_hidden_states_img)
        query_proj_img = query_proj_raw
        # Check norm_added_k too
        # Checkpointing
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")
        key_proj_img = checkpoint_name(key_proj_img, "key_proj_img")
        value_proj_img = checkpoint_name(value_proj_img, "value_proj_img")
        query_proj_img = checkpoint_name(query_proj_img, "query_proj_img")

        # Attention - tensors are (B, S, D)
        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output_text = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)
        with self.conditional_named_scope("cross_attn_img_apply"):
          # Pass encoder_attention_mask_img for image cross-attention to mask padded tokens
          attn_output_img = self.attention_op.apply_attention(
              query_proj_img,
              key_proj_img,
              value_proj_img,
              attention_mask=encoder_attention_mask_img,
          )

        attn_output = attn_output_text + attn_output_img
      else:
        # No image embeddings, only text cross-attention
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")

        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)

    attn_output = attn_output.astype(dtype=dtype)
    attn_output = checkpoint_name(attn_output, "attn_output")

    with jax.named_scope("proj_attn"):
      hidden_states = self.proj_attn(attn_output)
      if self.drop_out.rate > 0:
        hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    return hidden_states

  def compute_kv(
      self,
      encoder_hidden_states: jax.Array,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    is_i2v_cross_attention = self.added_kv_proj_dim is not None

    if not is_i2v_cross_attention:
      with jax.named_scope("key_proj"):
        key_proj = self.key(encoder_hidden_states)
      with jax.named_scope("value_proj"):
        value_proj = self.value(encoder_hidden_states)

      if self.qk_norm:
        with self.conditional_named_scope("attn_k_norm"):
          key_proj = self.norm_k(key_proj)

      return {"text": (key_proj, value_proj)}
    else:
      # Image embeddings are padded to multiples of 128 (v5p and below) or 256 (v6e and above) for TPU flash attention
      alignment = self.alignment
      if self.image_seq_len is not None:
        image_seq_len_actual = self.image_seq_len
      else:
        image_seq_len_actual = 257
      padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment

      if encoder_attention_mask is None:
        padded_img_len = image_seq_len_actual

      encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
      encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]

      # Text K/V
      with self.conditional_named_scope("proj_key"):
        key_proj_text = self.key(encoder_hidden_states_text)
      if self.qk_norm:
        with self.conditional_named_scope("attn_k_norm"):
          key_proj_text = self.norm_k(key_proj_text)
      with self.conditional_named_scope("proj_value"):
        value_proj_text = self.value(encoder_hidden_states_text)

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        with self.conditional_named_scope("add_proj_k"):
          key_proj_img = self.add_k_proj(encoder_hidden_states_img)
        with self.conditional_named_scope("norm_add_k"):
          key_proj_img = self.norm_added_k(key_proj_img)
        with self.conditional_named_scope("add_proj_v"):
          value_proj_img = self.add_v_proj(encoder_hidden_states_img)

        return {
            "text": (key_proj_text, value_proj_text),
            "image": (key_proj_img, value_proj_img),
        }
      else:
        return {"text": (key_proj_text, value_proj_text)}


class FlaxFluxAttention(nn.Module):
  query_dim: int
  heads: int = 8
  dim_head: int = 64
  dropout: float = 0.0
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  out_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  precision: jax.lax.Precision = None
  qkv_bias: bool = False
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False

  def setup(self):
    if self.attention_kernel in {"flash", "cudnn_flash_te"} and self.mesh is None:
      raise ValueError(f"The flash attention kernel requires a value for mesh, but mesh is {self.mesh}")
    inner_dim = self.dim_head * self.heads
    scale = self.dim_head**-0.5

    self.attention_op = AttentionOp(
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        scale=scale,
        heads=self.heads,
        dim_head=self.dim_head,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        split_head_dim=self.split_head_dim,
        flash_block_sizes=self.flash_block_sizes,
        dtype=self.dtype,
        float32_qk_product=False,
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
    )

    kernel_axes = ("embed", "heads")
    qkv_init_kernel = nn.with_logical_partitioning(nn.initializers.lecun_normal(), kernel_axes)

    self.qkv = nn.Dense(
        inner_dim * 3,
        kernel_init=qkv_init_kernel,
        use_bias=self.qkv_bias,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="i_qkv",
        precision=self.precision,
    )

    self.encoder_qkv = nn.Dense(
        inner_dim * 3,
        kernel_init=qkv_init_kernel,
        use_bias=self.qkv_bias,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="e_qkv",
        precision=self.precision,
    )

    proj_attn_kernel_axes = ("heads", "embed")

    self.proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), proj_attn_kernel_axes),
        use_bias=True,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="i_proj",
        precision=self.precision,
    )

    self.encoder_proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), proj_attn_kernel_axes),
        use_bias=True,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="e_proj",
        precision=self.precision,
    )

    self.query_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )
    self.key_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )

    self.encoder_query_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )
    self.encoder_key_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )

  def __call__(
      self,
      hidden_states,
      encoder_hidden_states=None,
      attention_mask=None,
      image_rotary_emb=None,
  ):
    B, L = hidden_states.shape[:2]
    # Deduce dimensions cleanly from class attributes
    H, D = self.heads, self.dim_head

    qkv_proj = self.qkv(hidden_states)
    qkv_proj = checkpoint_name(qkv_proj, "img_qkv_proj")

    qkv_proj = qkv_proj.reshape(B, L, 3, H, D)
    query_proj, key_proj, value_proj = jnp.split(qkv_proj, 3, axis=2)
    query_proj = query_proj.squeeze(2)
    key_proj = key_proj.squeeze(2)
    value_proj = value_proj.squeeze(2)

    query_proj = self.query_norm(query_proj)
    key_proj = self.key_norm(key_proj)

    if encoder_hidden_states is not None:
      B_enc, L_txt = encoder_hidden_states.shape[:2]
      encoder_qkv_proj = self.encoder_qkv(encoder_hidden_states)
      encoder_qkv_proj = checkpoint_name(encoder_qkv_proj, "txt_qkv_proj")
      encoder_qkv_proj = encoder_qkv_proj.reshape(B_enc, L_txt, 3, H, D)
      enc_query_proj, enc_key_proj, enc_value_proj = jnp.split(encoder_qkv_proj, 3, axis=2)
      enc_query_proj = enc_query_proj.squeeze(2)
      enc_key_proj = enc_key_proj.squeeze(2)
      enc_value_proj = enc_value_proj.squeeze(2)

      encoder_query_proj = self.encoder_query_norm(enc_query_proj)
      encoder_key_proj = self.encoder_key_norm(enc_key_proj)

      query_proj = jnp.concatenate((encoder_query_proj, query_proj), axis=1)
      key_proj = jnp.concatenate((encoder_key_proj, key_proj), axis=1)
      value_proj = jnp.concatenate((enc_value_proj, value_proj), axis=1)

      # query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
      # key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
      # value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

    if not isinstance(image_rotary_emb, (tuple, list)):
      image_rotary_emb = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)

    query_proj = query_proj.swapaxes(1, 2)
    key_proj = key_proj.swapaxes(1, 2)
    query_proj, key_proj = apply_rope(query_proj, key_proj, image_rotary_emb)
    query_proj = query_proj.swapaxes(1, 2)
    key_proj = key_proj.swapaxes(1, 2)

    query_proj = query_proj.reshape(B, -1, H * D)
    key_proj = key_proj.reshape(B, -1, H * D)
    value_proj = value_proj.reshape(B, -1, H * D)

    if encoder_hidden_states is not None:
      query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
      key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
      value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

    attn_output = self.attention_op.apply_attention(query_proj, key_proj, value_proj, attention_mask=attention_mask)
    context_attn_output = None

    if encoder_hidden_states is not None:
      context_attn_output, attn_output = (
          attn_output[:, : encoder_hidden_states.shape[1]],
          attn_output[:, encoder_hidden_states.shape[1] :],
      )

      attn_output = self.proj_attn(attn_output)

      context_attn_output = self.encoder_proj_attn(context_attn_output)

    return attn_output, context_attn_output


# TODO(v2.0): Emit DeprecationWarning on these legacy symbol re-exports and purge in next major release.
# Re-export legacy Diffusers 2D UNet/transformer blocks for 100% backwards compatibility
from maxdiffusion.models.unet_transformer_blocks_flax import (  # noqa: F401
    FlaxAttention,
    FlaxBasicTransformerBlock,
    FlaxTransformer2DModel,
    FlaxFeedForward,
    FlaxGEGLU,
)
