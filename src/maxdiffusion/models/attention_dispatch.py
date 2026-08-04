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

"""Attention dispatching, KERNEL_REGISTRY, and backwards-compatible wrappers."""

import functools
import math
from typing import Any, Callable, Dict, Protocol, TypedDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.models import attention_utils
from maxdiffusion.models.attention_utils import (  # noqa: F401
    _query_chunk_attention,
    _reshape_batch_dim_to_heads,
    _reshape_data_for_flash,
    _reshape_heads_to_batch_dim,
    _reshape_heads_to_head_dim,
    _select_flash_block_sizes,
    INTERNAL_RING_AXIS,
    INTERNAL_ULYSSES_AXIS,
    jax_memory_efficient_attention,
)
from maxdiffusion.models.attention_strategies import (
    AttentionBackend,
    SingleShardStrategy,
    RingAttentionStrategy,
    UlyssesStrategy,
    DotProductAttentionStrategy,
    FlashAttentionStrategy,
)

LOG2E = math.log2(math.e)
from maxdiffusion.common_types import (
    Array,
    Mesh,
    DType,
    BlockSizes,
    AxisNames,
    CONTEXT,
)

Quant = Any

SAFE_MIN_ATTN_WEIGHT = -1e30


class AttentionContext(TypedDict, total=False):
  heads: int
  mesh: Mesh
  axis_names_q: AxisNames
  axis_names_kv: AxisNames
  flash_block_sizes: BlockSizes
  dtype: DType
  mask_padding_tokens: bool
  residual_checkpoint_name: str | None
  attention_mask: Array | None
  scale: float
  use_base2_exp: bool
  use_experimental_scheduler: bool
  ulysses_shards: int
  ulysses_attention_chunks: int
  dim_head: int
  split_head_dim: bool
  float32_qk_product: bool
  use_memory_efficient_attention: bool
  dpa_layer: Callable


class KernelRegistryCallable(Protocol):

  def __call__(self, q: Array, k: Array, v: Array, context: AttentionContext) -> Array:
    ...


KERNEL_REGISTRY: Dict[AttentionBackend, KernelRegistryCallable] = {}


def register_kernel(name: AttentionBackend) -> Callable[[KernelRegistryCallable], KernelRegistryCallable]:
  def decorator(func: KernelRegistryCallable) -> KernelRegistryCallable:
    KERNEL_REGISTRY[name] = func
    return func

  return decorator


def _check_attention_inputs(query: Array, key: Array, value: Array) -> None:
  if query.ndim < 3 or key.ndim < 3 or value.ndim < 3:
    raise ValueError("q, k, v must have at least 3 dimensions.")
  if key.ndim != value.ndim:
    raise ValueError("k, v must have same rank.")
  if query.shape[:-3] != key.shape[:-3] or key.shape[:-3] != value.shape[:-3]:
    raise ValueError("q, k, v batch dims must match.")
  if key.shape[-2] != value.shape[-2]:
    raise ValueError("k, v num_kv_heads must match.")
  if key.shape[-3] != value.shape[-3]:
    raise ValueError("k, v lengths must match.")
  if query.shape[-1] != key.shape[-1]:
    raise ValueError("q, k depths must match.")


def _apply_attention_dot(
    query: Array,
    key: Array,
    value: Array,
    dtype: DType,
    heads: int,
    dim_head: int,
    scale: float,
    split_head_dim: bool = True,
    float32_qk_product: bool = True,
    use_memory_efficient_attention: bool = False,
    attention_mask: Array = None,
):
  """Reference un-fused JAX dot-product attention."""
  if split_head_dim:
    b = key.shape[0]
    query_states = jnp.reshape(query, (b, -1, heads, dim_head))
    key_states = jnp.reshape(key, (b, -1, heads, dim_head))
    value_states = jnp.reshape(value, (b, -1, heads, dim_head))
  else:
    query_states = _reshape_heads_to_batch_dim(query, heads)
    key_states = _reshape_heads_to_batch_dim(key, heads)
    value_states = _reshape_heads_to_batch_dim(value, heads)

  strategy = DotProductAttentionStrategy(
      scale=scale,
      split_head_dim=split_head_dim,
      float32_qk_product=float32_qk_product,
      use_memory_efficient_attention=use_memory_efficient_attention,
  )
  attention_output = strategy(
      query_states,
      key_states,
      value_states,
      q_seq_len=query.shape[1],
      kv_seq_len=key.shape[1],
      attention_mask=attention_mask,
      dtype=dtype,
  )

  if not split_head_dim:
    attention_output = _reshape_batch_dim_to_heads(attention_output, heads)
  return attention_output


def _cudnn_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    heads: int,
    mesh: Mesh,
    dpa_layer: Callable,
) -> Array:
  """GPU TransformerEngine cuDNN flash attention."""
  org_q_shape = query.shape
  if query.ndim == 3 and query.shape[0] != 1 and query.shape[0] % heads == 0:
    query = _reshape_batch_dim_to_heads(query, heads)
    key = _reshape_batch_dim_to_heads(key, heads)
    value = _reshape_batch_dim_to_heads(value, heads)

  b, q_len, h, d = query.shape
  out = dpa_layer(query.reshape(b, q_len, h * d), key.reshape(b, -1, h * d), value.reshape(b, -1, h * d))
  out = out.reshape(b, q_len, h, d)
  if len(org_q_shape) == 3:
    out = _reshape_heads_to_batch_dim(out, heads)
  return out


def _tpu_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    attention_kernel: str = "flash",
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
) -> jax.Array:
  """TPU Flash Attention wrapper around Pallas kernels."""
  num_context_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
  attention_mask = attention_utils._prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and attention_kernel == AttentionBackend.TOKAMAX_RING_CUSTOM:
    raise NotImplementedError("tokamax_ring_custom does not support attention_mask.")
  block_sizes = _select_flash_block_sizes(query, key, flash_block_sizes, dtype, attention_kernel)

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)

  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
      out_specs=q_axis_names,
      check_vma=False,
  )
  def wrap_flash_attention(query, key, value):
    strategy = FlashAttentionStrategy(
        backend=attention_kernel,
        block_sizes=block_sizes,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=None,
        use_fixed_m=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name=residual_checkpoint_name,
        num_context_shards=num_context_shards,
    )
    return strategy(
        query,
        key,
        value,
        q_seq_len=query.shape[2],
        kv_seq_len=key.shape[2],
        attention_mask=attention_mask,
    )

  x = wrap_flash_attention(query, key, value)
  x = x[:, :, :orig_q_seq_len, :]
  x = jax.lax.with_sharding_constraint(x, q_axis_names)
  x = _reshape_heads_to_head_dim(x)
  return x


def _ulysses_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    use_custom_kernel: bool = False,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    use_fixed_m: bool = False,
    ulysses_attention_chunks: int = 1,
) -> jax.Array:
  """Ulysses attention using UlyssesStrategy."""
  attention_mask = attention_utils._prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and use_custom_kernel:
    raise NotImplementedError(
        "The custom dense splash kernel (use_custom_kernel) does not support attention_mask "
        "(padding is handled via orig_seq_len)."
    )
  num_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  inner = SingleShardStrategy(
      local_kernel=custom_splash.make_splash_mha if use_custom_kernel else splash_attention_kernel.make_splash_mha,
      block_sizes=flash_block_sizes,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      use_fixed_m=use_fixed_m,
      backend=AttentionBackend.CUSTOM if use_custom_kernel else AttentionBackend.TOKAMAX,
  )
  strategy = UlyssesStrategy(
      ulysses_shards=num_shards,
      num_ring_shards=1,
      ulysses_axis=CONTEXT,
      ring_axis=CONTEXT,
      ulysses_attention_chunks=ulysses_attention_chunks,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      use_fixed_m=use_fixed_m,
      inner_strategy=inner,
  )
  return strategy(
      query=query,
      key=key,
      value=value,
      heads=heads,
      mesh=mesh,
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      flash_block_sizes=flash_block_sizes,
      dtype=dtype,
      attention_mask=attention_mask,
  )


def _ulysses_ring_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    ulysses_shards: int = -1,
    ulysses_attention_chunks: int = 1,
) -> jax.Array:
  """Ulysses+Ring attention using TokaMax ring kernel."""
  if ulysses_shards <= 0:
    raise ValueError("Ulysses ring attention requires ulysses_shards to be set from config or command line.")
  attention_mask = attention_utils._prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  num_context_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  num_ring_shards = num_context_shards // ulysses_shards
  inner = RingAttentionStrategy(
      block_sizes=flash_block_sizes,
      ring_axis=INTERNAL_RING_AXIS,
      ulysses_axis=INTERNAL_ULYSSES_AXIS,
      num_ring_shards=num_ring_shards,
      num_ulysses_shards=ulysses_shards,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      backend=AttentionBackend.TOKAMAX,
  )
  strategy = UlyssesStrategy(
      ulysses_shards=ulysses_shards,
      num_ring_shards=num_ring_shards,
      ulysses_axis=INTERNAL_ULYSSES_AXIS,
      ring_axis=INTERNAL_RING_AXIS,
      ulysses_attention_chunks=ulysses_attention_chunks,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      inner_strategy=inner,
  )
  return strategy(
      query=query,
      key=key,
      value=value,
      heads=heads,
      mesh=mesh,
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      flash_block_sizes=flash_block_sizes,
      dtype=dtype,
      attention_mask=attention_mask,
  )


def _ulysses_ring_custom_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    ulysses_shards: int = -1,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    bidirectional: bool = False,
    use_fixed_m: bool = False,
    ulysses_attention_chunks: int = 1,
) -> jax.Array:
  """Ulysses+Ring attention using Custom Splash kernel and composed strategy."""
  num_context_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  num_ring_shards = num_context_shards // ulysses_shards
  if num_ring_shards == 1:
    inner = SingleShardStrategy(
        local_kernel=custom_splash.make_splash_mha,
        block_sizes=flash_block_sizes,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        use_fixed_m=use_fixed_m,
        backend=AttentionBackend.CUSTOM,
    )
  else:
    inner = RingAttentionStrategy(
        block_sizes=flash_block_sizes,
        ring_axis=INTERNAL_RING_AXIS,
        ulysses_axis=INTERNAL_ULYSSES_AXIS,
        num_ring_shards=num_ring_shards,
        num_ulysses_shards=ulysses_shards,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        use_fixed_m=use_fixed_m,
        bidirectional=bidirectional,
        backend=AttentionBackend.CUSTOM,
    )
  strategy = UlyssesStrategy(
      ulysses_shards=ulysses_shards,
      num_ring_shards=num_ring_shards,
      ulysses_axis=INTERNAL_ULYSSES_AXIS,
      ring_axis=INTERNAL_RING_AXIS,
      ulysses_attention_chunks=ulysses_attention_chunks,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      use_fixed_m=use_fixed_m,
      bidirectional=bidirectional,
      inner_strategy=inner,
  )
  return strategy(
      query=query,
      key=key,
      value=value,
      heads=heads,
      mesh=mesh,
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      flash_block_sizes=flash_block_sizes,
      dtype=dtype,
      attention_mask=attention_mask,
  )


# --- Registering Backwards-Compatible Kernels ---


@register_kernel(AttentionBackend.DOT_PRODUCT)
def dot_product_kernel(q, k, v, context):
  return _apply_attention_dot(
      q,
      k,
      v,
      context["dtype"],
      context["heads"],
      context["dim_head"],
      context["scale"],
      context["split_head_dim"],
      context["float32_qk_product"],
      context["use_memory_efficient_attention"],
      context["attention_mask"],
  )


def _ulysses_common_kwargs(context: AttentionContext) -> AttentionContext:
  """Unpacks standard keyword arguments from context for Ulysses attention wrappers."""
  return {
      "heads": context["heads"],
      "mesh": context["mesh"],
      "axis_names_q": context["axis_names_q"],
      "axis_names_kv": context["axis_names_kv"],
      "flash_block_sizes": context["flash_block_sizes"],
      "dtype": context["dtype"],
      "mask_padding_tokens": context["mask_padding_tokens"],
      "residual_checkpoint_name": context["residual_checkpoint_name"],
      "attention_mask": context["attention_mask"],
      "use_base2_exp": context.get("use_base2_exp", True),
      "use_experimental_scheduler": context.get("use_experimental_scheduler", False),
      "ulysses_attention_chunks": context["ulysses_attention_chunks"],
  }


def _flash_common_kwargs(context: AttentionContext, attention_kernel: str) -> AttentionContext:
  """Unpacks standard keyword arguments from context for FlashAttention wrappers."""
  return {
      "heads": context["heads"],
      "mesh": context["mesh"],
      "axis_names_q": context["axis_names_q"],
      "axis_names_kv": context["axis_names_kv"],
      "flash_block_sizes": context["flash_block_sizes"],
      "dtype": context["dtype"],
      "attention_kernel": attention_kernel,
      "mask_padding_tokens": context["mask_padding_tokens"],
      "residual_checkpoint_name": context["residual_checkpoint_name"],
      "attention_mask": context["attention_mask"],
      "use_base2_exp": context["use_base2_exp"],
      "use_experimental_scheduler": context["use_experimental_scheduler"],
  }


@register_kernel(AttentionBackend.ULYSSES_CUSTOM)
def ulysses_custom_kernel(q, k, v, context):
  return _ulysses_attention(q, k * context["scale"], v, use_custom_kernel=True, **_ulysses_common_kwargs(context))


@register_kernel(AttentionBackend.ULYSSES_RING_CUSTOM)
def ulysses_ring_custom_kernel(q, k, v, context):
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      ulysses_shards=context["ulysses_shards"],
      **_ulysses_common_kwargs(context),
  )


@register_kernel(AttentionBackend.ULYSSES_RING_CUSTOM_FIXED_M)
def ulysses_ring_custom_fixed_m_kernel(q, k, v, context):
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      ulysses_shards=context["ulysses_shards"],
      use_fixed_m=True,
      **_ulysses_common_kwargs(context),
  )


@register_kernel(AttentionBackend.ULYSSES_RING_CUSTOM_BIDIR)
def ulysses_ring_custom_bidir_kernel(q, k, v, context):
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      ulysses_shards=context["ulysses_shards"],
      bidirectional=True,
      **_ulysses_common_kwargs(context),
  )


@register_kernel(AttentionBackend.ULYSSES_CUSTOM_FIXED_M)
def ulysses_custom_fixed_m_kernel(q, k, v, context):
  return _ulysses_attention(
      q,
      k * context["scale"],
      v,
      use_custom_kernel=True,
      use_fixed_m=True,
      **_ulysses_common_kwargs(context),
  )


@register_kernel(AttentionBackend.ULYSSES)
def ulysses_kernel(q, k, v, context):
  return _ulysses_attention(q, k * context["scale"], v, **_ulysses_common_kwargs(context))


@register_kernel(AttentionBackend.ULYSSES_RING)
def ulysses_ring_kernel(q, k, v, context):
  return _ulysses_ring_attention(
      q,
      k * context["scale"],
      v,
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      heads=context["heads"],
      mesh=context["mesh"],
      axis_names_q=context["axis_names_q"],
      axis_names_kv=context["axis_names_kv"],
      flash_block_sizes=context["flash_block_sizes"],
      dtype=context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
  )


@register_kernel(AttentionBackend.FLASH)
def flash_kernel(q, k, v, context):
  return _tpu_flash_attention(q, k * context["scale"], v, **_flash_common_kwargs(context, "flash"))


@register_kernel(AttentionBackend.TOKAMAX_FLASH)
def tokamax_flash_kernel(q, k, v, context):
  return _tpu_flash_attention(q, k * context["scale"], v, **_flash_common_kwargs(context, "tokamax_flash"))


@register_kernel(AttentionBackend.TOKAMAX_RING)
def tokamax_ring_kernel(q, k, v, context):
  return _tpu_flash_attention(q, k * context["scale"], v, **_flash_common_kwargs(context, "tokamax_ring"))


@register_kernel(AttentionBackend.TOKAMAX_RING_CUSTOM)
def tokamax_ring_custom_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k * context["scale"],
      v,
      **_flash_common_kwargs(context, "tokamax_ring_custom"),
  )


@register_kernel(AttentionBackend.CUDNN_FLASH_TE)
def cudnn_flash_te_kernel(q, k, v, context):
  return _cudnn_flash_attention(q, k, v, context["heads"], context["mesh"], context["dpa_layer"])


def _apply_attention(
    query: Array,
    key: Array,
    value: Array,
    heads: int,
    dim_head: int,
    split_head_dim: bool,
    float32_qk_product: bool,
    attention_kernel: str,
    flash_min_seq_length: int,
    use_memory_efficient_attention: bool,
    scale: float,
    dtype: jnp.dtype,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dpa_layer: Callable,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: Array = None,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    ulysses_shards: int = -1,
    ulysses_attention_chunks: int = 1,
):
  """Routes to different attention kernels using a module-level registry."""
  _check_attention_inputs(query, key, value)
  seq_len_idx = 1
  if query.ndim == 4:
    seq_len_idx = 2

  can_use_flash_attention = True
  if attention_kernel not in (
      AttentionBackend.DOT_PRODUCT,
      AttentionBackend.CUDNN_FLASH_TE,
  ):
    can_use_flash_attention = (
        query.shape[seq_len_idx] >= flash_min_seq_length
        and key.shape[seq_len_idx] >= flash_min_seq_length
        and value.shape[seq_len_idx] >= flash_min_seq_length
    )

  effective_attention_kernel = attention_kernel
  if attention_kernel == AttentionBackend.DOT_PRODUCT or use_memory_efficient_attention or not can_use_flash_attention:
    effective_attention_kernel = AttentionBackend.DOT_PRODUCT

  # Masks enter the dispatcher as canonical [B, K] keep masks. Adapt them
  # only after fallback selection because a configured flash kernel may use
  # dot-product attention for short sequences.
  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
    attention_mask = attention_mask.astype(jnp.bool_)
    if effective_attention_kernel == AttentionBackend.DOT_PRODUCT:
      attention_bias = jnp.where(
          attention_mask,
          jnp.asarray(0.0, dtype=dtype),
          jnp.asarray(-10000.0, dtype=dtype),
      )
      if split_head_dim:
        attention_mask = attention_bias[:, None, None, :]
      else:
        attention_mask = jnp.repeat(attention_bias, heads, axis=0)[:, None, :]

  context = {
      "heads": heads,
      "mesh": mesh,
      "axis_names_q": axis_names_q,
      "axis_names_kv": axis_names_kv,
      "flash_block_sizes": flash_block_sizes,
      "dtype": dtype,
      "mask_padding_tokens": mask_padding_tokens,
      "residual_checkpoint_name": residual_checkpoint_name,
      "attention_mask": attention_mask,
      "scale": scale,
      "use_base2_exp": use_base2_exp,
      "use_experimental_scheduler": use_experimental_scheduler,
      "ulysses_shards": ulysses_shards,
      "ulysses_attention_chunks": ulysses_attention_chunks,
      "dim_head": dim_head,
      "split_head_dim": split_head_dim,
      "float32_qk_product": float32_qk_product,
      "use_memory_efficient_attention": use_memory_efficient_attention,
      "dpa_layer": dpa_layer,
  }

  if effective_attention_kernel in KERNEL_REGISTRY:
    return KERNEL_REGISTRY[effective_attention_kernel](query, key, value, context)

  try:
    kernel_enum = AttentionBackend(effective_attention_kernel)
    if kernel_enum in KERNEL_REGISTRY:
      return KERNEL_REGISTRY[kernel_enum](query, key, value, context)
  except ValueError:
    pass

  raise ValueError(f"Unexpected attention kernel {effective_attention_kernel=}.")
