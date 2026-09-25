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
import contextvars
import dataclasses
import functools
import math
from typing import Optional, Callable, Tuple, Any, Dict
import flax.linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from maxdiffusion import wan_runtime_options
from maxdiffusion.kernels.splash_attention import splash_attention_mask as tokamax_splash_attention_mask
from maxdiffusion.kernels.splash_attention import splash_attention_kernel as tokamax_splash_attention_kernel
from maxdiffusion.kernels.splash_attention import ring_attention_kernel as tokamax_ring_attention_kernel
from maxdiffusion.kernels.splash_attention import base as tokamax_splash_base
from maxdiffusion.kernels.fused_producers import fused_rmsnorm_rope
from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import (
    fused_rmsnorm_rope_pallas,
    resolve_rope_accum,
    rope_accum_is_measured,
    with_xla_backward,
)
from maxdiffusion.kernels import cross_attention_pallas
from einops import rearrange
from .. import common_types, max_logging
from maxdiffusion.tpu_utils import get_tpu_type, TpuType
from maxdiffusion.max_utils import safe_getattr


from ..kernels import custom_splash_attention as custom_splash
from ..kernels import custom_svg_attention_dispatch
from ..kernels import custom_svg_static_range_attention
from . import quantizations
from .modeling_flax_utils import get_activation

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

INTERNAL_RING_AXIS = "ring"
INTERNAL_ULYSSES_AXIS = "ulysses"


def _coerce_tokamax_block_sizes(block_sizes):
  if isinstance(block_sizes, dict):
    return splash_attention_kernel.BlockSizes(
        block_q=block_sizes.get("block_q", 512),
        block_kv=block_sizes.get("block_kv", 512),
        block_kv_compute=block_sizes.get("block_kv_compute", 512),
        block_q_dkv=block_sizes.get("block_q_dkv", 512),
        block_kv_dkv=block_sizes.get("block_kv_dkv", 512),
        block_kv_dkv_compute=block_sizes.get("block_kv_dkv_compute", 512),
        block_q_dq=block_sizes.get("block_q_dq", None),
        block_kv_dq=block_sizes.get("block_kv_dq", None),
        use_fused_bwd_kernel=block_sizes.get("use_fused_bwd_kernel", False),
    )
  # Tokamax requires fused bwd; convert if needed.
  if getattr(block_sizes, "use_fused_bwd_kernel", False):
    return block_sizes

  # Fall back if some fields are missing.
  bq = block_sizes.block_q
  bkv = getattr(block_sizes, "block_kv", bq)
  bkv_compute = getattr(block_sizes, "block_kv_compute", bkv)
  bq_dkv = getattr(block_sizes, "block_q_dkv", bq)
  bkv_dkv = getattr(block_sizes, "block_kv_dkv", bkv)
  bkv_dkv_compute = getattr(block_sizes, "block_kv_dkv_compute", bkv_compute)
  return splash_attention_kernel.BlockSizes(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_compute,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=bkv_dkv_compute,
      block_q_dq=None,
      block_kv_dq=None,
      use_fused_bwd_kernel=True,
  )


def _maybe_aqt_einsum(quant: Quant):
  return jnp.einsum if quant is None else quant.einsum()


def _check_attention_inputs(query: Array, key: Array, value: Array) -> None:
  """Check attention inputs."""

  assert key.ndim == value.ndim, "k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
  assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."


def _reshape_data_from_cudnn_flash(tensor):
  # reshapes from [b, s, h, d] back to [b, s, h * d]
  return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


def _reshape_data_for_cudnn_flash(tensor, heads):
  # reshapes from [b, s, h * d] to [b, s, h, d] (input format to flash format)
  if len(tensor.shape) == 3:
    batch, seq, dim_head = tensor.shape
    tensor = tensor.reshape(batch, seq, heads, dim_head // heads)
  else:
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _reshape_batch_dim_to_heads(tensor, heads):
  batch_size, seq_len, dim = tensor.shape
  head_size = heads
  tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  reshaped_tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _reshape_heads_to_batch_dim(tensor, heads):
  if tensor.ndim == 3:
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
  else:
    batch_size, head_size, seq_len, head_dim = tensor.shape
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, head_dim)
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _reshape_heads_to_head_dim(tensor):
  # takes a tensor of shape [b, h, s, d] and reshapes to [b, s, h * d]
  # This is used to transform the output of flash attention back into the format of other attention outputs
  b, h, s, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  reshaped_tensor = jnp.reshape(tensor, (b, -1, h * d))
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _unflatten_heads(tensor, heads):
  # reshapes from [b, s, h * d] to [b, h, s, d] (input format to flash format)
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  # Transpose to ('batch', 'heads', 'length', 'kv')
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _replace_mesh_axis(axis_spec, old_axis: str, new_axes: tuple[str, ...]):
  if axis_spec == old_axis:
    return new_axes
  if isinstance(axis_spec, tuple):
    replacement = []
    for axis in axis_spec:
      if axis == old_axis:
        replacement.extend(new_axes)
      else:
        replacement.append(axis)
    return tuple(replacement)
  return axis_spec


def _replace_mesh_axis_names(axis_names, old_axis: str, new_axes: tuple[str, ...]):
  return jax.sharding.PartitionSpec(*(_replace_mesh_axis(axis_name, old_axis, new_axes) for axis_name in axis_names))


# Attention kernels are traced once per layer per transformer, so an
# unconditional log would repeat dozens of times per run and be ignored.
_WARNED_ONCE: set[str] = set()


def _warn_once(key: str, message: str) -> None:
  """Logs `message` the first time `key` is seen in this process."""
  if key in _WARNED_ONCE:
    return
  _WARNED_ONCE.add(key)
  max_logging.log(message)


def resolve_k_centering(value, *, ring: bool) -> bool:
  """Resolves the `use_k_centering` setting for one attention path.

  `value` may be a bool, or "auto"/None (the config default). "auto" picks the
  cheap choice per path:
    * non-ring Ulysses (`ulysses_custom_fixed_m*`): ON. Centering is virtual --
      `q . k_mean` is folded into the kernel registers, with no collective and
      no HBM copy -- and it tightens the fixed-m bound.
    * ring paths (`ulysses_ring_custom*`): OFF. Centering there is
      materialized (R>1: `K - pmean(mean(K))` over (ulysses, ring) before the
      all-to-all; R==1: `K - mean(K)` after the all-to-all), costing a full-K
      HBM subtraction per layer, plus a cross-rank pmean when R>1.
  Strings "true"/"false" (e.g. from a command-line override) are accepted;
  unrecognised strings raise `ValueError`, other values are coerced with `bool()`.
  """
  if value is None:
    return not ring
  if isinstance(value, str):
    v = value.strip().lower()
    if v == "auto":
      return not ring
    if v in ("true", "1", "yes"):
      return True
    if v in ("false", "0", "no"):
      return False
    raise ValueError(f"use_k_centering must be a bool or 'auto', got {value!r}.")
  return bool(value)


def _validate_implicit_ulysses_degree(requested_ulysses_shards: int, context_shards: int, kernel_name: str) -> None:
  """Rejects a `ulysses_shards` request the non-ring Ulysses path cannot honour.

  The non-ring kernels always shard heads across the *entire* context mesh
  axis, so their Ulysses degree is implicitly `context_shards`. Silently
  ignoring a different explicit request has previously caused benchmarks to
  believe they were measuring U=2 while actually measuring U=4.
  """
  if requested_ulysses_shards is None or requested_ulysses_shards <= 0:
    return  # Unset: the implicit degree is what the caller wants.
  if requested_ulysses_shards == context_shards:
    return  # Explicit request agrees with what this path will do.
  raise ValueError(
      f"attention='{kernel_name}' cannot honour ulysses_shards={requested_ulysses_shards}: "
      f"the non-ring Ulysses path always splits heads across the full context mesh axis, "
      f"so its Ulysses degree is fixed at context_shards={context_shards}. "
      f"Either set ulysses_shards={context_shards} (or leave it unset), or switch to a ring "
      f"variant such as 'ulysses_ring_custom_fixed_m_per_q_block', which accepts "
      f"ulysses_shards=U and forms a ring of degree R=context_shards/U."
  )


def _largest_ulysses_shards_for_real_ring(context_shards: int, heads: int | None = None, kv_heads: int | None = None):
  """Largest Ulysses degree that still leaves a real ring (R > 1), or None if impossible.

  A usable Ulysses degree U must divide the context shard count *and* both head
  counts, mirroring the constraints the ring path itself enforces. Returning the
  largest such U below `context_shards` yields the smallest ring degree R > 1,
  which is normally the cheapest real ring for a given mesh.
  """
  for candidate in range(context_shards - 1, 0, -1):
    if context_shards % candidate != 0:
      continue
    if heads is not None and heads % candidate != 0:
      continue
    if kv_heads is not None and kv_heads % candidate != 0:
      continue
    return candidate
  return None


def _warn_if_ring_is_degenerate(
    num_ring_shards: int,
    num_ulysses_shards: int,
    context_shards: int,
    heads: int | None = None,
    kv_heads: int | None = None,
) -> None:
  """Warns when a ring variant collapses to R=1 and is really running plain Ulysses."""
  if num_ring_shards != 1:
    return

  if context_shards <= 1:
    advice = (
        "There is only one context shard, so no ring is possible on this mesh; "
        "increase ici_context_parallelism to use a ring."
    )
  else:
    suggestion = _largest_ulysses_shards_for_real_ring(context_shards, heads, kv_heads)
    if suggestion is None:
      advice = (
          f"No ulysses_shards below context_shards={context_shards} divides both the mesh and the "
          f"head counts (heads={heads}, kv_heads={kv_heads}), so this mesh cannot form a real ring."
      )
    else:
      advice = f"For a real ring set ulysses_shards={suggestion} (ring degree R={context_shards // suggestion})."

  _warn_once(
      f"degenerate_ring:{context_shards}:{num_ulysses_shards}",
      f"[attention] Ring degree R=1 (context_shards={context_shards} / ulysses_shards={num_ulysses_shards}). "
      f"This ring variant is degenerate: no KV is rotated, and the result is mathematically equivalent to "
      f"the corresponding non-ring Ulysses kernel (fixed-m numerics may differ: 'auto' K-centering is off "
      f"for ring variants and on for non-ring). Do NOT report this as a ring-attention result. " + advice,
  )


def _create_internal_ulysses_ring_mesh(
    mesh: Mesh,
    ring_shards: int,
    ulysses_shards: int,
    ring_axis: str = INTERNAL_RING_AXIS,
    ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
) -> Mesh:
  """Split the public context mesh axis into private ring and Ulysses axes."""
  mesh_axis_names = tuple(mesh.axis_names)
  context_axis_index = mesh_axis_names.index(CONTEXT)
  devices = mesh.devices
  new_shape = devices.shape[:context_axis_index] + (ring_shards, ulysses_shards) + devices.shape[context_axis_index + 1 :]
  new_axis_names = (
      mesh_axis_names[:context_axis_index] + (ring_axis, ulysses_axis) + mesh_axis_names[context_axis_index + 1 :]
  )
  return Mesh(devices.reshape(new_shape), new_axis_names)


def _reshape_data_for_flash(tensor, heads, num_context_shards=1):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  if tensor.ndim != 4:
    tensor = _unflatten_heads(tensor, heads)

  org_seq_len = tensor.shape[2]

  # Pad sequence dimension so it is evenly divisible by the context mesh axis,
  # which shard_map requires.
  if num_context_shards <= 1:
    return tensor, org_seq_len
  rem = org_seq_len % num_context_shards
  if rem == 0:
    return tensor, org_seq_len
  pad_width = [(0, 0)] * tensor.ndim
  pad_width[2] = (0, num_context_shards - rem)
  return jnp.pad(tensor, pad_width), org_seq_len


def _pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  tensor, _ = _reshape_data_for_flash(tensor, heads)

  # Pad head_dim to 128 if less than that.
  kv_size = tensor.shape[-1]
  head_dim_pad = 0
  if kv_size < 128:
    head_dim_pad = 128 - kv_size

  # Pad seq_len with sharding constraints.
  seq_len = tensor.shape[2]

  # 1. First, pad seq_len to be a multiple of flash_block_size
  rem = seq_len % flash_block_size
  if rem != 0:
    seq_len_padded_pre = seq_len + (flash_block_size - rem)
  else:
    seq_len_padded_pre = seq_len

  # 2. Ensure num_blocks is divisible by num_shards
  num_blocks = seq_len_padded_pre // flash_block_size
  if num_blocks % num_shards != 0:
    num_blocks += num_shards - (num_blocks % num_shards)

  final_padded_len = num_blocks * flash_block_size
  seq_len_pad = final_padded_len - seq_len

  if kv_size < 128 or seq_len_pad != 0:
    npad = ((0, 0), (0, 0), (0, seq_len_pad), (0, head_dim_pad))
    tensor = jnp.pad(tensor, npad)

  return tensor, kv_size, seq_len


def _flash_sequence_length(tensor: Array) -> int:
  if tensor.ndim == 3:
    return tensor.shape[1]
  if tensor.ndim == 4:
    return tensor.shape[2]
  raise ValueError(f"Flash attention expects rank-3 or rank-4 inputs, got rank {tensor.ndim}.")


def _select_flash_block_sizes(
    query: Array,
    key: Array,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype,
    attention_kernel: str,
    preserve_asymmetric_block_sizes: bool = False,
) -> BlockSizes:
  """Select Flash/Splash block sizes.

  Existing MaxDiffusion behavior is preserved by default. When
  preserve_asymmetric_block_sizes=True, explicitly configured block sizes are
  honored even when Q and KV have different sequence lengths; the existing
  padding path makes the tensors compatible with those block sizes.
  """
  query_seq_len = _flash_sequence_length(query)
  key_seq_len = _flash_sequence_length(key)

  q_max_block_size = 1024 if dtype == jnp.bfloat16 else 512

  if key_seq_len != query_seq_len:
    kv_max_block_size = ((key_seq_len + 127) // 128) * 128
  else:
    kv_max_block_size = q_max_block_size

  # Preserve the existing Tokamax conversion behavior.
  if flash_block_sizes is not None and not hasattr(flash_block_sizes, "use_fused_bwd_kernel"):
    flash_block_sizes = _coerce_tokamax_block_sizes(flash_block_sizes)

  # Existing self-attention behavior: configured values are returned unchanged.
  if flash_block_sizes is not None and key_seq_len == query_seq_len:
    if attention_kernel in ("tokamax_flash", "tokamax_ring"):
      return _coerce_tokamax_block_sizes(flash_block_sizes)
    return flash_block_sizes

  # NEW: opt-in behavior required by Klein KV-cache.
  #
  # Q and KV may have different sequence lengths, but _pad_data_for_flash()
  # pads each sequence independently to the configured block size. Therefore
  # block_q/block_kv do not need to divide the original sequence lengths.
  if preserve_asymmetric_block_sizes and flash_block_sizes is not None:
    if attention_kernel in ("tokamax_flash", "tokamax_ring"):
      return _coerce_tokamax_block_sizes(flash_block_sizes)
    return flash_block_sizes

  # Existing MaxDiffusion cross-attention behavior.
  block_size_q = flash_block_sizes.block_q if flash_block_sizes is not None else q_max_block_size

  use_tokamax = attention_kernel in ("tokamax_flash", "tokamax_ring")

  return splash_attention_kernel.BlockSizes(
      block_q=block_size_q,
      block_kv_compute=min(kv_max_block_size, key_seq_len),
      block_kv=min(kv_max_block_size, key_seq_len),
      block_q_dkv=block_size_q,
      block_kv_dkv=min(kv_max_block_size, key_seq_len),
      block_kv_dkv_compute=min(kv_max_block_size, query_seq_len),
      block_q_dq=None if use_tokamax else block_size_q,
      block_kv_dq=None if use_tokamax else min(kv_max_block_size, query_seq_len),
      use_fused_bwd_kernel=use_tokamax,
  )


def convert_to_tokamax_splash_config(
    block_sizes: BlockSizes,
    q_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    k_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    v_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    residual_checkpoint_name: str | None = None,
    attn_logits_soft_cap: float | None = None,
    fuse_reciprocal: bool = True,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    max_logit_const: float | None = None,
    interpret: bool = False,
    dq_reduction_steps: int | None = None,
) -> tokamax_splash_attention_kernel.SplashConfig:
  assert block_sizes.use_fused_bwd_kernel, "Tokamax Splash attention only supports fused bwd kernel."
  return tokamax_splash_attention_kernel.SplashConfig(
      block_q=block_sizes.block_q,
      block_kv=block_sizes.block_kv,
      block_kv_compute=block_sizes.block_kv_compute,
      block_q_dkv=block_sizes.block_q_dkv,
      block_kv_dkv=block_sizes.block_kv_dkv,
      block_kv_dkv_compute=block_sizes.block_kv_dkv_compute,
      block_q_dq=None if block_sizes.use_fused_bwd_kernel else block_sizes.block_q_dq,
      block_kv_dq=None if block_sizes.use_fused_bwd_kernel else block_sizes.block_kv_dq,
      use_fused_bwd_kernel=block_sizes.use_fused_bwd_kernel,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      residual_checkpoint_name=residual_checkpoint_name,
      attn_logits_soft_cap=attn_logits_soft_cap,
      fuse_reciprocal=fuse_reciprocal,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      max_logit_const=max_logit_const,
      interpret=interpret,
      dq_reduction_steps=dq_reduction_steps,
  )


def _extract_custom_block_sizes(flash_block_sizes):
  """Pulls custom-kernel block sizes out of the (dict or BlockSizes-like) config.

  Mirrors the extraction used by the `ulysses_custom` path so the custom ring
  kernel honors the same `flash_block_sizes={...}` knobs.
  """
  bq = 4864
  bkv = 1024
  bkv_compute = 1024
  bkv_compute_in = 1024
  heads_per_tile = 1
  vmem_limit_bytes = None
  if flash_block_sizes is not None:
    if isinstance(flash_block_sizes, dict):
      get = flash_block_sizes.get
      bq = get("block_q", None) or bq
      bkv = get("block_kv", None) or bkv
      bkv_compute = get("block_kv_compute", None) or bkv_compute
      bkv_compute_in = get("block_kv_compute_in", None) or bkv_compute_in
      heads_per_tile = get("heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = get("vmem_limit_bytes", None) or vmem_limit_bytes
    else:
      bq = getattr(flash_block_sizes, "block_q", None) or bq
      bkv = getattr(flash_block_sizes, "block_kv", None) or bkv
      bkv_compute = getattr(flash_block_sizes, "block_kv_compute", None) or bkv_compute
      bkv_compute_in = getattr(flash_block_sizes, "block_kv_compute_in", None) or bkv_compute_in
      heads_per_tile = getattr(flash_block_sizes, "heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = getattr(flash_block_sizes, "vmem_limit_bytes", None) or vmem_limit_bytes
  # A BlockSizes object carries heads_per_tile=None when the config dict omitted
  # it; getattr then returns that None instead of the default, so coerce it back
  # to 1 (the custom-kernel default) to keep the `heads_per_tile > 1` guards safe.
  if heads_per_tile is None:
    heads_per_tile = 1
  bkv_compute_in = min(bkv_compute, bkv_compute_in)
  return bq, bkv, bkv_compute, bkv_compute_in, heads_per_tile, vmem_limit_bytes


def _build_padding_segment_ids(
    query_seq_len: int,
    q_padded_len: int,
    key_seq_len: int,
    kv_padded_len: int,
    attention_mask: jax.Array | None,
    segment_ids_cls=splash_attention_kernel.SegmentIds,
):
  """Build splash segment ids that mask q/kv padding and the attention mask.

  Padding tokens get segment id 0, valid tokens 1. An optional attention_mask
  (batch, kv_len) is folded into the kv segment ids. Positions beyond an
  explicit mask are invalid, including sequence padding introduced before the
  local flash kernel. Shared by flash, Ulysses, and Ulysses+ring kernels.
  """
  q_indices = jax.lax.broadcasted_iota(jnp.int32, (q_padded_len,), 0)
  q_segment_ids = (q_indices < query_seq_len).astype(jnp.int32)

  kv_indices = jax.lax.broadcasted_iota(jnp.int32, (kv_padded_len,), 0)
  kv_segment_ids = (kv_indices < key_seq_len).astype(jnp.int32)

  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
    mask_len = min(key_seq_len, attention_mask.shape[1])
    kv_mask_for_batch = attention_mask[:, :mask_len].astype(jnp.int32)
    # An explicit mask is authoritative. This also masks sequence padding
    # introduced before shard_map to make the KV length context-divisible.
    if key_seq_len > mask_len:
      kv_mask_for_batch = jnp.concatenate(
          [
              kv_mask_for_batch,
              jnp.zeros((attention_mask.shape[0], key_seq_len - mask_len), jnp.int32),
          ],
          axis=1,
      )
    # Tokens past key_seq_len are padding.
    if kv_padded_len > key_seq_len:
      kv_mask_for_batch = jnp.concatenate(
          [
              kv_mask_for_batch,
              jnp.zeros(
                  (attention_mask.shape[0], kv_padded_len - key_seq_len),
                  jnp.int32,
              ),
          ],
          axis=1,
      )
    kv_segment_ids = (kv_segment_ids[None, :] * kv_mask_for_batch).astype(jnp.int32)
    q_segment_ids = jnp.broadcast_to(q_segment_ids[None, :], (attention_mask.shape[0], q_padded_len))

  return segment_ids_cls(q=q_segment_ids, kv=kv_segment_ids)


def _prepare_attention_mask_for_shard_map(
    attention_mask: jax.Array | None,
    batch_size: int,
    padded_kv_len: int,
) -> jax.Array | None:
  """Broadcasts and pads a canonical keep mask before entering shard_map."""
  if attention_mask is None:
    return None
  if attention_mask.ndim != 2:
    raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
  if attention_mask.shape[0] == 1 and batch_size != 1:
    attention_mask = jnp.broadcast_to(attention_mask, (batch_size, attention_mask.shape[1]))
  elif attention_mask.shape[0] != batch_size:
    raise ValueError(
        f"attention_mask batch dimension must be 1 or match the attention batch ({batch_size}), "
        f"got {attention_mask.shape[0]}."
    )

  attention_mask = attention_mask.astype(jnp.bool_)

  # `attention_mask.shape[1]` represents the true original sequence length of the KV states,
  # while `padded_kv_len` represents the padded sequence length required for XLA compilation/divisibility.
  # We pad the mask with `False` (masked out) to cover the padded dummy tokens.
  if attention_mask.shape[1] < padded_kv_len:
    attention_mask = jnp.pad(
        attention_mask,
        ((0, 0), (0, padded_kv_len - attention_mask.shape[1])),
        constant_values=False,
    )
  elif attention_mask.shape[1] > padded_kv_len:
    # If the user-provided mask exceeds the required length, we truncate it.
    attention_mask = attention_mask[:, :padded_kv_len]
  return attention_mask


def _mesh_axis_in_spec(axis_spec, mesh_axis: str) -> bool:
  """Returns whether a logical-to-mesh axis entry contains mesh_axis."""
  if isinstance(axis_spec, tuple):
    return mesh_axis in axis_spec
  return axis_spec == mesh_axis


def _ulysses_head_chunk_ranges(num_heads: int, ulysses_shards: int, num_chunks: int):
  """Build head-axis ranges for chunked Ulysses all-to-all.

  The Ulysses all-to-all splits each local chunk's head axis over
  `ulysses_shards`, so every returned range length is a multiple of
  `ulysses_shards`. When `num_chunks` does not evenly divide the number of
  Ulysses-sized head groups, earlier chunks get the floor-sized range and the
  final chunk carries the remainder.

  Returns:
    A list of `(start, end)` half-open ranges over the head axis. Concatenating
    tensors sliced with these ranges along the head axis restores the original
    head layout. For `num_chunks <= 1`, returns `[(0, num_heads)]`, which is the
    unchunked all-to-all path.
  """
  if num_chunks <= 1:
    return [(0, num_heads)]
  if num_heads % ulysses_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of heads to be divisible by the Ulysses shard count, "
        f"got heads={num_heads} and ulysses_shards={ulysses_shards}."
    )

  head_groups = num_heads // ulysses_shards
  num_chunks = min(num_chunks, head_groups)
  regular_groups_per_chunk = max(1, head_groups // num_chunks)

  ranges = []
  start_group = 0
  for chunk_idx in range(num_chunks):
    end_group = head_groups if chunk_idx == num_chunks - 1 else min(start_group + regular_groups_per_chunk, head_groups)
    if start_group >= end_group:
      break
    ranges.append((start_group * ulysses_shards, end_group * ulysses_shards))
    start_group = end_group
  return ranges


@dataclasses.dataclass(frozen=True)
class TokenPadding:
  """Pad tokens a model inserted into its self-attention sequence.

  The sequence is `num_segments` equal segments laid end to end: one per ring
  shard on the Ulysses-ring path, a single segment otherwise. Each segment holds
  `real_len` real tokens followed by `padded_len - real_len` pad tokens.
  """

  real_len: int
  padded_len: int
  num_segments: int = 1

  def __post_init__(self):
    if not 0 < self.real_len <= self.padded_len or self.num_segments < 1:
      raise ValueError(f"Invalid token padding: {self}.")

  @property
  def total_len(self) -> int:
    return self.padded_len * self.num_segments


# Kernels whose self-attention honours TokenPadding. The pads sit where these
# kernels already expect a ragged KV tail (the end of the sequence, or of each
# ring segment), so passing the real length masks the pad keys, and the fixed-m
# norm reductions skip the pad rows. Any other kernel would attend to the pads.
TOKEN_PADDING_KERNELS = frozenset({
    "ulysses_custom",
    "ulysses_custom_fixed_m",
    "ulysses_custom_fixed_m_per_q_block",
    "ulysses_ring_custom",
    "ulysses_ring_custom_fixed_m",
    "ulysses_ring_custom_fixed_m_per_q_block",
})

_ACTIVE_TOKEN_PADDING: contextvars.ContextVar[Optional[TokenPadding]] = contextvars.ContextVar(
    "maxdiffusion_active_token_padding", default=None
)


@contextlib.contextmanager
def self_attention_token_padding(padding: Optional[TokenPadding]):
  """Declares, while the transformer blocks are traced, that the video tokens carry `padding`."""
  token = _ACTIVE_TOKEN_PADDING.set(padding)
  try:
    yield
  finally:
    _ACTIVE_TOKEN_PADDING.reset(token)


def _active_token_padding(seq_len: int, num_segments: int) -> Optional[TokenPadding]:
  """The declared padding when it describes this `seq_len`-token KV sequence, else None.

  Only self-attention over the padded video tokens matches. Cross-attention
  keys (text or image tokens) have their own length and carry no pads.
  """
  padding = _ACTIVE_TOKEN_PADDING.get()
  if padding is None or padding.total_len != seq_len:
    return None
  if padding.num_segments != num_segments:
    raise ValueError(
        f"The tokens were padded as {padding.num_segments} segment(s), but this attention splits the sequence into "
        f"{num_segments} ring segment(s), so the pads would not sit at the tail of each segment."
    )
  return padding


ULYSSES_OUT_A2A_MODES = ("flat", "chunked")


def _ulysses_seq_to_heads(
    x: jax.Array, *, axis_name, seq_axis: int, num_shards: int, mode: Optional[str] = None
) -> jax.Array:
  """Inverse Ulysses exchange: `all_to_all(x, split_axis=seq_axis, concat_axis=1, tiled=True)`.

  Shard c receives the c-th sequence chunk of every local head. `x` is
  [B, H_local, ...] with the full sequence on `seq_axis`; the result is
  [B, num_shards * H_local, ...] with the sequence cut to one chunk.

  `wan_ulysses_out_a2a` picks how the collective is written:
    * "flat": a plain split of the sequence axis. XLA reorders the whole output
      so the sequence axis is major before the collective, then reorders it back
      for the output projection. That is four relayout copies per layer at the
      720p shard shape (~1.5 ms on v6e, ~0.7 ms on tpu7x).
    * "chunked": split the sequence axis into (num_shards, chunk) and exchange
      over a leading chunk axis. Each shard's block is then contiguous in
      whatever minor-dimension order the producer used. The received
      [src, H_local, ...] blocks merge into [src * H_local, ...], which is the
      order "flat" produces. Same data movement, same values.
  """
  if mode is None:
    mode = wan_runtime_options.get("wan_ulysses_out_a2a")
  if mode not in ULYSSES_OUT_A2A_MODES:
    raise ValueError(f"wan_ulysses_out_a2a must be one of {ULYSSES_OUT_A2A_MODES}, got {mode!r}.")
  seq_len = x.shape[seq_axis]
  if mode == "flat" or num_shards == 1 or seq_len % num_shards:
    return jax.lax.all_to_all(x, axis_name=axis_name, split_axis=seq_axis, concat_axis=1, tiled=True)
  x = x.reshape(x.shape[:seq_axis] + (num_shards, seq_len // num_shards) + x.shape[seq_axis + 1 :])
  x = jnp.moveaxis(x, seq_axis, 1)  # [B, num_shards, H_local, ...]
  x = jax.lax.all_to_all(x, axis_name=axis_name, split_axis=1, concat_axis=1, tiled=True)
  return x.reshape((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _run_chunked_ulysses_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    num_heads: int,
    ulysses_shards: int,
    ulysses_attention_chunks: int,
    attention_fn,
) -> jax.Array:
  """Runs Ulysses attention chunked or unchunked along the head axis.

  Splits the attention compute and communication into head-group chunks so XLA
  can overlap communication and compute.

  Args:
    query: The query tensor, [B, H, S, D].
    key: The key tensor, [B, H, S, D].
    value: The value tensor, [B, H, S, D].
    num_heads: The number of heads in query.
    ulysses_shards: The Ulysses/context shard count.
    ulysses_attention_chunks: Number of head-group chunks to split into.
    attention_fn: The local Ulysses attention function to call on each chunk,
      taking (query, key, value) and returning the attention output.

  Returns:
    The concatenated attention output tensor.
  """
  if query.shape[1] != key.shape[1] and ulysses_attention_chunks > 1:
    raise NotImplementedError(
        f"GQA (query heads {query.shape[1]} != key heads {key.shape[1]}) with "
        f"ulysses_attention_chunks={ulysses_attention_chunks} > 1 is not supported."
    )
  head_chunk_ranges = _ulysses_head_chunk_ranges(num_heads, ulysses_shards, ulysses_attention_chunks)
  if len(head_chunk_ranges) > 1:
    chunk_outputs = [
        attention_fn(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
        )
        for start, end in head_chunk_ranges
    ]
    return jnp.concatenate(chunk_outputs, axis=1)
  else:
    return attention_fn(query, key, value)


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
    is_causal: bool = False,
    preserve_asymmetric_block_sizes: bool = False,
    spatiotemporal_config: Optional[dict] = None,
    spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
) -> jax.Array:
  """TPU Flash Attention"""

  num_context_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and attention_kernel == "tokamax_ring_custom":
    raise NotImplementedError("tokamax_ring_custom does not support attention_mask.")
  block_sizes = _select_flash_block_sizes(
      query,
      key,
      flash_block_sizes,
      dtype,
      attention_kernel,
      preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
  )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))

  def wrap_flash_attention(query, key, value, attention_mask):
    if attention_kernel == "tokamax_ring_custom":
      # Ring attention backed by the custom dense splash kernel. q stays local,
      # k/v rotate over the "context" axis (handled inside the ring kernel).
      (
          bq,
          bkv,
          bkv_compute,
          bkv_compute_in,
          heads_per_tile,
          vmem_limit_bytes,
      ) = _extract_custom_block_sizes(flash_block_sizes)
      if heads_per_tile > 1:
        raise NotImplementedError("tokamax_ring_custom currently supports heads_per_tile == 1 only.")
      query_local = query * LOG2E if use_base2_exp else query
      query_local, kv_size, query_seq_len = _pad_data_for_flash(query_local, heads, bq)
      key_local, _, key_seq_len = _pad_data_for_flash(key, heads, bkv)
      value_local, _, _ = _pad_data_for_flash(value, heads, bkv)

      bsizes = custom_splash._BlockSizes(
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkv_compute,
          block_kv_compute_in=bkv_compute_in,
      )
      ring_kernel = tokamax_ring_attention_kernel.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          ring_axis="context",
      )
      vmapped_ring = jax.vmap(ring_kernel, in_axes=(0, 0, 0))
      attention_output = vmapped_ring(query_local, key_local, value_local)
      return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    uses_fused_kernel = block_sizes.use_fused_bwd_kernel
    block_q_sizes = (
        block_sizes.block_q,
        block_sizes.block_q_dkv,
    )
    block_kv_sizes = (
        block_sizes.block_kv,
        block_sizes.block_kv_dkv,
    )
    if uses_fused_kernel:
      block_q_sizes += (block_sizes.block_q_dkv,)
      block_kv_sizes += (block_sizes.block_kv_dkv,)
    else:
      block_q_sizes += (block_sizes.block_q_dq,)
      block_kv_sizes += (block_sizes.block_kv_dq,)

    block_q = max(*block_q_sizes)
    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)

    block_kv = max(*block_kv_sizes)
    key, _, key_seq_len = _pad_data_for_flash(key, heads, block_kv)
    value, _, _ = _pad_data_for_flash(value, heads, block_kv)

    if is_causal:
      mask = splash_attention_mask.CausalMask((query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])
    else:
      mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

    segment_ids_cls = (
        tokamax_splash_base.SegmentIds if attention_kernel == "tokamax_ring" else splash_attention_kernel.SegmentIds
    )
    segment_ids = _build_padding_segment_ids(
        query_seq_len,
        query.shape[2],
        key_seq_len,
        key.shape[2],
        attention_mask,
        segment_ids_cls,
    )

    # make_splash_mha is wrapped around shardmap and seq and head is already
    # sharded based on in_specs, therefore setting head_shards=1 and q_seq_shards=1.
    if attention_kernel == "tokamax_flash":
      if is_causal:
        mask = tokamax_splash_attention_mask.CausalMask(
            (query.shape[2], key.shape[2]),
        )
      else:
        mask = tokamax_splash_attention_mask.FullMask(
            _shape=(query.shape[2], key.shape[2]),
        )
      splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
          mask=mask,
          q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
          config=convert_to_tokamax_splash_config(
              block_sizes,
              residual_checkpoint_name=residual_checkpoint_name,
              use_base2_exp=use_base2_exp,
              use_experimental_scheduler=use_experimental_scheduler,
          ),
          save_residuals=False,
      )
    elif attention_kernel == "tokamax_ring":
      mask = tokamax_splash_attention_mask.FullMask(
          _shape=(query.shape[2], key.shape[2]),
      )
      splash_kernel = tokamax_ring_attention_kernel.make_ring_attention(
          mask=mask,
          is_mqa=False,
          config=convert_to_tokamax_splash_config(
              block_sizes,
              residual_checkpoint_name=residual_checkpoint_name,
              use_base2_exp=use_base2_exp,
              use_experimental_scheduler=use_experimental_scheduler,
          ),
          save_residuals=False,
          ring_axis=CONTEXT,
          # Padding-only IDs are identical on each shard. Explicit masks differ
          # by KV shard and must rotate together with K/V.
          rotate_segment_ids=attention_mask is not None,
      )
    else:
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,  # the sizes of the axis is sharding over heads
          q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
          block_sizes=block_sizes,
          save_residuals=True if "ring" in attention_kernel else False,
          residual_checkpoint_name=residual_checkpoint_name,
          interpret=(jax.default_backend() == "cpu"),
      )

    segment_ids_in_axes = 0 if attention_mask is not None else None
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))

    if not mask_padding_tokens and attention_mask is None:
      segment_ids = None
    if attention_kernel in ["flash", "tokamax_flash", "tokamax_ring"]:
      attention_output = vmapped_splash(query, key, value, segment_ids)
    else:
      if num_context_shards > 1:
        out, (lse,) = vmapped_splash(query, key, value, segment_ids)
        m = lse.astype(jnp.float32)
        l = jnp.exp(lse - m)
        o = out.astype(jnp.float32) * l[..., None]

        perm = [(j, (j + 1) % num_context_shards) for j in range(num_context_shards)]

        k1 = jax.lax.ppermute(key, axis_name=CONTEXT, perm=perm)
        v1 = jax.lax.ppermute(value, axis_name=CONTEXT, perm=perm)

        def ring_scan_body(carry, _):
          m, l, o, k_current, v_current = carry
          k_next = jax.lax.ppermute(k_current, axis_name=CONTEXT, perm=perm)
          v_next = jax.lax.ppermute(v_current, axis_name=CONTEXT, perm=perm)

          out_chunk, (lse_chunk,) = vmapped_splash(query, k_current, v_current, segment_ids)

          m_chunk = lse_chunk.astype(jnp.float32)
          m_old = m
          m = jnp.maximum(m_old, m_chunk)

          exp_m_diff = jnp.exp(m_old - m)
          exp_m_chunk_diff = jnp.exp(m_chunk - m)

          l = l * exp_m_diff + jnp.exp(lse_chunk - m)
          o = o * exp_m_diff[..., None]
          o += exp_m_chunk_diff[..., None] * out_chunk.astype(jnp.float32)

          # Return the updated state for the next iteration
          return (m, l, o, k_next, v_next), None

        initial_carry = (m, l, o, k1, v1)
        (m_final, l_final, o_final, _, _), _ = jax.lax.scan(
            ring_scan_body, initial_carry, None, length=num_context_shards - 1
        )

        attention_output = o_final / l_final[..., None]
      else:
        raise ValueError("ring attention requires context > 1")
    return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

  data_dim = mesh.shape["data"] if "data" in mesh.shape else 1
  fsdp_dim = mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1
  devices_in_batch_sharding = data_dim * fsdp_dim
  # This warning might show up when doing model eval for example, when calculating model flops
  # and that is expected.
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_flash_attention = shard_map.shard_map(
        lambda q, k, v: wrap_flash_attention(q, k, v, None),
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=q_axis_names,
        check_rep=False,
    )
    x = sharded_flash_attention(query, key, value)
  else:
    sharded_flash_attention = shard_map.shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names, mask_axis_names),
        out_specs=q_axis_names,
        check_rep=False,
    )
    x = sharded_flash_attention(query, key, value, attention_mask)
  # Trim back to original sequence length after context-axis padding.
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)

  return x


# ---------------------------------------------------------------------------
# Ulysses sequence-parallel attention
# ---------------------------------------------------------------------------


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
    preserve_asymmetric_block_sizes: bool = False,
    spatiotemporal_config: Optional[dict] = None,
    spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
    per_q_block: bool = True,
    kv_heads: Optional[int] = None,
    ulysses_shards: int = -1,
    kernel_name: str = "ulysses_custom",
    use_k_centering: bool = True,
    qk_prescaled: bool = False,
    wan_ulysses_out_a2a: Optional[str] = None,
) -> jax.Array:
  """Ulysses sequence-parallel attention.

  Tensors arrive sequence-sharded on the context axis.  Inside a shard_map the
  all-to-all collectives trade sequence shards for head shards, run local
  splash attention on the full sequence with a subset of heads, then
  all-to-all back.

  The Ulysses degree of this path is implicitly the full context mesh axis; an
  explicit `ulysses_shards` that disagrees is rejected rather than ignored.
  """
  axis_name = CONTEXT
  num_shards = mesh.shape[axis_name]
  _validate_implicit_ulysses_degree(ulysses_shards, num_shards, kernel_name)
  if kv_heads is None:
    kv_heads = heads

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_shards)
  key, orig_kv_seq_len = _reshape_data_for_flash(key, kv_heads, num_shards)
  value, _ = _reshape_data_for_flash(value, kv_heads, num_shards)
  # Pad tokens inserted by the model (see `TokenPadding`) sit at the tail of the
  # sequence, like the shard padding above, and are masked the same way: the
  # custom kernel reads only the first `real_kv_seq_len` keys.
  token_padding = _active_token_padding(orig_kv_seq_len, num_segments=1)
  if token_padding is not None and (not use_custom_kernel or spatiotemporal_config is not None):
    raise NotImplementedError(f"{kernel_name}: token padding needs the dense custom kernel, which masks the pad keys.")
  real_kv_seq_len = orig_kv_seq_len if token_padding is None else token_padding.real_len
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and use_custom_kernel:
    raise NotImplementedError(
        "The custom dense splash kernel (use_custom_kernel) does not support attention_mask "
        "(it only handles padding via orig_seq_len); got a non-None attention_mask."
    )
  num_q_heads = query.shape[1]
  num_kv_heads = key.shape[1]
  # Ulysses only redistributes existing heads across the context mesh, so
  # indivisible head counts are rejected.
  if num_q_heads % num_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of query heads to be divisible by the context shard count, "
        f"got q_heads={num_q_heads} and context_shards={num_shards}."
    )
  if num_kv_heads % num_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of KV heads to be divisible by the context shard count, "
        f"got kv_heads={num_kv_heads} and context_shards={num_shards}."
    )
  num_heads = num_q_heads

  if not use_custom_kernel:
    block_sizes = _select_flash_block_sizes(
        query,
        key,
        flash_block_sizes,
        dtype,
        "flash",
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))
  mask_needs_ulysses_gather = _mesh_axis_in_spec(kv_axis_names[2], axis_name)

  def wrap_ulysses_attention(query, key, value, attention_mask):
    # Apply the base-2 rescale of Q *before* the all-to-all. A scalar elementwise
    # multiply commutes exactly with the collective (which is pure data movement),
    # so this is bit-identical. Done after the a2a it sat between the collective
    # and the kernel and XLA wrapped it in relayout copies; done before, it fuses
    # into the producer of Q and its 185MB round-trip disappears.
    if use_custom_kernel and use_base2_exp and not qk_prescaled:
      query = query * LOG2E
    # Swap sharding: each device gives up a slice of heads and gathers
    # a slice of sequence, so the local kernel sees the full sequence.
    query = jax.lax.all_to_all(query, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    key = jax.lax.all_to_all(key, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    value = jax.lax.all_to_all(value, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    if attention_mask is not None and mask_needs_ulysses_gather:
      attention_mask = jax.lax.all_gather(attention_mask, axis_name=axis_name, axis=1, tiled=True)

    if use_custom_kernel:
      if attention_mask is not None:
        raise NotImplementedError(
            "The custom dense splash kernel (use_custom_kernel) does not support attention_mask "
            "(it only handles padding via orig_seq_len); got a non-None attention_mask."
        )
      (
          bq,
          bkv,
          bkv_compute,
          bkv_compute_in,
          heads_per_tile,
          vmem_limit_bytes,
      ) = _extract_custom_block_sizes(flash_block_sizes)

      # NOTE: the base-2 rescale of Q is applied before the all-to-all above.
      raw_key = key
      raw_query = query
      raw_value = value
      context_q_seq_len = raw_query.shape[2]
      # With token padding this excludes the pad keys: everything below (the
      # kernel's ragged-tail mask, k_mean, the norm bounds) reads this length.
      actual_kv_seq_len = real_kv_seq_len

      real_key = raw_key[:, :, :actual_kv_seq_len, :]

      recenter, safe_bound = custom_splash.get_fixed_m_constants(actual_kv_seq_len)

      query, kv_size, query_seq_len = _pad_data_for_flash(raw_query, heads, bq)
      k_mean = None
      if use_fixed_m and use_k_centering:
        # Virtual k-centering (output-invariant): project q^T \bar{k} inside the
        # kernel registers without writing back / materializing (K - \bar{k}) in HBM.
        # Computed strictly on real (unpadded) tokens, indexed by KV head.
        k_mean = jnp.mean(real_key.astype(jnp.float32), axis=2)
        pad_d = max(0, query.shape[-1] - k_mean.shape[-1])
        if pad_d > 0:
          k_mean = jnp.pad(k_mean, ((0, 0), (0, 0), (0, pad_d)))
      # When actual_kv_seq_len is aligned to 8 sublanes, K/V are passed with NO
      # sequence padding. The fixed-m kernel slices the ragged KV tail
      # (`last_compute_body_fixed` in custom_splash_attention.py) using slice
      # lengths derived from the unpadded `orig_kv_seq_len`, so it never reads a
      # padded K/V row; materialising the pad cost 2 x 185MB of HBM traffic per
      # layer for nothing. Passing flash_block_size=1 makes only the sequence
      # pad a no-op -- the head_dim->128 pad, the reshape and the returned
      # (tensor, kv_size, seq_len) contract are all preserved.
      kv_pad_size = 1 if actual_kv_seq_len % 8 == 0 else bkv
      key, _, key_seq_len = _pad_data_for_flash(raw_key, heads, kv_pad_size)
      value, _, _ = _pad_data_for_flash(raw_value, heads, kv_pad_size)

      mk_arr = None
      all_fixed = None
      if use_fixed_m:
        mk_arr, all_fixed = _compute_fixed_m_metadata(
            query,
            real_key,
            block_q=bq,
            safe_bound=safe_bound,
            recenter=recenter,
            per_q_block=per_q_block,
            k_mean=k_mean,
            # Use the unpadded V: `all_fixed` gates the whole kernel through a
            # lax.cond, so anything feeding it sits on the critical path. Reading
            # the padded copy chained a 193MB pad + reduction behind the V
            # all-to-all and left that collective fully exposed. The padding is
            # zeros and the check is a max of squares, so this is output-invariant.
            # Model pad tokens are not zeros, so they are sliced off instead.
            value=raw_value if token_padding is None else raw_value[:, :, :actual_kv_seq_len, :],
        )

      bsizes = custom_splash._BlockSizes(
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkv_compute,
          block_kv_compute_in=bkv_compute_in,
      )

      if use_fixed_m:
        splash_kernel_uniform = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=True,
            uniform_fixed_m=True,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )
        splash_kernel_hybrid = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=True,
            uniform_fixed_m=False,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )

        def _run_uniform(q, k, v, m, km):
          return jax.vmap(splash_kernel_uniform, in_axes=(0, 0, 0, 0, 0))(q, k, v, m, km)

        def _run_hybrid(q, k, v, m, km):
          return jax.vmap(splash_kernel_hybrid, in_axes=(0, 0, 0, 0, 0))(q, k, v, m, km)

        attention_output = jax.lax.cond(
            all_fixed,
            _run_uniform,
            _run_hybrid,
            query,
            key,
            value,
            mk_arr,
            k_mean,
        )
      else:
        splash_kernel = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=False,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )
        vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0))
        attention_output = vmapped_splash(query, key, value)
      wan_splash_transpose_out = wan_runtime_options.get("wan_splash_transpose_out")
      if wan_splash_transpose_out:
        attention_output = attention_output[:, :, :context_q_seq_len, :kv_size].astype(query.dtype)
        # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
        # Sequence axis is at index 2 (sublanes), heads axis is at index 1.
        attention_output = _ulysses_seq_to_heads(
            attention_output, axis_name=axis_name, seq_axis=2, num_shards=num_shards, mode=wan_ulysses_out_a2a
        )
      else:
        attention_output = attention_output[:, :, :kv_size, :context_q_seq_len].astype(query.dtype)
        # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
        # Sequence axis is at index 3, heads axis is at index 1.
        attention_output = _ulysses_seq_to_heads(
            attention_output, axis_name=axis_name, seq_axis=3, num_shards=num_shards, mode=wan_ulysses_out_a2a
        )
      return attention_output
    else:
      # Run the same local splash kernel as standard TPU flash attention, but now
      # on full-sequence / fewer-heads tensors produced by the all-to-all above.
      uses_fused_kernel = block_sizes.use_fused_bwd_kernel
      block_q_sizes = (block_sizes.block_q, block_sizes.block_q_dkv)
      block_kv_sizes = (block_sizes.block_kv, block_sizes.block_kv_dkv)
      if uses_fused_kernel:
        block_q_sizes += (block_sizes.block_q_dkv,)
        block_kv_sizes += (block_sizes.block_kv_dkv,)
      else:
        block_q_sizes += (block_sizes.block_q_dq,)
        block_kv_sizes += (block_sizes.block_kv_dq,)

      block_q = max(*block_q_sizes)
      query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)
      block_kv = max(*block_kv_sizes)
      key, _, key_seq_len = _pad_data_for_flash(key, heads, block_kv)
      value, _, _ = _pad_data_for_flash(value, heads, block_kv)

      mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

      segment_ids = _build_padding_segment_ids(query_seq_len, query.shape[2], key_seq_len, key.shape[2], attention_mask)
      if not mask_padding_tokens and attention_mask is None:
        segment_ids = None

      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=block_sizes,
          save_residuals=False,
          residual_checkpoint_name=residual_checkpoint_name,
          interpret=(jax.default_backend() == "cpu"),
      )
      segment_ids_in_axes = 0 if attention_mask is not None else None
      vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
      attention_output = vmapped_splash(query, key, value, segment_ids)
      attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

      # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
      attention_output = jax.lax.all_to_all(
          attention_output,
          axis_name=axis_name,
          split_axis=2,
          concat_axis=1,
          tiled=True,
      )
      return attention_output

  devices_in_batch_sharding = mesh.shape["data"] * (mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1)
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  # Fold the (CFG) batch into the heads axis around the Ulysses exchange.
  # Each (batch, head) pair is an independent attention problem, so
  # [B, H, S, D] -> [1, B*H, S, D] is mathematically identity — but it makes
  # XLA compile the attention path as the batch=1 case. At batch=2 XLA
  # otherwise places the size-2 batch in the tile sublanes ({3,0,1,2:T(2,128)}
  # instead of T(8,128)) which quadruples the cost of every op touching the
  # a2a tensors inside the scanned layers (measured 7.0 -> expected ~3.5
  # s/step at 720p 81f cp8 CFG).
  batch = query.shape[0]
  # Only foldable when nothing else shards the batch: with data/fsdp > 1 the
  # shard_map still maps axis 0 onto those mesh axes, and a folded (size-1)
  # batch is not divisible by them. Those configs are already batch=1 per
  # device inside the shard_map, so they never hit the T(2,128) tiling problem
  # the fold exists to avoid.
  # Folding batch into heads destroys the one-mask-per-example association.
  # Keep the optimization for the common unmasked path only.
  fold_batch = (
      attention_mask is None
      and batch > 1
      and devices_in_batch_sharding == 1
      and num_q_heads == num_kv_heads
      and (batch * num_heads) % num_shards == 0
  )
  if fold_batch:
    query = query.reshape(1, batch * num_heads, *query.shape[2:])
    key = key.reshape(1, batch * num_heads, *key.shape[2:])
    value = value.reshape(1, batch * num_heads, *value.shape[2:])
    effective_num_heads = batch * num_heads
  else:
    effective_num_heads = num_heads

  wan_splash_transpose_out = wan_runtime_options.get("wan_splash_transpose_out")
  out_q_axis_names = (
      q_axis_names
      if (not use_custom_kernel or wan_splash_transpose_out)
      else jax.sharding.PartitionSpec(q_axis_names[0], q_axis_names[1], q_axis_names[3], q_axis_names[2])
  )

  if attention_mask is None:
    sharded_ulysses_attention = jax.shard_map(
        lambda q, k, v: wrap_ulysses_attention(q, k, v, None),
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=out_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_attention(q, k, v):
      return sharded_ulysses_attention(q, k, v)

  else:
    sharded_ulysses_attention = jax.shard_map(
        wrap_ulysses_attention,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names, mask_axis_names),
        out_specs=out_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_attention(q, k, v):
      return sharded_ulysses_attention(q, k, v, attention_mask)

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      effective_num_heads,
      num_shards,
      ulysses_attention_chunks,
      run_ulysses_attention,
  )

  if use_custom_kernel and not wan_splash_transpose_out:
    if fold_batch:
      x = x.reshape(batch, num_heads, *x.shape[2:])
    x = x[:, :, :, :orig_q_seq_len]
    b, h, d, s = x.shape
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape(b, -1, h * d)
    axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
    x = jax.lax.with_sharding_constraint(x, axis_names)
  else:
    if fold_batch:
      x = x.reshape(batch, num_heads, *x.shape[2:])
    x = x[:, :, :orig_q_seq_len, :]
    x = _reshape_heads_to_head_dim(x)

  return x


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
    ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
    ring_axis: str = INTERNAL_RING_AXIS,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    ulysses_shards: int = -1,
    ulysses_attention_chunks: int = 1,
    preserve_asymmetric_block_sizes: bool = False,
    kv_heads: int | None = None,
) -> jax.Array:
  """2D context-parallel attention using a private Ulysses x ring mesh.

  Public configs only shard sequence on the context axis.  Internally this
  reshapes that same device axis into hidden ring and Ulysses axes, runs the
  Ulysses all-to-all over the hidden Ulysses axis, and rotates K/V over the
  hidden ring axis.
  """
  if kv_heads is None:
    kv_heads = heads

  context_axis = CONTEXT
  if context_axis not in mesh.shape:
    raise ValueError(f"Ulysses ring attention requires mesh axis {context_axis!r}, got mesh axes {mesh.shape}.")

  num_context_shards = mesh.shape[context_axis]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("Ulysses ring attention requires ulysses_shards to be set from config or command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the requested Ulysses shard count to divide the context shard count, "
        f"got context_shards={num_context_shards} and ulysses_shards={num_ulysses_shards}."
    )
  if heads % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the number of query heads to be divisible by the requested Ulysses shard count, "
        f"got heads={heads} and ulysses_shards={num_ulysses_shards}."
    )
  if kv_heads % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the number of KV heads to be divisible by the requested Ulysses shard count, "
        f"got kv_heads={kv_heads} and ulysses_shards={num_ulysses_shards}."
    )
  num_ring_shards = num_context_shards // num_ulysses_shards
  _warn_if_ring_is_degenerate(
      num_ring_shards,
      num_ulysses_shards,
      num_context_shards,
      heads=heads,
      kv_heads=kv_heads,
  )
  internal_mesh = _create_internal_ulysses_ring_mesh(
      mesh,
      ring_shards=num_ring_shards,
      ulysses_shards=num_ulysses_shards,
      ring_axis=ring_axis,
      ulysses_axis=ulysses_axis,
  )
  internal_sequence_axes = (ring_axis, ulysses_axis)
  num_sequence_shards = num_context_shards

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_sequence_shards)
  key, _ = _reshape_data_for_flash(key, kv_heads, num_sequence_shards)
  value, _ = _reshape_data_for_flash(value, kv_heads, num_sequence_shards)
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  num_heads = query.shape[1]

  block_sizes = _select_flash_block_sizes(
      query,
      key,
      flash_block_sizes,
      dtype,
      "tokamax_ring",
      preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
  )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  internal_q_axis_names = _replace_mesh_axis_names(q_axis_names, context_axis, internal_sequence_axes)
  internal_kv_axis_names = _replace_mesh_axis_names(kv_axis_names, context_axis, internal_sequence_axes)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))
  internal_mask_axis_names = _replace_mesh_axis_names(mask_axis_names, context_axis, internal_sequence_axes)
  mask_needs_ulysses_gather = _mesh_axis_in_spec(internal_mask_axis_names[1], ulysses_axis)

  def wrap_ulysses_ring_attention(query, key, value, attention_mask):
    # Swap sharding: each device gives up a slice of heads and gathers
    # a slice of sequence, so the local kernel sees the full sequence.
    query = jax.lax.all_to_all(query, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    key = jax.lax.all_to_all(key, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    value = jax.lax.all_to_all(value, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    if attention_mask is not None and mask_needs_ulysses_gather:
      attention_mask = jax.lax.all_gather(attention_mask, axis_name=ulysses_axis, axis=1, tiled=True)

    uses_fused_kernel = block_sizes.use_fused_bwd_kernel
    block_q_sizes = (block_sizes.block_q, block_sizes.block_q_dkv)
    block_kv_sizes = (block_sizes.block_kv, block_sizes.block_kv_dkv)
    if uses_fused_kernel:
      block_q_sizes += (block_sizes.block_q_dkv,)
      block_kv_sizes += (block_sizes.block_kv_dkv,)
    else:
      block_q_sizes += (block_sizes.block_q_dq,)
      block_kv_sizes += (block_sizes.block_kv_dq,)

    block_q = max(*block_q_sizes)
    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)
    block_kv = max(*block_kv_sizes)
    key, _, key_seq_len = _pad_data_for_flash(key, kv_heads, block_kv)
    value, _, _ = _pad_data_for_flash(value, kv_heads, block_kv)

    q_padded_len = query.shape[2]
    kv_padded_len = key.shape[2]
    total_kv_len = kv_padded_len * num_ring_shards

    # Mask q/kv padding via segment ids, same as the tokamax_ring kernel.
    # Padding-only IDs are identical per shard; explicit KV masks rotate.
    segment_ids = _build_padding_segment_ids(
        query_seq_len,
        q_padded_len,
        key_seq_len,
        kv_padded_len,
        attention_mask,
        tokamax_splash_base.SegmentIds,
    )

    if not mask_padding_tokens and attention_mask is None:
      segment_ids = None

    mask = tokamax_splash_attention_mask.FullMask(_shape=(q_padded_len, total_kv_len))

    splash_kernel = tokamax_ring_attention_kernel.make_ring_attention(
        mask=mask,
        is_mqa=False,
        config=convert_to_tokamax_splash_config(
            block_sizes,
            residual_checkpoint_name=residual_checkpoint_name,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
        ),
        save_residuals=False,
        ring_axis=ring_axis,
        kv_seq_shards=num_ring_shards,
        rotate_segment_ids=attention_mask is not None,
    )
    segment_ids_in_axes = 0 if attention_mask is not None else None
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
    attention_output = vmapped_splash(query, key, value, segment_ids)
    attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
    attention_output = jax.lax.all_to_all(
        attention_output,
        axis_name=ulysses_axis,
        split_axis=2,
        concat_axis=1,
        tiled=True,
    )
    return attention_output

  devices_in_batch_sharding = mesh.shape["data"] * (mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1)
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_ulysses_ring_attention = jax.shard_map(
        lambda q, k, v: wrap_ulysses_ring_attention(q, k, v, None),
        mesh=internal_mesh,
        in_specs=(
            internal_q_axis_names,
            internal_kv_axis_names,
            internal_kv_axis_names,
        ),
        out_specs=internal_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_ring_attention(q, k, v):
      return sharded_ulysses_ring_attention(q, k, v)

  else:
    sharded_ulysses_ring_attention = jax.shard_map(
        wrap_ulysses_ring_attention,
        mesh=internal_mesh,
        in_specs=(
            internal_q_axis_names,
            internal_kv_axis_names,
            internal_kv_axis_names,
            internal_mask_axis_names,
        ),
        out_specs=internal_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_ring_attention(q, k, v):
      return sharded_ulysses_ring_attention(q, k, v, attention_mask)

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      num_heads,
      num_ulysses_shards,
      ulysses_attention_chunks,
      run_ulysses_ring_attention,
  )
  x = jax.lax.with_sharding_constraint(x, q_axis_names)
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)

  return x


def _slice_own_ulysses_heads(x: jax.Array, ulysses_axis: str, num_ulysses_shards: int, axis: int) -> jax.Array:
  """Slices an all-heads array down to the heads this rank owns after the a2a.

  `all_to_all(split_axis=1, concat_axis=2, tiled=True)` hands rank `r` the head
  block `[r * H/U, (r+1) * H/U)`, so the same static block size with a
  rank-dependent offset recovers exactly the heads the rank now holds.
  """
  heads_per_dev = x.shape[axis] // num_ulysses_shards
  start = jax.lax.axis_index(ulysses_axis) * heads_per_dev
  return jax.lax.dynamic_slice_in_dim(x, start, heads_per_dev, axis=axis)


def _ring_fixed_m_norms_pre_a2a(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    ulysses_axis: str,
    ring_axis: str,
    num_ulysses_shards: int,
    num_ring_shards: int,
    block_q: int,
    per_q_block: bool,
    use_k_centering: bool = False,
    token_padding: Optional[TokenPadding] = None,
):
  """Computes all R>1 fixed-m norms and global eligibility predicates *pre* a2a.

  Inputs are the shard-local activations as they arrive from the QKV
  projections: `[B, H_all, S/(U*R), D]` -- every head, a 1/U slice of this ring
  shard's sequence. The equivalent post-a2a arrays are `[B, H_all/U, S/R, D]`:
  the same elements, redistributed. Both forms therefore admit the same
  reductions, but doing them here (including `v_max_sq`, `v_ok`, and
  `all_fixed_global`) is materially cheaper:

    * Every reduction reads the projection's natural output layout. After the
      all-to-all the arrays carry the collective's layout, and XLA inserts
      relayout copies to feed post-a2a reductions.
    * All reductions and the cross-chip `pmax` (and, with K-centering, `pmean`)
      collectives become independent of `all_to_all(query, key, value)`,
      allowing XLA's latency-hiding scheduler to overlap them with the
      all-to-all instead of serialising reductions and collectives between the
      all-to-all and the `lax.cond`. With `per_q_block=True` a small
      ulysses-axis `all_to_all` of the Q row norms is also issued.

  Note: callers must apply `jax.lax.optimization_barrier((query, key, value))`
  in the outer scope so both this function and the subsequent `all_to_all`
  consume the exact same barriered tensors.

  With `token_padding`, the pad rows at the tail of each ring segment (the tail
  of the last Ulysses ranks' local rows) are left out of every reduction: the
  kernel masks those keys and the model drops those queries, so they must not
  move the bounds. Pad Q/K rows are zero after RoPE anyway; pad V rows are not.

  Returns `(key_out, qn_dev, mk_all_sq_dev, v_ok, all_fixed_global)` sliced
  to the heads this Ulysses rank owns and ready for immediate `jax.lax.cond`
  dispatch post-a2a.
  """
  reduce_axes = (ulysses_axis, ring_axis)
  key_f32 = key.astype(jnp.float32)

  row_ok = None
  if token_padding is not None:
    local_len = key.shape[2]
    segment_row = jax.lax.axis_index(ulysses_axis) * local_len + jnp.arange(local_len)
    row_ok = segment_row < token_padding.real_len

  q_norm_sq = (query.astype(jnp.float32) ** 2).sum(axis=-1)
  v_sq = value.astype(jnp.float32) ** 2
  if row_ok is not None:
    q_norm_sq = jnp.where(row_ok[None, None, :], q_norm_sq, 0.0)
    v_sq = jnp.where(row_ok[None, None, :, None], v_sq, 0.0)
  qn_head_local = q_norm_sq.max(axis=-1)
  vn_local = v_sq.max()

  if use_k_centering:
    # Optional K-centering: computes global mean and subtracts before a2a.
    if row_ok is None:
      k_mean_all = jax.lax.pmean(jnp.mean(key_f32, axis=2), axis_name=reduce_axes)
    else:
      k_sum = jnp.where(row_ok[None, None, :, None], key_f32, 0.0).sum(axis=2)
      k_mean_all = jax.lax.psum(k_sum, axis_name=reduce_axes) / (token_padding.real_len * num_ring_shards)
    centered_f32 = key_f32 - k_mean_all[:, :, None, :]
    key_out = centered_f32.astype(key.dtype)
    kn_rows = jnp.sum(key_out.astype(jnp.float32) ** 2, axis=-1)
  else:
    # High-performance uncentered path: key is completely untouched, so all-to-all
    # starts immediately in parallel with norm reductions, eliminating the pmean
    # collective, 194MB/layer HBM subtraction, and collective serialization.
    key_out = key
    kn_rows = jnp.sum(key_f32**2, axis=-1)
  if row_ok is not None:
    kn_rows = jnp.where(row_ok[None, None, :], kn_rows, 0.0)
  kn_local = kn_rows.max(axis=-1)

  # Global Q/V/K max norms in a SINGLE (ulysses, ring) pmax.
  #
  # K needs only the ring-wide max (see the `mk_all_sq` note below), so no
  # ring-axis `all_gather` of per-shard K norms is issued. Its payload would be
  # only `heads` floats; the cost is compute, not collective time. On TPU a
  # ring-axis gather of these small arrays forces a relayout in the middle of
  # the QKV projection's fusion region, and XLA then fails to fuse across it.
  # Measured on an earlier revision of this PR (tpu7x-8, 40-step denoise),
  # dropping the gather recovered ~2.4 s (convolution fusion -1.64 s, loop
  # fusion -0.44 s, data formatting -0.36 s).
  qn_head_global, vn_global, kn_global = jax.lax.pmax((qn_head_local, vn_local, kn_local), axis_name=reduce_axes)

  if not per_q_block:
    qn_dev = _slice_own_ulysses_heads(qn_head_global, ulysses_axis, num_ulysses_shards, axis=1)
  else:
    batch, num_q_heads, local_seq = q_norm_sq.shape
    post_a2a_seq = local_seq * num_ulysses_shards
    num_q_blocks = -(-post_a2a_seq // block_q)
    padded_seq = num_q_blocks * block_q
    q_norm_sq_dev = jax.lax.all_to_all(q_norm_sq, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    pad_len = padded_seq - post_a2a_seq
    if pad_len > 0:
      q_norm_sq_dev = jnp.pad(q_norm_sq_dev, ((0, 0), (0, 0), (0, pad_len)))
    qn_dev = q_norm_sq_dev.reshape(batch, num_q_heads // num_ulysses_shards, num_q_blocks, block_q).max(axis=-1)

  # `mk_all_sq` is broadcast to (batch, ring, heads) only to take the ring
  # kernel's 2-D branch, which collapses it with `max(axis=0)`; every row is the
  # ring-wide max. Using the max over all K shards can only enlarge the
  # Cauchy-Schwarz bound, so fixed-m eligibility is more conservative, never
  # less, than a per-shard gate.
  mk_all_sq = jnp.broadcast_to(kn_global[:, None, :], (kn_global.shape[0], num_ring_shards, kn_global.shape[1]))

  # Slice down to the heads owned by this Ulysses rank post-a2a.
  qn_head_global_dev = _slice_own_ulysses_heads(qn_head_global, ulysses_axis, num_ulysses_shards, axis=1)
  mk_all_sq_dev = _slice_own_ulysses_heads(mk_all_sq, ulysses_axis, num_ulysses_shards, axis=2)
  vn_dev = vn_global

  num_q_heads_dev = qn_dev.shape[1]
  num_kv_heads_dev = mk_all_sq_dev.shape[2]
  if num_q_heads_dev != num_kv_heads_dev:
    if num_q_heads_dev % num_kv_heads_dev != 0:
      raise ValueError(
          f"num_q_heads ({num_q_heads_dev}) must be divisible by num_kv_heads ({num_kv_heads_dev}) for GQA ring fixed-m."
      )
    q_heads_per_kv_head = num_q_heads_dev // num_kv_heads_dev
    mk_all_sq_dev = jnp.repeat(mk_all_sq_dev, q_heads_per_kv_head, axis=2)

  # Evaluate global V safety and Cauchy-Schwarz fixed-m eligibility pre-a2a.
  # Because qn_head_global_dev, vn_dev, and mk_global_sq come from the single
  # (ulysses, ring) pmax above, v_ok and all_fixed_global are bit-identical
  # across all ring ranks with no further collectives. all_fixed_global is
  # also reduced over the batch, so it is an unbatched scalar under the ring
  # kernel's jax.vmap.
  effective_kv_seq_len = key.shape[2] * num_ulysses_shards * num_ring_shards
  if token_padding is not None:
    effective_kv_seq_len = token_padding.real_len * num_ring_shards
  global_recenter, global_bound = custom_splash.get_fixed_m_constants(effective_kv_seq_len)
  global_bound_sq = global_bound**2
  dtype_safe = custom_splash.fixed_m_dtype_is_safe(query.dtype, global_recenter)

  v_max_sq = vn_dev.max()
  v_ok = (v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2)) & dtype_safe
  mk_global_sq = mk_all_sq_dev.max(axis=1)

  bound_head_sq = qn_head_global_dev * mk_global_sq
  all_fixed_global = jnp.all(bound_head_sq <= global_bound_sq) & v_ok
  return key_out, qn_dev, mk_all_sq_dev, v_ok, all_fixed_global


def _compute_fixed_m_metadata(
    query: jax.Array,
    key: jax.Array,
    block_q: int,
    safe_bound: float | None = None,
    recenter: float | None = None,
    per_q_block: bool = True,
    k_mean: jax.Array | None = None,
    value: jax.Array | None = None,
    v_max_bound: float = 256.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes Cauchy-Schwarz norm bounds and per-Q-block (or per-head) fixed-m metadata.

  Args:
    query: Padded query activation, shape `(batch, local_q_heads, padded_q_len, head_dim)`.
    key: Key activation (raw unpadded or padded), shape `(batch, local_kv_heads, kv_len, head_dim)`.
      K norms are computed per KV head and repeated internally to Q heads for GQA.
    block_q: Query tile block size.
    safe_bound: Maximum safe norm product threshold before falling back to online softmax.
    recenter: Fixed-m dynamic recenter constant C(N).
    per_q_block: If True, evaluates gating independently per query tile. If False,
      evaluates monolithic gating per head.
    k_mean: Optional mean key vector for Virtual K-centering, KV-head indexed,
      shape `(batch, >= local_kv_heads, >= head_dim)`. It is not validated: it is
      sliced to `k_mean[:, :local_kv_heads, :head_dim]`, so a longer (e.g.
      pre-padded or Q-head-expanded) array is silently truncated, not rejected.
    value: Optional value activation, shape `(batch, local_kv_heads, kv_len, head_dim_v)`, used to
      verify that |V| <= v_max_bound to guarantee against FP32 overflow.
    v_max_bound: Maximum safe value magnitude (default 256.0).

  Returns:
    mk_arr: Gating metadata array of shape `(batch, 2, local_q_heads, num_q_blocks)`
      multiplexing precomputed block base shifts and binary fixed-m gating predicates into a single
      Pallas scalar prefetch memory slot:
        - `mk_arr[:, 0, h, i]`: Precomputed block base shift m_B = ceil(max_i ||q_i|| * max_j ||k_j||) - C.
        - `mk_arr[:, 1, h, i]`: Discrete eligibility predicate (1.0 for fixed-m, 0.0 for online).
    all_fixed: Boolean scalar indicating if all elements are eligible for uniform fixed-m.
  """
  batch_size, num_q_heads, q_len, _ = query.shape
  num_kv_heads = key.shape[1]
  if safe_bound is None or recenter is None:
    rec, bnd = custom_splash.get_fixed_m_constants(key.shape[2], v_max_bound=v_max_bound)
    if safe_bound is None:
      safe_bound = bnd
    if recenter is None:
      recenter = rec
  safe_bound_sq = safe_bound**2
  if k_mean is not None:
    centered_k = key.astype(jnp.float32) - k_mean[:, :num_kv_heads, None, : key.shape[-1]]
    mk_h_sq = (centered_k**2).sum(axis=-1).max(axis=-1)
  else:
    mk_h_sq = (key.astype(jnp.float32) ** 2).sum(axis=-1).max(axis=-1)  # (batch, num_kv_heads)

  if num_q_heads != num_kv_heads:
    if num_q_heads % num_kv_heads != 0:
      raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA fixed-m.")
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    mk_h_sq = jnp.repeat(mk_h_sq, q_heads_per_kv_head, axis=1)  # (batch, num_q_heads)

  # Fixed-m weights reach 2**recenter before being narrowed to the activation
  # dtype for the S@V matmul. If that dtype's exponent range cannot hold them
  # (fp16, fp8), the FP32 bound analysis is irrelevant -- the narrowing itself
  # overflows to inf -- so disqualify every head up front.
  dtype_safe = custom_splash.fixed_m_dtype_is_safe(query.dtype, recenter)
  v_ok = 1.0 if dtype_safe else 0.0
  if dtype_safe and value is not None:
    v_max_sq = (value.astype(jnp.float32) ** 2).max()
    v_ok = (v_max_sq <= (v_max_bound**2)).astype(jnp.float32)

  # The kernel's grid is ceil(q_len / block_q); callers pad Q to a multiple of
  # block_q first, so floor == ceil here. Fail loudly if that contract breaks
  # rather than silently dropping the ragged tail's gating metadata.
  if q_len % block_q != 0:
    raise ValueError(
        f"_compute_fixed_m_metadata expects query padded to a multiple of block_q, got q_len={q_len}, block_q={block_q}."
    )
  num_q_blocks = q_len // block_q
  if per_q_block:
    norm_sq = (query.astype(jnp.float32) ** 2).sum(axis=-1)  # (batch, num_q_heads, q_len)
    qn_max_sq = norm_sq.reshape(batch_size, num_q_heads, num_q_blocks, block_q).max(
        axis=-1
    )  # (batch, num_q_heads, num_q_blocks)
    bound_sq = qn_max_sq * mk_h_sq[:, :, None]
    fixed_ok = (bound_sq <= safe_bound_sq).astype(jnp.float32) * v_ok
    m_base = jnp.ceil(jnp.sqrt(bound_sq)) - recenter
    mk_arr = jnp.stack([m_base, fixed_ok], axis=1)  # (batch, 2, num_q_heads, num_q_blocks)
    all_fixed = jnp.all(fixed_ok > 0.5)
  else:
    qn_max_sq = (query.astype(jnp.float32) ** 2).sum(axis=-1).max(axis=-1)  # (batch, num_q_heads)
    bound_sq_1d = qn_max_sq * mk_h_sq
    fixed_ok_1d = (bound_sq_1d <= safe_bound_sq).astype(jnp.float32) * v_ok
    m_base_1d = jnp.ceil(jnp.sqrt(bound_sq_1d)) - recenter
    m_base_expanded = jnp.broadcast_to(m_base_1d[:, :, None], (batch_size, num_q_heads, num_q_blocks))
    fixed_ok_expanded = jnp.broadcast_to(fixed_ok_1d[:, :, None], (batch_size, num_q_heads, num_q_blocks))
    mk_arr = jnp.stack([m_base_expanded, fixed_ok_expanded], axis=1)  # (batch, 2, num_q_heads, num_q_blocks)
    all_fixed = jnp.all(fixed_ok_1d > 0.5)

  return mk_arr, all_fixed


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
    per_q_block: bool = True,
    kv_heads: int | None = None,
    use_k_centering: bool = False,
    qk_prescaled: bool = False,
    wan_ulysses_out_a2a: Optional[str] = None,
) -> jax.Array:
  """2D USP attention (Ulysses + Ring) using custom splash kernel with exact Fixed-m support."""
  if kv_heads is None:
    kv_heads = heads

  if attention_mask is not None:
    raise NotImplementedError("ulysses_ring_custom does not support attention_mask.")
  axis_name = "context"
  num_context_shards = mesh.shape[axis_name]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("ulysses_ring_custom requires ulysses_shards to be set from config or command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        f"ulysses_ring_custom requires ulysses_shards to divide the context shard count, "
        f"got context_shards={num_context_shards} and ulysses_shards={num_ulysses_shards}."
    )
  num_ring_shards = num_context_shards // num_ulysses_shards

  # K-centering for ring variants: `use_k_centering` is resolved by the caller
  # with `resolve_k_centering(..., ring=True)`, so "auto" is OFF regardless of R.
  # When forced on, K is *materially* centered (R>1: `K - pmean(mean(K))` over
  # (ulysses, ring) before the a2a; R==1: `K - mean(K)` after the a2a), and the
  # norms are taken from the centered K. No `k_mean` is passed to the kernel.
  _warn_if_ring_is_degenerate(
      num_ring_shards,
      num_ulysses_shards,
      num_context_shards,
      heads=heads,
      kv_heads=kv_heads,
  )

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, orig_kv_seq_len = _reshape_data_for_flash(key, kv_heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, kv_heads, num_context_shards)
  num_heads = query.shape[1]
  if num_heads % num_ulysses_shards != 0:
    raise ValueError(f"Ulysses+Ring requires query heads divisible by U={num_ulysses_shards}, got heads={num_heads}.")
  if kv_heads % num_ulysses_shards != 0:
    raise ValueError(f"Ulysses+Ring requires KV heads divisible by U={num_ulysses_shards}, got kv_heads={kv_heads}.")
  if num_ring_shards > 1 and (orig_q_seq_len % num_context_shards != 0 or orig_kv_seq_len % num_context_shards != 0):
    raise ValueError(
        f"2D Ulysses+Ring attention requires sequence length to be divisible by context_shards={num_context_shards}, "
        f"got orig_q_seq_len={orig_q_seq_len}, orig_kv_seq_len={orig_kv_seq_len}."
    )
  token_padding = _active_token_padding(orig_kv_seq_len, num_segments=num_ring_shards)

  (
      bq,
      bkv,
      bkv_compute,
      bkv_compute_in,
      heads_per_tile,
      vmem_limit_bytes,
  ) = _extract_custom_block_sizes(flash_block_sizes)
  if heads_per_tile > 1 and num_ring_shards > 1:
    raise NotImplementedError("heads_per_tile > 1 is not supported for multi-shard ring attention.")
  internal_mesh = _create_internal_ulysses_ring_mesh(mesh, num_ring_shards, num_ulysses_shards)
  ring_axis, ulysses_axis = INTERNAL_RING_AXIS, INTERNAL_ULYSSES_AXIS
  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  internal_q_axis_names = _replace_mesh_axis_names(q_axis_names, axis_name, (ring_axis, ulysses_axis))
  internal_kv_axis_names = _replace_mesh_axis_names(kv_axis_names, axis_name, (ring_axis, ulysses_axis))

  @functools.partial(
      jax.shard_map,
      mesh=internal_mesh,
      in_specs=(
          internal_q_axis_names,
          internal_kv_axis_names,
          internal_kv_axis_names,
      ),
      out_specs=internal_q_axis_names,
      check_vma=False,
  )
  def wrap_ulysses_ring_attention(query, key, value):
    # Apply the base-2 rescale of Q *before* the all-to-all. A scalar elementwise
    # multiply commutes exactly with the collective (which is pure data movement),
    # so this is bit-identical. Done after the a2a it sat between the collective
    # and the kernel and XLA wrapped it in relayout copies; done before, it fuses
    # into the producer of Q and its 185MB round-trip disappears.
    if use_base2_exp and not qk_prescaled:
      query = query * LOG2E

    # (0) R>1 fixed-m reductions and global eligibility predicates, computed
    # entirely on the pre-a2a layout so zero reductions or collectives sit
    # between `all_to_all` and `jax.lax.cond`.
    qn_dev, mk_all_sq, v_ok, all_fixed_global = None, None, None, None
    if use_fixed_m and num_ring_shards > 1:
      query, key, value = jax.lax.optimization_barrier((query, key, value))
      (
          key,
          qn_dev,
          mk_all_sq,
          v_ok,
          all_fixed_global,
      ) = _ring_fixed_m_norms_pre_a2a(
          query,
          key,
          value,
          ulysses_axis=ulysses_axis,
          ring_axis=ring_axis,
          num_ulysses_shards=num_ulysses_shards,
          num_ring_shards=num_ring_shards,
          block_q=bq,
          per_q_block=per_q_block,
          use_k_centering=use_k_centering,
          token_padding=token_padding,
      )

    # (1) Ulysses All-to-All: heads -> sequence
    a2a = functools.partial(jax.lax.all_to_all, axis_name=ulysses_axis, tiled=True)
    query = a2a(query, split_axis=1, concat_axis=2)
    key = a2a(key, split_axis=1, concat_axis=2)
    value = a2a(value, split_axis=1, concat_axis=2)

    # NOTE: the base-2 rescale of Q is applied before the all-to-all above.
    raw_key = key
    raw_query = query
    raw_value = value
    context_q_seq_len = raw_query.shape[2]
    actual_kv_seq_len = (
        (token_padding.real_len if token_padding is not None else orig_kv_seq_len)
        if num_ring_shards == 1
        else (token_padding.real_len if token_padding is not None else raw_key.shape[2])
    )

    if use_fixed_m and num_ring_shards == 1 and use_k_centering:
      # Optional K-centering: Center key directly in JAX so Pallas kernel runs with pristine 4 operands
      kbar = jnp.mean(
          raw_key[:, :, :actual_kv_seq_len, :].astype(jnp.float32),
          axis=2,
          keepdims=True,
      )
      raw_key = (raw_key.astype(jnp.float32) - kbar).astype(raw_key.dtype)
      real_key = raw_key[:, :, :actual_kv_seq_len, :]
    else:
      real_key = raw_key[:, :, :actual_kv_seq_len, :]

    query, kv_size, query_seq_len = _pad_data_for_flash(raw_query, heads, bq)
    # When actual_kv_seq_len is aligned to 8 sublanes, K/V are passed with NO
    # sequence padding. The kernel slices the ragged KV tail using slice lengths
    # derived from actual_kv_seq_len, avoiding sequence pad HBM copies and
    # redundant ppermute/MXU compute on padded keys.
    kv_pad_size = 1 if actual_kv_seq_len % 8 == 0 else bkv
    key, _, key_seq_len = _pad_data_for_flash(raw_key, kv_heads, kv_pad_size)
    value, _, _ = _pad_data_for_flash(raw_value, kv_heads, kv_pad_size)
    ring_kv_seq_len = actual_kv_seq_len if actual_kv_seq_len % 8 == 0 else key_seq_len

    mk_arr, all_fixed = None, None
    if use_fixed_m and num_ring_shards == 1:
      recenter, safe_bound = custom_splash.get_fixed_m_constants(actual_kv_seq_len)
      mk_arr, all_fixed = _compute_fixed_m_metadata(
          query,
          real_key,
          bq,
          safe_bound=safe_bound,
          recenter=recenter,
          per_q_block=per_q_block,
          k_mean=None,
          value=raw_value if token_padding is None else raw_value[:, :, :actual_kv_seq_len, :],
      )

    bsizes = custom_splash._BlockSizes(bq, bkv, bkv_compute, bkv_compute_in)

    # (2a) R=1: Dedicated single-device splash kernel with fixed-m or online softmax
    if num_ring_shards == 1:
      if use_fixed_m:
        splash_kernel_uniform = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=True,
            uniform_fixed_m=True,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )
        splash_kernel_hybrid = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=True,
            uniform_fixed_m=False,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )

        def _run_uniform(q, k, v, m):
          return jax.vmap(splash_kernel_uniform, in_axes=(0, 0, 0, 0))(q, k, v, m)

        def _run_hybrid(q, k, v, m):
          return jax.vmap(splash_kernel_hybrid, in_axes=(0, 0, 0, 0))(q, k, v, m)

        raw_out = jax.lax.cond(all_fixed, _run_uniform, _run_hybrid, query, key, value, mk_arr)
      else:
        splash_kernel = custom_splash.make_splash_mha(
            block_sizes=bsizes,
            orig_q_seq_len=context_q_seq_len,
            orig_kv_seq_len=actual_kv_seq_len,
            heads_per_tile=heads_per_tile,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=False,
            transpose_out=bool(wan_runtime_options.get("wan_splash_transpose_out")),
        )
        raw_out = jax.vmap(splash_kernel, in_axes=(0, 0, 0))(query, key, value)
      wan_splash_transpose_out = wan_runtime_options.get("wan_splash_transpose_out")
      if wan_splash_transpose_out:
        attention_output = raw_out
      else:
        attention_output = jnp.swapaxes(raw_out, 2, 3)

    # (2b) Ring: Cross-chip ppermute schedule with custom ring kernel
    else:
      if use_fixed_m:
        ring_kernel = tokamax_ring_attention_kernel.make_custom_ring_attention(
            block_sizes=bsizes,
            orig_q_seq_len=query_seq_len,
            orig_kv_seq_len=ring_kv_seq_len,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            ring_axis=ring_axis,
            ring_size=num_ring_shards,
            bidirectional=bidirectional,
            use_fixed_m=True,
            per_q_block=per_q_block,
            pregathered_mk=True,
            v_ok=v_ok,
            all_fixed_global=all_fixed_global,
        )
        attention_output = jax.vmap(ring_kernel, in_axes=(0, 0, 0, (0, 0)))(query, key, value, (qn_dev, mk_all_sq))
      else:
        ring_kernel = tokamax_ring_attention_kernel.make_custom_ring_attention(
            block_sizes=bsizes,
            orig_q_seq_len=query_seq_len,
            orig_kv_seq_len=ring_kv_seq_len,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            ring_axis=ring_axis,
            ring_size=num_ring_shards,
            bidirectional=bidirectional,
            use_fixed_m=False,
        )
        attention_output = jax.vmap(ring_kernel, in_axes=(0, 0, 0))(query, key, value)

    attention_output = attention_output[:, :, :context_q_seq_len, :kv_size].astype(query.dtype)

    # (3) Ulysses All-to-All back: sequence -> heads
    return _ulysses_seq_to_heads(
        attention_output,
        axis_name=ulysses_axis,
        seq_axis=2,
        num_shards=num_ulysses_shards,
        mode=wan_ulysses_out_a2a,
    )

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      heads,
      num_ulysses_shards,
      ulysses_attention_chunks,
      wrap_ulysses_ring_attention,
  )
  x = jax.lax.with_sharding_constraint(x, q_axis_names)
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)
  return x


def _apply_attention_dot(
    query: Array,
    key: Array,
    value: Array,
    dtype: jnp.dtype,
    heads: int,
    dim_head: int,
    scale: float,
    split_head_dim: bool,
    float32_qk_product: bool,
    use_memory_efficient_attention: bool,
    attention_mask: Array = None,
    kv_heads: int | None = None,
    qk_prescaled: bool = False,
    k_prescaled: bool = False,
    use_base2_exp: bool = False,
):
  """Apply Attention."""
  effective_kv_heads = kv_heads if kv_heads is not None else heads
  if split_head_dim:

    def _to_bshd(x: Array, n_heads: int) -> Array:
      """Normalise to [B, S, H, D].

      Callers that apply rotary embeddings hand us [B, H, S, D] (see
      `_unflatten_heads`), while the flat path supplies [B, S, H*D]. Only the
      latter can be reshaped into [B, S, H, D]; reinterpreting [B, H, S, D]
      that way keeps the shape legal but interleaves heads with tokens, so it
      corrupts the output silently. Transpose the 4-D case instead.
      """
      if x.ndim == 4:
        return jnp.swapaxes(x, 1, 2)
      return jnp.reshape(x, (x.shape[0], -1, n_heads, dim_head))

    query_states = _to_bshd(query, heads)
    key_states = _to_bshd(key, effective_kv_heads)
    value_states = _to_bshd(value, effective_kv_heads)
    if heads != effective_kv_heads:
      num_repeats = heads // effective_kv_heads
      key_states = jnp.repeat(key_states, num_repeats, axis=2)
      value_states = jnp.repeat(value_states, num_repeats, axis=2)
  else:
    query_states = _reshape_heads_to_batch_dim(query, heads)
    key_states = _reshape_heads_to_batch_dim(key, effective_kv_heads)
    value_states = _reshape_heads_to_batch_dim(value, effective_kv_heads)
    if heads != effective_kv_heads:
      num_repeats = heads // effective_kv_heads
      b = query.shape[0]
      s_k = key_states.shape[1]
      key_states = jnp.repeat(key_states.reshape(b, effective_kv_heads, s_k, -1), num_repeats, axis=1).reshape(
          b * heads, s_k, -1
      )
      value_states = jnp.repeat(
          value_states.reshape(b, effective_kv_heads, s_k, -1),
          num_repeats,
          axis=1,
      ).reshape(b * heads, s_k, -1)

  if float32_qk_product:
    query_states = query_states.astype(jnp.float32)
    key_states = key_states.astype(jnp.float32)

  if use_memory_efficient_attention and attention_mask is None:
    if k_prescaled or qk_prescaled:
      key_states = key_states / jnp.asarray(scale, dtype=key_states.dtype)
    if qk_prescaled and use_base2_exp:
      query_states = query_states / jnp.asarray(LOG2E, dtype=query_states.dtype)
    if not split_head_dim:
      query_states = query_states.transpose(1, 0, 2)
      key_states = key_states.transpose(1, 0, 2)
      value_states = value_states.transpose(1, 0, 2)

    # this if statement create a chunk size for each layer of the unet
    # the chunk size is equal to the query_length dimension of the deepest layer of the unet

    flatten_latent_dim = query_states.shape[-3]
    if flatten_latent_dim % 64 == 0:
      query_chunk_size = int(flatten_latent_dim / 64)
    elif flatten_latent_dim % 16 == 0:
      query_chunk_size = int(flatten_latent_dim / 16)
    elif flatten_latent_dim % 4 == 0:
      query_chunk_size = int(flatten_latent_dim / 4)
    else:
      query_chunk_size = int(flatten_latent_dim)

    hidden_states = jax_memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        query_chunk_size=query_chunk_size,
        key_chunk_size=4096 * 4,
    )

    if split_head_dim:
      b = hidden_states.shape[0]
      hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    else:
      hidden_states = hidden_states.transpose(1, 0, 2)
      hidden_states = _reshape_batch_dim_to_heads(hidden_states, heads)
    hidden_states = hidden_states.astype(dtype)
  else:
    preferred_element_type = jnp.float32 if float32_qk_product else None
    if split_head_dim:
      attention_scores = jnp.einsum(
          "b t n h, b f n h -> b n f t",
          key_states,
          query_states,
          preferred_element_type=preferred_element_type,
      )
    else:
      attention_scores = jnp.einsum(
          "b i d, b j d->b i j",
          query_states,
          key_states,
          preferred_element_type=preferred_element_type,
      )

    if qk_prescaled:
      if use_base2_exp:
        attention_scores = attention_scores / jnp.asarray(LOG2E, dtype=attention_scores.dtype)
    elif k_prescaled:
      pass
    elif scale != 1.0:
      attention_scores = attention_scores * scale
    if attention_mask is not None:
      attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)
    attention_probs = nn.softmax(attention_scores, axis=-1 if split_head_dim else 2)
    if attention_mask is not None:
      has_valid_key = jnp.any(attention_mask == 0, axis=-1, keepdims=True)
      attention_probs = jnp.where(has_valid_key, attention_probs, 0)

    attention_probs = attention_probs.astype(dtype)

    # attend to values
    if split_head_dim:
      hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value_states)
      b = hidden_states.shape[0]
      hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    else:
      hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
      hidden_states = _reshape_batch_dim_to_heads(hidden_states, heads)

  return hidden_states


def _cudnn_flash_attention(query: Array, key: Array, value: Array, heads: int, mesh: Mesh, dpa_layer: Callable) -> Array:
  """CUDNN Flash Attention with Transformer Engine.
  1. Stable API, supports GQA
  2. Supports head_dim till 128; head_dim=256 support will be added soon
  """
  # These imports are only meant to work in a GPU build.
  # copied from tpu_flash_attention
  query = _reshape_data_for_cudnn_flash(query, heads)
  key = _reshape_data_for_cudnn_flash(key, heads)
  value = _reshape_data_for_cudnn_flash(value, heads)

  query = nn.with_logical_constraint(query, (BATCH, LENGTH, HEAD, D_KV))
  key = nn.with_logical_constraint(key, (BATCH, LENGTH, HEAD, D_KV))
  value = nn.with_logical_constraint(value, (BATCH, LENGTH, HEAD, D_KV))

  out = dpa_layer(query, key, value, mask=None)
  return _reshape_data_from_cudnn_flash(out)


KERNEL_REGISTRY = {}


def register_kernel(name: str):
  def decorator(func):
    KERNEL_REGISTRY[name] = func
    return func

  return decorator


# Register existing kernels at module level with context dict
@register_kernel("dot_product")
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
      kv_heads=context.get("kv_heads", None),
      qk_prescaled=context.get("qk_prescaled", False),
      k_prescaled=context.get("k_prescaled", False),
      use_base2_exp=context.get("use_base2_exp", False),
  )


@register_kernel("ulysses_custom")
def ulysses_custom_kernel(q, k, v, context):
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_custom_kernel=True,
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      spatiotemporal_config=context.get("spatiotemporal_config"),
      spatiotemporal_shape=context.get("spatiotemporal_shape"),
      kv_heads=context.get("kv_heads", None),
      ulysses_shards=context.get("ulysses_shards", -1),
      kernel_name="ulysses_custom",
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_ring_custom")
def ulysses_ring_custom_kernel(q, k, v, context):
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_ring_custom_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      kv_heads=context.get("kv_heads", None),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_ring_custom_fixed_m")
def ulysses_ring_custom_fixed_m_kernel(q, k, v, context):
  """fixed-m variant of ulysses_ring_custom with monolithic per-head gating."""
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_ring_custom_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
      per_q_block=False,
      ulysses_attention_chunks=context.get("ulysses_attention_chunks", 1),
      kv_heads=context.get("kv_heads", None),
      use_k_centering=resolve_k_centering(context.get("use_k_centering"), ring=True),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_ring_custom_fixed_m_per_q_block")
def ulysses_ring_custom_fixed_m_per_q_block_kernel(q, k, v, context):
  """fixed-m variant of ulysses_ring_custom with per-Q-block gating."""
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_ring_custom_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
      per_q_block=True,
      ulysses_attention_chunks=context.get("ulysses_attention_chunks", 1),
      kv_heads=context.get("kv_heads", None),
      use_k_centering=resolve_k_centering(context.get("use_k_centering"), ring=True),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_ring_custom_bidir")
def ulysses_ring_custom_bidir_kernel(q, k, v, context):
  """Wrap-free (bidirectional) variant of ulysses_ring_custom: the ring streams
  K/V both directions one hop at a time, avoiding the diameter-length wrap hop
  on a non-wrapping ring axis. Same USP split as ulysses_ring_custom otherwise."""
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_ring_custom_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      bidirectional=True,
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      kv_heads=context.get("kv_heads", None),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_custom_fixed_m")
def ulysses_custom_fixed_m_kernel(q, k, v, context):
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_custom_kernel=True,
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
      per_q_block=False,
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      kv_heads=context.get("kv_heads", None),
      ulysses_shards=context.get("ulysses_shards", -1),
      kernel_name="ulysses_custom_fixed_m",
      use_k_centering=resolve_k_centering(context.get("use_k_centering"), ring=False),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_custom_fixed_m_per_q_block")
def ulysses_custom_fixed_m_per_q_block_kernel(q, k, v, context):
  qk_prescaled = context.get("qk_prescaled", False)
  return _ulysses_attention(
      q,
      k if (qk_prescaled or context.get("k_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_custom_kernel=True,
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
      per_q_block=True,
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      kv_heads=context.get("kv_heads", None),
      ulysses_shards=context.get("ulysses_shards", -1),
      kernel_name="ulysses_custom_fixed_m_per_q_block",
      use_k_centering=resolve_k_centering(context.get("use_k_centering"), ring=False),
      qk_prescaled=qk_prescaled,
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses")
def ulysses_kernel(q, k, v, context):
  return _ulysses_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
      kv_heads=context.get("kv_heads", None),
      ulysses_shards=context.get("ulysses_shards", -1),
      kernel_name="ulysses",
      wan_ulysses_out_a2a=context.get("wan_ulysses_out_a2a"),
  )


@register_kernel("ulysses_ring")
def ulysses_ring_kernel(q, k, v, context):
  return _ulysses_ring_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      ulysses_shards=context["ulysses_shards"],
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
      kv_heads=context.get("kv_heads", None),
  )


@register_kernel("flash")
def flash_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="flash",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("tokamax_flash")
def tokamax_flash_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_flash",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
      spatiotemporal_config=context.get("spatiotemporal_config"),
      spatiotemporal_shape=context.get("spatiotemporal_shape"),
  )


@register_kernel("tokamax_ring")
def tokamax_ring_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_ring",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("tokamax_ring_custom")
def tokamax_ring_custom_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k if (context.get("k_prescaled", False) or context.get("qk_prescaled", False)) else k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_ring_custom",
      mask_padding_tokens=context["mask_padding_tokens"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("cudnn_flash_te")
def cudnn_flash_te_kernel(q, k, v, context):
  if context.get("k_prescaled", False) or context.get("qk_prescaled", False):
    k = k / jnp.asarray(context["scale"], dtype=k.dtype)
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
    is_causal: bool = False,
    preserve_asymmetric_block_sizes: bool = False,
    spatiotemporal_config: Optional[dict] = None,
    spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
    kv_heads: Optional[int] = None,
    use_k_centering: bool | str = "auto",
    qk_prescaled: bool = False,
    k_prescaled: bool = False,
    wan_ulysses_out_a2a: Optional[str] = None,
):
  """Routes to different attention kernels using a module-level registry."""

  _check_attention_inputs(query, key, value)
  seq_len_idx = 1
  if query.ndim == 4:
    seq_len_idx = 2

  can_use_flash_attention = True
  if attention_kernel in [
      "flash",
      "tokamax_flash",
      "ulysses",
      "ulysses_custom",
      "ulysses_custom_fixed_m",
      "ulysses_custom_fixed_m_per_q_block",
      "ulysses_ring",
      "ulysses_ring_custom",
      "ulysses_ring_custom_fixed_m",
      "ulysses_ring_custom_fixed_m_per_q_block",
      "ulysses_ring_custom_bidir",
  ]:
    can_use_flash_attention = (
        query.shape[seq_len_idx] >= flash_min_seq_length
        and key.shape[seq_len_idx] >= flash_min_seq_length
        and value.shape[seq_len_idx] >= flash_min_seq_length
    )

  effective_attention_kernel = attention_kernel
  if attention_kernel == "dot_product" or use_memory_efficient_attention or not can_use_flash_attention:
    effective_attention_kernel = "dot_product"

  # Masks enter the dispatcher as canonical [B, K] keep masks. Adapt them
  # only after fallback selection because a configured flash kernel may use
  # dot-product attention for short sequences.
  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
    attention_mask = attention_mask.astype(jnp.bool_)
    if effective_attention_kernel == "dot_product":
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
      "kv_heads": kv_heads,
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
      "is_causal": is_causal,
      "preserve_asymmetric_block_sizes": preserve_asymmetric_block_sizes,
      "spatiotemporal_config": spatiotemporal_config,
      "spatiotemporal_shape": spatiotemporal_shape,
      "use_k_centering": use_k_centering,
      "qk_prescaled": qk_prescaled,
      "k_prescaled": k_prescaled,
      "wan_ulysses_out_a2a": wan_ulysses_out_a2a,
  }

  if spatiotemporal_config and spatiotemporal_config.get("use_svg_attention"):
    if effective_attention_kernel not in (
        "ulysses_custom",
        "ulysses_custom_fixed_m",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
    ):
      raise ValueError("Head-local SVG requires a custom Ulysses attention backend.")
    # Dense uses its configured ring split; SVG exchanges over the full context axis.
    return _head_local_svg_attention(query, key, value, context)

  # Module-level Registry lookup
  if effective_attention_kernel in KERNEL_REGISTRY:
    with jax.named_scope(f"kernel_{effective_attention_kernel}"):
      return KERNEL_REGISTRY[effective_attention_kernel](query, key, value, context)

  raise ValueError(f"Unexpected attention kernel {effective_attention_kernel=}.")


def _head_local_svg_attention(query, key, value, context):
  from .wan.transformers import svg_attention, svg_head_local

  cfg = context["spatiotemporal_config"]
  grid = context["spatiotemporal_shape"]
  mesh = context["mesh"]
  cp = mesh.shape[CONTEXT]
  if context["attention_mask"] is not None or cfg.get("global_stride", 0):
    raise ValueError("Head-local SVG does not support external or periodic masks.")
  if context["ulysses_attention_chunks"] != 1:
    raise ValueError("Head-local SVG does not implement chunked Ulysses attention.")
  q, k, v = (_unflatten_heads(x, context["heads"]) if x.ndim == 3 else x for x in (query, key, value))
  if grid is None or q.shape != k.shape or q.shape != v.shape or q.shape[2] != math.prod(grid):
    raise ValueError("Head-local SVG requires matched self-attention QKV and token grid.")
  batch, heads = q.shape[0], q.shape[1]
  # Fold unsharded batches into heads to match the dense Ulysses layout.
  devices_in_batch_sharding = mesh.shape["data"] * (mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1)
  fold_batch = batch > 1 and devices_in_batch_sharding == 1 and (batch * heads) % cp == 0
  local_heads = batch * heads if fold_batch else heads
  if local_heads % cp or q.shape[2] % cp:
    raise ValueError("SVG heads and sequence length must divide evenly across Ulysses shards.")
  q, k, v = (svg_head_local.inference_only(x) for x in (q, k, v))
  qspec = nn.logical_to_mesh_axes(context["axis_names_q"])
  kvspec = nn.logical_to_mesh_axes(context["axis_names_kv"])
  if qspec != kvspec or qspec[1:] != (None, CONTEXT, None):
    raise ValueError("Head-local SVG requires sequence sharding and unsharded heads.")
  profile_key = jax.random.PRNGKey(int(cfg["profile_seed"]))
  for index in (cfg.get("svg_layer_index"), cfg.get("svg_step_index")):
    if index is not None:
      profile_key = jax.random.fold_in(profile_key, jnp.asarray(index, jnp.uint32))
  if cfg.get("svg_step_index") is None and cfg.get("svg_timestep") is not None:
    profile_key = jax.random.fold_in(profile_key, jnp.max(jnp.asarray(cfg["svg_timestep"])).astype(jnp.uint32))
  with jax.named_scope("svg_routing"):
    route = svg_attention.svg_profile_temporal_heads(
        q,
        k,
        v,
        grid,
        int(cfg["profile_query_count"]),
        profile_key,
        context["scale"],
        sample_max_row=int(cfg.get("sample_max_row", 10000)),
    )
  if fold_batch:
    q, k, v = (x.reshape(1, local_heads, *x.shape[2:]) for x in (q, k, v))
    route = route.reshape(1, local_heads)
  # Sparse and dense attention use independently configured tiles.
  bq, bkv, bc, bci, hpt, vmem = _extract_custom_block_sizes(
      cfg.get("custom_flash_block_sizes") or context["flash_block_sizes"]
  )
  if hpt != 1:
    raise ValueError("SVG requires heads_per_tile=1.")
  blocks = custom_svg_static_range_attention.SVGBlockSizes(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bc,
      block_kv_compute_in=bci,
  )

  def core(q, k, v):
    if not context.get("k_prescaled", False):
      k = k * context["scale"]
    if context["use_base2_exp"] and not context.get("qk_prescaled", False):
      q = q * LOG2E
    q, dim, n = _pad_data_for_flash(q, q.shape[1], bq)
    k, _, nk = _pad_data_for_flash(k, k.shape[1], bkv)
    v, _, _ = _pad_data_for_flash(v, v.shape[1], bkv)
    kernel = custom_svg_attention_dispatch.make_svg_static_range_mha(
        block_sizes=blocks,
        orig_q_seq_len=n,
        orig_kv_seq_len=nk,
        band_width=int(cfg["band_width"]),
        frame_size=int(grid[1] * grid[2]),
        include_first_frame=bool(cfg.get("include_first_frame", True)),
        bkv_compute_in=bci,
        use_base2_exp=context["use_base2_exp"],
        use_experimental_scheduler=context["use_experimental_scheduler"],
        vmem_limit_bytes=vmem,
    )
    # Report executed tile fraction, which differs from real attention-pair density.
    executed = kernel.union_main_tiles + kernel.tail_cleanup_tiles
    total = -(-n // bq) * -(-nk // bkv)
    with jax.named_scope(f"svg_kernel_c_tiles{executed}of{total}_d{executed / total:.3f}"):
      out = jax.vmap(kernel)(q, k, v)
    return jnp.swapaxes(out, 2, 3)[:, :, :n, :dim].astype(q.dtype)

  out = svg_head_local.exchange_local(
      q,
      k,
      v,
      route,
      mesh=mesh,
      qspec=qspec,
      kvspec=kvspec,
      ulysses_axis=CONTEXT,
      place=lambda q, k, v, r: svg_attention.svg_placement_permute(q, k, v, r, grid),
      restore=lambda o, r: svg_attention.svg_placement_unpermute(o, r, grid),
      core=core,
  )
  if fold_batch:
    out = out.reshape(batch, heads, *out.shape[2:])
  return _reshape_heads_to_head_dim(out)


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
  """Multi-head dot product attention with a limited number of queries."""
  num_kv, num_heads, k_features = key.shape[-3:]
  v_features = value.shape[-1]
  key_chunk_size = min(key_chunk_size, num_kv)
  query = query / jnp.sqrt(k_features)

  @functools.partial(jax.checkpoint, prevent_cse=False)
  def summarize_chunk(query, key, value):
    attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

    max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    max_score = jax.lax.stop_gradient(max_score)
    exp_weights = jnp.exp(attn_weights - max_score)

    exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
    max_score = jnp.einsum("...qhk->...qh", max_score)

    return (exp_values, exp_weights.sum(axis=-1), max_score)

  def chunk_scanner(chunk_idx):
    # julienne key array
    key_chunk = jax.lax.dynamic_slice(
        operand=key,
        start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
        slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
    )

    # julienne value array
    value_chunk = jax.lax.dynamic_slice(
        operand=value,
        start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
        slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
    )

    return summarize_chunk(query, key_chunk, value_chunk)

  chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

  global_max = jnp.max(chunk_max, axis=0, keepdims=True)
  max_diffs = jnp.exp(chunk_max - global_max)

  chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
  chunk_weights *= max_diffs

  all_values = chunk_values.sum(axis=0)
  all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

  return all_values / all_weights


def jax_memory_efficient_attention(
    query,
    key,
    value,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
):
  r"""
  Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
  https://github.com/AminRezaei0x443/memory-efficient-attention

  Args:
      query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
      key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
      value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
      precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
          numerical precision for computation
      query_chunk_size (`int`, *optional*, defaults to 1024):
          chunk size to divide query array value must divide query_length equally without remainder
      key_chunk_size (`int`, *optional*, defaults to 4096):
          chunk size to divide key and value array value must divide key_value_length equally without remainder

  Returns:
      (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
  """
  num_q, num_heads, q_features = query.shape[-3:]

  def chunk_scanner(chunk_idx, _):
    # julienne query array
    query_chunk = jax.lax.dynamic_slice(
        operand=query,
        start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
        slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
    )

    return (
        chunk_idx + query_chunk_size,  # unused ignore it
        _query_chunk_attention(
            query=query_chunk,
            key=key,
            value=value,
            precision=precision,
            key_chunk_size=key_chunk_size,
        ),
    )

  _, res = jax.lax.scan(
      f=chunk_scanner,
      init=0,
      xs=None,
      length=math.ceil(num_q / query_chunk_size),  # start counter  # stop counter
  )

  return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


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


# New Class for Wan I2V
class NNXSimpleFeedForward(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      activation_fn: str = "gelu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      sharding_specs: Optional[Any] = None,
  ):
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    net_0_kernel = safe_getattr(sharding_specs, "net_0_kernel", ("embed", "mlp"))
    net_0_bias = safe_getattr(sharding_specs, "net_0_bias", ("mlp",))
    net_2_kernel = safe_getattr(sharding_specs, "net_2_kernel", ("mlp", "embed"))
    net_2_bias = safe_getattr(sharding_specs, "net_2_bias", ("embed",))

    self.net_0 = nnx.Linear(
        dim,
        inner_dim,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_0_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_0_bias),
    )
    self.act = get_activation(activation_fn)
    self.net_2 = nnx.Linear(
        inner_dim,
        dim_out,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_2_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_2_bias),
    )

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.net_0(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


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
      kv_heads: Optional[int] = None,
      use_k_centering: bool | str = "auto",
      wan_ulysses_out_a2a: Optional[str] = None,
  ):
    self.dpa_layer = None
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.ulysses_shards = ulysses_shards
    self.ulysses_attention_chunks = ulysses_attention_chunks
    self.use_k_centering = use_k_centering
    self.wan_ulysses_out_a2a = wan_ulysses_out_a2a
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
    self.kv_heads = kv_heads
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
      preserve_asymmetric_block_sizes: bool = False,
      spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
      sparse_config_override: Optional[dict] = None,
      qk_prescaled: bool = False,
      k_prescaled: bool = False,
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
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
        spatiotemporal_config=sparse_config_override,
        spatiotemporal_shape=spatiotemporal_shape,
        kv_heads=self.kv_heads,
        use_k_centering=getattr(self, "use_k_centering", "auto"),
        qk_prescaled=qk_prescaled,
        k_prescaled=k_prescaled,
        wan_ulysses_out_a2a=getattr(self, "wan_ulysses_out_a2a", None),
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
  is_causal: bool = False
  kv_heads: Optional[int] = None
  use_k_centering: bool | str = "auto"

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

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      attention_mask: Array = None,
      preserve_asymmetric_block_sizes: bool = False,
      spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
      sparse_config_override: Optional[dict] = None,
      qk_prescaled: bool = False,
      k_prescaled: bool = False,
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
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
        ulysses_shards=self.ulysses_shards,
        ulysses_attention_chunks=self.ulysses_attention_chunks,
        is_causal=self.is_causal,
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
        spatiotemporal_config=sparse_config_override,
        spatiotemporal_shape=spatiotemporal_shape,
        kv_heads=self.kv_heads,
        use_k_centering=getattr(self, "use_k_centering", "auto"),
        qk_prescaled=qk_prescaled,
        k_prescaled=k_prescaled,
    )


@functools.lru_cache(maxsize=64)
def _build_sharded_fused_rope_producer(
    kernel_fn: Callable,
    mesh: jax.sharding.Mesh,
    act_spec: jax.sharding.PartitionSpec,
    replicated_spec: jax.sharding.PartitionSpec,
    freqs_spec: jax.sharding.PartitionSpec,
    out_spec: jax.sharding.PartitionSpec,
    q_heads: int,
    dim_head: int,
    eps: float,
    k_eps: Optional[float],
    norm_mode: str,
    rope_accum: str,
    block_s: Optional[int],
    head_block: Optional[int],
    q_prescale: float,
    k_prescale: float,
):
  """Builds and caches the `with_xla_backward(jax.shard_map(...))` RoPE producer."""
  sharded = jax.shard_map(
      functools.partial(
          kernel_fn,
          q_heads=q_heads,
          dim_head=dim_head,
          eps=eps,
          k_eps=k_eps,
          norm_mode=norm_mode,
          # Under jit, XLA on v6e contracts the reference's RoPE multiply-adds
          # into FP32 FMAs ("f32"), while XLA on tpu7x emits native bf16
          # multiply-adds ("dtype"). Matching the platform's contraction
          # yields 0-ULP bit-identical output on both v6e and tpu7x.
          rope_accum=rope_accum,
          block_s=block_s,
          head_block=head_block,
          q_prescale=q_prescale,
          k_prescale=k_prescale,
      ),
      mesh=mesh,
      in_specs=(act_spec, act_spec, replicated_spec, replicated_spec, freqs_spec),
      out_specs=(out_spec, out_spec),
      check_vma=False,
  )

  def xla_equivalent(q, k, q_scale, k_scale, freqs):
    """The same producer in unfused primitives, prescaling included."""
    out_q, out_k = fused_rmsnorm_rope(
        q, k, q_scale, k_scale, freqs, q_heads=q_heads, dim_head=dim_head, eps=eps, k_eps=k_eps
    )
    if q_prescale != 1.0:
      out_q = out_q * jnp.asarray(q_prescale, out_q.dtype)
    if k_prescale != 1.0:
      out_k = out_k * jnp.asarray(k_prescale, out_k.dtype)
    return out_q, out_k

  # `pallas_call` has no transpose rule; forward stays the kernel.
  return with_xla_backward(sharded, xla_equivalent)


@functools.lru_cache(maxsize=64)
def _build_sharded_pallas_cross_attention(
    kernel_fn: Callable,
    mesh: jax.sharding.Mesh,
    q_spec: jax.sharding.PartitionSpec,
    kv_spec: jax.sharding.PartitionSpec,
    local_heads: int,
    global_heads: int,
    dim_head: int,
    scale: float,
    k_prescaled: bool,
    head_block: int,
    is_cpu_interpret: bool,
    dtype: DType,
    split_head_dim: bool,
    float32_qk_product: bool,
):
  """Builds and caches the `with_xla_backward(jax.shard_map(...))` cross-attention wrapper."""
  sharded = jax.shard_map(
      functools.partial(
          kernel_fn,
          heads=local_heads,
          dim_head=dim_head,
          scale=scale,
          k_prescaled=k_prescaled,
          head_block=head_block,
          interpret=is_cpu_interpret,
      ),
      mesh=mesh,
      in_specs=(q_spec, kv_spec, kv_spec),
      out_specs=q_spec,
      check_vma=False,
  )

  def xla_equivalent(q, k, v):
    """The XLA dot-product path this kernel stands in for."""
    return _apply_attention_dot(
        q,
        k,
        v,
        dtype,
        global_heads,
        dim_head,
        scale,
        split_head_dim,
        float32_qk_product,
        False,
        k_prescaled=k_prescaled,
    )

  # `pallas_call` has no transpose rule; forward stays the kernel.
  return with_xla_backward(sharded, xla_equivalent)


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
        "use_svg_attention": False,
        "svg_implementation": "official_svg",
        "svg_spatial_density": 0.25,
        "svg_sample_max_row": 10000,
        "svg_profile_query_count": 64,
        "svg_profile_seed": 0,
        "svg_dense_layer_fraction": 0.0,
        "svg_dense_timestep_fraction": 0.0,
        "svg_active_start_step": -1,
        "svg_active_end_step": -1,
        "svg_active_start_layer": -1,
        "svg_active_end_layer": -1,
        "svg_num_train_timesteps": 1000,
        "svg_num_layers": 40,
        "svg_include_first_frame": True,
        "svg_global_stride": 0,
        "svg_global_offset": 0,
        "svg_high_noise_density": -1.0,
        "svg_low_noise_density": -1.0,
        "svg_flash_block_sizes": None,
        "use_k_centering": "auto",
        # Fused RMSNorm+RoPE+head-transpose Pallas producer. Off by default:
        # 0-ULP against the separately-jitted XLA producer on measured platforms,
        # but end-to-end output is equivalent, not identical (XLA fuses the
        # unfused producer differently). Enabled only where
        # rope_accum_is_measured() holds.
        "use_fused_rope_kernel": False,
        "fused_rope_block_s": 1024,
        "fused_rope_head_block": None,
        **(attention_config or {}),
    }

    self.use_svg_attention = attention_config["use_svg_attention"]
    self.svg_implementation = attention_config["svg_implementation"]
    self.svg_spatial_density = attention_config["svg_spatial_density"]
    self.svg_sample_max_row = attention_config["svg_sample_max_row"]
    self.svg_profile_query_count = attention_config["svg_profile_query_count"]
    self.svg_profile_seed = attention_config["svg_profile_seed"]
    self.svg_dense_layer_fraction = attention_config["svg_dense_layer_fraction"]
    self.svg_dense_timestep_fraction = attention_config["svg_dense_timestep_fraction"]
    self.svg_active_start_step = attention_config["svg_active_start_step"]
    self.svg_active_end_step = attention_config["svg_active_end_step"]
    self.svg_active_start_layer = attention_config["svg_active_start_layer"]
    self.svg_active_end_layer = attention_config["svg_active_end_layer"]
    self.svg_num_train_timesteps = attention_config["svg_num_train_timesteps"]
    self.svg_num_layers = attention_config["svg_num_layers"]
    self.svg_include_first_frame = attention_config["svg_include_first_frame"]
    self.svg_global_stride = attention_config["svg_global_stride"]
    self.svg_global_offset = attention_config["svg_global_offset"]
    self.svg_high_noise_density = attention_config["svg_high_noise_density"]
    self.svg_low_noise_density = attention_config["svg_low_noise_density"]
    self.svg_flash_block_sizes = attention_config["svg_flash_block_sizes"]
    self.is_self_attention = is_self_attention
    # Assigned before the check below, which references it: without this the
    # validation raises AttributeError instead of the intended message.
    self.mesh = mesh

    self.use_fused_rope_kernel = attention_config["use_fused_rope_kernel"]
    self.fused_rope_block_s = attention_config["fused_rope_block_s"]
    self.fused_rope_head_block = attention_config["fused_rope_head_block"]
    self.wan_cross_attn_kernel = attention_config.get("wan_cross_attn_kernel")
    self.wan_ulysses_out_a2a = attention_config.get("wan_ulysses_out_a2a")
    self.wan_cross_attn_cpu_interpret = attention_config.get("wan_cross_attn_cpu_interpret", False)

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
    self.is_self_attention = is_self_attention
    self.eps = eps

    cross_attention_remapped_to_flash = not is_self_attention and attention_kernel in (
        "tokamax_ring",
        "tokamax_ring_custom",
        "ulysses_ring",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
        "ulysses_ring_custom_fixed_m_per_q_block",
        "ulysses_ring_custom_bidir",
        "ulysses_custom",
        "ulysses_custom_fixed_m",
        "ulysses_custom_fixed_m_per_q_block",
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
    self.added_kv_proj_dim = added_kv_proj_dim  # New for I2V
    self.image_seq_len = image_seq_len  # New for I2V
    tpu_type = get_tpu_type()
    self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    self.precision = precision

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
        use_k_centering=attention_config["use_k_centering"],
        wan_ulysses_out_a2a=self.wan_ulysses_out_a2a,
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
    _prescale_unsupported_kernels = {"cudnn_flash_te"}
    self.cross_attn_prescale_kv = (
        wan_runtime_options.get("wan_cross_attn_prescale_kv")
        and getattr(self.attention_op, "attention_kernel", "") not in _prescale_unsupported_kernels
        and not getattr(self.attention_op, "use_memory_efficient_attention", False)
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

    # 5. Interleave the rotated pairs back into the last axis
    xq_out = jnp.concatenate([xq_out_0[..., None], xq_out_1[..., None]], axis=-1).reshape(xq.shape)
    xk_out = jnp.concatenate([xk_out_0[..., None], xk_out_1[..., None]], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

  def _fused_rope_producer(self):
    """Selects the RMSNorm+RoPE+transpose producer, falling back when unsafe.

    The Pallas kernel is a pure fusion of `fused_rmsnorm_rope`: ~1.9x faster at
    the Wan shard shape, and 0 ULP against the *separately jitted* XLA producer
    there. That equality does not survive inlining -- inside the 40-layer graph
    XLA fuses the unfused producer with its neighbours and rounds it differently
    (v6e 720p/81f, same seed: 53.8 dB PSNR after 1 denoise step, 34.1 dB after
    40). Equivalent, not identical; hence opt-in, on measured hardware only.

    Returns a wrapper calling the Pallas producer (returning
    `((q_out, k_out), fuse_qk_prescale)`) when every static precondition below
    holds, or `fused_rmsnorm_rope` (returning `(q_out, k_out)`) otherwise:
      * the mesh devices are TPUs (the Pallas kernel uses Mosaic TPU primitives)
        and `rope_accum_is_measured(self.mesh)` holds (or `wan_rope_accum` is
        explicitly set);
      * a mesh is available to wrap the custom call in `shard_map` -- a raw
        `pallas_call` on sharded operands would make GSPMD all-gather them;
      * the feature axis is unsharded. RMSNorm reduces across `heads * dim_head`
        and RoPE pairs lanes within a head, so a sharded feature axis would
        silently produce a per-shard norm instead of the true one;
      * `dim_head` is a whole number of lanes, which the kernel requires;
      * at call time inside the returned wrapper, the sequence dimension must
        divide evenly by its mesh axis (otherwise it falls back to
        `fused_rmsnorm_rope`); an indivisible batch dimension is treated as
        replicated.
    """
    is_tpu = self.mesh is not None and all(getattr(d, "platform", None) == "tpu" for d in self.mesh.devices.flat)
    # Matching the XLA producer's rounding is a per-platform property: on v4 the
    # kernel drifts by up to one bf16 ULP. Unverified hardware gets XLA.
    rounding_verified = is_tpu and rope_accum_is_measured(self.mesh)
    if (
        not self.use_fused_rope_kernel
        or not is_tpu
        or self.mesh is None
        or self.dim_head % 128 != 0
        or not rounding_verified
    ):
      if self.use_fused_rope_kernel:
        kinds = sorted({getattr(d, "device_kind", "?") for d in self.mesh.devices.flat}) if self.mesh is not None else []
        _warn_once(
            "fused_rope_kernel_unusable",
            f"fused RoPE kernel requested but unusable (is_tpu={is_tpu}, mesh={self.mesh is not None}, "
            f"dim_head={self.dim_head}, rounding_verified={rounding_verified}, device_kind={kinds}); "
            "using the XLA producer.",
        )
      return fused_rmsnorm_rope

    in_spec = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
    feature_axis = in_spec[2]
    if feature_axis is not None:
      axes = (feature_axis,) if isinstance(feature_axis, str) else tuple(feature_axis)
      if any(self.mesh.shape[a] > 1 for a in axes):
        _warn_once(
            "fused_rope_kernel_sharded_feature",
            f"fused RoPE kernel disabled: the feature axis is sharded over {axes}, which would turn the RMSNorm "
            "reduction into a per-shard norm; using the XLA producer.",
        )
        return fused_rmsnorm_rope

    replicated = jax.sharding.PartitionSpec()
    block_s = self.fused_rope_block_s
    head_block = self.fused_rope_head_block
    mesh = self.mesh

    def _shards_over(axis) -> int:
      """Total mesh width a single PartitionSpec entry splits a dimension by."""
      if axis is None:
        return 1
      names = (axis,) if isinstance(axis, str) else tuple(axis)
      return math.prod(mesh.shape[n] for n in names)

    batch_axis, seq_axis = in_spec[0], in_spec[1]
    batch_shards = _shards_over(batch_axis)
    seq_shards = _shards_over(seq_axis)

    def producer(q, k, q_scale, k_scale, freqs, *, q_heads, dim_head, eps, k_eps=None):
      # `shard_map` demands exact divisibility on every sharded dimension,
      # whereas the surrounding GSPMD program pads uneven splits. Wan runs a
      # global batch of 1 over a multi-way data axis, which GSPMD degenerates
      # into replication anyway -- so declare it replicated here rather than
      # giving up the kernel over a dimension no device actually splits.
      local_batch_axis = batch_axis if q.shape[0] % batch_shards == 0 else None
      divides = (
          q.shape[1] % seq_shards == 0
          and k.shape[1] % seq_shards == 0
          # Each shard rotates its own contiguous window of positions, which
          # only lines up if q, k and the table are split the same way.
          and freqs.shape[2] == q.shape[1]
          and q.shape[1] == k.shape[1]
      )
      if not divides:
        _warn_once(
            "fused_rope_kernel_indivisible",
            f"fused RoPE kernel disabled: q={q.shape} k={k.shape} freqs={freqs.shape} are not all splittable "
            f"{seq_shards} ways on the sequence axis; using the XLA producer.",
        )
        return fused_rmsnorm_rope(q, k, q_scale, k_scale, freqs, q_heads=q_heads, dim_head=dim_head, eps=eps, k_eps=k_eps)

      act_spec = jax.sharding.PartitionSpec(local_batch_axis, seq_axis, in_spec[2])
      # [B, S, H*D] -> [B, H, S, D]: the head axis inherits the feature axis'
      # sharding (unsharded, per the guard above) and `dim_head` is never split.
      out_spec = jax.sharding.PartitionSpec(local_batch_axis, in_spec[2], seq_axis, None)
      # `freqs_cis` is `[1, 1, S, dim_head // 2]` and reaches this point
      # replicated, but each shard owns a contiguous window of positions.
      # Splitting it on the same axis as the activations is what GSPMD does
      # implicitly for the reference's elementwise multiply; leaving it
      # replicated would make every shard rotate by positions [0, S_local),
      # which is right only on the first shard.
      freqs_spec = jax.sharding.PartitionSpec(None, None, seq_axis, None)

      attn_kernel = getattr(self.attention_op, "attention_kernel", "dot_product")
      min_seq_len = getattr(self.attention_op, "flash_min_seq_length", 4096)
      use_mem_eff = getattr(self.attention_op, "use_memory_efficient_attention", False)
      supported_custom_kernels = {
          "ulysses_custom",
          "ulysses_ring_custom",
          "ulysses_ring_custom_fixed_m",
          "ulysses_ring_custom_fixed_m_per_q_block",
          "ulysses_ring_custom_bidir",
          "ulysses_custom_fixed_m",
          "ulysses_custom_fixed_m_per_q_block",
      }
      can_prescale = attn_kernel in supported_custom_kernels and not use_mem_eff and q.shape[1] >= min_seq_len
      norm_mode_env = wan_runtime_options.get("wan_rope_norm_mode")
      fuse_qk_prescale = wan_runtime_options.get("wan_fuse_qk_prescale") and can_prescale
      q_prescale_val = float(LOG2E) if (fuse_qk_prescale and getattr(self.attention_op, "use_base2_exp", True)) else 1.0
      k_prescale_val = float(self.attention_op.scale) if fuse_qk_prescale else 1.0
      _warn_once(
          "fused_rope_kernel_active",
          f"fused RoPE Pallas kernel ACTIVE: q={q.shape}, per-shard seq={q.shape[1] // seq_shards}, "
          f"batch_spec={local_batch_axis}, block_s={block_s}, head_block={head_block or q_heads}, "
          f"norm_mode={norm_mode_env}, q_prescale={q_prescale_val}, k_prescale={k_prescale_val}.",
      )
      # Resolved through the shared helper so this and the AOT cache
      # fingerprint in generate_wan.py cannot disagree about which mode was
      # compiled (see `resolve_rope_accum`).
      rope_accum_env = resolve_rope_accum(mesh)
      differentiable = _build_sharded_fused_rope_producer(
          fused_rmsnorm_rope_pallas,
          mesh,
          act_spec,
          replicated,
          freqs_spec,
          out_spec,
          q_heads,
          dim_head,
          eps,
          k_eps,
          norm_mode_env,
          rope_accum_env,
          block_s,
          head_block,
          q_prescale_val,
          k_prescale_val,
      )
      return differentiable(q, k, q_scale, k_scale, freqs), fuse_qk_prescale

    return producer

  def _pallas_cross_attention(self, query, key, value, *, k_prescaled: bool):
    """Runs text cross-attention through the fused Pallas kernel, or returns None.

    Opt-in via `wan_cross_attn_kernel: "pallas"`. It replaces only what would
    otherwise be the XLA dot-product fallback -- unmasked MHA against a KV
    shorter than `flash_min_seq_length` -- because the kernel holds a head
    block's entire KV in VMEM. Returns None, meaning "use the XLA path", unless:
      * the mesh devices are TPUs (or `wan_cross_attn_cpu_interpret` is enabled)
        and a mesh is available for `shard_map`;
      * the inputs are flat `[B, S, H*D]` with a lane-aligned `dim_head`;
      * K/V carry the same heads as Q and are short enough to stay resident;
      * sequence and heads divide evenly by the mesh axes they are sharded over
        (an indivisible batch dimension is treated as replicated).
    """
    mode = getattr(self, "wan_cross_attn_kernel", None) or wan_runtime_options.get("wan_cross_attn_kernel")
    if mode == "xla":
      return None
    if mode != "pallas":
      raise ValueError(f"wan_cross_attn_kernel must be 'xla' or 'pallas', got {mode!r}.")

    op = self.attention_op
    mesh = self.mesh
    is_tpu = mesh is not None and all(getattr(d, "platform", None) == "tpu" for d in mesh.devices.flat)
    is_cpu_interpret = getattr(self, "wan_cross_attn_cpu_interpret", False) or bool(
        wan_runtime_options.get("wan_cross_attn_cpu_interpret")
    )
    flat = query.ndim == 3 and key.ndim == 3 and value.ndim == 3
    kv_len = key.shape[1] if flat else -1
    would_use_dot_product = not op.use_memory_efficient_attention and (
        op.attention_kernel == "dot_product" or min(query.shape[1], kv_len) < op.flash_min_seq_length
    )

    def _shards_over(axis) -> int:
      if axis is None:
        return 1
      names = (axis,) if isinstance(axis, str) else tuple(axis)
      return math.prod(mesh.shape[n] for n in names)

    reason = None
    if not (is_tpu or is_cpu_interpret):
      reason = "the mesh devices are not TPUs"
    elif not flat:
      reason = f"inputs are not flat [B, S, H*D] (q={query.shape}, k={key.shape}, v={value.shape})"
    elif self.dim_head % cross_attention_pallas.NUM_LANES:
      reason = f"dim_head={self.dim_head} is not a multiple of {cross_attention_pallas.NUM_LANES}"
    elif getattr(op, "kv_heads", None) not in (None, self.heads):
      reason = f"kv_heads={op.kv_heads} differs from heads={self.heads}"
    elif kv_len > cross_attention_pallas.MAX_KV_LEN:
      reason = f"KV length {kv_len} exceeds {cross_attention_pallas.MAX_KV_LEN}"
    elif kv_len % 8 != 0:
      reason = f"KV length {kv_len} is not a multiple of 8"
    elif not would_use_dot_product:
      reason = f"the configured {op.attention_kernel!r} kernel already handles this KV length"
    if reason is None:
      batch_axis, seq_axis, head_axis = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
      batch_shards, seq_shards, head_shards = (_shards_over(a) for a in (batch_axis, seq_axis, head_axis))
      if query.shape[1] % seq_shards or self.heads % head_shards:
        reason = f"q={query.shape} with {self.heads} heads does not split {seq_shards}x{head_shards} (seq x heads)"
      else:
        local_q_len = query.shape[1] // seq_shards
        if local_q_len <= cross_attention_pallas.DEFAULT_BLOCK_Q and local_q_len % 8 != 0:
          reason = f"local query length {local_q_len} <= {cross_attention_pallas.DEFAULT_BLOCK_Q} is not a multiple of 8"
    if reason is not None:
      _warn_once("pallas_cross_attn_unusable", f"Pallas cross-attention requested but unusable: {reason}; using XLA.")
      return None

    # As in `_fused_rope_producer`: GSPMD degenerates an uneven batch split
    # into replication, so declare it replicated rather than lose the kernel.
    local_batch_axis = batch_axis if query.shape[0] % batch_shards == 0 else None
    local_heads = self.heads // head_shards
    head_block = max(
        hb for hb in range(1, min(local_heads, cross_attention_pallas.DEFAULT_HEAD_BLOCK) + 1) if local_heads % hb == 0
    )
    q_spec = jax.sharding.PartitionSpec(local_batch_axis, seq_axis, head_axis)
    kv_spec = jax.sharding.PartitionSpec(local_batch_axis, None, head_axis)
    _warn_once(
        "pallas_cross_attn_active",
        f"Pallas cross-attention ACTIVE: q={query.shape}, kv={key.shape}, q_spec={q_spec}, kv_spec={kv_spec}, "
        f"local_heads={local_heads}, head_block={head_block}, block_q={cross_attention_pallas.DEFAULT_BLOCK_Q}, "
        f"k_prescaled={k_prescaled}.",
    )
    # `pallas_call` has no transpose rule; forward stays the kernel.
    differentiable = _build_sharded_pallas_cross_attention(
        cross_attention_pallas.cross_attention_pallas,
        mesh,
        q_spec,
        kv_spec,
        local_heads,
        self.heads,
        self.dim_head,
        float(op.scale),
        bool(k_prescaled),
        head_block,
        bool(is_cpu_interpret),
        op.dtype,
        bool(op.split_head_dim),
        bool(op.float32_qk_product),
    )
    with jax.named_scope("kernel_pallas_cross_attention"):
      return differentiable(query, key, value)

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
      spatiotemporal_shape: Optional[Tuple[int, int, int]] = None,
      svg_layer_index: Optional[int | jax.Array] = None,
      svg_timestep: Optional[int | float | jax.Array] = None,
      svg_step_index: Optional[int | jax.Array] = None,
  ) -> jax.Array:
    same_kv_source = encoder_hidden_states is None or encoder_hidden_states is hidden_states
    hidden_states = nn.with_logical_constraint(hidden_states, (BATCH, LENGTH, HEAD))
    if encoder_hidden_states is not None:
      encoder_hidden_states = nn.with_logical_constraint(encoder_hidden_states, (BATCH, LENGTH, HEAD))
    dtype = hidden_states.dtype
    if not same_kv_source:
      is_self_attention = False
    else:
      is_self_attention = getattr(self, "is_self_attention", True)
    if self.use_svg_attention and is_self_attention:
      if not deterministic:
        raise ValueError("SVG attention supports deterministic inference only.")
      if spatiotemporal_shape is None:
        raise ValueError("SVG attention requires spatiotemporal_shape.")
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

      k_prescaled = False
      if is_self_attention:
        with jax.named_scope("key_proj"):
          key_proj = self.key(hidden_states)
        with jax.named_scope("value_proj"):
          value_proj = self.value(hidden_states)
      elif cached_kv is not None and "text" in cached_kv:
        key_proj, value_proj = cached_kv["text"]
        k_prescaled = self.cross_attn_prescale_kv
      else:
        with jax.named_scope("key_proj"):
          key_proj = self.key(encoder_hidden_states)
        with jax.named_scope("value_proj"):
          value_proj = self.value(encoder_hidden_states)

      qk_prescaled = False
      if rotary_emb is not None and self.qk_norm and is_self_attention:
        with self.conditional_named_scope("fused_rmsnorm_rope"):
          q_scale = self.norm_q.scale[...]
          k_scale = self.norm_k.scale[...]
          q_eps = getattr(self.norm_q, "epsilon", self.eps)
          k_eps = getattr(self.norm_k, "epsilon", self.eps)
          # The SVG sparse path does not understand prescaled Q/K, so SVG layers
          # keep the XLA fused_rmsnorm_rope producer rather than the Pallas kernel.
          producer = fused_rmsnorm_rope if self.use_svg_attention else self._fused_rope_producer()
          prod_out = producer(
              query_proj,
              key_proj,
              q_scale,
              k_scale,
              rotary_emb,
              q_heads=self.heads,
              dim_head=self.dim_head,
              eps=q_eps,
              k_eps=k_eps,
          )
          if isinstance(prod_out, tuple) and len(prod_out) == 2 and isinstance(prod_out[1], bool):
            (query_proj, key_proj), qk_prescaled = prod_out
          else:
            query_proj, key_proj = prod_out
          value_proj = _unflatten_heads(value_proj, self.heads)
      else:
        if self.qk_norm:
          with self.conditional_named_scope("attn_q_norm"):
            query_proj = self.norm_q(query_proj)
          if is_self_attention or cached_kv is None or "text" not in cached_kv:
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

      if self.use_svg_attention and is_self_attention and spatiotemporal_shape is not None:
        from .wan.transformers import svg_attention

        is_active = svg_attention.is_svg_active(
            step_index=svg_step_index,
            layer_index=svg_layer_index,
            timestep=svg_timestep,
            start_step=self.svg_active_start_step,
            end_step=self.svg_active_end_step,
            start_layer=self.svg_active_start_layer,
            end_layer=self.svg_active_end_layer,
            dense_layer_fraction=self.svg_dense_layer_fraction,
            dense_timestep_fraction=self.svg_dense_timestep_fraction,
            num_train_timesteps=self.svg_num_train_timesteps,
            num_layers=self.svg_num_layers,
        )

        def run_dense(_):
          return self.attention_op.apply_attention(
              query_proj,
              key_proj,
              value_proj,
              attention_mask=encoder_attention_mask,
              qk_prescaled=qk_prescaled,
              k_prescaled=k_prescaled,
          )

        def run_sparse_svg(_):
          execution_band_width = svg_attention.svg_execution_band_width(
              spatiotemporal_shape,
              self.svg_spatial_density,
          )
          sparse_config = {
              "use_svg_attention": True,
              "mask_type": "svg_spatial",
              "band_width": execution_band_width,
              "include_first_frame": self.svg_include_first_frame,
              "global_stride": self.svg_global_stride,
              "global_offset": self.svg_global_offset,
              "profile_query_count": self.svg_profile_query_count,
              "profile_seed": self.svg_profile_seed,
              "sample_max_row": self.svg_sample_max_row,
              "custom_flash_block_sizes": self.svg_flash_block_sizes,
              "svg_step_index": svg_step_index,
              "svg_layer_index": svg_layer_index,
              "svg_timestep": svg_timestep,
          }
          return self.attention_op.apply_attention(
              query_proj,
              key_proj,
              value_proj,
              attention_mask=encoder_attention_mask,
              spatiotemporal_shape=spatiotemporal_shape,
              sparse_config_override=sparse_config,
          )

        with jax.named_scope("apply_attention"):
          if isinstance(is_active, bool):
            attn_output = run_sparse_svg(None) if is_active else run_dense(None)
          else:
            attn_output = jax.lax.cond(is_active, run_sparse_svg, run_dense, operand=None)
      else:
        with jax.named_scope("apply_attention"):
          attn_output = None
          if not is_self_attention and encoder_attention_mask is None and not qk_prescaled:
            attn_output = self._pallas_cross_attention(query_proj, key_proj, value_proj, k_prescaled=k_prescaled)
          if attn_output is None:
            attn_output = self.attention_op.apply_attention(
                query_proj,
                key_proj,
                value_proj,
                attention_mask=encoder_attention_mask,
                qk_prescaled=qk_prescaled,
                k_prescaled=k_prescaled,
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
        k_prescaled_text = self.cross_attn_prescale_kv
      else:
        with self.conditional_named_scope("proj_key"):
          key_proj_text = self.key(encoder_hidden_states_text)
        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj_text = self.norm_k(key_proj_text)
        with self.conditional_named_scope("proj_value"):
          value_proj_text = self.value(encoder_hidden_states_text)
        k_prescaled_text = False

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        if cached_kv is not None and "image" in cached_kv:
          key_proj_img, value_proj_img = cached_kv["image"]
          k_prescaled_img = self.cross_attn_prescale_kv
        else:
          with self.conditional_named_scope("add_proj_k"):
            key_proj_img = self.add_k_proj(encoder_hidden_states_img)
          with self.conditional_named_scope("norm_add_k"):
            key_proj_img = self.norm_added_k(key_proj_img)
          with self.conditional_named_scope("add_proj_v"):
            value_proj_img = self.add_v_proj(encoder_hidden_states_img)
          k_prescaled_img = False
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
          attn_output_text = self._pallas_cross_attention(
              query_proj_text, key_proj_text, value_proj_text, k_prescaled=k_prescaled_text
          )
          if attn_output_text is None:
            attn_output_text = self.attention_op.apply_attention(
                query_proj_text, key_proj_text, value_proj_text, k_prescaled=k_prescaled_text
            )
        with self.conditional_named_scope("cross_attn_img_apply"):
          # Pass encoder_attention_mask_img for image cross-attention to mask padded tokens
          attn_output_img = self.attention_op.apply_attention(
              query_proj_img,
              key_proj_img,
              value_proj_img,
              attention_mask=encoder_attention_mask_img,
              k_prescaled=k_prescaled_img,
          )

        attn_output = attn_output_text + attn_output_img
      else:
        # No image embeddings, only text cross-attention
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")

        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output = self._pallas_cross_attention(
              query_proj_text, key_proj_text, value_proj_text, k_prescaled=k_prescaled_text
          )
          if attn_output is None:
            attn_output = self.attention_op.apply_attention(
                query_proj_text, key_proj_text, value_proj_text, k_prescaled=k_prescaled_text
            )

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
      if self.cross_attn_prescale_kv:
        key_proj = key_proj * jnp.asarray(self.attention_op.scale, key_proj.dtype)

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
      if self.cross_attn_prescale_kv:
        key_proj_text = key_proj_text * jnp.asarray(self.attention_op.scale, key_proj_text.dtype)
      with self.conditional_named_scope("proj_value"):
        value_proj_text = self.value(encoder_hidden_states_text)

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        with self.conditional_named_scope("add_proj_k"):
          key_proj_img = self.add_k_proj(encoder_hidden_states_img)
        with self.conditional_named_scope("norm_add_k"):
          key_proj_img = self.norm_added_k(key_proj_img)
        if self.cross_attn_prescale_kv:
          key_proj_img = key_proj_img * jnp.asarray(self.attention_op.scale, key_proj_img.dtype)
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
  ulysses_shards: int = -1
  ulysses_attention_chunks: int = 1

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
        ulysses_shards=self.ulysses_shards,
        ulysses_attention_chunks=self.ulysses_attention_chunks,
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


class FlaxAttention(nn.Module):
  r"""
  A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

  Parameters:
      query_dim (:obj:`int`):
          Input hidden states dimension
      heads (:obj:`int`, *optional*, defaults to 8):
          Number of heads
      dim_head (:obj:`int`, *optional*, defaults to 64):
          Hidden states dimension inside each head
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      quant (`AqtQuantization`, *optional*, defaults to None)

  """

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
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  precision: jax.lax.Precision = None
  quant: Quant = None

  def setup(self):
    if self.attention_kernel == "flash" and self.mesh is None:
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
        quant=self.quant,
    )

    qkv_init_kernel = nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads"))
    dot_general_cls = None
    if self.quant:
      dot_general_cls = self.quant.dot_general_cls()
    self.query = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_q",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.key = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_k",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.value = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_v",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("heads", "embed")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_out_0",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(
      self,
      hidden_states,
      context=None,
      deterministic=True,
      cross_attention_kwargs=None,
  ):
    context = hidden_states if context is None else context
    query_proj = self.query(hidden_states)
    key_proj = self.key(context)
    value_proj = self.value(context)

    query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
    key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
    value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

    hidden_states = self.attention_op.apply_attention(query_proj, key_proj, value_proj)

    hidden_states = self.proj_attn(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, (BATCH, LENGTH, HEAD))
    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxBasicTransformerBlock(nn.Module):
  r"""
  A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
  https://arxiv.org/abs/1706.03762


  Parameters:
      dim (:obj:`int`):
          Inner hidden states dimension
      n_heads (:obj:`int`):
          Number of heads
      d_head (:obj:`int`):
          Hidden states dimension inside each head
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      only_cross_attention (`bool`, defaults to `False`):
          Whether to only apply cross attention.
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      quant (`AqtQuantization`, *optional*, defaults to None)
  """

  dim: int
  n_heads: int
  d_head: int
  dropout: float = 0.0
  only_cross_attention: bool = False
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  precision: jax.lax.Precision = None
  quant: Quant = None

  def setup(self):
    # self attention (or cross_attention if only_cross_attention is True)
    self.attn1 = FlaxAttention(
        self.dim,
        self.n_heads,
        self.d_head,
        self.dropout,
        self.use_memory_efficient_attention,
        self.split_head_dim,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant=self.quant,
    )
    # cross attention
    self.attn2 = FlaxAttention(
        self.dim,
        self.n_heads,
        self.d_head,
        self.dropout,
        self.use_memory_efficient_attention,
        self.split_head_dim,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant=self.quant,
    )
    self.ff = FlaxFeedForward(
        dim=self.dim,
        dropout=self.dropout,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, context, deterministic=True, cross_attention_kwargs=None):
    # self attention
    residual = hidden_states
    if self.only_cross_attention:
      hidden_states = self.attn1(
          self.norm1(hidden_states),
          context,
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )
    else:
      hidden_states = self.attn1(
          self.norm1(hidden_states),
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )

    hidden_states = hidden_states + residual

    # cross attention
    residual = hidden_states
    hidden_states = self.attn2(
        self.norm2(hidden_states),
        context,
        deterministic=deterministic,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    hidden_states = hidden_states + residual

    # feed forward
    residual = hidden_states
    hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
    hidden_states = hidden_states + residual

    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxTransformer2DModel(nn.Module):
  r"""
  A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
  https://arxiv.org/pdf/1506.02025.pdf


  Parameters:
      in_channels (:obj:`int`):
          Input number of channels
      n_heads (:obj:`int`):
          Number of heads
      d_head (:obj:`int`):
          Hidden states dimension inside each head
      depth (:obj:`int`, *optional*, defaults to 1):
          Number of transformers block
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      use_linear_projection (`bool`, defaults to `False`): tbd
      only_cross_attention (`bool`, defaults to `False`): tbd
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      quant (`AqtQuantization`, *optional*, defaults to None)
            Configures AQT quantization github.com/google/aqt.
  """

  in_channels: int
  n_heads: int
  d_head: int
  depth: int = 1
  dropout: float = 0.0
  use_linear_projection: bool = False
  only_cross_attention: bool = False
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  norm_num_groups: int = 32
  precision: jax.lax.Precision = None
  hidden_state_axis_names: AxisNames = (BATCH, LENGTH, D_KV)
  quant: Quant = (None,)

  def setup(self):
    self.norm = nn.GroupNorm(
        num_groups=self.norm_num_groups,
        epsilon=1e-5,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )

    conv_kernel_init = nn.with_logical_partitioning(
        nn.initializers.lecun_normal(), ("keep_1", "keep_2", "conv_in", "conv_out")
    )

    inner_dim = self.n_heads * self.d_head
    if self.use_linear_projection:
      self.proj_in = nn.Dense(
          inner_dim,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "hidden")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
    else:
      self.proj_in = nn.Conv(
          inner_dim,
          kernel_init=conv_kernel_init,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )

    self.transformer_blocks = [
        FlaxBasicTransformerBlock(
            inner_dim,
            self.n_heads,
            self.d_head,
            dropout=self.dropout,
            only_cross_attention=self.only_cross_attention,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            split_head_dim=self.split_head_dim,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            precision=self.precision,
            quant=self.quant,
        )
        for _ in range(self.depth)
    ]

    if self.use_linear_projection:
      self.proj_out = nn.Dense(
          inner_dim,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("hidden", "embed")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
    else:
      self.proj_out = nn.Conv(
          inner_dim,
          kernel_init=conv_kernel_init,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )

    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, context, deterministic=True, cross_attention_kwargs=None):
    batch, height, width, channels = hidden_states.shape
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    if self.use_linear_projection:
      hidden_states = hidden_states.reshape(batch, height * width, channels)
      hidden_states = self.proj_in(hidden_states)
    else:
      hidden_states = self.proj_in(hidden_states)
      hidden_states = hidden_states.reshape(batch, height * width, channels)

    for transformer_block in self.transformer_blocks:
      hidden_states = transformer_block(
          hidden_states,
          context,
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )

    if self.use_linear_projection:
      hidden_states = self.proj_out(hidden_states)
      hidden_states = hidden_states.reshape(batch, height, width, channels)
    else:
      hidden_states = hidden_states.reshape(batch, height, width, channels)
      hidden_states = self.proj_out(hidden_states)

    hidden_states = nn.with_logical_constraint(hidden_states, self.hidden_state_axis_names)

    hidden_states = hidden_states + residual
    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxFeedForward(nn.Module):
  r"""
  Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
  [`FeedForward`] class, with the following simplifications:
  - The activation function is currently hardcoded to a gated linear unit from:
  https://arxiv.org/abs/2002.05202
  - `dim_out` is equal to `dim`.
  - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

  Parameters:
      dim (:obj:`int`):
          Inner hidden states dimension
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
  """

  dim: int
  dropout: float = 0.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    # The second linear layer needs to be called
    # net_2 for now to match the index of the Sequential layer
    self.net_0 = FlaxGEGLU(
        self.dim,
        self.dropout,
        self.dtype,
        self.weights_dtype,
        precision=self.precision,
    )
    self.net_2 = nn.Dense(
        self.dim,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.net_0(hidden_states, deterministic=deterministic)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


class FlaxGEGLU(nn.Module):
  r"""
  Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
  https://arxiv.org/abs/2002.05202.

  Parameters:
      dim (:obj:`int`):
          Input hidden states dimension
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
  """

  dim: int
  dropout: float = 0.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    inner_dim = self.dim * 4
    self.proj = nn.Dense(
        inner_dim * 2,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.proj(hidden_states)
    hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
    return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)
