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

"""Utility functions, tensor reshaping, padding, segment ID generation,
and block size adapters for attention kernels in maxdiffusion.
"""

import dataclasses
import functools
import math
from typing import Any, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from maxdiffusion import common_types
from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as pallas_splash_kernel


class CustomBlockSizes(NamedTuple):
  """Structured block sizes for custom dense splash and ring kernels."""

  bq: int
  bkv: int
  bkv_compute: int
  bkv_compute_in: int
  heads_per_tile: int
  vmem_limit_bytes: Optional[int]

  @property
  def block_q(self) -> int:
    return self.bq

  @property
  def block_kv(self) -> int:
    return self.bkv

  @property
  def block_kv_compute(self) -> int:
    return self.bkv_compute

  @property
  def block_kv_compute_in(self) -> int:
    return self.bkv_compute_in


LOG2E = math.log2(math.e)
SAFE_MIN_LSE = -1e30
SAFE_MIN_ATTN_WEIGHT = -1e30
SAFE_MIN_PROB = 1e-30

Array = common_types.Array
DType = common_types.DType
BlockSizes = common_types.BlockSizes
AxisNames = common_types.AxisNames
Quant = Any

CONTEXT = common_types.CONTEXT
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

INTERNAL_RING_AXIS = "ring"
INTERNAL_ULYSSES_AXIS = "ulysses"


@dataclasses.dataclass
class AttentionBlockSizes:
  """Unified block size adapter for attention kernels."""

  block_q: int = 4864
  block_kv: int = 1024
  block_kv_compute: int = 1024
  block_kv_compute_in: int = 1024
  heads_per_tile: int = 1
  vmem_limit_bytes: Optional[int] = None
  use_fused_bwd_kernel: bool = True
  block_q_dkv: Optional[int] = None
  block_kv_dkv: Optional[int] = None
  block_kv_dkv_compute: Optional[int] = None
  block_q_dq: Optional[int] = None
  block_kv_dq: Optional[int] = None

  @property
  def has_backward_blocks(self) -> bool:
    return self.use_fused_bwd_kernel

  def to_tokamax(self) -> pallas_splash_kernel.BlockSizes:
    return pallas_splash_kernel.BlockSizes(
        block_q=self.block_q,
        block_kv=self.block_kv,
        block_kv_compute=self.block_kv_compute,
        block_q_dkv=self.block_q_dkv if self.block_q_dkv is not None else self.block_q,
        block_kv_dkv=self.block_kv_dkv if self.block_kv_dkv is not None else self.block_kv,
        block_kv_dkv_compute=self.block_kv_dkv_compute if self.block_kv_dkv_compute is not None else self.block_kv_compute,
        block_q_dq=self.block_q_dq,
        block_kv_dq=self.block_kv_dq,
        use_fused_bwd_kernel=self.use_fused_bwd_kernel,
    )

  def to_custom_splash(self) -> custom_splash._BlockSizes:
    return custom_splash._BlockSizes(
        block_q=self.block_q,
        block_kv=self.block_kv,
        block_kv_compute=self.block_kv_compute,
        block_kv_compute_in=self.block_kv_compute_in,
    )


def _coerce_tokamax_block_sizes(block_sizes):
  """Tokamax requires fused bwd; convert if needed."""
  if getattr(block_sizes, "use_fused_bwd_kernel", False):
    return block_sizes

  bq = block_sizes.block_q
  bkv = getattr(block_sizes, "block_kv", bq)
  bkv_compute = getattr(block_sizes, "block_kv_compute", bkv)
  bq_dkv = getattr(block_sizes, "block_q_dkv", bq)
  bkv_dkv = getattr(block_sizes, "block_kv_dkv", bkv)
  bkv_dkv_compute = getattr(block_sizes, "block_kv_dkv_compute", bkv_compute)
  return pallas_splash_kernel.BlockSizes(
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


def _extract_custom_block_sizes(flash_block_sizes) -> CustomBlockSizes:
  """Pulls custom-kernel block sizes out of the config."""
  bq = 4864
  bkv = 1024
  bkv_compute = 1024
  bkv_compute_in = 1024
  heads_per_tile = 1
  vmem_limit_bytes = None
  if flash_block_sizes is not None:
    if isinstance(flash_block_sizes, dict):
      get = flash_block_sizes.get
      bq = get("block_q", None) or get("bq", None) or bq
      bkv = get("block_kv", None) or get("bkv", None) or bkv
      bkv_compute = get("block_kv_compute", None) or get("bkv_compute", None) or bkv_compute
      bkv_compute_in = get("block_kv_compute_in", None) or get("bkv_compute_in", None) or bkv_compute_in
      heads_per_tile = get("heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = get("vmem_limit_bytes", None) or vmem_limit_bytes
    else:
      bq = getattr(flash_block_sizes, "block_q", None) or getattr(flash_block_sizes, "bq", None) or bq
      bkv = getattr(flash_block_sizes, "block_kv", None) or getattr(flash_block_sizes, "bkv", None) or bkv
      bkv_compute = (
          getattr(flash_block_sizes, "block_kv_compute", None)
          or getattr(flash_block_sizes, "bkv_compute", None)
          or bkv_compute
      )
      bkv_compute_in = (
          getattr(flash_block_sizes, "block_kv_compute_in", None)
          or getattr(flash_block_sizes, "bkv_compute_in", None)
          or bkv_compute_in
      )
      heads_per_tile = getattr(flash_block_sizes, "heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = getattr(flash_block_sizes, "vmem_limit_bytes", None) or vmem_limit_bytes
  if heads_per_tile is None:
    heads_per_tile = 1
  return CustomBlockSizes(bq, bkv, bkv_compute, bkv_compute_in, heads_per_tile, vmem_limit_bytes)


def _reshape_data_from_cudnn_flash(tensor):
  return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


def _reshape_data_for_cudnn_flash(tensor, heads):
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
  axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
  return jax.lax.with_sharding_constraint(reshaped_tensor, axis_names)


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
  axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
  return jax.lax.with_sharding_constraint(reshaped_tensor, axis_names)


def _reshape_heads_to_head_dim(tensor):
  b, h, s, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  reshaped_tensor = jnp.reshape(tensor, (b, -1, h * d))
  axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, HEAD))
  return jax.lax.with_sharding_constraint(reshaped_tensor, axis_names)


def _unflatten_heads(tensor, heads):
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
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
  return PartitionSpec(*(_replace_mesh_axis(axis_name, old_axis, new_axes) for axis_name in axis_names))


def _create_internal_ulysses_ring_mesh(
    mesh: Mesh,
    ring_shards: int,
    ulysses_shards: int,
    ring_axis: str = INTERNAL_RING_AXIS,
    ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
) -> Mesh:
  mesh_axis_names = tuple(mesh.axis_names)
  context_axis_index = mesh_axis_names.index(CONTEXT)
  devices = mesh.devices
  new_shape = devices.shape[:context_axis_index] + (ring_shards, ulysses_shards) + devices.shape[context_axis_index + 1 :]
  new_axis_names = (
      mesh_axis_names[:context_axis_index] + (ring_axis, ulysses_axis) + mesh_axis_names[context_axis_index + 1 :]
  )
  return Mesh(devices.reshape(new_shape), new_axis_names)


def _reshape_data_for_flash(tensor, heads, num_context_shards=1):
  if tensor.ndim != 4:
    tensor = _unflatten_heads(tensor, heads)
  org_seq_len = tensor.shape[2]
  if num_context_shards <= 1:
    return tensor, org_seq_len
  rem = org_seq_len % num_context_shards
  if rem == 0:
    return tensor, org_seq_len
  pad_width = [(0, 0)] * tensor.ndim
  pad_width[2] = (0, num_context_shards - rem)
  return jnp.pad(tensor, pad_width), org_seq_len


def _pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1):
  tensor, _ = _reshape_data_for_flash(tensor, heads)
  kv_size = tensor.shape[-1]
  head_dim_pad = 0
  if kv_size < 128:
    head_dim_pad = 128 - kv_size

  seq_len = tensor.shape[2]
  rem = seq_len % flash_block_size
  if rem != 0:
    seq_len_padded_pre = seq_len + (flash_block_size - rem)
  else:
    seq_len_padded_pre = seq_len

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
) -> BlockSizes:
  query_seq_len = _flash_sequence_length(query)
  key_seq_len = _flash_sequence_length(key)

  q_max_block_size = 1024 if dtype == jnp.bfloat16 else 512
  if key_seq_len != query_seq_len:
    kv_max_block_size = ((key_seq_len + 127) // 128) * 128
  else:
    kv_max_block_size = q_max_block_size

  if flash_block_sizes and key_seq_len == query_seq_len:
    if attention_kernel in ["tokamax_flash", "tokamax_ring"]:
      return _coerce_tokamax_block_sizes(flash_block_sizes)
    return flash_block_sizes

  block_size_q = flash_block_sizes.block_q if flash_block_sizes else q_max_block_size
  use_tokamax = attention_kernel in ["tokamax_flash", "tokamax_ring"]
  return pallas_splash_kernel.BlockSizes(
      block_q=block_size_q,
      block_kv_compute=min(kv_max_block_size, key_seq_len),
      block_kv=min(kv_max_block_size, key_seq_len),
      block_q_dkv=block_size_q,
      block_kv_dkv=min(kv_max_block_size, key_seq_len),
      block_kv_dkv_compute=min(kv_max_block_size, query_seq_len),
      block_q_dq=None if use_tokamax else block_size_q,
      block_kv_dq=None if use_tokamax else min(kv_max_block_size, query_seq_len),
      use_fused_bwd_kernel=True if use_tokamax else False,
  )


def convert_to_tokamax_splash_config(
    block_sizes: BlockSizes,
    q_layout: splash_attention_kernel.QKVLayout = splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    k_layout: splash_attention_kernel.QKVLayout = splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    v_layout: splash_attention_kernel.QKVLayout = splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    residual_checkpoint_name: str | None = None,
    attn_logits_soft_cap: float | None = None,
    fuse_reciprocal: bool = True,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    max_logit_const: float | None = None,
    interpret: bool = False,
    dq_reduction_steps: int | None = None,
) -> splash_attention_kernel.SplashConfig:
  assert block_sizes.use_fused_bwd_kernel, "Tokamax Splash attention only supports fused bwd kernel."
  return splash_attention_kernel.SplashConfig(
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


def _build_padding_segment_ids(
    query_seq_len: int,
    q_padded_len: int,
    key_seq_len: int,
    kv_padded_len: int,
    attention_mask: jax.Array | None,
    segment_ids_cls=splash_attention_kernel.SegmentIds,
):
  q_indices = jax.lax.broadcasted_iota(jnp.int32, (q_padded_len,), 0)
  q_segment_ids = (q_indices < query_seq_len).astype(jnp.int32)

  kv_indices = jax.lax.broadcasted_iota(jnp.int32, (kv_padded_len,), 0)
  kv_segment_ids = (kv_indices < key_seq_len).astype(jnp.int32)

  if attention_mask is not None:
    mask_len = min(key_seq_len, attention_mask.shape[1])
    kv_mask_for_batch = attention_mask[0, :mask_len]
    if key_seq_len > mask_len:
      kv_mask_for_batch = jnp.concatenate(
          [kv_mask_for_batch, jnp.ones((key_seq_len - mask_len,), jnp.int32)],
          axis=0,
      )
    if kv_padded_len > key_seq_len:
      kv_mask_for_batch = jnp.concatenate(
          [
              kv_mask_for_batch,
              jnp.zeros((kv_padded_len - key_seq_len,), jnp.int32),
          ],
          axis=0,
      )
    kv_segment_ids = (kv_segment_ids * kv_mask_for_batch).astype(jnp.int32)

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


def _ulysses_head_chunk_ranges(num_heads: int, ulysses_shards: int, num_chunks: int):
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


def _run_chunked_ulysses_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    num_heads: int,
    ulysses_shards: int,
    ulysses_attention_chunks: int,
    attention_fn,
) -> jax.Array:
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
  return attention_fn(query, key, value)


def _max_row_norm_per_head(x: jax.Array) -> jax.Array:
  """Largest row L2 norm per head of a `[B, H, S, D]` activation."""
  row_sq = jnp.square(x).sum(axis=-1, dtype=jnp.float32)
  return jnp.sqrt(row_sq.max(axis=(0, 2))) * 1.01


def _query_chunk_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    precision: jax.lax.Precision,
    key_chunk_size: int = 4096,
) -> jax.Array:
  """Multi-head dot product attention with a limited number of queries."""
  num_kv, num_heads, k_features = key.shape[-3:]
  v_features = value.shape[-1]
  key_chunk_size = min(key_chunk_size, num_kv)
  key_pad_len = (key_chunk_size - (num_kv % key_chunk_size)) % key_chunk_size
  if key_pad_len > 0:
    k_pad_width = [(0, 0)] * key.ndim
    k_pad_width[-3] = (0, key_pad_len)
    key = jnp.pad(key, k_pad_width)
    v_pad_width = [(0, 0)] * value.ndim
    v_pad_width[-3] = (0, key_pad_len)
    value = jnp.pad(value, v_pad_width)
  padded_num_kv = key.shape[-3]
  valid_mask = jnp.arange(padded_num_kv) < num_kv

  query = query / jnp.sqrt(k_features)

  @functools.partial(jax.checkpoint, prevent_cse=False)
  def summarize_chunk(query, key, value, mask_chunk):
    attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)
    if key_pad_len > 0:
      attn_weights = jnp.where(mask_chunk, attn_weights, SAFE_MIN_ATTN_WEIGHT)
    max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    max_score = jax.lax.stop_gradient(max_score)
    exp_weights = jnp.exp(attn_weights - max_score)
    if key_pad_len > 0:
      exp_weights = jnp.where(mask_chunk, exp_weights, 0.0)
    exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
    max_score = jnp.squeeze(max_score, axis=-1)
    return (exp_values, exp_weights.sum(axis=-1), max_score)

  def chunk_scanner(chunk_idx):
    key_chunk = jax.lax.dynamic_slice(
        operand=key,
        start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],
        slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],
    )
    value_chunk = jax.lax.dynamic_slice(
        operand=value,
        start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],
        slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],
    )
    mask_chunk = jax.lax.dynamic_slice_in_dim(valid_mask, chunk_idx, key_chunk_size, axis=0)
    return summarize_chunk(query, key_chunk, value_chunk, mask_chunk)

  chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, padded_num_kv, key_chunk_size))
  global_max = jnp.max(chunk_max, axis=0, keepdims=True)
  max_diffs = jnp.exp(chunk_max - global_max)
  chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
  chunk_weights *= max_diffs
  all_values = chunk_values.sum(axis=0)
  all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
  return all_values / all_weights


def jax_memory_efficient_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
) -> jax.Array:
  """Memory-efficient attention using chunking."""
  num_q, num_heads, q_features = query.shape[-3:]
  query_chunk_size = min(query_chunk_size, num_q)
  pad_len = (query_chunk_size - (num_q % query_chunk_size)) % query_chunk_size
  if pad_len > 0:
    pad_width = [(0, 0)] * query.ndim
    pad_width[-3] = (0, pad_len)
    query = jnp.pad(query, pad_width)
  padded_q_len = query.shape[-3]

  def chunk_scanner(chunk_idx, _):
    query_chunk = jax.lax.dynamic_slice(
        operand=query,
        start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],
        slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, padded_q_len), num_heads, q_features],
    )
    return (
        chunk_idx + query_chunk_size,
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
      length=math.ceil(padded_q_len / query_chunk_size),
  )
  out = jnp.concatenate(res, axis=-3)
  if pad_len > 0:
    out = out[..., :num_q, :, :]
  return out
