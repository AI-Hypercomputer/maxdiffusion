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

"""Custom Pallas flash attention kernel for TPU."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

# Default block sizes (tuned for 720p Wan2.1 on v6e/v7x)
DEFAULT_BQSIZE = 3328
DEFAULT_BKVSIZE = 2816
# Cranked up to 1024 for massive MXU throughput
DEFAULT_BKVCOMPUTESIZE = 1024
# Kept at 256 to protect VPU registers (V1 Optimization)
DEFAULT_BKVCOMPUTEINSIZE = 256


def _exp2(x: jax.Array, use_base2_exp: bool = True) -> jax.Array:
  if use_base2_exp:
    return jnp.exp2(x)
  return jnp.exp(x)


class _BlockSizes:
  __slots__ = ("block_q", "block_kv", "block_kv_compute")

  def __init__(self, block_q: int, block_kv: int, block_kv_compute: int | None = None):
    self.block_q = block_q
    self.block_kv = block_kv
    self.block_kv_compute = block_kv_compute if block_kv_compute is not None else block_kv


def _flash_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    q_seq_len: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  _, _, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def compute_body(kv_compute_index, _):
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    q = q_ref[...]
    o_prev = o_scratch_ref[:]

    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    k_chunk = k_ref[slice_k, :]

    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]

    # --- V1 VPU REGISTER TILING ---
    step = bkv_compute_in
    for i in range(0, qk.shape[0], step):
      qk_slice = qk[i : i + step]

      m_curr = qk_slice.max(axis=0)[None, :]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = _exp2(qk_slice - m_next[0:1], use_base2_exp=use_base2_exp)
      l_curr = s_curr.sum(axis=0, keepdims=True)

      alpha = jnp.exp2(m_prev - m_next)
      l_next = l_curr + alpha * l_prev

      sv_dims = (((0,), (0,)), ((), ()))
      o_curr = lax.dot_general(
          v_chunk[i : i + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )

      alpha_o = alpha[0:1, ...]
      o_prev = alpha_o * o_prev + o_curr

      m_prev, l_prev = m_next, l_next
    # --- END V1 TILING ---

    m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[:] = o_prev

  def last_compute_body(kv_compute_index):
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    q = q_ref[...]
    o_prev = o_scratch_ref[:]

    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(kv_compute_index * bkv_compute, slice_k_len)
    k_chunk = k_ref[slice_k, :]

    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]

    # --- V1 VPU REGISTER TILING ---
    step = bkv_compute_in
    for i in range(0, qk.shape[0], step):
      qk_slice = qk[i : i + step]

      m_curr = qk_slice.max(axis=0)[None, :]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = _exp2(qk_slice - m_next[0:1], use_base2_exp=use_base2_exp)
      l_curr = s_curr.sum(axis=0, keepdims=True)

      alpha = jnp.exp2(m_prev - m_next)
      l_next = l_curr + alpha * l_prev

      sv_dims = (((0,), (0,)), ((), ()))
      o_curr = lax.dot_general(
          v_chunk[i : i + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )

      alpha_o = alpha[0:1, ...]
      o_prev = alpha_o * o_prev + o_curr

      m_prev, l_prev = m_next, l_next
    # --- END V1 TILING ---

    m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[:] = o_prev

  assert bkv % bkv_compute == 0

  @pl.when(j != grid_width - 1)
  def body():
    lax.fori_loop(0, (bkv // bkv_compute), compute_body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def last_body():
    if kv_seq_len % bkv == 0:
      iter_num = bkv // bkv_compute
      lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
    else:
      remain_kv_seq_len = kv_seq_len % bkv
      iter_num = (remain_kv_seq_len + bkv_compute - 1) // bkv_compute
      if remain_kv_seq_len % bkv_compute == 0:
        lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
      else:
        lax.fori_loop(0, iter_num - 1, compute_body, None, unroll=True)
        last_compute_body(iter_num - 1)

  @pl.when(j == grid_width - 1)
  def end():
    l = l_scratch_ref[...]
    l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
    o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def _flash_attention_kernel_mhpt(
    q_ref,
    k_ref,
    v_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    q_seq_len: int,
    kv_seq_len: int,
    heads_per_tile: int,
    use_base2_exp: bool = True,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  _, _, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def compute_body(kv_compute_index, _):
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)

    for h_local in range(heads_per_tile):
      m_prev = m_scratch_ref[h_local]
      l_prev = l_scratch_ref[h_local]
      q = q_ref[h_local]
      o_prev = o_scratch_ref[h_local]

      k_chunk = k_ref[h_local, slice_k, :]
      qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
      v_chunk = v_ref[h_local, slice_k, :]

      # --- V1 VPU REGISTER TILING ---
      step = bkv_compute_in
      for i in range(0, qk.shape[0], step):
        qk_slice = qk[i : i + step]

        m_curr = qk_slice.max(axis=0)[None, :]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = _exp2(qk_slice - m_next[0:1], use_base2_exp=use_base2_exp)
        l_curr = s_curr.sum(axis=0, keepdims=True)

        alpha = jnp.exp2(m_prev - m_next)
        l_next = l_curr + alpha * l_prev

        sv_dims = (((0,), (0,)), ((), ()))
        o_curr = lax.dot_general(
            v_chunk[i : i + step],
            s_curr.astype(q_ref.dtype),
            sv_dims,
            preferred_element_type=float32,
        )

        alpha_o = alpha[0:1, ...]
        o_prev = alpha_o * o_prev + o_curr

        m_prev, l_prev = m_next, l_next
      # --- END V1 TILING ---

      m_scratch_ref[h_local] = m_prev
      l_scratch_ref[h_local] = l_prev
      o_scratch_ref[h_local] = o_prev

  def last_compute_body(kv_compute_index):
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(kv_compute_index * bkv_compute, slice_k_len)

    for h_local in range(heads_per_tile):
      m_prev = m_scratch_ref[h_local]
      l_prev = l_scratch_ref[h_local]
      q = q_ref[h_local]
      o_prev = o_scratch_ref[h_local]

      k_chunk = k_ref[h_local, slice_k, :]
      qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
      v_chunk = v_ref[h_local, slice_k, :]

      # --- V1 VPU REGISTER TILING ---
      step = bkv_compute_in
      for i in range(0, qk.shape[0], step):
        qk_slice = qk[i : i + step]

        m_curr = qk_slice.max(axis=0)[None, :]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = _exp2(qk_slice - m_next[0:1], use_base2_exp=use_base2_exp)
        l_curr = s_curr.sum(axis=0, keepdims=True)

        alpha = jnp.exp2(m_prev - m_next)
        l_next = l_curr + alpha * l_prev

        sv_dims = (((0,), (0,)), ((), ()))
        o_curr = lax.dot_general(
            v_chunk[i : i + step],
            s_curr.astype(q_ref.dtype),
            sv_dims,
            preferred_element_type=float32,
        )

        alpha_o = alpha[0:1, ...]
        o_prev = alpha_o * o_prev + o_curr

        m_prev, l_prev = m_next, l_next
      # --- END V1 TILING ---

      m_scratch_ref[h_local] = m_prev
      l_scratch_ref[h_local] = l_prev
      o_scratch_ref[h_local] = o_prev

  assert bkv % bkv_compute == 0

  @pl.when(j != grid_width - 1)
  def body():
    lax.fori_loop(0, (bkv // bkv_compute), compute_body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def last_body():
    if kv_seq_len % bkv == 0:
      iter_num = bkv // bkv_compute
      lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
    else:
      remain_kv_seq_len = kv_seq_len % bkv
      iter_num = (remain_kv_seq_len + bkv_compute - 1) // bkv_compute
      if remain_kv_seq_len % bkv_compute == 0:
        lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
      else:
        lax.fori_loop(0, iter_num - 1, compute_body, None, unroll=True)
        last_compute_body(iter_num - 1)

  @pl.when(j == grid_width - 1)
  def end():
    for h_local in range(heads_per_tile):
      l = l_scratch_ref[h_local]
      l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
      o_ref[h_local] = (o_scratch_ref[h_local] * l_inv).astype(o_ref.dtype)


def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_experimental_scheduler: bool = True,
    use_base2_exp: bool = True,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  num_kv_heads = k.shape[0]
  padded_kv_seq_len = k.shape[1]

  actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
  actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def out_index_map(h, i, j, *_):
    return h, 0, i

  def k_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)

  def v_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
      pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, actual_q_seq_len), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
      pl.BlockSpec((None, head_dim_v, bq), out_index_map),
  ]
  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  grid = (num_q_heads, grid_height, grid_width)

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bq=bq,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          q_seq_len=actual_q_seq_len,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
      ),
      out_shape=out_shapes,
  )(q, k, v)
  return all_out[-1]


def _splash_attention_forward_mhpt(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    heads_per_tile: int,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = True,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  num_kv_heads = k.shape[0]
  actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
  actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else k.shape[1]
  hpt = heads_per_tile

  assert num_q_heads % hpt == 0, f"num_heads {num_q_heads} must be divisible by heads_per_tile {hpt}"
  assert num_q_heads == num_kv_heads, "MHPT currently requires num_q_heads == num_kv_heads (no GQA)"

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def k_index_map(h, i, j, *_):
    return (h, j, 0)

  def v_index_map(h, i, j, *_):
    return (h, j, 0)

  def out_index_map(h, i, j, *_):
    return (h, 0, i)

  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_qk), k_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_v), v_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, actual_q_seq_len), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, head_dim_v, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, head_dim_v, bq), out_index_map),
  ]
  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  grid = (num_q_heads // hpt, grid_height, grid_width)

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel_mhpt,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bq=bq,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          q_seq_len=actual_q_seq_len,
          kv_seq_len=actual_kv_seq_len,
          heads_per_tile=hpt,
          use_base2_exp=use_base2_exp,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
      ),
      out_shape=out_shapes,
  )(q, k, v)
  return all_out[-1]


def make_splash_mha(
    block_sizes: _BlockSizes,
    bkv_compute_in: int = DEFAULT_BKVCOMPUTEINSIZE,
    orig_q_seq_len: int | None = None,
    orig_kv_seq_len: int | None = None,
    heads_per_tile: int = 1,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = True,
):
  def _splash_attention(q, k, v):
    if heads_per_tile > 1:
      return _splash_attention_forward_mhpt(
          q, k, v, block_sizes, bkv_compute_in, heads_per_tile, q_seq_len=orig_q_seq_len, kv_seq_len=orig_kv_seq_len, use_base2_exp=use_base2_exp, use_experimental_scheduler=use_experimental_scheduler
      )
    return _splash_attention_forward(
        q, k, v, block_sizes, bkv_compute_in, q_seq_len=orig_q_seq_len, kv_seq_len=orig_kv_seq_len, use_base2_exp=use_base2_exp, use_experimental_scheduler=use_experimental_scheduler
    )

  return _splash_attention


# ---------------------------------------------------------------------------
# High-level attention function with shard_map
# ---------------------------------------------------------------------------


def tpu_custom_attention(
    query,
    key,
    value,
    mesh,
    *,
    scale=None,
    block_q=DEFAULT_BQSIZE,
    block_kv=DEFAULT_BKVSIZE,
    block_kv_compute=DEFAULT_BKVCOMPUTESIZE,
    block_kv_compute_in=DEFAULT_BKVCOMPUTEINSIZE,
    heads_per_tile=1,
):
  _LOG2_E = 1.44269504
  num_heads = query.shape[1]

  def _attention_on_slices(q, k, v):
    scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    q = q * scale_factor * _LOG2_E

    def _pad_to_multiple(x, multiple, axis):
      seq_len = x.shape[axis]
      pad_len = (multiple - seq_len % multiple) % multiple
      if pad_len == 0:
        return x, seq_len
      pad_width = [(0, 0)] * x.ndim
      pad_width[axis] = (0, pad_len)
      return jnp.pad(x, pad_width), seq_len

    def _kernel_3d(q_3d, k_3d, v_3d):
      q_orig_len = q_3d.shape[1]
      kv_orig_len = k_3d.shape[1]

      q_3d_padded, _ = _pad_to_multiple(q_3d, block_q, axis=1)
      k_3d_padded, _ = _pad_to_multiple(k_3d, block_kv, axis=1)
      v_3d_padded, _ = _pad_to_multiple(v_3d, block_kv, axis=1)

      padded_q_seq_len = q_3d_padded.shape[1]
      padded_kv_seq_len = k_3d_padded.shape[1]

      bsizes = _BlockSizes(
          block_q=min(block_q, padded_q_seq_len),
          block_kv=min(block_kv, padded_kv_seq_len),
          block_kv_compute=min(block_kv_compute, padded_kv_seq_len),
      )
      splash_kernel = make_splash_mha(
          block_sizes=bsizes,
          bkv_compute_in=block_kv_compute_in,
          orig_q_seq_len=q_orig_len,
          orig_kv_seq_len=kv_orig_len,
          heads_per_tile=heads_per_tile,
      )
      out = splash_kernel(
          q_3d_padded.astype(jnp.bfloat16),
          k_3d_padded,
          v_3d_padded,
      )
      out = jnp.swapaxes(out, 1, 2)
      return out[:, :q_orig_len, ...]

    return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q, k, v)

  batch_size = query.shape[0]
  if num_heads < mesh.size:
    q_partition_spec = P()
    kv_partition_spec = P()
    out_constraint = P()
  else:
    axis_names = mesh.axis_names
    if len(axis_names) == 1:
      tp_axis = axis_names[0]
      q_partition_spec = P(None, tp_axis, None, None)
      kv_partition_spec = P(None, tp_axis, None, None)
      out_constraint = P(None, None, tp_axis, None)
    elif len(axis_names) == 2:
      dp_axis, tp_axis = axis_names[0], axis_names[1]
      dp_size = mesh.shape[dp_axis]
      if batch_size >= dp_size:
        q_partition_spec = P(dp_axis, tp_axis, None, None)
        kv_partition_spec = P(dp_axis, tp_axis, None, None)
        out_constraint = P(dp_axis, None, tp_axis, None)
      else:
        all_axes = tuple(axis_names)
        q_partition_spec = P(None, all_axes, None, None)
        kv_partition_spec = P(None, all_axes, None, None)
        out_constraint = P(None, None, all_axes, None)
    else:
      q_partition_spec = P(axis_names[0], axis_names[1], axis_names[2], None)
      kv_partition_spec = P(axis_names[0], axis_names[1], None, None)
      out_constraint = P(axis_names[0], None, (axis_names[1], axis_names[2]), None)

  sharded_fn = shard_map(
      _attention_on_slices,
      mesh=mesh,
      in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
      out_specs=q_partition_spec,
      check_rep=False,
  )
  out = sharded_fn(query, key, value)
  out = jax.lax.with_sharding_constraint(out, out_constraint)
  return out


# ---------------------------------------------------------------------------
# TorchAX SDPA wrapper
# ---------------------------------------------------------------------------


def make_custom_splash_sdpa(mesh, env, **kwargs):
  bq = kwargs.get("block_q", DEFAULT_BQSIZE)
  bkv = kwargs.get("block_kv", DEFAULT_BKVSIZE)
  bkv_compute = kwargs.get("block_kv_compute", DEFAULT_BKVCOMPUTESIZE)
  bkv_compute_in = kwargs.get("block_kv_compute_in", DEFAULT_BKVCOMPUTEINSIZE)
  hpt = kwargs.get("heads_per_tile", 1)
  use_k_smooth = kwargs.get("use_k_smooth", True)

  def _simple_attention(q, k, v, scale=None):
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    attn = jnp.einsum("bhsd,bhtd->bhst", q * s, k)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.einsum("bhst,bhtd->bhsd", attn, v)

  def _sdpa(
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
  ):
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    num_heads = jquery.shape[1]

    if num_heads <= 8:
      result = _simple_attention(jquery, jkey, jvalue, scale=scale)
      return env.j2t_iso(result)

    if use_k_smooth:
      key_mean = jnp.mean(jkey, axis=2, keepdims=True)
      jkey = jkey - key_mean

    result = tpu_custom_attention(
        jquery,
        jkey,
        jvalue,
        mesh,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv_compute,
        block_kv_compute_in=bkv_compute_in,
        heads_per_tile=hpt,
    )
    return env.j2t_iso(result)

  return _sdpa
