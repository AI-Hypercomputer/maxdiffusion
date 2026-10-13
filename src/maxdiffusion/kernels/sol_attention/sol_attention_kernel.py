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

Pallas TPU kernel for Sol-Attn (arXiv:2607.24027, Algorithm 1).

One grid program per (head, query block). Each program keeps a single
online-softmax state and runs two branches into it:

  1. Approximate branch (outer loop of Algorithm 1): one GEMM of the query
     block against all pooled keys. Unselected blocks contribute their
     zeroth-order term (Eqs. 9-10) through a log-count bias; the selected
     blocks (read from the SMEM routing row) are masked on chip so nothing is
     double counted.
  2. Exact branch (inner loop): walks the selected key/value blocks of this
     query block through a ring of manual HBM->VMEM DMAs, so only routed
     blocks are ever read.

Routing (pooling + thresholds) runs in a tiny XLA pre-pass, see `routing.py`;
only the compacted per-query-block index rows reach the kernel.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion.kernels.sol_attention import routing as sol_routing

NUM_LANES = 128
NUM_SUBLANES = 8
# The hot loop fuses selected blocks into score tiles of about this many keys;
# measured on v6e, 1024 beats 256/512 by 10-25% for 128-256 token blocks.
TARGET_TILE_KEYS = 1024
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
LOG2E = 1.4426950408889634
_MASK_VALUE = sol_routing.MASK_VALUE
_HALF_MASK_VALUE = 0.5 * sol_routing.MASK_VALUE


def _round_up(x: int, multiple: int) -> int:
  return -(-x // multiple) * multiple


def _resolve_pipelining(block_kv: int, group_size: Optional[int], num_buffers: Optional[int]) -> tuple[int, int]:
  if group_size is None:
    group_size = max(1, TARGET_TILE_KEYS // block_kv)
  if num_buffers is None:
    num_buffers = 2 * group_size
  if group_size < 1 or num_buffers % group_size or num_buffers < 2 * group_size:
    raise ValueError(f"num_buffers={num_buffers} must be a multiple of group_size={group_size} and >= 2 * group_size.")
  return group_size, num_buffers


def _sol_kernel(
    # Inputs.
    q_ref,  # VMEM [bq, d], pre-scaled by scale * log2(e)
    k_pooled_ref,  # VMEM [nk_pad, d]
    v_pooled_ref,  # VMEM [nk_pad, d]
    log_count_ref,  # VMEM [nk_pad, 1] f32: log2(tokens in block j), MASK for padding blocks
    rows_hbm,  # HBM  [H, Nq, 8, row_cols] int32; each (8, row_cols) row is one tile-aligned DMA
    k_hbm,  # HBM  [H, Lp, d]
    v_hbm,  # HBM  [H, Lp, d]
    # Outputs.
    o_ref,  # VMEM [d, bq] (transposed)
    # Scratch.
    row_smem,  # SMEM [2, 8, row_cols] int32, double-buffered across grid steps
    k_buf,  # VMEM [num_buffers, bkv, d]
    v_buf,  # VMEM [num_buffers, bkv, d]
    bias_scr,  # VMEM [nk_pad, 1] f32
    o_scr,  # VMEM [d, bq] f32
    row_sem,  # DMA semaphores [2]
    kv_sem,  # DMA semaphores [2, num_buffers]
    *,
    seq_len: int,
    block_kv: int,
    num_buffers: int,
    group_size: int,
):
  """One (head, query block) program: approximate branch + exact sparse loop."""
  # Scores are laid out [keys, queries] so softmax reductions run over
  # sublanes (cheap VPU ops) instead of lanes, as in the dense splash kernel.
  h = pl.program_id(0)
  i = pl.program_id(1)
  num_q_blocks = pl.num_programs(1)
  step = h * num_q_blocks + i
  num_steps = pl.num_programs(0) * num_q_blocks
  row_slot = lax.rem(step, 2)
  row_cols = row_smem.shape[2]

  def row_copy(n, slot):
    return pltpu.make_async_copy(
        rows_hbm.at[n // num_q_blocks, lax.rem(n, num_q_blocks)], row_smem.at[slot], row_sem.at[slot]
    )

  # The routing row of step n+1 is prefetched during step n (grid runs in
  # order), so only the very first step pays the DMA latency.
  @pl.when(step == 0)
  def fetch_first_row():
    row_copy(step, row_slot).start()

  row_copy(step, row_slot).wait()

  @pl.when(step + 1 < num_steps)
  def prefetch_next_row():
    row_copy(step + 1, 1 - row_slot).start()

  def row_entry(n):
    return row_smem[row_slot, n // row_cols, lax.rem(n, row_cols)]

  count = row_entry(0)

  def kv_copies(t):
    """K/V block DMAs for the t-th selected block, into ring slot t % num_buffers."""
    slot = lax.rem(t, num_buffers)
    start = pl.multiple_of(row_entry(t + 1) * block_kv, block_kv)
    k_copy = pltpu.make_async_copy(k_hbm.at[h, pl.ds(start, block_kv), :], k_buf.at[slot], kv_sem.at[0, slot])
    v_copy = pltpu.make_async_copy(v_hbm.at[h, pl.ds(start, block_kv), :], v_buf.at[slot], kv_sem.at[1, slot])
    return k_copy, v_copy

  def start_kv_copies(t):
    for c in kv_copies(t):
      c.start()

  # Ring of num_buffers block slots, consumed group_size blocks at a time.
  # Prime all but the last group; these DMAs overlap the approximate branch.
  for t in range(num_buffers - group_size):
    pl.when(t < count)(functools.partial(start_kv_copies, t))

  # Routing mask on chip: selected blocks get no approximate term.
  bias_scr[...] = log_count_ref[...]

  def mask_selected(t, carry):
    bias_scr[pl.ds(row_entry(t + 1), 1), :] = jnp.full((1, 1), _MASK_VALUE, jnp.float32)
    return carry

  lax.fori_loop(0, count, mask_selected, 0)

  q = q_ref[...]
  sv_dims = (((0,), (0,)), ((), ()))

  # Approximate branch: one GEMM over all pooled keys, zeroth-order term of
  # Eqs. (9)-(10) via the log2(count) bias.
  s = lax.dot_general(k_pooled_ref[...], q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
  s = s + bias_scr[...]
  m = jnp.max(s, axis=0, keepdims=True)
  p = jnp.where(s > _HALF_MASK_VALUE, jnp.exp2(s - m), 0.0)
  l = jnp.sum(p, axis=0, keepdims=True)
  o_scr[...] = lax.dot_general(v_pooled_ref[...], p.astype(v_pooled_ref.dtype), sv_dims, preferred_element_type=jnp.float32)

  ragged = seq_len % block_kv != 0
  last_block = seq_len // block_kv

  def exact_blocks(t0, m_prev, l_prev, size: int, prefetch: bool, mask_tail: bool):
    """Online-softmax update with selected blocks t0..t0+size-1.

    Their ring slots are contiguous, so `size` routed blocks form one
    [size * bkv, bq] score tile: routing stays fine-grained while each loop
    iteration amortizes the softmax rescale over a larger tile.
    """
    slot = lax.rem(t0, num_buffers)
    copies = [kv_copies(t0 + j) for j in range(size)]
    if prefetch:
      for j in range(size):
        nxt = t0 + num_buffers - group_size + j
        pl.when(nxt < count)(functools.partial(start_kv_copies, nxt))
    for k_copy, _ in copies:
      k_copy.wait()
    k = k_buf[pl.ds(slot, size)].reshape(size * block_kv, k_buf.shape[-1])
    s = lax.dot_general(k, q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    if mask_tail:
      valid = lax.broadcasted_iota(jnp.int32, s.shape, 0) < seq_len - last_block * block_kv
      s = jnp.where(valid, s, _MASK_VALUE)
    m_next = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
    alpha = jnp.exp2(m_prev - m_next)
    p = jnp.exp2(s - m_next)
    l_next = alpha * l_prev + jnp.sum(p, axis=0, keepdims=True)
    for _, v_copy in copies:
      v_copy.wait()
    v = v_buf[pl.ds(slot, size)].reshape(size * block_kv, v_buf.shape[-1])
    o_scr[...] = alpha * o_scr[...] + lax.dot_general(v, p.astype(v.dtype), sv_dims, preferred_element_type=jnp.float32)
    return m_next, l_next

  # Exact branch over routed blocks only, sharing the online-softmax state.
  # Indices are ascending, so a selected ragged tail block is always last; it
  # runs as a masked epilogue to keep the hot loop branch-free (a per-block
  # branch costs ~40% here). Every block's DMA is issued by the prologue or
  # by a hot-loop iteration, so the leftover/tail steps never prefetch.
  if ragged:
    has_tail = jnp.logical_and(count > 0, row_entry(count) == last_block)
    num_main = count - has_tail.astype(jnp.int32)
  else:
    num_main = count
  num_groups = num_main // group_size
  m, l = lax.fori_loop(
      0,
      num_groups,
      lambda g, c: exact_blocks(g * group_size, *c, size=group_size, prefetch=True, mask_tail=False),
      (m, l),
  )
  if group_size > 1:
    m, l = lax.fori_loop(
        num_groups * group_size,
        num_main,
        lambda t, c: exact_blocks(t, *c, size=1, prefetch=False, mask_tail=False),
        (m, l),
    )
  if ragged:
    m, l = lax.cond(
        has_tail,
        lambda c: exact_blocks(num_main, *c, size=1, prefetch=False, mask_tail=True),
        lambda c: c,
        (m, l),
    )
  o_ref[...] = (o_scr[...] * pl.reciprocal(l, approx=False)).astype(o_ref.dtype)


def sol_attention_from_routing(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    routing: sol_routing.SolRouting,
    *,
    seq_len: int,
    scale: float,
    block_sizes: sol_routing.SolBlockSizes,
    vmem_limit_bytes: Optional[int] = None,
    num_buffers: Optional[int] = None,
    group_size: Optional[int] = None,
    interpret: bool = False,
) -> jax.Array:
  """Runs the kernel on padded [H, Lp, d] tensors with precomputed routing.

  Returns the padded [H, Lp, d] output.
  """
  group_size, num_buffers = _resolve_pipelining(block_sizes.block_kv, group_size, num_buffers)
  num_heads, padded_len, head_dim = q.shape
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  nq, nk = padded_len // bq, padded_len // bkv
  nk_pad = _round_up(nk, NUM_LANES)
  # Row tables are (8, row_cols) tiles so HBM->SMEM DMAs are layout aligned.
  row_cols = _round_up(-(-(nk + 1) // NUM_SUBLANES), NUM_LANES)
  row_width = NUM_SUBLANES * row_cols

  def pad_blocks(x):
    widths = [(0, 0)] * x.ndim
    widths[-2] = (0, nk_pad - nk)
    return jnp.pad(x, widths)

  # exp2 everywhere: fold log2(e) into the query scale and the count bias.
  q = (q.astype(jnp.float32) * (scale * LOG2E)).astype(q.dtype)
  k_pooled = pad_blocks(routing.k_pooled).astype(k.dtype)
  v_pooled = pad_blocks(routing.v_pooled).astype(v.dtype)
  counts = jnp.pad(routing.kv_counts, (0, nk_pad - nk))
  log_count = jnp.where(counts > 0, jnp.log2(jnp.maximum(counts, 1).astype(jnp.float32)), _MASK_VALUE)[:, None]
  rows = sol_routing.compact_selection(routing.selected, row_width).reshape(num_heads, nq, NUM_SUBLANES, row_cols)

  kernel = functools.partial(_sol_kernel, seq_len=seq_len, block_kv=bkv, num_buffers=num_buffers, group_size=group_size)
  out = pl.pallas_call(
      kernel,
      grid=(num_heads, nq),
      in_specs=[
          pl.BlockSpec((None, bq, head_dim), lambda h, i: (h, i, 0)),
          pl.BlockSpec((None, nk_pad, head_dim), lambda h, i: (h, 0, 0)),
          pl.BlockSpec((None, nk_pad, head_dim), lambda h, i: (h, 0, 0)),
          pl.BlockSpec((nk_pad, 1), lambda h, i: (0, 0)),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec((None, head_dim, bq), lambda h, i: (h, 0, i)),
      out_shape=jax.ShapeDtypeStruct((num_heads, head_dim, padded_len), q.dtype),
      scratch_shapes=[
          pltpu.SMEM((2, NUM_SUBLANES, row_cols), jnp.int32),
          pltpu.VMEM((num_buffers, bkv, head_dim), k.dtype),
          pltpu.VMEM((num_buffers, bkv, head_dim), v.dtype),
          pltpu.VMEM((nk_pad, 1), jnp.float32),
          pltpu.VMEM((head_dim, bq), jnp.float32),
          pltpu.SemaphoreType.DMA((2,)),
          pltpu.SemaphoreType.DMA((2, num_buffers)),
      ],
      compiler_params=pltpu.CompilerParams(
          # Sequential grid: each step prefetches the next step's routing row.
          dimension_semantics=("arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=interpret,
      name="sol_attention",
  )(q, k_pooled, v_pooled, log_count, rows, k, v)
  return jnp.swapaxes(out, 1, 2)


def sol_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    scale: float,
    beta: float,
    block_sizes: sol_routing.SolBlockSizes = sol_routing.SolBlockSizes(),
    always_exact: Optional[jax.Array] = None,
    vmem_limit_bytes: Optional[int] = None,
    num_buffers: Optional[int] = None,
    group_size: Optional[int] = None,
    interpret: bool = False,
    return_density: bool = False,
):
  """Sol-Attn self-attention on [..., L, d] tensors (any leading dims).

  Args:
    q, k, v: Query/key/value [..., L, d]; all leading dims are folded into one
      head axis. Keys must cover the full sequence (no sequence sharding).
    scale: Attention logit scale (usually head_dim ** -0.5).
    beta: Standardized threshold cutoff. Expected density ~= 1 - Phi(beta).
    block_sizes: Query/key block granularity.
    always_exact: Optional bool [Nk] of key blocks always computed exactly.
    vmem_limit_bytes: Optional Mosaic VMEM limit.
    num_buffers: Exact-branch K/V DMA ring depth in blocks (default 2 * group_size).
    group_size: Selected blocks fused into one score tile per loop step
      (default: TARGET_TILE_KEYS // block_kv).
    interpret: Run the Pallas kernel in interpret mode (CPU testing).
    return_density: Also return the exact-branch block density (scalar).

  Returns:
    Output [..., L, d] in q.dtype (and the density if requested).
  """
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError(f"Sol-Attn needs matching self-attention q/k/v, got {q.shape}, {k.shape}, {v.shape}.")
  lead = q.shape[:-2]
  seq_len, head_dim = q.shape[-2:]
  padded_len = sol_routing.padded_length(seq_len, block_sizes)
  qp, kp, vp = (sol_routing.pad_sequence(x.reshape(-1, seq_len, head_dim), padded_len) for x in (q, k, v))
  with jax.named_scope("sol_routing"):
    routing = sol_routing.compute_routing(
        qp,
        kp,
        vp,
        seq_len=seq_len,
        scale=scale,
        beta=beta,
        block_sizes=block_sizes,
        always_exact=always_exact,
    )
  with jax.named_scope("sol_kernel"):
    out = sol_attention_from_routing(
        qp,
        kp,
        vp,
        routing,
        seq_len=seq_len,
        scale=scale,
        block_sizes=block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        num_buffers=num_buffers,
        group_size=group_size,
        interpret=interpret,
    )
  out = out[:, :seq_len].reshape(*lead, seq_len, head_dim)
  if return_density:
    return out, routing.density()
  return out
