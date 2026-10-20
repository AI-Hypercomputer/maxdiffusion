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
softmax state and runs two branches into it:

  1. Approximate branch (outer loop of Algorithm 1): one GEMM of the query
     block against all pooled keys. Unselected blocks contribute their
     zeroth-order term (Eqs. 9-10) through a log-count bias; the selected
     blocks (read from the SMEM routing row) are masked on chip so nothing is
     double counted.
  2. Exact branch (inner loop): walks the selected key/value blocks of this
     query block through a ring of manual HBM->VMEM DMAs, so only routed
     blocks are ever read.

The softmax offset is fixed per query from the pooled scores and key block
radii (see `_sol_kernel`), so the hot loop needs no running max or rescale.

Pooling runs in a fused single-pass kernel (`pooling.py`) and routing in a tiny
XLA pre-pass (`routing.py`); only the compacted per-query-block index rows
reach this kernel. Inputs need no padding to the block sizes.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion.kernels.sol_attention import pooling as sol_pooling
from maxdiffusion.kernels.sol_attention import routing as sol_routing

NUM_LANES = 128
NUM_SUBLANES = 8
# The hot loop fuses selected blocks into score tiles of about this many keys;
# measured on v6e, 1024 beats 256/512 by 10-25% for 128-256 token blocks.
TARGET_TILE_KEYS = 1024
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
LOG2E = 1.4426950408889634
# Fixed softmax offset (log2 units): max exp2 argument (f32 overflows at 128),
# and the widest bound gap per query tile before falling back to online softmax.
FIXED_M_HEADROOM = 80.0
FIXED_M_WINDOW = 120.0
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
    q_ref,  # VMEM [bq, d]; the last block may extend past the array end
    k_pooled_ref,  # VMEM [nk_pad, d]
    v_pooled_ref,  # VMEM [nk_pad, d]
    log_count_ref,  # VMEM [nk_pad, 1] f32: log2(tokens in block j), MASK for padding blocks
    radius_ref,  # VMEM [nk_pad, 1] f32: key block radius (see routing.block_radius)
    rows_hbm,  # HBM  [H, Nq, 8, row_cols] int32; each (8, row_cols) row is one tile-aligned DMA
    k_hbm,  # HBM  [H, L, d], L >= seq_len (no padding needed)
    v_hbm,  # HBM  [H, L, d]
    # Outputs.
    o_ref,  # VMEM [bq, d]
    # Scratch.
    row_smem,  # SMEM [2, 8, row_cols] int32, double-buffered across grid steps
    k_buf,  # VMEM [num_buffers, bkv, d]
    v_buf,  # VMEM [num_buffers, bkv, d]
    sel_scr,  # VMEM [nk_pad, 1] f32: 1 for blocks routed to the exact branch
    o_scr,  # VMEM [d, bq] f32
    row_sem,  # DMA semaphores [2]
    kv_sem,  # DMA semaphores [2, num_buffers]
    *,
    seq_len: int,
    block_kv: int,
    num_buffers: int,
    group_size: int,
    q_scale: float,
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

  kv_len = k_hbm.shape[1]

  def kv_start(blk):
    # A ragged tail block reads the last block_kv rows of an unpadded array;
    # the epilogue masks the rows that belong to the previous block.
    return jnp.minimum(blk * block_kv, kv_len - block_kv)

  def kv_copies(t):
    """K/V block DMAs for the t-th selected block, into ring slot t % num_buffers."""
    slot = lax.rem(t, num_buffers)
    start = kv_start(row_entry(t + 1))
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

  # Routing flags on chip, from the SMEM index row.
  sel_scr[...] = jnp.zeros_like(sel_scr)

  def flag_selected(t, carry):
    sel_scr[pl.ds(row_entry(t + 1), 1), :] = jnp.ones((1, 1), jnp.float32)
    return carry

  lax.fori_loop(0, count, flag_selected, 0)

  # exp2 everywhere: log2(e) is folded into the query scale and count bias.
  qf = q_ref[...].astype(jnp.float32) * q_scale
  q = qf.astype(q_ref.dtype)
  sv_dims = (((0,), (0,)), ((), ()))

  # Pooled-key scores [nk_pad, bq]: they drive both the approximate branch
  # and the per-query softmax offset m.
  s = lax.dot_general(k_pooled_ref[...], q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
  sel = sel_scr[...] > 0.5
  log_count = log_count_ref[...]
  approx_logits = s + log_count
  # Fixed offset instead of a running max: no per-tile max reduction and no
  # rescaling of the accumulators in the hot loop (~20% of kernel time).
  # m_est <= the true max (a block's mean score never exceeds its max), and
  # ub >= every token score (q.k_t <= q.mean(K_j) + ||q|| * radius_j). With
  # m = max(m_est, ub - headroom), no exp2 argument exceeds the headroom (no
  # f32 overflow), and if ub - m_est <= window the largest term is
  # >= 2^(headroom - window) (no underflow). Tiles with a wider window, e.g.
  # from an outlier key that inflates a block radius, fall back to the online
  # (running-max) softmax.
  ones = jnp.ones((NUM_SUBLANES, qf.shape[1]), jnp.float32)
  q_norm = jnp.sqrt(lax.dot_general(ones, qf * qf, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)[:1])
  m_est = jnp.max(jnp.where(sel, s, approx_logits), axis=0, keepdims=True)
  ub = jnp.max(jnp.where(sel, s + radius_ref[...] * q_norm, _MASK_VALUE), axis=0, keepdims=True)
  fixed = jnp.max(ub - m_est) <= FIXED_M_WINDOW
  m = jnp.where(fixed, jnp.maximum(m_est, ub - FIXED_M_HEADROOM), m_est)

  # Approximate branch: zeroth-order term of Eqs. (9)-(10) for unselected,
  # non-padding blocks via the log2(count) bias. Its logits are <= m_est.
  keep = jnp.logical_and(jnp.logical_not(sel), log_count > _HALF_MASK_VALUE)
  p = jnp.where(keep, jnp.exp2(approx_logits - m), 0.0)
  l = jnp.sum(p, axis=0, keepdims=True)
  o_scr[...] = lax.dot_general(v_pooled_ref[...], p.astype(v_pooled_ref.dtype), sv_dims, preferred_element_type=jnp.float32)

  ragged = seq_len % block_kv != 0
  last_block = seq_len // block_kv

  def exact_blocks(t0, carry, size: int, prefetch: bool, mask_tail: bool, online: bool):
    """Accumulates selected blocks t0..t0+size-1 into the softmax state.

    Their ring slots are contiguous, so `size` routed blocks form one
    [size * bkv, bq] score tile: routing stays fine-grained while each loop
    iteration works on a large tile. carry is (m, l) if online else l.
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
      rows = kv_start(last_block) + lax.broadcasted_iota(jnp.int32, s.shape, 0)
      valid = jnp.logical_and(rows >= last_block * block_kv, rows < seq_len)
      s = jnp.where(valid, s, _MASK_VALUE)
    if online:
      m_prev, l_prev = carry
      m_next = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
      alpha = jnp.exp2(m_prev - m_next)
    else:
      m_next, l_prev, alpha = m, carry, None
    p = jnp.exp2(s - m_next)
    l_next = jnp.sum(p, axis=0, keepdims=True) + (l_prev if alpha is None else alpha * l_prev)
    for _, v_copy in copies:
      v_copy.wait()
    v = v_buf[pl.ds(slot, size)].reshape(size * block_kv, v_buf.shape[-1])
    pv = lax.dot_general(v, p.astype(v.dtype), sv_dims, preferred_element_type=jnp.float32)
    if online:
      o_scr[...] = alpha * o_scr[...] + pv
      return m_next, l_next
    o_scr[...] += pv
    return l_next

  # Exact branch over routed blocks only, sharing the softmax state.
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

  def exact_branch(online: bool):
    step = functools.partial(exact_blocks, online=online)
    carry = (m, l) if online else l
    carry = lax.fori_loop(
        0,
        num_groups,
        lambda g, c: step(g * group_size, c, size=group_size, prefetch=True, mask_tail=False),
        carry,
    )
    if group_size > 1:
      carry = lax.fori_loop(
          num_groups * group_size,
          num_main,
          lambda t, c: step(t, c, size=1, prefetch=False, mask_tail=False),
          carry,
      )
    if ragged:
      carry = lax.cond(
          has_tail,
          lambda c: step(num_main, c, size=1, prefetch=False, mask_tail=True),
          lambda c: c,
          carry,
      )
    return carry[1] if online else carry

  l = lax.cond(fixed, lambda: exact_branch(online=False), lambda: exact_branch(online=True))
  # Transposing on chip saves an XLA relayout of the whole output.
  o_ref[...] = (o_scr[...] * pl.reciprocal(l, approx=False)).T.astype(o_ref.dtype)


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
  """Runs the kernel on [H, L, d] tensors (L >= seq_len) with precomputed routing.

  The tensors need no padding to the block sizes; zero-padded inputs work too.

  Returns:
    The [H, L, d] output (rows past seq_len are unspecified).
  """
  group_size, num_buffers = _resolve_pipelining(block_sizes.block_kv, group_size, num_buffers)
  num_heads, length, head_dim = q.shape
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  nq, nk = routing.selected.shape[1:]
  if nq != -(-length // bq) or nk != -(-length // bkv) or length < max(bq, bkv) or seq_len > length:
    raise ValueError(f"Routing {routing.selected.shape} does not match q {q.shape} with blocks {block_sizes}.")
  nk_pad = _round_up(nk, NUM_LANES)
  # Row tables are (8, row_cols) tiles so HBM->SMEM DMAs are layout aligned.
  row_cols = _round_up(-(-(nk + 1) // NUM_SUBLANES), NUM_LANES)
  row_width = NUM_SUBLANES * row_cols

  def pad_blocks(x):
    widths = [(0, 0)] * x.ndim
    widths[-2] = (0, nk_pad - nk)
    return jnp.pad(x, widths)

  k_pooled = pad_blocks(routing.k_pooled).astype(k.dtype)
  v_pooled = pad_blocks(routing.v_pooled).astype(v.dtype)
  counts = jnp.pad(routing.kv_counts, (0, nk_pad - nk))
  log_count = jnp.where(counts > 0, jnp.log2(jnp.maximum(counts, 1).astype(jnp.float32)), _MASK_VALUE)[:, None]
  rows = sol_routing.compact_selection(routing.selected, row_width).reshape(num_heads, nq, NUM_SUBLANES, row_cols)
  radius = routing.k_radius if routing.k_radius is not None else jnp.zeros((num_heads, nk), jnp.float32)
  radius = jnp.pad(radius.astype(jnp.float32), ((0, 0), (0, nk_pad - nk)))[..., None]

  kernel = functools.partial(
      _sol_kernel,
      seq_len=seq_len,
      block_kv=bkv,
      num_buffers=num_buffers,
      group_size=group_size,
      q_scale=scale * LOG2E,
  )
  return pl.pallas_call(
      kernel,
      grid=(num_heads, nq),
      in_specs=[
          pl.BlockSpec((None, bq, head_dim), lambda h, i: (h, i, 0)),
          pl.BlockSpec((None, nk_pad, head_dim), lambda h, i: (h, 0, 0)),
          pl.BlockSpec((None, nk_pad, head_dim), lambda h, i: (h, 0, 0)),
          pl.BlockSpec((nk_pad, 1), lambda h, i: (0, 0)),
          pl.BlockSpec((None, nk_pad, 1), lambda h, i: (h, 0, 0)),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec((None, bq, head_dim), lambda h, i: (h, i, 0)),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
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
  )(q, k_pooled, v_pooled, log_count, radius, rows, k, v)


def compute_routing_unpadded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    scale: float,
    beta: float,
    block_sizes: sol_routing.SolBlockSizes,
    always_exact: Optional[jax.Array] = None,
    interpret: bool = False,
) -> sol_routing.SolRouting:
  """Routing for unpadded [H, L, d] tensors, pooling with the fused kernel."""
  seq_len = q.shape[1]
  q_pooled, k_pooled, v_pooled, k_radius = sol_pooling.pool_qkv(
      q, k, v, block_q=block_sizes.block_q, block_kv=block_sizes.block_kv, interpret=interpret
  )
  kv_counts = sol_routing.block_counts(seq_len, k_pooled.shape[1] * block_sizes.block_kv, block_sizes.block_kv)
  return sol_routing.route_pooled(
      q_pooled, k_pooled, v_pooled, kv_counts, scale=scale, beta=beta, always_exact=always_exact, k_radius=k_radius
  )


def _needs_padding(seq_len: int, block_sizes: sol_routing.SolBlockSizes) -> bool:
  """Short or unaligned sequences fall back to zero-padding (cheap at that size)."""
  return seq_len % NUM_SUBLANES != 0 or seq_len < sol_pooling.slab_rows(block_sizes.block_q, block_sizes.block_kv)


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
    interpret: Run the Pallas kernels in interpret mode (CPU testing).
    return_density: Also return the exact-branch block density (scalar).

  Returns:
    Output [..., L, d] in q.dtype (and the density if requested).
  """
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError(f"Sol-Attn needs matching self-attention q/k/v, got {q.shape}, {k.shape}, {v.shape}.")
  lead = q.shape[:-2]
  seq_len, head_dim = q.shape[-2:]
  q, k, v = (x.reshape(-1, seq_len, head_dim) for x in (q, k, v))
  if _needs_padding(seq_len, block_sizes):
    padded_len = sol_routing.padded_length(seq_len, block_sizes)
    q, k, v = (sol_routing.pad_sequence(x, padded_len) for x in (q, k, v))
    with jax.named_scope("sol_routing"):
      routing = sol_routing.compute_routing(
          q, k, v, seq_len=seq_len, scale=scale, beta=beta, block_sizes=block_sizes, always_exact=always_exact
      )
  else:
    with jax.named_scope("sol_routing"):
      routing = compute_routing_unpadded(
          q, k, v, scale=scale, beta=beta, block_sizes=block_sizes, always_exact=always_exact, interpret=interpret
      )
  with jax.named_scope("sol_kernel"):
    out = sol_attention_from_routing(
        q,
        k,
        v,
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
