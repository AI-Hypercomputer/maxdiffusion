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

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))


LN2 = float(np.log(2.0))


class _BlockSizes:
  __slots__ = (
      "block_q",
      "block_kv",
      "block_kv_compute",
      "block_kv_compute_in",
      "block_q_dkv",
      "block_kv_dkv",
      "block_kv_dkv_compute",
      "block_kv_dkv_compute_in",
      "block_q_dq",
      "block_kv_dq",
      "block_kv_dq_compute",
      "block_kv_dq_compute_in",
      "use_fused_bwd_kernel",
      "dq_reduction_steps",
  )

  def __init__(
      self,
      block_q: int,
      block_kv: int,
      block_kv_compute: int | None = None,
      block_kv_compute_in: int = 256,
      block_q_dkv: int | None = None,
      block_kv_dkv: int | None = None,
      block_kv_dkv_compute: int | None = None,
      block_kv_dkv_compute_in: int | None = None,
      block_q_dq: int | None = None,
      block_kv_dq: int | None = None,
      block_kv_dq_compute: int | None = None,
      block_kv_dq_compute_in: int | None = None,
      use_fused_bwd_kernel: bool = True,
      dq_reduction_steps: int | None = 3,
  ):
    self.block_q = block_q
    self.block_kv = block_kv
    self.block_kv_compute = block_kv_compute if block_kv_compute is not None else block_kv
    self.block_kv_compute_in = block_kv_compute_in
    self.block_q_dkv = block_q_dkv if block_q_dkv is not None else block_q
    self.block_kv_dkv = block_kv_dkv if block_kv_dkv is not None else block_kv
    self.block_kv_dkv_compute = (
        block_kv_dkv_compute if block_kv_dkv_compute is not None else self.block_kv_dkv
    )
    self.block_kv_dkv_compute_in = (
        block_kv_dkv_compute_in if block_kv_dkv_compute_in is not None else block_kv_compute_in
    )
    self.block_q_dq = block_q_dq if block_q_dq is not None else block_q
    self.block_kv_dq = block_kv_dq if block_kv_dq is not None else block_kv
    self.block_kv_dq_compute = (
        block_kv_dq_compute if block_kv_dq_compute is not None else self.block_kv_dq
    )
    self.block_kv_dq_compute_in = (
        block_kv_dq_compute_in if block_kv_dq_compute_in is not None else block_kv_compute_in
    )
    self.use_fused_bwd_kernel = use_fused_bwd_kernel
    self.dq_reduction_steps = dq_reduction_steps

  def _as_tuple(self):
    return (
        self.block_q,
        self.block_kv,
        self.block_kv_compute,
        self.block_kv_compute_in,
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
        self.block_kv_dkv_compute_in,
        self.block_q_dq,
        self.block_kv_dq,
        self.block_kv_dq_compute,
        self.block_kv_dq_compute_in,
        self.use_fused_bwd_kernel,
        self.dq_reduction_steps,
    )

  def __eq__(self, other):
    if not isinstance(other, _BlockSizes):
      return False
    return self._as_tuple() == other._as_tuple()

  def __hash__(self):
    return hash(self._as_tuple())


# Fixed-m softmax-bound constants. Instead of tracking the online-softmax
# running max per KV block, eligible heads subtract a precomputed per-query
# upper bound on the logits (Cauchy-Schwarz: max_j q_i.k_j <= ||q_i|| *
# max_j||k_j||). _FIXED_M_RECENTER (C) shifts the exp2 exponents up so the
# largest surviving term stays above the f32 subnormal-flush floor 2^-126:
# with k-smoothing the per-row max is >= 0, so the max term has exponent
# >= -ceil(bound) + C, which stays > -126 while ceil(bound) <=
# _FIXED_M_SAFE_BOUND (= C + 126 - 1 of margin). Heads whose worst-case bound
# exceeds the gate fall back to online softmax (the "sink" heads).
_FIXED_M_RECENTER = 88.0
_FIXED_M_SAFE_BOUND = 213.0
# Ring-path gate: the ring processes UN-smoothed K shards (no ring rank holds
# the full K to compute a mean, and a per-shard mean would shift each hop's
# logits differently, breaking the cross-shard merge). Without k-smoothing the
# per-row max logit has no >=0 guarantee, so the safe bound halves (calibrated
# for ring_size=2, matching DiffusionServing's ring gate).
_FIXED_M_RING_SAFE_BOUND = _FIXED_M_SAFE_BOUND / 2.0


def _flash_attention_kernel(
    mk_ref,
    q_ref,
    k_ref,
    v_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    l_ring_ref=None,
    m_ring_ref=None,
    *,
    mask_value: float,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
    fuse_reciprocal: bool = True,
    use_fixed_m: bool = False,
    uniform_fixed_m: bool = False,
    save_lse: bool = False,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  h, _, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp
  sv_dims = (((0,), (0,)), ((), ()))

  # `uniform_fixed_m` is the caller's compile-time promise that EVERY head
  # passed the gate (the ring's accumulate merge runs under exactly that
  # predicate). It buys two things at once: the per-head dispatch collapses to
  # a single body per block, and the fixed bound stays PINNED through the
  # ragged last KV block, so every hop reports the identical m and the hops
  # combine by plain accumulation.
  #
  # Both must move together. Pinning without the uniform promise needs a
  # SECOND body in the last block (fixed and online), and a two-body last
  # block degrades the instruction schedule of the WHOLE grid -- measured 3x
  # slower end to end, which is the cliff the design doc's D3 warns about.
  # Keeping this one flag rather than two makes that combination unspellable.
  fixed_only = use_fixed_m and uniform_fixed_m
  if uniform_fixed_m and not use_fixed_m:
    raise ValueError("uniform_fixed_m requires use_fixed_m.")

  # Per-head dispatch: heads inside the no-flush window run fixed-m, the rest
  # keep online softmax. Branch once per head (body level), never per step.
  is_fixed = (mk_ref[1, h] > 0.5) if (use_fixed_m and not fixed_only) else False

  def _write_fixed_m():
    # Per-query Cauchy-Schwarz bound m_i = ceil(||q_i|| * max_j||k_j||) - C.
    qf = q_ref[...].astype(float32)
    qn = jnp.sqrt((qf * qf).sum(axis=1))[None, :]  # (1, bq) per-query norm
    bound = qn * mk_ref[0, h]
    m_fixed = jnp.ceil(bound) - _FIXED_M_RECENTER
    m_scratch_ref[...] = jnp.broadcast_to(m_fixed, m_scratch_ref.shape)

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    if fixed_only:
      _write_fixed_m()
    elif use_fixed_m:

      @pl.when(is_fixed)
      def _init_fixed():
        _write_fixed_m()

      @pl.when(jnp.logical_not(is_fixed))
      def _init_online():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

    else:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

  def _online_inner(qk, v_chunk, m_prev, l_prev, o_prev):
    # Standard online-softmax tiling over the VPU register block.
    step = bkv_compute_in
    for i in range(0, qk.shape[0], step):
      qk_slice = qk[i : i + step]

      m_curr = qk_slice.max(axis=0)[None, :]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = exp(qk_slice - m_next[0:1])
      l_curr = s_curr.sum(axis=0, keepdims=True)

      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev

      o_curr = lax.dot_general(
          v_chunk[i : i + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )
      o_prev = alpha[0:1, ...] * o_prev + o_curr
      m_prev, l_prev = m_next, l_next
    return m_prev, l_prev, o_prev

  def _fixed_inner(qk, v_chunk, m_fix, l_prev, o_prev):
    # Fixed-m fast path: m is constant, so no reduce-max and no alpha rescale.
    step = bkv_compute_in
    for i in range(0, qk.shape[0], step):
      qk_slice = qk[i : i + step]

      s_curr = exp(qk_slice - m_fix[0:1])
      l_curr = s_curr.sum(axis=0, keepdims=True)

      o_curr = lax.dot_general(
          v_chunk[i : i + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )
      o_prev = o_prev + o_curr
      l_prev = l_prev + l_curr
    return l_prev, o_prev

  def compute_body_online(kv_compute_index, _):
    q = q_ref[...]
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    qk = lax.dot_general(k_ref[slice_k, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]
    m_prev, l_prev, o_prev = _online_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
    m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[:] = o_prev

  def compute_body_fixed(kv_compute_index, _):
    q = q_ref[...]
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    qk = lax.dot_general(k_ref[slice_k, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]
    l_prev, o_prev = _fixed_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
    l_scratch_ref[...] = l_prev
    o_scratch_ref[:] = o_prev

  def last_compute_body_online(kv_compute_index):
    q = q_ref[...]
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(kv_compute_index * bkv_compute, slice_k_len)
    qk = lax.dot_general(k_ref[slice_k, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]
    m_prev, l_prev, o_prev = _online_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
    m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[:] = o_prev

  def last_compute_body_fixed(kv_compute_index):
    # Ragged tail for the pinned fixed-m path: exact slice (padded keys are
    # never touched -- with a pinned m their exp2(0 - m_fixed) would be huge
    # garbage, so slicing, not masking, is load-bearing here).
    q = q_ref[...]
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(kv_compute_index * bkv_compute, slice_k_len)
    qk = lax.dot_general(k_ref[slice_k, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_ref[slice_k, :]
    l_prev, o_prev = _fixed_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
    l_scratch_ref[...] = l_prev
    o_scratch_ref[:] = o_prev

  assert bkv % bkv_compute == 0

  if fixed_only:

    @pl.when(j != grid_width - 1)
    def _body_uniform_fixed():
      lax.fori_loop(0, (bkv // bkv_compute), compute_body_fixed, None, unroll=True)

  elif use_fixed_m:

    @pl.when((j != grid_width - 1) & is_fixed)
    def _body_fixed():
      lax.fori_loop(0, (bkv // bkv_compute), compute_body_fixed, None, unroll=True)

    @pl.when((j != grid_width - 1) & jnp.logical_not(is_fixed))
    def _body_online():
      lax.fori_loop(0, (bkv // bkv_compute), compute_body_online, None, unroll=True)

  else:

    @pl.when(j != grid_width - 1)
    def body():
      lax.fori_loop(0, (bkv // bkv_compute), compute_body_online, None, unroll=True)

  # Exactly ONE of these runs in the final KV block -- never both (see the
  # note on `uniform_fixed_m` above).
  #
  # `_last_online` is the hybrid default: a fixed-m head arrives with
  # m_scratch = ceil(bound) - C, and since that is an upper bound on every
  # logit the online step's max leaves it unchanged and its rescale factor is
  # exp2(0) = 1, so running the last block online is exact for fixed heads too.
  # `_last_fixed` keeps the bound pinned instead, which is what lets the ring's
  # accumulate merge assume every hop reports the identical m.
  def _last_online():
    if kv_seq_len % bkv == 0:
      iter_num = bkv // bkv_compute
      lax.fori_loop(0, iter_num, compute_body_online, None, unroll=True)
    else:
      remain_kv_seq_len = kv_seq_len % bkv
      iter_num = (remain_kv_seq_len + bkv_compute - 1) // bkv_compute
      if remain_kv_seq_len % bkv_compute == 0:
        lax.fori_loop(0, iter_num, compute_body_online, None, unroll=True)
      else:
        lax.fori_loop(0, iter_num - 1, compute_body_online, None, unroll=True)
        last_compute_body_online(iter_num - 1)

  def _last_fixed():
    if kv_seq_len % bkv == 0:
      iter_num = bkv // bkv_compute
      lax.fori_loop(0, iter_num, compute_body_fixed, None, unroll=True)
    else:
      remain_kv_seq_len = kv_seq_len % bkv
      iter_num = (remain_kv_seq_len + bkv_compute - 1) // bkv_compute
      if remain_kv_seq_len % bkv_compute == 0:
        lax.fori_loop(0, iter_num, compute_body_fixed, None, unroll=True)
      else:
        lax.fori_loop(0, iter_num - 1, compute_body_fixed, None, unroll=True)
        last_compute_body_fixed(iter_num - 1)

  if fixed_only:
    # ONE body in the last block (the whole point of uniform mode).
    @pl.when(j == grid_width - 1)
    def last_body_uniform_fixed():
      _last_fixed()

  else:

    @pl.when(j == grid_width - 1)
    def last_body():
      _last_online()

  @pl.when(j == grid_width - 1)
  def end():
    l = l_scratch_ref[...]
    if fuse_reciprocal:
      l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      # Ring path: emit the un-normalized numerator plus the running softmax
      # stats (max logit `m` and linear denominator `l`) so the outer ring loop
      # can merge shard contributions and normalize only once at the very end.
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if l_ring_ref is not None:
      if save_lse:
        log = jnp.log2 if use_base2_exp else jnp.log
        l_ring_ref[...] = (m_scratch_ref[...] + log(l)).astype(l_ring_ref.dtype)
      else:
        l_ring_ref[...] = l.astype(l_ring_ref.dtype)
    if m_ring_ref is not None:
      m_ring_ref[...] = m_scratch_ref[...].astype(m_ring_ref.dtype)


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
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    kv_seq_len: int,
    heads_per_tile: int,
    use_base2_exp: bool = True,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  _, _, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

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
        s_curr = exp(qk_slice - m_next[0:1])
        l_curr = s_curr.sum(axis=0, keepdims=True)

        alpha = exp(m_prev - m_next)
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
        s_curr = exp(qk_slice - m_next[0:1])
        l_curr = s_curr.sum(axis=0, keepdims=True)

        alpha = exp(m_prev - m_next)
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
      l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
      o_ref[h_local] = (o_scratch_ref[h_local] * l_inv).astype(o_ref.dtype)


def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    use_fixed_m: bool = False,
    mk: jax.Array | None = None,
    save_residuals: bool = False,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  # Scalar-prefetch operand carrying per-head fixed-m data:
  #   mk[0, h] = max_j||k_j|| (Cauchy-Schwarz factor), mk[1, h] = eligibility.
  # A dummy is supplied for online callers; the kernel ignores it.
  if mk is None:
    mk = jnp.zeros((2, num_q_heads), jnp.float32)
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in
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

  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  active_q_len = grid_height * bq
  active_kv_len = grid_width * bkv

  q_in = jnp.pad(q, ((0, 0), (0, active_q_len - q.shape[1]), (0, 0))) if q.shape[1] < active_q_len else q
  k_in = jnp.pad(k, ((0, 0), (0, active_kv_len - k.shape[1]), (0, 0))) if k.shape[1] < active_kv_len else k
  v_in = jnp.pad(v, ((0, 0), (0, active_kv_len - v.shape[1]), (0, 0))) if v.shape[1] < active_kv_len else v

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
      pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, active_q_len), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
      pl.BlockSpec((None, head_dim_v, bq), out_index_map),
  ]
  if save_residuals:
    out_shapes.append(jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, active_q_len), jnp.float32))
    out_specs.append(pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map))

  grid = (num_q_heads, grid_height, grid_width)

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
          use_fixed_m=use_fixed_m,
          save_lse=save_residuals,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=out_shapes,
  )(mk, q_in, k_in, v_in)
  out = all_out[3][:, :, :actual_q_seq_len]
  if save_residuals:
    return out, all_out[4][:, :, :actual_q_seq_len]
  return out


def _splash_attention_forward_ring(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    use_fixed_m: bool = False,
    mk: jax.Array | None = None,
    uniform_fixed_m: bool = False,
):
  """Ring-specific forward path that returns pre-reciprocal fp32 accumulators.

  Mirrors `_splash_attention_forward`, but instead of normalizing the output by
  the softmax denominator inside the kernel, it returns the un-normalized
  numerator (`out`) together with the per-row max logit (`m`) and linear softmax
  denominator (`l`). The outer ring loop merges these shard contributions and
  normalizes only once at the very end (see
  `ring_attention_kernel._custom_ring_attention_forward`).

  Returns:
    A tuple `(out, m, l)` where
      - `out` has shape `(num_q_heads, q_seq_len, head_dim_v)` (fp32, un-normalized),
      - `m` and `l` have shape `(num_q_heads, q_seq_len)` (fp32).
  """
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in
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

  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  active_q_len = grid_height * bq
  active_kv_len = grid_width * bkv

  q_in = jnp.pad(q, ((0, 0), (0, active_q_len - q.shape[1]), (0, 0))) if q.shape[1] < active_q_len else q
  k_in = jnp.pad(k, ((0, 0), (0, active_kv_len - k.shape[1]), (0, 0))) if k.shape[1] < active_kv_len else k
  v_in = jnp.pad(v, ((0, 0), (0, active_kv_len - v.shape[1]), (0, 0))) if v.shape[1] < active_kv_len else v

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
      pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, active_q_len), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, active_q_len), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, active_q_len), jnp.float32),
  ]
  out_specs = [
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
      pl.BlockSpec((None, head_dim_v, bq), out_index_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
  ]
  grid = (num_q_heads, grid_height, grid_width)

  # Scalar-prefetch operand carrying per-head fixed-m data (same convention as
  # `_splash_attention_forward`): mk[0, h] = max_j||k_j|| over ALL ring shards
  # (the caller all-reduces this over the ring axis), mk[1, h] = eligibility.
  # A dummy is supplied for online callers; the kernel ignores it.
  if mk is None:
    mk = jnp.zeros((2, num_q_heads), jnp.float32)

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
          fuse_reciprocal=False,
          use_fixed_m=use_fixed_m,
          uniform_fixed_m=uniform_fixed_m,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=out_shapes,
  )(mk, q_in, k_in, v_in)
  out = jnp.swapaxes(all_out[3][:, :, :actual_q_seq_len], 1, 2)  # (h, head_dim_v, s) -> (h, s, head_dim_v)
  l = all_out[4][:, 0, :actual_q_seq_len]  # (h, s)
  m = all_out[5][:, 0, :actual_q_seq_len]  # (h, s)
  return out, m, l


def _splash_attention_forward_mhpt(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    heads_per_tile: int,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in
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
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
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
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=out_shapes,
  )(q, k, v)
  return all_out[-1]


def _flash_attention_dq_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    lse_ref,
    di_ref,
    dq_scratch_ref,
    dq_ref,
    *,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
):
  float32 = jnp.float32
  _, _, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  @pl.when(j == 0)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def _dq_inner(qk, k_chunk, v_chunk, do, lse, di, dq_prev):
    step = bkv_compute_in
    for idx in range(0, qk.shape[0], step):
      qk_slice = qk[idx : idx + step]
      v_slice = v_chunk[idx : idx + step]
      k_slice = k_chunk[idx : idx + step]

      p_curr = exp(qk_slice - lse)
      dp_curr = lax.dot_general(
          v_slice,
          do.astype(v_slice.dtype),
          (((1,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      ds_curr = p_curr * (dp_curr - di)
      dq_curr = lax.dot_general(
          ds_curr.astype(k_slice.dtype),
          k_slice,
          (((0,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      dq_prev = dq_prev + dq_curr
    return dq_prev

  def compute_body(kv_compute_index, _):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    dq_scratch_ref[...] = _dq_inner(qk, k_chunk, v_chunk, do, lse, di, dq_scratch_ref[...])

  def last_compute_body(kv_compute_index):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(kv_compute_index * bkv_compute, slice_k_len)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    dq_scratch_ref[...] = _dq_inner(qk, k_chunk, v_chunk, do, lse, di, dq_scratch_ref[...])

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
    if use_base2_exp:
      dq_ref[...] = (dq_scratch_ref[...] * LN2).astype(dq_ref.dtype)
    else:
      dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attention_dkv_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    lse_ref,
    di_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dk_ref,
    dv_ref,
    *,
    grid_width: int,
    total_q_steps: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
):
  float32 = jnp.float32
  _, j, step_q = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  @pl.when(step_q == 0)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def _dkv_inner(base_offset, qk, q, v_chunk, do, lse, di):
    step = bkv_compute_in
    for idx in range(0, qk.shape[0], step):
      sub_len = min(step, qk.shape[0] - idx)
      sub_slice = pl.ds(base_offset + idx, sub_len)
      qk_slice = qk[idx : idx + sub_len]
      v_slice = v_chunk[idx : idx + sub_len]

      p_curr = exp(qk_slice - lse)
      dv_curr = lax.dot_general(
          p_curr.astype(do.dtype),
          do,
          NT_DIM_NUMBERS,
          preferred_element_type=float32,
      )
      dv_scratch_ref[sub_slice, :] = dv_scratch_ref[sub_slice, :] + dv_curr

      dp_curr = lax.dot_general(
          v_slice,
          do.astype(v_slice.dtype),
          (((1,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      ds_curr = p_curr * (dp_curr - di)
      dk_curr = lax.dot_general(
          ds_curr.astype(q.dtype),
          q,
          (((1,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      dk_scratch_ref[sub_slice, :] = dk_scratch_ref[sub_slice, :] + dk_curr

  def compute_body(kv_compute_index, _):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    _dkv_inner(base_offset, qk, q, v_chunk, do, lse, di)

  def last_compute_body(kv_compute_index):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    base_offset = kv_compute_index * bkv_compute
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(base_offset, slice_k_len)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    _dkv_inner(base_offset, qk, q, v_chunk, do, lse, di)

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

  @pl.when(step_q == total_q_steps - 1)
  def end():
    if use_base2_exp:
      dk_ref[...] = (dk_scratch_ref[...] * LN2).astype(dk_ref.dtype)
    else:
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
    dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)


def _flash_attention_bwd_fused_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    lse_ref,
    di_ref,
    dq_alias_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    grid_width: int,
    grid_height: int,
    q_heads_per_kv_head: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
    use_dq_aliasing: bool = False,
):
  """Fused backward kernel iterating over KV outer (grid_width) and Q inner (num_q_heads, grid_height).

  Computes dQ, dK, and dV in a single pass without masks, ignoring KV tail padding via
  slice bounds and accumulating dK/dV in VMEM across Q steps and KV head groups.
  """
  float32 = jnp.float32
  j, h_q, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  q_head_in_group = lax.rem(h_q, q_heads_per_kv_head)
  should_init_dkv = jnp.logical_and(i == 0, q_head_in_group == 0)
  should_write_dkv = jnp.logical_and(
      i == grid_height - 1, q_head_in_group == q_heads_per_kv_head - 1
  )

  @pl.when(should_init_dkv)
  def init_dkv():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def _bwd_fused_inner(base_offset, qk, k_chunk, v_chunk, q, do, lse, di):
    step = bkv_compute_in
    dq_acc = dq_scratch_ref[...]
    for idx in range(0, qk.shape[0], step):
      sub_len = min(step, qk.shape[0] - idx)
      sub_slice = pl.ds(base_offset + idx, sub_len)
      qk_slice = qk[idx : idx + sub_len]
      k_slice = k_chunk[idx : idx + sub_len]
      v_slice = v_chunk[idx : idx + sub_len]

      p_curr = exp(qk_slice - lse)
      dv_curr = lax.dot_general(
          p_curr.astype(do.dtype),
          do,
          NT_DIM_NUMBERS,
          preferred_element_type=float32,
      )
      dv_scratch_ref[sub_slice, :] = dv_scratch_ref[sub_slice, :] + dv_curr

      dp_curr = lax.dot_general(
          v_slice,
          do.astype(v_slice.dtype),
          (((1,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      ds_curr = p_curr * (dp_curr - di)
      dk_curr = lax.dot_general(
          ds_curr.astype(q.dtype),
          q,
          (((1,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      dk_scratch_ref[sub_slice, :] = dk_scratch_ref[sub_slice, :] + dk_curr

      dq_curr = lax.dot_general(
          ds_curr.astype(k_slice.dtype),
          k_slice,
          (((0,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      dq_acc = dq_acc + dq_curr
    dq_scratch_ref[...] = dq_acc

  def compute_body(kv_compute_index, _):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    base_offset = kv_compute_index * bkv_compute
    slice_k = pl.ds(base_offset, bkv_compute)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    _bwd_fused_inner(base_offset, qk, k_chunk, v_chunk, q, do, lse, di)

  def last_compute_body(kv_compute_index):
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[0:1, :]
    di = di_ref[0:1, :]
    base_offset = kv_compute_index * bkv_compute
    slice_k_len = kv_seq_len % bkv_compute
    slice_k = pl.ds(base_offset, slice_k_len)
    k_chunk = k_ref[slice_k, :]
    v_chunk = v_ref[slice_k, :]
    qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
    _bwd_fused_inner(base_offset, qk, k_chunk, v_chunk, q, do, lse, di)

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

  if use_base2_exp:
    dq_val = (dq_scratch_ref[...] * LN2).astype(dq_ref.dtype)
  else:
    dq_val = dq_scratch_ref[...].astype(dq_ref.dtype)

  if use_dq_aliasing:

    @pl.when(j < 3)
    def write_dq_first():
      dq_ref[...] = dq_val

    @pl.when(j >= 3)
    def write_dq_acc():
      dq_ref[...] = dq_alias_ref[...] + dq_val

  else:
    dq_ref[...] = dq_val

  @pl.when(should_write_dkv)
  def end_dkv():
    if use_base2_exp:
      dk_ref[...] = (dk_scratch_ref[...] * LN2).astype(dk_ref.dtype)
    else:
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
    dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)


def _splash_attention_bwd_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    lse: jax.Array,
    di: jax.Array,
    block_sizes: _BlockSizes,
    actual_q_seq_len: int,
    actual_kv_seq_len: int,
    padded_q_seq_len: int,
    padded_kv_seq_len: int,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
):
  num_q_heads, _, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  num_kv_heads = k.shape[0]
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  bq = block_sizes.block_q_dkv
  bkv = block_sizes.block_kv_dkv
  bkv_compute = block_sizes.block_kv_dkv_compute
  bkv_compute_in = block_sizes.block_kv_dkv_compute_in

  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  active_q_len = grid_height * bq
  active_kv_len = grid_width * bkv

  if actual_q_seq_len < active_q_len:
    pad_q = active_q_len - actual_q_seq_len
    q_bwd = jnp.pad(q[:, :actual_q_seq_len, :], ((0, 0), (0, pad_q), (0, 0)))
    do_bwd = jnp.pad(do[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    lse_bwd = jnp.pad(lse[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    di_bwd = jnp.pad(di[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
  elif q.shape[1] > active_q_len:
    q_bwd = q[:, :active_q_len, :]
    do_bwd = do[:, :, :active_q_len]
    lse_bwd = lse[:, :, :active_q_len]
    di_bwd = di[:, :, :active_q_len]
  else:
    q_bwd, do_bwd, lse_bwd, di_bwd = q, do, lse, di

  k_bwd = jnp.pad(k, ((0, 0), (0, active_kv_len - k.shape[1]), (0, 0))) if k.shape[1] < active_kv_len else k
  v_bwd = jnp.pad(v, ((0, 0), (0, active_kv_len - v.shape[1]), (0, 0))) if v.shape[1] < active_kv_len else v

  use_dq_aliasing = (
      block_sizes.dq_reduction_steps == 3 and grid_width > 3
  )

  if use_dq_aliasing:
    dq_index_map = lambda j, h_q, i, *_: (j % 3, h_q, i, 0)
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_dtype = jnp.float32
    dq_shape = jax.ShapeDtypeStruct((3, num_q_heads, active_q_len, head_dim_qk), dq_dtype)
    dq_init = lax.empty((3, num_q_heads, active_q_len, head_dim_qk), dtype=dq_dtype)
  else:
    dq_index_map = lambda j, h_q, i, *_: (j, h_q, i, 0)
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = None
    dq_dtype = q.dtype if grid_width == 1 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((grid_width, num_q_heads, active_q_len, head_dim_qk), dq_dtype)
    dq_init = None

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), lambda j, h_q, i, *_: (h_q, i, 0)),
      pl.BlockSpec((None, bkv, head_dim_qk), lambda j, h_q, i, *_: (h_q // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((None, bkv, head_dim_v), lambda j, h_q, i, *_: (h_q // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((None, head_dim_v, bq), lambda j, h_q, i, *_: (h_q, 0, i)),
      pl.BlockSpec((None, NUM_SUBLANES, bq), lambda j, h_q, i, *_: (h_q, 0, i)),
      pl.BlockSpec((None, NUM_SUBLANES, bq), lambda j, h_q, i, *_: (h_q, 0, i)),
      dq_alias_spec,
  ]

  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct((num_kv_heads, active_kv_len, head_dim_qk), k.dtype),
      jax.ShapeDtypeStruct((num_kv_heads, active_kv_len, head_dim_v), v.dtype),
      jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32),
      jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),
      jax.ShapeDtypeStruct((bkv, head_dim_v), jnp.float32),
  ]
  out_specs = [
      dq_spec,
      pl.BlockSpec((None, bkv, head_dim_qk), lambda j, h_q, i, *_: (h_q // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((None, bkv, head_dim_v), lambda j, h_q, i, *_: (h_q // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0)),
      pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),
      pl.BlockSpec((bkv, head_dim_v), lambda *_: (0, 0)),
  ]
  grid = (grid_width, num_q_heads, grid_height)
  input_output_aliases = {6: 0} if use_dq_aliasing else {}

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_bwd_fused_kernel,
          grid_width=grid_width,
          grid_height=grid_height,
          q_heads_per_kv_head=q_heads_per_kv_head,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
          use_dq_aliasing=use_dq_aliasing,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=out_shapes,
      input_output_aliases=input_output_aliases,
  )(q_bwd, k_bwd, v_bwd, do_bwd, lse_bwd, di_bwd, dq_init)

  dq_unreduced, dk, dv = all_out[0], all_out[1], all_out[2]
  if grid_width == 1:
    dq = dq_unreduced[0].astype(q.dtype)
  else:
    dq = dq_unreduced.sum(axis=0).astype(q.dtype)

  if active_q_len > padded_q_seq_len:
    dq = dq[:, :padded_q_seq_len, :]
  elif active_q_len < padded_q_seq_len:
    dq = jnp.pad(dq, ((0, 0), (0, padded_q_seq_len - active_q_len), (0, 0)))

  if active_kv_len > padded_kv_seq_len:
    dk = dk[:, :padded_kv_seq_len, :]
    dv = dv[:, :padded_kv_seq_len, :]
  elif active_kv_len < padded_kv_seq_len:
    pad_kv = padded_kv_seq_len - active_kv_len
    dk = jnp.pad(dk, ((0, 0), (0, pad_kv), (0, 0)))
    dv = jnp.pad(dv, ((0, 0), (0, pad_kv), (0, 0)))

  return dq, dk, dv


def _splash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    block_sizes: _BlockSizes,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    di: jax.Array | None = None,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  num_kv_heads = k.shape[0]
  padded_kv_seq_len = k.shape[1]

  actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
  actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if di is None:
    di_vec = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=1)
    di = jnp.broadcast_to(di_vec[:, None, :], (num_q_heads, NUM_SUBLANES, o.shape[2]))

  if block_sizes.use_fused_bwd_kernel:
    return _splash_attention_bwd_fused(
        q=q,
        k=k,
        v=v,
        do=do,
        lse=lse,
        di=di,
        block_sizes=block_sizes,
        actual_q_seq_len=actual_q_seq_len,
        actual_kv_seq_len=actual_kv_seq_len,
        padded_q_seq_len=padded_q_seq_len,
        padded_kv_seq_len=padded_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
    )

  # --- 1. Compute dQ ---
  bq_dq, bkv_dq = block_sizes.block_q_dq, block_sizes.block_kv_dq
  bkv_dq_compute = block_sizes.block_kv_dq_compute
  bkv_dq_compute_in = block_sizes.block_kv_dq_compute_in
  grid_width_dq = (actual_kv_seq_len + bkv_dq - 1) // bkv_dq
  grid_height_dq = (actual_q_seq_len + bq_dq - 1) // bq_dq
  active_q_len_dq = grid_height_dq * bq_dq
  active_kv_len_dq = grid_width_dq * bkv_dq

  if actual_q_seq_len < active_q_len_dq:
    pad_q = active_q_len_dq - actual_q_seq_len
    q_dq = jnp.pad(q[:, :actual_q_seq_len, :], ((0, 0), (0, pad_q), (0, 0)))
    do_dq = jnp.pad(do[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    lse_dq = jnp.pad(lse[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    di_dq = jnp.pad(di[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
  elif q.shape[1] > active_q_len_dq:
    q_dq = q[:, :active_q_len_dq, :]
    do_dq = do[:, :, :active_q_len_dq]
    lse_dq = lse[:, :, :active_q_len_dq]
    di_dq = di[:, :, :active_q_len_dq]
  else:
    q_dq, do_dq, lse_dq, di_dq = q, do, lse, di

  k_dq = jnp.pad(k, ((0, 0), (0, active_kv_len_dq - k.shape[1]), (0, 0))) if k.shape[1] < active_kv_len_dq else k
  v_dq = jnp.pad(v, ((0, 0), (0, active_kv_len_dq - v.shape[1]), (0, 0))) if v.shape[1] < active_kv_len_dq else v

  dq_in_specs = [
      pl.BlockSpec((None, bq_dq, head_dim_qk), lambda h, i, j, *_: (h, i, 0)),
      pl.BlockSpec((None, bkv_dq, head_dim_qk), lambda h, i, j, *_: (h // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((None, bkv_dq, head_dim_v), lambda h, i, j, *_: (h // q_heads_per_kv_head, j, 0)),
      pl.BlockSpec((None, head_dim_v, bq_dq), lambda h, i, j, *_: (h, 0, i)),
      pl.BlockSpec((None, NUM_SUBLANES, bq_dq), lambda h, i, j, *_: (h, 0, i)),
      pl.BlockSpec((None, NUM_SUBLANES, bq_dq), lambda h, i, j, *_: (h, 0, i)),
  ]
  dq_out_shapes = [
      jax.ShapeDtypeStruct((bq_dq, head_dim_qk), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, active_q_len_dq, head_dim_qk), q.dtype),
  ]
  dq_out_specs = [
      pl.BlockSpec((bq_dq, head_dim_qk), lambda *_: (0, 0)),
      pl.BlockSpec((None, bq_dq, head_dim_qk), lambda h, i, j, *_: (h, i, 0)),
  ]
  dq_grid = (num_q_heads, grid_height_dq, grid_width_dq)

  _, dq = pl.pallas_call(
      functools.partial(
          _flash_attention_dq_kernel,
          grid_width=grid_width_dq,
          bkv=bkv_dq,
          bkv_compute=bkv_dq_compute,
          bkv_compute_in=bkv_dq_compute_in,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=dq_in_specs,
          out_specs=dq_out_specs,
          grid=dq_grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=dq_out_shapes,
  )(q_dq, k_dq, v_dq, do_dq, lse_dq, di_dq)

  if active_q_len_dq > padded_q_seq_len:
    dq = dq[:, :padded_q_seq_len, :]
  elif active_q_len_dq < padded_q_seq_len:
    dq = jnp.pad(dq, ((0, 0), (0, padded_q_seq_len - active_q_len_dq), (0, 0)))

  # --- 2. Compute dK, dV ---
  bq_dkv, bkv_dkv = block_sizes.block_q_dkv, block_sizes.block_kv_dkv
  bkv_dkv_compute = block_sizes.block_kv_dkv_compute
  bkv_dkv_compute_in = block_sizes.block_kv_dkv_compute_in
  grid_width_dkv = (actual_kv_seq_len + bkv_dkv - 1) // bkv_dkv
  grid_height_dkv = (actual_q_seq_len + bq_dkv - 1) // bq_dkv
  active_q_len_dkv = grid_height_dkv * bq_dkv
  active_kv_len_dkv = grid_width_dkv * bkv_dkv
  total_q_steps = q_heads_per_kv_head * grid_height_dkv

  if actual_q_seq_len < active_q_len_dkv:
    pad_q = active_q_len_dkv - actual_q_seq_len
    q_dkv = jnp.pad(q[:, :actual_q_seq_len, :], ((0, 0), (0, pad_q), (0, 0)))
    do_dkv = jnp.pad(do[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    lse_dkv = jnp.pad(lse[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
    di_dkv = jnp.pad(di[:, :, :actual_q_seq_len], ((0, 0), (0, 0), (0, pad_q)))
  elif q.shape[1] > active_q_len_dkv:
    q_dkv = q[:, :active_q_len_dkv, :]
    do_dkv = do[:, :, :active_q_len_dkv]
    lse_dkv = lse[:, :, :active_q_len_dkv]
    di_dkv = di[:, :, :active_q_len_dkv]
  else:
    q_dkv, do_dkv, lse_dkv, di_dkv = q, do, lse, di

  k_dkv = jnp.pad(k, ((0, 0), (0, active_kv_len_dkv - k.shape[1]), (0, 0))) if k.shape[1] < active_kv_len_dkv else k
  v_dkv = jnp.pad(v, ((0, 0), (0, active_kv_len_dkv - v.shape[1]), (0, 0))) if v.shape[1] < active_kv_len_dkv else v

  def q_step_map(h_kv, j, step_q, *_):
    h_q = h_kv * q_heads_per_kv_head + (step_q // grid_height_dkv)
    i = step_q % grid_height_dkv
    return (h_q, i, 0)

  def do_step_map(h_kv, j, step_q, *_):
    h_q = h_kv * q_heads_per_kv_head + (step_q // grid_height_dkv)
    i = step_q % grid_height_dkv
    return (h_q, 0, i)

  dkv_in_specs = [
      pl.BlockSpec((None, bq_dkv, head_dim_qk), q_step_map),
      pl.BlockSpec((None, bkv_dkv, head_dim_qk), lambda h_kv, j, step_q, *_: (h_kv, j, 0)),
      pl.BlockSpec((None, bkv_dkv, head_dim_v), lambda h_kv, j, step_q, *_: (h_kv, j, 0)),
      pl.BlockSpec((None, head_dim_v, bq_dkv), do_step_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq_dkv), do_step_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq_dkv), do_step_map),
  ]
  dkv_out_shapes = [
      jax.ShapeDtypeStruct((bkv_dkv, head_dim_qk), jnp.float32),
      jax.ShapeDtypeStruct((bkv_dkv, head_dim_v), jnp.float32),
      jax.ShapeDtypeStruct((num_kv_heads, active_kv_len_dkv, head_dim_qk), k.dtype),
      jax.ShapeDtypeStruct((num_kv_heads, active_kv_len_dkv, head_dim_v), v.dtype),
  ]
  dkv_out_specs = [
      pl.BlockSpec((bkv_dkv, head_dim_qk), lambda *_: (0, 0)),
      pl.BlockSpec((bkv_dkv, head_dim_v), lambda *_: (0, 0)),
      pl.BlockSpec((None, bkv_dkv, head_dim_qk), lambda h_kv, j, step_q, *_: (h_kv, j, 0)),
      pl.BlockSpec((None, bkv_dkv, head_dim_v), lambda h_kv, j, step_q, *_: (h_kv, j, 0)),
  ]
  dkv_grid = (num_kv_heads, grid_width_dkv, total_q_steps)

  _, _, dk, dv = pl.pallas_call(
      functools.partial(
          _flash_attention_dkv_kernel,
          grid_width=grid_width_dkv,
          total_q_steps=total_q_steps,
          bkv=bkv_dkv,
          bkv_compute=bkv_dkv_compute,
          bkv_compute_in=bkv_dkv_compute_in,
          kv_seq_len=actual_kv_seq_len,
          use_base2_exp=use_base2_exp,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=dkv_in_specs,
          out_specs=dkv_out_specs,
          grid=dkv_grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=dkv_out_shapes,
  )(q_dkv, k_dkv, v_dkv, do_dkv, lse_dkv, di_dkv)

  if active_kv_len_dkv > padded_kv_seq_len:
    dk = dk[:, :padded_kv_seq_len, :]
    dv = dv[:, :padded_kv_seq_len, :]
  elif active_kv_len_dkv < padded_kv_seq_len:
    pad_kv = padded_kv_seq_len - active_kv_len_dkv
    dk = jnp.pad(dk, ((0, 0), (0, pad_kv), (0, 0)))
    dv = jnp.pad(dv, ((0, 0), (0, pad_kv), (0, 0)))

  return dq, dk, dv


@functools.partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "block_sizes",
        "q_seq_len",
        "kv_seq_len",
        "use_base2_exp",
        "use_experimental_scheduler",
        "vmem_limit_bytes",
    ),
)
def _splash_attention_custom(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
):
  return _splash_attention_forward(
      q,
      k,
      v,
      block_sizes,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      vmem_limit_bytes=vmem_limit_bytes,
      save_residuals=False,
  )


def _splash_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
):
  out, lse = _splash_attention_forward(
      q,
      k,
      v,
      block_sizes,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      vmem_limit_bytes=vmem_limit_bytes,
      save_residuals=True,
  )
  return out, (q, k, v, out, lse)


def _splash_attention_bwd(
    block_sizes: _BlockSizes,
    q_seq_len: int | None,
    kv_seq_len: int | None,
    use_base2_exp: bool,
    use_experimental_scheduler: bool,
    vmem_limit_bytes: int | None,
    residuals,
    do: jax.Array,
):
  q, k, v, out, lse = residuals
  dq, dk, dv = _splash_attention_backward(
      q,
      k,
      v,
      out,
      lse,
      do,
      block_sizes,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  return dq, dk, dv


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


def make_splash_mha(
    block_sizes: _BlockSizes,
    orig_q_seq_len: int | None = None,
    orig_kv_seq_len: int | None = None,
    heads_per_tile: int = 1,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    use_fixed_m: bool = False,
):
  def _splash_attention(q, k, v, mk=None):
    if heads_per_tile > 1:
      if use_fixed_m:
        raise NotImplementedError("fixed-m is not supported with heads_per_tile > 1")
      return _splash_attention_forward_mhpt(
          q,
          k,
          v,
          block_sizes,
          heads_per_tile,
          q_seq_len=orig_q_seq_len,
          kv_seq_len=orig_kv_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
      )
    if use_fixed_m or mk is not None:
      return _splash_attention_forward(
          q,
          k,
          v,
          block_sizes,
          q_seq_len=orig_q_seq_len,
          kv_seq_len=orig_kv_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          use_fixed_m=use_fixed_m,
          mk=mk,
      )
    return _splash_attention_custom(
        q,
        k,
        v,
        block_sizes=block_sizes,
        q_seq_len=orig_q_seq_len,
        kv_seq_len=orig_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
    )

  return _splash_attention
