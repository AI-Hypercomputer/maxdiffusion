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

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))


class _BlockSizes:
  __slots__ = ("block_q", "block_kv", "block_kv_compute", "block_kv_compute_in")

  def __init__(
      self,
      block_q: int,
      block_kv: int,
      block_kv_compute: int | None = None,
      block_kv_compute_in: int = 256,
  ):
    self.block_q = block_q
    self.block_kv = block_kv
    self.block_kv_compute = block_kv_compute if block_kv_compute is not None else block_kv
    self.block_kv_compute_in = block_kv_compute_in


FP32_OUTPUT_HEADROOM_BITS = 8.0  # Assumes activation |V| <= 2**FP32_OUTPUT_HEADROOM_BITS = 256.0


def get_fixed_m_constants(kv_seq_len: int, is_ring: bool = False) -> tuple[float, float]:
  """Computes dynamic fixed-m constants C(N) and safe bounds based on KV sequence length.

  Derivation:
    1. Overflow Ceiling: C(N) = 127.0 - ceil(log2(N)) - FP32_OUTPUT_HEADROOM_BITS guarantees that:
       - Denominator accumulator: l = sum_j 2^{z_j - m} <= N * 2^C(N) <= 2^{127 - 8} = 2^{119}
       - Output accumulator: |o_d| = |sum_j V_{j,d} 2^{z_j - m}| <= V_max * N * 2^C(N) <= 2^{127} < 2^{128} (FP32 Max)
       for any activation magnitude |V| <= 256.0 (2^8).
    2. Subnormal Underflow Floor: Requiring the minimal shifted exponent to stay >= -125.0
       (1 bit of safety margin above IEEE-754 normal floor -126.0):
       - Ulysses (Centered, M >= 0): ceil(U) <= C(N) + 125.0 = W(N).
       - Ring (Uncentered, M >= -U): U + ceil(U) <= W(N) => U <= floor(W(N) / 2).
  """
  if kv_seq_len is None or kv_seq_len <= 0:
    raise ValueError(f"kv_seq_len must be a positive integer to compute dynamic fixed-m constants, got {kv_seq_len=}")

  fp32_max_exp = 128.0
  fp32_min_normal_exp = -126.0
  output_headroom_bits = FP32_OUTPUT_HEADROOM_BITS

  max_accumulation_bits = float(math.ceil(math.log2(float(kv_seq_len))))

  # C(N) = 127.0 - max_accumulation_bits - output_headroom_bits
  recenter = fp32_max_exp - max_accumulation_bits - output_headroom_bits - 1.0

  # Safe window W(N) = C(N) - (-126.0) - 1.0 = C(N) + 125.0
  safe_window = recenter - fp32_min_normal_exp - 1.0

  if is_ring:
    # For integer threshold K = floor(W / 2), U <= K guarantees U + ceil(U) <= 2K <= W
    safe_bound = float(int(safe_window // 2))
  else:
    safe_bound = safe_window

  return recenter, safe_bound


def _flash_attention_kernel(
    mk_ref,
    q_ref,
    k_ref,
    v_ref,
    k_mean_ref,
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
    fixed_m_recenter: float | None = None,
    q_heads_per_kv_head: int = 1,
):
  """Pallas Mosaic TPU flash attention kernel with fixed-m support.

  Scalar Prefetch Multiplexing:
    `mk_ref` is a multiplexed scalar prefetch buffer of shape `(2, num_heads, num_q_blocks)`
    passing both the precomputed block fixed-m base shift and discrete predicate in a single scalar memory slot:
      - `mk_ref[0, h, i]`: Precomputed block shift m_B = ceil(max_i ||q_i|| * max_j ||k_j||) - C.
      - `mk_ref[1, h, i]`: Gating eligibility predicate (1.0 for fixed-m, 0.0 for online).
  """
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp
  sv_dims = (((0,), (0,)), ((), ()))

  # `uniform_fixed_m` is the caller's compile-time promise that EVERY head
  # passed the gate (the ring's accumulate merge runs under exactly that
  # predicate). It buys two things at once: the per-head dispatch collapses to
  # a single body per block, and the fixed bound stays PINNED through the
  # ragged last KV block, so every hop reports the identical m and the hops
  # combine by plain accumulation.
  fixed_only = use_fixed_m and uniform_fixed_m
  if uniform_fixed_m and not use_fixed_m:
    raise ValueError("uniform_fixed_m requires use_fixed_m.")

  if use_fixed_m and fixed_m_recenter is None:
    raise ValueError("fixed_m_recenter must be specified when use_fixed_m=True.")

  # Per-(head, Q-block) dispatch: heads / Q-blocks inside the no-flush window run
  # fixed-m, the rest keep online softmax.
  if use_fixed_m and not fixed_only:
    is_fixed = mk_ref[1, h, i] > 0.5
  else:
    is_fixed = False

  def _write_fixed_m():
    # Precomputed block bound m_B = ceil(max_i ||q_i|| * max_j ||k_j||) - C.
    # Virtual K-centering applies the row-specific projection: m_i = m_B + q_i^T \bar{k}.
    m_base = mk_ref[0, h, i]
    if k_mean_ref is not None:
      qf = q_ref[...].astype(float32)
      km = k_mean_ref[h // q_heads_per_kv_head, :].astype(float32)
      mu = (qf * km[None, :]).sum(axis=1)[None, :]
      m_fixed = m_base + mu
    else:
      m_fixed = m_base
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

  if bkv % bkv_compute != 0:
    raise ValueError(f"block_kv ({bkv}) must be divisible by block_kv_compute ({bkv_compute})")

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

  if bkv % bkv_compute != 0:
    raise ValueError(f"block_kv ({bkv}) must be divisible by block_kv_compute ({bkv_compute})")

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
    fixed_m_recenter: float | None = None,
    uniform_fixed_m: bool = False,
    k_mean: jax.Array | None = None,
):
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in
  num_kv_heads = k.shape[0]
  padded_kv_seq_len = k.shape[1]

  actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
  actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA.")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  grid = (num_q_heads, grid_height, grid_width)

  if use_fixed_m and fixed_m_recenter is None:
    raise ValueError("`fixed_m_recenter` must be explicitly specified when `use_fixed_m=True`.")

  # Scalar-prefetch operand carrying per-head / per-Q-block fixed-m data:
  #   mk[0, h, i] = m_B (precomputed block fixed-m base shift), mk[1, h, i] = eligibility.
  # A dummy is supplied for online callers; the kernel ignores it.
  if use_fixed_m and mk is None:
    raise ValueError("`mk` metadata array is required when `use_fixed_m=True`.")
  if mk is None:
    mk = jnp.zeros((2, num_q_heads, grid_height), jnp.float32)
  elif mk.ndim == 2:
    mk = jnp.broadcast_to(mk[:, :, None], (2, num_q_heads, grid_height))

  if mk.shape[0] != 2 or mk.shape[1] != num_q_heads or mk.shape[2] != grid_height:
    raise ValueError(f"mk must have shape (2, {num_q_heads}, {grid_height}), got {mk.shape}")

  if k_mean is None:
    k_mean = jnp.zeros((num_kv_heads, head_dim_qk), dtype=jnp.float32)
  elif k_mean.shape[0] == num_q_heads and num_q_heads != num_kv_heads:
    k_mean = k_mean[::q_heads_per_kv_head]

  if k_mean.shape[0] != num_kv_heads or k_mean.shape[1] != head_dim_qk:
    raise ValueError(f"k_mean must have shape ({num_kv_heads}, {head_dim_qk}), got {k_mean.shape}")

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
      pl.BlockSpec((k_mean.shape[0], head_dim_qk), lambda *_: (0, 0)),
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
          uniform_fixed_m=uniform_fixed_m,
          fixed_m_recenter=fixed_m_recenter,
          q_heads_per_kv_head=q_heads_per_kv_head,
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
  )(mk, q, k, v, k_mean)
  return all_out[-1]


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
    fixed_m_recenter: float | None = None,
    k_mean: jax.Array | None = None,
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
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA.")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if use_fixed_m and fixed_m_recenter is None:
    raise ValueError("`fixed_m_recenter` must be explicitly specified when `use_fixed_m=True`.")

  if use_fixed_m and mk is None:
    raise ValueError("`mk` metadata array is required when `use_fixed_m=True`.")

  if k_mean is None:
    k_mean = jnp.zeros((num_kv_heads, head_dim_qk), dtype=jnp.float32)
  elif k_mean.shape[0] == num_q_heads and num_q_heads != num_kv_heads:
    k_mean = k_mean[::q_heads_per_kv_head]

  if k_mean.shape[0] != num_kv_heads or k_mean.shape[1] != head_dim_qk:
    raise ValueError(f"k_mean must have shape ({num_kv_heads}, {head_dim_qk}), got {k_mean.shape}")

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
      pl.BlockSpec((k_mean.shape[0], head_dim_qk), lambda *_: (0, 0)),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, actual_q_seq_len), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, actual_q_seq_len), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, actual_q_seq_len), jnp.float32),
  ]
  out_specs = [
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
      pl.BlockSpec((None, head_dim_v, bq), out_index_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
      pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
  ]
  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  grid = (num_q_heads, grid_height, grid_width)

  # Scalar-prefetch operand carrying per-head / per-Q-block fixed-m data:
  #   mk[0, h, i] = max_j||k_j|| over ALL ring shards
  # (the caller all-reduces this over the ring axis), mk[1, h, i] = eligibility.
  # A dummy is supplied for online callers; the kernel ignores it.
  if use_fixed_m and mk is None:
    raise ValueError("`mk` metadata array is required when `use_fixed_m=True`.")
  if mk is None:
    mk = jnp.zeros((2, num_q_heads, grid_height), jnp.float32)
  elif mk.ndim == 2:
    mk = jnp.broadcast_to(mk[:, :, None], (2, num_q_heads, grid_height))

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
          fixed_m_recenter=fixed_m_recenter,
          q_heads_per_kv_head=q_heads_per_kv_head,
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
  )(mk, q, k, v, k_mean)
  out = jnp.swapaxes(all_out[3], 1, 2)  # (h, head_dim_v, s) -> (h, s, head_dim_v)
  l = all_out[4][:, 0, :]  # (h, s)
  m = all_out[5][:, 0, :]  # (h, s)
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

  if num_q_heads % hpt != 0:
    raise ValueError(f"num_heads {num_q_heads} must be divisible by heads_per_tile {hpt}")
  if num_q_heads != num_kv_heads:
    raise ValueError(f"MHPT currently requires num_q_heads == num_kv_heads (no GQA), got {num_q_heads=} vs {num_kv_heads=}")

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


def make_splash_mha(
    block_sizes: _BlockSizes,
    orig_q_seq_len: int | None = None,
    orig_kv_seq_len: int | None = None,
    heads_per_tile: int = 1,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    use_fixed_m: bool = False,
    uniform_fixed_m: bool = False,
):
  if use_fixed_m and not use_base2_exp:
    raise NotImplementedError(
        "fixed-m softmax bounds are derived strictly for base-2 exponents. Please set use_base2_exp=True."
    )

  recenter, _ = get_fixed_m_constants(orig_kv_seq_len, is_ring=False)

  def _splash_attention(q, k, v, mk=None, k_mean=None):
    if use_fixed_m and mk is None:
      raise ValueError("`mk` metadata array is required when `use_fixed_m=True`.")
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
        uniform_fixed_m=uniform_fixed_m,
        fixed_m_recenter=recenter,
        k_mean=k_mean,
    )

  return _splash_attention
