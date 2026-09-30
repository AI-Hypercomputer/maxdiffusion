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

"""Ring attention with the KV permutation INSIDE the Pallas kernel (contrast with
`kernels/splash_attention/ring_attention_kernel.py`, which does the ring hop as
an XLA `lax.ppermute` + a per-hop online-softmax merge -- see the `(2c)` comment
at its `attention_flax.py` call site for that comparison). Here the hop is a
`pltpu.make_async_remote_copy` issued from inside the kernel body, so a single
pallas_call sees every ring shard. Grid, per the PCP design note -- ring
INNERMOST, under the kv block loop:

    grid = (b, h, i, j, r)
      for each q block i:
        for each kv block j:
          for each hop r:      <-- one bkv-sized KV block arrives over ICI
            accumulate online softmax in VMEM

`r` innermost keeps the online-softmax accumulator `(m, l, o)` in VMEM for the
whole q block (no cross-hop merge, one HBM write) and keeps only ONE kv block
per tensor in flight (`2 x bkv x head_dim`, not `2 x kv_seq x head_dim` -- a
whole-shard buffer would need ~40 MB of the 64 MB VMEM budget at 2D-ring shapes
and OOM).

The price is ICI volume. `r` under `i` means the whole KV shard is rotated once
per q block, so the wire traffic is `num_q_blocks x (R-1) x |KV shard|` against
the external design's `(R-1) x |KV shard|`. That trade is why this design wins
for LLM inference -- GQA shrinks the KV shard by the group ratio G, so rotating
it `num_q_blocks` times is cheap -- and is a much closer call for diffusion
self-attention, where G = 1 and KV is exactly as large as Q.

Flow control (one-credit protocol, full detail at `_release()` below): a rank
signals its upstream neighbour after finishing a slot; a push waits for one
credit, so a sender can run at most one hop ahead of its receiver.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion.kernels import custom_splash_attention as custom_splash

DEFAULT_MASK_VALUE = custom_splash.DEFAULT_MASK_VALUE
NUM_SUBLANES = custom_splash.NUM_SUBLANES
NT_DIM_NUMBERS = custom_splash.NT_DIM_NUMBERS

# collective_id namespace for this kernel's barrier semaphore. Must not collide
# with another Pallas collective live in the same program; nothing else in
# MaxDiffusion uses in-kernel collectives today.
_COLLECTIVE_ID = 7


def _neighbor(axis_names, ring_axis, offset):
  """Mesh-index tuple of the rank `offset` steps along `ring_axis`."""
  ring_size = lax.axis_size(ring_axis)
  idx = lax.axis_index(ring_axis)
  nxt = lax.rem(idx + offset + ring_size, ring_size)
  return tuple(nxt if a == ring_axis else lax.axis_index(a) for a in axis_names)


def _internal_ring_kernel(
    # scalar prefetch
    mk_ref,  # (2, num_q_heads) f32: [0,h]=global max_j||k_j|| over the WHOLE ring,
    #                                 [1,h]=eligibility (unused when uniform)
    # inputs
    q_ref,  # VMEM (bq, head_dim_qk)      -- BlockSpec pipelined
    k_hbm,  # ANY  (batch * num_kv_heads, kv_pad, head_dim_qk)
    v_hbm,  # ANY  (batch * num_kv_heads, kv_pad, head_dim_v)
    # outputs
    o_ref,  # VMEM (head_dim_v, bq)
    # scratch
    k_buf,  # VMEM (2, bkv, head_dim_qk)
    v_buf,  # VMEM (2, bkv, head_dim_v)
    m_scratch_ref,  # VMEM (NUM_SUBLANES, bq) f32
    l_scratch_ref,  # VMEM (NUM_SUBLANES, bq) f32
    o_scratch_ref,  # VMEM (head_dim_v, bq)  f32
    local_sem,  # DMA (2,)   [k, v]
    send_sem,  # DMA (2,)   [k, v]
    recv_sem,  # DMA (2,)   [k, v]
    credit_sem,  # REGULAR
    *,
    mask_value: float,
    grid_width: int,
    ring_size: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    kv_seq_len: int,
    q_heads_per_kv_head: int,
    num_kv_heads: int,
    use_base2_exp: bool,
    use_fixed_m: bool,
    axis_names: tuple[str, ...],
    ring_axis: str,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  b, h, i, j, r = (pl.program_id(n) for n in range(5))
  exp = jnp.exp2 if use_base2_exp else jnp.exp
  sv_dims = (((0,), (0,)), ((), ()))

  # The KV HBM refs arrive flattened to (batch * num_kv_heads, kv_pad, d) so one
  # `pl.ds` picks a shard: `.at[]` on an ANY-space ref does not squeeze, so the
  # source and destination ranks have to line up by construction.
  hk = b * num_kv_heads + h // q_heads_per_kv_head

  # Hop counter within one (b, h, i). Slot parity follows it so the block a hop
  # computes from and the block the next hop streams into are always different.
  t = j * ring_size + r
  slot = lax.rem(t, 2)
  nslot = 1 - slot

  is_first_hop = (j == 0) & (r == 0)
  is_last_hop = (j == grid_width - 1) & (r == ring_size - 1)

  upstream = _neighbor(axis_names, ring_axis, -1)
  downstream = _neighbor(axis_names, ring_axis, +1)

  def _local_load(dst_slot, block):
    """Own KV block, HBM -> VMEM: what the stock BlockSpec pipeline would fetch.
    The ring never re-reads a neighbour's block from HBM, only over ICI."""
    src = (pl.ds(hk, 1), pl.ds(block * bkv, bkv))
    dst = pl.ds(dst_slot, 1)
    return (
        pltpu.make_async_copy(k_hbm.at[src], k_buf.at[dst], local_sem.at[0]),
        pltpu.make_async_copy(v_hbm.at[src], v_buf.at[dst], local_sem.at[1]),
    )

  def _remote_push(src_slot, dst_slot):
    """One ring hop: the KV block I hold now -> the downstream rank's next slot."""
    src, dst = pl.ds(src_slot, 1), pl.ds(dst_slot, 1)
    return (
        pltpu.make_async_remote_copy(k_buf.at[src], k_buf.at[dst], send_sem.at[0], recv_sem.at[0], device_id=downstream),
        pltpu.make_async_remote_copy(v_buf.at[src], v_buf.at[dst], send_sem.at[1], recv_sem.at[1], device_id=downstream),
    )

  # ------------------------------------------------------------------- DMA --
  # (0) Once per launch: rendezvous with both ring neighbours before any remote
  # write or remote semaphore signal can be issued (the Pallas all_gather
  # example's `main_barrier`). At R == 2 upstream and downstream are the same
  # rank, which the 2-signal / 2-wait form still handles.
  @pl.when((b == 0) & (h == 0) & (i == 0) & is_first_hop)
  def _barrier():
    sem = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(sem, 1, device_id=upstream)
    pl.semaphore_signal(sem, 1, device_id=downstream)
    pl.semaphore_wait(sem, 2)

  # (1) First hop of a q block has no prefetch behind it: load and block. Not
  # prefetching across the (b, h, i) boundary keeps the head/block index out of
  # the DMA schedule at the cost of one exposed 2 x bkv HBM read per q block.
  @pl.when(is_first_hop)
  def _prime():
    for dma in _local_load(0, 0):
      dma.start()
    for dma in _local_load(0, 0):
      dma.wait()

  # (2) Wait for the block this hop computes from.
  @pl.when(jnp.logical_not(is_first_hop))
  def _await_current():
    @pl.when(r == 0)
    def _await_local():
      # My own block j, prefetched from HBM by hop (j-1, R-1).
      for dma in _local_load(slot, j):
        dma.wait()

    @pl.when(r > 0)
    def _await_remote():
      # Pushed by the upstream rank one hop ago: its source slot was `nslot`,
      # my destination slot is `slot`. Same shapes => same semaphore credit.
      for dma in _remote_push(nslot, slot):
        dma.wait_recv()

  # (3) Retire my own previous send before its source buffer is reused as this
  # hop's DMA destination (hop t-1's source slot == hop t's destination slot).
  # Hop t-1 pushed iff its r was < R-1: always true when r > 0, never when
  # r == 0 (the r == R-1 hop reloads from HBM instead of pushing).
  @pl.when(r > 0)
  def _retire_send():
    for dma in _remote_push(nslot, slot):
      dma.wait_send()

  # (3b) Release `nslot` -- the block I finished computing at hop t-1 -- to the
  # upstream rank, which is about to push into it.
  #
  # THIS MUST LIVE IN THE PROLOGUE OF HOP t, NOT THE EPILOGUE OF HOP t-1.
  # Emitted from the epilogue it sits in the same grid step as the compute that
  # reads `k_buf[slot]`, and nothing orders a `semaphore_signal` after those
  # vector loads -- Mosaic may hoist it, letting the upstream overwrite a buffer
  # that is still being read. That is a genuine race: it reproduced at R=8 with
  # a ragged tail, non-deterministically (rel err 0.30 / 0.41 / 0.52 on repeat
  # runs of one config), while R<=4 happened to schedule safely. Grid iteration
  # order IS a real ordering guarantee, so releasing here puts the signal
  # provably after hop t-1's compute.
  #
  # Condition: release iff a push targets `nslot` at THIS hop, i.e. r < R-1.
  # At the last hop of a q block (r == R-1) the slot is refilled from HBM
  # instead, so no credit is owed -- which also closes the ledger at zero
  # (Pallas checks semaphores are drained at kernel exit) with no seed needed:
  # at hop 0 `nslot` has never been written, so releasing it is correct and it
  # is exactly the credit the downstream rank's first push consumes.
  @pl.when((r < ring_size - 1) & (ring_size > 1))
  def _release():
    pl.semaphore_signal(credit_sem, 1, device_id=upstream)

  # (4) Issue the next hop's transfer.
  @pl.when(jnp.logical_not(is_last_hop))
  def _prefetch_next():
    @pl.when(r < ring_size - 1)
    def _push():
      # One credit == "the downstream rank has retired the slot I am about to
      # write". Without it a push lands on a lagging neighbour's live block.
      pl.semaphore_wait(credit_sem, 1)
      for dma in _remote_push(slot, nslot):
        dma.start()

    @pl.when(r == ring_size - 1)
    def _reload_own():
      # End of a ring cycle: the next kv block starts from my own shard again.
      for dma in _local_load(nslot, j + 1):
        dma.start()

  # ------------------------------------------------------------- accumulate --
  @pl.when(is_first_hop)
  def _init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    if use_fixed_m:
      # Cauchy-Schwarz bound m_i = ceil(||q_i|| * max_j||k_j||) - C, pinned for
      # the WHOLE ring. Unlike the external ring this needs no per-hop gating and
      # no LSE merge: one kernel sees every shard and the accumulator never
      # leaves VMEM, so every hop is already subtracting the identical m.
      # `mk_ref[0, h]` is max||k|| reduced over the ring AND ulysses axes by the
      # caller, so the bound covers keys this rank never holds.
      qf = q_ref[...].astype(float32)
      qn = jnp.sqrt((qf * qf).sum(axis=1))[None, :]
      m_fixed = jnp.ceil(qn * mk_ref[0, h]) - custom_splash._FIXED_M_RECENTER  # pylint: disable=protected-access
      m_scratch_ref[...] = jnp.broadcast_to(m_fixed, m_scratch_ref.shape)
    else:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

  def _online_inner(qk, v_chunk, m_prev, l_prev, o_prev):
    step = bkv_compute_in
    for c in range(0, qk.shape[0], step):
      qk_slice = qk[c : c + step]
      m_curr = qk_slice.max(axis=0)[None, :]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = exp(qk_slice - m_next[0:1])
      l_curr = s_curr.sum(axis=0, keepdims=True)
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      o_curr = lax.dot_general(
          v_chunk[c : c + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )
      o_prev = alpha[0:1, ...] * o_prev + o_curr
      m_prev, l_prev = m_next, l_next
    return m_prev, l_prev, o_prev

  def _fixed_inner(qk, v_chunk, m_fix, l_prev, o_prev):
    # m is constant: no reduce-max over the block and no alpha rescale of o.
    step = bkv_compute_in
    for c in range(0, qk.shape[0], step):
      s_curr = exp(qk[c : c + step] - m_fix[0:1])
      l_prev = l_prev + s_curr.sum(axis=0, keepdims=True)
      o_prev = o_prev + lax.dot_general(
          v_chunk[c : c + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )
    return l_prev, o_prev

  def _step(offset, length):
    q = q_ref[...]
    sl = pl.ds(offset, length)
    qk = lax.dot_general(k_buf[slot, sl, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    v_chunk = v_buf[slot, sl, :]
    if use_fixed_m:
      l_prev, o_prev = _fixed_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
      l_scratch_ref[...] = l_prev
    else:
      m_prev, l_prev, o_prev = _online_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[:])
      m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[:] = o_prev

  def compute_body(kv_compute_index, _):
    _step(kv_compute_index * bkv_compute, bkv_compute)

  assert bkv % bkv_compute == 0

  @pl.when(j != grid_width - 1)
  def _body():
    lax.fori_loop(0, bkv // bkv_compute, compute_body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def _last_body():
    # Ragged tail. `kv_seq_len` is the un-padded shard length and every ring rank
    # pads identically, so the same tail applies on every hop.
    if kv_seq_len % bkv == 0:
      lax.fori_loop(0, bkv // bkv_compute, compute_body, None, unroll=True)
    else:
      remain = kv_seq_len % bkv
      iter_num = (remain + bkv_compute - 1) // bkv_compute
      if remain % bkv_compute == 0:
        lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
      else:
        lax.fori_loop(0, iter_num - 1, compute_body, None, unroll=True)
        _step((iter_num - 1) * bkv_compute, remain % bkv_compute)

  # -------------------------------------------------------------- epilogue --
  # Nothing to release here: the credit for this hop's slot is emitted in the
  # NEXT hop's prologue (see 3b), which is the only placement that is provably
  # ordered after this hop's compute.
  @pl.when(is_last_hop)
  def _write_out():
    l = l_scratch_ref[...]
    l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
    o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def internal_ring_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: "custom_splash._BlockSizes",
    *,
    q_seq_len: int,
    kv_seq_len: int,
    ring_axis: str,
    ring_size: int,
    axis_names: tuple[str, ...],
    use_base2_exp: bool = True,
    use_fixed_m: bool = False,
    mk: jax.Array | None = None,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> jax.Array:
  """Single-launch ring attention; the hop is a remote DMA inside the kernel.

  Args:
    q: `(batch, num_q_heads, q_seq_padded, head_dim_qk)`, already LOG2E-scaled
      by the caller when `use_base2_exp`.
    k, v: `(batch, num_kv_heads, kv_seq_padded, head_dim)` local ring shard.
    q_seq_len / kv_seq_len: un-padded lengths (grid bounds / ragged tail).
    ring_axis / ring_size: mesh axis the KV rotates over and its (static) size.
    axis_names: `mesh.axis_names` of the enclosing shard_map, needed to spell a
      neighbour as a full mesh-index tuple.

  Returns:
    `(batch, num_q_heads, q_seq_len, head_dim_v)`, softmax-normalized.
  """
  batch, num_q_heads, _, head_dim_qk = q.shape
  num_kv_heads = k.shape[1]
  kv_pad = k.shape[2]
  head_dim_v = v.shape[-1]
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in

  # Scalar-prefetch operand: mk[0,h] = max_j||k_j|| over EVERY ring shard,
  # mk[1,h] = per-head eligibility. A dummy keeps the signature uniform when the
  # online path is compiled.
  if mk is None:
    mk = jnp.zeros((2, num_q_heads), jnp.float32)

  grid_width = (kv_seq_len + bkv - 1) // bkv
  grid_height = (q_seq_len + bq - 1) // bq
  grid = (batch, num_q_heads, grid_height, grid_width, ring_size)

  # `*_` absorbs the scalar-prefetch operand, which Pallas appends to every
  # index_map's argument list once num_scalar_prefetch > 0.
  def q_index_map(b, h, i, j, r, *_):
    return (b, h, i, 0)

  def out_index_map(b, h, i, j, r, *_):
    return (b, h, 0, i)

  in_specs = [
      pl.BlockSpec((None, None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  out_specs = pl.BlockSpec((None, None, head_dim_v, bq), out_index_map)
  out_shape = jax.ShapeDtypeStruct((batch, num_q_heads, head_dim_v, q_seq_len), q.dtype)

  scratch_shapes = [
      pltpu.VMEM((2, bkv, head_dim_qk), k.dtype),
      pltpu.VMEM((2, bkv, head_dim_v), v.dtype),
      pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((head_dim_v, bq), jnp.float32),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.REGULAR,
  ]

  out = pl.pallas_call(
      functools.partial(
          _internal_ring_kernel,
          mask_value=mask_value,
          grid_width=grid_width,
          ring_size=ring_size,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          kv_seq_len=kv_seq_len,
          q_heads_per_kv_head=q_heads_per_kv_head,
          num_kv_heads=num_kv_heads,
          use_base2_exp=use_base2_exp,
          use_fixed_m=use_fixed_m,
          axis_names=tuple(axis_names),
          ring_axis=ring_axis,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          # Every dimension is "arbitrary": the in-kernel ring makes grid order
          # semantically load-bearing (hop r must follow hop r-1, and the credit
          # protocol assumes every rank walks the grid in lockstep), so no
          # dimension may be reordered or split across cores. In particular `h`
          # cannot be "parallel" the way the stock kernel has it.
          dimension_semantics=("arbitrary",) * 5,
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes,
          collective_id=_COLLECTIVE_ID,
          has_side_effects=True,
      ),
      out_shape=out_shape,
  )(mk, q, k.reshape(batch * num_kv_heads, kv_pad, head_dim_qk), v.reshape(batch * num_kv_heads, kv_pad, head_dim_v))
  return jnp.swapaxes(out, 2, 3)


def make_internal_ring_attention(
    *,
    block_sizes: "custom_splash._BlockSizes",
    orig_q_seq_len: int,
    orig_kv_seq_len: int,
    ring_axis: str,
    ring_size: int,
    axis_names: tuple[str, ...],
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    use_fixed_m: bool = False,
):
  """Batched `(b, h, s, d) -> (b, h, s, d)` callable. Deliberately NOT vmapped:
  the batch axis is a grid dimension, because vmapping a pallas_call that owns
  semaphores and a collective_id would duplicate the collective per batch
  element."""

  def _ring(q, k, v, mk=None):
    return internal_ring_attention_forward(
        q,
        k,
        v,
        block_sizes,
        q_seq_len=orig_q_seq_len,
        kv_seq_len=orig_kv_seq_len,
        ring_axis=ring_axis,
        ring_size=ring_size,
        axis_names=axis_names,
        use_base2_exp=use_base2_exp,
        use_fixed_m=use_fixed_m,
        mk=mk,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
        mask_value=mask_value,
    )

  return _ring
