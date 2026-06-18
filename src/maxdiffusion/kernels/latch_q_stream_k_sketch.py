"""
Latch-Q / Stream-K flash-attention inner loop — DESIGN SKETCH (WIP, UNTESTED).

Goal: kill the register spilling that dominates the current kernel. The post-RA
LLO dump for bq=5888/cin=256 showed ~43k spill ops (85% of all VMEM traffic,
9,465 spill slots) because the bq=5888-wide score canvas is ~23x the 64-VREG
file and `bkv_compute_in` only tiles the kv axis — the bq lane-width is never
tiled, so the allocator spills everything.

This sketch tiles the *bq* axis into 256-wide strips processed with the low-level
MXU primitives, keeping the live working set at one [256,256] tile (~64 VREGs) and
evicting each strip MRB->VPU before the next QK lands. That is the missing axis.

Status: this is a STRUCTURAL skeleton to reconcile with the working Colab
baseline (Kunjan's "Latch Q & Stream K"). The index algebra and primitive order
are worked out; the exact accumulator/staging-register *addresses* and the f32
accumulator byte offsets are marked TODO and must be validated on device — these
primitives have hard rules (no data left in staging/acc on exit) and there is no
bounds checking. Do NOT wire this into production before it round-trips a
numerical diff vs _flash_attention_kernel.

Key constraints (from jax 0.10.0 pltpu primitive docstrings):
  matmul_push_rhs(rhs[256,256], staging_register, mxu_index, transpose=False)
  matmul_acc_lhs(acc_addr, lhs[M,256], mxu_index, load_staged_rhs=None)
  matmul_pop(acc_addr, shape, dtype, mxu_index) -> [M,256] f32, zeroes acc
  - out[m,n] = sum_k lhs[m,k] * staged_rhs[k,n]   (contraction over the 256 dim)
  - load_staged_rhs=None REUSES the loaded RHS (this is the "don't re-latch" win)
  - RHS 256x256; LHS M x 256; accumulator f32/i32; nothing left resident on exit

Layout (kept identical to _flash_attention_kernel so the wrapper/grid is reused):
  TRANSPOSED. scores live as [kv, q]; output o as [head_dim, q].
  QK (mxu0): lhs=K[kv,hd], rhs=Q_strip pushed transpose=True -> [hd,q]
             => out[kv,q] = sum_hd K[kv,hd]*Q[q,hd]               (== current qk)
  PV (mxu1): lhs=V[kv,hd] used as [hd,kv], rhs=p[kv,q]
             => out[hd,q] = sum_kv V[kv,hd]*p[kv,q]               (== current o)
  Both contractions are over head_dim => REQUIRES head_dim == 256 to fill the MXU.
"""


import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TILE = 256  # MXU-native systolic dimension
MXU_QK = 0  # dual-MXU: QK on MXU0 ...
MXU_PV = 1  # ... PV on MXU1, so QK(next) overlaps PV(curr)
STAGE_Q = 0  # staging register holding the latched Q strip
STAGE_P = 0  # staging register on MXU1 for the per-tile p

# Accumulator base addresses (f32 slices). TODO(device): confirm these don't
# overlap and fit the acc file; one [256,256] f32 tile = 256 sublane-rows.
ACC_QK = 0
ACC_PV = 0


def _strip_flash(q_strip, k_ref, v_ref, kv_seq_len, mask_value, exp):
  """One 256-wide bq strip: stream all KV with latched Q, online softmax.

  q_strip: [TILE, head_dim(=256)] — the 256 query rows for this strip.
  k_ref/v_ref: VMEM refs, [kv_seq_len, head_dim].
  Returns o_strip [head_dim, TILE] (un-normalized) and l [1, TILE] for the caller
  to divide by — matching the current kernel's deferred 1/l epilogue.
  """
  head_dim = q_strip.shape[1]
  assert head_dim == TILE, "Latch-Q-Stream-K needs head_dim==256 to fill the MXU"

  # --- Latch Q ONCE for the whole KV stream (the core idea) -------------------
  # transpose=True so the staged RHS is Q^T [hd, q]; reused across every kv tile.
  pltpu.matmul_push_rhs(q_strip, STAGE_Q, MXU_QK, transpose=True)

  # Running online-softmax state for these 256 queries. Kept tiny + resident:
  #   m,l : [1, TILE]   o : [head_dim, TILE]
  m = jnp.full((1, TILE), mask_value, jnp.float32)
  l = jnp.zeros((1, TILE), jnp.float32)
  o = jnp.zeros((head_dim, TILE), jnp.float32)

  num_kv_tiles = kv_seq_len // TILE  # TODO: handle ragged tail like last_compute_body
  first = True

  def body(t, carry):
    nonlocal first
    m, l, o = carry
    kv0 = t * TILE
    k_tile = k_ref[pl.ds(kv0, TILE), :]  # [kv=256, hd=256] -> lhs for QK
    v_tile = v_ref[pl.ds(kv0, TILE), :]  # [kv=256, hd=256] -> lhs for PV

    # --- QK on MXU0: load staged Q only on the FIRST tile, then reuse ----------
    # load_staged_rhs=STAGE_Q on the first acc; None afterwards = no re-latch.
    pltpu.matmul_acc_lhs(ACC_QK, k_tile, MXU_QK, load_staged_rhs=STAGE_Q if first else None)
    qk = pltpu.matmul_pop(ACC_QK, (TILE, TILE), jnp.float32, MXU_QK)  # [kv, q]
    first = False

    # --- online softmax on the [kv, q] tile (reduce over kv = axis 0) ----------
    m_curr = qk.max(axis=0, keepdims=True)  # [1, q]
    m_next = jnp.maximum(m, m_curr)
    p = exp(qk - m_next)  # [kv, q]  (EPU)
    alpha = exp(m - m_next)  # [1, q]
    l = alpha * l + p.sum(axis=0, keepdims=True)

    # --- PV on MXU1: overlaps the next QK on MXU0 -----------------------------
    # rhs = p [kv, q] (changes every tile -> must push+load each time);
    # lhs = V used as [hd, kv]; out[hd, q] = sum_kv V[kv,hd] p[kv,q].
    pltpu.matmul_push_rhs(p.astype(v_tile.dtype), STAGE_P, MXU_PV)
    pltpu.matmul_acc_lhs(ACC_PV, v_tile, MXU_PV, load_staged_rhs=STAGE_P)
    o_curr = pltpu.matmul_pop(ACC_PV, (head_dim, TILE), jnp.float32, MXU_PV)

    # --- rescale + accumulate o (VPU). The online-softmax rescale is why PV is
    # popped per-tile rather than accumulated in the MXU across all kv tiles —
    # the running-max correction can't be applied inside the MXU accumulator.
    # OPTIMIZATION TARGET: a 2-pass / max-deferred scheme could let PV accumulate
    # in-MXU and drop this per-tile rescale (see assignment: redundant round-trips).
    o = alpha * o + o_curr
    m = m_next
    return m, l, o

  m, l, o = lax.fori_loop(0, num_kv_tiles, body, (m, l, o), unroll=True)
  return o, l  # caller does o / l (deferred normalize, as today)


# ---------------------------------------------------------------------------
# Open design questions to settle against the working baseline / on device:
#
# 1. head_dim==256. With the current head_dim==128 you must either pad the
#    contraction (back to eff=0.5, defeating the point) or pack two heads into
#    the 256 contraction lane. This kernel only pays off WITH the hdim-256
#    surgery — they compose; sequence them together.
#
# 2. Staging/accumulator register budget. STAGE_*/ACC_* are placeholders. Each
#    MXU has its own staging set; confirm one in-flight Q (mxu0) + one in-flight
#    p (mxu1) + the two f32 acc tiles fit, and that nothing is left resident on
#    exit (hard rule, no bounds check).
#
# 3. Dual-MXU overlap. QK(t+1) on MXU0 should issue while PV(t) runs on MXU1.
#    With unroll=True the scheduler can interleave; verify in the LLO that the
#    two vmatpush/vmatmul streams target mxu0 vs mxu1 and actually overlap.
#
# 4. Redundant-latch audit (the assignment): Q is latched once per strip here
#    (load_staged_rhs only on first tile). If the baseline re-pushes Q per kv
#    tile, that's the first thing to delete. p MUST re-push each tile (it
#    changes) — that one is not redundant.
#
# 5. Tail handling: num_kv_tiles assumes kv_seq_len % 256 == 0. Mirror the
#    current kernel's last_compute_body for the ragged tail, OR pad kv to 256.
#
# 6. Success metric: re-dump packed-bundles-post-ra and confirm spill stores/
#    fills (#allocationN_spill) drop from ~43k toward ~0. That, not MXU %, is
#    the number that predicts the speedup.
# ---------------------------------------------------------------------------
