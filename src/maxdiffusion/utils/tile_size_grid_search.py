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

Tile-size (block_q / block_kv) grid search for flash/splash/ring attention kernels.

Two hardware granularities drive the candidate math (do NOT conflate them):
  * VPU_LANE = 128 — the VMEM/vector lane width and the kernel's HARD floor: every
    block size must be a multiple of 128 or the splash kernel raises.
  * MXU_TILE = 256 — the systolic matmul array is 256x256 on v7x.  A block
    dimension that is an *odd* multiple of 128 (e.g. 896 = 3.5x256) leaves the last
    MXU pass half-empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence
import csv
import os
import sys
import jax
import numpy as np
from jax.experimental import multihost_utils

# --- the two granularities (see module docstring) --------------------------------
VPU_LANE = 128  # kernel hard floor: block sizes must be multiples of this
MXU_TILE = 256  # 256x256 MXU: multiples of this fully pack the systolic array

# Kernel-family sets live in ONE place (attention_kernel_registry); they are
# re-exported here because the benches and tests import them from this module.
from maxdiffusion.attention_kernel_registry import (  # pylint: disable=g-importing-member
    INTERNAL_PERM_KERNELS,
    PURE_RING_ATTENTION_KERNELS,
    ULYSSES_RING_ATTENTION_KERNELS,
    auto_block_q_sub,
)


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _ceil_to(x: int, m: int) -> int:
  return _ceil_div(x, m) * m


def _floor_to(x: int, m: int) -> int:
  return (x // m) * m


def local_tiled_seq_len(full_seq: int, attention: str, context_shards: int, ulysses_shards: int) -> int:
  """Returns the sequence length tiled by one local ring kernel invocation."""
  context_shards = max(1, context_shards)
  context_local_seq = _ceil_div(full_seq, context_shards)
  if attention in PURE_RING_ATTENTION_KERNELS:
    return context_local_seq
  elif attention in ULYSSES_RING_ATTENTION_KERNELS:
    if ulysses_shards < 1:
      raise ValueError(f"{attention} requires ulysses_shards >= 1, got {ulysses_shards}.")
    if context_shards % ulysses_shards:
      raise ValueError(
          f"context_shards={context_shards} must be divisible by ulysses_shards={ulysses_shards} for {attention}."
      )
    # Production pads the global sequence to a context-shard multiple, then
    # Ulysses gathers `ulysses_shards` local sequence chunks per head shard.
    return context_local_seq * ulysses_shards
  else:
    return full_seq


@dataclass(frozen=True)
class Padding:
  seq_len: int
  block: int
  n_blocks: int
  padded_len: int
  pad: int
  pad_pct: float


def padding_of(seq_len: int, block: int) -> Padding:
  """How `block` tiles `seq_len`: block count, padded length, and wasted rows."""
  n = _ceil_div(seq_len, block)
  padded = n * block
  return Padding(
      seq_len,
      block,
      n,
      padded,
      padded - seq_len,
      100.0 * (padded - seq_len) / seq_len,
  )


def block_for_count(seq_len: int, n: int, align: int = MXU_TILE) -> Optional[int]:
  """Smallest multiple of `align` that tiles `seq_len` into EXACTLY n blocks, else None.

  (Rounding seq/n up to `align` can bump the block big enough to yield n-1 blocks; such
  counts have no clean aligned block and are skipped.)"""
  b = max(align, _ceil_to(_ceil_div(seq_len, n), align))
  return b if _ceil_div(seq_len, b) == n else None


def full_axis_candidates(
    seq_len: int,
    *,
    step: int = MXU_TILE,
    min_block: int = MXU_TILE,
    max_block: Optional[int] = None,
) -> list[int]:
  """Full sweep: [min_block, min_block+step, ..., max_block].  `max_block` defaults
  to the single-block ceiling (round_up(seq_len, step)); nothing bigger helps.  Default
  step=256 keeps the count sane (step=128 ~doubles it; use only for a fine characterization).
  """
  if step % VPU_LANE:
    raise ValueError(f"step must be a multiple of {VPU_LANE}, got {step}")
  max_block = max_block or _ceil_to(seq_len, step)
  return list(range(min_block, max_block + 1, step))


def bq_candidates(
    seq_len: int,
    *,
    k: int = 3,
    spread: int = 2,
    align: int = VPU_LANE,
    max_block: Optional[int] = None,
    min_block: int = VPU_LANE,
) -> list[int]:
  """BQ: VMEM-capped fewest-tile ladder + geometric spread-down.

  `max_block` = the BQ VMEM ceiling (e.g. `vmem_bq_ceiling(min_bkv)`); defaults to the
  single-block size.  Takes the `k` fewest-tile blocks <= max_block (fewer Q-tiles win),
  then `spread` progressively-halved blocks (snapped to the min-padding aligned block for
  that tile count) so a MODERATE optimum -- e.g. ulysses bq~5888, well below single-tile --
  is also sampled.  Returned high->low, deduped.  align=128 (256-mult preferred, not
  required)."""
  cap = _floor_to(min(max_block or _ceil_to(seq_len, align), _ceil_to(seq_len, align)), align)
  max_n = _ceil_div(seq_len, min_block)
  out: list[int] = []
  n = _ceil_div(seq_len, cap)  # fewest tiles that fit the cap
  while len(out) < k and n <= max_n:
    b = block_for_count(seq_len, n, align)
    if b is not None and min_block <= b <= cap and b not in out:
      out.append(b)
    n += 1
  b = out[-1] if out else cap
  for _ in range(spread):  # geometric spread toward moderate blocks
    b = _floor_to(b // 2, align)
    if b < min_block:
      break
    snapped = block_for_count(seq_len, _ceil_div(seq_len, b), align) or b
    if min_block <= snapped <= cap and snapped not in out:
      out.append(snapped)
  return sorted(set(out), reverse=True)


# ---------------------------------------------------------------------------
# VMEM ceiling model.
#
# Replaces a single `score_tile <= vmem * score_fraction` fraction, which cannot
# be right: VMEM also holds terms that scale with bq INDEPENDENTLY of bkv (the q
# block double-buffered, the fp32 `o` accumulator, the output block). Proof --
# two tiles with an identical score-tile product: 9472x1024 FITS, 18944x512 OOMs.
# A one-term model gets tuned to the worst corner and under-caps everywhere else.
#
# Fit: `used/bq = a*(4*bkv) + b`, one (a, b) per family below -- they differ
# because the external ring keeps fp32 online-softmax residual windows while
# the internal-permutation kernel carries (m, l, o) in one VMEM scratch.
# Measured: external OOMs at 9472/1280 (3/3 reps) where internal runs it.
#
# CAVEAT -- external + fixed-m. There the 1-block bench disagrees with e2e: dense
# sweeps proposed tiles that measured WORSE end to end in 3/3 cases (1.570 vs
# 1.547, 1.610 vs 1.528, 1.900 vs 1.898 s/step). A correct ceiling widens the
# search space, and for that combination a wider space can surface a tile the
# bench over-rates. Verify any fixed-m winner end to end before trusting it.
_VMEM_FIT = {  # family -> (score scale a, per-bq bytes b)
    "external": (0.973, 1989),
    "internal": (0.982, 1295),
}


# Measured ceilings from the sweeps (~3,300 points), MIN across configs so the
# search never proposes a tile that OOMs for some shape. Preferred over the
# fit wherever the exact bq was swept -- a 2-term model provably cannot bracket
# both ends (at bq=9472 internal needs b<=1226 to reach its true 1408 ceiling; at
# bq=18944 it needs b>1367 to avoid over-capping -- contradictory), and the rung
# that matters most, 9472, is exactly where the fit under-caps.
#
# NOTE: these are for a 64 MiB VMEM budget. They are scaled linearly for other
# budgets, which is only approximate -- the per-bq term does not scale with the
# score tile. For a different VMEM size the fit is used instead.
_MEASURED_BKV_CEILING = {  # (family, bq) -> min ceiling across configs @64MB
    "external": {
        1024: 4096,
        1280: 4096,
        1536: 4096,
        1792: 4096,
        2048: 4096,
        2304: 4096,
        2560: 4096,
        2816: 4096,
        3072: 4096,
        3328: 4096,
        3584: 3968,
        3840: 3840,
        4096: 3328,
        4352: 3200,
        4608: 2944,
        4864: 2816,
        5376: 2560,
        5632: 2304,
        6144: 2048,
        6400: 2048,
        7168: 1664,
        7680: 1536,
        7936: 1536,
        8192: 1408,
        8448: 1408,
        8704: 1280,
        8960: 1280,
        9472: 1152,
        10240: 1024,
        11008: 896,
        11264: 896,
        12032: 768,
        12800: 768,
        14080: 640,
        14336: 512,
        15360: 512,
        15872: 512,
        17152: 384,
        18944: 256,
        22272: 256,
        25344: 256,
    },
    "internal": {
        1024: 4096,
        1280: 4096,
        1536: 4096,
        1792: 4096,
        2048: 4096,
        2304: 4096,
        2560: 4096,
        2816: 4096,
        3072: 4096,
        3328: 4096,
        3584: 4096,
        3840: 3968,
        4096: 3456,
        4352: 3328,
        4608: 3072,
        4864: 2944,
        5376: 2688,
        5632: 2432,
        6144: 2304,
        6400: 2176,
        7168: 1920,
        7680: 1664,
        7936: 1664,
        8704: 1536,
        9472: 1408,
        11264: 1024,
        12800: 896,
        14336: 768,
        15872: 640,
        18944: 384,
    },
}

# The sweeps ran at the kernel's vmem_limit_bytes, which is 64 MiB (67,108,864),
# NOT 64e6. Getting this wrong silently disables the table and falls back to the
# fit -- caught by tile_size_grid_search_test.test_bkv_ceiling_is_per_family.
_MEASURED_VMEM_BYTES = 64 * 1024 * 1024


def vmem_family(attention: str) -> str:
  """Which calibrated VMEM fit applies to this attention kernel."""
  return "internal" if attention in INTERNAL_PERM_KERNELS else "external"


def vmem_bkv_ceiling(
    bq: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    align: int = VPU_LANE,
    family: str = "external",
) -> int:
  """Largest bkv that fits at this bq. Returns 0 when bq alone exhausts VMEM.

  align defaults to VPU_LANE (128), not MXU_TILE: the measured optima include
  1152, 1280 and 1408, none of which a 256-aligned ladder can express.
  """
  # Prefer the measured ceiling when this exact bq was swept at this VMEM size.
  if abs(vmem_bytes - _MEASURED_VMEM_BYTES) < 512 * 1024:
    hit = _MEASURED_BKV_CEILING.get(family, {}).get(bq)
    if hit is not None:
      return _floor_to(hit, align)
  a, b = _VMEM_FIT[family]
  per_row = vmem_bytes / max(bq, 1) - b
  if per_row <= 0:
    return 0
  return max(0, _floor_to(int(per_row / (a * dtype_bytes)), align))


def vmem_bq_ceiling(
    bkv: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    align: int = VPU_LANE,
    family: str = "external",
) -> int:
  """Largest bq that fits at this bkv (inverse of `vmem_bkv_ceiling`)."""
  a, b = _VMEM_FIT[family]
  return max(align, _floor_to(int(vmem_bytes / (a * bkv * dtype_bytes + b)), align))


def bkv_candidates(seq_len: int, *, k: int = 3, align: int = VPU_LANE, max_block: int) -> list[int]:
  """BKV(=bkv_compute): the MXU compute tile.  Largest tiles that fit VMEM, descending by
  `align` from `max_block` (a VMEM ceiling, e.g. `vmem_bkv_ceiling`).  Capped at the
  single-block size (no point tiling KV bigger than the sequence).  align=128 by default
  so a 128-multiple that fits (e.g. 1152) is considered alongside 256-multiples (1024).
  """
  top = min(_floor_to(max_block, align), _ceil_to(seq_len, align))
  out: list[int] = []
  b = top
  while b >= align and len(out) < k:
    out.append(b)
    b -= align
  return out


# ================================================================================
# Internal-permutation (iperm) kernels with a real ring: resident-Q VMEM model.
# ================================================================================
# With ring_shards > 1 every iperm kernel runs ONE pallas_call per head for the
# whole ring. The ENTIRE local Q shard -- R = ceil_to(q_seq, block_q) rows, so
# block_q only matters through the padding it adds -- stays resident in VMEM
# and is walked in `block_q_sub`-row chunks while K/V stream through in
# `block_kv` blocks. The per-(bq, bkv) tile model above never sees R or
# block_q_sub, so for these kernels it is wrong in both directions: it passes
# the production OOM (R=48640, block_kv=1024) and it caps away resident blocks
# that fit. Calibrated replacement (bytes; bf16 Q/K/V, head_dim 128):
#
#   VMEM(R, q_sub, bkv) = 1024*R + 4.02*q_sub*block_kv_compute + 1024*bkv
#
#   1024*R    per resident row: the bf16 Q row plus the f32 accumulator and
#             online-softmax statistics the kernel keeps for every row.
#   4.02*q*c  the f32 (block_q_sub x block_kv_compute) score/probability tile.
#   1024*bkv  the double-buffered bf16 K and V blocks.
#
# Fitted on the `Used` figure of the Mosaic VMEM-OOM reports (126) from ten
# iperm-hybrid sweeps (dp1cp8 and dp2cp4, 2.5-12.5 s) plus the production
# failure. Production (R=48640, auto q_sub=12160, bkv=1024): 96.2 MiB predicted,
# 98.0 MiB measured (-1.8%). Predicted-minus-used spans [-7.0, +2.5] MiB, the
# large misses on huge tiles far past the budget; the 12.5 s sweep (largest R)
# is within [-0.9, -0.6] MiB. Against a 64 MiB budget every swept config that
# OOMed predicts >= 103.7% and every config that ran predicts <= 98.9%, so
# IPERM_VMEM_MARGIN = 0.98 admits no config that OOMed and covers the
# production under-prediction. It does exclude the four configs that ran at
# 98-98.9%, all slow (e.g. R=28160, q_sub=7040, bkv=1280: 233.9 ms vs 146.6 ms
# at bkv=1024); 0.97 would exclude the measured 10 s winner (97.5%).
IPERM_VMEM_MARGIN = 0.98
_IPERM_ROW_BYTES = 1024
_IPERM_SCORE_BYTES = 4.02
_IPERM_KV_BYTES = 1024
# block_q_outer < R switches the kernel from the single-buffered resident Q to
# double-buffered Q/output blocks (RESIDENT_SINGLE_BUFFER gate in
# internal_ring_attention.py). No sweep has exercised it yet, so this is an
# UNCALIBRATED upper estimate (1.5x the per-row cost plus 3 MiB of slack); the
# run-time OOM classification is the backstop.
_IPERM_OUTER_ROW_BYTES = 1536
_IPERM_OUTER_SLACK_BYTES = 3 * 1024 * 1024

# Planner knobs (see `iperm_candidates`).
IPERM_MAX_CANDIDATES = 30
IPERM_SAFE_FRACTION = 0.85  # the "safe" anchor leaves >= 15% VMEM headroom
IPERM_MAX_PAD_FRACTION = 0.04  # resident lengths considered: up to 4% over q_seq
IPERM_RESIDENT_TOP_K = 3  # best-scoring resident lengths kept from that sweep
IPERM_MAX_OUTER_BLOCKS = 8  # block_q_outer: at most 8 ring re-walks
IPERM_BKV_ANCHORS = (1024, 1280)
IPERM_BKV_EXTRAS = (512, 768, 1536, 2048)
# block_q_sub values that won (or tied for best) across the measured sweeps.
IPERM_KNOWN_GOOD_Q_SUB = (7040, 6272, 5760, 4736, 3840, 3584, 2816, 2688, 1920, 1664)
# block_q_sub admissibility (see `_q_sub_admissible`). Every candidate walks R
# in at least two chunks: a single whole-block chunk is the instruction-memory
# cliff (the static program grows with R; ~2.1x slower). Formula-driven
# candidates also stay inside the measured regime:
#  * q_sub * block_kv_compute <= 8 Mi elements -- the largest score tile that
#    measured fast is 3840 x 2176; 7040 x 1280 ran 233.9 ms vs 146.6 ms at
#    7040 x 1024.
#  * q_sub >= 1024 once R >= 8192 -- at those sizes every measured q_sub
#    below 1024 ran 1.2-2.1x slower than the shape's best (e.g. 384: 453.6 ms
#    vs 234.3 ms at 10 s).
IPERM_MAX_SCORE_TILE = 8 * 1024 * 1024
IPERM_MIN_Q_SUB = 1024
IPERM_MIN_Q_SUB_FROM_RESIDENT = 8192
# When the resident length is not on the existing bq ladder, block_q is picked
# as the largest divisor of R up to this size that still pads q_seq to R.
# block_q is still the flash (cross-attention) tile, and this keeps it in the
# range the sweeps exercised.
_IPERM_PREFERRED_MAX_BQ = 8192


def iperm_vmem_bytes(
    resident: int,
    block_q_sub: int,
    block_kv: int,
    block_kv_compute: Optional[int] = None,
) -> float:
  """Predicted VMEM bytes of a whole-shard-resident iperm launch (see above)."""
  cmp = block_kv if block_kv_compute is None else block_kv_compute
  return _IPERM_ROW_BYTES * resident + _IPERM_SCORE_BYTES * block_q_sub * cmp + _IPERM_KV_BYTES * block_kv


def iperm_outer_vmem_bytes(
    block_q_outer: int,
    block_q_sub: int,
    block_kv: int,
    block_kv_compute: Optional[int] = None,
) -> float:
  """Predicted VMEM bytes of an iperm launch split by block_q_outer (uncalibrated, conservative)."""
  cmp = block_kv if block_kv_compute is None else block_kv_compute
  return (
      _IPERM_OUTER_ROW_BYTES * block_q_outer
      + _IPERM_SCORE_BYTES * block_q_sub * cmp
      + _IPERM_KV_BYTES * block_kv
      + _IPERM_OUTER_SLACK_BYTES
  )


def ring_shards_for(attention: str, context_shards: int, ulysses_shards: int) -> int:
  """Ring size the attention kernel runs: CP for pure ring, CP / U for Ulysses x ring, else 1."""
  if attention in PURE_RING_ATTENTION_KERNELS:
    return max(1, context_shards)
  if attention in ULYSSES_RING_ATTENTION_KERNELS:
    return max(1, context_shards // max(1, ulysses_shards))
  return 1


def iperm_uses_resident_q(attention: str, ring_shards: Optional[int]) -> bool:
  """True when the resident-Q model (not the per-tile one) governs this kernel's VMEM.

  iperm kernels only take the whole-ring, whole-shard-resident path with more
  than one ring shard; with one they fall back to the standard splash kernel.
  An unknown ring size is treated as resident -- the conservative model.
  """
  return attention in INTERNAL_PERM_KERNELS and (ring_shards is None or ring_shards > 1)


def _divisors_128(n: int) -> list[int]:
  """Ascending VPU_LANE-multiple divisors of n (itself a VPU_LANE multiple)."""
  if n <= 0 or n % VPU_LANE:
    raise ValueError(f"expected a positive multiple of {VPU_LANE}, got {n}")
  m = n // VPU_LANE
  small, large = [], []
  i = 1
  while i * i <= m:
    if m % i == 0:
      small.append(i * VPU_LANE)
      if i != m // i:
        large.append((m // i) * VPU_LANE)
    i += 1
  return small + large[::-1]


def _iperm_cost(resident: int, block_q_sub: int, block_kv: int, outer_blocks: int = 1) -> float:
  """Relative cost proxy that ONLY orders and caps candidates; the bench picks the winner.

  Rows to compute, plus a per-chunk overhead that falls with bigger block_q_sub
  and block_kv (the measured sweeps' dominant trend), plus a ring re-walk per
  extra outer block.
  """
  return resident * (1.0 + 300.0 / block_q_sub) * (1.0 + 100.0 / block_kv) * (1.0 + 0.25 * (outer_blocks - 1))


@dataclass(frozen=True)
class TileCandidate:
  """One configuration the search compiles and times.

  block_q_sub / block_q_outer are None for kernels that don't read them (and
  None block_q_sub means the iperm kernel's own auto rule). `resident` (the Q
  rows one launch holds in VMEM: R, or block_q_outer when set),
  `pred_vmem_bytes` and `tag` are planner metadata carried into the results.
  """

  bq: int
  bkv: int
  bkv_compute: int
  block_q_sub: Optional[int] = None
  block_q_outer: Optional[int] = None
  resident: Optional[int] = None
  pred_vmem_bytes: Optional[float] = None
  tag: str = ""


@dataclass(frozen=True)
class IpermStructure:
  """Result of the pre-compilation structural check for a resident-Q iperm shape."""

  q_seq: int
  min_resident: int
  min_resident_bytes: float
  budget_bytes: float
  fits_resident: bool
  fits_outer: bool
  message: str


def iperm_structural_check(q_seq: int, *, vmem_bytes: int, margin: float = IPERM_VMEM_MARGIN) -> IpermStructure:
  """Can a whole-shard-resident Q block fit VMEM AT ALL for this local q_seq?

  The cheapest resident launch is R = ceil_to(q_seq, 128) with block_q_sub =
  block_kv = 128; the 1024*R term alone decides it. If even that exceeds the
  budget, no (block_q, block_kv, block_q_sub) can fit and the search must use
  block_q_outer < R -- decided here, before anything is compiled.
  `fits_outer` is the same floor for the smallest block_q_outer the planner
  allows (R / IPERM_MAX_OUTER_BLOCKS): when it is False not even splitting the
  shard helps, and the fix is a different sharding (more ring shards).
  """
  r_min = _ceil_to(q_seq, VPU_LANE)
  need = iperm_vmem_bytes(r_min, VPU_LANE, VPU_LANE)
  outer_min = _ceil_to(_ceil_div(r_min, IPERM_MAX_OUTER_BLOCKS), VPU_LANE)
  need_outer = iperm_outer_vmem_bytes(outer_min, VPU_LANE, VPU_LANE)
  budget = margin * vmem_bytes
  mib = 1024 * 1024
  fits_resident = need <= budget
  fits_outer = need_outer <= budget
  where = f"{budget / mib:.1f} MiB budget ({margin:.0%} of {vmem_bytes / mib:.0f} MiB)"
  if fits_resident:
    msg = f"whole-shard-resident Q fits: R >= {r_min} rows needs >= {need / mib:.1f} MiB of the {where}"
  elif fits_outer:
    msg = (
        f"whole-shard-resident Q cannot fit VMEM at local q_seq={q_seq}: R >= {r_min} resident rows "
        f"need >= {need / mib:.1f} MiB even at block_q_sub=block_kv={VPU_LANE}, over the {where}. "
        f"No (block_q, block_kv, block_q_sub) choice fixes this; it needs block_q_outer < R "
        "(the kernel grid splits the shard and re-walks the ring per outer block), or more ring "
        "shards / fewer Ulysses shards."
    )
  else:
    msg = (
        f"iperm cannot fit VMEM at local q_seq={q_seq} with any tiling: the resident block needs "
        f">= {need / mib:.1f} MiB and even block_q_outer={outer_min} ({IPERM_MAX_OUTER_BLOCKS} outer blocks) "
        f"needs >= {need_outer / mib:.1f} MiB, over the {where}. Use more ring shards / fewer Ulysses shards, "
        "or a non-iperm kernel."
    )
  return IpermStructure(q_seq, r_min, need, budget, fits_resident, fits_outer, msg)


def _bq_for_resident(q_seq: int, resident: int, ladder_bqs: Sequence[int]) -> int:
  """A block_q that pads q_seq to exactly `resident` rows.

  Prefers a bq from the existing ladder (the ones the sweeps validated), then
  the single-tile bq, then the largest divisor of R <= _IPERM_PREFERRED_MAX_BQ.
  Any divisor d of R with d > R - q_seq pads to R exactly.
  """
  hits = [bq for bq in ladder_bqs if _ceil_to(q_seq, bq) == resident]
  if hits:
    return max(hits)
  if resident == _ceil_to(q_seq, VPU_LANE):
    return resident
  valid = [d for d in _divisors_128(resident) if d > resident - q_seq]
  preferred = [d for d in valid if d <= _IPERM_PREFERRED_MAX_BQ]
  return max(preferred) if preferred else min(valid)


def _q_sub_admissible(rows: int, q_sub: int, block_kv: int, *, formula: bool) -> bool:
  """Whether block_q_sub may walk a `rows`-row Q block (resident R or block_q_outer).

  Always: a lane-multiple divisor of `rows` giving at least two chunks. With
  `formula` (every candidate except the explicit anchors) also the measured
  regime: IPERM_MAX_SCORE_TILE and IPERM_MIN_Q_SUB. VMEM is checked separately.
  """
  if q_sub < VPU_LANE or q_sub % VPU_LANE or rows % q_sub:
    return False
  if q_sub > max(VPU_LANE, rows // 2):
    return False
  if formula:
    if q_sub * block_kv > IPERM_MAX_SCORE_TILE:
      return False
    if rows >= IPERM_MIN_Q_SUB_FROM_RESIDENT and q_sub < IPERM_MIN_Q_SUB:
      return False
  return True


def _fitting_q_subs(resident: int, block_kv: int, budget: float, *, formula: bool = True) -> list[int]:
  """Admissible block_q_sub values (descending) whose resident launch fits `budget`."""
  return [
      q
      for q in reversed(_divisors_128(resident))
      if _q_sub_admissible(resident, q, block_kv, formula=formula) and iperm_vmem_bytes(resident, q, block_kv) <= budget
  ]


def iperm_candidates(
    q_seq: int,
    kv_seq: int,
    *,
    vmem_bytes: int,
    margin: float = IPERM_VMEM_MARGIN,
    max_candidates: int = IPERM_MAX_CANDIDATES,
    ladder_bqs: Sequence[int] = (),
) -> list[TileCandidate]:
  """Joint (block_q, block_kv, block_q_sub[, block_q_outer]) candidates for an iperm kernel with a ring.

  Every candidate is predicted to fit `margin * vmem_bytes` (see
  `iperm_vmem_bytes`); nothing predicted to OOM is proposed. block_q only sets
  the resident length R, so the planner searches R directly:

    * R values: the single-tile R (ceil_to(q_seq, 128)), the
      IPERM_RESIDENT_TOP_K best-scoring R in [q_seq, q_seq * 1.04] (an R with
      a large fitting divisor beats "least padding": the 10 s winner pads 2%
      to R=37632 for q_sub=6272), and every R the existing bq ladder reaches.
    * block_q_sub, per R: the largest admissible value that fits (the
      "frontier"), the next one down, and the repeat sweep winners
      (IPERM_KNOWN_GOOD_Q_SUB) wherever they divide R; plus the kernel's auto
      rule (`auto_block_q_sub`) when it fits.
    * block_kv: 1024 and 1280 always, plus IPERM_BKV_EXTRAS at the frontier.
    * one "safe" anchor at <= IPERM_SAFE_FRACTION of VMEM.

  Anchors are always kept; the rest is ordered by `_iperm_cost` and capped at
  `max_candidates`. When no resident block can fit at all
  (`iperm_structural_check`) the candidates are block_q_outer < R instead;
  when one fits only with a tiny block_q_sub, both kinds are returned. An
  empty list means not even block_q_outer fits: a structural failure.
  """
  budget = margin * vmem_bytes
  kv_cap = _ceil_to(kv_seq, VPU_LANE)
  anchor_bkvs = sorted({min(b, kv_cap) for b in IPERM_BKV_ANCHORS})
  extra_bkvs = sorted({min(b, kv_cap) for b in IPERM_BKV_EXTRAS} - set(anchor_bkvs))
  ref_bkv = min(1024, kv_cap)
  ladder_bqs = [bq for bq in ladder_bqs if bq >= VPU_LANE and bq % VPU_LANE == 0]

  def outer_candidates() -> list[TileCandidate]:
    return _iperm_outer_candidates(
        q_seq,
        budget=budget,
        max_candidates=max_candidates,
        ladder_bqs=ladder_bqs,
        bkvs=(*anchor_bkvs, min(512, kv_cap)),
    )

  if not iperm_structural_check(q_seq, vmem_bytes=vmem_bytes, margin=margin).fits_resident:
    return outer_candidates()

  single = _ceil_to(q_seq, VPU_LANE)
  scored: list[tuple[float, int]] = []
  for r in range(single, max(single, _floor_to(int(q_seq * (1 + IPERM_MAX_PAD_FRACTION)), VPU_LANE)) + 1, VPU_LANE):
    fits = _fitting_q_subs(r, ref_bkv, budget)
    if fits:
      scored.append((_iperm_cost(r, fits[0], ref_bkv), r))
  scored.sort()
  primary = scored[0][1] if scored else single
  residents = [primary, single]
  residents += [r for _, r in scored[:IPERM_RESIDENT_TOP_K]]
  residents += sorted({_ceil_to(q_seq, bq) for bq in ladder_bqs})
  residents = list(dict.fromkeys(residents))  # dedupe, keep order
  bq_of = {r: _bq_for_resident(q_seq, r, ladder_bqs) for r in residents}

  picked: dict[tuple, tuple[int, float, TileCandidate]] = {}

  def add(
      priority: int,
      resident: int,
      q_sub: Optional[int],
      bkv: int,
      tag: str,
      *,
      formula: bool = True,
      bq: Optional[int] = None,
  ) -> None:
    if q_sub is None or not _q_sub_admissible(resident, q_sub, bkv, formula=formula):
      return
    pred = iperm_vmem_bytes(resident, q_sub, bkv)
    if pred > budget:
      return
    cand = TileCandidate(bq or bq_of[resident], bkv, bkv, q_sub, None, resident, pred, tag)
    key = (cand.bq, cand.bkv, cand.block_q_sub, cand.block_q_outer)
    if key not in picked or priority < picked[key][0]:
      picked[key] = (priority, _iperm_cost(resident, q_sub, bkv), cand)

  def first(*seqs: list[int]) -> Optional[int]:
    return next((seq[0] for seq in seqs if seq), None)

  # Priority 0: anchors, always kept.
  for bkv in anchor_bkvs:
    fits = _fitting_q_subs(primary, bkv, budget)
    add(0, primary, first(fits), bkv, "frontier")
    if bkv == ref_bkv:
      add(0, primary, first(fits[1:]), bkv, "below-frontier")
  add(0, primary, auto_block_q_sub(primary), ref_bkv, "auto-q-sub", formula=False)
  single_fits = _fitting_q_subs(single, ref_bkv, budget)
  single_any = _fitting_q_subs(single, ref_bkv, budget, formula=False)
  # Literally block_q = R (one tile): it is also the cross-attention flash tile,
  # so it is not interchangeable with a ladder bq that pads to the same R.
  add(0, single, first(single_fits, single_any), ref_bkv, "single-tile", formula=False, bq=single)
  for r in residents:
    for q_sub in IPERM_KNOWN_GOOD_Q_SUB:
      add(0, r, q_sub, ref_bkv, "known-good", formula=False)
  safe_budget = IPERM_SAFE_FRACTION * vmem_bytes
  for bkv in dict.fromkeys(b for b in (ref_bkv, 512, 256, VPU_LANE) if b <= kv_cap):
    safe = first(_fitting_q_subs(primary, bkv, safe_budget), _fitting_q_subs(primary, bkv, safe_budget, formula=False))
    if safe is not None:
      add(0, primary, safe, bkv, "safe", formula=False)
      break
  # Priority 1: the frontier at the anchor block_kv for every other R, and the
  # known-good block_q_sub values at the second anchor block_kv.
  for r in residents:
    for bkv in anchor_bkvs:
      add(1, r, first(_fitting_q_subs(r, bkv, budget)), bkv, "frontier")
      if bkv != ref_bkv:
        for q_sub in IPERM_KNOWN_GOOD_Q_SUB:
          add(1, r, q_sub, bkv, "known-good", formula=False)
  # Priority 2: the frontier at the extra block_kv values.
  for r in residents:
    for bkv in extra_bkvs:
      add(2, r, first(_fitting_q_subs(r, bkv, budget)), bkv, "frontier-bkv")

  ranked = sorted(picked.values(), key=lambda t: (t[0], t[1]))
  anchors = [c for p, _, c in ranked if p == 0]
  rest = [c for p, _, c in ranked if p > 0]
  out = anchors + rest[: max(0, max_candidates - len(anchors))]
  if not scored:
    # A resident block fits only with a block_q_sub below the measured regime:
    # also measure the block_q_outer path, which may well be faster.
    out += outer_candidates()
  return out


def _iperm_outer_candidates(
    q_seq: int,
    *,
    budget: float,
    max_candidates: int,
    ladder_bqs: Sequence[int],
    bkvs: Sequence[int],
) -> list[TileCandidate]:
  """block_q_outer < R candidates, for shapes whose resident block can't fit (well).

  For the single-tile R and every ladder R: the two largest block_q_outer
  (divisors of R, at most IPERM_MAX_OUTER_BLOCKS outer blocks) that have an
  admissible, fitting block_q_sub, at each block_kv in `bkvs`, under
  `iperm_outer_vmem_bytes`.
  """
  single = _ceil_to(q_seq, VPU_LANE)
  residents = list(dict.fromkeys([single, *sorted({_ceil_to(q_seq, bq) for bq in ladder_bqs})]))
  scored: list[tuple[float, TileCandidate]] = []
  seen = set()
  for r in residents:
    bq = _bq_for_resident(q_seq, r, ladder_bqs)
    feasible_outers = 0
    for outer in reversed(_divisors_128(r)):
      if outer >= r:
        continue
      blocks = r // outer
      if blocks > IPERM_MAX_OUTER_BLOCKS or feasible_outers >= 2:
        break
      any_fit = False
      for bkv in dict.fromkeys(bkvs):
        fits = [
            q
            for q in reversed(_divisors_128(outer))
            if _q_sub_admissible(outer, q, bkv, formula=True) and iperm_outer_vmem_bytes(outer, q, bkv) <= budget
        ]
        if not fits:
          continue
        any_fit = True
        pred = iperm_outer_vmem_bytes(outer, fits[0], bkv)
        cand = TileCandidate(bq, bkv, bkv, fits[0], outer, outer, pred, "outer")
        key = (cand.bq, cand.bkv, cand.block_q_sub, cand.block_q_outer)
        if key not in seen:
          seen.add(key)
          scored.append((_iperm_cost(r, fits[0], bkv, blocks), cand))
      feasible_outers += int(any_fit)
  scored.sort(key=lambda t: t[0])
  return [c for _, c in scored[:max_candidates]]


# ================================================================================
# Per-model plug: a BlockBenchmark builds a ONE-block model and times a
# forward for a given (bq, bkv).  Only this layer is model-specific; add a model by
# implementing it (WanBlockBenchmark below is the reference).
# ================================================================================
@dataclass
class BenchResult:
  bq: int
  bkv: int
  bkv_compute: int
  status: str  # "ok" | "oom" | "error"
  mean_ms: Optional[float] = None  # steady-state, COMPILE + WARMUP EXCLUDED
  std_ms: Optional[float] = None
  times_ms: list[float] = field(default_factory=list)
  compile_ms: Optional[float] = None  # first-call (compile+first-exec) wall time, reported separately
  detail: str = ""
  # Internal-permutation kernels only (None elsewhere): the in-kernel Q chunk
  # and the outer Q block the candidate ran with -- None block_q_sub is the
  # kernel's auto rule -- plus the planner's metadata (see TileCandidate).
  block_q_sub: Optional[int] = None
  block_q_outer: Optional[int] = None
  resident: Optional[int] = None
  pred_vmem_bytes: Optional[float] = None
  tag: str = ""

  def key(self) -> tuple:
    """Identity of the configuration: every field production is set to."""
    return (self.bq, self.bkv, self.bkv_compute, self.block_q_sub, self.block_q_outer)

  def describe(self) -> str:
    s = f"bq={self.bq} bkv={self.bkv}"
    if self.bkv_compute not in (None, self.bkv):
      s += f" cmp={self.bkv_compute}"
    if self.block_q_sub is not None:
      s += f" q_sub={self.block_q_sub}"
    if self.block_q_outer is not None:
      s += f" q_outer={self.block_q_outer}"
    return s

  def as_candidate(self) -> dict:
    """The candidate dict `apply_candidate_to_config` (and the e2e pass) consume."""
    return {
        "block_q": self.bq,
        "block_kv": self.bkv,
        "block_kv_compute": self.bkv if self.bkv_compute is None else self.bkv_compute,
        "block_q_sub": self.block_q_sub,
        "block_q_outer": self.block_q_outer,
        "proxy_ms": self.mean_ms,
    }

  def csv_row(self) -> dict:
    return {
        "bq": self.bq,
        "bkv": self.bkv,
        "bkv_compute": self.bkv_compute,
        "block_q_sub": self.block_q_sub,
        "block_q_outer": self.block_q_outer,
        "resident": self.resident,
        "pred_vmem_mib": None if self.pred_vmem_bytes is None else round(self.pred_vmem_bytes / (1024 * 1024), 2),
        "tag": self.tag,
        "status": self.status,
        "mean_ms": self.mean_ms,
        "std_ms": self.std_ms,
        "compile_ms": self.compile_ms,
        "detail": self.detail,
    }


CSV_FIELDS = (
    "bq",
    "bkv",
    "bkv_compute",
    "block_q_sub",
    "block_q_outer",
    "resident",
    "pred_vmem_mib",
    "tag",
    "status",
    "mean_ms",
    "std_ms",
    "compile_ms",
    "detail",
)


def time_callable(fn, *, iters: int = 10, warmup: int = 2, sync=lambda x: x):
  """Correct microbenchmark that ALWAYS excludes compilation and warmup from mean_ms.

  Call #1 is executed untimed and absorbs the JIT compile / first-touch cost (returned
  separately as compile_ms).  `warmup-1` further untimed calls reach steady state.  ONLY
  the subsequent `iters` calls are timed -> `mean_ms` never contains compile or warmup.
  `sync(result)` must block until the async result is materialised (jax.block_until_ready);
  it defaults to identity so this stays jax-free and unit-testable.

  Returns (mean_ms, std_ms, times_ms, compile_ms)."""
  import statistics
  import time

  if iters < 1:
    raise ValueError(f"iters must be >= 1, got {iters}.")
  t0 = time.perf_counter()
  sync(fn())  # call #1: compilation happens HERE, untimed
  compile_ms = (time.perf_counter() - t0) * 1e3
  for _ in range(max(0, warmup - 1)):  # extra warmups -> steady state, untimed
    sync(fn())
  times: list[float] = []
  for _ in range(iters):  # the ONLY timed calls
    t = time.perf_counter()
    sync(fn())
    times.append((time.perf_counter() - t) * 1e3)
  mean = sum(times) / len(times)
  std = statistics.pstdev(times) if len(times) > 1 else 0.0
  return mean, std, times, compile_ms


class BlockBenchmark:
  """Interface a model implements so the grid search can drive it.  `run` must build (or
  reuse) a single-block model with the given block sizes, execute a forward `warmup`+`iters`
  times, and return timings.  Catch out-of-VMEM and return status="oom" (don't raise).
  """

  label: str = "block"

  def tiled_seq_lens(self) -> tuple[int, int]:
    """(q_seq, kv_seq) the kernel actually TILES — per-shard, variant-aware (e.g.
    full_seq*U/CP for ring, full_seq for pure ulysses).  The candidate math runs on this.
    """
    raise NotImplementedError

  def vmem_bytes(self) -> int:
    raise NotImplementedError

  def dtype_bytes(self) -> int:
    return 2  # bf16 q/k/v; the score tile is f32 (4) -> see _MEASURED_BKV_CEILING

  def attention_kernel(self) -> str:
    """The attention kernel name the block runs (selects the candidate planner/VMEM model)."""
    return getattr(self, "_attention", "") or ""

  def ring_shards(self) -> Optional[int]:
    """Ring size the kernel runs, or None if unknown (iperm then assumes the resident path)."""
    context_shards = getattr(self, "_context_shards", None)
    if context_shards is None:
      return None
    ulysses_shards = int(getattr(self, "_ulysses_shards", 1) or 1)
    return ring_shards_for(self.attention_kernel(), int(context_shards), ulysses_shards)

  def run(
      self,
      bq: int,
      bkv: int,
      *,
      bkv_compute: Optional[int] = None,
      iters: int = 10,
      warmup: int = 2,
      block_q_sub: Optional[int] = None,
      block_q_outer: Optional[int] = None,
  ) -> BenchResult:
    """Build/reuse the 1-block model with these block sizes and time a forward.  MUST use
    `time_callable` (or equivalent) so `mean_ms` EXCLUDES compilation + warmup; report the
    one-time compile cost in `compile_ms`.  Catch out-of-VMEM -> status='oom' (don't raise).

    `block_q_sub` / `block_q_outer` are only passed (as keywords) for candidates that set
    them -- the internal-permutation kernels' in-kernel Q chunk and outer Q block -- and must
    reach the kernel through the same block-size object production builds
    (`candidate_flash_block_sizes`), never be dropped.
    """
    raise NotImplementedError


# ================================================================================
# Orchestrator
# ================================================================================
@dataclass
class SearchResult:
  best: Optional[BenchResult]
  results: list[BenchResult]
  q_seq: int
  kv_seq: int
  mode: str
  # Resident-Q iperm shapes only: the pre-compilation structural verdict.
  structure: Optional[IpermStructure] = None
  candidates: list[TileCandidate] = field(default_factory=list)
  csv_path: Optional[str] = None

  def ranked(self) -> list[BenchResult]:
    """Successful results, best (the broadcast winner) first, then by proxy time."""
    ok = sorted((r for r in self.results if r.status == "ok" and r.mean_ms is not None), key=lambda r: r.mean_ms)
    if self.best is None:
      return ok
    return [self.best] + [r for r in ok if r.key() != self.best.key()]

  def ranked_candidates(self) -> list[dict]:
    """`ranked()` as candidate dicts (see BenchResult.as_candidate)."""
    return [r.as_candidate() for r in self.ranked()]


_STATUS_TO_CODE = {"ok": 0, "oom": 1, "error": 2}


def _aggregate_process_measurements(
    measurements: np.ndarray,
) -> tuple[str, Optional[float], Optional[float], Optional[float]]:
  """Aggregates candidate measurements, rejecting a candidate that fails on any host."""
  measurements = np.asarray(measurements).reshape((-1, 4))
  status_codes = measurements[:, 0].astype(np.int32)
  if np.any(status_codes == _STATUS_TO_CODE["error"]):
    return "error", None, None, None
  if np.any(status_codes == _STATUS_TO_CODE["oom"]):
    return "oom", None, None, None
  return (
      "ok",
      float(np.max(measurements[:, 1])),
      float(np.max(measurements[:, 2])),
      float(np.max(measurements[:, 3])),
  )


def _aggregate_process_result(result: BenchResult) -> BenchResult:
  if jax.process_count() == 1:
    return result

  local = np.asarray(
      [
          _STATUS_TO_CODE.get(result.status, _STATUS_TO_CODE["error"]),
          result.mean_ms if result.mean_ms is not None else np.inf,
          result.std_ms if result.std_ms is not None else np.inf,
          result.compile_ms if result.compile_ms is not None else np.inf,
      ],
      dtype=np.float32,
  )
  gathered = multihost_utils.process_allgather(local, tiled=False)
  status, mean_ms, std_ms, compile_ms = _aggregate_process_measurements(gathered)
  failed_hosts = int(np.count_nonzero(np.asarray(gathered).reshape((-1, 4))[:, 0]))
  detail = result.detail if status == "ok" else f"{status} on {failed_hosts}/{jax.process_count()} process(es)"
  if status != "ok" and result.detail:
    detail += f"; this process: {result.detail}"
  # `replace` keeps every other field (block_q_sub, block_q_outer, planner metadata).
  return replace(result, status=status, mean_ms=mean_ms, std_ms=std_ms, compile_ms=compile_ms, detail=detail)


def _broadcast_winner(best: Optional[BenchResult], results: list[BenchResult]) -> Optional[BenchResult]:
  if jax.process_count() == 1:
    return best

  # Payload: (found, bq, bkv, bkv_compute, block_q_sub, block_q_outer); None -> -1.
  is_source = jax.process_index() == 0
  payload = np.zeros((6,), dtype=np.int64)
  if is_source and best is not None:
    payload[:] = (1, *(-1 if v is None else int(v) for v in best.key()))
  payload = np.asarray(multihost_utils.broadcast_one_to_all(payload, is_source=is_source))
  if payload[0] == 0:
    return None

  winner_key = tuple(None if int(v) < 0 else int(v) for v in payload[1:])
  for result in results:
    if result.key() == winner_key and result.status == "ok":
      return result
  raise RuntimeError(f"Process 0 selected tile candidate {winner_key}, but it is unavailable on this process.")


def smart_grid(
    q_seq: int,
    kv_seq: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    k_bq: int = 3,
    k_bkv: int = 4,
    spread_bq: int = 2,
    min_bkv_ref: int = 1024,
    family: str = "external",
    align: int = VPU_LANE,
    bkv_anchors: tuple[int, ...] = (1024, 1280),
) -> list[tuple[int, int]]:
  """Nested candidate pairs: BQ = VMEM-capped fewest-tile ladder + spread; then for EACH bq,
  BKV = largest-that-fits at that bq (so bkv is VMEM-correct for its partner, not globally).
  Pairs whose score tile still overflows are OOM-pruned at run time.  cmp is locked = bkv.

  `min_bkv_ref` sets the BQ VMEM cap via the bkv it is expected to pair with.  Use a REALISTIC
  bkv (1024, a good MXU tile) rather than the smallest possible: with a tiny ref the cap is huge,
  so the fewest-tile ladder starts at the single-tile end (which OOMs for a large per-shard seq)
  and the feasible moderate-BQ optimum (e.g. bq=9472 at seq 37800) falls in the ladder's gap.

  Two guaranteed candidates are added on top of the ladders, because the ladders' own bounds
  are calibrated constants that don't track the actual shape being searched, and every prior
  sweep (see report/ and memory) agrees on what they'd otherwise silently exclude:

    * The true single-tile BQ (`ceil_to(q_seq, align)`), UNCAPPED by `bq_cap` above. `bq_cap`
      depends only on `min_bkv_ref`/`vmem_bytes`/family -- never on `q_seq` -- so it is a FIXED
      ceiling regardless of duration/shape. Once a sequence's single-tile size exceeds that
      fixed constant, `bq_candidates` cannot propose it even if it would fit and win (single
      Q-tile is the dominant lever in essentially every measured sweep). OOM-pruning is the
      safety net if it doesn't fit.
    * `bkv_anchors` (1024, 1280) for every bq. `bkv_candidates` only walks down from its own
      `bkv_cap` by `align` for `k_bkv` steps, so it samples a narrow band near the ceiling --
      which is nowhere near 1024 whenever `bkv_cap` sits high (small bq pushes it up). 1024 wins
      in nearly every measured sweep regardless of shape; the ladder can structurally miss it.
  """
  bq_cap = vmem_bq_ceiling(min_bkv_ref, vmem_bytes=vmem_bytes, dtype_bytes=dtype_bytes, family=family)
  bqs = bq_candidates(q_seq, k=k_bq, spread=spread_bq, max_block=bq_cap)
  single_tile_bq = _ceil_to(q_seq, align)
  if single_tile_bq not in bqs:
    bqs = sorted({*bqs, single_tile_bq}, reverse=True)
  pairs: list[tuple[int, int]] = []
  for bq in bqs:
    bkv_cap = vmem_bkv_ceiling(bq, vmem_bytes=vmem_bytes, dtype_bytes=dtype_bytes, family=family)
    bkvs = bkv_candidates(kv_seq, k=k_bkv, max_block=bkv_cap)
    for anchor in bkv_anchors:
      snapped = _floor_to(min(anchor, bkv_cap, _ceil_to(kv_seq, align)), align)
      if snapped >= align and snapped not in bkvs:
        bkvs.append(snapped)
    for bkv in bkvs:
      pairs.append((bq, bkv))
  return pairs


def full_grid(q_seq: int, kv_seq: int, *, step: int = MXU_TILE, max_configs: Optional[int] = None) -> list[tuple[int, int]]:
  """Mode-1 full 2D sweep: every (bq, bkv) in the step-`step` product.  WARNING: this is
  O(N^2) in seq/step (per-shard 9450 -> ~1.4k combos; 75600 -> ~88k).  `max_configs` caps it.
  """
  bqs = full_axis_candidates(q_seq, step=step)
  bkvs = full_axis_candidates(kv_seq, step=step)
  pairs = [(bq, bkv) for bq in bqs for bkv in bkvs]
  if max_configs and len(pairs) > max_configs:
    pairs = pairs[:max_configs]
  return pairs


def plan_candidates(
    bench: BlockBenchmark,
    *,
    mode: str = "smart",
    k: int = 3,
    step: int = MXU_TILE,
    max_configs: Optional[int] = None,
    log=print,
) -> tuple[list[TileCandidate], Optional[IpermStructure], str]:
  """The candidates `grid_search` runs on `bench`, plus the resident-Q structural verdict
  (None unless the iperm resident model governs this shape) and the VMEM family label.

  Iperm kernels with a ring hold the whole padded shard resident, so the per-tile
  (bq, bkv) model that `smart_grid` encodes does not describe them: their candidates come
  from the joint planner (`iperm_candidates`), seeded with `smart_grid`'s bq ladder only
  for the resident lengths it reaches. Every other kernel keeps the (bq, bkv) ladders.
  """
  if mode not in ("smart", "full"):
    raise ValueError(f"mode must be 'smart' or 'full', got {mode!r}")
  q_seq, kv_seq = bench.tiled_seq_lens()
  vmem = bench.vmem_bytes()
  # The VMEM fit is per kernel family (see `_VMEM_FIT`); read the attention name
  # off the bench when it exposes one, else assume the external ring.
  attention = bench.attention_kernel()
  family = vmem_family(attention)
  resident_q = iperm_uses_resident_q(attention, bench.ring_shards())
  structure = iperm_structural_check(q_seq, vmem_bytes=vmem) if resident_q else None
  if mode == "full":
    log(
        "Warning: tile_search mode is 'full', not 'smart' -- this is an exhaustive O(N^2) 2D BQ x BKV"
        " sweep (often hundreds to thousands of configs, each separately compiled) and is meant for"
        " one-off characterization; use mode='smart' for routine tuning."
    )
    if resident_q:
      log(
          "Warning: mode='full' sweeps (block_q, block_kv) only; block_q_sub stays on the iperm "
          "kernel's auto rule (and block_q_outer unset). mode='smart' searches them jointly."
      )
    pairs = full_grid(q_seq, kv_seq, step=step, max_configs=max_configs)
    return [TileCandidate(bq, bkv, bkv) for bq, bkv in pairs], structure, family
  pairs = smart_grid(q_seq, kv_seq, vmem_bytes=vmem, dtype_bytes=4, k_bq=k, k_bkv=max(k, 4), family=family)
  if not resident_q:
    return [TileCandidate(bq, bkv, bkv) for bq, bkv in pairs], None, family
  ladder = sorted({bq for bq, _ in pairs})
  return iperm_candidates(q_seq, kv_seq, vmem_bytes=vmem, ladder_bqs=ladder), structure, "iperm-resident"


def _run_candidate(bench: BlockBenchmark, cand: TileCandidate, *, iters: int, warmup: int) -> BenchResult:
  """Runs one candidate; the knobs it does not set are not passed (benches may not take them)."""
  extra = {}
  if cand.block_q_sub is not None:
    extra["block_q_sub"] = cand.block_q_sub
  if cand.block_q_outer is not None:
    extra["block_q_outer"] = cand.block_q_outer
  r = bench.run(cand.bq, cand.bkv, bkv_compute=cand.bkv_compute, iters=iters, warmup=warmup, **extra)
  return replace(
      r,
      block_q_sub=cand.block_q_sub,
      block_q_outer=cand.block_q_outer,
      resident=cand.resident,
      pred_vmem_bytes=cand.pred_vmem_bytes,
      tag=cand.tag,
  )


def grid_search(
    bench: BlockBenchmark,
    *,
    mode: str = "smart",
    out_dir: Optional[str] = None,
    iters: int = 10,
    warmup: int = 2,
    k: int = 3,
    step: int = MXU_TILE,
    max_configs: Optional[int] = None,
    log=print,
) -> SearchResult:
  """Run the tile-size grid on `bench`, write CSV to `out_dir` (or pretty-print), return the
  winner (lowest mean_ms among status=='ok').  `mode`: 'smart' (candidate ladders) | 'full'.
  """
  q_seq, kv_seq = bench.tiled_seq_lens()
  cands, structure, family = plan_candidates(bench, mode=mode, k=k, step=step, max_configs=max_configs, log=log)
  vmem = bench.vmem_bytes()
  if structure is not None:
    log(f"[tile-search] {bench.label}: {structure.message}")

  log(
      f"[tile-search] {bench.label}: q_seq={q_seq} kv_seq={kv_seq} mode={mode} "
      f"family={family} -> {len(cands)} configs (iters={iters})"
  )
  results: list[BenchResult] = []
  for i, cand in enumerate(cands, 1):
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices(f"tile_search_candidate_{i}_start")
    r = _run_candidate(bench, cand, iters=iters, warmup=warmup)
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices(f"tile_search_candidate_{i}_complete")
      r = _aggregate_process_result(r)
    results.append(r)
    tag = "" if cand.bq % MXU_TILE == 0 and cand.bkv % MXU_TILE == 0 else " [½MXU]"
    if cand.pred_vmem_bytes is not None:
      tag += f" [{cand.tag}, pred {cand.pred_vmem_bytes / vmem:.0%} VMEM]"
    compile_note = f"  (compile {r.compile_ms/1e3:.0f}s, excluded)" if r.compile_ms else ""
    log(
        f"  [{i}/{len(cands)}] {r.describe()}{tag}: "
        + (f"{r.mean_ms:.2f}ms{compile_note}" if r.status == "ok" else f"{r.status} {r.detail}".rstrip())
    )

  ok = [r for r in results if r.status == "ok" and r.mean_ms is not None]
  best = min(ok, key=lambda r: r.mean_ms) if ok and jax.process_index() == 0 else None
  best = _broadcast_winner(best, results)
  csv_path = _emit(results, best, q_seq, kv_seq, mode, out_dir, log)
  return SearchResult(best, results, q_seq, kv_seq, mode, structure=structure, candidates=cands, csv_path=csv_path)


def _emit(results, best, q_seq, kv_seq, mode, out_dir, log) -> Optional[str]:
  path = None
  if out_dir:
    path = os.path.join(out_dir, "tile_size_grid_search.csv")
    if jax.process_index() == 0:
      os.makedirs(out_dir, exist_ok=True)
      with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in sorted(results, key=lambda r: (r.mean_ms is None, r.mean_ms or 0)):
          w.writerow(r.csv_row())
      log(f"[tile-search] wrote {path}")
  else:
    log(f"[tile-search] results (q_seq={q_seq}, kv_seq={kv_seq}, mode={mode}):")
    for r in sorted(results, key=lambda r: (r.mean_ms is None, r.mean_ms or 0)):
      knobs = ""
      if r.block_q_sub is not None or r.block_q_outer is not None:
        knobs = f" q_sub={r.block_q_sub!s:>5} q_outer={r.block_q_outer!s:>5}"
      log(
          f"    bq={r.bq:>6} bkv={r.bkv:>5} cmp={r.bkv_compute!s:>5}{knobs}  "
          + (f"{r.mean_ms:7.2f} ms  (±{r.std_ms:.2f})" if r.status == "ok" else f"  {r.status}")
      )
  if best:
    log(f"[tile-search] WINNER: {best.describe()} bkv_compute={best.bkv_compute} -> {best.mean_ms:.2f} ms")
  else:
    log("[tile-search] no config succeeded (all OOM/error)")
  return path


# --------------------------------------------------------------------------------
# Applying a candidate: ONE path for the bench and for production
# --------------------------------------------------------------------------------
def candidate_flash_block_sizes(
    flash_block_sizes: Mapping[str, Any],
    attention: str,
    cand: Mapping[str, Any],
    *,
    vmem_limit_bytes: Optional[int] = None,
) -> dict:
  """The raw `flash_block_sizes` mapping that runs candidate `cand` (keys as in
  `BenchResult.as_candidate`). Benches build their kernel block sizes from this AND
  production writes exactly this into the config, so both resolve the same object via
  `max_utils.get_flash_block_sizes`."""
  from maxdiffusion import max_utils  # pylint: disable=g-import-not-at-top  # keeps this module light

  return max_utils.flash_block_sizes_for_candidate(
      dict(flash_block_sizes or {}),
      attention,
      int(cand["block_q"]),
      int(cand["block_kv"]),
      cand.get("block_kv_compute"),
      vmem_limit_bytes=vmem_limit_bytes,
      block_q_sub=cand.get("block_q_sub"),
      block_q_outer=cand.get("block_q_outer"),
  )


def resolve_block_sizes(attention: str, flash_block_sizes: Mapping[str, Any]):
  """What production's `max_utils.get_flash_block_sizes` builds from this mapping."""
  from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
  from maxdiffusion import max_utils  # pylint: disable=g-import-not-at-top

  return max_utils.get_flash_block_sizes(SimpleNamespace(attention=attention, flash_block_sizes=flash_block_sizes))


DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def tile_search_vmem_limit_bytes(config) -> int:
  """The scoped-VMEM limit candidates are sized and benchmarked under, and that production is
  then run with (apply_candidate_to_config writes it): `tile_search_vmem_limit_bytes`, else the
  configured `flash_block_sizes.vmem_limit_bytes`, else 64 MiB. A candidate sized to ~98% of
  64 MiB must not ship under Mosaic's (much smaller) default scoped limit."""
  keys = config.get_keys()
  fbs = getattr(config, "flash_block_sizes", None) or {}
  return int(keys.get("tile_search_vmem_limit_bytes") or fbs.get("vmem_limit_bytes") or DEFAULT_VMEM_LIMIT_BYTES)


def apply_candidate_to_config(config, cand: Mapping[str, Any], *, vmem_limit_bytes: Optional[int] = None) -> dict:
  """Writes candidate `cand` into `config.flash_block_sizes` exactly the way
  maybe_tune_block_sizes does, then checks that production's block-size object carries it
  (block_q, block_kv, and for the custom kernels block_q_sub / block_q_outer -- a field the
  carrier does not know is silently dropped, requirement 5). Returns the new mapping.

  `vmem_limit_bytes=None` keeps whatever limit the config already carries.
  """
  fbs = candidate_flash_block_sizes(config.flash_block_sizes, config.attention, cand, vmem_limit_bytes=vmem_limit_bytes)
  config.get_keys()["flash_block_sizes"] = fbs  # config is immutable via setattr; mutate raw dict
  effective = resolve_block_sizes(config.attention, config.flash_block_sizes)
  want = {"block_q": int(cand["block_q"]), "block_kv": int(cand["block_kv"])}
  if "custom" in config.attention:
    want["block_q_sub"] = cand.get("block_q_sub")
    want["block_q_outer"] = cand.get("block_q_outer")
  got = {name: getattr(effective, name, None) for name in want}
  if got != want:
    raise RuntimeError(f"[tile-search] applied candidate {want}, but production block sizes resolved to {got}.")
  return fbs


def format_using_line(cand: Mapping[str, Any]) -> str:
  """The single parseable line announcing the configuration production will run."""
  q_sub = cand.get("block_q_sub")
  q_outer = cand.get("block_q_outer")
  return (
      f"[tile-search] using block_q={int(cand['block_q'])} block_kv={int(cand['block_kv'])} "
      f"block_kv_compute={int(cand['block_kv_compute'])} "
      f"block_q_sub={'auto' if q_sub is None else int(q_sub)} "
      f"block_q_outer={'none' if q_outer is None else int(q_outer)} "
      f"(block-bench {float(cand['proxy_ms']):.2f} ms)"
  )


def tune_block_sizes(
    config,
    bench: BlockBenchmark,
    *,
    vmem_limit_bytes: Optional[int] = None,
    log=print,
) -> list[dict]:
  """maybe_tune_block_sizes' model-agnostic core: proxy-search `bench` with the config's
  tile_search_* settings, apply the best candidate to `config.flash_block_sizes`, print the
  parseable `using` line, and return the proxy-ranked successful candidates, best first
  (dicts as in `BenchResult.as_candidate`).
  """
  keys = config.get_keys()
  result = grid_search(
      bench,
      mode=keys.get("tile_search_mode", "smart"),
      iters=keys.get("tile_search_iters", 10),
      out_dir=(keys.get("tile_search_out", "") or None),
      log=log,
  )
  ranked = result.ranked_candidates()
  if not ranked:
    log("[tile-search] no config succeeded; keeping configured flash_block_sizes")
    return ranked
  apply_candidate_to_config(config, ranked[0], vmem_limit_bytes=vmem_limit_bytes)
  log(format_using_line(ranked[0]))
  return ranked


# --------------------------------------------------------------------------------
# tiny self-demo: `python -m maxdiffusion.utils.tile_size_grid_search 9450`
# --------------------------------------------------------------------------------
def _mxu_tag(b: int) -> str:
  return f"{b // MXU_TILE}x256 packed" if b % MXU_TILE == 0 else f"{b // VPU_LANE}x128 (½ MXU pass)"


def _demo(seq_len: int, _bq: int, vmem_mb: int) -> None:
  vmem = vmem_mb * 1024 * 1024
  print(f"seq_len (per-shard tiled) = {seq_len}   VPU_LANE={VPU_LANE}  MXU_TILE={MXU_TILE}  vmem={vmem_mb}MB")
  pairs = smart_grid(seq_len, seq_len, vmem_bytes=vmem, dtype_bytes=4)
  print(f"\nSMART grid: {len(pairs)} (bq, bkv=cmp) pairs (bkv is largest-fits PER bq):")
  last_bq = None
  for bq, bkv in pairs:
    if bq != last_bq:
      pq = padding_of(seq_len, bq)
      print(f"  bq={bq:>6} ({pq.n_blocks} Q-tile(s), pad {pq.pad_pct:.1f}%, {_mxu_tag(bq)}):")
      last_bq = bq
    pk = padding_of(seq_len, bkv)
    print(f"      bkv={bkv:>5}  {pk.n_blocks} tile(s)  {_mxu_tag(bkv)}  score[{bq},{bkv}]f32={bq*bkv*4/1e6:.0f}MB")
  full = full_axis_candidates(seq_len)
  print(f"\nFULL sweep (step 256): {len(full)}x{len(full)} = {len(full)**2} combos (mode='full')")


if __name__ == "__main__":
  _seq = int(sys.argv[1]) if len(sys.argv) > 1 else 9450
  _bq = int(sys.argv[2]) if len(sys.argv) > 2 else 9472
  _vm = int(sys.argv[3]) if len(sys.argv) > 3 else 64
  _demo(_seq, _bq, _vm)
