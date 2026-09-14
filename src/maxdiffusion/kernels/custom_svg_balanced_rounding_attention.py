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

Budget-preserving tile-native boundary rounding for SVG attention.

FULL tiles retain exact SVG semantics. Each exact SVG BOUNDARY tile is rounded
UP (execute the full real rectangle with padding-only masking) or DOWN (omit
it). The host policy chooses UP tiles so rounded pair work matches the exact
boundary-pair budget as closely as possible.

Production execution then unions FULL tiles with selected rounded BOUNDARY
tiles. Tiles fully inside the physical sequence run together through one
mask-free Pallas partial; only tiles touching the Q/KV sequence tail use the
padding-only cleanup partial. This keeps the hot path uniform and avoids a
FULL-vs-BOUNDARY decision inside the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels import custom_svg_balanced_rounding_partial as partial_impl
from maxdiffusion.kernels import custom_svg_static_range_attention as exact_svg


@dataclass(frozen=True)
class BoundaryTileStat:
  qi: int
  kj: int
  real_pairs: int
  exact_pairs: int

  @property
  def alpha(self):
    return float(self.exact_pairs) / max(float(self.real_pairs), 1.0)


def _pack_rows(rows, qtiles):
  width = max(1, max((len(r) for r in rows), default=0))
  table = np.zeros((qtiles, width), dtype=np.int32)
  active = np.zeros((qtiles,), dtype=np.int32)
  for qi, row in enumerate(rows):
    if row:
      table[qi, : len(row)] = np.asarray(row, np.int32)
      active[qi] = len(row)
  return table, active


def _exact_pairs_for_tile(
    *,
    qi,
    kj,
    q_seq_len,
    kv_seq_len,
    bq,
    bkv,
    band_width,
    frame_size,
    include_first_frame,
):
  q0, k0 = qi * bq, kj * bkv
  q1, k1 = min(q_seq_len, q0 + bq), min(kv_seq_len, k0 + bkv)
  if q1 <= q0 or k1 <= k0:
    return 0, 0

  qs = np.arange(q0, q1, dtype=np.int64)
  lo = np.maximum(k0, qs - int(band_width))
  hi = np.minimum(k1 - 1, qs + int(band_width))
  local = np.maximum(0, hi - lo + 1)
  if include_first_frame:
    sink_hi = min(k1 - 1, int(frame_size) - 1)
    sink_count = max(0, sink_hi - k0 + 1)
    if sink_count:
      overlap = np.maximum(
          0,
          np.minimum(hi, sink_hi) - np.maximum(lo, k0) + 1,
      )
      exact = local + sink_count - overlap
    else:
      exact = local
  else:
    exact = local
  return int((q1 - q0) * (k1 - k0)), int(exact.sum())


def build_boundary_stats(
    *,
    orig_q_seq_len,
    orig_kv_seq_len,
    block_sizes,
    band_width,
    frame_size,
    include_first_frame=True,
):
  bq, bkv = int(block_sizes.block_q), int(block_sizes.block_kv)
  fm, fa, bm, ba = exact_svg._classify_tiles(
      int(orig_q_seq_len),
      int(orig_kv_seq_len),
      bq,
      bkv,
      int(band_width),
      int(frame_size),
      bool(include_first_frame),
  )
  stats = []
  for qi in range(bm.shape[0]):
    for slot in range(int(ba[qi])):
      kj = int(bm[qi, slot])
      rp, ep = _exact_pairs_for_tile(
          qi=qi,
          kj=kj,
          q_seq_len=int(orig_q_seq_len),
          kv_seq_len=int(orig_kv_seq_len),
          bq=bq,
          bkv=bkv,
          band_width=int(band_width),
          frame_size=int(frame_size),
          include_first_frame=bool(include_first_frame),
      )
      stats.append(BoundaryTileStat(qi, kj, rp, ep))
  return fm, fa, bm, ba, stats


def _closest_prefix(items, target):
  ordered = sorted(
      items,
      key=lambda x: (-x.alpha, -x.exact_pairs, x.qi, x.kj),
  )
  best_k, best_err, running = 0, abs(float(target)), 0
  for k, tile in enumerate(ordered, 1):
    running += tile.real_pairs
    err = abs(float(running) - float(target))
    if err < best_err:
      best_k, best_err = k, err
  return {(x.qi, x.kj) for x in ordered[:best_k]}


def select_boundary_tiles(stats, *, policy="global_balanced", budget_scale=1.0):
  policy = str(policy).lower().strip()
  if policy == "up":
    return {(x.qi, x.kj) for x in stats}
  if policy == "down":
    return set()
  if policy == "nearest":
    return {(x.qi, x.kj) for x in stats if x.alpha >= 0.5}

  target = float(budget_scale) * sum(x.exact_pairs for x in stats)
  if policy == "global_balanced":
    return _closest_prefix(stats, target)
  if policy == "row_balanced":
    out = set()
    for qi in sorted({x.qi for x in stats}):
      row = [x for x in stats if x.qi == qi]
      out |= _closest_prefix(
          row,
          float(budget_scale) * sum(x.exact_pairs for x in row),
      )
    return out
  raise ValueError(f"unknown SVG boundary rounding policy {policy!r}")


def build_selected_boundary_table(
    *,
    stats,
    qtiles,
    policy="global_balanced",
    budget_scale=1.0,
):
  selected = select_boundary_tiles(
      stats,
      policy=policy,
      budget_scale=budget_scale,
  )
  rows = [[] for _ in range(qtiles)]
  for tile in stats:
    if (tile.qi, tile.kj) in selected:
      rows[tile.qi].append(tile.kj)
  for row in rows:
    row.sort()

  table, active = _pack_rows(rows, qtiles)
  exact = int(sum(x.exact_pairs for x in stats))
  rounded = int(sum(x.real_pairs for x in stats if (x.qi, x.kj) in selected))
  retained = int(sum(x.exact_pairs for x in stats if (x.qi, x.kj) in selected))
  target = float(budget_scale) * exact
  report = {
      "policy": policy,
      "budget_scale": float(budget_scale),
      "boundary_tiles_total": len(stats),
      "boundary_tiles_selected": len(selected),
      "exact_boundary_pairs": exact,
      "rounded_boundary_pairs": rounded,
      "target_boundary_pairs": target,
      "budget_error_pairs": rounded - target,
      "budget_error_fraction": (rounded - target) / max(target, 1.0),
      "retained_exact_pairs": retained,
      "dropped_exact_pairs": exact - retained,
      "added_outside_pairs": rounded - retained,
      "boundary_exact_recall": retained / max(exact, 1),
      "rounded_precision": retained / max(rounded, 1),
  }
  return table, active, report


def build_union_tail_tables(
    *,
    full_table,
    full_active,
    selected_boundary_table,
    selected_boundary_active,
    orig_q_seq_len,
    orig_kv_seq_len,
    block_sizes,
):
  """Split the rounded physical tile set into mask-free main and tail tables.

  FULL and selected rounded BOUNDARY tiles are semantically identical once the
  boundary decision has been made: both execute the whole real rectangle. They
  therefore share one mask-free table whenever the whole hardware tile lies
  inside the physical Q/KV sequence. Only tiles touching sequence padding are
  sent to the padding-aware cleanup table.
  """
  qtiles = full_table.shape[0]
  bq = int(block_sizes.block_q)
  bkv = int(block_sizes.block_kv)
  main_rows = [[] for _ in range(qtiles)]
  tail_rows = [[] for _ in range(qtiles)]

  for qi in range(qtiles):
    row = {
        *(int(x) for x in full_table[qi, : int(full_active[qi])]),
        *(int(x) for x in selected_boundary_table[qi, : int(selected_boundary_active[qi])]),
    }
    q_full = (qi + 1) * bq <= int(orig_q_seq_len)
    for kj in sorted(row):
      k_full = (kj + 1) * bkv <= int(orig_kv_seq_len)
      (main_rows if q_full and k_full else tail_rows)[qi].append(kj)

  main_table, main_active = _pack_rows(main_rows, qtiles)
  tail_table, tail_active = _pack_rows(tail_rows, qtiles)
  return main_table, main_active, tail_table, tail_active


def make_svg_balanced_rounding_mha(
    *,
    policy="global_balanced",
    budget_scale=1.0,
    block_sizes,
    orig_q_seq_len,
    orig_kv_seq_len,
    band_width,
    frame_size,
    include_first_frame=True,
    bkv_compute_in=None,
    use_base2_exp=True,
    use_experimental_scheduler=False,
    vmem_limit_bytes=None,
):
  fm, fa, bm, ba, stats = build_boundary_stats(
      orig_q_seq_len=orig_q_seq_len,
      orig_kv_seq_len=orig_kv_seq_len,
      block_sizes=block_sizes,
      band_width=band_width,
      frame_size=frame_size,
      include_first_frame=include_first_frame,
  )
  sm, sa, budget = build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy=policy,
      budget_scale=budget_scale,
  )
  mm, ma, tm, ta = build_union_tail_tables(
      full_table=fm,
      full_active=fa,
      selected_boundary_table=sm,
      selected_boundary_active=sa,
      orig_q_seq_len=orig_q_seq_len,
      orig_kv_seq_len=orig_kv_seq_len,
      block_sizes=block_sizes,
  )
  common = {
      "block_sizes": block_sizes,
      "orig_q_seq_len": orig_q_seq_len,
      "orig_kv_seq_len": orig_kv_seq_len,
      "band_width": band_width,
      "frame_size": frame_size,
      "include_first_frame": include_first_frame,
      "bkv_compute_in": bkv_compute_in,
      "use_base2_exp": use_base2_exp,
      "use_experimental_scheduler": use_experimental_scheduler,
      "vmem_limit_bytes": vmem_limit_bytes,
  }
  main = exact_svg._make_partial(
      table_np=mm,
      active_np=ma,
      mask_mode="none",
      **common,
  )
  tail = partial_impl.make_padding_partial_from_table(
      table_np=tm,
      active_np=ta,
      **common,
  )
  tail_tiles = int(ta.sum())

  def attention(q, k, v):
    with jax.named_scope("svg_union_main"):
      om, lm = main(q, k, v)
    if tail_tiles == 0:
      return om
    with jax.named_scope("svg_tail_cleanup"):
      ot, lt = tail(q, k, v)
    with jax.named_scope("svg_lse_merge"):
      m = jnp.maximum(lm, lt)
      exp = jnp.exp2 if use_base2_exp else jnp.exp
      wm, wt = exp(lm - m), exp(lt - m)
      den = wm + wt
      return (om.astype(jnp.float32) * (wm / den)[:, None, :] + ot.astype(jnp.float32) * (wt / den)[:, None, :]).astype(
          q.dtype
      )

  attention.full_tiles = int(fa.sum())
  attention.original_boundary_tiles = int(ba.sum())
  attention.selected_boundary_tiles = int(sa.sum())
  attention.union_main_tiles = int(ma.sum())
  attention.tail_cleanup_tiles = tail_tiles
  attention.rounding_budget = budget
  attention.rounding_policy = policy
  attention.rounding_budget_scale = float(budget_scale)
  return attention
