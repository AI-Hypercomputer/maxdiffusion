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

from dataclasses import dataclass, field
from typing import Optional
import csv
import os
import sys
import jax

# --- the two granularities (see module docstring) --------------------------------
VPU_LANE = 128  # kernel hard floor: block sizes must be multiples of this
MXU_TILE = 256  # 256x256 MXU: multiples of this fully pack the systolic array


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _ceil_to(x: int, m: int) -> int:
  return _ceil_div(x, m) * m


def _floor_to(x: int, m: int) -> int:
  return (x // m) * m


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


def vmem_bq_ceiling(
    bkv: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    score_fraction: float = 0.65,
    align: int = VPU_LANE,
) -> int:
  """Approx largest bq whose score tile [bq, bkv]*dtype_bytes fits VMEM (mirror of
  `vmem_bkv_ceiling`).  Use a SMALL reference bkv so the fewest-tile BQ ladder isn't
  over-constrained; big-bq x big-bkv corners are OOM-pruned by the orchestrator."""
  budget = int(vmem_bytes * score_fraction)
  return max(align, _floor_to(budget // (bkv * dtype_bytes), align))


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


def vmem_bkv_ceiling(
    bq: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    score_fraction: float = 0.65,
    align: int = MXU_TILE,
) -> int:
  """Approx largest bkv_compute whose score tile [bq, bkv]*dtype_bytes fits the fraction
  of VMEM left after the kernel's other resident tiles (ring fp32 residual windows, K/V,
  Q, accumulators).  APPROXIMATE — OOM-pruning in the orchestrator is the real guard.
  Default 0.65 fits the ulysses_ring_custom kernel (measured: bkv 1152 fits, 1280 OOMs
  at bq=9472 / 64 MB)."""
  budget = int(vmem_bytes * score_fraction)
  return max(align, _floor_to(budget // (bq * dtype_bytes), align))


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

  def csv_row(self) -> dict:
    return {
        "bq": self.bq,
        "bkv": self.bkv,
        "bkv_compute": self.bkv_compute,
        "status": self.status,
        "mean_ms": self.mean_ms,
        "std_ms": self.std_ms,
        "compile_ms": self.compile_ms,
        "detail": self.detail,
    }


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
    return 2  # bf16 q/k/v; score tile is f32 (4) -> handled by score_fraction tuning

  def run(
      self,
      bq: int,
      bkv: int,
      *,
      bkv_compute: Optional[int] = None,
      iters: int = 10,
      warmup: int = 2,
  ) -> BenchResult:
    """Build/reuse the 1-block model with these block sizes and time a forward.  MUST use
    `time_callable` (or equivalent) so `mean_ms` EXCLUDES compilation + warmup; report the
    one-time compile cost in `compile_ms`.  Catch out-of-VMEM -> status='oom' (don't raise).
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


def smart_grid(
    q_seq: int,
    kv_seq: int,
    *,
    vmem_bytes: int,
    dtype_bytes: int = 4,
    k_bq: int = 3,
    k_bkv: int = 3,
    spread_bq: int = 2,
    score_fraction: float = 0.65,
    min_bkv_ref: int = 1024,
) -> list[tuple[int, int]]:
  """Nested candidate pairs: BQ = VMEM-capped fewest-tile ladder + spread; then for EACH bq,
  BKV = largest-that-fits at that bq (so bkv is VMEM-correct for its partner, not globally).
  Pairs whose score tile still overflows are OOM-pruned at run time.  cmp is locked = bkv.

  `min_bkv_ref` sets the BQ VMEM cap via the bkv it is expected to pair with.  Use a REALISTIC
  bkv (1024, a good MXU tile) rather than the smallest possible: with a tiny ref the cap is huge,
  so the fewest-tile ladder starts at the single-tile end (which OOMs for a large per-shard seq)
  and the feasible moderate-BQ optimum (e.g. bq=9472 at seq 37800) falls in the ladder's gap.
  """
  bq_cap = vmem_bq_ceiling(
      min_bkv_ref,
      vmem_bytes=vmem_bytes,
      dtype_bytes=dtype_bytes,
      score_fraction=score_fraction,
  )
  bqs = bq_candidates(q_seq, k=k_bq, spread=spread_bq, max_block=bq_cap)
  pairs: list[tuple[int, int]] = []
  for bq in bqs:
    bkv_cap = vmem_bkv_ceiling(
        bq,
        vmem_bytes=vmem_bytes,
        dtype_bytes=dtype_bytes,
        score_fraction=score_fraction,
    )
    for bkv in bkv_candidates(kv_seq, k=k_bkv, max_block=bkv_cap):
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
  if mode == "smart":
    pairs = smart_grid(q_seq, kv_seq, vmem_bytes=bench.vmem_bytes(), dtype_bytes=4, k_bq=k, k_bkv=k)
  elif mode == "full":
    log(
        "Warning: tile_search mode is 'full', not 'smart' -- this is an exhaustive O(N^2) 2D BQ x BKV"
        " sweep (often hundreds to thousands of configs, each separately compiled) and is meant for"
        " one-off characterization; use mode='smart' for routine tuning."
    )
    pairs = full_grid(q_seq, kv_seq, step=step, max_configs=max_configs)
  else:
    raise ValueError(f"mode must be 'smart' or 'full', got {mode!r}")

  log(f"[tile-search] {bench.label}: q_seq={q_seq} kv_seq={kv_seq} mode={mode} " f"-> {len(pairs)} configs (iters={iters})")
  results: list[BenchResult] = []
  for i, (bq, bkv) in enumerate(pairs, 1):
    r = bench.run(bq, bkv, bkv_compute=bkv, iters=iters, warmup=warmup)
    results.append(r)
    tag = "" if bq % MXU_TILE == 0 and bkv % MXU_TILE == 0 else " [½MXU]"
    compile_note = f"  (compile {r.compile_ms/1e3:.0f}s, excluded)" if r.compile_ms else ""
    log(
        f"  [{i}/{len(pairs)}] bq={bq} bkv={bkv}{tag}: "
        + (f"{r.mean_ms:.2f}ms{compile_note}" if r.status == "ok" else r.status)
    )

  ok = [r for r in results if r.status == "ok" and r.mean_ms is not None]
  best = min(ok, key=lambda r: r.mean_ms) if ok else None
  _emit(results, best, q_seq, kv_seq, mode, out_dir, log)
  return SearchResult(best, results, q_seq, kv_seq, mode)


def _emit(results, best, q_seq, kv_seq, mode, out_dir, log) -> None:
  if out_dir:
    if jax.process_index() == 0:
      os.makedirs(out_dir, exist_ok=True)
      path = os.path.join(out_dir, "tile_size_grid_search.csv")
      with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bq",
                "bkv",
                "bkv_compute",
                "status",
                "mean_ms",
                "std_ms",
                "compile_ms",
                "detail",
            ],
        )
        w.writeheader()
        for r in sorted(results, key=lambda r: (r.mean_ms is None, r.mean_ms or 0)):
          w.writerow(r.csv_row())
      log(f"[tile-search] wrote {path}")
  else:
    log(f"[tile-search] results (q_seq={q_seq}, kv_seq={kv_seq}, mode={mode}):")
    for r in sorted(results, key=lambda r: (r.mean_ms is None, r.mean_ms or 0)):
      log(
          f"    bq={r.bq:>6} bkv={r.bkv:>5} cmp={r.bkv_compute:>5}  "
          + (f"{r.mean_ms:7.2f} ms  (±{r.std_ms:.2f})" if r.status == "ok" else f"  {r.status}")
      )
  if best:
    log(f"[tile-search] WINNER: bq={best.bq} bkv={best.bkv} bkv_compute={best.bkv_compute} " f"-> {best.mean_ms:.2f} ms")
  else:
    log("[tile-search] no config succeeded (all OOM/error)")


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
