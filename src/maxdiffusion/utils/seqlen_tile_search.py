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

End-to-end self-attention tile search for the external-ring fixed-m kernel.

Searches (block_q, block_kv) jointly; block_kv_compute = block_kv and
block_kv_compute_in = min(1024, block_kv). Candidates are timed on full model
steps rather than a one-block proxy, which mis-ranks fixed-m tiles by 1-5% end
to end.

  stage 1 (screen):  every candidate, SCREEN_STEPS timed steps. A candidate whose
                     first timed step is ABORT_RATIO x slower than the best so far
                     stops there.
  stage 2 (confirm): the TOP_K screened tiles, CONFIRM_STEPS timed steps each;
                     the fastest median wins.
"""

import math
from typing import Callable, NamedTuple, Optional

ALIGN = 256  # MXU tile; every candidate block is a multiple of it
MIN_BQ = 2048
# Searched first so early abort has a good reference: 4096x2048 was near the best at
# every measured length; 6400x2048 is the production tile for 75,600 tokens on DP2xCP4.
ANCHORS = ((4096, 2048), (6400, 2048))
BKV_LADDER = (2048, 1536, 1024, 1280, 2560, 3072)
EXTRA_SPLITS = {2048: 2, 1536: 1, 1024: 1}  # tiles beyond the fewest-tile split, per block_kv
SCREEN_STEPS = 2
CONFIRM_STEPS = 6
TOP_K = 3
ABORT_RATIO = 1.15

# Largest block_kv that compiles at each block_q under a 64 MiB VMEM limit (external-ring
# kernels, minimum across configs). Unswept block_q values use the next swept block_q
# above them, which is conservative since the ceiling only falls as block_q grows.
_BKV_CEILING_64MIB = {
    1024: 4096, 1280: 4096, 1536: 4096, 1792: 4096, 2048: 4096, 2304: 4096,
    2560: 4096, 2816: 4096, 3072: 4096, 3328: 4096, 3584: 3968, 3840: 3840,
    4096: 3328, 4352: 3200, 4608: 2944, 4864: 2816, 5376: 2560, 5632: 2304,
    6144: 2048, 6400: 2048, 7168: 1664, 7680: 1536, 7936: 1536, 8192: 1408,
    8448: 1408, 8704: 1280, 8960: 1280, 9472: 1152, 10240: 1024, 11008: 896,
    11264: 896, 12032: 768, 12800: 768, 14080: 640, 14336: 512, 15360: 512,
    15872: 512, 17152: 384, 18944: 256, 22272: 256, 25344: 256,
}  # fmt: skip
_SWEPT_BQ = sorted(_BKV_CEILING_64MIB)


class Tile(NamedTuple):
  block_q: int
  block_kv: int

  @property
  def name(self) -> str:
    return f"t{self.block_q}x{self.block_kv}"


def bkv_ceiling(block_q: int) -> int:
  """Largest block_kv that fits VMEM at this block_q (0 if none was measured)."""
  for swept in _SWEPT_BQ:
    if swept >= block_q:
      return _BKV_CEILING_64MIB[swept]
  return 0


def max_block_q(block_kv: int) -> int:
  """Largest ALIGN-multiple block_q whose ceiling admits block_kv."""
  fits = [bq for bq in range(ALIGN, _SWEPT_BQ[-1] + 1, ALIGN) if bkv_ceiling(bq) >= block_kv]
  return max(fits) if fits else 0


def block_for_count(seq_len: int, n: int) -> Optional[int]:
  """Smallest ALIGN-multiple block that tiles seq_len into exactly n blocks, else None."""
  b = ALIGN * math.ceil(seq_len / (ALIGN * n))
  return b if math.ceil(seq_len / b) == n else None


def tile_candidates(chunk_len: int) -> list:
  """(block_q, block_kv) candidates for one ring chunk, ANCHORS first.

  Per block_kv in BKV_LADDER: the even split into the fewest block_q tiles that fit
  VMEM, plus EXTRA_SPLITS more tiles. When the chunk is ALIGN-aligned, also the
  largest exact divisor that fits (zero padding) at block_kv 2048/1536/1024 -- the
  tiles an aligned bucket length makes possible.
  """
  out = []

  def add(bq, bkv):
    if bq and MIN_BQ <= bq <= max_block_q(bkv) and Tile(bq, bkv) not in out:
      out.append(Tile(bq, bkv))

  for bq, bkv in ANCHORS:
    add(bq, bkv)
  for bkv in BKV_LADDER:
    n = math.ceil(chunk_len / max_block_q(bkv))
    for extra in range(EXTRA_SPLITS.get(bkv, 0) + 1):
      add(block_for_count(chunk_len, n + extra), bkv)
  if chunk_len % ALIGN == 0:
    for bkv in (2048, 1536, 1024):
      divisors = [d for d in range(MIN_BQ, max_block_q(bkv) + 1, ALIGN) if chunk_len % d == 0]
      if divisors:
        add(max(divisors), bkv)
  return out


def parse_tile_suffix(name: str):
  """'<case>_t<bq>x<bkv>' -> (case, Tile); '<case>_bq<N>' -> (case, Tile(N, None)); else (name, None)."""
  case, sep, tail = name.rpartition("_t")
  if sep and "x" in tail:
    bq, _, bkv = tail.partition("x")
    if bq.isdigit() and bkv.isdigit():
      return case, Tile(int(bq), int(bkv))
  case, sep, tail = name.rpartition("_bq")
  if sep and tail.isdigit():
    return case, Tile(int(tail), None)
  return name, None


Measure = Callable[[Tile, int, Optional[float]], dict]


def _ok(rec: dict) -> bool:
  return "error" not in rec and not rec.get("aborted") and rec.get("step_median_s") is not None


def run_search(
    candidates,
    measure: Measure,
    *,
    screen_steps: int = SCREEN_STEPS,
    confirm_steps: int = CONFIRM_STEPS,
    top_k: int = TOP_K,
    abort_ratio: float = ABORT_RATIO,
) -> dict:
  """Two-stage search. `measure(tile, timed_steps, abort_above_s)` returns a speed record
  ({"step_median_s", ...}, {"aborted": True, ...} or {"error": ...}); it may return a
  cached record (resume). Returns {"winner", "screen", "confirm"} keyed by Tile."""
  screen, best = {}, math.inf
  for tile in candidates:
    rec = measure(tile, screen_steps, best * abort_ratio if math.isfinite(best) else None)
    screen[tile] = rec
    if _ok(rec):
      best = min(best, rec["step_median_s"])
  ranked = sorted((t for t, r in screen.items() if _ok(r)), key=lambda t: screen[t]["step_median_s"])
  confirm = {tile: measure(tile, confirm_steps, None) for tile in ranked[:top_k]}
  done = [t for t, r in confirm.items() if _ok(r)]
  winner = min(done, key=lambda t: confirm[t]["step_median_s"]) if done else None
  return {"winner": winner, "screen": screen, "confirm": confirm}
