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

import time
import unittest

from maxdiffusion.utils.tile_size_grid_search import (
    MXU_TILE,
    VPU_LANE,
    BenchResult,
    BlockBenchmark,
    bkv_candidates,
    bq_candidates,
    grid_search,
    padding_of,
    smart_grid,
    time_callable,
    vmem_bkv_ceiling,
)

# per-shard seq the ring U=1 kernel tiles (75600 / 8); the empirical winner is bq=9472, bkv=1024.
RING_SEQ = 9450
VMEM_64MB = 64 * 1024 * 1024


class _MockRingBench(BlockBenchmark):
  """Synthetic benchmark encoding the measured ring behaviour: fewer Q-tiles is faster, the
  bkv_compute sweet spot is ~1024, odd-128 blocks pay a half-MXU-pass penalty, and a score
  tile that overflows VMEM OOMs.  Lets us test the orchestrator with no TPU."""

  label = "mock-ring"

  def __init__(self, seq=RING_SEQ, vmem=VMEM_64MB):
    self.seq, self._vmem = seq, vmem

  def tiled_seq_lens(self):
    return (self.seq, self.seq)

  def vmem_bytes(self):
    return self._vmem

  def run(self, bq, bkv, *, bkv_compute=None, iters=10, warmup=2):
    cmp = bkv_compute or bkv
    if bq * cmp * 4 + 15e6 > self._vmem:  # score tile f32 + ~15MB ring overhead
      return BenchResult(bq, bkv, cmp, "oom")
    n_q = padding_of(self.seq, bq).n_blocks
    n_kv = padding_of(self.seq, bkv).n_blocks
    ms = 60 + 4.0 * n_q + 0.9 * n_kv - 3.0 * min(cmp, 1024) / 1024
    ms += 1.4 * (bq % MXU_TILE != 0) + 1.4 * (bkv % MXU_TILE != 0)
    return BenchResult(bq, bkv, cmp, "ok", mean_ms=round(ms, 2), std_ms=0.2, compile_ms=25000.0)


class PaddingMathTest(unittest.TestCase):

  def test_padding_of(self):
    p = padding_of(RING_SEQ, 1024)
    self.assertEqual(p.n_blocks, 10)
    self.assertEqual(p.padded_len, 10240)
    self.assertEqual(p.pad, 790)

  def test_single_tile_is_low_pad(self):
    self.assertEqual(padding_of(RING_SEQ, 9472).pad, 22)  # 37 * 256


class CandidateTest(unittest.TestCase):

  def test_bq_fewest_tile_ladder(self):
    # single-block ceiling fits, so the ladder starts at n=1 (bq=9472) and includes the winner.
    bqs = bq_candidates(RING_SEQ, k=3, spread=0)
    self.assertEqual(bqs[0], 9472)
    self.assertTrue(all(b % VPU_LANE == 0 for b in bqs))

  def test_bkv_largest_fits_includes_winner(self):
    ceil = vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB)
    bkvs = bkv_candidates(RING_SEQ, k=3, max_block=ceil)
    self.assertIn(1024, bkvs)  # the measured optimum, largest 256-mult that fits at bq=9472

  def test_smart_grid_pairs_winner(self):
    pairs = smart_grid(RING_SEQ, RING_SEQ, vmem_bytes=VMEM_64MB, dtype_bytes=4)
    self.assertIn((9472, 1024), pairs)

  def test_candidates_not_strictly_256(self):
    # 128-multiples (e.g. 896) must be admissible, not filtered out.
    bkvs = bkv_candidates(RING_SEQ, k=3, max_block=vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB))
    self.assertTrue(any(b % MXU_TILE != 0 for b in bkvs))


class TimingTest(unittest.TestCase):

  def test_compile_excluded_from_mean(self):
    calls = {"n": 0}

    def fake_fn():
      calls["n"] += 1
      time.sleep(0.20 if calls["n"] == 1 else 0.01)  # call #1 = "compile"
      return calls["n"]

    mean, _, times, compile_ms = time_callable(fake_fn, iters=5, warmup=2)
    self.assertGreater(compile_ms, 150.0)  # the 200ms first call is captured here...
    self.assertLess(mean, 30.0)  # ...and NOT in the steady-state mean
    self.assertEqual(len(times), 5)


class OrchestratorTest(unittest.TestCase):

  def test_smart_search_picks_measured_winner(self):
    res = grid_search(_MockRingBench(), mode="smart", iters=10, log=lambda *a, **k: None)
    self.assertIsNotNone(res.best)
    self.assertEqual((res.best.bq, res.best.bkv), (9472, 1024))

  def test_oom_configs_pruned_not_raised(self):
    # a tiny VMEM budget OOMs the big pairs; search must still return (or None), never raise.
    res = grid_search(
        _MockRingBench(vmem=8 * 1024 * 1024),
        mode="smart",
        iters=2,
        log=lambda *a, **k: None,
    )
    self.assertTrue(any(r.status == "oom" for r in res.results))


if __name__ == "__main__":
  unittest.main()
