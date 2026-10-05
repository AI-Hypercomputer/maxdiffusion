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
import numpy as np

from maxdiffusion.utils.tile_size_grid_search import (
    MXU_TILE,
    VPU_LANE,
    IPERM_VMEM_MARGIN,
    _aggregate_process_measurements,
    BenchResult,
    BlockBenchmark,
    auto_block_q_sub,
    bkv_candidates,
    bq_candidates,
    grid_search,
    iperm_candidates,
    iperm_outer_vmem_bytes,
    iperm_structural_check,
    iperm_vmem_bytes,
    local_tiled_seq_len,
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

  def test_bkv_ceiling_is_per_family_and_measured(self):
    # Measured at 64 MB, bq=9472: external tops out at 1152, internal at 1408.
    # The old single-fraction model capped BOTH at 1024, hiding the internal
    # kernel's optimum (9472/1408) -- worth 3-12% e2e. See report/internal_perm.
    self.assertEqual(vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB, family="external"), 1152)
    self.assertEqual(vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB, family="internal"), 1408)
    self.assertGreater(
        vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB, family="internal"),
        vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB, family="external"),
    )

  def test_bkv_largest_fits_includes_winner(self):
    ceil = vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB)
    bkvs = bkv_candidates(RING_SEQ, k=4, max_block=ceil)
    self.assertIn(1024, bkvs)  # still offered: 1024 beats 1152 in all 13 measured configs

  def test_smart_grid_pairs_winner(self):
    pairs = smart_grid(RING_SEQ, RING_SEQ, vmem_bytes=VMEM_64MB, dtype_bytes=4)
    self.assertIn((9472, 1024), pairs)

  def test_smart_grid_always_offers_single_tile_bq(self):
    # bq_cap in smart_grid is a FIXED constant (from min_bkv_ref/vmem_bytes/family only --
    # never from q_seq), so for a long enough sequence the ladder alone would never propose
    # the single-tile bq even though it's the dominant lever in every measured sweep. A
    # duration well beyond any prior sweep (e.g. a long video) must still get it offered.
    long_seq = 30000
    pairs = smart_grid(long_seq, RING_SEQ, vmem_bytes=VMEM_64MB, dtype_bytes=4, family="internal")
    bqs = {bq for bq, _ in pairs}
    single_tile = -(-long_seq // VPU_LANE) * VPU_LANE
    self.assertIn(single_tile, bqs)

  def test_smart_grid_always_offers_bkv_anchor(self):
    # bkv_candidates only walks down from its own ceiling by align*k_bkv steps, so once a
    # small bq pushes that ceiling well above 1024, the ladder alone can't reach 1024 even
    # though it wins in nearly every measured sweep regardless of shape. The smallest bq
    # the ladder proposes has the largest bkv_cap, so it's the sharpest case.
    pairs = smart_grid(RING_SEQ, RING_SEQ, vmem_bytes=VMEM_64MB, dtype_bytes=4)
    smallest_bq = min(bq for bq, _ in pairs)
    bkvs_at_smallest_bq = {bkv for bq, bkv in pairs if bq == smallest_bq}
    self.assertIn(1024, bkvs_at_smallest_bq)

  def test_candidates_not_strictly_256(self):
    # 128-multiples (e.g. 896) must be admissible, not filtered out.
    bkvs = bkv_candidates(RING_SEQ, k=3, max_block=vmem_bkv_ceiling(9472, vmem_bytes=VMEM_64MB))
    self.assertTrue(any(b % MXU_TILE != 0 for b in bkvs))

  def test_ring_sequence_length_uses_actual_ring_shards(self):
    self.assertEqual(local_tiled_seq_len(6144, "tokamax_ring", context_shards=8, ulysses_shards=-1), 768)
    self.assertEqual(local_tiled_seq_len(6144, "ulysses_ring", context_shards=8, ulysses_shards=2), 1536)
    self.assertEqual(local_tiled_seq_len(6144, "flash", context_shards=8, ulysses_shards=-1), 6144)

  def test_ring_sequence_length_matches_production_padding(self):
    self.assertEqual(local_tiled_seq_len(10, "tokamax_ring", context_shards=4, ulysses_shards=-1), 3)
    self.assertEqual(local_tiled_seq_len(10, "ulysses_ring", context_shards=4, ulysses_shards=2), 6)

  def test_hybrid_ring_requires_compatible_shards(self):
    with self.assertRaisesRegex(ValueError, "must be divisible"):
      local_tiled_seq_len(6144, "ulysses_ring_custom", context_shards=8, ulysses_shards=3)


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

  def test_rejects_nonpositive_iters(self):
    with self.assertRaises(ValueError):
      time_callable(lambda: 1, iters=0)


class OrchestratorTest(unittest.TestCase):

  def test_process_measurements_use_slowest_successful_host(self):
    status, mean_ms, std_ms, compile_ms = _aggregate_process_measurements(
        np.asarray([
            [0, 10.0, 0.5, 100.0],
            [0, 12.0, 0.7, 120.0],
        ])
    )
    self.assertEqual(status, "ok")
    self.assertEqual((mean_ms, std_ms, compile_ms), (12.0, 0.7, 120.0))

  def test_process_measurements_reject_candidate_if_any_host_fails(self):
    status, mean_ms, _, _ = _aggregate_process_measurements(
        np.asarray([
            [0, 10.0, 0.5, 100.0],
            [1, np.inf, np.inf, np.inf],
        ])
    )
    self.assertEqual(status, "oom")
    self.assertIsNone(mean_ms)

  def test_smart_search_picks_measured_winner(self):
    # Expectation updated when the VMEM ceiling was corrected. The mock's cost
    # model rewards fewer KV blocks (0.9 * n_kv) and caps the compute bonus at
    # min(cmp, 1024), so 1280 genuinely beats 1024 under its own physics -- the
    # old assertion only held because the 0.65-fraction ceiling never OFFERED
    # anything above 1024. It was testing the cap, not the search.
    res = grid_search(_MockRingBench(), mode="smart", iters=10, log=lambda *a, **k: None)
    self.assertIsNotNone(res.best)
    self.assertEqual(res.best.bq, 9472)
    self.assertGreaterEqual(res.best.bkv, 1024)

  def test_oom_configs_pruned_not_raised(self):
    # a tiny VMEM budget OOMs the big pairs; search must still return (or None), never raise.
    res = grid_search(
        _MockRingBench(vmem=8 * 1024 * 1024),
        mode="smart",
        iters=2,
        log=lambda *a, **k: None,
    )
    self.assertTrue(any(r.status == "oom" for r in res.results))

  def test_broadcast_winner_handles_none_bkv_compute(self):
    from unittest import mock
    from maxdiffusion.utils.tile_size_grid_search import _broadcast_winner

    cand = BenchResult(bq=1024, bkv=512, bkv_compute=None, status="ok", mean_ms=10.0)
    with (
        mock.patch("jax.process_count", return_value=2),
        mock.patch("jax.process_index", return_value=0),
        mock.patch("jax.experimental.multihost_utils.broadcast_one_to_all", side_effect=lambda x, is_source: x),
    ):
      out = _broadcast_winner(cand, [cand])
    self.assertIs(out, cand)


MIB = 1024 * 1024

# Measured iperm-hybrid sweep winners (64 MiB VMEM): local q_seq -> (resident R,
# block_q_sub, block_kv). Any block_q with ceil_to(q_seq, block_q) == R runs the
# same self-attention kernel.
IPERM_MEASURED_WINNERS = {
    27900: (28160, 7040, 1024),  # dp2cp4 7.5 s, 146.6 ms
    36900: (37632, 6272, 1024),  # dp2cp4 10 s, 234.3 ms (next best 453.6 ms)
    22950: (23040, 5760, 1280),  # dp1cp8 12.5 s, 175.3 ms
    9450: (9600, 1920, 1280),  # dp1cp8 5 s, 50.2 ms
}


def _internal_ladder(q_seq):
  """The bq ladder the production search passes to the iperm planner."""
  return sorted({bq for bq, _ in smart_grid(q_seq, q_seq, vmem_bytes=VMEM_64MB, dtype_bytes=4, family="internal")})


class IpermPlannerTest(unittest.TestCase):
  """The resident-Q VMEM model, structural precheck and joint (bq, bkv, q_sub) planner."""

  def test_vmem_model_reproduces_production_oom(self):
    # R=48640 (bq=12160), auto q_sub=12160, bkv=1024: Mosaic reported 98.04 MiB used.
    pred = iperm_vmem_bytes(48640, 12160, 1024)
    self.assertGreater(pred, VMEM_64MB)
    self.assertAlmostEqual(pred / MIB, 98.04, delta=0.03 * 98.04)
    # ...which is exactly the config the unset default resolves to.
    self.assertEqual(auto_block_q_sub(48640), 12160)

  def test_vmem_model_reproduces_documented_q_sub_wall(self):
    # attention_flax: at R=30720, q_sub=7680 ran (104.65 ms) and 10240 OOMed at 71.17M.
    budget = IPERM_VMEM_MARGIN * VMEM_64MB
    self.assertLessEqual(iperm_vmem_bytes(30720, 7680, 1024), budget)
    self.assertAlmostEqual(iperm_vmem_bytes(30720, 10240, 1024) / MIB, 71.17, delta=0.02 * 71.17)

  def test_measured_winners_are_candidates(self):
    for q_seq, (resident, q_sub, bkv) in IPERM_MEASURED_WINNERS.items():
      for ladder in (_internal_ladder(q_seq), ()):
        with self.subTest(q_seq=q_seq, ladder=bool(ladder)):
          cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=ladder)
          hits = [c for c in cands if (c.resident, c.block_q_sub, c.bkv, c.block_q_outer) == (resident, q_sub, bkv, None)]
          self.assertTrue(hits, f"winner R={resident} q_sub={q_sub} bkv={bkv} missing")
          self.assertEqual(-(-q_seq // hits[0].bq) * hits[0].bq, resident)

  def test_block_q_sub_is_searched_jointly(self):
    # Requirement 1: several measured block_q_sub per resident length, not one derived value.
    q_seq = 27900
    cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
    q_subs_at_winner_r = {c.block_q_sub for c in cands if c.resident == 28160}
    self.assertGreaterEqual(len(q_subs_at_winner_r), 2)
    self.assertTrue(q_subs_at_winner_r - {auto_block_q_sub(28160)})

  def test_every_candidate_is_predicted_to_fit_and_well_formed(self):
    budget = IPERM_VMEM_MARGIN * VMEM_64MB
    for q_seq in (4950, 9450, 13950, 18450, 22950, 27900, 36900, 45900, 48600, 60000, 70000, 90000):
      cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
      self.assertTrue(cands, q_seq)
      for c in cands:
        with self.subTest(q_seq=q_seq, cand=c):
          resident = -(-q_seq // c.bq) * c.bq
          self.assertEqual(c.bkv_compute, c.bkv)
          if c.block_q_outer is None:
            rows = resident
            self.assertEqual(c.resident, resident)
            self.assertLessEqual(iperm_vmem_bytes(rows, c.block_q_sub, c.bkv), budget)
          else:
            rows = c.block_q_outer
            self.assertLess(rows, resident)
            self.assertEqual(resident % rows, 0)
            self.assertLessEqual(iperm_outer_vmem_bytes(rows, c.block_q_sub, c.bkv), budget)
          self.assertEqual(rows % c.block_q_sub, 0)
          self.assertEqual(c.block_q_sub % VPU_LANE, 0)
          # Never the whole block as one chunk: that is the instruction-memory cliff.
          self.assertLessEqual(c.block_q_sub, rows // 2)

  def test_anchors_always_present(self):
    # Requirement 7: the single-tile block_q and block_kv 1024 / 1280, at every shape.
    for q_seq in (9450, 22950, 27900, 36900, 45900, 48600):
      with self.subTest(q_seq=q_seq):
        cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
        single_tile = -(-q_seq // VPU_LANE) * VPU_LANE
        self.assertIn(single_tile, {c.bq for c in cands})
        self.assertTrue({1024, 1280} <= {c.bkv for c in cands})

  def test_production_oom_shape_gets_fitting_q_sub(self):
    # The U=2 production shape: every old candidate and the unset default OOMed. The
    # resident block itself fits; it is block_q_sub that has to shrink.
    q_seq = 48600
    self.assertTrue(iperm_structural_check(q_seq, vmem_bytes=VMEM_64MB).fits_resident)
    cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
    self.assertTrue(cands)
    self.assertNotIn((48640, 12160), {(c.resident, c.block_q_sub) for c in cands})
    self.assertTrue(all(c.block_q_outer is None for c in cands))

  def test_structural_check_separates_failure_classes(self):
    fits = iperm_structural_check(48600, vmem_bytes=VMEM_64MB)
    needs_outer = iperm_structural_check(70000, vmem_bytes=VMEM_64MB)
    hopeless = iperm_structural_check(600000, vmem_bytes=VMEM_64MB)
    self.assertEqual((fits.fits_resident, fits.fits_outer), (True, True))
    self.assertEqual((needs_outer.fits_resident, needs_outer.fits_outer), (False, True))
    self.assertEqual((hopeless.fits_resident, hopeless.fits_outer), (False, False))
    self.assertIn("block_q_outer", needs_outer.message)
    self.assertIn("ring shards", hopeless.message)

  def test_no_resident_fit_switches_to_block_q_outer(self):
    # Requirement 2/3: decided before compiling, and it yields block_q_outer candidates.
    q_seq = 70000
    cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
    self.assertTrue(cands)
    self.assertTrue(all(c.block_q_outer is not None and c.block_q_outer < c.bq * -(-q_seq // c.bq) for c in cands))

  def test_nothing_fits_returns_no_candidates(self):
    self.assertEqual(iperm_candidates(600000, 600000, vmem_bytes=VMEM_64MB, ladder_bqs=()), [])


if __name__ == "__main__":
  unittest.main()
