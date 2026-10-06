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
    FAILURE_DETAIL_CHARS,
    IPERM_FALLBACK_MAX_CANDIDATES,
    IPERM_VMEM_MARGIN,
    _aggregate_process_measurements,
    BenchResult,
    BlockBenchmark,
    TileSearchError,
    auto_block_q_sub,
    bkv_candidates,
    bq_candidates,
    classify_failure,
    failure_result,
    grid_search,
    iperm_candidates,
    iperm_outer_vmem_bytes,
    iperm_structural_check,
    iperm_vmem_bytes,
    local_tiled_seq_len,
    measured_vmem_bytes,
    outer_fallback_candidates,
    padding_of,
    smart_grid,
    time_callable,
    tune_block_sizes,
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
    # Requirement 7: the single-tile block_q and block_kv 1024 / 1280, at every resident-fitting shape.
    for q_seq in (9450, 22950, 27900, 36900):
      with self.subTest(q_seq=q_seq):
        cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
        single_tile = -(-q_seq // VPU_LANE) * VPU_LANE
        self.assertIn(single_tile, {c.bq for c in cands})
        self.assertTrue({1024, 1280} <= {c.bkv for c in cands})

  def test_resident_q_sub_wall_shape_gets_fitting_q_sub(self):
    # At R=36992..37888 (dp2cp4 10.0s, q_seq=36900), resident Q fits (epilogue ~58 MiB < 62.7 MiB),
    # while oversized q_sub * bkv tiles are pruned and fitting q_sub values are offered.
    q_seq = 36900
    self.assertTrue(iperm_structural_check(q_seq, vmem_bytes=VMEM_64MB).fits_resident)
    cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
    self.assertTrue(cands)
    self.assertNotIn((37888, 9472, 1024), {(c.resident, c.block_q_sub, c.bkv) for c in cands})
    self.assertTrue(all(c.block_q_outer is None for c in cands))

  def test_structural_check_separates_failure_classes(self):
    fits = iperm_structural_check(36900, vmem_bytes=VMEM_64MB)
    needs_outer = iperm_structural_check(45900, vmem_bytes=VMEM_64MB)
    hopeless = iperm_structural_check(600000, vmem_bytes=VMEM_64MB)
    self.assertEqual((fits.fits_resident, fits.fits_outer), (True, True))
    self.assertEqual((needs_outer.fits_resident, needs_outer.fits_outer), (False, True))
    self.assertEqual((hopeless.fits_resident, hopeless.fits_outer), (False, False))
    self.assertIn("block_q_outer", needs_outer.message)
    self.assertIn("ring shards", hopeless.message)

  def test_no_resident_fit_switches_to_block_q_outer(self):
    # Requirement 2/3: at dp2cp4 12.5s (q_seq=45900, R>=45952 -> 70.3 MiB epilogue VMEM),
    # decided before compiling, and yields block_q_outer candidates including the measured winner.
    for q_seq in (45900, 70000):
      cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
      self.assertTrue(cands)
      self.assertTrue(all(c.block_q_outer is not None and c.block_q_outer < c.bq * -(-q_seq // c.bq) for c in cands))

  def test_nothing_fits_returns_no_candidates(self):
    self.assertEqual(iperm_candidates(600000, 600000, vmem_bytes=VMEM_64MB, ladder_bqs=()), [])


IPERM_HYBRID = "ulysses_ring_custom_iperm_fixed_m_hybrid"


class _MockIpermBench(BlockBenchmark):
  """Resident-Q iperm bench (dp2cp4 ring with U=2 -> 2 ring shards) that records the knobs
  every run receives. Larger block_q_sub and less resident padding are faster."""

  label = "mock-iperm"

  def __init__(self, seq=27900, vmem=VMEM_64MB, attention=IPERM_HYBRID, context_shards=4, ulysses_shards=2):
    self.seq, self._vmem = seq, vmem
    self._attention = attention
    self._context_shards = context_shards
    self._ulysses_shards = ulysses_shards
    self.calls = []

  def tiled_seq_lens(self):
    return (self.seq, self.seq)

  def vmem_bytes(self):
    return self._vmem

  def run(self, bq, bkv, *, bkv_compute=None, iters=10, warmup=2, block_q_sub=None, block_q_outer=None):
    self.calls.append((bq, bkv, bkv_compute, block_q_sub, block_q_outer))
    cmp = bkv_compute or bkv
    resident = -(-self.seq // bq) * bq
    ms = 100.0 + 0.01 * (resident - self.seq) + 1e5 / (block_q_sub or 128) + 0.001 * abs(cmp - 1024)
    return BenchResult(bq, bkv, cmp, "ok", mean_ms=round(ms, 4), std_ms=0.1, compile_ms=1.0)


class _FakeConfig:
  """pyconfig-like: attribute reads go through the raw keys dict maybe_tune mutates."""

  def __init__(self, **keys):
    self.__dict__["_keys"] = dict(keys)

  def get_keys(self):
    return self._keys

  def __getattr__(self, name):
    try:
      return self.__dict__["_keys"][name]
    except KeyError as e:
      raise AttributeError(name) from e


def _quiet(*_a, **_k):
  pass


class JointSearchPropagationTest(unittest.TestCase):
  """block_q_sub / block_q_outer travel planner -> bench -> result -> CSV/broadcast -> config."""

  def test_iperm_search_passes_joint_knobs_to_bench(self):
    bench = _MockIpermBench()
    res = grid_search(bench, mode="smart", iters=1, log=_quiet)
    self.assertIsNotNone(res.structure)
    self.assertTrue(res.structure.fits_resident)
    planned = {(c.bq, c.bkv, c.bkv_compute, c.block_q_sub, c.block_q_outer) for c in res.candidates}
    self.assertEqual(set(bench.calls), planned)
    self.assertTrue(all(call[3] is not None for call in bench.calls))
    for r in res.results:
      self.assertIn(r.key(), planned)
      self.assertIsNotNone(r.resident)
      self.assertIsNotNone(r.pred_vmem_bytes)
    best = res.best.as_candidate()
    self.assertEqual(min(res.results, key=lambda r: r.mean_ms).key(), res.best.key())
    self.assertIsNotNone(best["block_q_sub"])
    self.assertEqual(res.ranked_candidates()[0], best)
    self.assertEqual(len(res.ranked_candidates()), len(res.results))

  def test_legacy_bench_without_knob_kwargs_still_runs(self):
    # _MockRingBench.run has no block_q_sub/block_q_outer parameters.
    res = grid_search(_MockRingBench(), mode="smart", iters=1, log=_quiet)
    self.assertIsNone(res.structure)
    self.assertTrue(all(r.block_q_sub is None and r.block_q_outer is None for r in res.results))
    self.assertIsNone(res.best.as_candidate()["block_q_sub"])

  def test_csv_carries_joint_knobs(self):
    import csv
    import tempfile

    with tempfile.TemporaryDirectory() as d:
      res = grid_search(_MockIpermBench(), mode="smart", iters=1, out_dir=d, log=_quiet)
      with open(res.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    self.assertEqual(len(rows), len(res.results))
    self.assertEqual(int(rows[0]["block_q_sub"]), res.best.block_q_sub)
    self.assertEqual(rows[0]["block_q_outer"], "")
    self.assertTrue(all(row["pred_vmem_mib"] and row["resident"] for row in rows))

  def test_broadcast_winner_distinguishes_block_q_sub(self):
    from unittest import mock
    from maxdiffusion.utils.tile_size_grid_search import _broadcast_winner

    small = BenchResult(7040, 1024, 1024, "ok", mean_ms=2.0, block_q_sub=3520)
    big = BenchResult(7040, 1024, 1024, "ok", mean_ms=1.0, block_q_sub=7040)
    payload = np.asarray([1, 7040, 1024, 1024, 7040, -1])
    with (
        mock.patch("jax.process_count", return_value=2),
        mock.patch("jax.process_index", return_value=1),
        mock.patch("jax.experimental.multihost_utils.broadcast_one_to_all", return_value=payload),
    ):
      self.assertIs(_broadcast_winner(None, [small, big]), big)


class BlockSizePropagationTest(unittest.TestCase):
  """Requirement 5: no tunable field is dropped, and the bench runs what production runs."""

  def test_every_custom_field_round_trips_through_get_flash_block_sizes(self):
    import dataclasses
    from types import SimpleNamespace
    from maxdiffusion import max_utils

    names = [f.name for f in dataclasses.fields(max_utils.CustomFlashBlockSizes)]
    self.assertTrue({"block_q", "block_kv", "block_kv_compute", "block_q_sub", "block_q_outer"} <= set(names))
    fbs = {name: 128 * (i + 1) for i, name in enumerate(names)}
    out = max_utils.get_flash_block_sizes(SimpleNamespace(attention=IPERM_HYBRID, flash_block_sizes=fbs))
    self.assertIsInstance(out, max_utils.CustomFlashBlockSizes)
    self.assertEqual({name: getattr(out, name) for name in names}, fbs)

  def test_candidate_reaches_custom_kernel_unchanged(self):
    from maxdiffusion.models.attention_flax import _extract_custom_block_sizes
    from maxdiffusion.utils.tile_size_grid_search import apply_candidate_to_config, resolve_block_sizes

    cfg = _FakeConfig(attention=IPERM_HYBRID, flash_block_sizes={"heads_per_tile": 1})
    cand = {"block_q": 7040, "block_kv": 1024, "block_kv_compute": 1024, "block_q_sub": 3520, "block_q_outer": 14080}
    apply_candidate_to_config(cfg, cand, vmem_limit_bytes=VMEM_64MB)
    out = resolve_block_sizes(cfg.attention, cfg.flash_block_sizes)
    self.assertEqual((out.block_q_sub, out.block_q_outer), (3520, 14080))
    self.assertEqual(_extract_custom_block_sizes(out), (7040, 1024, 1024, 1024, 1, VMEM_64MB))

  def test_bench_and_production_build_identical_block_sizes(self):
    from types import SimpleNamespace
    from maxdiffusion import max_utils
    from maxdiffusion.utils.ltx2_block_benchmark import LTX2BlockBenchmark
    from maxdiffusion.utils.tile_size_grid_search import apply_candidate_to_config
    from maxdiffusion.utils.wan_block_benchmark import WanBlockBenchmark

    base = {"heads_per_tile": 2, "block_q_sub": 999, "block_q_outer": 4096, "vmem_limit_bytes": 1}
    cands = (
        {"block_q": 7040, "block_kv": 1024, "block_kv_compute": 1024, "block_q_sub": 3520, "block_q_outer": 14080},
        {"block_q": 1792, "block_kv": 1280, "block_kv_compute": 1280, "block_q_sub": None, "block_q_outer": None},
    )
    for attention in (IPERM_HYBRID, "ulysses_custom", "flash", "tokamax_flash"):
      for cand in cands:
        with self.subTest(attention=attention, cand=cand):
          cfg = _FakeConfig(attention=attention, flash_block_sizes=dict(base))
          apply_candidate_to_config(cfg, cand, vmem_limit_bytes=VMEM_64MB)
          production = max_utils.get_flash_block_sizes(cfg)
          for bench_cls in (WanBlockBenchmark, LTX2BlockBenchmark):
            bench = object.__new__(bench_cls)
            bench._config = SimpleNamespace(flash_block_sizes=dict(base))
            bench._attention = attention
            bench._vmem = VMEM_64MB
            built = bench._flash_block_sizes(
                cand["block_q"],
                cand["block_kv"],
                cand["block_kv_compute"],
                block_q_sub=cand["block_q_sub"],
                block_q_outer=cand["block_q_outer"],
            )
            self.assertEqual(built, production, bench_cls.__name__)
          if "custom" in attention:
            self.assertEqual(production.heads_per_tile, 2)  # inherited from the config, not forced
            self.assertEqual(production.vmem_limit_bytes, VMEM_64MB)

  def test_apply_candidate_clears_stale_knobs(self):
    from maxdiffusion.utils.tile_size_grid_search import apply_candidate_to_config, resolve_block_sizes

    cfg = _FakeConfig(attention=IPERM_HYBRID, flash_block_sizes={"block_q_sub": 999, "block_q_outer": 4096})
    cand = {"block_q": 1792, "block_kv": 1024, "block_kv_compute": 1024, "block_q_sub": None, "block_q_outer": None}
    fbs = apply_candidate_to_config(cfg, cand)
    self.assertNotIn("block_q_sub", fbs)
    self.assertNotIn("block_q_outer", fbs)
    out = resolve_block_sizes(cfg.attention, cfg.flash_block_sizes)
    self.assertEqual((out.block_q_sub, out.block_q_outer), (None, None))

  def test_apply_candidate_fails_loud_when_a_field_is_dropped(self):
    from unittest import mock
    from maxdiffusion import max_utils
    from maxdiffusion.utils.tile_size_grid_search import apply_candidate_to_config

    def drops_q_sub(config):
      fbs = config.flash_block_sizes
      return max_utils.CustomFlashBlockSizes(block_q=fbs["block_q"], block_kv=fbs["block_kv"])

    cfg = _FakeConfig(attention=IPERM_HYBRID, flash_block_sizes={})
    cand = {"block_q": 7040, "block_kv": 1024, "block_kv_compute": 1024, "block_q_sub": 3520, "block_q_outer": None}
    with mock.patch.object(max_utils, "get_flash_block_sizes", side_effect=drops_q_sub):
      with self.assertRaisesRegex(RuntimeError, "block_q_sub"):
        apply_candidate_to_config(cfg, cand)

  def test_tune_block_sizes_applies_winner_and_logs_one_using_line(self):
    import re

    cfg = _FakeConfig(
        attention=IPERM_HYBRID,
        flash_block_sizes={"heads_per_tile": 1},
        tile_search_mode="smart",
        tile_search_iters=1,
    )
    lines = []
    ranked = tune_block_sizes(cfg, _MockIpermBench(), vmem_limit_bytes=VMEM_64MB, log=lines.append)
    using = [line for line in lines if line.startswith("[tile-search] using ")]
    self.assertEqual(len(using), 1)
    m = re.fullmatch(
        r"\[tile-search\] using block_q=(\d+) block_kv=(\d+) block_kv_compute=(\d+) "
        r"block_q_sub=(\d+|auto) block_q_outer=(\d+|none) \(block-bench ([\d.]+) ms\)",
        using[0],
    )
    self.assertIsNotNone(m, using[0])
    best = ranked[0]
    self.assertEqual(
        m.groups()[:5],
        tuple(str(best[k]) for k in ("block_q", "block_kv", "block_kv_compute", "block_q_sub")) + ("none",),
    )
    fbs = cfg.flash_block_sizes
    for k in ("block_q", "block_kv", "block_kv_compute", "block_q_sub"):
      self.assertEqual(fbs[k], best[k])
    self.assertEqual(fbs["vmem_limit_bytes"], VMEM_64MB)

  def test_tile_search_vmem_limit_bytes_resolution(self):
    from maxdiffusion.utils.tile_size_grid_search import DEFAULT_VMEM_LIMIT_BYTES, tile_search_vmem_limit_bytes

    self.assertEqual(tile_search_vmem_limit_bytes(_FakeConfig(flash_block_sizes={})), DEFAULT_VMEM_LIMIT_BYTES)
    self.assertEqual(tile_search_vmem_limit_bytes(_FakeConfig(flash_block_sizes={"vmem_limit_bytes": 5})), 5)
    cfg = _FakeConfig(flash_block_sizes={"vmem_limit_bytes": 5}, tile_search_vmem_limit_bytes=7)
    self.assertEqual(tile_search_vmem_limit_bytes(cfg), 7)


# The message every over-budget candidate in the iperm sweeps failed with.
_XLA_VMEM_OOM = (
    "RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
    "Used 98.04M of 64.00M vmem. Exceeded vmem capacity by 34.04M."
)
_MOSAIC_ERROR = "Mosaic failed to compile TPU kernel: unsupported VMEM layout for tpu.matmul"


class _XlaRuntimeError(RuntimeError):
  """Stands in for jax.errors.JaxRuntimeError."""


class _FailingIpermBench(_MockIpermBench):
  """_MockIpermBench whose resident-Q candidates raise `resident_exc` and whose block_q_outer
  candidates raise `outer_exc` (None: they run)."""

  def __init__(self, resident_exc, outer_exc=None, **kwargs):
    super().__init__(**kwargs)
    self.resident_exc, self.outer_exc = resident_exc, outer_exc

  def run(self, bq, bkv, *, block_q_outer=None, **kwargs):
    result = super().run(bq, bkv, block_q_outer=block_q_outer, **kwargs)  # records the call
    exc = self.resident_exc if block_q_outer is None else self.outer_exc
    if exc is not None:
      raise exc
    return result


class _RaisingRingBench(_MockRingBench):

  def run(self, bq, bkv, **kwargs):
    raise ValueError("block_kv must divide the padded kv length")


class FailureHandlingTest(unittest.TestCase):
  """OOM vs error classification, the runtime block_q_outer fallback, and failing loudly."""

  def _cfg(self, **keys):
    base = {"attention": IPERM_HYBRID, "flash_block_sizes": {"heads_per_tile": 1}, "tile_search_iters": 1}
    return _FakeConfig(**{**base, **keys})

  def test_only_memory_exhaustion_is_oom(self):
    self.assertEqual(classify_failure("JaxRuntimeError: " + _XLA_VMEM_OOM), "oom")
    self.assertEqual(classify_failure("Scoped allocation of 70.1M exceeded scoped vmem limit by 6.1M"), "oom")
    self.assertEqual(classify_failure("out of memory allocating 2.0G"), "oom")
    # Mosaic lowering / verification errors mention Mosaic and VMEM but are not about size:
    # the old benches filed them under OOM, which would send the search after smaller tiles.
    for msg in (_MOSAIC_ERROR, "INTERNAL: Mosaic: failed to legalize 'tpu.memref_slice' (vmem)", "ValueError: bad"):
      with self.subTest(msg=msg):
        self.assertEqual(classify_failure(msg), "error")

  def test_failure_result_is_one_line_and_keeps_measured_vmem(self):
    exc = _XlaRuntimeError(_XLA_VMEM_OOM + "\n\nLargest program allocations in vmem:\n" + "  alloc\n" * 200)
    r = failure_result(7040, 1024, 1024, exc)
    self.assertEqual(r.status, "oom")
    self.assertNotIn("\n", r.detail)
    self.assertTrue(r.detail.startswith("_XlaRuntimeError: RESOURCE_EXHAUSTED"))
    self.assertLessEqual(len(r.detail), FAILURE_DETAIL_CHARS)
    used, capacity = measured_vmem_bytes(r.detail)
    self.assertAlmostEqual(used / MIB, 98.04)
    self.assertAlmostEqual(capacity / MIB, 64.0)
    late = failure_result(1, 1, 1, RuntimeError("RESOURCE_EXHAUSTED: " + "x" * 400 + " Used 70.50M of 64.00M vmem."))
    self.assertAlmostEqual(measured_vmem_bytes(late.detail)[0] / MIB, 70.5)
    self.assertIsNone(measured_vmem_bytes(_MOSAIC_ERROR))

  def test_model_benches_use_the_shared_classifier(self):
    import contextlib
    from unittest import mock
    from maxdiffusion.utils.ltx2_block_benchmark import LTX2BlockBenchmark
    from maxdiffusion.utils.wan_block_benchmark import WanBlockBenchmark

    for bench_cls in (WanBlockBenchmark, LTX2BlockBenchmark):
      for message, status in ((_MOSAIC_ERROR, "error"), (_XLA_VMEM_OOM, "oom")):
        with self.subTest(bench=bench_cls.__name__, status=status):
          bench = object.__new__(bench_cls)
          bench._mesh = contextlib.nullcontext()
          bench._build_model = mock.Mock(side_effect=_XlaRuntimeError(message))
          with mock.patch("traceback.print_exc"):
            r = bench.run(7040, 1024, block_q_sub=3520)
          self.assertEqual((r.bq, r.bkv, r.status), (7040, 1024, status))
          self.assertIn(message[:40], r.detail)

  def test_a_raising_bench_does_not_abort_the_search(self):
    res = grid_search(_RaisingRingBench(), mode="smart", iters=1, log=_quiet)
    self.assertIsNone(res.best)
    self.assertTrue(res.results)
    self.assertTrue(all(r.status == "error" and "ValueError" in r.detail for r in res.results))
    self.assertEqual(res.outer_fallback, 0)

  def test_resident_oom_falls_back_to_block_q_outer(self):
    # Requirement 2 at run time: the model said the resident block fits, the compiler disagreed.
    bench = _FailingIpermBench(_XlaRuntimeError(_XLA_VMEM_OOM))
    lines = []
    res = grid_search(bench, mode="smart", iters=1, log=lines.append)
    self.assertTrue(res.structure.fits_resident)
    first = res.candidates[: len(res.candidates) - res.outer_fallback]
    fallback = res.candidates[len(first) :]
    self.assertTrue(first and all(c.block_q_outer is None for c in first))
    self.assertTrue(0 < len(fallback) <= IPERM_FALLBACK_MAX_CANDIDATES)
    budget = IPERM_VMEM_MARGIN * VMEM_64MB
    for c in fallback:
      self.assertEqual(c.tag, "outer-fallback")
      self.assertLessEqual(iperm_outer_vmem_bytes(c.block_q_outer, c.block_q_sub, c.bkv), budget)
    self.assertTrue({(c.bq, c.bkv, c.bkv_compute, c.block_q_sub, c.block_q_outer) for c in fallback} <= set(bench.calls))
    self.assertEqual([r.status for r in res.results[: len(first)]], ["oom"] * len(first))
    self.assertIsNotNone(res.best)
    self.assertIsNotNone(res.best.block_q_outer)
    self.assertEqual(res.best.tag, "outer-fallback")
    self.assertIsNotNone(res.ranked_candidates()[0]["block_q_outer"])
    self.assertEqual(sum("falling back to" in line for line in lines), 1)
    total = len(res.candidates)
    self.assertTrue(any(line.startswith(f"  [{total}/{total}] ") for line in lines))

  def test_fallback_winner_reaches_production(self):
    cfg = self._cfg()
    ranked = tune_block_sizes(
        cfg, _FailingIpermBench(_XlaRuntimeError(_XLA_VMEM_OOM)), vmem_limit_bytes=VMEM_64MB, log=_quiet
    )
    self.assertIsNotNone(ranked[0]["block_q_outer"])
    self.assertEqual(cfg.flash_block_sizes["block_q_outer"], ranked[0]["block_q_outer"])
    self.assertEqual(cfg.flash_block_sizes["block_q_sub"], ranked[0]["block_q_sub"])

  def test_no_fallback_for_non_memory_errors(self):
    bench = _FailingIpermBench(_XlaRuntimeError(_MOSAIC_ERROR))
    res = grid_search(bench, mode="smart", iters=1, log=_quiet)
    self.assertEqual(res.outer_fallback, 0)
    self.assertTrue(all(r.status == "error" for r in res.results))
    self.assertTrue(all(call[4] is None for call in bench.calls))

  def test_fallback_only_when_nothing_ran_and_something_ooms(self):
    from dataclasses import replace

    q_seq = 27900
    structure = iperm_structural_check(q_seq, vmem_bytes=VMEM_64MB)
    cands = iperm_candidates(q_seq, q_seq, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(q_seq))
    oom = [BenchResult(c.bq, c.bkv, c.bkv_compute, "oom") for c in cands]

    def fallback(results, structure=structure, cands=cands, q=q_seq):
      return outer_fallback_candidates(q, q, vmem_bytes=VMEM_64MB, structure=structure, cands=cands, results=results)

    self.assertTrue(fallback(oom))
    self.assertEqual(fallback([replace(oom[0], status="ok", mean_ms=1.0)] + oom[1:]), [])
    self.assertEqual(fallback([replace(r, status="error") for r in oom]), [])
    self.assertEqual(fallback(oom, structure=None), [])  # not a resident-Q iperm search
    # The first round already measured block_q_outer (no resident fit): nothing left to try.
    big = 70000
    outer_round = iperm_candidates(big, big, vmem_bytes=VMEM_64MB, ladder_bqs=_internal_ladder(big))
    big_oom = [BenchResult(c.bq, c.bkv, c.bkv_compute, "oom") for c in outer_round]
    big_structure = iperm_structural_check(big, vmem_bytes=VMEM_64MB)
    self.assertEqual(fallback(big_oom, structure=big_structure, cands=outer_round, q=big), [])

  def test_all_oom_raises_with_model_underprediction(self):
    cfg = self._cfg()
    oom = _XlaRuntimeError(_XLA_VMEM_OOM)
    with self.assertRaises(TileSearchError) as ctx:
      tune_block_sizes(cfg, _FailingIpermBench(oom, outer_exc=oom), vmem_limit_bytes=VMEM_64MB, log=_quiet)
    msg = str(ctx.exception)
    self.assertIsInstance(ctx.exception, RuntimeError)
    self.assertIn("under-predicts", msg)
    self.assertIn("the compiler reported 98.0 MiB", msg)  # measured, next to the prediction
    self.assertIn("block_q_outer fallback", msg)
    self.assertIn("tile_search_fail_open", msg)
    self.assertEqual(cfg.flash_block_sizes, {"heads_per_tile": 1})  # nothing applied

  def test_all_errors_raise_as_a_kernel_problem(self):
    cfg = self._cfg()
    with self.assertRaises(TileSearchError) as ctx:
      tune_block_sizes(cfg, _FailingIpermBench(_XlaRuntimeError(_MOSAIC_ERROR)), vmem_limit_bytes=VMEM_64MB, log=_quiet)
    msg = str(ctx.exception)
    self.assertIn("not a tile-size one", msg)
    self.assertIn("unsupported VMEM layout", msg)
    self.assertNotIn("under-predicts", msg)

  def test_structural_failure_is_reported_without_compiling(self):
    bench = _MockIpermBench(seq=600000)
    with self.assertRaises(TileSearchError) as ctx:
      tune_block_sizes(self._cfg(), bench, vmem_limit_bytes=VMEM_64MB, log=_quiet)
    self.assertIn("Structural", str(ctx.exception))
    self.assertIn("ring shards", str(ctx.exception))
    self.assertEqual(bench.calls, [])

  def test_ladder_oom_blames_the_ladder_ceilings(self):
    with self.assertRaises(TileSearchError) as ctx:
      tune_block_sizes(self._cfg(attention="flash"), _MockRingBench(vmem=8 * MIB), log=_quiet)
    self.assertIn("_MEASURED_BKV_CEILING", str(ctx.exception))

  def test_fail_open_logs_the_diagnosis_and_keeps_the_config(self):
    for flag in (True, "true", "1"):
      with self.subTest(flag=flag):
        cfg = self._cfg(tile_search_fail_open=flag)
        lines = []
        bench = _FailingIpermBench(_XlaRuntimeError(_MOSAIC_ERROR))
        self.assertEqual(tune_block_sizes(cfg, bench, vmem_limit_bytes=VMEM_64MB, log=lines.append), [])
        failed = [line for line in lines if line.startswith("[tile-search] FAILED: ")]
        self.assertEqual(len(failed), 1)
        self.assertIn("not a tile-size one", failed[0])
        self.assertFalse(any(line.startswith("[tile-search] using ") for line in lines))
        self.assertEqual(cfg.flash_block_sizes, {"heads_per_tile": 1})
    with self.assertRaises(TileSearchError):
      tune_block_sizes(
          self._cfg(tile_search_fail_open="false"),
          _FailingIpermBench(_XlaRuntimeError(_MOSAIC_ERROR)),
          vmem_limit_bytes=VMEM_64MB,
          log=_quiet,
      )


if __name__ == "__main__":
  unittest.main()
