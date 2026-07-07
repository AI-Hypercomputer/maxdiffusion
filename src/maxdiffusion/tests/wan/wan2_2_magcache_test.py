"""
Copyright 2025 Google LLC

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

import os
import time
import unittest

import numpy as np
import pytest
from absl.testing import absltest

from maxdiffusion import max_logging
from maxdiffusion.pipelines.wan.wan_pipeline import init_magcache, magcache_step
from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanMagCacheValidationTest(unittest.TestCase):
  """Tests that use_magcache validation raises correct errors."""

  def _make_pipeline(self):
    return WanPipeline2_2.__new__(WanPipeline2_2)

  def test_magcache_with_both_scales_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(prompt=["test"], guidance_scale_low=1.0, guidance_scale_high=1.0, use_magcache=True)
    self.assertIn("use_magcache", str(ctx.exception))

  def test_magcache_with_low_scale_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(prompt=["test"], guidance_scale_low=0.5, guidance_scale_high=4.0, use_magcache=True)
    self.assertIn("use_magcache", str(ctx.exception))

  def test_magcache_with_high_scale_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(prompt=["test"], guidance_scale_low=3.0, guidance_scale_high=1.0, use_magcache=True)
    self.assertIn("use_magcache", str(ctx.exception))

  def test_magcache_mutually_exclusive_with_cfg_cache(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=3.0,
          guidance_scale_high=4.0,
          use_cfg_cache=True,
          use_magcache=True,
      )
    self.assertIn("mutually exclusive", str(ctx.exception))

  def test_magcache_mutually_exclusive_with_sen_cache(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=3.0,
          guidance_scale_high=4.0,
          use_sen_cache=True,
          use_magcache=True,
      )
    self.assertIn("mutually exclusive", str(ctx.exception))

  def test_magcache_with_valid_scales_no_validation_error(self):
    """Both guidance_scales > 1.0 should pass validation (may fail later without model)."""
    pipeline = self._make_pipeline()
    try:
      pipeline(prompt=["test"], guidance_scale_low=3.0, guidance_scale_high=4.0, use_magcache=True)
    except ValueError as e:
      if "use_magcache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      pass

  def test_no_magcache_with_low_scales_no_error(self):
    """use_magcache=False should never raise our ValueError."""
    pipeline = self._make_pipeline()
    try:
      pipeline(prompt=["test"], guidance_scale_low=0.5, guidance_scale_high=0.5, use_magcache=False)
    except ValueError as e:
      if "use_magcache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      pass


class WanMagCacheScheduleTest(unittest.TestCase):
  """Tests the MagCache dual-phase schedule (retention zones + boundary reset).

  Mirrors the deterministic, host-side schedule in run_inference_2_2's MagCache
  branch: forced-compute (retention) zone at the start of EACH phase, an explicit
  cache reset at the high->low boundary, and global-step indexing into mag_ratios.
  The actual skip decision (magcache_step) is exercised separately below.
  """

  def _get_magcache_schedule(self, num_inference_steps, retention_ratio=0.2, boundary_ratio=0.875, num_train_timesteps=1000):
    boundary = boundary_ratio * num_train_timesteps
    timesteps = np.linspace(num_train_timesteps - 1, 0, num_inference_steps, dtype=np.int32)
    step_uses_high = [bool(timesteps[s] >= boundary) for s in range(num_inference_steps)]
    high_noise_steps = sum(step_uses_high)

    high_warmup_end = int(high_noise_steps * retention_ratio)
    low_warmup_end = high_noise_steps + int((num_inference_steps - high_noise_steps) * retention_ratio)

    force_compute, is_boundary_list = [], []
    cached_residual_is_none = True  # no residual until the first compute
    for step in range(num_inference_steps):
      is_boundary = step > 0 and step_uses_high[step] != step_uses_high[step - 1]
      if is_boundary:
        cached_residual_is_none = True  # boundary reset
      in_warmup = step < high_warmup_end or (high_noise_steps <= step < low_warmup_end)
      forced = in_warmup or is_boundary or cached_residual_is_none
      force_compute.append(forced)
      is_boundary_list.append(is_boundary)
      cached_residual_is_none = False  # a residual exists after any step (computed or reused)

    return {
        "step_uses_high": step_uses_high,
        "high_noise_steps": high_noise_steps,
        "high_warmup_end": high_warmup_end,
        "low_warmup_end": low_warmup_end,
        "force_compute": force_compute,
        "is_boundary": is_boundary_list,
    }

  def test_first_step_always_forced(self):
    """Step 0 has no cached residual yet, so it must compute."""
    sched = self._get_magcache_schedule(40)
    self.assertTrue(sched["force_compute"][0])

  def test_high_phase_warmup_forced(self):
    """The first retention_ratio fraction of the high-noise phase is forced."""
    sched = self._get_magcache_schedule(40)
    self.assertTrue(all(sched["force_compute"][: sched["high_warmup_end"]]))

  def test_low_phase_warmup_forced(self):
    """The first retention_ratio fraction of the low-noise phase (post-boundary) is forced."""
    sched = self._get_magcache_schedule(40)
    high, low_end = sched["high_noise_steps"], sched["low_warmup_end"]
    self.assertTrue(all(sched["force_compute"][high:low_end]), "Low-phase warmup zone must be forced")

  def test_boundary_is_forced(self):
    """Every high<->low transition step must compute (residual reset)."""
    sched = self._get_magcache_schedule(40)
    suh = sched["step_uses_high"]
    for s in range(1, 40):
      if suh[s] != suh[s - 1]:
        self.assertTrue(sched["is_boundary"][s], f"step {s} should be a boundary")
        self.assertTrue(sched["force_compute"][s], f"Boundary step {s} must be forced")

  def test_exactly_one_boundary(self):
    """A monotone timestep schedule crosses the boundary exactly once."""
    sched = self._get_magcache_schedule(40)
    self.assertEqual(sum(sched["is_boundary"]), 1)

  def test_cacheable_window_exists(self):
    """With enough steps, some steps are eligible to skip (not forced)."""
    sched = self._get_magcache_schedule(40)
    self.assertGreater(sum(not f for f in sched["force_compute"]), 0, "Expected some cacheable steps")

  def test_warmup_zones_scale_with_steps(self):
    for n in [20, 40, 80]:
      sched = self._get_magcache_schedule(n)
      self.assertTrue(all(sched["force_compute"][: sched["high_warmup_end"]]), f"high warmup forced @ {n}")
      self.assertTrue(
          all(sched["force_compute"][sched["high_noise_steps"] : sched["low_warmup_end"]]),
          f"low warmup forced @ {n}",
      )

  def test_global_step_indexing_in_bounds(self):
    """mag_ratios is indexed by GLOBAL step as [2*step] (cond) / [2*step+1] (uncond)."""
    n = 40
    mag_ratios = np.ones(2 * n)
    for step in range(n):
      _ = mag_ratios[step * 2]
      _ = mag_ratios[step * 2 + 1]  # must not raise
    self.assertEqual(len(mag_ratios), 2 * n)


class WanMagCacheCoreTest(unittest.TestCase):
  """Pure-host tests for init_magcache / magcache_step (no TPU, CI-safe).

  These confirm the skip schedule is deterministic given constant mag_ratios,
  which is the property that lets MagCache live in the host-side denoise loop.
  """

  def test_init_passthrough_when_double_length(self):
    """A curve already of length 2*steps is used verbatim (no interpolation)."""
    n = 5
    base = list(np.linspace(1.0, 0.9, 2 * n))
    out = init_magcache(n, 0.2, base)
    mag_ratios = out[8]
    self.assertEqual(len(mag_ratios), 2 * n)
    np.testing.assert_allclose(mag_ratios, np.array(base))

  def test_init_interpolates_when_mismatched(self):
    """A shorter curve is nearest-interpolated up to length 2*steps."""
    n = 40
    base = list(np.linspace(1.0, 0.8, 2 * 20))  # 20-step curve, run is 40 steps
    out = init_magcache(n, 0.2, base)
    self.assertEqual(len(out[8]), 2 * n)

  def test_init_skip_warmup(self):
    self.assertEqual(init_magcache(40, 0.2, list(np.ones(80)))[7], int(40 * 0.2))

  def test_disabled_never_skips(self):
    """use_magcache=False forces a compute and leaves accumulators untouched."""
    n = 10
    mag_ratios = np.ones(2 * n)
    state = init_magcache(n, 0.2, list(mag_ratios))[:6]
    skip, new_state = magcache_step(3, mag_ratios, state, magcache_thresh=1.0, magcache_K=99, use_magcache=False)
    self.assertFalse(skip)
    self.assertEqual(new_state, state)

  def test_skips_when_under_threshold(self):
    """Ratios ~1.0 with a generous threshold should skip."""
    n = 10
    mag_ratios = np.ones(2 * n)
    state = init_magcache(n, 0.2, list(mag_ratios))[:6]
    skip, _ = magcache_step(3, mag_ratios, state, magcache_thresh=0.5, magcache_K=99, use_magcache=True)
    self.assertTrue(skip)

  def test_resets_when_over_threshold(self):
    """A ratio far from 1.0 exceeds the error budget -> no skip + accumulators reset."""
    n = 10
    mag_ratios = np.full(2 * n, 0.5)  # err = |1 - 0.5| = 0.5 per step
    state = init_magcache(n, 0.2, list(mag_ratios))[:6]
    skip, new_state = magcache_step(3, mag_ratios, state, magcache_thresh=0.04, magcache_K=99, use_magcache=True)
    self.assertFalse(skip)
    self.assertEqual(new_state, (1.0, 1.0, 0.0, 0.0, 0, 0))

  def test_K_caps_consecutive_skips(self):
    """Even with err=0 (ratio 1.0), no more than magcache_K consecutive skips."""
    n = 20
    K = 2
    mag_ratios = np.ones(2 * n)
    state = init_magcache(n, 0.0, list(mag_ratios))[:6]
    consecutive = 0
    max_consecutive = 0
    for step in range(n):
      skip, state = magcache_step(step, mag_ratios, state, magcache_thresh=1.0, magcache_K=K, use_magcache=True)
      if skip:
        consecutive += 1
        max_consecutive = max(max_consecutive, consecutive)
      else:
        consecutive = 0
    self.assertLessEqual(max_consecutive, K, f"skipped {max_consecutive} in a row, K={K}")

  def test_requires_both_cond_and_uncond_under_threshold(self):
    """If the uncond branch blows the budget, the step is not skipped even if cond is fine."""
    n = 10
    mag_ratios = np.ones(2 * n)
    mag_ratios[3 * 2 + 1] = 0.5  # uncond at step 3 is far from 1.0
    state = init_magcache(n, 0.2, list(mag_ratios))[:6]
    skip, _ = magcache_step(3, mag_ratios, state, magcache_thresh=0.04, magcache_K=99, use_magcache=True)
    self.assertFalse(skip)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class WanMagCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: MagCache should be faster with PSNR >= 30 dB, SSIM >= 0.95.

  Runs on TPU (WAN 2.2 27B T2V, 720p). Skipped in CI (GitHub Actions) — run with:
    python -m pytest src/maxdiffusion/tests/wan/wan_mag_cache_test.py::WanMagCacheSmokeTest -v
  """

  @classmethod
  def setUpClass(cls):
    from maxdiffusion import pyconfig
    from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "..", "configs", "base_wan_27b.yml"),
            "num_inference_steps=40",
            "height=720",
            "width=1280",
            "num_frames=81",
            "fps=16",
            "guidance_scale_low=3.0",
            "guidance_scale_high=4.0",
            "boundary_ratio=0.875",
            "flow_shift=12.0",
            "seed=118445",
            "attention=flash",
            "remat_policy=FULL",
            "allow_split_physical_axes=True",
            "skip_jax_distributed_system=True",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "per_device_batch_size=0.125",
            "ici_data_parallelism=1",
            "ici_fsdp_parallelism=1",
            "ici_context_parallelism=8",
            "ici_tensor_parallelism=1",
            "flash_min_seq_length=0",
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointer2_2(config=cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    for use_cache in [False, True]:  # warm up both XLA code paths
      cls.pipeline(
          prompt=cls.prompt,
          negative_prompt=cls.negative_prompt,
          height=cls.config.height,
          width=cls.config.width,
          num_frames=cls.config.num_frames,
          num_inference_steps=cls.config.num_inference_steps,
          guidance_scale_low=cls.config.guidance_scale_low,
          guidance_scale_high=cls.config.guidance_scale_high,
          use_magcache=use_cache,
      )

  def _run_pipeline(self, use_magcache):
    t0 = time.perf_counter()
    videos = self.pipeline(
        prompt=self.prompt,
        negative_prompt=self.negative_prompt,
        height=self.config.height,
        width=self.config.width,
        num_frames=self.config.num_frames,
        num_inference_steps=self.config.num_inference_steps,
        guidance_scale_low=self.config.guidance_scale_low,
        guidance_scale_high=self.config.guidance_scale_high,
        use_magcache=use_magcache,
    )
    return videos, time.perf_counter() - t0

  def test_magcache_speedup_and_fidelity(self):
    videos_baseline, t_baseline = self._run_pipeline(use_magcache=False)
    videos_cached, t_cached = self._run_pipeline(use_magcache=True)

    speedup = t_baseline / t_cached
    max_logging.log(f"Baseline: {t_baseline:.2f}s, MagCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    self.assertGreater(speedup, 1.0, f"MagCache should be faster. Speedup={speedup:.3f}x")

    v1 = np.array(videos_baseline[0], dtype=np.float64)
    v2 = np.array(videos_cached[0], dtype=np.float64)

    mse = np.mean((v1 - v2) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    max_logging.log(f"PSNR: {psnr:.2f} dB")
    self.assertGreaterEqual(psnr, 30.0, f"PSNR={psnr:.2f} dB < 30 dB")

    C1, C2 = 0.01**2, 0.03**2
    ssim_scores = []
    for f in range(v1.shape[0]):
      mu1, mu2 = np.mean(v1[f]), np.mean(v2[f])
      sigma1_sq, sigma2_sq = np.var(v1[f]), np.var(v2[f])
      sigma12 = np.mean((v1[f] - mu1) * (v2[f] - mu2))
      ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
      ssim_scores.append(float(ssim))
    mean_ssim = np.mean(ssim_scores)
    max_logging.log(f"SSIM: mean={mean_ssim:.4f}, min={np.min(ssim_scores):.4f}")
    self.assertGreaterEqual(mean_ssim, 0.95, f"Mean SSIM={mean_ssim:.4f} < 0.95")

  @classmethod
  def tearDownClass(cls):
    del cls.pipeline
    import gc

    gc.collect()


if __name__ == "__main__":
  absltest.main()
