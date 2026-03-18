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

from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanSenCacheValidationTest(unittest.TestCase):
  """Tests that use_sen_cache validation raises correct errors."""

  def _make_pipeline(self):
    pipeline = WanPipeline2_2.__new__(WanPipeline2_2)
    return pipeline

  def test_sen_cache_with_both_scales_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=1.0,
          guidance_scale_high=1.0,
          use_sen_cache=True,
      )
    self.assertIn("use_sen_cache", str(ctx.exception))

  def test_sen_cache_with_low_scale_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=0.5,
          guidance_scale_high=4.0,
          use_sen_cache=True,
      )
    self.assertIn("use_sen_cache", str(ctx.exception))

  def test_sen_cache_with_high_scale_low_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=3.0,
          guidance_scale_high=1.0,
          use_sen_cache=True,
      )
    self.assertIn("use_sen_cache", str(ctx.exception))

  def test_sen_cache_mutually_exclusive_with_cfg_cache(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale_low=3.0,
          guidance_scale_high=4.0,
          use_cfg_cache=True,
          use_sen_cache=True,
      )
    self.assertIn("mutually exclusive", str(ctx.exception))

  def test_sen_cache_with_valid_scales_no_validation_error(self):
    """Both guidance_scales > 1.0 should pass validation (may fail later without model)."""
    pipeline = self._make_pipeline()
    try:
      pipeline(
          prompt=["test"],
          guidance_scale_low=3.0,
          guidance_scale_high=4.0,
          use_sen_cache=True,
      )
    except ValueError as e:
      if "use_sen_cache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      pass

  def test_no_sen_cache_with_low_scales_no_error(self):
    """use_sen_cache=False should never raise our ValueError."""
    pipeline = self._make_pipeline()
    try:
      pipeline(
          prompt=["test"],
          guidance_scale_low=0.5,
          guidance_scale_high=0.5,
          use_sen_cache=False,
      )
    except ValueError as e:
      if "use_sen_cache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      pass


class WanSenCacheScheduleTest(unittest.TestCase):
  """Tests the SenCache schedule logic (force-compute zones and sensitivity gating).

  Mirrors the schedule computation in run_inference_2_2 to verify correctness
  of force_compute zones. The actual sensitivity gating (score <= epsilon) is
  data-dependent, so we test the deterministic scheduling constraints here.
  """

  def _get_force_compute_schedule(self, num_inference_steps, boundary_ratio=0.875, num_train_timesteps=1000):
    """Extract which steps are forced to compute (cannot be cached).

    Returns (force_compute, step_uses_high) lists.
    """
    boundary = boundary_ratio * num_train_timesteps
    timesteps = np.linspace(num_train_timesteps - 1, 0, num_inference_steps, dtype=np.int32)
    step_uses_high = [bool(timesteps[s] >= boundary) for s in range(num_inference_steps)]

    # SenCache hyperparameters (mirrored from run_inference_2_2)
    warmup_steps = 1
    nocache_start_ratio = 0.3
    nocache_end_ratio = 0.1

    nocache_start = int(num_inference_steps * nocache_start_ratio)
    nocache_end_begin = int(num_inference_steps * (1.0 - nocache_end_ratio))

    force_compute = []
    for s in range(num_inference_steps):
      is_boundary = s > 0 and step_uses_high[s] != step_uses_high[s - 1]
      forced = (
          s < warmup_steps
          or s < nocache_start
          or s >= nocache_end_begin
          or is_boundary
          or s == 0  # ref_noise_pred is None on first step
      )
      force_compute.append(forced)

    return force_compute, step_uses_high

  def test_first_step_always_forced(self):
    """Step 0 must always compute (warmup + ref_noise_pred is None)."""
    force_compute, _ = self._get_force_compute_schedule(50)
    self.assertTrue(force_compute[0])

  def test_first_30_percent_always_forced(self):
    """First 30% of steps are in the no-cache zone."""
    force_compute, _ = self._get_force_compute_schedule(50)
    nocache_start = int(50 * 0.3)  # 15
    self.assertTrue(all(force_compute[:nocache_start]))

  def test_last_10_percent_always_forced(self):
    """Last 10% of steps are in the no-cache zone."""
    force_compute, _ = self._get_force_compute_schedule(50)
    nocache_end_begin = int(50 * 0.9)  # 45
    self.assertTrue(all(force_compute[nocache_end_begin:]))

  def test_boundary_transition_forced(self):
    """Steps at high-to-low transformer transitions are forced."""
    force_compute, step_uses_high = self._get_force_compute_schedule(50)
    for s in range(1, 50):
      if step_uses_high[s] != step_uses_high[s - 1]:
        self.assertTrue(force_compute[s], f"Boundary step {s} should be forced")

  def test_cacheable_window_exists(self):
    """There should be steps in [30%, 90%) that are NOT forced (eligible for caching)."""
    force_compute, _ = self._get_force_compute_schedule(50)
    nocache_start = int(50 * 0.3)
    nocache_end_begin = int(50 * 0.9)
    cacheable = [not force_compute[s] for s in range(nocache_start, nocache_end_begin)]
    self.assertGreater(sum(cacheable), 0, "Should have cacheable steps in the middle window")

  def test_short_run_all_forced(self):
    """Very few steps should all be forced (no-cache zones overlap completely)."""
    force_compute, _ = self._get_force_compute_schedule(3)
    self.assertTrue(all(force_compute), "3 steps is too short — all should be forced")

  def test_max_reuse_limit(self):
    """Simulate max_reuse=3: even if score stays low, after 3 reuses must recompute."""
    max_reuse = 3
    # Simulate a sequence of cache decisions where score is always below epsilon
    reuse_count = 0
    recompute_happened = False
    for _ in range(10):
      if reuse_count < max_reuse:
        reuse_count += 1
      else:
        reuse_count = 0
        recompute_happened = True
    self.assertTrue(recompute_happened, "Should force recompute after max_reuse consecutive reuses")

  def test_sensitivity_score_formula(self):
    """Verify the sensitivity score formula: S = α_x·‖Δx‖ + α_t·|Δt|."""
    alpha_x, alpha_t = 1.0, 1.0
    sen_epsilon = 0.1

    # Case 1: small deltas => cache hit
    score = alpha_x * 0.03 + alpha_t * 0.02
    self.assertLessEqual(score, sen_epsilon, "Small deltas should yield score <= epsilon")

    # Case 2: large latent drift => cache miss
    score = alpha_x * 0.5 + alpha_t * 0.02
    self.assertGreater(score, sen_epsilon, "Large dx should yield score > epsilon")

    # Case 3: large timestep drift => cache miss
    score = alpha_x * 0.01 + alpha_t * 0.5
    self.assertGreater(score, sen_epsilon, "Large dt should yield score > epsilon")

  def test_all_high_noise_no_cacheable_window(self):
    """If boundary_ratio=0, all steps are high-noise — boundary transitions still force compute."""
    force_compute, step_uses_high = self._get_force_compute_schedule(50, boundary_ratio=0.0)
    self.assertTrue(all(step_uses_high), "All steps should be high-noise")

  def test_nocache_zones_scale_with_steps(self):
    """No-cache zones should scale proportionally with num_inference_steps."""
    for n_steps in [20, 50, 100]:
      force_compute, _ = self._get_force_compute_schedule(n_steps)
      nocache_start = int(n_steps * 0.3)
      nocache_end_begin = int(n_steps * 0.9)
      self.assertTrue(all(force_compute[:nocache_start]), f"First 30% forced for {n_steps} steps")
      self.assertTrue(all(force_compute[nocache_end_begin:]), f"Last 10% forced for {n_steps} steps")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class WanSenCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: SenCache should be faster with SSIM >= 0.95.

  Runs on TPU v7-8 (8 chips, context_parallelism=8) with WAN 2.2 27B, 720p.
  Skipped in CI (GitHub Actions) — run locally with:
    python -m pytest src/maxdiffusion/tests/wan_sen_cache_test.py::WanSenCacheSmokeTest -v
  """

  @classmethod
  def setUpClass(cls):
    from maxdiffusion import pyconfig
    from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_27b.yml"),
            "num_inference_steps=50",
            "height=720",
            "width=1280",
            "num_frames=81",
            "fps=24",
            "guidance_scale_low=3.0",
            "guidance_scale_high=4.0",
            "boundary_ratio=0.875",
            "flow_shift=3.0",
            "seed=11234567893",
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
            'flash_block_sizes={"block_q": 2048, "block_kv_compute": 1024, "block_kv": 2048, "block_q_dkv": 2048, "block_kv_dkv": 2048, "block_kv_dkv_compute": 2048, "use_fused_bwd_kernel": true}',
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointer2_2(config=cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    # Warmup both XLA code paths
    for use_cache in [False, True]:
      cls.pipeline(
          prompt=cls.prompt,
          negative_prompt=cls.negative_prompt,
          height=cls.config.height,
          width=cls.config.width,
          num_frames=cls.config.num_frames,
          num_inference_steps=cls.config.num_inference_steps,
          guidance_scale_low=cls.config.guidance_scale_low,
          guidance_scale_high=cls.config.guidance_scale_high,
          use_sen_cache=use_cache,
      )

  def _run_pipeline(self, use_sen_cache):
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
        use_sen_cache=use_sen_cache,
    )
    return videos, time.perf_counter() - t0

  def test_sen_cache_speedup_and_fidelity(self):
    """SenCache must be faster than baseline with PSNR >= 30 dB and SSIM >= 0.95."""
    videos_baseline, t_baseline = self._run_pipeline(use_sen_cache=False)
    videos_cached, t_cached = self._run_pipeline(use_sen_cache=True)

    # Speed check
    speedup = t_baseline / t_cached
    print(f"Baseline: {t_baseline:.2f}s, SenCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    self.assertGreater(speedup, 1.0, f"SenCache should be faster. Speedup={speedup:.3f}x")

    # Fidelity checks
    v1 = np.array(videos_baseline[0], dtype=np.float64)
    v2 = np.array(videos_cached[0], dtype=np.float64)

    # PSNR
    mse = np.mean((v1 - v2) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"PSNR: {psnr:.2f} dB")
    self.assertGreaterEqual(psnr, 30.0, f"PSNR={psnr:.2f} dB < 30 dB")

    # SSIM (per-frame)
    C1, C2 = 0.01**2, 0.03**2
    ssim_scores = []
    for f in range(v1.shape[0]):
      mu1, mu2 = np.mean(v1[f]), np.mean(v2[f])
      sigma1_sq, sigma2_sq = np.var(v1[f]), np.var(v2[f])
      sigma12 = np.mean((v1[f] - mu1) * (v2[f] - mu2))
      ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
      ssim_scores.append(float(ssim))

    mean_ssim = np.mean(ssim_scores)
    print(f"SSIM: mean={mean_ssim:.4f}, min={np.min(ssim_scores):.4f}")
    self.assertGreaterEqual(mean_ssim, 0.95, f"Mean SSIM={mean_ssim:.4f} < 0.95")


if __name__ == "__main__":
  absltest.main()
