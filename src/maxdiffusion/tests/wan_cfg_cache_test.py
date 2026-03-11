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

from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanCfgCacheValidationTest(unittest.TestCase):
  """Tests that use_cfg_cache=True with guidance_scale <= 1.0 raises ValueError."""

  def _make_pipeline(self):
    """Create a WanPipeline2_1 instance with mocked internals."""
    pipeline = WanPipeline2_1.__new__(WanPipeline2_1)
    return pipeline

  def test_cfg_cache_with_guidance_scale_1_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale=1.0,
          use_cfg_cache=True,
      )
    self.assertIn("use_cfg_cache", str(ctx.exception))

  def test_cfg_cache_with_guidance_scale_0_raises(self):
    pipeline = self._make_pipeline()
    with self.assertRaises(ValueError) as ctx:
      pipeline(
          prompt=["test"],
          guidance_scale=0.0,
          use_cfg_cache=True,
      )
    self.assertIn("use_cfg_cache", str(ctx.exception))

  def test_cfg_cache_with_valid_guidance_scale_no_validation_error(self):
    """guidance_scale > 1.0 should pass validation (may fail later without model)."""
    pipeline = self._make_pipeline()
    try:
      pipeline(
          prompt=["test"],
          guidance_scale=5.0,
          use_cfg_cache=True,
      )
    except ValueError as e:
      if "use_cfg_cache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      # Other errors expected (no model loaded).
      pass

  def test_no_cfg_cache_with_low_guidance_no_error(self):
    """use_cfg_cache=False should never raise our ValueError regardless of guidance_scale."""
    pipeline = self._make_pipeline()
    try:
      pipeline(
          prompt=["test"],
          guidance_scale=0.5,
          use_cfg_cache=False,
      )
    except ValueError as e:
      if "use_cfg_cache" in str(e):
        self.fail(f"Unexpected validation error: {e}")
    except Exception:
      pass


class WanCfgCacheScheduleTest(unittest.TestCase):
  """Tests that CFG cache schedule produces the correct full/cache step pattern.

  Verifies the schedule logic in run_inference_2_1 without running any model.
  """

  def _get_cache_schedule(self, num_inference_steps, height=480):
    """Extract the cache schedule from run_inference_2_1's logic.

    Mirrors the schedule computation in run_inference_2_1 to verify correctness.
    """
    if height >= 720:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = int(num_inference_steps * 0.9)
    else:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = num_inference_steps - 2

    first_full_step_seen = False
    schedule = []
    for s in range(num_inference_steps):
      is_cache = (
          first_full_step_seen
          and s >= cfg_cache_start_step
          and s < cfg_cache_end_step
          and (s - cfg_cache_start_step) % cfg_cache_interval != 0
      )
      schedule.append(is_cache)
      if not is_cache:
        first_full_step_seen = True
    return schedule

  def test_480p_50_steps_schedule(self):
    """480p, 50 steps: cache starts at step 16, ends at step 48."""
    schedule = self._get_cache_schedule(50, height=480)
    self.assertEqual(len(schedule), 50)
    # First 16 steps should all be full CFG
    self.assertTrue(all(not s for s in schedule[:16]))
    # Last 2 steps should be full CFG
    self.assertTrue(all(not s for s in schedule[48:]))
    # There should be some cache steps in the middle
    cache_count = sum(schedule)
    self.assertGreater(cache_count, 0, "Should have cache steps in 480p/50 steps")

  def test_720p_50_steps_schedule(self):
    """720p, 50 steps: more conservative — cache ends at step 45."""
    schedule = self._get_cache_schedule(50, height=720)
    self.assertEqual(len(schedule), 50)
    # First 16 steps should all be full CFG
    self.assertTrue(all(not s for s in schedule[:16]))
    # Last 10% of steps (45-49) should be full CFG
    self.assertTrue(all(not s for s in schedule[45:]))
    cache_count = sum(schedule)
    self.assertGreater(cache_count, 0, "Should have cache steps in 720p/50 steps")

  def test_720p_has_fewer_cache_steps_than_480p(self):
    """720p should be more conservative (fewer cache steps) than 480p."""
    schedule_480 = self._get_cache_schedule(50, height=480)
    schedule_720 = self._get_cache_schedule(50, height=720)
    self.assertGreater(sum(schedule_480), sum(schedule_720))

  def test_cache_interval_is_5(self):
    """Every 5th step after start should be a full CFG step (not cached)."""
    schedule = self._get_cache_schedule(50, height=480)
    start = int(50 / 3)  # 16
    end = 48
    for s in range(start, end):
      if (s - start) % 5 == 0:
        self.assertFalse(schedule[s], f"Step {s} should be full CFG (interval=5)")

  def test_short_run_no_cache(self):
    """Very few steps should have no cache steps."""
    schedule = self._get_cache_schedule(3, height=480)
    self.assertEqual(sum(schedule), 0, "3 steps is too short for cache")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class WanCfgCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: CFG cache should be faster with SSIM >= 0.95.

  Runs on TPU v7-8 (8 chips, context_parallelism=8) with WAN 2.1 14B, 720p.
  Skipped in CI (GitHub Actions) — run locally with:
    python -m pytest src/maxdiffusion/tests/wan_cfg_cache_test.py::WanCfgCacheSmokeTest -v
  """

  @classmethod
  def setUpClass(cls):
    from maxdiffusion import pyconfig
    from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
            "num_inference_steps=50",
            "height=720",
            "width=1280",
            "num_frames=81",
            "fps=24",
            "guidance_scale=5.0",
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
    checkpoint_loader = WanCheckpointer2_1(config=cls.config)
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
          guidance_scale=cls.config.guidance_scale,
          use_cfg_cache=use_cache,
      )

  def _run_pipeline(self, use_cfg_cache):
    t0 = time.perf_counter()
    videos = self.pipeline(
        prompt=self.prompt,
        negative_prompt=self.negative_prompt,
        height=self.config.height,
        width=self.config.width,
        num_frames=self.config.num_frames,
        num_inference_steps=self.config.num_inference_steps,
        guidance_scale=self.config.guidance_scale,
        use_cfg_cache=use_cfg_cache,
    )
    return videos, time.perf_counter() - t0

  def test_cfg_cache_speedup_and_fidelity(self):
    """CFG cache must be faster than baseline with mean SSIM >= 0.95."""
    videos_baseline, t_baseline = self._run_pipeline(use_cfg_cache=False)
    videos_cached, t_cached = self._run_pipeline(use_cfg_cache=True)

    # Speed check
    speedup = t_baseline / t_cached
    print(f"Baseline: {t_baseline:.2f}s, CFG cache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    self.assertGreater(speedup, 1.0, f"CFG cache should be faster. Speedup={speedup:.3f}x")

    # Fidelity check (per-frame SSIM)
    v1 = np.array(videos_baseline[0], dtype=np.float64)
    v2 = np.array(videos_cached[0], dtype=np.float64)

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
