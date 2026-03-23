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

import os
import time
import unittest

import numpy as np
import pytest

from maxdiffusion import pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
import jax

try:
  jax.distributed.initialize()
except Exception:
  pass

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_metrics(v1_baseline, v2_cached):
  """Helper to calculate Speedup, PSNR and SSIM between baseline and cached videos."""
  num_videos = len(v1_baseline)
  all_psnr = []
  all_ssim = []

  for i in range(num_videos):
    v1 = np.array(v1_baseline[i], dtype=np.float64)
    v2 = np.array(v2_cached[i], dtype=np.float64)

    # PSNR
    mse = np.mean((v1 - v2) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    all_psnr.append(psnr)

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
    all_ssim.append(mean_ssim)

  avg_psnr = np.mean(all_psnr)
  avg_ssim = np.mean(all_ssim)
  print(f"PSNR (avg of {num_videos} videos): {avg_psnr:.2f} dB")
  print(f"SSIM (avg of {num_videos} videos): mean={avg_ssim:.4f}")
  return avg_psnr, avg_ssim


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class Wan21T2VMagCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: MagCache for Wan 2.1 T2V 14B."""

  @classmethod
  def setUpClass(cls):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
            "num_inference_steps=50",
            "height=720",
            "width=1280",
            "num_frames=81",
            "seed=11234567893",
            "attention=flash",
            "remat_policy=FULL",
            "allow_split_physical_axes=True",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "per_device_batch_size=0.25",
            "ici_data_parallelism=4",
            "ici_fsdp_parallelism=1",
            "ici_context_parallelism=4",
            "ici_tensor_parallelism=1",
            "flash_min_seq_length=0",
            'flash_block_sizes={"block_q":3024,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":3024,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"use_fused_bwd_kernel":true}',
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointer2_1(cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    print("Warming up paths...")
    for use_cache in [False, True]:
      cls.pipeline(
          prompt=cls.prompt,
          negative_prompt=cls.negative_prompt,
          height=cls.config.height,
          width=cls.config.width,
          num_frames=cls.config.num_frames,
          num_inference_steps=cls.config.num_inference_steps,
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
        use_magcache=use_magcache,
    )
    return videos, time.perf_counter() - t0

  def test_magcache_speedup_and_fidelity(self):
    videos_baseline, t_baseline = self._run_pipeline(use_magcache=False)
    videos_cached, t_cached = self._run_pipeline(use_magcache=True)

    speedup = t_baseline / t_cached
    print(f"[Wan 2.1 T2V 14B] Baseline: {t_baseline:.2f}s, MagCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    calculate_metrics(videos_baseline, videos_cached)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class Wan22T2VMagCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: MagCache for Wan 2.2 T2V 14B/27B."""

  @classmethod
  def setUpClass(cls):
    from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_27b.yml"),
            "num_inference_steps=40",
            "height=720",
            "width=1280",
            "num_frames=81",
            "guidance_scale_low=3.0",
            "guidance_scale_high=4.0",
            "boundary_ratio=0.875",
            "seed=11234567893",
            "attention=flash",
            "remat_policy=FULL",
            "allow_split_physical_axes=True",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "per_device_batch_size=0.25",
            "ici_data_parallelism=4",
            "ici_fsdp_parallelism=1",
            "ici_context_parallelism=4",
            "ici_tensor_parallelism=1",
            "flash_min_seq_length=0",
            'flash_block_sizes={"block_q":3024,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":3024,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"use_fused_bwd_kernel":true}',
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointer2_2(cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    print("Warming up paths...")
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
    print(f"[Wan 2.2 T2V 27B] Baseline: {t_baseline:.2f}s, MagCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    calculate_metrics(videos_baseline, videos_cached)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class Wan21I2VMagCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: MagCache for Wan 2.1 I2V 14B."""

  @classmethod
  def setUpClass(cls):
    from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
    from maxdiffusion.utils.loading_utils import load_image

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_i2v_14b.yml"),
            "num_inference_steps=50",
            "height=720",
            "width=1280",
            "pretrained_model_name_or_path=Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            "num_frames=81",
            "seed=11234567893",
            "attention=flash",
            "remat_policy=FULL",
            "allow_split_physical_axes=True",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "per_device_batch_size=0.25",
            "ici_data_parallelism=4",
            "ici_fsdp_parallelism=1",
            "ici_context_parallelism=4",
            "ici_tensor_parallelism=1",
            "flash_min_seq_length=0",
            'flash_block_sizes={"block_q":3024,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":3024,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"use_fused_bwd_kernel":true}',
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointerI2V_2_1(cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.image = load_image(cls.config.image_url)
    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    print("Warming up paths...")
    for use_cache in [False, True]:
      cls.pipeline(
          prompt=cls.prompt,
          image=cls.image,
          negative_prompt=cls.negative_prompt,
          height=cls.config.height,
          width=cls.config.width,
          num_frames=cls.config.num_frames,
          num_inference_steps=cls.config.num_inference_steps,
          use_magcache=use_cache,
      )

  def _run_pipeline(self, use_magcache):
    t0 = time.perf_counter()
    videos = self.pipeline(
        prompt=self.prompt,
        image=self.image,
        negative_prompt=self.negative_prompt,
        height=self.config.height,
        width=self.config.width,
        num_frames=self.config.num_frames,
        num_inference_steps=self.config.num_inference_steps,
        use_magcache=use_magcache,
    )
    return videos, time.perf_counter() - t0

  def test_magcache_speedup_and_fidelity(self):
    videos_baseline, t_baseline = self._run_pipeline(use_magcache=False)
    videos_cached, t_cached = self._run_pipeline(use_magcache=True)

    speedup = t_baseline / t_cached
    print(f"[Wan 2.1 I2V 14B] Baseline: {t_baseline:.2f}s, MagCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    calculate_metrics(videos_baseline, videos_cached)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU v7-8 and model weights")
class Wan22I2VMagCacheSmokeTest(unittest.TestCase):
  """End-to-end smoke test: MagCache for Wan 2.2 I2V 27B."""

  @classmethod
  def setUpClass(cls):
    from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
    from maxdiffusion.utils.loading_utils import load_image

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_i2v_27b.yml"),
            "num_inference_steps=40",
            "height=720",
            "width=1280",
            "num_frames=81",
            "guidance_scale_low=3.0",
            "guidance_scale_high=4.0",
            "boundary_ratio=0.875",
            "seed=11234567893",
            "attention=flash",
            "remat_policy=FULL",
            "allow_split_physical_axes=True",
            "weights_dtype=bfloat16",
            "activations_dtype=bfloat16",
            "per_device_batch_size=0.25",
            "ici_data_parallelism=4",
            "ici_fsdp_parallelism=1",
            "ici_context_parallelism=4",
            "ici_tensor_parallelism=1",
            "flash_min_seq_length=0",
            'flash_block_sizes={"block_q":3024,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":3024,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"use_fused_bwd_kernel":true}',
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointerI2V_2_2(cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.image = load_image(cls.config.image_url)
    cls.prompt = [cls.config.prompt] * cls.config.global_batch_size_to_train_on
    cls.negative_prompt = [cls.config.negative_prompt] * cls.config.global_batch_size_to_train_on

    print("Warming up paths...")
    for use_cache in [False, True]:
      cls.pipeline(
          prompt=cls.prompt,
          image=cls.image,
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
        image=self.image,
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
    print(f"[Wan 2.2 I2V 27B] Baseline: {t_baseline:.2f}s, MagCache: {t_cached:.2f}s, Speedup: {speedup:.3f}x")
    calculate_metrics(videos_baseline, videos_cached)
