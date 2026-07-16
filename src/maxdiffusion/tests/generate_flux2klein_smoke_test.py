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
import unittest
import pytest

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from maxdiffusion import pyconfig
from maxdiffusion import generate_flux2klein

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT = "anime corgi eating sushi in the mountains"


class GenerateFlux2KleinSmokeTest(unittest.TestCase):
  """End-to-end smoke test for Flux2Klein 4B and 9B."""

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_flux2klein_4b_smoke(self):
    """End-to-end smoke test for Flux.2-klein-4B image generation at 1024x1024."""
    ref_path = os.path.join(THIS_DIR, "images", "ref_flux2klein_4b.png")
    self.assertTrue(os.path.exists(ref_path), f"Reference image not found: {ref_path}")
    base_image = np.array(Image.open(ref_path)).astype(np.uint8)

    output_dir = "/mnt/data/smoke_test_4b" if os.path.exists("/mnt/data") else "/tmp/smoke_test_4b"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "flux2klein_generated_image.png")
    if os.path.exists(out_path):
      os.remove(out_path)

    pyconfig._config = None
    pyconfig.config = None
    args = [
        None,
        os.path.join(THIS_DIR, "..", "configs", "base_flux2klein.yml"),
        "run_name=smoke_test_4b",
        f"output_dir={output_dir}",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
        f"prompt={PROMPT}",
        "height=512",
        "width=512",
        "batch_size=1",
        "seed=42",
        "ici_fsdp_parallelism=-1",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
    ]

    generate_flux2klein.main(args)

    self.assertTrue(os.path.exists(out_path), "Smoke test 4B failed to produce output image!")
    test_image = np.array(Image.open(out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 4B] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.75)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_flux2klein_9b_smoke(self):
    """End-to-end smoke test for Flux.2-klein-9B image generation at 1024x1024."""
    ref_path = os.path.join(THIS_DIR, "images", "ref_flux2klein_9b.png")
    self.assertTrue(os.path.exists(ref_path), f"Reference image not found: {ref_path}")
    base_image = np.array(Image.open(ref_path)).astype(np.uint8)

    output_dir = "/mnt/data/smoke_test_9b" if os.path.exists("/mnt/data") else "/tmp/smoke_test_9b"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "flux2klein_generated_image.png")
    if os.path.exists(out_path):
      os.remove(out_path)

    pyconfig._config = None
    pyconfig.config = None
    args = [
        None,
        os.path.join(THIS_DIR, "..", "configs", "base_flux2klein_9B.yml"),
        "run_name=smoke_test_9b",
        f"output_dir={output_dir}",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
        f"prompt={PROMPT}",
        "height=512",
        "width=512",
        "batch_size=1",
        "seed=42",
        "ici_fsdp_parallelism=-1",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
    ]

    generate_flux2klein.main(args)

    self.assertTrue(os.path.exists(out_path), "Smoke test 9B failed to produce output image!")
    test_image = np.array(Image.open(out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 9B] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.80)


if __name__ == "__main__":
  unittest.main()
