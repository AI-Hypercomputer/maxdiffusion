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
import jax

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
    ref_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_4b.png")
    self.assertTrue(os.path.exists(ref_path), f"Reference image not found: {ref_path}")
    base_image = np.array(Image.open(ref_path)).astype(np.uint8)

    output_dir = "/tmp/smoke_test_4b"
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
        f"prompt={PROMPT}",
        "height=512",
        "width=512",
        f"per_device_batch_size={1.0 / jax.device_count()}",
        "seed=42",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "num_reps=5",
        "text_encoder_attention=dot_product",
    ]

    generate_flux2klein.main(args)

    rep_out_path = os.path.join(output_dir, "rep_1_flux2klein_generated_image.png")
    final_out_path = rep_out_path if os.path.exists(rep_out_path) else out_path
    self.assertTrue(os.path.exists(final_out_path), "Smoke test 4B failed to produce output image!")
    test_image = np.array(Image.open(final_out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 4B] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.8)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_flux2klein_9b_smoke(self):
    """End-to-end smoke test for Flux.2-klein-9B image generation at 1024x1024."""
    ref_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_9b.png")
    self.assertTrue(os.path.exists(ref_path), f"Reference image not found: {ref_path}")
    base_image = np.array(Image.open(ref_path)).astype(np.uint8)

    output_dir = "/tmp/smoke_test_9b"
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
        f"prompt={PROMPT}",
        "height=512",
        "width=512",
        f"per_device_batch_size={1.0 / jax.device_count()}",
        "seed=42",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "num_reps=5",
        "text_encoder_attention=dot_product",
    ]

    generate_flux2klein.main(args)

    rep_out_path = os.path.join(output_dir, "rep_1_flux2klein_generated_image.png")
    final_out_path = rep_out_path if os.path.exists(rep_out_path) else out_path
    self.assertTrue(os.path.exists(final_out_path), "Smoke test 9B failed to produce output image!")
    test_image = np.array(Image.open(final_out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 9B] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.8)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_flux2klein_4b_image_edit_smoke(self):
    """End-to-end smoke test for Flux.2-klein-4B image editing at 512x512."""
    ref_gold_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_4b_image_edit.png")
    self.assertTrue(os.path.exists(ref_gold_path), f"Golden reference image not found: {ref_gold_path}")
    base_image = np.array(Image.open(ref_gold_path)).astype(np.uint8)

    input_img_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_4b.png")
    self.assertTrue(os.path.exists(input_img_path), f"Input reference image not found: {input_img_path}")

    output_dir = "/tmp/smoke_test_image_edit_4b"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "flux2klein_generated_image.png")
    if os.path.exists(out_path):
      os.remove(out_path)

    pyconfig._config = None
    pyconfig.config = None
    args = [
        None,
        os.path.join(THIS_DIR, "..", "configs", "base_flux2klein.yml"),
        "run_name=smoke_test_image_edit_4b",
        f"output_dir={output_dir}",
        "jax_cache_dir=/tmp/cache_dir",
        f"image_paths=['{input_img_path}']",
        "prompt=change the lighting to evening",
        "height=512",
        "width=512",
        f"per_device_batch_size={1.0 / jax.device_count()}",
        "seed=42",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "num_reps=5",
        "text_encoder_attention=dot_product",
    ]

    generate_flux2klein.main(args)

    rep_out_path = os.path.join(output_dir, "rep_1_flux2klein_generated_image.png")
    final_out_path = rep_out_path if os.path.exists(rep_out_path) else out_path
    self.assertTrue(os.path.exists(final_out_path), "Smoke test 4B image edit failed to produce output image!")
    test_image = np.array(Image.open(final_out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 4B IMAGE EDIT] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.8)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_flux2klein_9b_image_edit_smoke(self):
    """End-to-end smoke test for Flux.2-klein-9B image editing at 512x512."""
    ref_gold_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_9b_image_edit.png")
    self.assertTrue(os.path.exists(ref_gold_path), f"Golden reference image not found: {ref_gold_path}")
    base_image = np.array(Image.open(ref_gold_path)).astype(np.uint8)

    input_img_path = os.path.join(THIS_DIR, "images", "flux2klein", "ref_flux2klein_4b.png")
    self.assertTrue(os.path.exists(input_img_path), f"Input reference image not found: {input_img_path}")

    output_dir = "/tmp/smoke_test_image_edit_9b"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "flux2klein_generated_image.png")
    if os.path.exists(out_path):
      os.remove(out_path)

    pyconfig._config = None
    pyconfig.config = None
    args = [
        None,
        os.path.join(THIS_DIR, "..", "configs", "base_flux2klein_9B.yml"),
        "run_name=smoke_test_image_edit_9b",
        f"output_dir={output_dir}",
        "jax_cache_dir=/tmp/cache_dir",
        f"image_paths=['{input_img_path}']",
        "prompt=change the lighting to evening",
        "height=512",
        "width=512",
        f"per_device_batch_size={1.0 / jax.device_count()}",
        "seed=42",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "num_reps=5",
        "text_encoder_attention=dot_product",
    ]

    generate_flux2klein.main(args)

    rep_out_path = os.path.join(output_dir, "rep_1_flux2klein_generated_image.png")
    final_out_path = rep_out_path if os.path.exists(rep_out_path) else out_path
    self.assertTrue(os.path.exists(final_out_path), "Smoke test 9B image edit failed to produce output image!")
    test_image = np.array(Image.open(final_out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST 9B IMAGE EDIT] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.8)


if __name__ == "__main__":
  unittest.main()
