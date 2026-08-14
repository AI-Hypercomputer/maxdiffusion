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
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import prepare_image_latents


class TestFlux2KleinImageConditioningParity(unittest.TestCase):
  """Verifies Phase 3: Multi-image reference conditioning and token packing."""

  def setUp(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    self.golden_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data/golden_image_edit_data") else "golden_image_edit_data"
    self.assertTrue(os.path.exists(self.golden_dir), f"Golden data directory not found: {self.golden_dir}")

    candidates = [
        "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        "/mnt/hyperdisk_weights/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"),
    ]
    self.vae_path = None
    for c in candidates:
      if os.path.exists(c):
        snaps = os.listdir(c)
        if snaps:
          self.vae_path = os.path.join(c, snaps[0], "vae", "diffusion_pytorch_model.safetensors")
          if os.path.exists(self.vae_path):
            break
    self.assertIsNotNone(self.vae_path, "VAE safetensors file not found!")

  def test_multi_image_conditioning_parity(self):
    """Tests prepare_image_latents with 4 reference images against Diffusers golden states."""
    print("\n" + "=" * 80)
    print("🧪 Running Phase 3: Multi-Image Reference Conditioning & Packing Parity Test...")
    print("=" * 80)

    # 1. Instantiate NNX VAE & Load Weights
    nnx_vae = NNXAutoencoderKLFlux2(dtype=jnp.float32, param_dtype=jnp.float32)
    bn_mean, bn_std = load_and_convert_flux2klein_nnx_vae_weights(
        self.vae_path, nnx_vae, dtype=jnp.float32
    )

    # 2. Load 4 preprocessed reference images
    ref_images = []
    for i in range(4):
      img_path = os.path.join(self.golden_dir, f"preprocessed_image_{i}.npy")
      self.assertTrue(os.path.exists(img_path), f"Missing golden image {img_path}")
      img_np = np.load(img_path)
      ref_images.append(jnp.array(img_np, dtype=jnp.float32))

    # 3. Call prepare_image_latents
    image_latents_concat, image_latent_ids = prepare_image_latents(
        nnx_vae, ref_images, bn_mean, bn_std, scale=10
    )

    # 4. Verify Shapes
    # 4 images of 512x512 -> 4 * (32 * 32) = 4096 tokens, each of dim 128
    self.assertEqual(image_latents_concat.shape, (1, 4096, 128))
    self.assertEqual(image_latent_ids.shape, (1, 4096, 4))
    print(f" -> Output image_latents_concat shape: {image_latents_concat.shape}")
    print(f" -> Output image_latent_ids shape: {image_latent_ids.shape}")

    # 5. Verify 4D RoPE IDs against Diffusers Golden Ground Truth
    golden_ids = np.load(os.path.join(self.golden_dir, "image_latent_ids.npy"))
    ids_diff = np.max(np.abs(np.array(image_latent_ids) - golden_ids))
    print(f" -> 4D RoPE Position IDs (T=10..40): Exact Match Diff = {ids_diff}")
    self.assertEqual(ids_diff, 0, "Position IDs do not match Diffusers exactly!")

    # 6. Verify Concatenated Latent Tokens against Diffusers Golden Ground Truth
    golden_latents = np.load(os.path.join(self.golden_dir, "image_latents_concat.npy"))
    latents_diff = np.abs(np.array(image_latents_concat) - golden_latents)
    max_latents_diff = np.max(latents_diff)
    mean_latents_diff = np.mean(latents_diff)
    print(f" -> Concatenated Latent Tokens (4096x128): Max Diff = {max_latents_diff:.6e}, Mean Diff = {mean_latents_diff:.6e}")
    self.assertLess(max_latents_diff, 0.25, "Concatenated reference tokens exceed bfloat16 tolerance!")
    self.assertLess(mean_latents_diff, 0.015, "Mean difference exceeds bfloat16 tolerance!")

    print("=" * 80)
    print("✅ Phase 3: Multi-Image Conditioning & Packing Parity PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
