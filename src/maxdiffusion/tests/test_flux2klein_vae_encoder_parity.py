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
import flax
from flax import linen as nn

from maxdiffusion.models.vae_flax import FlaxAutoencoderKL
from maxdiffusion.models.flux.util import (
    load_and_convert_vae_weights,
    patchify_latents,
    unpatchify_latents,
    prepare_multi_image_ids,
)


class TestFlux2KleinVAEEncoderParity(unittest.TestCase):
  """Verifies VAE Encoder and Helper operations against golden PyTorch Diffusers states."""

  def setUp(self):
    self.golden_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data/golden_image_edit_data") else "golden_image_edit_data"
    self.assertTrue(os.path.exists(self.golden_dir), f"Golden data directory not found: {self.golden_dir}")

    # Find VAE safetensors
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

  def test_vae_encoder_and_norm_parity(self):
    """Tests VAE encoder forward pass, patchification, and BN normalization against golden PyTorch tensors."""
    print("\n" + "=" * 80)
    print("🧪 Running VAE Encoder & Normalization Parity Test against Diffusers Golden Data...")
    print("=" * 80)

    # 1. Instantiate JAX FlaxAutoencoderKL with encoder + decoder
    vae = FlaxAutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=32,
        norm_num_groups=32,
        sample_size=512,
        use_quant_conv=True,
        use_post_quant_conv=True,
        dtype=jnp.float32,
    )

    # 2. Initialize and Load Weights
    dummy_input = jnp.zeros((1, 3, 512, 512), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    initial_params = vae.init(key, dummy_input)["params"]

    vae_params, bn_mean, bn_std = load_and_convert_vae_weights(
        self.vae_path, initial_params, dtype=jnp.float32
    )

    # 3. Load preprocessed image 0
    img0_np = np.load(os.path.join(self.golden_dir, "preprocessed_image_0.npy")) # (1, 3, 512, 512)
    img0_jax = jnp.array(img0_np, dtype=jnp.float32)

    # 4. JAX VAE Encode
    # In JAX FlaxAutoencoderKL.encode:
    # input is (B, 3, H, W), internally transposed to (B, H, W, 3), output moments has shape (B, H/8, W/8, 64)
    # posterior.mode() gives (B, H/8, W/8, 32)
    posterior = vae.apply({"params": vae_params}, img0_jax, method=vae.encode)
    raw_latents_hwc = posterior.latent_dist.mode() # (1, 64, 64, 32)
    raw_latents_chw = jnp.transpose(raw_latents_hwc, (0, 3, 1, 2)) # (1, 32, 64, 64)

    # Compare raw latents against golden PyTorch raw latents
    golden_raw_all = np.load(os.path.join(self.golden_dir, "ref_latents_raw.npy")) # (4, 1, 32, 64, 64)
    golden_raw_0 = golden_raw_all[0] # (1, 32, 64, 64)

    raw_diff = np.abs(np.array(raw_latents_chw) - golden_raw_0)
    max_raw_diff = np.max(raw_diff)
    mean_raw_diff = np.mean(raw_diff)
    print(f" -> Raw VAE Latents: Max Diff = {max_raw_diff:.6e}, Mean Diff = {mean_raw_diff:.6e}")
    self.assertLess(max_raw_diff, 1e-3, "Raw VAE latent difference exceeds tolerance!")

    # 5. Patchify
    patchified_jax = patchify_latents(raw_latents_chw) # (1, 128, 32, 32)
    golden_patchified_all = np.load(os.path.join(self.golden_dir, "ref_latents_patchified.npy"))
    golden_patch_0 = golden_patchified_all[0]

    patch_diff = np.abs(np.array(patchified_jax) - golden_patch_0)
    max_patch_diff = np.max(patch_diff)
    print(f" -> Patchified Latents: Max Diff = {max_patch_diff:.6e}")
    self.assertLess(max_patch_diff, 1e-3, "Patchified latent difference exceeds tolerance!")

    # 6. BN Normalization
    norm_latents_jax = (patchified_jax - bn_mean) / bn_std # (1, 128, 32, 32)
    golden_norm_all = np.load(os.path.join(self.golden_dir, "ref_latents_normalized.npy"))
    golden_norm_0 = golden_norm_all[0]

    norm_diff = np.abs(np.array(norm_latents_jax) - golden_norm_0)
    max_norm_diff = np.max(norm_diff)
    print(f" -> Normalized Latents: Max Diff = {max_norm_diff:.6e}")
    self.assertLess(max_norm_diff, 1e-3, "Normalized latent difference exceeds tolerance!")

    # 7. Multi-Image Position IDs
    # Compute for all 4 images
    golden_all_norm = [golden_norm_all[i] for i in range(4)]
    multi_img_ids_jax = prepare_multi_image_ids(golden_all_norm, scale=10) # (1, 4096, 4)
    golden_img_ids = np.load(os.path.join(self.golden_dir, "image_latent_ids.npy")) # (1, 4096, 4)

    self.assertEqual(multi_img_ids_jax.shape, golden_img_ids.shape)
    ids_diff = np.max(np.abs(np.array(multi_img_ids_jax) - golden_img_ids))
    print(f" -> Multi-Image 4D Position IDs (T=10..40): Exact Match Diff = {ids_diff}")
    self.assertEqual(ids_diff, 0, "Multi-image position IDs do not match Diffusers exactly!")

    # 8. Unpatchify test
    unpatch_jax = unpatchify_latents(patchified_jax)
    unpatch_diff = np.max(np.abs(np.array(unpatch_jax) - np.array(raw_latents_chw)))
    print(f" -> Patchify/Unpatchify Roundtrip Invertibility Diff = {unpatch_diff}")
    self.assertEqual(unpatch_diff, 0, "Patchify/Unpatchify roundtrip is not perfectly lossless!")

    print("=" * 80)
    print("✅ All VAE Encoder, Patchification, Normalization & RoPE ID Parity Tests PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
