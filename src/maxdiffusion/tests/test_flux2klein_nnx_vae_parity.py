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
import torch
import jax
import jax.numpy as jnp
from flax import nnx
from diffusers import AutoencoderKLFlux2

from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import (
    patchify_latents,
    unpatchify_latents,
    prepare_multi_image_ids,
)


class TestFlux2KleinNNXVAEParity(unittest.TestCase):
  """Verifies NNXAutoencoderKLFlux2 against golden PyTorch Diffusers states."""

  def setUp(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    self.golden_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data/golden_image_edit_data") else "golden_image_edit_data"

    candidates = [
        "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        "/mnt/hyperdisk_weights/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"),
    ]
    self.vae_dir = None
    for c in candidates:
      if os.path.exists(c):
        snaps = os.listdir(c)
        if snaps:
          self.vae_dir = os.path.join(c, snaps[0], "vae")
          self.vae_path = os.path.join(self.vae_dir, "diffusion_pytorch_model.safetensors")
          if os.path.exists(self.vae_path):
            break
    self.assertIsNotNone(self.vae_path, "VAE safetensors file not found!")

  def test_nnx_vae_float32_parity(self):
    """Tests float32 mathematical parity between PyTorch AutoencoderKLFlux2 and NNXAutoencoderKLFlux2."""
    print("\n" + "=" * 80)
    print("🧪 Running Exact Float32 Parity Test (PyTorch vs NNX)...")
    print("=" * 80)

    # 1. Load PyTorch VAE in float32
    pt_vae = AutoencoderKLFlux2.from_pretrained(self.vae_dir, torch_dtype=torch.float32).eval()

    # 2. Load NNX VAE in float32
    nnx_vae = NNXAutoencoderKLFlux2(dtype=jnp.float32, param_dtype=jnp.float32)
    load_and_convert_flux2klein_nnx_vae_weights(self.vae_path, nnx_vae, dtype=jnp.float32)

    # 3. Create test input
    np.random.seed(42)
    img_np = np.random.randn(1, 3, 512, 512).astype(np.float32)
    pt_img = torch.tensor(img_np, dtype=torch.float32)

    # 4. Compare Encoder
    with torch.no_grad():
      pt_raw = pt_vae.encode(pt_img).latent_dist.mode().numpy()
    jax_raw = np.array(nnx_vae.encode(jnp.array(img_np, dtype=jnp.float32)))

    enc_diff = np.abs(jax_raw - pt_raw)
    max_enc_diff = np.max(enc_diff)
    mean_enc_diff = np.mean(enc_diff)
    print(f" -> Float32 VAE Encode: Max Diff = {max_enc_diff:.6e}, Mean Diff = {mean_enc_diff:.6e}")
    self.assertLess(max_enc_diff, 1e-4, "Float32 VAE Encoder difference exceeds tolerance!")

    # 5. Compare Decoder
    with torch.no_grad():
      pt_dec = pt_vae.decode(torch.tensor(pt_raw, dtype=torch.float32)).sample.numpy()
    jax_dec = np.array(nnx_vae.decode(jnp.array(pt_raw, dtype=jnp.float32)))

    dec_diff = np.abs(jax_dec - pt_dec)
    max_dec_diff = np.max(dec_diff)
    mean_dec_diff = np.mean(dec_diff)
    print(f" -> Float32 VAE Decode: Max Diff = {max_dec_diff:.6e}, Mean Diff = {mean_dec_diff:.6e}")
    self.assertLess(max_dec_diff, 1e-4, "Float32 VAE Decoder difference exceeds tolerance!")

  def test_helpers_parity(self):
    """Tests patchify, unpatchify, and multi-image RoPE position IDs."""
    print("\n" + "=" * 80)
    print("🧪 Running Image Edit Helpers Parity Test...")
    print("=" * 80)

    # Invertibility
    dummy = jnp.zeros((1, 32, 64, 64), dtype=jnp.float32)
    p = patchify_latents(dummy)
    self.assertEqual(p.shape, (1, 128, 32, 32))
    u = unpatchify_latents(p)
    self.assertEqual(u.shape, (1, 32, 64, 64))
    self.assertEqual(np.max(np.abs(np.array(u) - np.array(dummy))), 0)

    if os.path.exists(self.golden_dir):
      golden_all_norm = [np.load(os.path.join(self.golden_dir, "ref_latents_normalized.npy"))[i] for i in range(4)]
      multi_img_ids_jax = prepare_multi_image_ids(golden_all_norm, scale=10)
      golden_img_ids = np.load(os.path.join(self.golden_dir, "image_latent_ids.npy"))
      ids_diff = np.max(np.abs(np.array(multi_img_ids_jax) - golden_img_ids))
      print(f" -> Multi-Image 4D Position IDs (T=10..40): Exact Match Diff = {ids_diff}")
      self.assertEqual(ids_diff, 0)

    print("=" * 80)
    print("✅ All NNX VAE & Helper Tests PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
