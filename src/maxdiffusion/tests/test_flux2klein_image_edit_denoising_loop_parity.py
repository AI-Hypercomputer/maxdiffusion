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
from PIL import Image
import torch
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2

from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import (
    load_and_convert_flux_klein_nnx_weights,
    unpatchify_latents,
)


class TestFlux2KleinImageEditDenoisingLoopParity(unittest.TestCase):
  """Verifies Phase 5: 4-Step Euler Denoising Loop and Final Decoded Image Parity in Float32."""

  def setUp(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    self.golden_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data/golden_image_edit_data") else "golden_image_edit_data"
    self.assertTrue(os.path.exists(self.golden_dir), f"Golden data directory not found: {self.golden_dir}")

    candidates = [
        "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        "/mnt/hyperdisk_weights/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"),
    ]
    self.model_dir = None
    for c in candidates:
      if os.path.exists(c):
        snaps = os.listdir(c)
        if snaps:
          self.model_dir = os.path.join(c, snaps[0])
          self.transformer_path = os.path.join(self.model_dir, "transformer")
          self.vae_dir = os.path.join(self.model_dir, "vae")
          self.vae_path = os.path.join(self.vae_dir, "diffusion_pytorch_model.safetensors")
          if os.path.exists(self.transformer_path) and os.path.exists(self.vae_path):
            break
    self.assertIsNotNone(self.model_dir, "FLUX.2-Klein 4B model directory not found!")

  def test_4step_euler_denoising_loop_parity(self):
    """Executes the full 4-step Euler trajectory in PyTorch and JAX in float32 and asserts step-by-step and pixel alignment."""
    print("\n" + "=" * 80)
    print("🧪 Running Phase 5: 4-Step Euler Denoising Loop & Visual Parity Test (FLUX.2-Klein 4B)...")
    print("=" * 80)

    # 1. Load PyTorch Diffusers Transformer & VAE in float32
    print(" -> Loading PyTorch Transformer and VAE in float32...")
    pt_t = Flux2Transformer2DModel.from_pretrained(
        self.transformer_path, torch_dtype=torch.float32
    ).eval()
    pt_vae = AutoencoderKLFlux2.from_pretrained(
        self.vae_dir, torch_dtype=torch.float32
    ).eval()

    # 2. Instantiate NNX Transformer & VAE in float32
    print(" -> Instantiating Flax NNX Transformer & VAE in float32...")
    rngs = nnx.Rngs(0)
    transformer = NNXFlux2KleinTransformer2DModel(
        rngs=rngs,
        patch_size=1,
        in_channels=128,
        num_layers=5,
        num_single_layers=20,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=7680,
        pooled_projection_dim=None,
        guidance_embeds=False,
        axes_dim=(32, 32, 32, 32),
        scale_shift_order="scale_shift",
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
    )
    transformer_state = load_and_convert_flux_klein_nnx_weights(
        self.transformer_path,
        nnx.state(transformer, nnx.Param),
        num_double_layers=5,
        num_single_layers=20,
        dtype=jnp.float32,
    )
    nnx.update(transformer, transformer_state)

    nnx_vae = NNXAutoencoderKLFlux2(dtype=jnp.float32, param_dtype=jnp.float32)
    bn_mean, bn_std = load_and_convert_flux2klein_nnx_vae_weights(
        self.vae_path, nnx_vae, dtype=jnp.float32
    )

    # 3. Load Inputs
    prompt_embeds = np.load(os.path.join(self.golden_dir, "prompt_embeds.npy"))
    text_ids = np.load(os.path.join(self.golden_dir, "text_ids.npy"))
    initial_noise = np.load(os.path.join(self.golden_dir, "initial_noise_latents.npy"))
    gen_ids = np.load(os.path.join(self.golden_dir, "gen_latent_ids.npy"))
    ref_latents_concat = np.load(os.path.join(self.golden_dir, "image_latents_concat.npy"))
    ref_latent_ids = np.load(os.path.join(self.golden_dir, "image_latent_ids.npy"))
    timesteps = np.load(os.path.join(self.golden_dir, "timesteps.npy"))

    joint_image_ids = np.concatenate([gen_ids, ref_latent_ids], axis=1)

    pt_latents = torch.tensor(initial_noise, dtype=torch.float32)
    joint_ids_pt = torch.tensor(joint_image_ids, dtype=torch.float32)
    ref_latents_pt = torch.tensor(ref_latents_concat, dtype=torch.float32)
    p_embed_pt = torch.tensor(prompt_embeds, dtype=torch.float32)
    t_ids_pt = torch.tensor(text_ids, dtype=torch.float32)

    jax_latents = jnp.array(initial_noise, dtype=jnp.float32)
    joint_ids_jax = jnp.array(joint_image_ids, dtype=jnp.float32)
    ref_latents_jax = jnp.array(ref_latents_concat, dtype=jnp.float32)
    p_embed_jax = jnp.array(prompt_embeds, dtype=jnp.float32)
    t_ids_jax = jnp.array(text_ids, dtype=jnp.float32)

    # 4. Synchronous Step-by-Step Denoising Loop
    for i, t in enumerate(timesteps):
      t_curr = t
      t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0.0

      # PyTorch Step
      with torch.no_grad():
        joint_img_pt = torch.cat([pt_latents, ref_latents_pt], dim=1)
        out_pt = pt_t(
            hidden_states=joint_img_pt,
            encoder_hidden_states=p_embed_pt,
            timestep=torch.tensor([t_curr / 1000.0], dtype=torch.float32),
            img_ids=joint_ids_pt,
            txt_ids=t_ids_pt,
        ).sample
        noise_pred_pt = out_pt[:, :pt_latents.shape[1], :]
        pt_latents = pt_latents + ((t_prev - t_curr) / 1000.0) * noise_pred_pt

      # JAX Step
      joint_img_jax = jnp.concatenate([jax_latents, ref_latents_jax], axis=1)
      out_jax = transformer(
          hidden_states=joint_img_jax,
          encoder_hidden_states=p_embed_jax,
          timestep=jnp.array([t_curr / 1000.0], dtype=jnp.float32),
          img_ids=joint_ids_jax,
          txt_ids=t_ids_jax,
          guidance=None,
      ).sample
      noise_pred_jax = out_jax[:, :jax_latents.shape[1], :]
      jax_latents = jax_latents + ((t_prev - t_curr) / 1000.0) * noise_pred_jax

      # Compare Step Output
      step_diff = np.abs(np.array(jax_latents) - pt_latents.numpy())
      max_step_diff = np.max(step_diff)
      mean_step_diff = np.mean(step_diff)
      print(f" -> Denoising Step {i+1}/4 (t={t_curr:.2f} -> {t_prev:.2f}): Max Diff = {max_step_diff:.6e}, Mean Diff = {mean_step_diff:.6e}")
      self.assertLess(max_step_diff, 1e-3, f"Step {i+1} latents exceed float32 tolerance!")

    # 5. Decode Both in Float32
    print(" -> Decoding PyTorch and JAX final latents through VAE Decoders...")
    
    # JAX Decode
    latents_spatial_jax = rearrange(jax_latents, "b (h w) c -> b c h w", h=32, w=32)
    latents_denorm_jax = latents_spatial_jax * bn_std + bn_mean
    unpatchified_jax = unpatchify_latents(latents_denorm_jax)
    decoded_jax = nnx_vae.decode(unpatchified_jax)
    img_jax_hwc = np.transpose(np.array(decoded_jax[0]), (1, 2, 0))
    img_jax_uint8 = np.clip((img_jax_hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    jax_image_path = os.path.join(self.golden_dir, "jax_edited_image_fp32.png")
    Image.fromarray(img_jax_uint8).save(jax_image_path)
    print(f" -> Saved JAX Float32 image to: {jax_image_path}")

    # PyTorch Decode
    latents_spatial_pt = rearrange(pt_latents.numpy(), "b (h w) c -> b c h w", h=32, w=32)
    latents_denorm_pt = latents_spatial_pt * np.array(bn_std) + np.array(bn_mean)
    unpatchified_pt = unpatchify_latents(jnp.array(latents_denorm_pt))
    with torch.no_grad():
      decoded_pt = pt_vae.decode(torch.tensor(np.array(unpatchified_pt), dtype=torch.float32)).sample.numpy()
    img_pt_hwc = np.transpose(decoded_pt[0], (1, 2, 0))
    img_pt_uint8 = np.clip((img_pt_hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    pt_image_path = os.path.join(self.golden_dir, "pytorch_edited_image_fp32.png")
    Image.fromarray(img_pt_uint8).save(pt_image_path)
    print(f" -> Saved PyTorch Float32 image to: {pt_image_path}")

    # 6. Exact Visual & Pixel Parity Assertions
    pixel_diff = np.abs(img_jax_uint8.astype(np.float32) - img_pt_uint8.astype(np.float32))
    max_pixel_diff = np.max(pixel_diff)
    mean_pixel_diff = np.mean(pixel_diff)
    print(f" -> Float32 Pixel Difference (JAX vs PyTorch): Max = {max_pixel_diff:.4f} / 255, Mean = {mean_pixel_diff:.4f} / 255")

    self.assertLess(mean_pixel_diff, 0.5, "Mean pixel difference between JAX and PyTorch in float32 must be < 0.5 / 255!")
    self.assertLess(max_pixel_diff, 3.0, "Max pixel difference between JAX and PyTorch in float32 must be < 3.0 / 255!")

    print("=" * 80)
    print("✅ Phase 5: 4-Step Euler Denoising Loop & EXACT Visual Parity PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
