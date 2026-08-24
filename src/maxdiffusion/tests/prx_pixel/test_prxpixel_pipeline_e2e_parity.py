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

"""End-to-End Trajectory Parity Test between PyTorch Diffusers and MaxDiffusion JAX."""

import os
import unittest
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import jax
import jax.numpy as jnp

from diffusers import PRXPixelPipeline
from maxdiffusion.checkpointing.prx_pixel_checkpointer import PRXPixelCheckpointer

SNAPSHOT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")
OUTPUT_DIR = "/tmp/prxpixel_parity_output"


class TestPRXPixelPipelineE2EParity(unittest.TestCase):
  """Validates E2E generation parity between PyTorch Diffusers and MaxDiffusion JAX."""

  @classmethod
  def setUpClass(cls):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "highest")

  def test_e2e_trajectory_parity_4steps(self):
    """Runs 4-step generation on both PyTorch Diffusers CPU and MaxDiffusion JAX with identical initial noise."""
    if not os.path.exists(SNAPSHOT_DIR):
      self.skipTest(f"Snapshot directory missing at {SNAPSHOT_DIR}")

    prompt = "A sleek modern sports car on a winding mountain road at dusk, cinematic lighting"
    neg_prompt = ""
    H, W = 512, 512
    steps = 4
    cfg = 4.5

    print("\n" + "=" * 80)
    print("🚀 [STEP 1/2] Running PyTorch Diffusers PRXPixelPipeline...")
    print("=" * 80)

    pipe_pt = PRXPixelPipeline.from_pretrained(
        SNAPSHOT_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe_pt.text_encoder = pipe_pt.text_encoder.to(torch.bfloat16).to("cpu")
    pipe_pt.transformer = pipe_pt.transformer.to(torch.bfloat16).to("cpu")

    # Generate initial deterministic noise tensor on CPU with 2.0x noise scale
    torch.manual_seed(42)
    initial_noise_pt = 2.0 * torch.randn((1, 3, H, W), dtype=torch.bfloat16, device="cpu")

    pt_latents_history = [initial_noise_pt.clone()]
    def pt_callback(pipe, step, timestep, callback_kwargs):
      pt_latents_history.append(callback_kwargs["latents"].clone())
      return callback_kwargs

    pipe_pt._callback_tensor_inputs = ["latents"]
    with torch.no_grad():
      out_pt_np = pipe_pt(
          prompt=prompt,
          negative_prompt=neg_prompt,
          height=H,
          width=W,
          num_inference_steps=steps,
          guidance_scale=cfg,
          latents=initial_noise_pt,
          use_resolution_binning=False,
          output_type="np",
          callback_on_step_end=pt_callback,
          callback_on_step_end_tensor_inputs=["latents"],
      ).images[0]

    out_pt = Image.fromarray((out_pt_np * 255.0).round().astype(np.uint8))
    pt_img_path = os.path.join(OUTPUT_DIR, "diffusers_pt_output.png")
    out_pt.save(pt_img_path)
    print(f"✅ PyTorch image saved to: {pt_img_path}")

    print("\n" + "=" * 80)
    print("🚀 [STEP 2/2] Running MaxDiffusion Production Pipeline...")
    print("=" * 80)

    pipe_jax = PRXPixelCheckpointer.load_pipeline(SNAPSHOT_DIR, dtype=jnp.bfloat16, device="cpu")

    initial_noise_jax = jnp.asarray(initial_noise_pt.float().numpy(), dtype=jnp.bfloat16)

    jax_latents_history = []
    def jax_callback(pipe, step, timestep, callback_kwargs):
      jax_latents_history.append(np.asarray(callback_kwargs["latents"].astype(jnp.float32)))

    out_jax_np = pipe_jax(
        prompt=prompt,
        negative_prompt=neg_prompt,
        height=H,
        width=W,
        num_inference_steps=steps,
        guidance_scale=cfg,
        latents=initial_noise_jax,
        output_type="np",
        callback_on_step_end=jax_callback,
    )[0]

    out_jax = Image.fromarray((out_jax_np * 255.0).round().astype(np.uint8))
    jax_img_path = os.path.join(OUTPUT_DIR, "maxdiffusion_jax_output.png")
    out_jax.save(jax_img_path)
    print(f"✅ MaxDiffusion image saved to: {jax_img_path}")

    # Compute step-by-step relative L2 parity
    print("\n" + "=" * 80)
    print("📈 STEP-BY-STEP TRAJECTORY RELATIVE L2 PARITY:")
    print(f"{'Step':<6} | {'Timestep (t)':<14} | {'Rel L2':<12} | {'Max Abs':<12} | {'Cos Sim':<10}")
    print("-" * 80)

    for i in range(steps):
      pt_lat = pt_latents_history[i + 1].float().cpu().numpy()
      jax_lat = jax_latents_history[i]
      rel_l2 = np.linalg.norm(jax_lat - pt_lat) / np.linalg.norm(pt_lat)
      max_abs = np.max(np.abs(jax_lat - pt_lat))
      cos_sim = np.dot(jax_lat.flatten(), pt_lat.flatten()) / (np.linalg.norm(jax_lat) * np.linalg.norm(pt_lat))
      t_val = pipe_pt.scheduler.timesteps[i].item()
      print(f"{i:<6} | {t_val:<14.2f} | {rel_l2:<12.4e} | {max_abs:<12.4e} | {cos_sim:<10.6f}")
      self.assertLess(rel_l2, 0.05, f"Step {i} Relative L2 error {rel_l2:.4e} exceeds 0.05 threshold!")

    print("=" * 80)

    # Compute continuous SSIM and PSNR (data_range=1.0)
    score_ssim = ssim(out_pt_np, out_jax_np, channel_axis=2, data_range=1.0)
    score_psnr = psnr(out_pt_np, out_jax_np, data_range=1.0)
    score_ssim_uint8 = ssim(np.array(out_pt), np.array(out_jax), channel_axis=2)

    print("\n" + "=" * 80)
    print("📊 QUANTITATIVE E2E PARITY RESULTS:")
    print(f"  • SSIM Score (float [0, 1]): {score_ssim:.4f}")
    print(f"  • SSIM Score (uint8 RGB):    {score_ssim_uint8:.4f}")
    print(f"  • PSNR:                      {score_psnr:.2f} dB")
    print("=" * 80)

    self.assertGreater(score_ssim, 0.95, f"SSIM score {score_ssim:.4f} is below 0.95 threshold!")
    self.assertGreater(score_psnr, 35.0, f"PSNR {score_psnr:.2f} dB is below 35.0 dB threshold!")
    print("\n🎉 [PHASE 11 COMPLETE] MaxDiffusion PRXPixel matches PyTorch Diffusers with high fidelity!")


if __name__ == "__main__":
  unittest.main()
