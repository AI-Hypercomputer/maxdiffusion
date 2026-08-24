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

from diffusers import PRXPipeline
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
    print("🚀 [STEP 1/2] Running PyTorch Diffusers Reference Pipeline...")
    print("=" * 80)

    from tools.prxpixel.dump_prxpixel_reference import build_pytorch_transformer_model
    pipe_pt = PRXPipeline.from_pretrained(
        SNAPSHOT_DIR,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )
    pipe_pt.transformer = build_pytorch_transformer_model(SNAPSHOT_DIR, dtype=torch.bfloat16)
    pipe_pt.text_encoder = pipe_pt.text_encoder.to(torch.bfloat16).to("cpu")
    pipe_pt.transformer = pipe_pt.transformer.to(torch.bfloat16).to("cpu")

    # Generate initial deterministic noise tensor on CPU
    torch.manual_seed(42)
    initial_noise_pt = torch.randn((1, 3, H, W), dtype=torch.bfloat16, device="cpu")

    with torch.no_grad():
      out_pt_raw = pipe_pt(
          prompt=prompt,
          negative_prompt=neg_prompt,
          height=H,
          width=W,
          num_inference_steps=steps,
          guidance_scale=cfg,
          latents=initial_noise_pt,
          use_resolution_binning=False,
          output_type="pt",
      ).images[0]
      pt_np = out_pt_raw.float().cpu().numpy()
      pt_np = (np.clip(pt_np, -1.0, 1.0) + 1.0) * 127.5
      pt_np = np.transpose(pt_np.astype(np.uint8), (1, 2, 0))
      out_pt = Image.fromarray(pt_np)

    pt_img_path = os.path.join(OUTPUT_DIR, "diffusers_pt_output.png")
    out_pt.save(pt_img_path)
    print(f"✅ PyTorch image saved to: {pt_img_path}")

    print("\n" + "=" * 80)
    print("🚀 [STEP 2/2] Running MaxDiffusion JAX Pipeline...")
    print("=" * 80)

    pipe_jax = PRXPixelCheckpointer.load_pipeline(SNAPSHOT_DIR, dtype=jnp.bfloat16, device="cpu")

    initial_noise_jax = jnp.asarray(initial_noise_pt.float().numpy(), dtype=jnp.float32)

    out_jax = pipe_jax(
        prompt=prompt,
        negative_prompt=neg_prompt,
        height=H,
        width=W,
        num_inference_steps=steps,
        guidance_scale=cfg,
        latents=initial_noise_jax,
        output_type="pil",
    )[0]

    jax_img_path = os.path.join(OUTPUT_DIR, "maxdiffusion_jax_output.png")
    out_jax.save(jax_img_path)
    print(f"✅ MaxDiffusion image saved to: {jax_img_path}")

    # Compute SSIM and PSNR
    arr_pt = np.array(out_pt)
    arr_jax = np.array(out_jax)

    score_ssim = ssim(arr_pt, arr_jax, channel_axis=2)
    score_psnr = psnr(arr_pt, arr_jax)

    print("\n" + "=" * 80)
    print("📊 QUANTITATIVE E2E PARITY RESULTS:")
    print(f"  • SSIM Score:  {score_ssim:.4f}")
    print(f"  • PSNR:        {score_psnr:.2f} dB")
    print("=" * 80)

    self.assertGreater(score_ssim, 0.80, f"SSIM score {score_ssim:.4f} is below 0.80 threshold!")
    print("\n🎉 [PHASE 11 COMPLETE] MaxDiffusion PRXPixel matches PyTorch Diffusers with high fidelity!")


if __name__ == "__main__":
  unittest.main()
