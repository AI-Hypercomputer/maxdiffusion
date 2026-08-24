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

"""Step-0 Tensor-Level Debugger: Compares every intermediate tensor in Step 0."""

import os
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import jax
import jax.numpy as jnp
from PIL import Image

from diffusers import PRXPipeline
from tools.prxpixel.dump_prxpixel_reference import build_pytorch_transformer_model
from tools.prxpixel.compare_prxpixel_tensors import compute_metrics
from maxdiffusion.checkpointing.prx_pixel_checkpointer import PRXPixelCheckpointer

SNAPSHOT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")


def main():
  prompt = "A sleek modern sports car on a winding mountain road at dusk, cinematic lighting"
  H, W = 512, 512
  steps = 4
  guidance_scale = 4.5

  print("=" * 105)
  print("🔍 STEP-0 TENSOR-LEVEL PARITY INVESTIGATION")
  print("=" * 105)

  # ---------------------------------------------------------------------------
  # 1. PyTorch Step 0 Execution
  # ---------------------------------------------------------------------------
  pipe_pt = PRXPipeline.from_pretrained(SNAPSHOT_DIR, transformer=None, torch_dtype=torch.bfloat16)
  pipe_pt.transformer = build_pytorch_transformer_model(SNAPSHOT_DIR, dtype=torch.bfloat16)
  pipe_pt.text_encoder = pipe_pt.text_encoder.to(torch.bfloat16).to("cpu")
  pipe_pt.transformer = pipe_pt.transformer.to(torch.bfloat16).to("cpu")

  torch.manual_seed(42)
  latents_pt = torch.randn((1, 3, H, W), dtype=torch.bfloat16, device="cpu")

  # Encode prompt in PyTorch
  with torch.no_grad():
    text_emb, text_mask, uncond_emb, uncond_mask = pipe_pt.encode_prompt(
        prompt, device=torch.device("cpu"), do_classifier_free_guidance=True, negative_prompt=""
    )
    ca_embed_pt = torch.cat([uncond_emb, text_emb], dim=0)
    ca_mask_pt = torch.cat([uncond_mask, text_mask], dim=0)

    pipe_pt.scheduler.set_timesteps(steps, device=torch.device("cpu"))
    timesteps_pt = pipe_pt.scheduler.timesteps
    t_0 = timesteps_pt[0]
    t_cont_pt = (t_0.float() / pipe_pt.scheduler.config.num_train_timesteps).view(1).repeat(2)

    latents_in_pt = torch.cat([latents_pt, latents_pt], dim=0)

    noise_pred_pt = pipe_pt.transformer(
        hidden_states=latents_in_pt,
        timestep=t_cont_pt,
        encoder_hidden_states=ca_embed_pt,
        attention_mask=ca_mask_pt,
        return_dict=False,
    )[0]

    noise_uncond_pt, noise_text_pt = noise_pred_pt.chunk(2, dim=0)
    pred_cfg_pt = noise_uncond_pt + guidance_scale * (noise_text_pt - noise_uncond_pt)

    latents_next_pt = pipe_pt.scheduler.step(pred_cfg_pt, t_0, latents_pt).prev_sample

  # ---------------------------------------------------------------------------
  # 2. JAX Step 0 Execution
  # ---------------------------------------------------------------------------
  pipe_jax = PRXPixelCheckpointer.load_pipeline(SNAPSHOT_DIR, dtype=jnp.bfloat16, device="cpu")

  prompt_embeds_jax, attention_mask_jax = pipe_jax.encode_prompt(
      prompt, negative_prompt="", do_classifier_free_guidance=True
  )

  sigmas_np = np.linspace(1.0, 1.0 / steps, steps)
  shift = 3.0
  sigmas_np = shift * sigmas_np / (1.0 + (shift - 1.0) * sigmas_np)
  sigmas_jax = jnp.asarray(np.concatenate([sigmas_np, [0.0]]).astype(np.float32))
  timesteps_jax = sigmas_jax[:-1] * 1000.0

  t_0_jax = timesteps_jax[0]
  t_cont_jax = (t_0_jax.astype(jnp.float32) / 1000.0).reshape((1,))

  latents_jax = jnp.asarray(latents_pt.float().numpy(), dtype=jnp.float32)
  latents_in_jax = jnp.concatenate([latents_jax, latents_jax], axis=0).astype(jnp.bfloat16)
  t_cont_in_jax = jnp.broadcast_to(t_cont_jax, (2,))

  pred_jax = pipe_jax.transformer(
      hidden_states=latents_in_jax,
      timestep=t_cont_in_jax,
      encoder_hidden_states=prompt_embeds_jax,
      attention_mask=attention_mask_jax,
  )

  pred_uncond_jax, pred_cond_jax = jnp.split(pred_jax, 2, axis=0)
  pred_cfg_jax = pred_uncond_jax + guidance_scale * (pred_cond_jax - pred_uncond_jax)

  dt_jax = sigmas_jax[1] - sigmas_jax[0]
  latents_next_jax = latents_jax + dt_jax * pred_cfg_jax.astype(jnp.float32)

  # ---------------------------------------------------------------------------
  # 3. Compare Every Tensor
  # ---------------------------------------------------------------------------
  tensors_to_compare = [
      ("prompt_embeds (ca_embed)", prompt_embeds_jax.astype(jnp.float32), ca_embed_pt.float()),
      ("attention_mask (ca_mask)", attention_mask_jax, ca_mask_pt),
      ("latents_in", latents_in_jax.astype(jnp.float32), latents_in_pt.float()),
      ("noise_uncond (raw)", pred_uncond_jax.astype(jnp.float32), noise_uncond_pt.float()),
      ("noise_text (raw)", pred_cond_jax.astype(jnp.float32), noise_text_pt.float()),
      ("pred_cfg", pred_cfg_jax.astype(jnp.float32), pred_cfg_pt.float()),
      ("latents_step_1 (x_1)", latents_next_jax, latents_next_pt.float()),
  ]

  header = f"{'Tensor Name':<32} | {'Shape':<16} | {'Max Abs':<11} | {'Mean Abs':<11} | {'Rel L2':<11} | {'Cos Sim':<9}"
  print(header)
  print("=" * 105)

  for name, jax_t, pt_t in tensors_to_compare:
    jax_arr = np.asarray(jax_t)
    pt_arr = pt_t.numpy() if isinstance(pt_t, torch.Tensor) else np.asarray(pt_t)
    m = compute_metrics(jax_arr, pt_arr)
    print(f"{name:<32} | {str(jax_arr.shape):<16} | {m['max_abs']:<11.4e} | {m['mean_abs']:<11.4e} | {m['rel_l2']:<11.4e} | {m['cos_sim']:<9.6f}")


if __name__ == "__main__":
  main()
