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

"""CLI Entry Point for Generating Images with PRXPixel on TPU / JAX."""

import argparse
import os
import time
import jax
import jax.numpy as jnp
from PIL import Image

from maxdiffusion.checkpointing.prx_pixel_checkpointer import PRXPixelCheckpointer

DEFAULT_SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")


def main():
  parser = argparse.ArgumentParser(description="Generate images with PRXPixel in pure Flax NNX on TPU.")
  parser.add_argument("--prompt", type=str, default="A majestic lion sitting on a rock in the savannah at golden hour sunset, 8k, photorealistic")
  parser.add_argument("--negative_prompt", type=str, default="")
  parser.add_argument("--model_path", type=str, default=DEFAULT_SNAPSHOT)
  parser.add_argument("--height", type=int, default=1024)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--num_inference_steps", type=int, default=28)
  parser.add_argument("--guidance_scale", type=float, default=4.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output_file", type=str, default="prxpixel_generated_image.png")
  parser.add_argument("--dtype", type=str, default="bfloat16")
  args = parser.parse_args()

  jax.config.update("jax_default_matmul_precision", "highest")
  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

  print("=" * 80)
  print("🚀 Launching PRXPixel (Photoroom/prxpixel-t2i) Image Generation")
  print(f"   Prompt: '{args.prompt}'")
  print(f"   Resolution: {args.height}x{args.width} | Steps: {args.num_inference_steps} | CFG: {args.guidance_scale}")
  print(f"   Dtype: {args.dtype} | Seed: {args.seed}")
  print("=" * 80)

  t0 = time.perf_counter()
  pipeline = PRXPixelCheckpointer.load_pipeline(args.model_path, dtype=dtype)
  load_time = time.perf_counter() - t0
  print(f"⏱️ Model Loading & Placement Time: {load_time:.2f}s")

  rng = jax.random.PRNGKey(args.seed)

  # ---------------------------------------------------------------------------
  # Pass 1: Compilation & Warmup Run
  # ---------------------------------------------------------------------------
  print("\n🎨 [PASS 1/2] Running Compilation & Warmup Pass...")
  t_pass1_start = time.perf_counter()
  images_warmup = pipeline(
      prompt=args.prompt,
      negative_prompt=args.negative_prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      guidance_scale=args.guidance_scale,
      generator=rng,
  )
  pass1_time = time.perf_counter() - t_pass1_start
  print(f"⏱️ Pass 1 (Compilation + Execution) Time: {pass1_time:.2f}s")

  out_dir = os.path.dirname(args.output_file) or "."
  out_name = os.path.basename(args.output_file)
  warmup_out = os.path.join(out_dir, f"warmup_{out_name}")
  images_warmup[0].save(warmup_out)
  print(f"✅ Warmup image saved to: {warmup_out}")

  # ---------------------------------------------------------------------------
  # Pass 2: Warmed-Up Steady-State Run
  # ---------------------------------------------------------------------------
  print("\n🎨 [PASS 2/2] Running Warmed-Up Steady-State Inference...")
  t_pass2_start = time.perf_counter()
  images_steady = pipeline(
      prompt=args.prompt,
      negative_prompt=args.negative_prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      guidance_scale=args.guidance_scale,
      generator=rng,
  )
  pass2_time = time.perf_counter() - t_pass2_start
  per_step_ms = (pass2_time / args.num_inference_steps) * 1000.0

  images_steady[0].save(args.output_file)
  print(f"✅ Steady-state image saved to: {args.output_file}")

  # ---------------------------------------------------------------------------
  # Performance Summary
  # ---------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("📊 PRXPIXEL LATENCY & PERFORMANCE BENCHMARK:")
  print(f"  • Model Load & Placement Time:  {load_time:.2f}s")
  print(f"  • Pass 1 (Compile + Warmup):     {pass1_time:.2f}s")
  print(f"  • Pass 2 (Steady-State Latency): {pass2_time:.2f}s")
  print(f"  • Per-Step Latency:              {per_step_ms:.1f} ms/step")
  print(f"  • Resolution:                    {args.height}x{args.width}")
  print(f"  • Inference Steps:               {args.num_inference_steps}")
  print(f"  • Dtype:                         {args.dtype}")
  print("=" * 80)


if __name__ == "__main__":
  main()
