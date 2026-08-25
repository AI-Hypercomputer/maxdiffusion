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

"""Runs PRXPixel CFG Guidance Scale Sweep (3 Prompts x 4 CFG Levels) on TPU."""

import os
import sys
import time
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw

from maxdiffusion.checkpointing.prx_pixel_checkpointer import PRXPixelCheckpointer

SNAPSHOT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")
OUTPUT_DIR = "/mnt/data/maxdiffusion/prxpixel_testoutputs"

PROMPTS = [
    ("sports_car", "A sleek modern sports car on a winding mountain road at dusk, cinematic lighting"),
    ("punk_musician", "a portrait of a punk musician with green mohawk and nose ring, leather jacket, harsh flash photography"),
    ("male_model", "an adult male model wearing a short-sleeve t-shirt and jeans under studio lighting with white background"),
]

CFG_SCALES = [1.0, 2.0, 3.0, 4.0]
STEPS = 28
HEIGHT = 512
WIDTH = 512
SEED = 42


def create_comparison_strip(images, cfg_scales, title, out_path):
  """Creates a 4-column side-by-side comparison image with header banners."""
  w, h = images[0].size
  header_h = 50
  padding = 10
  total_w = w * len(images) + padding * (len(images) - 1)
  total_h = h + header_h

  combined = Image.new("RGB", (total_w, total_h), (240, 240, 240))
  draw = ImageDraw.Draw(combined)

  for i, (img, cfg) in enumerate(zip(images, cfg_scales)):
    x_offset = i * (w + padding)
    combined.paste(img, (x_offset, header_h))
    text = f"CFG = {cfg:.1f}"
    draw.text((x_offset + w // 2 - 35, 18), text, fill=(0, 0, 0))

  combined.save(out_path)
  print(f"  🖼️ Saved comparison strip: {out_path}", flush=True)


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  jax.config.update("jax_default_matmul_precision", "highest")

  print("=" * 90, flush=True)
  print("🚀 Starting PRXPixel CFG Guidance Scale Sweep on TPU", flush=True)
  print(f"   Prompts: {len(PROMPTS)} | CFG Scales: {CFG_SCALES} | Steps: {STEPS} | Resolution: {HEIGHT}x{WIDTH}", flush=True)
  print(f"   Output Directory: {OUTPUT_DIR}", flush=True)
  print(f"   Devices: {jax.devices()}", flush=True)
  print("=" * 90, flush=True)

  t0 = time.perf_counter()
  pipeline = PRXPixelCheckpointer.load_pipeline(SNAPSHOT_DIR, dtype=jnp.bfloat16)
  load_time = time.perf_counter() - t0
  print(f"⏱️ Model Load Time: {load_time:.2f}s\n", flush=True)

  # Warmup run to trigger JIT compilation
  print("🔥 JIT Warming up pipeline...", flush=True)
  t_w0 = time.perf_counter()
  _ = pipeline(
      prompt="warmup test prompt",
      height=HEIGHT,
      width=WIDTH,
      num_inference_steps=STEPS,
      guidance_scale=4.0,
      generator=jax.random.PRNGKey(0),
  )
  print(f"⏱️ Warmup JIT Time: {time.perf_counter() - t_w0:.2f}s\n", flush=True)

  for prompt_idx, (tag, prompt_text) in enumerate(PROMPTS, 1):
    print("=" * 90, flush=True)
    print(f"📸 [{prompt_idx}/{len(PROMPTS)}] Prompt: \"{prompt_text}\"", flush=True)
    print("=" * 90, flush=True)

    prompt_images = []
    # Deterministic initial noise per prompt
    initial_noise = 2.0 * jax.random.normal(
        jax.random.PRNGKey(SEED),
        shape=(1, 3, HEIGHT, WIDTH),
        dtype=jnp.bfloat16,
    )

    for cfg in CFG_SCALES:
      t_start = time.perf_counter()
      img = pipeline(
          prompt=prompt_text,
          negative_prompt="",
          height=HEIGHT,
          width=WIDTH,
          num_inference_steps=STEPS,
          guidance_scale=cfg,
          latents=initial_noise,
          output_type="pil",
      )[0]
      gen_time = time.perf_counter() - t_start

      img_filename = f"{tag}_cfg{int(cfg)}.png"
      img_path = os.path.join(OUTPUT_DIR, img_filename)
      img.save(img_path)
      prompt_images.append(img)
      print(f"  ✅ CFG {cfg:.1f} generated in {gen_time:.2f}s -> {img_filename}", flush=True)

    # Generate 4-column comparison strip
    strip_filename = f"{tag}_cfg_comparison_1_to_4.png"
    strip_path = os.path.join(OUTPUT_DIR, strip_filename)
    create_comparison_strip(prompt_images, CFG_SCALES, prompt_text, strip_path)

  print("\n" + "=" * 90, flush=True)
  print("🎉 All 12 CFG sweep images & comparison strips successfully generated!", flush=True)
  print(f"📁 Files saved in: {OUTPUT_DIR}", flush=True)
  print("=" * 90, flush=True)


if __name__ == "__main__":
  main()
