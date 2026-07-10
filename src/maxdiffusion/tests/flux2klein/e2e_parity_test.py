# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import gc
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Set HF_HOME cache path early
if not os.environ.get("HF_HOME"):
  if os.path.exists("/mnt/data/hf_cache"):
    os.environ["HF_HOME"] = "/mnt/data/hf_cache"


def compute_psnr(img1, img2):
  img1_np = np.array(img1).astype(np.float64)
  img2_np = np.array(img2).astype(np.float64)
  mse = np.mean((img1_np - img2_np) ** 2)
  if mse == 0:
    return float("inf")
  return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
  img1_gray = np.array(img1.convert("L"))
  img2_gray = np.array(img2.convert("L"))
  return ssim(img1_gray, img2_gray)


def run_pytorch_pipeline(model_id, prompt, batch_size, width, height, num_inference_steps, seed, latents_pt_packed, prefix):
  # Locate cached model files
  cache_dir = f"/mnt/data/hf_cache/hub/models--{model_id.replace('/', '--')}/snapshots"
  if not os.path.exists(cache_dir):
    raise FileNotFoundError(f"Hugging Face cache directory not found: {cache_dir}")
  snapshots = os.listdir(cache_dir)
  snapshot_dir = os.path.join(cache_dir, snapshots[0])
  print(f"\n[PyTorch] Loading '{model_id}' weights from: {snapshot_dir}")

  from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

  # Run in FP32
  print("[PyTorch] Running Leg 1: FP32...")
  pipe_fp32 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.float32, local_files_only=True)
  pipe_fp32.to("cpu")
  with torch.no_grad():
    pt_images_fp32 = pipe_fp32(
        prompt=[prompt] * batch_size,
        width=width,
        height=height,
        latents=latents_pt_packed,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images

  fp32_paths = []
  for idx, img in enumerate(pt_images_fp32):
    path = f"/tmp/{prefix}_pt_fp32_b{idx}.png"
    img.save(path)
    fp32_paths.append(path)
    print(f" -> Saved: {path}")

  del pipe_fp32
  gc.collect()

  # Run in BF16
  print("[PyTorch] Running Leg 2: BF16...")
  pipe_bf16 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.bfloat16, local_files_only=True)
  pipe_bf16.to("cpu")
  with torch.no_grad():
    pt_images_bf16 = pipe_bf16(
        prompt=[prompt] * batch_size,
        width=width,
        height=height,
        latents=latents_pt_packed.to(torch.bfloat16),
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images

  bf16_paths = []
  for idx, img in enumerate(pt_images_bf16):
    path = f"/tmp/{prefix}_pt_bf16_b{idx}.png"
    img.save(path)
    bf16_paths.append(path)
    print(f" -> Saved: {path}")

  del pipe_bf16
  gc.collect()

  return fp32_paths, bf16_paths


def main():
  prompt = "An animated person dancing in the fields"
  width = 512
  height = 512
  num_inference_steps = 4
  batch_size = 4
  seed = 123

  print("=" * 80)
  # 1. Generate identical starting noise on CPU using NumPy
  print(f"Generating shared starting noise on CPU (seed={seed}, batch_size={batch_size})...")
  np.random.seed(seed)
  latents_numpy = np.random.randn(batch_size, 32, height // 8, width // 8).astype(np.float32)

  # Save latents to file for JAX pipeline to read
  latents_file_path = "/tmp/shared_noise_b4.npy"
  np.save(latents_file_path, latents_numpy)
  print(f" -> Saved shared noise to JAX-compatible file: {latents_file_path}")

  # Prepare packed latents for PyTorch
  latents_unpacked_pt = torch.from_numpy(latents_numpy)
  latents_pt_packed = latents_unpacked_pt.view(batch_size, 32, height // 16, 2, width // 16, 2)
  latents_pt_packed = latents_pt_packed.permute(0, 1, 3, 5, 2, 4)
  latents_pt_packed = latents_pt_packed.reshape(batch_size, 128, height // 16, width // 16)

  # =========================================================================
  # PART I: FLUX.2-KLEIN-4B PARITY
  # =========================================================================
  print("\n" + "#" * 80)
  print("🎬 STARTING PARITY EVALUATION FOR FLUX.2-KLEIN-4B")
  print("#" * 80)

  # 1. Run PyTorch 4B (FP32 & BF16)
  pt_4b_fp32_paths, pt_4b_bf16_paths = run_pytorch_pipeline(
      model_id="black-forest-labs/FLUX.2-klein-4B",
      prompt=prompt,
      batch_size=batch_size,
      width=width,
      height=height,
      num_inference_steps=num_inference_steps,
      seed=seed,
      latents_pt_packed=latents_pt_packed,
      prefix="4b",
  )

  # 2. Run JAX 4B (via generate_flux2klein.py)
  print("\n[JAX 4B] Executing pipeline script generate_flux2klein.py...")
  cmd_jax_4b = [
      "python3",
      "src/maxdiffusion/generate_flux2klein.py",
      "src/maxdiffusion/configs/base_flux2klein.yml",
      f"prompt={prompt}",
      "output_dir=/tmp/",
      f"seed={seed}",
      f"height={height}",
      f"width={width}",
      f"num_inference_steps={num_inference_steps}",
      f"batch_size={batch_size}",
      f"latents_path={latents_file_path}",
      "weights_dtype=bfloat16",
      "activations_dtype=bfloat16",
      "precision=DEFAULT",
      "output_name=jax_4b_dancing.png",
  ]
  print(f"Executing: {' '.join(cmd_jax_4b)}")
  subprocess.run(cmd_jax_4b, check=True)

  jax_4b_paths = [f"/tmp/jax_4b_dancing_b{b}.png" for b in range(batch_size)]
  print(f"[JAX 4B] Execution complete! Verifying outputs at: {jax_4b_paths}")

  # =========================================================================
  # PART II: FLUX.2-KLEIN-9B PARITY
  # =========================================================================
  print("\n" + "#" * 80)
  print("🎬 STARTING PARITY EVALUATION FOR FLUX.2-KLEIN-9B")
  print("#" * 80)

  # 1. Run PyTorch 9B (FP32 & BF16)
  pt_9b_fp32_paths, pt_9b_bf16_paths = run_pytorch_pipeline(
      model_id="black-forest-labs/FLUX.2-klein-9B",
      prompt=prompt,
      batch_size=batch_size,
      width=width,
      height=height,
      num_inference_steps=num_inference_steps,
      seed=seed,
      latents_pt_packed=latents_pt_packed,
      prefix="9b",
  )

  # 2. Run JAX 9B (via generate_flux2klein.py)
  print("\n[JAX 9B] Executing pipeline script generate_flux2klein.py...")
  cmd_jax_9b = [
      "python3",
      "src/maxdiffusion/generate_flux2klein.py",
      "src/maxdiffusion/configs/base_flux2klein_9B.yml",
      f"prompt={prompt}",
      "output_dir=/tmp/",
      f"seed={seed}",
      f"height={height}",
      f"width={width}",
      f"num_inference_steps={num_inference_steps}",
      f"batch_size={batch_size}",
      f"latents_path={latents_file_path}",
      "weights_dtype=bfloat16",
      "activations_dtype=bfloat16",
      "precision=DEFAULT",
      "output_name=jax_9b_dancing.png",
  ]
  print(f"Executing: {' '.join(cmd_jax_9b)}")
  subprocess.run(cmd_jax_9b, check=True)

  jax_9b_paths = [f"/tmp/jax_9b_dancing_b{b}.png" for b in range(batch_size)]
  print(f"[JAX 9B] Execution complete! Verifying outputs at: {jax_9b_paths}")

  # =========================================================================
  # PART III: METRICS EVALUATION & COMPARISON REPORT
  # =========================================================================
  print("\n" + "=" * 80)
  print("📊 BATCHED VISUAL ALIGNMENT COMPARISON REPORT")
  print("=" * 80)

  report = []
  report.append("# 📈 Flux.2-klein Batched (Batch-4) E2E Parity Report")
  report.append(f"Prompt: '{prompt}'\n")

  report.append("## 4B Model Parity (JAX TPU vs PyTorch CPU)")
  report.append("| Batch Index | JAX vs PyTorch FP32 SSIM | JAX vs PyTorch FP32 PSNR | JAX vs PyTorch BF16 SSIM |")
  report.append("| :--- | :--- | :--- | :--- |")
  for b in range(batch_size):
    img_pt_fp32 = Image.open(pt_4b_fp32_paths[b])
    img_pt_bf16 = Image.open(pt_4b_bf16_paths[b])
    img_jax = Image.open(jax_4b_paths[b])

    ssim_fp32 = compute_ssim(img_jax, img_pt_fp32)
    psnr_fp32 = compute_psnr(img_jax, img_pt_fp32)
    ssim_bf16 = compute_ssim(img_jax, img_pt_bf16)

    report.append(f"| Batch Element {b} | {ssim_fp32:.6f} | {psnr_fp32:.2f} dB | {ssim_bf16:.6f} |")
    print(f"  [4B Batch {b}] JAX vs PyTorch FP32 SSIM: {ssim_fp32:.6f} | JAX vs PyTorch BF16 SSIM: {ssim_bf16:.6f}")

  report.append("\n## 9B Model Parity (JAX TPU vs PyTorch CPU)")
  report.append("| Batch Index | JAX vs PyTorch FP32 SSIM | JAX vs PyTorch FP32 PSNR | JAX vs PyTorch BF16 SSIM |")
  report.append("| :--- | :--- | :--- | :--- |")
  for b in range(batch_size):
    img_pt_fp32 = Image.open(pt_9b_fp32_paths[b])
    img_pt_bf16 = Image.open(pt_9b_bf16_paths[b])
    img_jax = Image.open(jax_9b_paths[b])

    ssim_fp32 = compute_ssim(img_jax, img_pt_fp32)
    psnr_fp32 = compute_psnr(img_jax, img_pt_fp32)
    ssim_bf16 = compute_ssim(img_jax, img_pt_bf16)

    report.append(f"| Batch Element {b} | {ssim_fp32:.6f} | {psnr_fp32:.2f} dB | {ssim_bf16:.6f} |")
    print(f"  [9B Batch {b}] JAX vs PyTorch FP32 SSIM: {ssim_fp32:.6f} | JAX vs PyTorch BF16 SSIM: {ssim_bf16:.6f}")

  report_path = "src/maxdiffusion/tests/flux2klein/flux_batched_hummingbird_parity_report.md"
  with open(report_path, "w") as rf:
    rf.write("\n".join(report))

  print("\n" + "=" * 80)
  print(f"SUCCESS! Batched parity report written to: {report_path}")
  print("=" * 80)


if __name__ == "__main__":
  main()
