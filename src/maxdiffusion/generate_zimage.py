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

"""Generate an image with Z-Image or Z-Image-Turbo."""

import time
from typing import Sequence

from absl import app
import jax
from jax.sharding import Mesh

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.checkpointing.z_image_checkpointer import ZImageCheckpointer
from maxdiffusion.max_utils import create_device_mesh


def call_pipeline(config, pipeline, num_inference_steps=None, return_timings=False):
  if num_inference_steps is None:
    num_inference_steps = config.num_inference_steps
  # Using global_batch_size_to_train_on so not to create more config variables:
  # per_device_batch_size images are generated on every device.
  return pipeline(
      [config.prompt] * config.global_batch_size_to_train_on,
      height=config.height,
      width=config.width,
      num_inference_steps=num_inference_steps,
      guidance_scale=config.guidance_scale,
      seed=config.seed,
      max_sequence_length=config.max_sequence_length,
      vae_decode_chunk=config.vae_decode_chunk,
      return_timings=return_timings,
  )


def run(config, commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  load_start = time.perf_counter()
  mesh = Mesh(create_device_mesh(config), config.mesh_axes)
  pipeline = ZImageCheckpointer(config, mesh).load_pipeline()
  load_time = time.perf_counter() - load_start
  max_logging.log(f"load_time: {load_time:.1f}s")

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width},"
      f" images: {config.global_batch_size_to_train_on}"
  )

  # Nothing in the pipeline reads the profiler flags -- the profiler is started
  # explicitly around run 3 below -- so the config is never mutated here.
  profiling_requested = config.get_keys().get("enable_profiler", False) or config.get_keys().get(
      "enable_ml_diagnostics", False
  )

  # ---------------------------------------------------------
  # Run 1: warmup compilation, nothing profiled.
  # ---------------------------------------------------------
  s0 = time.perf_counter()
  # Warm up at the real step count. A shorter warmup would leave the
  # step-count-dependent work (the sigma schedule, and anything XLA has not
  # already cached at these shapes) to compile inside the timed run.
  max_logging.log(f"🚀 Starting warmup compilation pass ({config.num_inference_steps} denoising steps)...")
  _ = call_pipeline(config, pipeline)
  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  # ---------------------------------------------------------
  # Run 2: the real generation, profiling still disabled.
  # ---------------------------------------------------------
  s0 = time.perf_counter()
  max_logging.log("🚀 Starting full-length generation pass...")
  images, trace = call_pipeline(config, pipeline, return_timings=True)
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  generation_time_per_image = generation_time / len(images)
  max_logging.log(f"generation time per image: {generation_time_per_image}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    writer.add_scalar("inference/generation_time_per_image", generation_time_per_image, global_step=0)

  image_paths = max_utils.save_images(config, images)

  summary = [
      f"\n{'=' * 50}",
      "  TIMING SUMMARY",
      f"{'=' * 50}",
      f"  Load (checkpoint):   {load_time:>7.1f}s",
      f"  Compile:             {compile_time:>7.1f}s",
      f"  Inference:           {generation_time:>7.1f}s",
      f"  Per image ({len(images):>2d}):      {generation_time_per_image:>7.1f}s",
  ]
  if trace:
    summary.extend([
        f"  {'─' * 40}",
        f"    Text Encoding:     {trace.get('text_encode', 0.0):>7.1f}s",
        f"    Denoise Total:     {trace.get('denoise', 0.0):>7.1f}s",
        f"    VAE Decode:        {trace.get('vae_decode', 0.0):>7.1f}s",
        f"    Host Formatting:   {trace.get('host_post', 0.0):>7.1f}s",
    ])
  summary.append(f"{'=' * 50}")
  max_logging.log("\n".join(summary))

  # ---------------------------------------------------------
  # Run 3: profiled pass, only if profiling was originally enabled.
  # ---------------------------------------------------------
  if profiling_requested:
    profiling_steps = config.get_keys().get("profiler_steps", 5)
    max_logging.log(f"🚀 Warmup for profiling pass ({profiling_steps} denoising steps)...")
    _ = call_pipeline(config, pipeline, num_inference_steps=profiling_steps)

    max_logging.log(f"🚀 Starting profiling run ({profiling_steps} denoising steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()
    s0 = time.perf_counter()
    _ = call_pipeline(config, pipeline, num_inference_steps=profiling_steps)
    generation_time_with_profiler = time.perf_counter() - s0
    profiler.stop()

    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

  return image_paths


def main(argv: Sequence[str]) -> None:
  commit_hash = max_utils.get_git_commit_hash()
  pyconfig.initialize(argv)
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)
