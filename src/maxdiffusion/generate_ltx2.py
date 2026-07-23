# Copyright 2025 Google LLC
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

from typing import Sequence
import jax
import jax.numpy as jnp
import time
import os
import subprocess
from maxdiffusion.checkpointing.ltx2_checkpointer import LTX2Checkpointer
from maxdiffusion import aot_cache, pyconfig, max_logging, max_utils
from absl import app
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
import flax
from maxdiffusion.utils.export_utils import export_to_video_with_audio
from maxdiffusion.loaders.ltx2_lora_nnx_loader import LTX2NNXLoraLoader


def upload_video_to_gcs(output_dir: str, video_path: str):
  """
  Uploads a local video file to a specified Google Cloud Storage bucket.
  """
  try:
    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    source_file_path = f"./{video_path}"
    destination_blob_name = os.path.join(folder_name, "videos", video_path)

    blob = bucket.blob(destination_blob_name)

    max_logging.log(f"Uploading {source_file_path} to {bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    max_logging.log(f"Upload complete {source_file_path}.")

  except GoogleAPIError as e:
    max_logging.log(f"A storage error occurred during upload: {e}")


def delete_file(file_path: str):
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_git_commit_hash():
  """Tries to get the current Git commit hash."""
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    return commit_hash
  except subprocess.CalledProcessError:
    max_logging.log("Warning: 'git rev-parse HEAD' failed. Not running in a git repo?")
    return None
  except FileNotFoundError:
    max_logging.log("Warning: 'git' command not found.")
    return None


jax.config.update("jax_use_shardy_partitioner", True)


def call_pipeline(config, pipeline, prompt, negative_prompt):
  generator = jax.random.key(config.seed) if hasattr(config, "seed") else jax.random.key(0)
  guidance_scale = config.guidance_scale if hasattr(config, "guidance_scale") else 3.0

  out = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=guidance_scale,
      guidance_rescale=getattr(config, "guidance_rescale", 0.0),
      generator=generator,
      frame_rate=getattr(config, "fps", 24.0),
      decode_timestep=getattr(config, "decode_timestep", 0.0),
      decode_noise_scale=getattr(config, "decode_noise_scale", None),
      max_sequence_length=getattr(config, "max_sequence_length", 1024),
      audio_guidance_scale=getattr(config, "audio_guidance_scale", None),
      audio_guidance_rescale=getattr(config, "audio_guidance_rescale", None),
      stg_scale=getattr(config, "stg_scale", 0.0),
      audio_stg_scale=getattr(config, "audio_stg_scale", None),
      modality_scale=getattr(config, "modality_scale", 1.0),
      audio_modality_scale=getattr(config, "audio_modality_scale", None),
      use_cross_timestep=getattr(config, "use_cross_timestep", None),
      noise_scale=getattr(config, "noise_scale", 1.0),
      dtype=jnp.bfloat16 if getattr(config, "activations_dtype", "bfloat16") == "bfloat16" else jnp.float32,
      output_type=getattr(config, "upsampler_output_type", "pil"),
  )
  return out


def maybe_tune_block_sizes(config):
  """If enable_tile_search, run a fast one-DiT-block tile-size grid search and overwrite
  flash_block_sizes' block_q/block_kv/block_kv_compute with the winner IN PLACE.
  """
  keys = config.get_keys()
  val = keys.get("enable_tile_search", False)
  if str(val).lower() not in ("true", "1", "yes"):
    return
  from maxdiffusion.utils.tile_size_grid_search import grid_search
  from maxdiffusion.utils.ltx2_block_benchmark import LTX2BlockBenchmark

  vmem = config.flash_block_sizes.get("vmem_limit_bytes", None) if config.flash_block_sizes else None
  if vmem is None:
    import os
    import re

    m = re.search(r"--xla_tpu_scoped_vmem_limit_kib=(\d+)", os.environ.get("LIBTPU_INIT_ARGS", ""))
    vmem = int(m.group(1)) * 1024 if m else 32 * 1024 * 1024

  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  bench = LTX2BlockBenchmark.from_config(config, mesh, vmem_limit_bytes=vmem)
  max_logging.log(f"[tile-search] tuning block sizes for {bench.label} (vmem={vmem/1024/1024:.1f}MB) before inference...")
  result = grid_search(
      bench,
      mode=keys.get("tile_search_mode", "smart"),
      iters=keys.get("tile_search_iters", 10),
      out_dir=(keys.get("tile_search_out", "") or None),
      log=max_logging.log,
  )
  if result.best is None:
    max_logging.log("[tile-search] no config succeeded; keeping configured flash_block_sizes")
    return
  fbs = dict(config.flash_block_sizes) if config.flash_block_sizes else {}
  fbs.update({
      "block_q": result.best.bq,
      "block_kv": result.best.bkv,
      "block_kv_compute": result.best.bkv_compute,
      "block_kv_compute_in": result.best.bkv_compute,
      "vmem_limit_bytes": vmem,
  })
  config.get_keys()["flash_block_sizes"] = fbs
  max_logging.log(
      f"[tile-search] using block_q={result.best.bq} block_kv={result.best.bkv} "
      f"(block-bench {result.best.mean_ms:.2f} ms)"
  )


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  if pipeline is None:
    maybe_tune_block_sizes(config)

  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  checkpoint_loader = LTX2Checkpointer(config=config)
  load_time = 0.0
  if pipeline is None:
    t0_load = time.perf_counter()
    # Use the config flag to determine if the upsampler should be loaded
    run_latent_upsampler = getattr(config, "run_latent_upsampler", False)
    pipeline, _, _ = checkpoint_loader.load_checkpoint(load_upsampler=run_latent_upsampler)

    # If LoRA is specified, inject layers and load weights.
    if (
        getattr(config, "enable_lora", False)
        and hasattr(config, "lora_config")
        and config.lora_config
        and config.lora_config.get("lora_model_name_or_path")
    ):
      lora_loader = LTX2NNXLoraLoader()
      lora_config = config.lora_config
      paths = lora_config["lora_model_name_or_path"]
      weights = lora_config.get("weight_name", [None] * len(paths))
      scales = lora_config.get("scale", [1.0] * len(paths))
      ranks = lora_config.get("rank", [64] * len(paths))

      for i in range(len(paths)):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            paths[i],
            transformer_weight_name=weights[i],
            rank=ranks[i],
            scale=scales[i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )
    load_time = time.perf_counter() - t0_load

  pipeline.enable_vae_slicing()
  pipeline.enable_vae_tiling()

  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on to map prompts
  prompt = getattr(config, "prompt", "A cat playing piano")
  prompt = [prompt] * getattr(config, "global_batch_size_to_train_on", 1)

  negative_prompt = getattr(config, "negative_prompt", "")
  negative_prompt = [negative_prompt] * getattr(config, "global_batch_size_to_train_on", 1)

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {getattr(config, 'model_name', 'ltx-video')}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {getattr(config, 'model_type', 'T2V')}")
  if getattr(config, "run_latent_upsampler", False):
    max_logging.log(f"upsampler model path: {config.upsampler_model_path}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  original_enable_profiler = config.get_keys().get("enable_profiler", False)
  original_enable_mld = config.get_keys().get("enable_ml_diagnostics", False)
  original_num_steps = config.get_keys().get("num_inference_steps", 40)

  # Per-shape AOT executable cache
  aot_cache.install(
      getattr(config, "aot_cache_dir", ""),
      meta={
          "model": config.pretrained_model_name_or_path,
          "attention": getattr(config, "attention", ""),
          "flash_block_sizes": str(getattr(config, "flash_block_sizes", "")),
          "mesh_shape": str(pipeline.mesh.shape) if pipeline and hasattr(pipeline, "mesh") and pipeline.mesh else "",
          "weights_dtype": str(getattr(config, "weights_dtype", "bfloat16")),
          "activations_dtype": str(getattr(config, "activations_dtype", "bfloat16")),
          "scan_layers": str(getattr(config, "scan_layers", True)),
          "jax": jax.__version__,
      },
      mesh=pipeline.mesh if pipeline else None,
  )
  aot_cache.wait_for_loads()

  # ---------------------------------------------------------
  # Run 1: Warmup Compilation (Original steps, NO profiling)
  # ---------------------------------------------------------
  config.get_keys()["enable_profiler"] = False
  config.get_keys()["enable_ml_diagnostics"] = False
  warmup_steps = min(2, original_num_steps)
  config.get_keys()["num_inference_steps"] = warmup_steps

  max_logging.log(f"🚀 Starting warmup compilation pass ({warmup_steps} steps)...")
  with aot_cache.warmup_mode():
    _ = call_pipeline(config, pipeline, prompt, negative_prompt)

  aot_cache.save_pending()
  config.get_keys()["num_inference_steps"] = original_num_steps

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  # ---------------------------------------------------------
  # Run 2: Actual Generation (Original steps, NO profiling)
  # ---------------------------------------------------------

  s0 = time.perf_counter()
  max_logging.log("🚀 Starting actual full-length generation pass...")
  out = call_pipeline(config, pipeline, prompt, negative_prompt)
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_devices = jax.device_count()
    num_videos = num_devices * config.per_device_batch_size
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")

  # out should have .frames and .audio
  videos = out.frames if hasattr(out, "frames") else out[0]
  audios = out.audio if hasattr(out, "audio") else None

  saved_video_path = []
  audio_sample_rate = (
      getattr(pipeline.vocoder.config, "output_sampling_rate", 24000)
      if getattr(pipeline, "vocoder", None) is not None
      else 24000
  )
  fps = getattr(config, "fps", 24)

  # Export videos
  for i in range(len(videos)):
    model_name = getattr(config, "model_name", "ltx2") or "ltx2"
    model_name_prefix = model_name.replace(".", "_")
    video_path = f"{filename_prefix}{model_name_prefix}_output_{getattr(config, 'seed', 0)}_{i}.mp4"
    audio_i = audios[i] if audios is not None else None

    audio_format = getattr(config, "audio_format", "s16")

    export_to_video_with_audio(
        video=videos[i],
        fps=fps,
        audio=audio_i,
        audio_sample_rate=audio_sample_rate,
        output_path=video_path,
        audio_format=audio_format,
    )

    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  timing_str = (
      f"\n{'=' * 50}\n"
      f"  TIMING SUMMARY\n"
      f"{'=' * 50}\n"
      f"  Load (checkpoint):   {load_time:>7.1f}s\n"
      f"  Compile:             {compile_time:>7.1f}s\n"
      f"  {'─' * 40}\n"
      f"  Inference:           {generation_time:>7.1f}s\n"
  )
  if hasattr(out, "timings") and out.timings:
    timing_str += (
        f"    Text Encoding:     {out.timings.get('Text Encoding', 0.0):>7.1f}s\n"
        f"    Preparation:       {out.timings.get('Preparation', 0.0):>7.1f}s\n"
        f"    Connectors:        {out.timings.get('Connectors', 0.0):>7.1f}s\n"
        f"    Denoising:         {out.timings.get('Denoising', 0.0):>7.1f}s\n"
    )
    if out.timings.get("Latent Upsampler", 0.0) > 0.0:
      timing_str += f"    Latent Upsampler:  {out.timings.get('Latent Upsampler', 0.0):>7.1f}s\n"
    timing_str += (
        f"    Latent Processing: {out.timings.get('Latent Processing', 0.0):>7.1f}s\n"
        f"    Video VAE:         {out.timings.get('Video VAE', 0.0):>7.1f}s\n"
        f"    Video Post:        {out.timings.get('Video Post', 0.0):>7.1f}s\n"
        f"    Audio VAE:         {out.timings.get('Audio VAE', 0.0):>7.1f}s\n"
        f"    Vocoder:           {out.timings.get('Vocoder', 0.0):>7.1f}s\n"
    )
  timing_str += f"{'=' * 50}"
  max_logging.log(timing_str)

  # Free memory before profiling
  del out
  del videos
  del audios

  # ---------------------------------------------------------
  # Run 3: Profiling Run (Only if profiling was originally enabled)
  # ---------------------------------------------------------
  if original_enable_profiler or original_enable_mld:
    skip_first_n_steps_for_profiler = config.get_keys().get("skip_first_n_steps_for_profiler", 0)
    if skip_first_n_steps_for_profiler != 0:
      max_logging.log(
          "\n⚠️ WARNING: 'skip_first_n_steps_for_profiler' is ignored because 'scan_diffusion_loop' is enabled! The profiler will capture all steps in this profile run.\n"
      )

    profiling_steps = config.get_keys().get("profiler_steps", 5)

    config.get_keys()["enable_profiler"] = False
    config.get_keys()["enable_ml_diagnostics"] = False
    config.get_keys()["num_inference_steps"] = profiling_steps

    max_logging.log(f"🚀 Warmup for profiling pass ({profiling_steps} steps)...")
    _ = call_pipeline(config, pipeline, prompt, negative_prompt)

    config.get_keys()["enable_profiler"] = original_enable_profiler
    config.get_keys()["enable_ml_diagnostics"] = original_enable_mld

    max_logging.log(f"🚀 Starting Profiling run ({profiling_steps} steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()

    _ = call_pipeline(config, pipeline, prompt, negative_prompt)

    profiler.stop()

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)
