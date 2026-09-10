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
import json
import jax
import jax.numpy as jnp
import time
import os
import subprocess
import uuid
from maxdiffusion.checkpointing.ltx2_checkpointer import LTX2Checkpointer
from maxdiffusion import aot_cache, pyconfig, max_logging, max_utils
from absl import app
import flax

from maxdiffusion.utils.export_utils import export_to_video_with_audio
from maxdiffusion.loaders.ltx2_lora_nnx_loader import LTX2NNXLoraLoader


def get_git_commit_hash():
  """Returns HEAD only when the tracked source tree is clean."""
  source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  try:
    commit_hash = (
        subprocess.check_output(["git", "-C", source_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        .strip()
        .decode("utf-8")
    )
    tracked_changes = subprocess.check_output(
        ["git", "-C", source_root, "status", "--porcelain", "--untracked-files=no"],
        stderr=subprocess.DEVNULL,
    ).strip()
    untracked_source = subprocess.check_output(
        ["git", "-C", source_root, "ls-files", "--others", "--exclude-standard", "--", "src/maxdiffusion"],
        stderr=subprocess.DEVNULL,
    ).strip()
    if tracked_changes or untracked_source:
      max_logging.log("Warning: Git source changes detected; persistent LTX2 AOT cache reuse is disabled.")
      return f"dirty:{commit_hash}"
    return commit_hash
  except subprocess.CalledProcessError as e:
    max_logging.log(f"Warning: unable to determine a clean Git revision ({e}). Not running in a Git repo?")
    return None
  except FileNotFoundError as e:
    max_logging.log(f"Warning: 'git' command not found ({e}).")
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
      output_type=getattr(config, "upsampler_output_type", "np_uint8"),
  )
  return out


def maybe_tune_block_sizes(config):
  """Tunes and applies the exact block-size fields consumed by production inference."""
  if not _tile_search_enabled(config):
    return
  keys = config.get_keys()
  from maxdiffusion.utils.tile_size_grid_search import grid_search
  from maxdiffusion.utils.ltx2_block_benchmark import LTX2BlockBenchmark

  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  vmem_limit_bytes = int(
      keys.get("tile_search_vmem_limit_bytes") or config.flash_block_sizes.get("vmem_limit_bytes") or 64 * 1024 * 1024
  )
  bench = LTX2BlockBenchmark.from_config(config, mesh, vmem_limit_bytes=vmem_limit_bytes)
  max_logging.log(f"[tile-search] tuning block sizes for {bench.label} before inference...")
  result = grid_search(
      bench,
      mode=keys.get("tile_search_mode", "smart"),
      iters=keys.get("tile_search_iters", 10),
      out_dir=(keys.get("tile_search_out", "") or None),
      log=max_logging.log,
  )
  if result.best is None:
    raise RuntimeError(
        "[tile-search] tuning was explicitly enabled, but no candidate succeeded. "
        "Inspect the per-candidate errors instead of running with an untuned configuration."
    )
  fbs = max_utils.flash_block_sizes_for_candidate(
      config.flash_block_sizes,
      config.attention,
      result.best.bq,
      result.best.bkv,
      result.best.bkv_compute,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  config.get_keys()["flash_block_sizes"] = fbs
  effective_block_sizes = max_utils.get_flash_block_sizes(config)
  if effective_block_sizes is None or effective_block_sizes.block_q != result.best.bq:
    effective_bq = None if effective_block_sizes is None else effective_block_sizes.block_q
    raise RuntimeError(f"[tile-search] selected block_q={result.best.bq}, but production resolved block_q={effective_bq}.")
  max_logging.log(
      f"[tile-search] using block_q={result.best.bq} block_kv={result.best.bkv} "
      f"(block-bench {result.best.mean_ms:.2f} ms)"
  )


def _tile_search_enabled(config) -> bool:
  return str(config.get_keys().get("enable_tile_search", False)).lower() in ("true", "1", "yes")


def _canonical_aot_value(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _non_reusable_aot_revision():
  """Returns a unique identity so unversioned source can never hit old HLO."""
  return f"unversioned:{uuid.uuid4().hex}"


def _resolve_ltx2_aot_source_revision(config, commit_hash=None):
  """Prefers an explicit Git revision, then a packaged-build revision."""
  for revision in (commit_hash, getattr(config, "aot_build_revision", None)):
    if revision is not None and str(revision).strip():
      return str(revision).strip()
  return None


def _is_reusable_aot_revision(source_revision) -> bool:
  if source_revision is None or not str(source_revision).strip():
    return False
  return not str(source_revision).startswith(("dirty:", "unversioned:"))


def ltx2_aot_metadata(config, pipeline, source_revision=None):
  """Returns every graph- and topology-shaping input to LTX2 AOT caching.

  We deliberately exclude values such as model weights and RNG state: they are
  executable inputs, not compilation inputs. `use_kv_cache` is also excluded
  because it is a static argument of `run_diffusion_loop` and therefore part
  of that executable's per-call signature. If no clean source/build revision
  is available, a per-run identity prevents stale executable reuse.
  """
  source_revision = str(source_revision).strip() if source_revision is not None else ""
  if not source_revision:
    source_revision = _non_reusable_aot_revision()

  transformer_config = dict(getattr(pipeline.transformer, "config", {}))
  for key in (
      "rngs",
      "mesh",
      "dtype",
      "weights_dtype",
      "precision",
      "flash_block_sizes",
      "flash_min_seq_length",
      "scan_layers",
      "attention_kernel",
      "a2v_attention_kernel",
      "v2a_attention_kernel",
      "ulysses_shards",
      "ulysses_attention_chunks",
      "remat_policy",
      "names_which_can_be_saved",
      "names_which_can_be_offloaded",
      "sharding_specs",
      "enable_jax_named_scopes",
  ):
    transformer_config.pop(key, None)

  config_keys = [
      "attention",
      "a2v_attention_kernel",
      "v2a_attention_kernel",
      "flash_block_sizes",
      "flash_min_seq_length",
      "use_base2_exp",
      "use_experimental_scheduler",
      "enable_jax_named_scopes",
      "ulysses_shards",
      "ulysses_attention_chunks",
      "scan_layers",
      "scan_diffusion_loop",
      "precision",
      "remat_policy",
      "names_which_can_be_saved",
      "names_which_can_be_offloaded",
      "spatio_temporal_guidance_blocks",
      "logical_axis_rules",
      "sharding",
      "weights_dtype",
      "activations_dtype",
  ]
  config_dict = {k: getattr(config, k, None) for k in config_keys}

  device = jax.devices()[0]
  return {
      "source_revision": source_revision,
      "model": config.pretrained_model_name_or_path,
      "transformer_architecture": _canonical_aot_value(transformer_config),
      "config": _canonical_aot_value(config_dict),
      "mesh_shape": str(pipeline.mesh.shape),
      "mesh_axes": _canonical_aot_value(pipeline.mesh.axis_names),
      "backend": jax.default_backend(),
      "device_kind": getattr(device, "device_kind", ""),
      "process_count": str(jax.process_count()),
      "jax": jax.__version__,
      "jaxlib": getattr(jax.lib, "__version__", ""),
  }


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  if pipeline is not None and _tile_search_enabled(config):
    raise ValueError(
        "enable_tile_search cannot be used with a prebuilt pipeline because block sizes are static in its graph. "
        "Tune before constructing the pipeline, or call run without the pipeline argument."
    )
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

  if getattr(config, "enable_vae_slicing", False):
    pipeline.enable_vae_slicing()
  if getattr(config, "enable_vae_tiling", False):
    pipeline.enable_vae_tiling()

  s0 = time.perf_counter()

  # Load prompts from prompt_file or default prompt
  prompt_file = getattr(config, "prompt_file", "")
  default_prompt = getattr(config, "prompt", "A cat playing piano")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=default_prompt)
  batch_size = getattr(config, "global_batch_size_to_train_on", 1)
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  # Using global_batch_size_to_train_on to map prompts
  warmup_prompt = [prompts[0]] * batch_size
  negative_prompt_str = getattr(config, "negative_prompt", "")
  warmup_negative_prompt = [negative_prompt_str] * batch_size

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
  max_logging.log(f"total prompts to generate: {len(prompts)}")
  max_logging.log("============================================================")

  original_enable_profiler = config.get_keys().get("enable_profiler", False)
  original_enable_mld = config.get_keys().get("enable_ml_diagnostics", False)
  original_num_steps = config.get_keys().get("num_inference_steps", 40)

  # Per-shape AOT executable cache
  detected_revision = commit_hash if commit_hash is not None else get_git_commit_hash()
  source_revision = _resolve_ltx2_aot_source_revision(config, detected_revision)
  aot_cache_dir = getattr(config, "aot_cache_dir", "")
  if aot_cache_dir and not _is_reusable_aot_revision(source_revision):
    max_logging.log(
        "[aot] No clean Git commit or aot_build_revision was supplied; "
        "persistent LTX2 AOT caching is disabled for this run."
    )
    aot_cache_dir = ""
  aot_metadata = ltx2_aot_metadata(config, pipeline, source_revision=source_revision)
  max_logging.log(f"[aot] LTX2 cache metadata: {_canonical_aot_value(aot_metadata)}")
  aot_cache.install(
      aot_cache_dir,
      meta=aot_metadata,
      mesh=pipeline.mesh,
  )
  aot_cache.wait_for_loads()

  # ---------------------------------------------------------
  # Run 1: Warmup Compilation (Original steps, NO profiling)
  # ---------------------------------------------------------
  config.get_keys()["enable_profiler"] = False
  config.get_keys()["enable_ml_diagnostics"] = False

  # When scan_diffusion_loop=True the entire denoising loop is compiled as a
  # single jax.lax.scan whose iteration count is baked into the XLA program via
  # array shapes.  A 2-step warmup would compile a *different* program than the
  # real N-step run, forcing a wasteful second compilation.  Only reduce warmup
  # steps when scan_diffusion_loop=False (Python loop), where compilation
  # happens per-step and is independent of the total iteration count.
  scan_diffusion_loop = getattr(config, "scan_diffusion_loop", True)
  if scan_diffusion_loop:
    warmup_steps = original_num_steps
  else:
    warmup_steps = min(2, original_num_steps)
  config.get_keys()["num_inference_steps"] = warmup_steps

  max_logging.log(f"🚀 Starting warmup compilation pass ({warmup_steps} steps)...")
  with aot_cache.warmup_mode():
    _ = call_pipeline(config, pipeline, warmup_prompt, warmup_negative_prompt)

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
  saved_video_path = []
  audio_sample_rate = (
      getattr(pipeline.vocoder.config, "output_sampling_rate", 24000)
      if getattr(pipeline, "vocoder", None) is not None
      else 24000
  )
  fps = getattr(config, "fps", 24)
  audio_format = getattr(config, "audio_format", "s16")
  model_name = getattr(config, "model_name", "ltx2") or "ltx2"
  model_name_prefix = model_name.replace(".", "_")
  gcs_output_path = max_utils.get_gcs_output_path(config)

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [negative_prompt_str] * batch_size
    out = call_pipeline(config, pipeline, prompt, negative_prompt)
    videos = out.frames if hasattr(out, "frames") else out[0]
    audios = out.audio if hasattr(out, "audio") else None
    for i in range(len(videos)):
      video_path = f"{filename_prefix}{model_name_prefix}_output_{getattr(config, 'seed', 0)}_{i}.mp4"
      audio_i = audios[i] if audios is not None else None
      export_to_video_with_audio(
          video=videos[i],
          fps=fps,
          audio=audio_i,
          audio_sample_rate=audio_sample_rate,
          output_path=video_path,
          audio_format=audio_format,
      )
      saved_video_path.append(video_path)
      if gcs_output_path:
        max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
  else:
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [negative_prompt_str] * batch_size

      out = call_pipeline(config, pipeline, padded_chunk, negative_prompt)
      videos = out.frames if hasattr(out, "frames") else out[0]
      audios = out.audio if hasattr(out, "audio") else None
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = f"{filename_prefix}{model_name_prefix}_output_{getattr(config, 'seed', 0)}_{prompt_idx}.mp4"
        audio_j = audios[j] if audios is not None else None
        export_to_video_with_audio(
            video=videos[j],
            fps=fps,
            audio=audio_j,
            audio_sample_rate=audio_sample_rate,
            output_path=video_path,
            audio_format=audio_format,
        )
        saved_video_path.append(video_path)
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")

  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_videos = len(saved_video_path)
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")

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

    profiler_prompt = [prompts[0]] * batch_size
    profiler_negative_prompt = [negative_prompt_str] * batch_size

    max_logging.log(f"🚀 Warmup for profiling pass ({profiling_steps} steps)...")
    _ = call_pipeline(config, pipeline, profiler_prompt, profiler_negative_prompt)

    config.get_keys()["enable_profiler"] = original_enable_profiler
    config.get_keys()["enable_ml_diagnostics"] = original_enable_mld

    max_logging.log(f"🚀 Starting Profiling run ({profiling_steps} steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()

    _ = call_pipeline(config, pipeline, profiler_prompt, profiler_negative_prompt)

    profiler.stop()

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  commit_hash = max_utils.get_git_commit_hash()
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)
