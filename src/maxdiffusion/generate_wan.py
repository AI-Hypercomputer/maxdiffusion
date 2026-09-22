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

import time
import os
import uuid
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion import aot_cache, pyconfig, max_logging, max_utils, wan_runtime_options
from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import resolve_rope_accum
from absl import app
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image
import flax
from maxdiffusion.common_types import WAN2_1, WAN2_2
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2


jax.config.update("jax_use_shardy_partitioner", True)


import functools
import hashlib


def _non_reusable_aot_revision():
  """Returns a unique identity so unversioned/dirty development source can never hit old HLO."""
  return f"unversioned:{uuid.uuid4().hex}"


@functools.lru_cache(maxsize=1)
def _compute_wan_source_hash() -> str | None:
  """Computes a deterministic SHA-256 content hash of all non-test package source files."""
  try:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    hasher = hashlib.sha256()
    py_files = []
    for root, dirs, files in os.walk(pkg_dir):
      dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
      for f in files:
        if f.endswith(".py"):
          py_files.append(os.path.join(root, f))
    for path in sorted(set(py_files)):
      rel = os.path.relpath(path, pkg_dir).replace(os.sep, "/")
      hasher.update(rel.encode("utf-8"))
      with open(path, "rb") as f:
        hasher.update(f.read())
    return f"src:{hasher.hexdigest()[:16]}"
  except Exception:  # noqa: BLE001
    return None


def _get_pkg_version(module_name: str) -> str:
  try:
    mod = __import__(module_name)
    return str(getattr(mod, "__version__", "unknown"))
  except ImportError:
    return "not_installed"


def _resolve_wan_aot_source_revision(config, commit_hash=None):
  """Prefers explicit aot_build_revision, then package source hash, then git commit hash."""
  explicit = getattr(config, "aot_build_revision", None)
  if explicit is not None and str(explicit).strip():
    return str(explicit).strip()
  src_hash = _compute_wan_source_hash()
  if src_hash is not None:
    return src_hash
  clean_commit = str(commit_hash).strip() if commit_hash is not None and str(commit_hash).strip() else None
  if clean_commit is not None:
    return clean_commit
  return None


def _is_reusable_aot_revision(source_revision) -> bool:
  if source_revision is None or not str(source_revision).strip():
    return False
  s = str(source_revision).strip()
  if s.startswith(("dirty:", "unversioned:")) or s.endswith("-dirty"):
    return False
  return True


def format_video_output_path(
    output_dir: str,
    run_name: str,
    seed: int,
    index: int,
    filename_prefix: str = "",
) -> str:
  """Formats the target mp4 path for a generated video."""
  if output_dir and not output_dir.startswith("gs://"):
    return os.path.join(output_dir, f"{filename_prefix}{run_name}_{seed}_{index}.mp4")
  return f"{filename_prefix}wan_output_{seed}_{index}.mp4"


def _build_wan_aot_metadata(config, mesh, source_revision) -> dict[str, str]:
  """Builds the install-time configuration metadata dictionary for Wan AOT caching."""
  first_dev = jax.devices()[0] if jax.devices() else None
  platform_version = getattr(first_dev, "platform_version", "unknown") if first_dev else "unknown"
  try:
    default_matmul_precision = str(
        jax.config.read("jax_default_matmul_precision")
        if hasattr(jax.config, "read")
        else getattr(jax.config, "jax_default_matmul_precision", "default")
    )
  except Exception:  # noqa: BLE001
    default_matmul_precision = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "default")
  try:
    default_prng_impl = str(
        jax.config.read("jax_default_prng_impl")
        if hasattr(jax.config, "read")
        else getattr(jax.config, "jax_default_prng_impl", "threefry2x32")
    )
  except Exception:  # noqa: BLE001
    default_prng_impl = "default"

  return {
      "model": str(getattr(config, "pretrained_model_name_or_path", "")),
      "wan_transformer_pretrained_model_name_or_path": str(
          getattr(config, "wan_transformer_pretrained_model_name_or_path", "")
      ),
      "attention": str(getattr(config, "attention", "")),
      # Kernel block sizes change the lowered graph, not the input
      # shapes — they must key the executable or a re-tuned config
      # would silently hit stale binaries.
      "flash_block_sizes": str(getattr(config, "flash_block_sizes", {})),
      "mesh_shape": str(mesh.shape if mesh is not None else ()),
      "vae_spatial": str(getattr(config, "vae_spatial", 8)),
      "vae_decode_chunk": str(getattr(config, "vae_decode_chunk", 1)),
      "vae_encode_chunk": str(getattr(config, "vae_encode_chunk", 0)),
      "replicate_vae": str(getattr(config, "replicate_vae", False)),
      "vae_logical_axis_rules": str(getattr(config, "vae_logical_axis_rules", ())),
      "vae_weights_dtype": str(getattr(config, "vae_weights_dtype", "bfloat16")),
      "vae_dtype": str(getattr(config, "vae_dtype", "bfloat16")),
      "weights_dtype": str(getattr(config, "weights_dtype", "")),
      "activations_dtype": str(getattr(config, "activations_dtype", "")),
      "scan_layers": str(getattr(config, "scan_layers", True)),
      "remat_policy": str(getattr(config, "remat_policy", "NONE")),
      "ulysses_shards": str(getattr(config, "ulysses_shards", 1)),
      "ulysses_attention_chunks": str(getattr(config, "ulysses_attention_chunks", 1)),
      "use_k_centering": str(getattr(config, "use_k_centering", "auto")),
      "use_kv_cache": str(getattr(config, "use_kv_cache", False)),
      "use_cfg_cache": str(getattr(config, "use_cfg_cache", False)),
      "use_magcache": str(getattr(config, "use_magcache", False)),
      "use_sen_cache": str(getattr(config, "use_sen_cache", False)),
      "use_qwix_quantization": str(getattr(config, "use_qwix_quantization", False)),
      "quantization": str(getattr(config, "quantization", "")),
      "qwix_module_path": str(getattr(config, "qwix_module_path", "")),
      "enable_lora": str(getattr(config, "enable_lora", False)),
      "lora_config": str(getattr(config, "lora_config", {})),
      "flash_min_seq_length": str(getattr(config, "flash_min_seq_length", 4096)),
      "mask_padding_tokens": str(getattr(config, "mask_padding_tokens", True)),
      "precision": str(getattr(config, "precision", "default")),
      "split_head_dim": str(getattr(config, "split_head_dim", True)),
      "logical_axis_rules": str(getattr(config, "logical_axis_rules", ())),
      "attention_sharding_uniform": str(getattr(config, "attention_sharding_uniform", True)),
      "allow_split_physical_axes": str(getattr(config, "allow_split_physical_axes", False)),
      "device_kind": str(first_dev.device_kind if first_dev else "unknown"),
      "platform_version": str(platform_version),
      "process_count": str(jax.process_count()),
      "use_base2_exp": str(getattr(config, "use_base2_exp", True)),
      "use_experimental_scheduler": str(getattr(config, "use_experimental_scheduler", False)),
      "use_fused_rope_kernel": str(getattr(config, "use_fused_rope_kernel", "auto")),
      "fused_rope_block_s": str(getattr(config, "fused_rope_block_s", 512)),
      "fused_rope_head_block": str(getattr(config, "fused_rope_head_block", 2)),
      "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
      "xla_flags": os.environ.get("XLA_FLAGS", ""),
      # Graph-changing Wan switches (YAML `wan_*` keys, see wan_runtime_options).
      **{
          name: str(getattr(config, name)) if getattr(config, name, None) is not None else value
          for name, value in wan_runtime_options.snapshot().items()
          if name != "wan_rope_accum"
      },
      # The RoPE accumulation mode changes the lowered graph's rounding, and
      # its default is platform-dependent, so the RESOLVED mode is recorded
      # rather than the raw env var. Recording the raw value would both
      # over-invalidate (an explicit "f32" on v6e is the same executable as
      # the unset default) and fail to describe what was actually compiled.
      "wan_rope_accum": str(resolve_rope_accum(mesh)),
      "jax": jax.__version__,
      "jaxlib": _get_pkg_version("jaxlib"),
      "flax": _get_pkg_version("flax"),
      "qwix": _get_pkg_version("qwix"),
      "tokamax": _get_pkg_version("tokamax"),
      "default_matmul_precision": default_matmul_precision,
      "default_prng_impl": default_prng_impl,
      "source_revision": source_revision if source_revision else _non_reusable_aot_revision(),
  }


def call_pipeline(config, pipeline, prompt, negative_prompt, num_inference_steps=None):
  model_key = config.model_name
  model_type = config.model_type
  if num_inference_steps is None:
    num_inference_steps = config.num_inference_steps
  if model_type == "I2V":
    image = load_image(config.image_url)
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
          use_kv_cache=config.use_kv_cache,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
          use_kv_cache=config.use_kv_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
      )
    else:
      raise ValueError(f"Unsupported model_name for I2V in config: {model_key}")
  elif model_type == "T2V":
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_cfg_cache=config.use_cfg_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
          use_kv_cache=config.use_kv_cache,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
          use_kv_cache=config.use_kv_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
      )
    else:
      raise ValueError(f"Unsupported model_name for T2V in config: {model_key}")


def inference_generate_video(config, pipeline, filename_prefix=""):
  s0 = time.perf_counter()
  prompt_file = getattr(config, "prompt_file", "")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=config.prompt)
  batch_size = config.global_batch_size_to_train_on
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width},"
      f" frames: {config.num_frames}, total prompts: {len(prompts)}, video prefix: {filename_prefix}"
  )

  gcs_output_path = max_utils.get_gcs_output_path(config)
  saved_video_paths = []

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [config.negative_prompt] * batch_size
    videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    max_logging.log(f"video {filename_prefix}, generation time: {(time.perf_counter() - s0):.2f}s")
    for i in range(len(videos)):
      video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
      export_to_video(videos[i], video_path, fps=config.fps)
      saved_video_paths.append(video_path)
      if gcs_output_path:
        max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
        max_utils.delete_file(f"./{video_path}")
  else:
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [config.negative_prompt] * batch_size

      videos = call_pipeline(config, pipeline, padded_chunk, negative_prompt)
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = f"{filename_prefix}wan_output_{config.seed}_{prompt_idx}.mp4"
        export_to_video(videos[j], video_path, fps=config.fps)
        saved_video_paths.append(video_path)
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
          max_utils.delete_file(f"./{video_path}")
    max_logging.log(f"all videos {filename_prefix}, total generation time: {(time.perf_counter() - s0):.2f}s")

  return saved_video_paths


def maybe_tune_block_sizes(config):
  """If enable_tile_search, run a fast one-DiT-block tile-size grid search and overwrite
  flash_block_sizes' block_q/block_kv/block_kv_compute with the winner IN PLACE, before the
  transformer (which bakes block sizes in at construction) is built.

  Flags are read defensively so this is a safe no-op (grid search OFF) for any config that
  doesn't declare them -- not every WAN yaml carries the tile_search_* keys."""
  keys = config.get_keys()
  if not keys.get("enable_tile_search", False):
    return
  from maxdiffusion.utils.tile_size_grid_search import grid_search
  from maxdiffusion.utils.wan_block_benchmark import WanBlockBenchmark

  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  bench = WanBlockBenchmark.from_config(config, mesh)
  max_logging.log(f"[tile-search] tuning block sizes for {bench.label} before inference...")
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
  fbs = dict(config.flash_block_sizes)
  fbs.update({
      "block_q": result.best.bq,
      "block_kv": result.best.bkv,
      "block_kv_compute": result.best.bkv_compute,
      "block_kv_compute_in": result.best.bkv_compute,
  })
  config.get_keys()["flash_block_sizes"] = fbs  # config is immutable via setattr; mutate raw dict
  max_logging.log(
      f"[tile-search] using block_q={result.best.bq} block_kv={result.best.bkv} "
      f"(block-bench {result.best.mean_ms:.2f} ms)"
  )


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  # Graph-changing Wan switches come from the config; load them before any
  # model is built or traced so every consumer sees the same values.
  wan_runtime_options.configure_from_config(config)
  model_key = config.model_name
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

  if pipeline is None:
    load_start = time.perf_counter()
    model_type = config.model_type
    if model_key == WAN2_1:
      pipeline_cls = WanPipelineI2V_2_1 if model_type == "I2V" else WanPipeline2_1
      pretrained_state_sources = (("wan_state", "transformer"),)
      pretrained_config_transformer_attr = "transformer"
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_1(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_1(config=config)
    elif model_key == WAN2_2:
      pipeline_cls = WanPipelineI2V_2_2 if model_type == "I2V" else WanPipeline2_2
      pretrained_state_sources = (
          ("low_noise_transformer_state", "low_noise_transformer"),
          ("high_noise_transformer_state", "high_noise_transformer"),
      )
      # WAN 2.2 training checkpoints save `wan_config` from the low-noise transformer.
      pretrained_config_transformer_attr = "low_noise_transformer"
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_2(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_2(config=config)
    else:
      raise ValueError(f"Unsupported model_name for checkpointer: {model_key}")
    checkpoint_step = checkpoint_loader.checkpoint_manager.latest_step()
    if checkpoint_step is not None:
      pipeline, _, _ = checkpoint_loader.load_checkpoint(checkpoint_step)
    else:
      pipeline = checkpoint_loader.load_pretrained_pipeline_or_diffusers(
          config,
          pipeline_cls,
          pretrained_state_sources,
          pretrained_config_transformer_attr,
      )
    load_time = time.perf_counter() - load_start
    max_logging.log(f"load_time: {load_time:.1f}s")
  else:
    load_time = 0.0

  # If LoRA is specified, inject layers and load weights.
  if (
      config.enable_lora
      and hasattr(config, "lora_config")
      and config.lora_config
      and config.lora_config["lora_model_name_or_path"]
  ):
    if model_key == WAN2_1:
      lora_loader = Wan2_1NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            transformer_weight_name=lora_config["weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

    if model_key == WAN2_2:
      lora_loader = Wan2_2NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            high_noise_weight_name=lora_config["high_noise_weight_name"][i],
            low_noise_weight_name=lora_config["low_noise_weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

  # Per-shape AOT executable cache: deserialization starts on background
  # threads now and overlaps the remaining setup; unknown shapes silently
  # fall back to jit and are serialized by save_pending() after warmup and
  # again after generation (persistent cache only).
  detected_revision = commit_hash if commit_hash is not None else max_utils.get_git_commit_hash()
  source_revision = _resolve_wan_aot_source_revision(config, detected_revision)
  aot_cache_dir = getattr(config, "aot_cache_dir", "")
  if aot_cache_dir and not _is_reusable_aot_revision(source_revision):
    max_logging.log(
        "[aot] Persistent Wan AOT caching is disabled for this development run; "
        "using ephemeral cache for zero-execution warmup."
    )
    aot_cache_dir = ""

  # If persistent cache is not enabled, use an ephemeral directory so zero-execution
  # warmup and weight priming still operate without touching shared disk storage.
  if not aot_cache_dir:
    import atexit
    import shutil
    import tempfile

    install_cache_dir = tempfile.mkdtemp(prefix="wan_aot_ephemeral_")
    atexit.register(shutil.rmtree, install_cache_dir, ignore_errors=True)
  else:
    install_cache_dir = aot_cache_dir

  aot_metadata = _build_wan_aot_metadata(config, pipeline.mesh, source_revision)
  aot_cache.install(
      install_cache_dir,
      meta={**aot_metadata, **aot_cache.extract_svg_meta(config, pipeline)},
      mesh=pipeline.mesh,
  )
  # Deserialization is seconds and warmup must see the loaded executables
  # to hit them; without this the first call races the loader threads.
  aot_cache.wait_for_loads()

  s0 = time.perf_counter()

  # Disable profiler for the first two runs to avoid duplicate uploads
  original_enable_profiler = config.enable_profiler if "enable_profiler" in config.get_keys() else False
  config.get_keys()["enable_profiler"] = False

  prompt_file = getattr(config, "prompt_file", "")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=config.prompt)
  batch_size = config.global_batch_size_to_train_on
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  # Using global_batch_size_to_train_on so not to create more config variables
  warmup_prompt = [prompts[0]] * batch_size
  warmup_negative_prompt = [config.negative_prompt] * batch_size

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width},"
      f" frames: {config.num_frames}, total prompts: {len(prompts)}"
  )
  # Warmup with 2 denoising steps instead of a full run. The step count only
  # changes the Python loop trip count, not traced shapes. With flow_shift=12
  # both warmup timesteps (t=999, 923) are above the WAN 2.2 boundary (875), so
  # the loop itself only reaches the high-noise transformer; run_inference_2_2
  # additionally compiles the forward pass of any phase the warmup schedule
  # skips. Together this compiles every executable of the full run (both
  # transformers, text encoder, VAE decode) at a fraction of the cost.
  warmup_steps = min(2, config.num_inference_steps)
  max_logging.log(f"Compile warmup: {warmup_steps} denoising steps")
  # Zero-execution warmup: wrapped transformer passes lower+compile (or
  # reuse the deserialized AOT executable) and return sharded zeros, so
  # the warmup pays compile time only, never real denoise compute. The
  # returned videos are garbage by design and are discarded below.
  with aot_cache.warmup_mode():
    videos = call_pipeline(
        config,
        pipeline,
        warmup_prompt,
        warmup_negative_prompt,
        num_inference_steps=warmup_steps,
    )
  if isinstance(videos, tuple):
    videos, warmup_trace = videos
    warmup_str = ", ".join(f"{stage}={seconds:.1f}s" for stage, seconds in warmup_trace.items())
    max_logging.log(f"Warmup breakdown: {warmup_str}")

  # Serialize newly-compiled shapes synchronously inside warmup-accounted time
  # (a background save would stall the first request). Skipped for the
  # ephemeral (non-persistent) cache dir.
  if aot_cache_dir:
    aot_cache.save_pending()
  else:
    aot_cache.clear_pending()

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log(f"total prompts to generate: {len(prompts)}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  s0 = time.perf_counter()
  saved_video_path = []
  gcs_output_path = max_utils.get_gcs_output_path(config)

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [config.negative_prompt] * batch_size
    outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(outputs, tuple):
      videos, trace = outputs
    else:
      videos = outputs
      trace = {}
    for i in range(len(videos)):
      video_path = format_video_output_path(
          getattr(config, "output_dir", ""),
          config.run_name,
          config.seed,
          i,
          filename_prefix,
      )
      saved_video_path.append(video_path)
      if jax.process_index() == 0:
        import numpy as np

        if getattr(config, "output_dir", "") and not config.output_dir.startswith("gs://"):
          os.makedirs(config.output_dir, exist_ok=True)
        frames_np = np.asarray(videos[i])
        export_to_video(frames_np, video_path, fps=config.fps)
        max_logging.log(f"Saved video to {video_path}")
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
  else:
    trace = {}
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [config.negative_prompt] * batch_size

      outputs = call_pipeline(config, pipeline, padded_chunk, negative_prompt)
      if isinstance(outputs, tuple):
        videos, trace = outputs
      else:
        videos = outputs
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = format_video_output_path(
            getattr(config, "output_dir", ""),
            config.run_name,
            config.seed,
            prompt_idx,
            filename_prefix,
        )
        saved_video_path.append(video_path)
        if jax.process_index() == 0:
          import numpy as np

          if getattr(config, "output_dir", "") and not config.output_dir.startswith("gs://"):
            os.makedirs(config.output_dir, exist_ok=True)
          frames_np = np.asarray(videos[j])
          export_to_video(frames_np, video_path, fps=config.fps)
          max_logging.log(f"Saved video to {video_path}")
          if gcs_output_path:
            max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")

  generation_time = time.perf_counter() - s0
  if aot_cache_dir:
    aot_cache.save_pending()
  else:
    aot_cache.clear_pending()
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_videos = len(saved_video_path)
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar(
          "inference/generation_time_per_video",
          generation_time_per_video,
          global_step=0,
      )
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")
  summary = [
      f"\n{'=' * 50}",
      "  TIMING SUMMARY",
      f"{'=' * 50}",
      f"  Load (checkpoint):   {load_time:>7.1f}s",
      f"  Compile:             {compile_time:>7.1f}s",
      f"  Inference:           {generation_time:>7.1f}s",
  ]
  if trace:
    vae_decode_total = trace.get("vae_decode", 0.0)
    vae_decode_tpu = trace.get("vae_decode_tpu", 0.0)
    vae_decode_post = vae_decode_total - vae_decode_tpu
    summary.extend([
        f"  {'─' * 40}",
        f"  Conditioning:        {trace.get('conditioning', 0.0):>7.1f}s",
        f"    - VAE Encode:      {trace.get('vae_encode', 0.0):>7.1f}s",
        f"  Denoise Total:       {trace.get('denoise_total', 0.0):>7.1f}s",
        f"  VAE Decode:          {vae_decode_total:>7.1f}s",
        f"    - TPU Compute:     {vae_decode_tpu:>7.1f}s",
        f"    - Host Formatting: {vae_decode_post:>7.1f}s",
    ])
  summary.append(f"{'=' * 50}")
  max_logging.log("\n".join(summary))

  s0 = time.perf_counter()
  # Restore original profiler setting for the profiling run
  config.get_keys()["enable_profiler"] = original_enable_profiler
  if original_enable_profiler:
    # Injecting user requested XLA tracing flags
    xla_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = "--xla_enable_mxu_trace=true --xla_jf_dump_llo_html=true --xla_tpu_enable_llo_profiling=true"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {new_flags}"
    max_logging.log(f"Injected XLA_FLAGS for profiling: {new_flags}")

    profiler_prompt = [prompts[0]] * batch_size
    profiler_negative_prompt = [config.negative_prompt] * batch_size
    videos = call_pipeline(config, pipeline, profiler_prompt, profiler_negative_prompt)
    if isinstance(videos, tuple):
      videos = videos[0]
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar(
          "inference/generation_time_with_profiler",
          generation_time_with_profiler,
          global_step=0,
      )

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
  with transformer_engine_context():
    app.run(main)
