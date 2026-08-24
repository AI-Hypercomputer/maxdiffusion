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

"""Text-to-image inference entry point for Ideogram 4."""

import json
import os
from functools import partial
from typing import Sequence
import jax
from jax.sharding import Mesh

import time

import subprocess
import numpy as np
from PIL import Image
from flax import nnx
from absl import app

from maxdiffusion import pyconfig, max_logging, max_utils
from maxdiffusion.checkpointing.ideogram_checkpointer import IdeogramCheckpointer
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import _denoise_step, _decode_latents


def _add_sharding_rule(vs: nnx.Variable, logical_axis_rules) -> nnx.Variable:
  vs.set_metadata(sharding_rules=logical_axis_rules)
  return vs


def create_sharded_logical_model(model, logical_axis_rules, mesh):
  """Shard a model from its own `nnx.with_partitioning` metadata.

  This used to match hardcoded path substrings ("qkv.kernel", ...) because the
  nnx modules carried no axis names, which meant renaming a layer silently
  dropped it to replicated and made the whole `logical_axis_rules` block in
  base_ideogram.yml inert. The modules now declare logical axes, so the repo's
  standard `nnx.get_partition_spec` path (see wan_pipeline.py) applies.

  Placement is an eager per-leaf `device_put`, not wan's jitted
  `with_sharding_constraint`: the pipeline loads weights to host memory
  (`load_transformer_weights(..., "cpu")`) to keep HBM low, and jit rejects
  CPU-committed inputs under a TPU sharding constraint.
  """
  if model is None:
    return None
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.tree.map(
      lambda x, p: jax.device_put(x, jax.sharding.NamedSharding(mesh, p)),
      state,
      pspecs,
  )
  return nnx.merge(graphdef, sharded_state, rest_of_state)


def maybe_tune_block_sizes(config):
  """If enable_tile_search, run a fast one-block tile-size grid search and
  overwrite flash_block_sizes' block_q/block_kv with the winner IN PLACE, before
  the transformer (which bakes block sizes in at construction) is built.

  Flags are read defensively so this is a safe no-op for a config that predates
  them. Only meaningful for attention='flash' -- the dense einsum path ignores
  block sizes entirely.
  """
  keys = config.get_keys()
  if not keys.get("enable_tile_search", False):
    return
  if config.attention != "flash":
    max_logging.log(f"[tile-search] attention={config.attention} ignores block sizes; skipping search")
    return

  from maxdiffusion.utils.ideogram_block_benchmark import IdeogramBlockBenchmark
  from maxdiffusion.utils.tile_size_grid_search import grid_search

  mesh = Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  bench = IdeogramBlockBenchmark.from_config(config, mesh, text_tokens=keys.get("tile_search_text_tokens", 512))
  max_logging.log(f"[tile-search] tuning block sizes for {bench.label} (seq={bench.seq_len}) before inference...")
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
  fbs.update({"block_q": result.best.bq, "block_kv": result.best.bkv})
  config.get_keys()["flash_block_sizes"] = fbs  # config is immutable via setattr; mutate raw dict
  max_logging.log(
      f"[tile-search] using block_q={result.best.bq} block_kv={result.best.bkv} "
      f"(block-bench {result.best.mean_ms:.2f} ms)"
  )


def get_git_commit_hash():
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    return commit_hash
  except subprocess.CalledProcessError:
    max_logging.log("Warning: 'git rev-parse HEAD' failed.")
    return None
  except FileNotFoundError:
    max_logging.log("Warning: 'git' command not found.")
    return None


jax.config.update("jax_use_shardy_partitioner", True)


def call_pipeline(config, pipeline, prompt, negative_prompt=None, mesh=None):
  seed = getattr(config, "seed", 42)
  height = getattr(config, "height", 256)
  width = getattr(config, "width", 256)
  num_inference_steps = getattr(config, "num_inference_steps", 50)
  guidance_scale = getattr(config, "guidance_scale", 7.0)

  # Convert single prompt to list of prompts to match pipeline batch dimension
  if isinstance(prompt, str):
    if mesh is not None:
      # Must come from the RESOLVED mesh, not the raw config: the idiomatic
      # "use all devices" value is -1, and multiplying by it yields a negative
      # prompt count, so `[prompt] * n` silently produces an empty batch.
      data_parallelism = mesh.shape["data"]
    else:
      data_parallelism = getattr(config, "dcn_data_parallelism", 1) * getattr(config, "ici_data_parallelism", 1)
      if data_parallelism < 1:
        raise ValueError(
            f"data parallelism resolved to {data_parallelism}; pass the mesh to call_pipeline so a "
            "-1 ('use all devices') config value can be resolved to a real axis size."
        )
    num_prompts = getattr(config, "per_device_batch_size", 1) * data_parallelism
    prompts = [prompt] * num_prompts
    if negative_prompt is None:
      negative_prompts = [""] * num_prompts
    elif isinstance(negative_prompt, str):
      negative_prompts = [negative_prompt] * num_prompts
    else:
      negative_prompts = negative_prompt
  else:
    prompts = prompt
    negative_prompts = negative_prompt

  return pipeline.generate(
      prompts=prompts,
      negative_prompts=negative_prompts,
      height=height,
      width=width,
      num_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      seed=seed,
  )


def run(config, filename_prefix="", commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")
    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")

  # Must run before the pipeline is built: the transformer bakes block sizes in
  # at construction time.
  maybe_tune_block_sizes(config)

  t0_load = time.perf_counter()
  max_logging.log("Loading pipeline weights for Ideogram via checkpointer...")

  checkpointer = IdeogramCheckpointer(config)
  pipeline, _, _ = checkpointer.load_checkpoint(load_transformer=True)

  load_time = time.perf_counter() - t0_load
  max_logging.log(f"Model loaded: {load_time:.1f}s")

  # Apply sharding over the device mesh
  max_logging.log("Applying sharding constraints to models...")
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  logical_axis_rules = tuple(tuple(rule) for rule in config.logical_axis_rules)
  with mesh:
    pipeline.conditional_transformer = create_sharded_logical_model(
        pipeline.conditional_transformer, logical_axis_rules, mesh
    )
    pipeline.unconditional_transformer = create_sharded_logical_model(
        pipeline.unconditional_transformer, logical_axis_rules, mesh
    )
    pipeline.autoencoder = create_sharded_logical_model(pipeline.autoencoder, logical_axis_rules, mesh)

  s0 = time.perf_counter()
  prompt = getattr(config, "prompt", "A cute dog")
  negative_prompt = getattr(config, "negative_prompt", "")

  max_logging.log(f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}")
  max_logging.log("===================== Model details =======================")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log("============================================================")

  original_enable_profiler = config.get_keys().get("enable_profiler", False)
  original_enable_mld = config.get_keys().get("enable_ml_diagnostics", False)
  original_num_steps = config.get_keys().get("num_inference_steps", 40)

  # 1. Warmup. The denoise step is compiled per *step*, not per loop, so the
  # executable a 2-step warmup builds is exactly the one the timed run reuses.
  # (Under the old fori_loop the trip count was baked into the while bound and
  # the sigmas array changed shape with num_steps, so a 2-step warmup compiled
  # nothing the 50-step run could use and `generation_time` was mostly compile.)
  config.get_keys()["enable_profiler"] = False
  config.get_keys()["enable_ml_diagnostics"] = False
  config.get_keys()["num_inference_steps"] = 2

  max_logging.log("🚀 Starting warmup compilation pass (2 steps)...")
  with mesh:
    warm, _ = call_pipeline(config, pipeline, prompt, negative_prompt, mesh=mesh)
  jax.block_until_ready(warm)

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")

  # 2. Actual Generation
  # pylint: disable=protected-access
  cache_before = (_denoise_step._cache_size(), _decode_latents._cache_size())

  config.get_keys()["num_inference_steps"] = original_num_steps
  s0 = time.perf_counter()
  max_logging.log(f"🚀 Starting actual full-length generation pass ({original_num_steps} steps)...")
  with mesh:
    out_images, trace = call_pipeline(config, pipeline, prompt, negative_prompt, mesh=mesh)
  jax.block_until_ready(out_images)
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")

  cache_after = (_denoise_step._cache_size(), _decode_latents._cache_size())
  if cache_after != cache_before:
    max_logging.log(
        f"WARNING: the timed run triggered recompilation (jit cache {cache_before} -> {cache_after}); "
        "generation_time includes compile and is not steady-state."
    )
  else:
    max_logging.log(f"No recompilation during the timed run. Per-step: {generation_time / original_num_steps * 1e3:.1f} ms")

  # Save images
  saved_image_paths = []
  actual_prefix = filename_prefix
  if not actual_prefix and getattr(config, "run_name", None):
    actual_prefix = getattr(config, "run_name") + "_"

  for i, out_image in enumerate(out_images):
    # Honor output_dir instead of dropping images in the current directory.
    out_dir = getattr(config, "output_dir", "") or "."
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, f"{actual_prefix}ideogram_output_{getattr(config, 'seed', 42)}_{i}.png")
    image_np = np.array(out_image)
    image_np = (image_np * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img.save(image_path)
    saved_image_paths.append(image_path)
    max_logging.log(f"Saved image to {image_path}")

  summary = [
      f"\n{'=' * 50}",
      "  TIMING SUMMARY",
      f"{'=' * 50}",
      f"  Load (checkpoint):   {load_time:>7.1f}s",
      f"  Compile:             {compile_time:>7.1f}s",
      f"  {'─' * 40}",
      f"  Inference:           {generation_time:>7.1f}s",
  ]
  if trace:
    steps = original_num_steps
    summary.extend([
        f"  {'─' * 40}",
        f"  Conditioning:        {trace.get('conditioning', 0.0):>7.1f}s",
        f"    - Input Prep:      {trace.get('input_prep', 0.0):>7.1f}s",
        f"    - Text Encode:     {trace.get('text_encode', 0.0):>7.1f}s",
        f"  Denoise Total:       {trace.get('denoise_total', 0.0):>7.1f}s  ({steps} steps)",
        f"    - First Step:      {trace.get('denoise_first_step', 0.0) * 1e3:>7.1f}ms",
        f"    - Steady/Step:     {trace.get('denoise_per_step', 0.0) * 1e3:>7.1f}ms",
        f"  VAE Decode:          {trace.get('vae_decode', 0.0):>7.1f}s",
    ])
  summary.append(f"{'=' * 50}")
  max_logging.log("\n".join(summary))

  # Machine-readable copy so sweep drivers do not have to regex the log.
  if jax.process_index() == 0:
    timing = {
        "load_time": load_time,
        "compile_time": compile_time,
        "generation_time": generation_time,
        "num_inference_steps": original_num_steps,
        "height": config.height,
        "width": config.width,
        "attention": config.attention,
        "flash_block_sizes": dict(config.flash_block_sizes) if config.flash_block_sizes else None,
        "per_device_batch_size": config.per_device_batch_size,
        "mesh": {name: int(size) for name, size in zip(config.mesh_axes, devices_array.shape)},
        "num_devices": jax.device_count(),
        "recompiled": cache_after != cache_before,
        **{f"trace_{k}": v for k, v in trace.items()},
    }
    out_dir = getattr(config, "output_dir", "") or "."
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{actual_prefix or 'ideogram_'}timing.json")
    with open(json_path, "w", encoding="utf-8") as fh:
      json.dump(timing, fh, indent=2)
    max_logging.log(f"Wrote timing json to {json_path}")

  # 3. Profiling Run
  if original_enable_profiler or original_enable_mld:
    profiling_steps = config.get_keys().get("profiler_steps", 5)
    config.get_keys()["enable_profiler"] = original_enable_profiler
    config.get_keys()["enable_ml_diagnostics"] = original_enable_mld
    config.get_keys()["num_inference_steps"] = profiling_steps

    max_logging.log(f"🚀 Starting Profiling run ({profiling_steps} steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()
    _ = call_pipeline(config, pipeline, prompt, negative_prompt, mesh=mesh)
    profiler.stop()

  return saved_image_paths


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)
