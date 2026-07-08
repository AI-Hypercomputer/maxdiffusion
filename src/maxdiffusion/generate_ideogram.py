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
from jax.sharding import Mesh

import time

import subprocess
import numpy as np
from PIL import Image
from flax import nnx
from absl import app

from maxdiffusion import pyconfig, max_logging, max_utils
from maxdiffusion.checkpointing.ideogram_checkpointer import IdeogramCheckpointer


def _add_sharding_rule(vs: nnx.Variable, logical_axis_rules) -> nnx.Variable:
  vs.set_metadata(sharding_rules=logical_axis_rules)
  return vs


def create_sharded_logical_model(model, logical_axis_rules, mesh):
  if model is None:
    return None
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)

  def map_leaf(path, leaf):
    if not isinstance(leaf, nnx.Variable):
      return jax.sharding.PartitionSpec()
    path_str = ".".join([str(p.key) if hasattr(p, "key") else str(p) for p in path])
    # Manually implement FSDP by matching layer names since nnx.Linear lacks axis_names
    if "qkv.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    elif "o.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    elif "w1.kernel" in path_str or "w3.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    elif "w2.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    elif "adaln_modulation.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    elif "final_layer.linear.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    elif "input_proj.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    elif "llm_cond_proj.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    elif "embed_image_indicator.embedding" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")

    # Fallback to replicated for small arrays like biases or norms
    leaf_shape = leaf.shape if hasattr(leaf, "shape") else leaf.value.shape
    if len(leaf_shape) == 1:
      return jax.sharding.PartitionSpec(None)
    elif len(leaf_shape) == 2:
      return jax.sharding.PartitionSpec(None, None)
    else:
      return jax.sharding.PartitionSpec(*([None] * len(leaf_shape)))

  pspecs = jax.tree_util.tree_map_with_path(map_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))

  sharded_state = jax.tree.map(
      lambda x, p: x.replace(
          value=jax.device_put(x.get_value() if hasattr(x, "get_value") else x.value, jax.sharding.NamedSharding(mesh, p))
      ),
      state,
      pspecs,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model


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


def call_pipeline(config, pipeline, prompt, negative_prompt=None):
  seed = getattr(config, "seed", 42)
  height = getattr(config, "height", 256)
  width = getattr(config, "width", 256)
  num_inference_steps = getattr(config, "num_inference_steps", 50)
  guidance_scale = getattr(config, "guidance_scale", 7.0)

  # Convert single prompt to list of prompts to match pipeline batch dimension
  if isinstance(prompt, str):
    data_parallelism = getattr(config, "dcn_data_parallelism", 1) * getattr(config, "ici_data_parallelism", 1)
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

  images = pipeline.generate(
      prompts=prompts,
      negative_prompts=negative_prompts,
      height=height,
      width=width,
      num_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      seed=seed,
  )
  return images


def run(config, filename_prefix="", commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")
    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")

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

  # 1. Warmup Compilation
  config.get_keys()["enable_profiler"] = False
  config.get_keys()["enable_ml_diagnostics"] = False
  config.get_keys()["num_inference_steps"] = 2  # lower for warmup

  max_logging.log("🚀 Starting warmup compilation pass (2 steps)...")
  with mesh:
    _ = call_pipeline(config, pipeline, prompt, negative_prompt)

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")

  # 2. Actual Generation
  config.get_keys()["num_inference_steps"] = original_num_steps
  s0 = time.perf_counter()
  max_logging.log(f"🚀 Starting actual full-length generation pass ({original_num_steps} steps)...")
  with mesh:
    out_images = call_pipeline(config, pipeline, prompt, negative_prompt)
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")

  # Save images
  saved_image_paths = []
  actual_prefix = filename_prefix
  if not actual_prefix and getattr(config, "run_name", None):
    actual_prefix = getattr(config, "run_name") + "_"

  for i in range(len(out_images)):
    image_path = f"{actual_prefix}ideogram_output_{getattr(config, 'seed', 42)}_{i}.png"
    image_np = np.array(out_images[i])
    image_np = (image_np * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img.save(image_path)
    saved_image_paths.append(image_path)
    max_logging.log(f"Saved image to {image_path}")

  timing_str = (
      f"\n{'=' * 50}\n"
      f"  TIMING SUMMARY\n"
      f"{'=' * 50}\n"
      f"  Load (checkpoint):   {load_time:>7.1f}s\n"
      f"  Compile:             {compile_time:>7.1f}s\n"
      f"  {'─' * 40}\n"
      f"  Inference:           {generation_time:>7.1f}s\n"
      f"{'=' * 50}"
  )
  max_logging.log(timing_str)

  # 3. Profiling Run
  if original_enable_profiler or original_enable_mld:
    profiling_steps = config.get_keys().get("profiler_steps", 5)
    config.get_keys()["enable_profiler"] = original_enable_profiler
    config.get_keys()["enable_ml_diagnostics"] = original_enable_mld
    config.get_keys()["num_inference_steps"] = profiling_steps

    max_logging.log(f"🚀 Starting Profiling run ({profiling_steps} steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()
    _ = call_pipeline(config, pipeline, prompt, negative_prompt)
    profiler.stop()

  return saved_image_paths


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)
