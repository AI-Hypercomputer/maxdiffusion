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
import subprocess
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image
from google.cloud import storage
import flax
from maxdiffusion.common_types import WAN2_1, WAN2_2
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2


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

  except Exception as e:
    max_logging.log(f"An error occurred: {e}")


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
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}, video: {filename_prefix}"
  )

  videos = call_pipeline(config, pipeline, prompt, negative_prompt)

  max_logging.log(f"video {filename_prefix}, compile time: {(time.perf_counter() - s0)}")
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
      # Delete local files to avoid storing too manys videos
      delete_file(f"./{video_path}")
  return


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  model_key = config.model_name
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
          config, pipeline_cls, pretrained_state_sources, pretrained_config_transformer_attr
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

  s0 = time.perf_counter()

  # Disable profiler for the first two runs to avoid duplicate uploads
  original_enable_profiler = config.enable_profiler if "enable_profiler" in config.get_keys() else False
  config.get_keys()["enable_profiler"] = False

  # Using global_batch_size_to_train_on so not to create more config variables
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )
  # Warmup with 2 denoising steps instead of a full run: step 0 runs the
  # high-noise transformer and step 1 crosses the boundary to the low-noise
  # one (WAN 2.2), so every executable of the full run (both transformers,
  # text encoder, VAE decode) gets compiled at a fraction of the cost. The
  # step count only changes the Python loop trip count, not traced shapes.
  warmup_steps = min(2, config.num_inference_steps)
  max_logging.log(f"Compile warmup: {warmup_steps} denoising steps")
  videos = call_pipeline(config, pipeline, prompt, negative_prompt, num_inference_steps=warmup_steps)
  if isinstance(videos, tuple):
    videos, warmup_trace = videos
    max_logging.log("Warmup breakdown: " + ", ".join(f"{stage}={seconds:.1f}s" for stage, seconds in warmup_trace.items()))

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  s0 = time.perf_counter()
  outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
  if isinstance(outputs, tuple):
    videos, trace = outputs
  else:
    videos = outputs
    trace = {}
  generation_time = time.perf_counter() - s0
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
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
  if max_utils.profiler_enabled(config):
    # Injecting user requested XLA tracing flags
    xla_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = "--xla_enable_mxu_trace=true --xla_jf_dump_llo_html=true --xla_tpu_enable_llo_profiling=true"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {new_flags}"
    max_logging.log(f"Injected XLA_FLAGS for profiling: {new_flags}")

    videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(videos, tuple):
      videos = videos[0]
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

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
  with transformer_engine_context():
    app.run(main)
