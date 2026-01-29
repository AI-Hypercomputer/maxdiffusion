"""
Copyright 2025 Google LLC

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

from typing import Sequence
import jax
import time
import os
import subprocess
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image
from google.cloud import storage
import flax
from maxdiffusion.common_types import WAN2_1, WAN2_2
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader
from maxdiffusion.inference.loader import InferenceLoader
from maxdiffusion.inference.runner import DiffusionRunner


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

  loaded_model = None
  if pipeline is None:
    max_logging.log("Initializing InferenceLoader...")
    loaded_model = InferenceLoader.load(config)
    pipeline = loaded_model["pipeline"]
  else:
    # If pipeline passed explicitly (e.g. from test), wrap it
    # But InferenceLoader logic assumes it creates it.
    # We construct a dummy loaded_model dict
    loaded_model = {
        "pipeline": pipeline,
        "mesh": getattr(config, "mesh", None) # Fallback
    }

  # If LoRA is specified, inject layers and load weights.
  # TODO: Move this into InferenceLoader._load_wan eventually
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
    # Update loaded model with modified pipeline
    loaded_model["pipeline"] = pipeline

  s0 = time.perf_counter()

  max_logging.log("Initializing DiffusionRunner...")
  runner = DiffusionRunner(loaded_model, config)

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )
  
  # Using global_batch_size_to_train_on logic is handled by Runner/Pipeline mostly now
  # But we can override args
  
  videos = runner.run()

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
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
      delete_file(f"./{video_path}")

  s0 = time.perf_counter()
  videos = runner.run()
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
  
  s0 = time.perf_counter()
  if config.enable_profiler:
    max_utils.activate_profiler(config)
    videos = runner.run()
    max_utils.deactivate_profiler(config)
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
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  app.run(main)