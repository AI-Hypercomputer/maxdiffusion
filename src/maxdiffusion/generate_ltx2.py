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
from maxdiffusion.checkpointing.ltx2_checkpointer import LTX2Checkpointer
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from google.cloud import storage
import flax
from maxdiffusion.pipelines.ltx2.ltx2_pipeline_utils import encode_video

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


def call_pipeline(config, pipeline, prompt, negative_prompt):
  # Set default generation arguments
  generator = jax.random.key(config.seed) if hasattr(config, "seed") else jax.random.key(0)
  
  negative_prompt = negative_prompt if hasattr(config, "negative_prompt") else None
  do_classifier_free_guidance = getattr(config, "do_classifier_free_guidance", True)
  guidance_scale = getattr(config, "guidance_scale", 3.0) if do_classifier_free_guidance else 1.0

  out = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=guidance_scale,
      generator=generator,
      frame_rate=getattr(config, "fps", 24.0),
      decode_timestep=getattr(config, "decode_timestep", 0.0),
      decode_noise_scale=getattr(config, "decode_noise_scale", None),
  )
  return out


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  if pipeline is None:
    checkpoint_loader = LTX2Checkpointer(config=config)
    pipeline, _, _ = checkpoint_loader.load_checkpoint()

  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on to map prompts
  prompt = getattr(config, "prompt", "A cat playing piano")
  prompt = [prompt] * getattr(config, "global_batch_size_to_train_on", 1)
  
  negative_prompt = getattr(config, "negative_prompt", "")
  negative_prompt = [negative_prompt] * getattr(config, "global_batch_size_to_train_on", 1)

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )
  
  out = call_pipeline(config, pipeline, prompt, negative_prompt)
  # out should have .frames and .audio
  videos = out.frames if hasattr(out, "frames") else out[0]
  audios = out.audio if hasattr(out, "audio") else None

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {getattr(config, 'model_name', 'ltx-video')}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {getattr(config, 'model_type', 'T2V')}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)
  
  saved_video_path = []
  audio_sample_rate = getattr(pipeline.audio_vae.config, "sample_rate", 16000) if hasattr(pipeline, "audio_vae") else 16000
  fps = getattr(config, "fps", 24)

  # Export videos
  for i in range(len(videos)):
    video_path = f"{filename_prefix}ltx2_output_{getattr(config, 'seed', 0)}_{i}.mp4"
    audio_i = audios[i] if audios is not None else None
    
    encode_video(
        video=videos[i], 
        fps=fps, 
        audio=audio_i, 
        audio_sample_rate=audio_sample_rate, 
        output_path=video_path
    )
    
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  s0 = time.perf_counter()
  call_pipeline(config, pipeline, prompt, negative_prompt)
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
  if getattr(config, "enable_profiler", False):
    max_utils.activate_profiler(config)
    call_pipeline(config, pipeline, prompt, negative_prompt)
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
