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
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from absl import flags
from maxdiffusion.utils import export_to_video
from google.cloud import storage
import flax

_MODEL_NAME = flags.DEFINE_enum(
    "model_name",
    default="wan2.1",
    enum_values=["wan2.1", "wan2.2"],
    help="The model version to run (wan2.1 or wan2.2). This determines the base config file.",
)

CONFIG_BASE_DIR = "src/maxdiffusion/configs"
MODEL_CONFIG_MAP = {
    "wan2.1": "base_wan_14b.yml",
    "wan2.2": "base_wan_27b.yml",
}

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


jax.config.update("jax_use_shardy_partitioner", True)


def inference_generate_video(config, pipeline, filename_prefix=""):
  s0 = time.perf_counter()
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}, video: {filename_prefix}"
  )

  videos = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=guidance_scale,
      guidance_scale_low=guidance_scale_low,
      guidance_scale_high=guidance_scale_high,
      boundary=boundary,
  )

  max_logging.log(f"video {filename_prefix}, compile time: {(time.perf_counter() - s0)}")
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
      # Delete local files to avoid storing too manys videos
      delete_file(f"./{video_path}")
  return


def run(config, pipeline=None, filename_prefix=""):
  print("seed: ", config.seed)
  from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer

  checkpoint_loader = WanCheckpointer(config, "WAN_CHECKPOINT")
  pipeline = checkpoint_loader.load_checkpoint()
  if pipeline is None:
    pipeline = WanPipeline.from_pretrained(config)
  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on so not to create more config variables
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on
  guidance_scale = config.guidance_scale if 'guidance_scale' in config.__dict__ else 5
  guidance_scale_low = config.guidance_scale_low if 'guidance_scale_low' in config.__dict__ else 3
  guidance_scale_high = config.guidance_scale_high if 'guidance_scale_high' in config.__dict__ else 4
  boundary = config.boundary_timestep if 'boundary_timestep' in config.__dict__ else 875

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )

  videos = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=guidance_scale,
      guidance_scale_low=guidance_scale_low,
      guidance_scale_high=guidance_scale_high,
      boundary=boundary,
  )

  print("compile time: ", (time.perf_counter() - s0))
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  s0 = time.perf_counter()
  videos = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=guidance_scale,
      guidance_scale_low=guidance_scale_low,
      guidance_scale_high=guidance_scale_high,
      boundary=boundary,
  )
  print("generation time: ", (time.perf_counter() - s0))

  s0 = time.perf_counter()
  if config.enable_profiler:
    max_utils.activate_profiler(config)
    videos = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=boundary,
    )
    max_utils.deactivate_profiler(config)
    print("generation time: ", (time.perf_counter() - s0))
  return saved_video_path


def main(argv: Sequence[str]) -> None:
  # Get the model name from the flag
  model_key = _MODEL_NAME.value
  config_filename = MODEL_CONFIG_MAP[model_key]
  selected_yaml_path = os.path.join(CONFIG_BASE_DIR, config_filename)

  max_logging.log(f"Using model: {model_key}, loading base config: {selected_yaml_path}")

  # Construct argv for pyconfig.initialize
  # argv[0] is the program name.
  # Insert the selected YAML path at index 1.
  # The rest of argv (argv[1:]) are the overrides.
  argv_for_pyconfig = list(argv[:1]) + [selected_yaml_path] + list(argv[1:])

  pyconfig.initialize(argv_for_pyconfig)
  flax.config.update("flax_always_shard_variable", False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
