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
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
import importlib
from maxdiffusion.utils import export_to_video
from google.cloud import storage
import flax


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
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  # tf.config.set_visible_devices([], "GPU")
if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
  max_logging.log("Enabling unsafe RNG bit generator for TPU SPMD.")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )

def get_pipeline(model_name: str):
  if model_name == "wan2.1":
    return importlib.import_module("maxdiffusion.pipelines.wan.wan_pipeline")
  elif model_name == "wan2.2":
    return importlib.import_module("maxdiffusion.pipelines.wan.wan_pipeline2_2")
  else:
    raise ValueError(f"Unsupported model_name in config: {model_name}")

def get_checkpointer(model_name: str):
  if model_name == "wan2.1":
    return importlib.import_module("maxdiffusion.checkpointing.wan_checkpointer")
  elif model_name == "wan2.2":
    return importlib.import_module("maxdiffusion.checkpointing.wan_checkpointer2_2")
  else:
    raise ValueError(f"Unsupported model_name in config: {model_name}")

def call_pipeline(config, pipeline, prompt, negative_prompt):
  model_key = config.model_name
  if model_key == "wan2.1":
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
    )
  elif model_key == "wan2.2":
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale_low=config.guidance_scale_low,
        guidance_scale_high=config.guidance_scale_high,
        boundary=config.boundary_timestep,
    )
  else:
    raise ValueError(f"Unsupported model_name in config: {model_key}")


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


def run(config, pipeline=None, filename_prefix=""):
  print("seed: ", config.seed)
  model_key = config.model_name
  config.tensorboard_dir = os.path.join(config.output_dir, "tensorboard")
  # Initialize TensorBoard writer
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

  checkpointer_lib = get_checkpointer(model_key)
  WanCheckpointer = checkpointer_lib.WanCheckpointer

  checkpoint_loader = WanCheckpointer(config, "WAN_CHECKPOINT")
  pipeline, _, _ = checkpoint_loader.load_checkpoint()

  if pipeline is None:
    pipeline_lib = get_pipeline(model_key)
    WanPipeline = pipeline_lib.WanPipeline
    pipeline = WanPipeline.from_pretrained(config)
  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on so not to create more config variables
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )

  videos = call_pipeline(config, pipeline, prompt, negative_prompt)

  compile_time = time.perf_counter() - s0
  print("compile_time: ", compile_time)
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  s0 = time.perf_counter()
  videos = call_pipeline(config, pipeline, prompt, negative_prompt)
  generation_time = time.perf_counter() - s0
  print("generation_time: ", generation_time)
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_devices = jax.device_count()
    num_videos = num_devices * config.per_device_batch_size
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      print(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")


  s0 = time.perf_counter()
  if config.enable_profiler:
    max_utils.activate_profiler(config)
    videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    max_utils.deactivate_profiler(config)
    generation_time_with_profiler = time.perf_counter() - s0
    print("generation_time_with_profiler: ", generation_time_with_profiler)
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  flax.config.update("flax_always_shard_variable", False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
