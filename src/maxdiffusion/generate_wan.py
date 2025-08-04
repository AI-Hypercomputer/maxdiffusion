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

import os
from typing import Sequence
import jax
import time
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from maxdiffusion.utils import export_to_video

jax.config.update("jax_use_shardy_partitioner", True)


def run(config, pipeline=None, filename_prefix=""):
  print("seed: ", config.seed)
  if pipeline is None:
    pipeline = WanPipeline.from_pretrained(config)
  s0 = time.perf_counter()

  # If global_batch_size % jax.device_count is not 0, use FSDP sharding.
  global_batch_size = config.global_batch_size
  if global_batch_size != 0:
    batch_multiplier = global_batch_size
  else:
    batch_multiplier = jax.device_count() * config.per_device_batch_size

  prompt = [config.prompt] * batch_multiplier
  negative_prompt = [config.negative_prompt] * batch_multiplier

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
      guidance_scale=config.guidance_scale,
  )

  print("compile time: ", (time.perf_counter() - s0))
  saved_video_path = []
  for i, video in enumerate(videos):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    if os.path.exists(f"{config.base_output_dir}"):
      video_path = f"{config.base_output_dir}/{video_path}"
    export_to_video(video, video_path, fps=config.fps)
    saved_video_path.append(video_path)

  s0 = time.perf_counter()
  videos = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=config.guidance_scale,
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
        guidance_scale=config.guidance_scale,
    )
    max_utils.deactivate_profiler(config)
    print("generation time: ", (time.perf_counter() - s0))
  return saved_video_path


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
