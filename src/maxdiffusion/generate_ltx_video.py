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

import numpy as np
from absl import app
from typing import Sequence, List, Optional, Union
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXVideoPipeline
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXMultiScalePipeline, ConditioningItem
import maxdiffusion.pipelines.ltx_video.crf_compressor as crf_compressor
from maxdiffusion import pyconfig, max_logging, max_utils
import torchvision.transforms.functional as TVF
import imageio
from datetime import datetime
import os
import time
from pathlib import Path
from PIL import Image
import torch
import jax


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

  # Calculate total padding needed
  pad_height = target_height - source_height
  pad_width = target_width - source_width

  # Calculate padding for each side
  pad_top = pad_height // 2
  pad_bottom = pad_height - pad_top  # Handles odd padding
  pad_left = pad_width // 2
  pad_right = pad_width - pad_left  # Handles odd padding
  padding = (pad_left, pad_right, pad_top, pad_bottom)
  return padding


def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
  """Load and process an image into a tensor.

  Args:
      image_input: Either a file path (str) or a PIL Image object
      target_height: Desired height of output tensor
      target_width: Desired width of output tensor
      just_crop: If True, only crop the image to the target size without resizing
  """
  if isinstance(image_input, str):
    image = Image.open(image_input).convert("RGB")
  elif isinstance(image_input, Image.Image):
    image = image_input
  else:
    raise ValueError("image_input must be either a file path or a PIL Image object")

  input_width, input_height = image.size
  aspect_ratio_target = target_width / target_height
  aspect_ratio_frame = input_width / input_height
  if aspect_ratio_frame > aspect_ratio_target:
    new_width = int(input_height * aspect_ratio_target)
    new_height = input_height
    x_start = (input_width - new_width) // 2
    y_start = 0
  else:
    new_width = input_width
    new_height = int(input_width / aspect_ratio_target)
    x_start = 0
    y_start = (input_height - new_height) // 2

  image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
  if not just_crop:
    image = image.resize((target_width, target_height))

  frame_tensor = TVF.to_tensor(image)  # PIL -> tensor (C, H, W), [0,1]
  frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
  frame_tensor_hwc = frame_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
  frame_tensor_hwc = crf_compressor.compress(frame_tensor_hwc)
  frame_tensor = frame_tensor_hwc.permute(2, 0, 1) * 255.0  # (H, W, C) -> (C, H, W)
  frame_tensor = (frame_tensor / 127.5) - 1.0
  # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
  return frame_tensor.unsqueeze(0).unsqueeze(2)


def prepare_conditioning(
    conditioning_media_paths: List[str],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    padding: tuple[int, int, int, int],
) -> Optional[List[ConditioningItem]]:
  """Prepare conditioning items based on input media paths and their parameters."""
  conditioning_items = []
  for path, strength, start_frame in zip(conditioning_media_paths, conditioning_strengths, conditioning_start_frames):
    num_input_frames = 1
    media_tensor = load_media_file(
        media_path=path,
        height=height,
        width=width,
        max_frames=num_input_frames,
        padding=padding,
        just_crop=True,
    )
    conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))
  return conditioning_items


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
  # Remove non-letters and convert to lowercase
  clean_text = "".join(char.lower() for char in text if char.isalpha() or char.isspace())

  # Split into words
  words = clean_text.split()

  # Build result string keeping track of length
  result = []
  current_length = 0

  for word in words:
    # Add word length plus 1 for underscore (except for first word)
    new_length = current_length + len(word)

    if new_length <= max_len:
      result.append(word)
      current_length += len(word)
    else:
      break

  return "-".join(result)


def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:
  media_tensor = load_image_to_tensor_with_resize_and_crop(media_path, height, width, just_crop=just_crop)
  media_tensor = torch.nn.functional.pad(media_tensor, padding)
  return media_tensor


def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
  base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
  for i in range(index_range):
    filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
    if not os.path.exists(filename):
      return filename
  raise FileExistsError(f"Could not find a unique filename after {index_range} attempts.")


def run(config):
  height_padded = ((config.height - 1) // 32 + 1) * 32
  width_padded = ((config.width - 1) // 32 + 1) * 32
  num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1
  padding = calculate_padding(config.height, config.width, height_padded, width_padded)
  prompt_enhancement_words_threshold = config.prompt_enhancement_words_threshold
  prompt_word_count = len(config.prompt.split())
  enhance_prompt = prompt_enhancement_words_threshold > 0 and prompt_word_count < prompt_enhancement_words_threshold

  pipeline = LTXVideoPipeline.from_pretrained(config, enhance_prompt=enhance_prompt)
  if config.pipeline_type == "multi-scale":
    pipeline = LTXMultiScalePipeline(pipeline)
  conditioning_media_paths = config.conditioning_media_paths if isinstance(config.conditioning_media_paths, List) else None
  conditioning_start_frames = config.conditioning_start_frames
  conditioning_strengths = None
  if conditioning_media_paths:
    if not conditioning_strengths:
      conditioning_strengths = [1.0] * len(conditioning_media_paths)
  conditioning_items = (
      prepare_conditioning(
          conditioning_media_paths=conditioning_media_paths,
          conditioning_strengths=conditioning_strengths,
          conditioning_start_frames=conditioning_start_frames,
          height=config.height,
          width=config.width,
          padding=padding,
      )
      if conditioning_media_paths
      else None
  )

  pipeline_args = {
      "height": height_padded,
      "width": width_padded,
      "num_frames": num_frames_padded,
      "is_video": True,
      "output_type": "pt",
      "config": config,
      "enhance_prompt": enhance_prompt,
      "conditioning_items": conditioning_items,
      "seed": config.seed,
  }


  # Warm-up call
  s0 = time.perf_counter()
  images = pipeline(**pipeline_args)
  max_logging.log(f"Warmup time: {time.perf_counter() - s0:.1f}s.")

  # Normal call
  s0 = time.perf_counter()
  images = pipeline(**pipeline_args)
  max_logging.log(f"Generation time: {time.perf_counter() - s0:.1f}s.")

  # Profiled call
  if config.enable_profiler:
    profile_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    profiler_output_path = f"gs://hjajoo-ai-ninja-bucket/ltx-video/profiler_traces/{profile_timestamp}"
    jax.profiler.start_trace(profiler_output_path)
    max_logging.log(f"JAX profiler started. Traces will be saved to: {profiler_output_path}")
    s0 = time.perf_counter()
    images = pipeline(**pipeline_args)
    jax.profiler.stop_trace()
    max_logging.log(f"JAX profiler stopped.")
    max_logging.log(f"Generation time with profiler: {time.perf_counter() - s0:.1f}s.")

  (pad_left, pad_right, pad_top, pad_bottom) = padding
  pad_bottom = -pad_bottom
  pad_right = -pad_right
  if pad_bottom == 0:
    pad_bottom = images.shape[3]
  if pad_right == 0:
    pad_right = images.shape[4]
  images = images[:, :, : config.num_frames, pad_top:pad_bottom, pad_left:pad_right]
  output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
  output_dir.mkdir(parents=True, exist_ok=True)

  for i in range(images.shape[0]):
    # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
    video_np = images[i].permute(1, 2, 3, 0).detach().float().numpy()
    # Unnormalizing images to [0, 255] range
    video_np = (video_np * 255).astype(np.uint8)
    fps = config.frame_rate
    height, width = video_np.shape[1:3]
    # In case a single image is generated
    if video_np.shape[0] == 1:
      output_filename = get_unique_filename(
          f"image_output_{i}",
          ".png",
          prompt=config.prompt,
          resolution=(height, width, config.num_frames),
          dir=output_dir,
      )
      imageio.imwrite(output_filename, video_np[0])
    else:
      output_filename = get_unique_filename(
          f"video_output_{i}",
          ".mp4",
          prompt=config.prompt,
          resolution=(height, width, config.num_frames),
          dir=output_dir,
      )
      # Write video
      with imageio.get_writer(output_filename, fps=fps) as video:
        for frame in video_np:
          video.append_data(frame)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
