# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Wan Animate inference entrypoint."""

import os
import time

from absl import app
import flax
import jax
import numpy as np
from PIL import Image

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.pipelines.wan.wan_pipeline_animate import WanAnimatePipeline
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image, load_video

jax.config.update("jax_use_shardy_partitioner", True)


def _get_animate_inference_settings(config):
  """Resolve animate-specific inference settings with upstream defaults."""
  return {
      "segment_frame_length": getattr(config, "segment_frame_length", 77),
      "prev_segment_conditioning_frames": getattr(config, "prev_segment_conditioning_frames", 5),
      "motion_encode_batch_size": getattr(config, "motion_encode_batch_size", None),
      "guidance_scale": getattr(config, "animate_guidance_scale", 1.0),
  }


def _frame_summary(name, frames):
  """Return a compact frame-count/size summary for logging."""
  if not frames:
    return f"{name}_frames=0"
  return f"{name}_frames={len(frames)}, {name}_frame_size={getattr(frames[0], 'size', None)}"


def run(config):
  """Run Wan Animate inference and write the generated videos to disk."""
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

  load_start = time.perf_counter()
  pipeline = WanAnimatePipeline.from_pretrained(config)
  load_time = time.perf_counter() - load_start
  max_logging.log(f"load_time: {load_time:.1f}s")

  # Setup inputs
  reference_image_path = getattr(config, "reference_image_path", "")
  if reference_image_path:
    image = load_image(reference_image_path)
    reference_image_source = reference_image_path
  else:
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    image = load_image(image_url)
    reference_image_source = image_url

  mode = getattr(config, "mode", "animate")
  pose_video_path = getattr(config, "pose_video_path", "")
  face_video_path = getattr(config, "face_video_path", "")
  background_video_path = getattr(config, "background_video_path", "")
  mask_video_path = getattr(config, "mask_video_path", "")

  num_frames = config.num_frames
  height = config.height
  width = config.width

  # face_video needs to match motion_encoder_size (probably 224x224 or 256x256)
  motion_encoder_size = pipeline.transformer.config.motion_encoder_size

  if pose_video_path and face_video_path:
    max_logging.log(f"Loading preprocessed videos from disk. pose_video={pose_video_path}, face_video={face_video_path}")
    pose_video = load_video(pose_video_path)
    face_video = load_video(face_video_path)
    num_frames = min(num_frames, len(pose_video), len(face_video))
    if num_frames == 0:
      raise ValueError("Loaded empty pose/face video. Check preprocessing outputs.")
    pose_video = pose_video[:num_frames]
    face_video = face_video[:num_frames]
  else:
    # Fallback path used for quick smoke tests only.
    max_logging.log(
        "No pose/face video paths provided; generating dummy videos for a smoke test only. "
        "For real outputs provide preprocessed pose_video_path and face_video_path."
    )
    pose_video = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)) for _ in range(num_frames)]
    face_video = [
        Image.fromarray(np.zeros((motion_encoder_size, motion_encoder_size, 3), dtype=np.uint8)) for _ in range(num_frames)
    ]

  background_video = None
  mask_video = None
  if mode == "replace":
    if not background_video_path or not mask_video_path:
      raise ValueError("Replace mode requires both `background_video_path` and `mask_video_path`.")
    background_video = load_video(background_video_path)[:num_frames]
    mask_video = load_video(mask_video_path)[:num_frames]

  max_logging.log(
      "Wan animate inputs: "
      f"reference_image={reference_image_source}, "
      f"image_size={getattr(image, 'size', None)}, "
      f"pose_video_path={pose_video_path or '<dummy>'}, "
      f"face_video_path={face_video_path or '<dummy>'}, "
      f"{_frame_summary('pose', pose_video)}, "
      f"{_frame_summary('face', face_video)}"
  )
  if mode == "replace":
    max_logging.log(
        "Wan replace inputs: "
        f"background_video_path={background_video_path}, "
        f"mask_video_path={mask_video_path}, "
        f"{_frame_summary('background', background_video)}, "
        f"{_frame_summary('mask', mask_video)}"
    )

  animate_settings = _get_animate_inference_settings(config)
  prompt = config.prompt
  negative_prompt = config.negative_prompt if animate_settings["guidance_scale"] > 1.0 else None

  max_logging.log(
      "Num steps: "
      f"{config.num_inference_steps}, height: {height}, width: {width}, frames: {num_frames}, "
      f"segment_frame_length: {animate_settings['segment_frame_length']}, "
      f"prev_segment_conditioning_frames: {animate_settings['prev_segment_conditioning_frames']}, "
      f"guidance_scale: {animate_settings['guidance_scale']}"
  )

  s0 = time.perf_counter()

  # First pass (compile)
  videos = pipeline(
      image=image,
      pose_video=pose_video,
      face_video=face_video,
      background_video=background_video,
      mask_video=mask_video,
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=height,
      width=width,
      segment_frame_length=animate_settings["segment_frame_length"],
      prev_segment_conditioning_frames=animate_settings["prev_segment_conditioning_frames"],
      motion_encode_batch_size=animate_settings["motion_encode_batch_size"],
      guidance_scale=animate_settings["guidance_scale"],
      num_inference_steps=config.num_inference_steps,
      mode=mode,
  )

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  s0 = time.perf_counter()
  videos = pipeline(
      image=image,
      pose_video=pose_video,
      face_video=face_video,
      background_video=background_video,
      mask_video=mask_video,
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=height,
      width=width,
      segment_frame_length=animate_settings["segment_frame_length"],
      prev_segment_conditioning_frames=animate_settings["prev_segment_conditioning_frames"],
      motion_encode_batch_size=animate_settings["motion_encode_batch_size"],
      guidance_scale=animate_settings["guidance_scale"],
      num_inference_steps=config.num_inference_steps,
      mode=mode,
  )

  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)

  filename_prefix = "animate_"
  os.makedirs(config.output_dir, exist_ok=True)
  for i, video in enumerate(videos):
    video_path = os.path.join(config.output_dir, f"{filename_prefix}wan_output_{config.seed}_{i}.mp4")
    export_to_video(video, video_path, fps=config.fps)
    max_logging.log(f"Saved video to {video_path}")

  if getattr(config, "enable_profiler", False):
    s0 = time.perf_counter()
    max_utils.activate_profiler(config)
    _ = pipeline(
        image=image,
        pose_video=pose_video,
        face_video=face_video,
        background_video=background_video,
        mask_video=mask_video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        segment_frame_length=animate_settings["segment_frame_length"],
        prev_segment_conditioning_frames=animate_settings["prev_segment_conditioning_frames"],
        motion_encode_batch_size=animate_settings["motion_encode_batch_size"],
        guidance_scale=animate_settings["guidance_scale"],
        num_inference_steps=config.num_inference_steps,
        mode=mode,
    )
    max_utils.deactivate_profiler(config)
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

  return videos


def main(argv) -> None:
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  run(pyconfig.config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
