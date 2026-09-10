# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Wan Animate inference entrypoint."""

import os
import time

from absl import app
import flax
import jax

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
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
  pipeline = WanCheckpointer.load_pretrained_pipeline_or_diffusers(
      config, WanAnimatePipeline, (("wan_state", "transformer"),), "transformer"
  )
  load_time = time.perf_counter() - load_start
  max_logging.log(f"load_time: {load_time:.1f}s")

  # Setup inputs
  reference_image_path = getattr(config, "reference_image_path", "")
  if reference_image_path:
    image = load_image(reference_image_path)
    reference_image_source = reference_image_path
  else:
    raise ValueError("Provide `reference_image_path`.")

  mode = getattr(config, "mode", "animate")
  pose_video_path = getattr(config, "pose_video_path", "")
  face_video_path = getattr(config, "face_video_path", "")
  background_video_path = getattr(config, "background_video_path", "")
  mask_video_path = getattr(config, "mask_video_path", "")

  num_frames = config.num_frames
  height = config.height
  width = config.width

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
    raise ValueError("Provide both `pose_video_path` and `face_video_path`.")

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
      f"pose_video_path={pose_video_path}, "
      f"face_video_path={face_video_path}, "
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
  prompt_file = getattr(config, "prompt_file", "")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=config.prompt)
  batch_size = max(1, getattr(config, "global_batch_size_to_train_on", 1))
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  warmup_prompt = [prompts[0]] * batch_size
  warmup_negative_prompt = [config.negative_prompt] * batch_size if animate_settings["guidance_scale"] > 1.0 else None

  max_logging.log(
      "Num steps: "
      f"{config.num_inference_steps}, height: {height}, width: {width}, frames: {num_frames}, "
      f"total prompts: {len(prompts)}, "
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
      prompt=warmup_prompt,
      negative_prompt=warmup_negative_prompt,
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
  filename_prefix = "animate_"
  gcs_output_path = max_utils.get_gcs_output_path(config)
  if not gcs_output_path:
    os.makedirs(config.output_dir, exist_ok=True)
  saved_video_paths = []

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [config.negative_prompt] * batch_size if animate_settings["guidance_scale"] > 1.0 else None
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
    for i, video in enumerate(videos):
      video_path = (
          f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
          if gcs_output_path
          else os.path.join(config.output_dir, f"{filename_prefix}wan_output_{config.seed}_{i}.mp4")
      )
      export_to_video(video, video_path, fps=config.fps)
      max_logging.log(f"Saved video to {video_path}")
      saved_video_paths.append(video_path)
      if gcs_output_path:
        max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
  else:
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [config.negative_prompt] * batch_size if animate_settings["guidance_scale"] > 1.0 else None
      videos = pipeline(
          image=image,
          pose_video=pose_video,
          face_video=face_video,
          background_video=background_video,
          mask_video=mask_video,
          prompt=padded_chunk,
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
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = (
            f"{filename_prefix}wan_output_{config.seed}_{prompt_idx}.mp4"
            if gcs_output_path
            else os.path.join(config.output_dir, f"{filename_prefix}wan_output_{config.seed}_{prompt_idx}.mp4")
        )
        export_to_video(videos[j], video_path, fps=config.fps)
        max_logging.log(f"Saved video to {video_path}")
        saved_video_paths.append(video_path)
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")

  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_videos = len(saved_video_paths)
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      max_logging.log(f"generation time per video: {generation_time_per_video}")

  if max_utils.profiler_enabled(config):
    s0 = time.perf_counter()
    with max_utils.Profiler(config, session_name="wan_animate_profile"):
      _ = pipeline(
          image=image,
          pose_video=pose_video,
          face_video=face_video,
          background_video=background_video,
          mask_video=mask_video,
          prompt=warmup_prompt,
          negative_prompt=warmup_negative_prompt,
          height=height,
          width=width,
          segment_frame_length=animate_settings["segment_frame_length"],
          prev_segment_conditioning_frames=animate_settings["prev_segment_conditioning_frames"],
          motion_encode_batch_size=animate_settings["motion_encode_batch_size"],
          guidance_scale=animate_settings["guidance_scale"],
          num_inference_steps=config.num_inference_steps,
          mode=mode,
      )
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
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
