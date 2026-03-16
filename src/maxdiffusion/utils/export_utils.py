"""
Copyright 2024 Google LLC

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

import io
import random
import struct
import tempfile
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import numpy as np

import PIL.Image
import PIL.ImageOps

from .import_utils import AV_IMPORT_ERROR, BACKENDS_MAPPING, is_av_available, is_imageio_available, is_opencv_available
from .logging import get_logger

if is_av_available():
  import av


global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager
def buffered_writer(raw_f):
  f = io.BufferedWriter(raw_f)
  yield f
  f.flush()


def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None) -> str:
  if output_gif_path is None:
    output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

  image[0].save(
      output_gif_path,
      save_all=True,
      append_images=image[1:],
      optimize=False,
      duration=100,
      loop=0,
  )
  return output_gif_path


def export_to_ply(mesh, output_ply_path: str = None):
  """
  Write a PLY file for a mesh.
  """
  if output_ply_path is None:
    output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

  coords = mesh.verts.detach().cpu().numpy()
  faces = mesh.faces.cpu().numpy()
  rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

  with buffered_writer(open(output_ply_path, "wb")) as f:
    f.write(b"ply\n")
    f.write(b"format binary_little_endian 1.0\n")
    f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    if rgb is not None:
      f.write(b"property uchar red\n")
      f.write(b"property uchar green\n")
      f.write(b"property uchar blue\n")
    if faces is not None:
      f.write(bytes(f"element face {len(faces)}\n", "ascii"))
      f.write(b"property list uchar int vertex_index\n")
    f.write(b"end_header\n")

    if rgb is not None:
      rgb = (rgb * 255.499).round().astype(int)
      vertices = [
          (*coord, *rgb)
          for coord, rgb in zip(
              coords.tolist(),
              rgb.tolist(),
          )
      ]
      format = struct.Struct("<3f3B")
      for item in vertices:
        f.write(format.pack(*item))
    else:
      format = struct.Struct("<3f")
      for vertex in coords.tolist():
        f.write(format.pack(*vertex))

    if faces is not None:
      format = struct.Struct("<B3I")
      for tri in faces.tolist():
        f.write(format.pack(len(tri), *tri))

  return output_ply_path


def export_to_obj(mesh, output_obj_path: str = None):
  if output_obj_path is None:
    output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

  verts = mesh.verts.detach().cpu().numpy()
  faces = mesh.faces.cpu().numpy()

  vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
  vertices = ["{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())]

  faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

  combined_data = ["v " + vertex for vertex in vertices] + faces

  with open(output_obj_path, "w") as f:
    f.writelines("\n".join(combined_data))


def _legacy_export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
):
  if is_opencv_available():
    import cv2
  else:
    raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
  if output_video_path is None:
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

  if isinstance(video_frames[0], np.ndarray):
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

  elif isinstance(video_frames[0], PIL.Image.Image):
    video_frames = [np.array(frame) for frame in video_frames]

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  h, w, c = video_frames[0].shape
  video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
  for i in range(len(video_frames)):
    img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
    video_writer.write(img)

  return output_video_path


def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> str:
  """
  quality:
      Video output quality. Default is 5. Uses variable bit rate. Highest quality is 10, lowest is 0. Set to None to
      prevent variable bitrate flags to FFMPEG so you can manually specify them using output_params instead.
      Specifying a fixed bitrate using `bitrate` disables this parameter.

  bitrate:
      Set a constant bitrate for the video encoding. Default is None causing `quality` parameter to be used instead.
      Better quality videos with smaller file sizes will result from using the `quality` variable bitrate parameter
      rather than specifying a fixed bitrate with this parameter.

  macro_block_size:
      Size constraint for video. Width and height, must be divisible by this number. If not divisible by this number
      imageio will tell ffmpeg to scale the image up to the next closest size divisible by this number. Most codecs
      are compatible with a macroblock size of 16 (default), some can go smaller (4, 8). To disable this automatic
      feature set it to None or 1, however be warned many players can't decode videos that are odd in size and some
      codecs will produce poor results or fail. See https://en.wikipedia.org/wiki/Macroblock.
  """
  # TODO: Dhruv. Remove by Diffusers release 0.33.0
  # Added to prevent breaking existing code
  if not is_imageio_available():
    logger.warning(
        (
            "It is recommended to use `export_to_video` with `imageio` and `imageio-ffmpeg` as a backend. \n"
            "These libraries are not present in your environment. Attempting to use legacy OpenCV backend to export video. \n"
            "Support for the OpenCV backend will be deprecated in a future Diffusers version"
        )
    )
    return _legacy_export_to_video(video_frames, output_video_path, fps)

  if is_imageio_available():
    import imageio
  else:
    raise ImportError(BACKENDS_MAPPING["imageio"][1].format("export_to_video"))

  try:
    imageio.plugins.ffmpeg.get_exe()
  except AttributeError:
    raise AttributeError(
        (
            "Found an existing imageio backend in your environment. Attempting to export video with imageio. \n"
            "Unable to find a compatible ffmpeg installation in your environment to use with imageio. Please install via `pip install imageio-ffmpeg"
        )
    )

  if output_video_path is None:
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

  if isinstance(video_frames[0], np.ndarray):
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

  elif isinstance(video_frames[0], PIL.Image.Image):
    video_frames = [np.array(frame) for frame in video_frames]

  with imageio.get_writer(
      output_video_path, fps=fps, quality=quality, bitrate=bitrate, macro_block_size=macro_block_size
  ) as writer:
    for frame in video_frames:
      writer.append_data(frame)

  return output_video_path


def _prepare_audio_stream(container, audio_sample_rate: int):
  """
  Prepare the audio stream for writing.
  """
  from fractions import Fraction

  audio_stream = container.add_stream("aac", rate=audio_sample_rate)
  audio_stream.codec_context.sample_rate = audio_sample_rate
  audio_stream.codec_context.layout = "stereo"
  audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
  return audio_stream


def _resample_audio(container, audio_stream, frame_in) -> None:
  cc = audio_stream.codec_context

  target_format = cc.format or "fltp"
  target_layout = cc.layout or "stereo"
  target_rate = cc.sample_rate or frame_in.sample_rate

  audio_resampler = av.audio.resampler.AudioResampler(
      format=target_format,
      layout=target_layout,
      rate=target_rate,
  )

  audio_next_pts = 0
  for rframe in audio_resampler.resample(frame_in):
    if rframe.pts is None:
      rframe.pts = audio_next_pts
    audio_next_pts += rframe.samples
    rframe.sample_rate = frame_in.sample_rate
    container.mux(audio_stream.encode(rframe))

  # flush audio encoder
  for packet in audio_stream.encode():
    container.mux(packet)


def _write_audio(
    container,
    audio_stream,
    samples: Any,
    audio_sample_rate: int,
    target_format: str = "s16",
) -> None:
  import numpy as np

  samples = np.asarray(samples)

  if samples.ndim == 1:
    samples = samples[:, None]

  # The Vocoder naturally outputs (Channels=2, Time)
  if samples.shape[0] == 2 and samples.shape[1] != 2:
    samples = samples.T  # Now (Time, 2)

  if samples.shape[1] != 2:
    raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

  if target_format == "s16":
    if samples.dtype != np.int16:
      samples = np.clip(samples, -1.0, 1.0)
      samples = (samples * 32767.0).astype(np.int16)
  elif target_format == "s32":
    if samples.dtype != np.int32:
      samples = np.clip(samples, -1.0, 1.0)
      samples = (samples * 2147483647.0).astype(np.int32)
  elif target_format in ["flt", "dbl", "fltp", "dblp"]:
    target_dtype = np.float32 if "flt" in target_format else np.float64
    if samples.dtype != target_dtype:
      samples = samples.astype(target_dtype)
  else:
    # Fallback to clip and scaling for other int formats if they were added, but raise for now
    raise ValueError(f"Unsupported target_format for converting numpy array: {target_format}")

  samples_np = np.ascontiguousarray(samples).reshape(1, -1)

  frame_in = av.AudioFrame.from_ndarray(
      samples_np,
      format=target_format,
      layout="stereo",
  )
  frame_in.sample_rate = audio_sample_rate

  _resample_audio(container, audio_stream, frame_in)


def export_to_video_with_audio(
    video: Any, fps: int, audio: Optional[Any], audio_sample_rate: Optional[int], output_path: str, audio_format: str = "s16"
) -> None:
  """
  Encodes video (and optionally audio) to a file using PyAV.
  Args:
      video: Video array-like [F, H, W, C] (frames, height, width, channels)
      fps: Frames per second
      audio: Audio array-like [C, L] or [L, C]
      audio_sample_rate: Audio sample rate
      output_path: Output file path
  """
  if not is_av_available():
    raise ImportError(AV_IMPORT_ERROR.format("export_to_video_with_audio"))

  video_np = np.asarray(video)

  if video_np.ndim == 4:
    # [F, H, W, C]
    _, height, width, _ = video_np.shape
  elif video_np.ndim == 5:
    # [B, F, H, W, C] -> take the first video in the batch
    video_np = video_np[0]
    _, height, width, _ = video_np.shape
  else:
    raise ValueError(f"export_to_video_with_audio expects a 4D or 5D video tensor, got {video_np.ndim}D")

  container = av.open(output_path, mode="w")
  stream = container.add_stream("libx264", rate=int(fps))
  stream.width = width
  stream.height = height
  stream.pix_fmt = "yuv420p"

  if audio is not None:
    if audio_sample_rate is None:
      raise ValueError("audio_sample_rate is required when audio is provided")

    audio_stream = _prepare_audio_stream(container, audio_sample_rate)

  for frame_array in video_np:
    # frame_array is [H, W, C]
    frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
    for packet in stream.encode(frame):
      container.mux(packet)

  # Flush encoder
  for packet in stream.encode():
    container.mux(packet)

  if audio is not None:
    _write_audio(container, audio_stream, audio, audio_sample_rate, target_format=audio_format)

  container.close()
