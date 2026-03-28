# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

"""Image processor for Wan-Animate reference (character) images.

Ported from:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/image_processor.py

Key difference vs. VaeImageProcessor: the reference image is letterboxed (padded
with black) instead of center-cropped, so the character is not accidentally cut off
when the image aspect ratio doesn't match the target output ratio.
"""

from typing import Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from ...image_processor import VaeImageProcessor
from ...utils import PIL_INTERPOLATION


class WanAnimateImageProcessor(VaeImageProcessor):
  r"""Image processor for the reference (character) image in Wan-Animate.

  Extends VaeImageProcessor with two changes:
    1. ``get_default_height_width`` snaps to multiples of
       ``vae_scale_factor * spatial_patch_size`` (instead of just
       ``vae_scale_factor``), which is what the Wan transformer expects.
    2. ``resize`` uses letterbox (pad with a solid fill colour) for PIL images
       rather than a stretch/crop, so the subject is not cut off.

  Args:
    do_resize: Whether to resize the image.
    vae_scale_factor: Spatial VAE downscale factor (typically 8).
    spatial_patch_size: Transformer patch size, e.g. ``(2, 2)``.  The
      target height/width will be multiples of
      ``vae_scale_factor * spatial_patch_size``.
    resample: PIL resampling filter name (e.g. ``"lanczos"``).
    do_normalize: Whether to normalise pixel values to ``[-1, 1]``.
    do_binarize: Whether to binarise the image to 0/1.
    do_convert_rgb: Whether to convert the image to RGB.
    do_convert_grayscale: Whether to convert the image to grayscale.
    fill_color: Solid colour used to fill letterbox bars.  Any value
      accepted by ``PIL.Image.new`` is valid; ``0`` gives black bars.
  """

  def __init__(
      self,
      do_resize: bool = True,
      vae_scale_factor: int = 8,
      spatial_patch_size: Tuple[int, int] = (2, 2),
      resample: str = "lanczos",
      do_normalize: bool = True,
      do_binarize: bool = False,
      do_convert_rgb: bool = False,
      do_convert_grayscale: bool = False,
      fill_color: Union[int, float, Tuple, str] = 0,
  ):
    super().__init__(
        do_resize=do_resize,
        vae_scale_factor=vae_scale_factor,
        resample=resample,
        do_normalize=do_normalize,
        do_binarize=do_binarize,
        do_convert_rgb=do_convert_rgb,
        do_convert_grayscale=do_convert_grayscale,
    )
    self.spatial_patch_size = spatial_patch_size
    self.fill_color = fill_color

  def _resize_and_fill(
      self,
      image: PIL.Image.Image,
      width: int,
      height: int,
  ) -> PIL.Image.Image:
    """Letterbox-resize *image* to (*width*, *height*).

    The image is scaled to the largest size that fits within the target
    dimensions while preserving its aspect ratio, then centred on a solid
    background of ``self.fill_color``.
    """
    ratio = width / height
    src_ratio = image.width / image.height

    src_w = width if ratio < src_ratio else image.width * height // image.height
    src_h = height if ratio >= src_ratio else image.height * width // image.width

    resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
    canvas = PIL.Image.new("RGB", (width, height), color=self.fill_color)
    canvas.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return canvas

  def get_default_height_width(
      self,
      image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> Tuple[int, int]:
    """Return the target (height, width) snapped to transformer-aligned multiples.

    The modulus is ``vae_scale_factor * spatial_patch_size`` on each axis,
    matching the Wan transformer's patchify requirements.
    """
    if height is None:
      if isinstance(image, PIL.Image.Image):
        height = image.height
      elif isinstance(image, torch.Tensor):
        height = image.shape[2]
      else:
        height = image.shape[1]

    if width is None:
      if isinstance(image, PIL.Image.Image):
        width = image.width
      elif isinstance(image, torch.Tensor):
        width = image.shape[3]
      else:
        width = image.shape[2]

    max_area = width * height
    aspect_ratio = height / width
    mod_h = self.config.vae_scale_factor * self.spatial_patch_size[0]
    mod_w = self.config.vae_scale_factor * self.spatial_patch_size[1]

    height = round(np.sqrt(max_area * aspect_ratio)) // mod_h * mod_h
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_w * mod_w

    return height, width

  def resize(
      self,
      image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
    """Resize image; PIL images are letterboxed, others use the parent's stretch."""
    if isinstance(image, PIL.Image.Image):
      return self._resize_and_fill(image, width, height)
    return super().resize(image, height=height, width=width)
