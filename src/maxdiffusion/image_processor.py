# ruff: noqa
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import warnings
from typing import List, Optional, Union

import numpy as np
import torch

import PIL.Image
from PIL import Image

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.FloatTensor],
]


def is_valid_image(image) -> bool:
  r"""
  Checks if the input is a valid image.

  A valid image can be:
  - A `PIL.Image.Image`.
  - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

  Args:
      image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
          The image to validate. It can be a PIL image, a NumPy array, or a torch tensor.

  Returns:
      `bool`:
          `True` if the input is a valid image, `False` otherwise.
  """
  return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)


def is_valid_image_imagelist(images):
  r"""
  Checks if the input is a valid image or list of images.

  The input can be one of the following formats:
  - A 4D tensor or numpy array (batch of images).
  - A valid single image: `PIL.Image.Image`, 2D `np.ndarray` or `torch.Tensor` (grayscale image), 3D `np.ndarray` or
    `torch.Tensor`.
  - A list of valid images.

  Args:
      images (`Union[np.ndarray, torch.Tensor, PIL.Image.Image, List]`):
          The image(s) to check. Can be a batch of images (4D tensor/array), a single image, or a list of valid
          images.

  Returns:
      `bool`:
          `True` if the input is valid, `False` otherwise.
  """
  if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
    return True
  elif is_valid_image(images):
    return True
  elif isinstance(images, list):
    return all(is_valid_image(image) for image in images)
  return False


class VaeImageProcessor(ConfigMixin):
  """
  Image processor for VAE.

  Args:
      do_resize (`bool`, *optional*, defaults to `True`):
          Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
          `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
      vae_scale_factor (`int`, *optional*, defaults to `8`):
          VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
      resample (`str`, *optional*, defaults to `lanczos`):
          Resampling filter to use when resizing the image.
      do_normalize (`bool`, *optional*, defaults to `True`):
          Whether to normalize the image to [-1,1].
      do_binarize (`bool`, *optional*, defaults to `False`):
          Whether to binarize the image to 0/1.
      do_convert_rgb (`bool`, *optional*, defaults to be `False`):
          Whether to convert the images to RGB format.
      do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
          Whether to convert the images to grayscale format.
  """

  config_name = CONFIG_NAME

  @register_to_config
  def __init__(
      self,
      do_resize: bool = True,
      vae_scale_factor: int = 8,
      resample: str = "lanczos",
      do_normalize: bool = True,
      do_binarize: bool = False,
      do_convert_rgb: bool = False,
      do_convert_grayscale: bool = False,
  ):
    super().__init__()
    if do_convert_rgb and do_convert_grayscale:
      raise ValueError(
          "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
          " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
          " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
      )
      self.config.do_convert_rgb = False

  @staticmethod
  def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
      images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
      # special case for grayscale (single channel) images
      pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
      pil_images = [Image.fromarray(image) for image in images]

    return pil_images

  @staticmethod
  def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
      images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

  @staticmethod
  def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
      images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

  @staticmethod
  def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

  @staticmethod
  def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0

  @staticmethod
  def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)

  @staticmethod
  def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to RGB format.
    """
    image = image.convert("RGB")

    return image

  @staticmethod
  def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to grayscale format.
    """
    image = image.convert("L")

    return image

  def get_default_height_width(
      self,
      image: [PIL.Image.Image, np.ndarray, torch.Tensor],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ):
    """
    This function return the height and width that are downscaled to the next integer multiple of
    `vae_scale_factor`.

    Args:
        image(`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
            The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
            shape `[batch, height, width]` or `[batch, height, width, channel]` if it is a pytorch tensor, should
            have shape `[batch, channel, height, width]`.
        height (`int`, *optional*, defaults to `None`):
            The height in preprocessed image. If `None`, will use the height of `image` input.
        width (`int`, *optional*`, defaults to `None`):
            The width in preprocessed. If `None`, will use the width of the `image` input.
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

    width, height = (
        x - x % self.config.vae_scale_factor for x in (width, height)
    )  # resize to integer multiple of vae_scale_factor

    return height, width

  def resize(
      self,
      image: [PIL.Image.Image, np.ndarray, torch.Tensor],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> [PIL.Image.Image, np.ndarray, torch.Tensor]:
    """
    Resize image.
    """
    if isinstance(image, PIL.Image.Image):
      image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
    elif isinstance(image, torch.Tensor):
      image = torch.nn.functional.interpolate(
          image,
          size=(height, width),
      )
    elif isinstance(image, np.ndarray):
      image = self.numpy_to_pt(image)
      image = torch.nn.functional.interpolate(
          image,
          size=(height, width),
      )
      image = self.pt_to_numpy(image)
    return image

  def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:
    """
    create a mask
    """
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image

  def preprocess(
      self,
      image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> torch.Tensor:
    """
    Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
    """
    supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

    # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
    if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
      if isinstance(image, torch.Tensor):
        # if image is a pytorch tensor could have 2 possible shapes:
        #    1. batch x height x width: we should insert the channel dimension at position 1
        #    2. channnel x height x width: we should insert batch dimension at position 0,
        #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
        #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
        image = image.unsqueeze(1)
      else:
        # if it is a numpy array, it could have 2 possible shapes:
        #   1. batch x height x width: insert channel dimension on last position
        #   2. height x width x channel: insert batch dimension on first position
        if image.shape[-1] == 1:
          image = np.expand_dims(image, axis=0)
        else:
          image = np.expand_dims(image, axis=-1)

    if isinstance(image, supported_formats):
      image = [image]
    elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
      raise ValueError(
          f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
      )

    if isinstance(image[0], PIL.Image.Image):
      if self.config.do_convert_rgb:
        image = [self.convert_to_rgb(i) for i in image]
      elif self.config.do_convert_grayscale:
        image = [self.convert_to_grayscale(i) for i in image]
      if self.config.do_resize:
        height, width = self.get_default_height_width(image[0], height, width)
        image = [self.resize(i, height, width) for i in image]
      image = self.pil_to_numpy(image)  # to np
      image = self.numpy_to_pt(image)  # to pt

    elif isinstance(image[0], np.ndarray):
      image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

      image = self.numpy_to_pt(image)

      height, width = self.get_default_height_width(image, height, width)
      if self.config.do_resize:
        image = self.resize(image, height, width)

    elif isinstance(image[0], torch.Tensor):
      image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

      if self.config.do_convert_grayscale and image.ndim == 3:
        image = image.unsqueeze(1)

      channel = image.shape[1]
      # don't need any preprocess if the image is latents
      if channel == 4:
        return image

      height, width = self.get_default_height_width(image, height, width)
      if self.config.do_resize:
        image = self.resize(image, height, width)

    # expected range [0,1], normalize to [-1,1]
    do_normalize = self.config.do_normalize
    if image.min() < 0 and do_normalize:
      warnings.warn(
          "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
          f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
          FutureWarning,
      )
      do_normalize = False

    if do_normalize:
      image = self.normalize(image)

    if self.config.do_binarize:
      image = self.binarize(image)

    return image

  def postprocess(
      self,
      image: torch.FloatTensor,
      output_type: str = "pil",
      do_denormalize: Optional[List[bool]] = None,
  ):
    if not isinstance(image, torch.Tensor):
      raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")
    if output_type not in ["latent", "pt", "np", "pil"]:
      deprecation_message = (
          f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
          "`pil`, `np`, `pt`, `latent`"
      )
      deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
      output_type = "np"

    if output_type == "latent":
      return image

    if do_denormalize is None:
      do_denormalize = [self.config.do_normalize] * image.shape[0]

    image = torch.stack([self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])])

    if output_type == "pt":
      return image

    image = self.pt_to_numpy(image)

    if output_type == "np":
      return image

    if output_type == "pil":
      return self.numpy_to_pil(image)


class VaeImageProcessorLDM3D(VaeImageProcessor):
  """
  Image processor for VAE LDM3D.

  Args:
      do_resize (`bool`, *optional*, defaults to `True`):
          Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
      vae_scale_factor (`int`, *optional*, defaults to `8`):
          VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
      resample (`str`, *optional*, defaults to `lanczos`):
          Resampling filter to use when resizing the image.
      do_normalize (`bool`, *optional*, defaults to `True`):
          Whether to normalize the image to [-1,1].
  """

  config_name = CONFIG_NAME

  @register_to_config
  def __init__(
      self,
      do_resize: bool = True,
      vae_scale_factor: int = 8,
      resample: str = "lanczos",
      do_normalize: bool = True,
  ):
    super().__init__()

  @staticmethod
  def numpy_to_pil(images):
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
      images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
      # special case for grayscale (single channel) images
      pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
      pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

    return pil_images

  @staticmethod
  def rgblike_to_depthmap(image):
    """
    Args:
        image: RGB-like depth image

    Returns: depth map

    """
    return image[:, :, 1] * 2**8 + image[:, :, 2]

  def numpy_to_depth(self, images):
    """
    Convert a NumPy depth image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
      images = images[None, ...]
    images_depth = images[:, :, :, 3:]
    if images.shape[-1] == 6:
      images_depth = (images_depth * 255).round().astype("uint8")
      pil_images = [Image.fromarray(self.rgblike_to_depthmap(image_depth), mode="I;16") for image_depth in images_depth]
    elif images.shape[-1] == 4:
      images_depth = (images_depth * 65535.0).astype(np.uint16)
      pil_images = [Image.fromarray(image_depth, mode="I;16") for image_depth in images_depth]
    else:
      raise Exception("Not supported")

    return pil_images

  def postprocess(
      self,
      image: torch.FloatTensor,
      output_type: str = "pil",
      do_denormalize: Optional[List[bool]] = None,
  ):
    if not isinstance(image, torch.Tensor):
      raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")
    if output_type not in ["latent", "pt", "np", "pil"]:
      deprecation_message = (
          f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
          "`pil`, `np`, `pt`, `latent`"
      )
      deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
      output_type = "np"

    if do_denormalize is None:
      do_denormalize = [self.config.do_normalize] * image.shape[0]

    image = torch.stack([self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])])

    image = self.pt_to_numpy(image)

    if output_type == "np":
      if image.shape[-1] == 6:
        image_depth = np.stack([self.rgblike_to_depthmap(im[:, :, 3:]) for im in image], axis=0)
      else:
        image_depth = image[:, :, :, 3:]
      return image[:, :, :, :3], image_depth

    if output_type == "pil":
      return self.numpy_to_pil(image), self.numpy_to_depth(image)
    else:
      raise Exception(f"This type {output_type} is not supported")
