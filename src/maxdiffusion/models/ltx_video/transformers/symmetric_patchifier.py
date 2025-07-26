# Copyright 2025 Lightricks Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Lightricks/LTX-Video/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the Torch version available at:
# https://github.com/Lightricks/LTX-Video/tree/main
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from diffusers.configuration_utils import ConfigMixin
from einops import rearrange
from torch import Tensor


class Patchifier(ConfigMixin, ABC):

  def __init__(self, patch_size: int):
    super().__init__()
    self._patch_size = (1, patch_size, patch_size)

  @abstractmethod
  def patchify(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError("Patchify method not implemented")

  @abstractmethod
  def unpatchify(
      self,
      latents: Tensor,
      output_height: int,
      output_width: int,
      out_channels: int,
  ) -> Tuple[Tensor, Tensor]:
    pass

  @property
  def patch_size(self):
    return self._patch_size

  def get_latent_coords(self, latent_num_frames, latent_height, latent_width, batch_size, device):
    """
    Return a tensor of shape [batch_size, 3, num_patches] containing the
        top-left corner latent coordinates of each latent patch.
    The tensor is repeated for each batch element.
    """
    latent_sample_coords = torch.meshgrid(
        torch.arange(0, latent_num_frames, self._patch_size[0], device=device),
        torch.arange(0, latent_height, self._patch_size[1], device=device),
        torch.arange(0, latent_width, self._patch_size[2], device=device),
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    latent_coords = rearrange(latent_coords, "b c f h w -> b c (f h w)", b=batch_size)
    return latent_coords


class SymmetricPatchifier(Patchifier):

  def patchify(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
    b, _, f, h, w = latents.shape
    latent_coords = self.get_latent_coords(f, h, w, b, latents.device)
    latents = rearrange(
        latents,
        "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
        p1=self._patch_size[0],
        p2=self._patch_size[1],
        p3=self._patch_size[2],
    )
    return latents, latent_coords

  def unpatchify(
      self,
      latents: Tensor,
      output_height: int,
      output_width: int,
      out_channels: int,
  ) -> Tuple[Tensor, Tensor]:
    output_height = output_height // self._patch_size[1]
    output_width = output_width // self._patch_size[2]
    latents = rearrange(
        latents,
        "b (f h w) (c p q) -> b c f (h p) (w q)",
        h=output_height,
        w=output_width,
        p=self._patch_size[1],
        q=self._patch_size[2],
    )
    return latents
