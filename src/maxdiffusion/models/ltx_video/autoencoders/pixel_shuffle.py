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
import torch.nn as nn
from einops import rearrange


class PixelShuffleND(nn.Module):

  def __init__(self, dims, upscale_factors=(2, 2, 2)):
    super().__init__()
    assert dims in [1, 2, 3], "dims must be 1, 2, or 3"
    self.dims = dims
    self.upscale_factors = upscale_factors

  def forward(self, x):
    if self.dims == 3:
      return rearrange(
          x,
          "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
          p1=self.upscale_factors[0],
          p2=self.upscale_factors[1],
          p3=self.upscale_factors[2],
      )
    elif self.dims == 2:
      return rearrange(
          x,
          "b (c p1 p2) h w -> b c (h p1) (w p2)",
          p1=self.upscale_factors[0],
          p2=self.upscale_factors[1],
      )
    elif self.dims == 1:
      return rearrange(
          x,
          "b (c p1) f h w -> b c (f p1) h w",
          p1=self.upscale_factors[0],
      )
