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
import torch
from torch import nn


class PixelNorm(nn.Module):

  def __init__(self, dim=1, eps=1e-8):
    super(PixelNorm, self).__init__()
    self.dim = dim
    self.eps = eps

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)
