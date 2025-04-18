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

from typing import Tuple, List
from flax import nnx
from ...configuration_utils import ConfigMixin, flax_register_to_config
from ..modeling_flax_utils import FlaxModelMixin

class WanEncoder3d(nnx.Module):
  pass

class WanCausalConv3d(nnx.Module):
  pass

class WanDecoder3d(nnx.Module):
  pass

class AutoencoderKLWan(nnx.Module, FlaxModelMixin, ConfigMixin):
  def __init__(
    self,
    base_dim: int = 96,
    z_dim: int = 16,
    dim_mult: Tuple[int] = [1,2,4,4],
    num_res_blocks: int = 2,
    attn_scales: List[float] = [],
    temporal_downsample: List[bool] = [False, True, True],
    dropout: float = 0.0,
    latents_mean: List[float] = [
      -0.7571,
      -0.7089,
      -0.9113,
      0.1075,
      -0.1745,
      0.9653,
      -0.1517,
      1.5508,
      0.4134,
      -0.0715,
      0.5517,
      -0.3632,
      -0.1922,
      -0.9497,
      0.2503,
      -0.2921,
    ],
    latents_std: List[float] = [
      2.8184,
      1.4541,
      2.3275,
      2.6558,
      1.2196,
      1.7708,
      2.6052,
      2.0743,
      3.2687,
      2.1526,
      2.8652,
      1.5579,
      1.6382,
      1.1253,
      2.8251,
      1.9160,
    ],
  ):
    self.z_dim = z_dim
    self.temporal_downsample = temporal_downsample
    self.temporal_upsample = temporal_downsample[::-1]

    self.encoder = WanEncoder3d(z_dim * 2, z_dim * 2, 1)
    self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

    self.decoder = WanDecoder3d(
      base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temporal_upsample, dropout
    )