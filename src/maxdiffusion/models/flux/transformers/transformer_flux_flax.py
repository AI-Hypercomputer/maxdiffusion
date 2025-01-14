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

from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn 
from chex import Array

from ..modules.layers import timestep_embedding, MLPEmbedder
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin, flax_register_to_config
from ....common_types import BlockSizes

class Identity(nn.Module):
  def __call__(self, x: Array) -> Array:
    return x

class FluxTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
  r"""
    The Tranformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.
  """
  patch_size: int = 1
  in_channels: int = 64
  num_layers: int = 19
  num_single_layers: int = 38
  attention_head_dim: int = 128
  num_attention_heads: int = 24
  joint_attention_dim: int = 4096
  pooled_projection_dim: int = 768
  guidance_embeds: bool = False
  axes_dims_rope: Tuple[int] = (16, 56, 56)
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    self.out_channels = self.in_channels
    self.inner_dim = self.num_attention_heads * self.attention_head_dim

    self.img_in = nn.Dense(
      self.inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )

    self.time_in = MLPEmbedder(
      hidden_dim=self.inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    ) 

    self.vector_in = MLPEmbedder(
      hidden_dim=self.inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    )

    self.guidance_in = (
      MLPEmbedder(
        hidden_dim=self.inner_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision
        )
      if self.guidance_embeds
      else Identity()
    )

    self.txt_in = nn.Dense(
      self.inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision
    )
  
  def __call__(
    self,
    img: Array,
    img_ids: Array,
    txt: Array,
    txt_ids: Array,
    timesteps: Array,
    y: Array,
    guidance: Array | None = None,
    return_dict: bool = True,
    train: bool = False):

    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256))

    if self.guidance_embeds:
      if guidance is None:
        raise ValueError(
          "Didn't get guidance strength for guidance distrilled model."
        )
      
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)