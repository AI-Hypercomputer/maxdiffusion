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

from ..modules.layers import timestep_embedding, MLPEmbedder, EmbedND, DoubleStreamBlock
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
  mlp_ratio: int = 4
  qkv_bias: bool = True
  guidance_embeds: bool = False
  axes_dims_rope: Tuple[int] = (16, 56, 56)
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  attention_kernel: str = "dot_product"
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  
  @nn.compact
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

    out_channels = self.in_channels
    inner_dim = self.num_attention_heads * self.attention_head_dim
    pe_dim = inner_dim // self.num_attention_heads

    #img = self.img_in(img)
    img = nn.Dense(
      inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )(img)

    #vec = self.time_in(timestep_embedding(timesteps, 256))
    vec = MLPEmbedder(
      hidden_dim=inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    )(timestep_embedding(timesteps, 256))

    if self.guidance_embeds:
      if guidance is None:
        raise ValueError(
          "Didn't get guidance strength for guidance distrilled model."
        )
      
      #vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
      vec = vec + MLPEmbedder(
        hidden_dim=inner_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision
      )(timestep_embedding(guidance, 256))
    
    #vec = vec + self.vector_in(y)
    vec = vec + MLPEmbedder(
      hidden_dim=inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    )(y)

    #txt = self.txt_in(txt)
    txt = nn.Dense(
      inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision
    )(txt)

    ids = jnp.concatenate((txt_ids, img_ids), axis=1)

    #pe_embedder
    pe = EmbedND(
      dim=pe_dim,
      theta=10000,
      axoes_dim=self.axes_dims_rope
    )(ids)

    img, text = nn.scan(
      DoubleStreamBlock,
      variable_broadcast='params',
      in_axes=0, out_axes=0,
      split_rngs={'params' : False}
    )(
      hidden_size=self.hidden_size,
      num_heads=self.num_attention_heads,
      mlp_ratio=self.mlp_ratio,
      attention_head_dim=self.attention_head_dim,
      flash_min_seq_length=self.flash_min_seq_length,
      flash_block_sizes=self.flash_block_sizes,
      mesh=self.mesh,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision,
      qkv_bias=self.qkv_bias,
      attention_kernel=self.attention_kernel,
      
    )(img=img, txt=txt, vec=vec, pe=pe)

    return img, text

    # img = jnp.concatenate((txt, img), axis=1)

    # img = nn.scan(
    #   SingleStreamBlock,
    #   variable_broadcast='params',
    #   in_axes=0, out_axes=0,
    #   split_rngs={'params' : False}
    # )(img, vec=vec, pe=pe)

    # img = img[:, txt.shape[1] :, ...]

    # img = LastLayer(

    # )(img, vec) # (N, T, patch_size ** 2 * out_channels)
    # return img