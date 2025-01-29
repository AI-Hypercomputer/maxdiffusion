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

from einops import repeat, rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn 
from chex import Array

from ..modules.layers import (
  timestep_embedding,
  MLPEmbedder,
  EmbedND,
  DoubleStreamBlock,
  SingleStreamBlock,
  LastLayer
)
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin, flax_register_to_config
from ....common_types import BlockSizes

class Identity(nn.Module):
  def __call__(self, x: Array) -> Array:
    return x

def scan_double_block_layers(
  inner_dim,
  num_heads,
  mlp_ratio,
  attention_head_dim,
  flash_min_seq_length,
  flash_block_sizes,
  mesh,
  dtype,
  weights_dtype,
  precision,
  qkv_bias,
  attention_kernel: str,
  num_layers: int):

  scan_fn = nn.scan(
    DoubleStreamBlock,
    variable_broadcast='params',
    in_axes=(
      nn.broadcast,
      nn.broadcast,
      nn.broadcast
    ),
    out_axes=nn.broadcast,
    split_rngs={'params' : False},
    length=num_layers
  )
  return scan_fn(
    hidden_size=inner_dim,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    attention_head_dim=attention_head_dim,
    flash_min_seq_length=flash_min_seq_length,
    flash_block_sizes=flash_block_sizes,
    mesh=mesh,
    dtype=dtype,
    weights_dtype=weights_dtype,
    precision=precision,
    qkv_bias=qkv_bias,
    attention_kernel=attention_kernel)

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

    img = nn.Dense(
      inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      ),
      name="img_in"
    )(img)

    vec = MLPEmbedder(
      hidden_dim=inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision,
      name="time_in"
    )(timestep_embedding(timesteps, 256))

    if self.guidance_embeds:
      if guidance is None:
        raise ValueError(
          "Didn't get guidance strength for guidance distrilled model."
        )
      
      vec = vec + MLPEmbedder(
        hidden_dim=inner_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        name="guidance_in"
      )(timestep_embedding(guidance, 256))
    
    vec = vec + MLPEmbedder(
      hidden_dim=inner_dim,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision,
      name="vector_in"
    )(y)

    txt = nn.Dense(
      inner_dim,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      ),
      name="text_in"
    )(txt)

    ids = jnp.concatenate((txt_ids, img_ids), axis=1)

    #pe_embedder
    pe = EmbedND(
      dim=pe_dim,
      theta=10000,
      axes_dim=self.axes_dims_rope
    )(ids)
    # breakpoint()
    # img, txt = DoubleStreamBlock(
    #   hidden_size=inner_dim,
    #   num_heads=self.num_attention_heads,
    #   mlp_ratio=self.mlp_ratio,
    #   attention_head_dim=self.attention_head_dim,
    #   flash_min_seq_length=self.flash_min_seq_length,
    #   flash_block_sizes=self.flash_block_sizes,
    #   mesh=self.mesh,
    #   dtype=self.dtype,
    #   weights_dtype=self.weights_dtype,
    #   precision=self.precision,
    #   qkv_bias=self.qkv_bias,
    #   attention_kernel=self.attention_kernel,
    #   name="double_blocks_0"
    # )(img=img, txt=txt, vec=vec, pe=pe)
    # # breakpoint()
    for _ in range(self.num_layers):
      img, txt = DoubleStreamBlock(
        hidden_size=inner_dim,
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
    # img, txt = nn.Sequential(
    #   [
    #     *[
    #       DoubleStreamBlock(
    #         hidden_size=inner_dim,
    #         num_heads=self.num_attention_heads,
    #         mlp_ratio=self.mlp_ratio,
    #         attention_head_dim=self.attention_head_dim,
    #         flash_min_seq_length=self.flash_min_seq_length,
    #         flash_block_sizes=self.flash_block_sizes,
    #         mesh=self.mesh,
    #         dtype=self.dtype,
    #         weights_dtype=self.weights_dtype,
    #         precision=self.precision,
    #         qkv_bias=self.qkv_bias,
    #         attention_kernel=self.attention_kernel,
    #       )(img=img, txt=txt, vec=vec, pe=pe)
    #       for _ in range(2)
    #     ]
    #   ]
    # )
    # breakpoint()
    # img, txt = scan_double_block_layers(
    #   inner_dim=inner_dim,
    #   num_heads=self.num_attention_heads,
    #   mlp_ratio=self.mlp_ratio,
    #   attention_head_dim=self.attention_head_dim,
    #   flash_min_seq_length=self.flash_min_seq_length,
    #   flash_block_sizes=self.flash_block_sizes,
    #   mesh=self.mesh,
    #   dtype=self.dtype,
    #   weights_dtype=self.weights_dtype,
    #   precision=self.precision,
    #   qkv_bias=self.qkv_bias,
    #   attention_kernel=self.attention_kernel,
    #   num_layers=self.num_layers
    # )(img, txt, vec, pe)
    img = jnp.concatenate((txt, img), axis=1)
    for _ in range(self.num_single_layers):
      img, SingleStreamBlock(
        hidden_size=inner_dim,
        num_heads=self.num_attention_heads,
        mlp_ratio=self.mlp_ratio,
        attention_head_dim=self.attention_head_dim,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        attention_kernel=self.attention_kernel
      )(img, vec, pe)

    img = img[:, txt.shape[1] :, ...]

    LastLayer(
      hidden_size=inner_dim,
      patch_size=1,
      out_channels=out_channels,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision,
      name="final_layer"
    )

    return img
  
  def init_weights(self, rngs, eval_only=True):
    scale_factor = 16
    resolution = 1024
    num_devices = len(jax.devices())
    batch_size = 1 * num_devices
    batch_image_shape = (
        batch_size,
        16,  # 16 to match jflux.get_noise
        2 * resolution // scale_factor,
        2 * resolution // scale_factor,
    )
    # bs, encoder_input, seq_length
    text_shape = (
        batch_size,
        256,
        4096,  # Sequence length of text encoder, how to get this programmatically?
    )
    text_ids_shape = (
        batch_size,
        256,
        3,  # Hardcoded to match jflux.prepare
    )
    vec_shape = (
        batch_size,
        768,  # Sequence length of clip, how to get this programmatically?
    )
    img = jnp.zeros(batch_image_shape, dtype=self.dtype)
    bs, c, h, w = img.shape
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = jnp.zeros((h // 2, w // 2, 3), dtype=self.dtype)
    img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt = jnp.zeros(text_shape, dtype=self.dtype)
    txt_ids = jnp.zeros(text_ids_shape, dtype=self.dtype)

    t_vec = jnp.full(bs, 0, dtype=self.dtype)

    vec = jnp.zeros(vec_shape, dtype=self.dtype)

    guidance_vec = jnp.full(bs, 4.0, dtype=self.dtype)

    if eval_only:
      return jax.eval_shape(
          self.init,
          rngs,
          img=img,
          img_ids=img_ids,
          txt=txt,
          txt_ids=txt_ids,
          y=vec,
          timesteps=t_vec,
          guidance=guidance_vec,
      )["params"]
    else:
      return self.init(
          rngs,
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          y=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )["params"]