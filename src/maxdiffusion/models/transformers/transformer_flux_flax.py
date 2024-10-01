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

'''This script is used an example of how to shard the UNET on TPU.'''

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from ...configuration_utils import ConfigMixin
from ..modeling_flax_utils import FlaxModelMixin
from ..normalization_flax import AdaLayerNormZeroSingle, AdaLayerNormContinuous
from ..attention_flax import FlaxAttention
from ..embeddings_flax import FluxPosEmbed, CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from ...common_types import BlockSizes
from ... import max_logging
from ...utils import BaseOutput

@flax.struct.dataclass
class Transformer2DModelOutput(BaseOutput):
  """
  The output of [`FluxTransformer2DModel`].

  Args:
  sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
    The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
  """

  sample: jnp.ndarray

class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    qk_norm: str ="rms_norm"
    eps: int = 1e-6
    flash_min_seq_length: int = 4096
    flash_block_sizes: BlockSizes = None
    mesh: jax.sharding.Mesh = None
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    mlp_ratio: int = 4.0
    attention_kernel: str = "dot_product"
    flash_min_seq_length: int = 4096
    flash_block_sizes: BlockSizes = None
    mesh: jax.sharding.Mesh = None
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    def setup(self):
        super.__init__()
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(self.dim)
        self.proj_mlp = nn.Dense(self.mlp_hidden_dim)
        self.act_mlp = nn.GELU
        self.proj_out = nn.Dense(self.dim)
        self.attn = FlaxAttention(
            query_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,


        )
    
    def __call__(self, hidden_states, temb, image_rotary_emb=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = jnp.expand_dims(x, axis=1)

class FluxTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    The Tranformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.

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
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=self.axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if self.guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision
        )
        self.context_embedder = nn.Dense(
            self.inner_dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision)
        self.x_embedder = nn.Dense(
            self.inner_dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision
        )

        self.tranformer_blocks = nn.scan(
            FluxTransformerBlock,
            variable_axes={"params" : 0},
            split_rngs={"params" : True},
            in_axes=(
              nn.broadcast,
              nn.broadcast,
              nn.broadcast,
              nn.broadcast, 
            ),
            length=self.num_layers
        )(
            dim=self.dim,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision
        )

        self.single_tranformer_blocks = nn.scan(
            FluxSingleTransformerBlock,
            variable_axes={"params" : 0},
            split_rngs={"params" : True},
            in_axes=(
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast
            ),
            length=self.num_single_layers
        )(
          dim=self.dim,
          num_attention_heads=self.num_attention_heads,
          attention_head_dim=self.attention_head_dim,
          attention_kernel=self.attention_kernel,
          flash_min_seq_length=self.flash_min_seq_length,
          flash_block_sizes=self.flash_block_sizes,
          mesh=self.mesh,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision
        )

        self.norm_out = AdaLayerNormContinuous(
          self.inner_dim,
          elementwise_affine=False,
          eps=1e-6,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision)
        
        self.proj_out = nn.Dense(self.patch_size**2 * self.out_channels, use_bias=True)
    
    def __call__(
      self,
      hidden_states,
      encoder_hidden_states,
      pooled_projections,
      timestep,
      img_ids,
      txt_ids,
      guidance,
      return_dict: bool = True,
      train: bool = False,
    ):
      
      hidden_states = self.x_embedder(hidden_states)

      timestep = timestep.astype(hidden_states.dtype) * 1000
      if guidance is not None:
        guidance = guidance.astype(hidden_states.dtype) * 1000
      else:
        guidance = None
    
      temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
      )
      encoder_hidden_states = self.context_embedder(encoder_hidden_states)
      if txt_ids.ndim == 3:
        max_logging.log(
          "Passing `txt_ids` 3d torch.Tensor is deprecated."
          "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
      if img_ids.ndim == 3:
        max_logging.log(
          "Passing `img_ids` 3d torch.Tensor is deprecated."
          "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]
      
      ids = jnp.concatenate((txt_ids, img_ids), axis=0)
      image_rotary_emb = self.pos_embed(ids)

      encoder_hidden_states, hidden_states = self.tranformer_blocks(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb
      )

      hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], dim=1)

      hidden_states = self.single_tranformer_blocks(
        hidden_states=hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb
      )

      hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

      hidden_states = self.norm_out(hidden_states, temb)
      output = self.proj_out(hidden_states)

      if not return_dict:
        return (output,)
     
      return Transformer2DModelOutput(sample=output)


