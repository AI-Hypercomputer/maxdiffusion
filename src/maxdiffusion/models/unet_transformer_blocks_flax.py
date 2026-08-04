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

"""Flax Transformer and UNet attention blocks for 2D models."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from .. import common_types
from .attention_flax import AttentionOp
from . import quantizations

Array = common_types.Array
Mesh = common_types.Mesh
DType = common_types.DType
BlockSizes = common_types.BlockSizes
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED
Quant = quantizations.AqtQuantization


class FlaxAttention(nn.Module):
  r"""
  A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

  Parameters:
      query_dim (:obj:`int`):
          Input hidden states dimension
      heads (:obj:`int`, *optional*, defaults to 8):
          Number of heads
      dim_head (:obj:`int`, *optional*, defaults to 64):
          Hidden states dimension inside each head
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      quant (`AqtQuantization`, *optional*, defaults to None)

  """

  query_dim: int
  heads: int = 8
  dim_head: int = 64
  dropout: float = 0.0
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD)
  precision: jax.lax.Precision = None
  quant: Quant = None

  def setup(self):
    if self.attention_kernel == "flash" and self.mesh is None:
      raise ValueError(f"The flash attention kernel requires a value for mesh, but mesh is {self.mesh}")
    inner_dim = self.dim_head * self.heads
    scale = self.dim_head**-0.5

    self.attention_op = AttentionOp(
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        scale=scale,
        heads=self.heads,
        dim_head=self.dim_head,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        split_head_dim=self.split_head_dim,
        flash_block_sizes=self.flash_block_sizes,
        dtype=self.dtype,
        quant=self.quant,
    )

    qkv_init_kernel = nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads"))
    dot_general_cls = None
    if self.quant:
      dot_general_cls = self.quant.dot_general_cls()
    self.query = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_q",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.key = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_k",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.value = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_v",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )

    self.proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("heads", "embed")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="to_out_0",
        precision=self.precision,
        dot_general_cls=dot_general_cls,
    )
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(
      self,
      hidden_states,
      context=None,
      deterministic=True,
      cross_attention_kwargs=None,
  ):
    context = hidden_states if context is None else context
    query_proj = self.query(hidden_states)
    key_proj = self.key(context)
    value_proj = self.value(context)

    query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
    key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
    value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

    hidden_states = self.attention_op.apply_attention(query_proj, key_proj, value_proj)

    hidden_states = self.proj_attn(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, (BATCH, LENGTH, HEAD))
    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxBasicTransformerBlock(nn.Module):
  r"""
  A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
  https://arxiv.org/abs/1706.03762


  Parameters:
      dim (:obj:`int`):
          Inner hidden states dimension
      n_heads (:obj:`int`):
          Number of heads
      d_head (:obj:`int`):
          Hidden states dimension inside each head
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      only_cross_attention (`bool`, defaults to `False`):
          Whether to only apply cross attention.
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      quant (`AqtQuantization`, *optional*, defaults to None)
  """

  dim: int
  n_heads: int
  d_head: int
  dropout: float = 0.0
  only_cross_attention: bool = False
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  precision: jax.lax.Precision = None
  quant: Quant = None

  def setup(self):
    # self attention (or cross_attention if only_cross_attention is True)
    self.attn1 = FlaxAttention(
        self.dim,
        self.n_heads,
        self.d_head,
        self.dropout,
        self.use_memory_efficient_attention,
        self.split_head_dim,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant=self.quant,
    )
    # cross attention
    self.attn2 = FlaxAttention(
        self.dim,
        self.n_heads,
        self.d_head,
        self.dropout,
        self.use_memory_efficient_attention,
        self.split_head_dim,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant=self.quant,
    )
    self.ff = FlaxFeedForward(
        dim=self.dim,
        dropout=self.dropout,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, context, deterministic=True, cross_attention_kwargs=None):
    # self attention
    residual = hidden_states
    if self.only_cross_attention:
      hidden_states = self.attn1(
          self.norm1(hidden_states),
          context,
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )
    else:
      hidden_states = self.attn1(
          self.norm1(hidden_states),
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )

    hidden_states = hidden_states + residual

    # cross attention
    residual = hidden_states
    hidden_states = self.attn2(
        self.norm2(hidden_states),
        context,
        deterministic=deterministic,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    hidden_states = hidden_states + residual

    # feed forward
    residual = hidden_states
    hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
    hidden_states = hidden_states + residual

    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxTransformer2DModel(nn.Module):
  r"""
  A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
  https://arxiv.org/pdf/1506.02025.pdf


  Parameters:
      in_channels (:obj:`int`):
          Input number of channels
      n_heads (:obj:`int`):
          Number of heads
      d_head (:obj:`int`):
          Hidden states dimension inside each head
      depth (:obj:`int`, *optional*, defaults to 1):
          Number of transformers block
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      use_linear_projection (`bool`, defaults to `False`): tbd
      only_cross_attention (`bool`, defaults to `False`): tbd
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
      use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
          enable memory efficient attention https://arxiv.org/abs/2112.05682
      split_head_dim (`bool`, *optional*, defaults to `False`):
          Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
          enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
      attention_kernel (`str`, *optional*, defaults to `dot_product`)
          Attention mechanism to be used.
      flash_min_seq_length (`int`, *optional*, defaults to 4096)
          Minimum seq length required to apply flash attention.
      flash_block_sizes (`BlockSizes`, *optional*, defaults to None)
          Overrides default block sizes for flash attention.
      mesh (`jax.sharding.mesh`, *optional*, defaults to `None`):
          jax mesh is required if attention is set to flash.
      quant (`AqtQuantization`, *optional*, defaults to None)
            Configures AQT quantization github.com/google/aqt.
  """

  in_channels: int
  n_heads: int
  d_head: int
  depth: int = 1
  dropout: float = 0.0
  use_linear_projection: bool = False
  only_cross_attention: bool = False
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  norm_num_groups: int = 32
  precision: jax.lax.Precision = None
  hidden_state_axis_names: AxisNames = (BATCH, LENGTH, D_KV)
  quant: Quant = (None,)

  def setup(self):
    self.norm = nn.GroupNorm(
        num_groups=self.norm_num_groups,
        epsilon=1e-5,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )

    conv_kernel_init = nn.with_logical_partitioning(
        nn.initializers.lecun_normal(), ("keep_1", "keep_2", "conv_in", "conv_out")
    )

    inner_dim = self.n_heads * self.d_head
    if self.use_linear_projection:
      self.proj_in = nn.Dense(
          inner_dim,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "hidden")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
    else:
      self.proj_in = nn.Conv(
          inner_dim,
          kernel_init=conv_kernel_init,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )

    self.transformer_blocks = [
        FlaxBasicTransformerBlock(
            inner_dim,
            self.n_heads,
            self.d_head,
            dropout=self.dropout,
            only_cross_attention=self.only_cross_attention,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            split_head_dim=self.split_head_dim,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            precision=self.precision,
            quant=self.quant,
        )
        for _ in range(self.depth)
    ]

    if self.use_linear_projection:
      self.proj_out = nn.Dense(
          inner_dim,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("hidden", "embed")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
    else:
      self.proj_out = nn.Conv(
          inner_dim,
          kernel_init=conv_kernel_init,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )

    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, context, deterministic=True, cross_attention_kwargs=None):
    batch, height, width, channels = hidden_states.shape
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    if self.use_linear_projection:
      hidden_states = hidden_states.reshape(batch, height * width, channels)
      hidden_states = self.proj_in(hidden_states)
    else:
      hidden_states = self.proj_in(hidden_states)
      hidden_states = hidden_states.reshape(batch, height * width, channels)

    for transformer_block in self.transformer_blocks:
      hidden_states = transformer_block(
          hidden_states,
          context,
          deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )

    if self.use_linear_projection:
      hidden_states = self.proj_out(hidden_states)
      hidden_states = hidden_states.reshape(batch, height, width, channels)
    else:
      hidden_states = hidden_states.reshape(batch, height, width, channels)
      hidden_states = self.proj_out(hidden_states)

    hidden_states = nn.with_logical_constraint(hidden_states, self.hidden_state_axis_names)

    hidden_states = hidden_states + residual
    return self.dropout_layer(hidden_states, deterministic=deterministic)


class FlaxFeedForward(nn.Module):
  r"""
  Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
  [`FeedForward`] class, with the following simplifications:
  - The activation function is currently hardcoded to a gated linear unit from:
  https://arxiv.org/abs/2002.05202
  - `dim_out` is equal to `dim`.
  - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

  Parameters:
      dim (:obj:`int`):
          Inner hidden states dimension
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
  """

  dim: int
  dropout: float = 0.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    # The second linear layer needs to be called
    # net_2 for now to match the index of the Sequential layer
    self.net_0 = FlaxGEGLU(
        self.dim,
        self.dropout,
        self.dtype,
        self.weights_dtype,
        precision=self.precision,
    )
    self.net_2 = nn.Dense(
        self.dim,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.net_0(hidden_states, deterministic=deterministic)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


class FlaxGEGLU(nn.Module):
  r"""
  Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
  https://arxiv.org/abs/2002.05202.

  Parameters:
      dim (:obj:`int`):
          Input hidden states dimension
      dropout (:obj:`float`, *optional*, defaults to 0.0):
          Dropout rate
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
          Parameters `dtype`
  """

  dim: int
  dropout: float = 0.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    inner_dim = self.dim * 4
    self.proj = nn.Dense(
        inner_dim * 2,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.proj(hidden_states)
    hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
    return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)
