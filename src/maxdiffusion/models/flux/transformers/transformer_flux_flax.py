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

from typing import Tuple
import jax
import math
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import repeat, rearrange
from ....configuration_utils import ConfigMixin, flax_register_to_config
from ...modeling_flax_utils import FlaxModelMixin
from ...normalization_flax import AdaLayerNormZeroSingle, AdaLayerNormContinuous, AdaLayerNormZero
from ...attention_flax import FlaxFluxAttention, apply_rope
from ...embeddings_flax import (FluxPosEmbed, CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings)
from .... import common_types
from ....common_types import BlockSizes
from ....utils import BaseOutput

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

@flax.struct.dataclass
class Transformer2DModelOutput(BaseOutput):
  """
  The output of [`FluxTransformer2DModel`].

  Args:
  sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
    The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
  """

  sample: jnp.ndarray


class FlaxSwiGLUFeedForward(nn.Module):
  dim: int
  dim_out: int
  mult: float = 3.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    inner_dim = int(self.dim * self.mult)
    self.linear_in = nn.Dense(
        inner_dim * 2,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_in",
    )
    self.linear_out = nn.Dense(
        self.dim_out,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_out",
    )

  def __call__(self, x):
    x = self.linear_in(x)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = nn.silu(x1) * x2
    x = self.linear_out(x)
    return x


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
  use_global_modulation: bool = False # Added flag!
  use_swiglu: bool = False # Added flag!

  def setup(self):
    self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

    if self.use_global_modulation:
      self.norm = nn.LayerNorm(
          use_bias=False,
          use_scale=False,
          epsilon=1e-6,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
      )
    else:
      self.norm = AdaLayerNormZeroSingle(
          self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
      )

    out_dim = self.dim * 3 + (2 * self.mlp_hidden_dim if self.use_swiglu else self.mlp_hidden_dim)
    self.linear1 = nn.Dense(
        out_dim,
        use_bias=not self.use_swiglu,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.mlp_act = nn.gelu
    self.linear2 = nn.Dense(
        self.dim,
        use_bias=not self.use_swiglu,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.attn = FlaxFluxAttention(
        query_dim=self.dim,
        heads=self.num_attention_heads,
        dim_head=self.attention_head_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        attention_kernel=self.attention_kernel,
        mesh=self.mesh,
        flash_block_sizes=self.flash_block_sizes,
    )

  def __call__(self, hidden_states, temb=None, image_rotary_emb=None, temb_mod=None):
    residual = hidden_states
    if self.use_global_modulation:
      shift_msa, scale_msa, gate = jnp.split(temb_mod, 3, axis=-1)
      # Unsqueeze sequence dimension for broadcasting when batch_size > 1
      shift_msa = jnp.expand_dims(shift_msa, axis=1)
      scale_msa = jnp.expand_dims(scale_msa, axis=1)
      gate = jnp.expand_dims(gate, axis=1)
      
      norm_hidden_states = self.norm(hidden_states)
      norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
    else:
      norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    qkv, mlp = jnp.split(self.linear1(norm_hidden_states), [3 * self.dim], axis=-1)
    mlp = nn.with_logical_constraint(mlp, ("activation_batch", "activation_length", "activation_embed"))
    qkv = nn.with_logical_constraint(qkv, ("activation_batch", "activation_length", "activation_embed"))

    B, L = hidden_states.shape[:2]
    H, D, K = self.num_attention_heads, qkv.shape[-1] // (self.num_attention_heads * 3), 3
    qkv_proj = qkv.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv_proj

    q = self.attn.query_norm(q)
    k = self.attn.key_norm(k)

    if image_rotary_emb is not None:
      # since this function returns image_rotary_emb and passes it between layers,
      # we do not want to modify it
      image_rotary_emb_reordered = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)
      q, k = apply_rope(q, k, image_rotary_emb_reordered)

    q = q.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
    k = k.transpose(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1)
    v = v.transpose(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)

    attn_output = self.attn.attention_op.apply_attention(q, k, v)

    if self.use_swiglu:
      mlp1, mlp2 = jnp.split(mlp, 2, axis=-1)
      mlp_activated = nn.silu(mlp1) * mlp2
    else:
      mlp_activated = self.mlp_act(mlp)

    attn_mlp = jnp.concatenate([attn_output, mlp_activated], axis=2)
    attn_mlp = nn.with_logical_constraint(attn_mlp, ("activation_batch", "activation_length", "activation_embed"))
    hidden_states = self.linear2(attn_mlp)
    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states
    if hidden_states.dtype == jnp.float16:
      hidden_states = jnp.clip(hidden_states, -65504, 65504)

    return hidden_states


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
  qk_norm: str = "rms_norm"
  eps: int = 1e-6
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  mlp_ratio: float = 4.0
  qkv_bias: bool = False
  attention_kernel: str = "dot_product"
  use_global_modulation: bool = False # Added flag!
  use_swiglu: bool = False # Added flag!

  def setup(self):
    if self.use_global_modulation:
      self.img_norm1 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=self.eps, dtype=self.dtype, param_dtype=self.weights_dtype)
      self.txt_norm1 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=self.eps, dtype=self.dtype, param_dtype=self.weights_dtype)
    else:
      self.img_norm1 = AdaLayerNormZero(self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision)
      self.txt_norm1 = AdaLayerNormZero(self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision)

    self.attn = FlaxFluxAttention(
        query_dim=self.dim,
        heads=self.num_attention_heads,
        dim_head=self.attention_head_dim,
        qkv_bias=self.qkv_bias,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        attention_kernel=self.attention_kernel,
        mesh=self.mesh,
        flash_block_sizes=self.flash_block_sizes,
    )

    self.img_norm2 = nn.LayerNorm(
        use_bias=False,
        use_scale=False,
        epsilon=self.eps,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    if self.use_swiglu:
      self.img_mlp = FlaxSwiGLUFeedForward(
          dim=self.dim,
          dim_out=self.dim,
          mult=self.mlp_ratio,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
          name="img_mlp",
      )
    else:
      self.img_mlp = nn.Sequential([
          nn.Dense(
              int(self.dim * self.mlp_ratio),
              use_bias=True,
              kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
              bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
              dtype=self.dtype,
              param_dtype=self.weights_dtype,
              precision=self.precision,
          ),
          nn.gelu,
          nn.Dense(
              self.dim,
              use_bias=True,
              kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
              bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
              dtype=self.dtype,
              param_dtype=self.weights_dtype,
              precision=self.precision,
          ),
      ], name="img_mlp")

    self.txt_norm2 = nn.LayerNorm(
        use_bias=False,
        use_scale=False,
        epsilon=self.eps,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    if self.use_swiglu:
      self.txt_mlp = FlaxSwiGLUFeedForward(
          dim=self.dim,
          dim_out=self.dim,
          mult=self.mlp_ratio,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
          name="txt_mlp",
      )
    else:
      self.txt_mlp = nn.Sequential([
          nn.Dense(
              int(self.dim * self.mlp_ratio),
              use_bias=True,
              kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
              bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
              dtype=self.dtype,
              param_dtype=self.weights_dtype,
              precision=self.precision,
          ),
          nn.gelu,
          nn.Dense(
              self.dim,
              use_bias=True,
              kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
              bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
              dtype=self.dtype,
              param_dtype=self.weights_dtype,
              precision=self.precision,
          ),
      ], name="txt_mlp")

    # let chunk size default to None
    self._chunk_size = None
    self._chunk_dim = 0

  def __call__(self, hidden_states, encoder_hidden_states, temb=None, image_rotary_emb=None,
               temb_mod_img=None, temb_mod_txt=None):
    if self.use_global_modulation:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = jnp.split(temb_mod_img, 6, axis=-1)
      (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) = jnp.split(temb_mod_txt, 6, axis=-1)
      
      # Unsqueeze sequence dimension for broadcasting when batch_size > 1
      shift_msa = jnp.expand_dims(shift_msa, axis=1)
      scale_msa = jnp.expand_dims(scale_msa, axis=1)
      gate_msa = jnp.expand_dims(gate_msa, axis=1)
      shift_mlp = jnp.expand_dims(shift_mlp, axis=1)
      scale_mlp = jnp.expand_dims(scale_mlp, axis=1)
      gate_mlp = jnp.expand_dims(gate_mlp, axis=1)
      
      c_shift_msa = jnp.expand_dims(c_shift_msa, axis=1)
      c_scale_msa = jnp.expand_dims(c_scale_msa, axis=1)
      c_gate_msa = jnp.expand_dims(c_gate_msa, axis=1)
      c_shift_mlp = jnp.expand_dims(c_shift_mlp, axis=1)
      c_scale_mlp = jnp.expand_dims(c_scale_mlp, axis=1)
      c_gate_mlp = jnp.expand_dims(c_gate_mlp, axis=1)
      
      norm_hidden_states = self.img_norm1(hidden_states)
      norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
      
      norm_encoder_hidden_states = self.txt_norm1(encoder_hidden_states)
      norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa
    else:
      norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.img_norm1(hidden_states, emb=temb)
      norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.txt_norm1(
          encoder_hidden_states, emb=temb
      )

    # Attention.
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    attn_output = gate_msa * attn_output
    hidden_states = hidden_states + attn_output
    norm_hidden_states = self.img_norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    ff_output = self.img_mlp(norm_hidden_states)
    ff_output = gate_mlp * ff_output

    hidden_states = hidden_states + ff_output
    # Process attention outputs for the `encoder_hidden_states`.
    context_attn_output = c_gate_msa * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.txt_norm2(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

    context_ff_output = self.txt_mlp(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
    if encoder_hidden_states.dtype == jnp.float16:
      encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return hidden_states, encoder_hidden_states


@flax_register_to_config
class FluxTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
  r"""
  The Transformer model introduced in Flux.

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
  scale_shift_order: str = "shift_scale"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  theta: int = 1000
  attention_kernel: str = "dot_product"
  eps = 1e-6
  joint_attention_bias: bool = True
  x_embedder_bias: bool = True
  proj_out_bias: bool = True
  use_global_modulation: bool = False # Added config flag!
  use_swiglu: bool = False # Added config flag!

  def setup(self):
    self.out_channels = self.in_channels
    self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

    self.pe_embedder = FluxPosEmbed(theta=self.theta, axes_dim=self.axes_dims_rope, dtype=self.dtype)

    text_time_guidance_cls = (
        CombinedTimestepGuidanceTextProjEmbeddings if self.guidance_embeds else CombinedTimestepTextProjEmbeddings
    )

    self.time_text_embed = text_time_guidance_cls(
        embedding_dim=self.inner_dim,
        pooled_projection_dim=self.pooled_projection_dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.txt_in = nn.Dense(
        self.inner_dim,
        use_bias=self.joint_attention_bias,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.img_in = nn.Dense(
        self.inner_dim,
        use_bias=self.x_embedder_bias,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    if self.use_global_modulation:
      self.double_stream_modulation_img = nn.Dense(
          6 * self.inner_dim,
          use_bias=False,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          name="double_stream_modulation_img",
      )
      self.double_stream_modulation_txt = nn.Dense(
          6 * self.inner_dim,
          use_bias=False,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          name="double_stream_modulation_txt",
      )
      self.single_stream_modulation = nn.Dense(
          3 * self.inner_dim,
          use_bias=False,
          kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          name="single_stream_modulation",
      )

    double_blocks = []
    for _ in range(self.num_layers):
      double_block = FluxTransformerBlock(
          dim=self.inner_dim,
          num_attention_heads=self.num_attention_heads,
          attention_head_dim=self.attention_head_dim,
          attention_kernel=self.attention_kernel,
          flash_min_seq_length=self.flash_min_seq_length,
          flash_block_sizes=self.flash_block_sizes,
          mesh=self.mesh,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
          use_global_modulation=self.use_global_modulation,
          use_swiglu=self.use_swiglu,
      )
      double_blocks.append(double_block)
    self.double_blocks = double_blocks

    single_blocks = []
    for _ in range(self.num_single_layers):
      single_block = FluxSingleTransformerBlock(
          dim=self.inner_dim,
          num_attention_heads=self.num_attention_heads,
          attention_head_dim=self.attention_head_dim,
          attention_kernel=self.attention_kernel,
          flash_min_seq_length=self.flash_min_seq_length,
          flash_block_sizes=self.flash_block_sizes,
          mesh=self.mesh,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
          mlp_ratio=self.mlp_ratio,
          use_global_modulation=self.use_global_modulation,
          use_swiglu=self.use_swiglu,
      )
      single_blocks.append(single_block)

    self.single_blocks = single_blocks

    self.norm_out = AdaLayerNormContinuous(
        self.inner_dim,
        elementwise_affine=False,
        eps=self.eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        scale_shift_order=self.scale_shift_order,
    )

    self.proj_out = nn.Dense(
        self.patch_size**2 * self.out_channels,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        use_bias=self.proj_out_bias,
    )

  def timestep_embedding(self, t: jax.Array, dim: int, max_period=10000, time_factor: float = 1000.0) -> jax.Array:
    """
    Generate timestep embeddings.

    Args:
        t: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        time_factor: Tensor of positional embeddings.

    Returns:
        timestep embeddings.
    """
    t = time_factor * t
    half = dim // 2

    freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=t.dtype) / half)

    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2:
      embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

    if jnp.issubdtype(t.dtype, jnp.floating):
      embedding = embedding.astype(t.dtype)

    return embedding

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
      return_intermediates: bool = False,
  ):
    hidden_states = self.img_in(hidden_states)
    timestep = self.timestep_embedding(timestep, 256, time_factor=1.0)

    timestep = nn.with_logical_constraint(timestep, ("activation_batch", None))

    if self.guidance_embeds:
      guidance = self.timestep_embedding(guidance, 256, time_factor=1.0)
    else:
      guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    temb = nn.with_logical_constraint(temb, ("activation_batch", None))

    if self.use_global_modulation:
      temb_silu = nn.silu(temb)
      double_stream_mod_img = self.double_stream_modulation_img(temb_silu)
      double_stream_mod_txt = self.double_stream_modulation_txt(temb_silu)
      single_stream_mod = self.single_stream_modulation(temb_silu)
    else:
      double_stream_mod_img, double_stream_mod_txt, single_stream_mod = None, None, None

    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]

    ids = jnp.concatenate((txt_ids, img_ids), axis=0)
    ids = nn.with_logical_constraint(ids, ("activation_batch", None))
    image_rotary_emb = self.pe_embedder(ids)
    image_rotary_emb = nn.with_logical_constraint(image_rotary_emb, (None, None))

    # Initialize intermediates collection if requested
    intermediates = {}
    if return_intermediates:
      intermediates["temb"] = temb
      intermediates["global_modulation"] = (double_stream_mod_img, double_stream_mod_txt, single_stream_mod)
      intermediates["double_block_inputs"] = []
      intermediates["double_block_outputs"] = []
      intermediates["single_block_outputs"] = []

    for double_block in self.double_blocks:
      if return_intermediates:
        intermediates["double_block_inputs"].append((hidden_states, encoder_hidden_states))
      hidden_states, encoder_hidden_states = double_block(
          hidden_states=hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          temb=temb,
          image_rotary_emb=image_rotary_emb,
          temb_mod_img=double_stream_mod_img,
          temb_mod_txt=double_stream_mod_txt,
      )
      if return_intermediates:
        intermediates["double_block_outputs"].append((hidden_states, encoder_hidden_states))

    hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))
    
    for single_block in self.single_blocks:
      hidden_states = single_block(
          hidden_states=hidden_states,
          temb=temb,
          image_rotary_emb=image_rotary_emb,
          temb_mod=single_stream_mod,
      )
      if return_intermediates:
        intermediates["single_block_outputs"].append(hidden_states)

    if return_intermediates:
      intermediates["before_split"] = hidden_states

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_intermediates:
      return output, intermediates

    if not return_dict:
      return (output,)

    return Transformer2DModelOutput(sample=output)

  def init_weights(self, rngs, max_sequence_length, eval_only=True):
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
        max_sequence_length,
        4096,  # Sequence length of text encoder, how to get this programmatically?
    )
    text_ids_shape = (
        batch_size,
        max_sequence_length,
        3,  # Hardcoded to match jflux.prepare
    )
    vec_shape = (
        batch_size,
        768,  # Sequence length of clip, how to get this programmatically?
    )
    img = jnp.zeros(batch_image_shape, dtype=self.dtype)
    bs, _, h, w = img.shape
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
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          pooled_projections=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )["params"]
    else:
      return self.init(
          rngs,
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          pooled_projections=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )["params"]
