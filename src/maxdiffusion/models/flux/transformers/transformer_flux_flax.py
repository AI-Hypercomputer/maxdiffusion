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
from ...gradient_checkpoint import GradientCheckpointType
from jax import checkpoint_policies as cp
from jax.ad_checkpoint import checkpoint_name

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


class MlpAndOutputBlock(nn.Module):
  dim: int
  mlp_ratio: float = 4.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  def setup(self):
    self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)
    self.lin_mlp = nn.Dense(
        self.mlp_hidden_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.mlp_act = nn.gelu
    self.linear2 = nn.Dense(
        self.dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, x, attn_output, gate, residual):
    mlp = self.lin_mlp(x)
    attn_mlp = jnp.concatenate([attn_output, self.mlp_act(mlp)], axis=2)
    attn_mlp = nn.with_logical_constraint(
        attn_mlp, ("activation_batch", None, "mlp")
    )
    hidden_states = self.linear2(attn_mlp)
    hidden_states = checkpoint_name(hidden_states, "lin2_hidden_states")
    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states
    return hidden_states


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
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False

  def setup(self):
    self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

    self.norm = AdaLayerNormZeroSingle(
        self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
    )

    self.lin_qkv = nn.Dense(
        self.dim * 3,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.mlp_and_out = nn.remat(MlpAndOutputBlock, prevent_cse=True)(
        dim=self.dim,
        mlp_ratio=self.mlp_ratio,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
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
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
    )

  def __call__(self, hidden_states, temb, image_rotary_emb=None):
    residual = hidden_states
    
    # FIX: Constrain inputs using valid config parameters (None skips sequence length axis parsing)
    hidden_states = nn.with_logical_constraint(
        hidden_states, ("activation_batch", None, "mlp")
    )
    
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    
    qkv = self.lin_qkv(norm_hidden_states)
    qkv = checkpoint_name(qkv, "lin1_norm_hidden_states")
    qkv = nn.with_logical_constraint(qkv, ("activation_batch", None, "mlp"))
    
    B, L = hidden_states.shape[:2]
    H, D, K = self.num_attention_heads, qkv.shape[-1] // (self.num_attention_heads * 3), 3
    
    qkv_proj = qkv.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv_proj

    q = self.attn.query_norm(q)
    k = self.attn.key_norm(k)

    if image_rotary_emb is not None:
      image_rotary_emb_reordered = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)
      q, k = apply_rope(q, k, image_rotary_emb_reordered)

    q = q.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
    k = k.transpose(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1)
    v = v.transpose(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)

    attn_output = self.attn.attention_op.apply_attention(q, k, v)
    attn_output = checkpoint_name(attn_output, "attn_output")

    hidden_states = self.mlp_and_out(norm_hidden_states, attn_output, gate, residual)
    
    if hidden_states.dtype == jnp.float16:
      hidden_states = jnp.clip(hidden_states, -65504, 65504)

    return hidden_states


class FluxTransformerBlock(nn.Module):
  dim: int
  num_attention_heads: int
  attention_head_dim: int
  qk_norm: str = "rms_norm"
  eps: float = 1e-6
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  mlp_ratio: float = 4.0
  qkv_bias: bool = False
  attention_kernel: str = "dot_product"
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False

  def setup(self):
    # These contain the parameter projections ("lin"), optimize them using your updated AdaLayerNorm class
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
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
    )

    # REMOVED: self.img_norm2 and self.txt_norm2 completely to stop HBM memory spilling.
    # The mathematical reductions are handled natively below.

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
    ])

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
    ])

  def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None):
    # Enforce active partitioning based on your FSDP setup config
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", None, "mlp"))
    encoder_hidden_states = nn.with_logical_constraint(encoder_hidden_states, ("activation_batch", None, "mlp"))

    # 1. First Adaptive Normalization Pass
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.img_norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.txt_norm1(
        encoder_hidden_states, emb=temb
    )

    # 2. Attention Mechanics
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    # --- IMAGE STREAM OPTIMIZATION (img_norm2) ---
    attn_output = gate_msa * attn_output
    hidden_states = hidden_states + attn_output
    
    # Fully fused LayerNorm + scale_mlp + shift_mlp compilation block
    img_mean = jnp.mean(hidden_states, axis=-1, keepdims=True)
    img_var = jnp.mean(jnp.square(hidden_states - img_mean), axis=-1, keepdims=True)
    img_inv_std = jax.lax.rsqrt(img_var + self.eps)
    
    norm_hidden_states = (hidden_states - img_mean) * img_inv_std * (1 + scale_mlp) + shift_mlp
    norm_hidden_states = nn.with_logical_constraint(norm_hidden_states, ("activation_batch", None, "mlp"))

    ff_output = self.img_mlp(norm_hidden_states)
    hidden_states = hidden_states + gate_mlp * ff_output

    # --- TEXT STREAM OPTIMIZATION (txt_norm2) ---
    context_attn_output = c_gate_msa * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    # Fully fused LayerNorm + c_scale_mlp + c_shift_mlp compilation block
    txt_mean = jnp.mean(encoder_hidden_states, axis=-1, keepdims=True)
    txt_var = jnp.mean(jnp.square(encoder_hidden_states - txt_mean), axis=-1, keepdims=True)
    txt_inv_std = jax.lax.rsqrt(txt_var + self.eps)

    norm_encoder_hidden_states = (encoder_hidden_states - txt_mean) * txt_inv_std * (1 + c_scale_mlp) + c_shift_mlp
    norm_encoder_hidden_states = nn.with_logical_constraint(norm_encoder_hidden_states, ("activation_batch", None, "mlp"))

    context_ff_output = self.txt_mlp(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
    
    # Safe numerical clipping limits for half precision math execution
    if encoder_hidden_states.dtype == jnp.float16 or encoder_hidden_states.dtype == jnp.bfloat16:
      encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
      hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states, encoder_hidden_states

class ScannedDoubleBlockWrapper(nn.Module):
    block_kwargs: dict
    
    @nn.compact
    def __call__(self, carry, _):
        hidden_states, encoder_hidden_states, temb, image_rotary_emb = carry
        
        # Instantiate the pure block (no remat here)
        block = FluxTransformerBlock(**self.block_kwargs)
        
        h_out, e_out = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        return (h_out, e_out, temb, image_rotary_emb), None


class ScannedSingleBlockWrapper(nn.Module):
    block_kwargs: dict
    
    @nn.compact
    def __call__(self, carry, _):
        hidden_states, temb, image_rotary_emb = carry
        
        # Instantiate the pure block
        block = FluxSingleTransformerBlock(**self.block_kwargs)
        h_out = block(
            hidden_states=hidden_states, 
            temb=temb, 
            image_rotary_emb=image_rotary_emb
        )
        return (h_out, temb, image_rotary_emb), None

@flax_register_to_config
class FluxTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
  r"""
  The Transformer model introduced in Flux.
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
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  theta: int = 1000
  attention_kernel: str = "dot_product"
  eps: float = 1e-6
  remat_policy: str = "None"
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False

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
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.img_in = nn.Dense(
        self.inner_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.gradient_checkpoint = GradientCheckpointType.from_str(self.remat_policy)

    # 2. Apply the policy to the Module classes
    #RematDoubleBlock = self.gradient_checkpoint.apply_linen(FluxTransformerBlock)
    #RematSingleBlock = self.gradient_checkpoint.apply_linen(FluxSingleTransformerBlock)

    # 1. Prepare the kwargs for the double blocks
    double_kwargs = {
        'dim': self.inner_dim,
        'num_attention_heads': self.num_attention_heads,
        'attention_head_dim': self.attention_head_dim,
        'attention_kernel': self.attention_kernel,
        'flash_min_seq_length': self.flash_min_seq_length,
        'flash_block_sizes': self.flash_block_sizes,
        'mesh': self.mesh,
        'dtype': self.dtype,
        'weights_dtype': self.weights_dtype,
        'precision': self.precision,
        'mlp_ratio': self.mlp_ratio,
        'qkv_bias': self.qkv_bias,
        'use_base2_exp': self.use_base2_exp,
        'use_experimental_scheduler': self.use_experimental_scheduler,
    }

    # 2. Force strict checkpointing on the Double Wrapper
    #RemattedDoubleWrapper = nn.remat(ScannedDoubleBlockWrapper, prevent_cse=True, policy=cp.checkpoint_dots_with_no_batch_dims)
    #RemattedDoubleWrapper = nn.remat(ScannedDoubleBlockWrapper, prevent_cse=True, policy=cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host"))
    RemattedDoubleWrapper = nn.remat(ScannedDoubleBlockWrapper, prevent_cse=True, policy=cp.save_any_names_but_these("img_qkv_proj", "txt_qkv_proj"))

    self.scanned_double_blocks = nn.scan(
        RemattedDoubleWrapper,
        variable_axes={'params': 0},
        split_rngs={'params': True, 'dropout': True},
        length=self.num_layers,
        metadata_params={'partition_name': None}
    )(block_kwargs=double_kwargs)

    # 3. Define pure kwargs for single blocks
    single_kwargs = {
        'dim': self.inner_dim,
        'num_attention_heads': self.num_attention_heads,
        'attention_head_dim': self.attention_head_dim,
        'attention_kernel': self.attention_kernel,
        'flash_min_seq_length': self.flash_min_seq_length,
        'flash_block_sizes': self.flash_block_sizes,
        'mesh': self.mesh,
        'dtype': self.dtype,
        'weights_dtype': self.weights_dtype,
        'precision': self.precision,
        'mlp_ratio': self.mlp_ratio,
        'use_base2_exp': self.use_base2_exp,
        'use_experimental_scheduler': self.use_experimental_scheduler,
    }

    # 4. Force strict checkpointing on the Single Wrapper
    #RemattedSingleWrapper = nn.remat(ScannedSingleBlockWrapper, prevent_cse=True, policy=cp.checkpoint_dots_with_no_batch_dims)
    #RemattedSingleWrapper = nn.remat(ScannedSingleBlockWrapper, prevent_cse=True, policy=cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host"))
    RemattedSingleWrapper = ScannedSingleBlockWrapper

    self.scanned_single_blocks = nn.scan(
        RemattedSingleWrapper,
        variable_axes={'params': 0},
        split_rngs={'params': True, 'dropout': True},
        length=self.num_single_layers,
        metadata_params={'partition_name': None}
    )(block_kwargs=single_kwargs)

    self.norm_out = AdaLayerNormContinuous(
        self.inner_dim,
        elementwise_affine=False,
        eps=self.eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.proj_out = nn.Dense(
        self.patch_size**2 * self.out_channels,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        use_bias=True,
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

    freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.bfloat16) / half).astype(dtype=t.dtype)

    args = t[:, None].astype(jnp.bfloat16) * freqs[None]
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
  ):
    hidden_states = self.img_in(hidden_states)
    timestep = self.timestep_embedding(timestep, 256)
    timestep = nn.with_logical_constraint(timestep, ("activation_batch", None))

    if self.guidance_embeds:
      guidance = self.timestep_embedding(guidance, 256)
    else:
      guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    temb = nn.with_logical_constraint(temb, ("activation_batch", None))

    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]

    ids = jnp.concatenate((txt_ids, img_ids), axis=0)
    ids = nn.with_logical_constraint(ids, ("activation_batch", None))
    image_rotary_emb = self.pe_embedder(ids)
    image_rotary_emb = nn.with_logical_constraint(image_rotary_emb, (None, None))

    carry = (hidden_states, encoder_hidden_states, temb, image_rotary_emb)
    carry, _ = self.scanned_double_blocks(carry, None)
    hidden_states, encoder_hidden_states, _, _ = carry

    hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))
    
    # Execute the 38 Single Blocks
    carry = (hidden_states, temb, image_rotary_emb)
    carry, _ = self.scanned_single_blocks(carry, None)
    hidden_states, _, _ = carry

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

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
        16, 
        2 * resolution // scale_factor,
        2 * resolution // scale_factor,
    )
    text_shape = (
        batch_size,
        max_sequence_length,
        4096, 
    )
    text_ids_shape = (
        batch_size,
        max_sequence_length,
        3, 
    )
    vec_shape = (
        batch_size,
        768, 
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