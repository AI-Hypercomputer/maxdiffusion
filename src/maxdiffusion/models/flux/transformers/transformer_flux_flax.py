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

from typing import Dict, Optional, Tuple
import jax
import math
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import repeat, rearrange
from ....configuration_utils import ConfigMixin, flax_register_to_config
from ...modeling_flax_utils import FlaxModelMixin
from ...normalization_flax import (
    AdaLayerNormZeroSingle,
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    NNXAdaLayerNormZeroSingle,
    NNXAdaLayerNormContinuous,
    NNXAdaLayerNormZero,
)
from ...attention_flax import FlaxFluxAttention as FluxAttention, FlaxFluxAttention, apply_rope
from flax import nnx
from ...embeddings_flax import (
    FluxPosEmbed,
    NNXFluxPosEmbed,
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepGuidanceTextProjEmbeddings as CombinedTimestepGuidanceTextEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    NNXCombinedTimestepGuidanceTextProjEmbeddings,
)
from .... import common_types
from ....common_types import BlockSizes
from ....utils import BaseOutput
from ...gradient_checkpoint import GradientCheckpointType, SKIP_GRADIENT_CHECKPOINT_KEY
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
    attn_mlp = nn.with_logical_constraint(attn_mlp, ("activation_batch", None, "mlp"))
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
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", None, "mlp"))

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    qkv = self.lin_qkv(norm_hidden_states)
    qkv = checkpoint_name(qkv, "lin1_norm_hidden_states")
    qkv = nn.with_logical_constraint(qkv, ("activation_batch", None, "mlp"))

    B, L = hidden_states.shape[:2]
    H, D, K = self.num_attention_heads, qkv.shape[-1] // (self.num_attention_heads * 3), 3

    qkv_proj = qkv.reshape(B, L, K, H, D)
    q, k, v = jnp.split(qkv_proj, 3, axis=2)
    q = q.squeeze(2).swapaxes(1, 2)
    k = k.squeeze(2).swapaxes(1, 2)
    v = v.squeeze(2).swapaxes(1, 2)

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
    h_out = block(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
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
  names_which_can_be_saved: tuple = ()
  names_which_can_be_offloaded: tuple = ()
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
    # RematDoubleBlock = self.gradient_checkpoint.apply_linen(FluxTransformerBlock)
    # RematSingleBlock = self.gradient_checkpoint.apply_linen(FluxSingleTransformerBlock)

    # 1. Prepare the kwargs for the double blocks
    double_kwargs = {
        "dim": self.inner_dim,
        "num_attention_heads": self.num_attention_heads,
        "attention_head_dim": self.attention_head_dim,
        "attention_kernel": self.attention_kernel,
        "flash_min_seq_length": self.flash_min_seq_length,
        "flash_block_sizes": self.flash_block_sizes,
        "mesh": self.mesh,
        "dtype": self.dtype,
        "weights_dtype": self.weights_dtype,
        "precision": self.precision,
        "mlp_ratio": self.mlp_ratio,
        "qkv_bias": self.qkv_bias,
        "use_base2_exp": self.use_base2_exp,
        "use_experimental_scheduler": self.use_experimental_scheduler,
    }

    double_policy = self.gradient_checkpoint.to_jax_policy(
        names_which_can_be_saved=self.names_which_can_be_saved,
        names_which_can_be_offloaded=self.names_which_can_be_offloaded,
        block_type="double",
    )

    if double_policy == SKIP_GRADIENT_CHECKPOINT_KEY:
      RemattedDoubleWrapper = ScannedDoubleBlockWrapper
    else:
      RemattedDoubleWrapper = nn.remat(ScannedDoubleBlockWrapper, prevent_cse=True, policy=double_policy)

    self.scanned_double_blocks = nn.scan(
        RemattedDoubleWrapper,
        variable_axes={"params": 0},
        split_rngs={"params": True, "dropout": True},
        length=self.num_layers,
        metadata_params={"partition_name": None},
    )(block_kwargs=double_kwargs)

    # 3. Define pure kwargs for single blocks
    single_kwargs = {
        "dim": self.inner_dim,
        "num_attention_heads": self.num_attention_heads,
        "attention_head_dim": self.attention_head_dim,
        "attention_kernel": self.attention_kernel,
        "flash_min_seq_length": self.flash_min_seq_length,
        "flash_block_sizes": self.flash_block_sizes,
        "mesh": self.mesh,
        "dtype": self.dtype,
        "weights_dtype": self.weights_dtype,
        "precision": self.precision,
        "mlp_ratio": self.mlp_ratio,
        "use_base2_exp": self.use_base2_exp,
        "use_experimental_scheduler": self.use_experimental_scheduler,
    }

    # 4. Force strict checkpointing on the Single Wrapper
    single_policy = self.gradient_checkpoint.to_jax_policy(
        names_which_can_be_saved=self.names_which_can_be_saved,
        names_which_can_be_offloaded=self.names_which_can_be_offloaded,
        block_type="single",
    )

    if single_policy == SKIP_GRADIENT_CHECKPOINT_KEY:
      RemattedSingleWrapper = ScannedSingleBlockWrapper
    else:
      RemattedSingleWrapper = nn.remat(ScannedSingleBlockWrapper, prevent_cse=True, policy=single_policy)

    self.scanned_single_blocks = nn.scan(
        RemattedSingleWrapper,
        variable_axes={"params": 0},
        split_rngs={"params": True, "dropout": True},
        length=self.num_single_layers,
        metadata_params={"partition_name": None},
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
    in_channels = self.in_channels // 4
    joint_attention_dim = self.joint_attention_dim
    pos_id_dim = 3
    pooled_projection_dim = self.pooled_projection_dim

    batch_image_shape = (
        batch_size,
        in_channels,
        2 * resolution // scale_factor,
        2 * resolution // scale_factor,
    )
    text_shape = (
        batch_size,
        max_sequence_length,
        joint_attention_dim,
    )
    text_ids_shape = (
        batch_size,
        max_sequence_length,
        pos_id_dim,
    )
    vec_shape = (
        batch_size,
        pooled_projection_dim,
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


class FlaxSwiGluFeedForward(nn.Module):
  dim: int
  hidden_dim: int
  out_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: float = None

  def setup(self):
    self.linear_in = nn.Dense(
        2 * self.hidden_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.linear_out = nn.Dense(
        self.out_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, x):
    x = self.linear_in(x)
    x1, x2 = jnp.split(x, 2, axis=-1)
    hidden = nn.silu(x1) * x2
    return self.linear_out(hidden)


class Flux2KleinSingleTransformerBlock(nn.Module):
  dim: int
  num_attention_heads: int
  attention_head_dim: int
  mlp_ratio: float = 3.0
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[Dict[str, int]] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: float = None
  use_global_modulation: bool = False
  use_swiglu: bool = True

  def setup(self):
    mlp_hidden_dim = int(self.dim * self.mlp_ratio)

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

    out_dim = self.dim * 3 + (2 * mlp_hidden_dim if self.use_swiglu else mlp_hidden_dim)
    self.linear1 = nn.Dense(
        out_dim,
        use_bias=not self.use_swiglu,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.mlp_act = jax.nn.gelu
    self.linear2 = nn.Dense(
        self.dim,
        use_bias=not self.use_swiglu,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.attn = FluxAttention(
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
      if isinstance(image_rotary_emb, (tuple, list)):
        image_rotary_emb_reordered = image_rotary_emb
      else:
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


class Flux2KleinTransformerBlock(nn.Module):
  dim: int
  num_attention_heads: int
  attention_head_dim: int
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[Dict[str, int]] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: float = None
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  use_global_modulation: bool = False

  def setup(self):
    if self.use_global_modulation:
      self.norm1 = nn.LayerNorm(
          use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
      )
      self.norm1_context = nn.LayerNorm(
          use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
      )
      self.norm2 = nn.LayerNorm(
          use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
      )
      self.norm2_context = nn.LayerNorm(
          use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
      )
    else:
      self.norm1 = AdaLayerNormZero(
          self.dim,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
      )
      self.norm1_context = AdaLayerNormZero(
          self.dim,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
      )
    self.attn = FluxAttention(
        query_dim=self.dim,
        heads=self.num_attention_heads,
        dim_head=self.attention_head_dim,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        qkv_bias=self.qkv_bias,
    )
    self.ff = FlaxSwiGluFeedForward(
        self.dim,
        int(self.dim * self.mlp_ratio),
        self.dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.ff_context = FlaxSwiGluFeedForward(
        self.dim,
        int(self.dim * self.mlp_ratio),
        self.dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(
      self,
      hidden_states,
      encoder_hidden_states,
      temb,
      image_rotary_emb=None,
      temb_mod_img=None,
      temb_mod_txt=None,
      return_intermediates: bool = False,
  ):
    if self.use_global_modulation:
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(temb_mod_img, 6, axis=-1)
      c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = jnp.split(temb_mod_txt, 6, axis=-1)

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

      norm1_hidden_states = self.norm1(hidden_states) * (1.0 + scale_msa) + shift_msa
      norm1_encoder_hidden_states = self.norm1_context(encoder_hidden_states) * (1.0 + c_scale_msa) + c_shift_msa

    else:
      norm1_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
      norm1_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
          encoder_hidden_states, temb
      )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm1_hidden_states,
        encoder_hidden_states=norm1_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    hidden_states = hidden_states + gate_msa * attn_output
    encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

    if self.use_global_modulation:
      norm2_hidden_states = self.norm2(hidden_states) * (1.0 + scale_mlp) + shift_mlp
      norm2_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (1.0 + c_scale_mlp) + c_shift_mlp

    else:
      norm2_hidden_states, gate_mlp, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
      norm2_encoder_hidden_states, c_gate_mlp, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
          encoder_hidden_states, temb
      )

    mlp_output = self.ff(norm2_hidden_states)
    encoder_mlp_output = self.ff_context(norm2_encoder_hidden_states)

    hidden_states = hidden_states + gate_mlp * mlp_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp * encoder_mlp_output

    if return_intermediates:
      return encoder_hidden_states, hidden_states, norm1_hidden_states, norm1_encoder_hidden_states

    return encoder_hidden_states, hidden_states


@flax_register_to_config
class Flux2KleinTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
  patch_size: int = 1
  in_channels: int = 128
  num_layers: int = 5
  num_single_layers: int = 20
  attention_head_dim: int = 128
  num_attention_heads: int = 24
  joint_attention_dim: int = 4096
  pooled_projection_dim: int = 768
  guidance_embeds: bool = True
  axes_dim: Tuple[int, ...] = (32, 32, 32, 32)
  theta: int = 10000
  qkv_bias: bool = True
  mlp_ratio: float = 3.0
  use_global_modulation: bool = True
  scale_shift_order: str = "scale_shift"
  proj_out_bias: bool = False
  joint_attention_bias: bool = False
  x_embedder_bias: bool = False
  use_swiglu: bool = True
  axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32)
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[Dict[str, int]] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: float = None

  def setup(self):
    self.inner_dim = self.num_attention_heads * self.attention_head_dim

    self.time_text_embed = CombinedTimestepGuidanceTextEmbeddings(
        embedding_dim=self.inner_dim,
        pooled_projection_dim=self.pooled_projection_dim,
        guidance_embeds=self.guidance_embeds,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

    if self.use_global_modulation:
      self.double_stream_modulation_img = nn.Dense(
          6 * self.inner_dim,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
      self.double_stream_modulation_txt = nn.Dense(
          6 * self.inner_dim,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )
      self.single_stream_modulation = nn.Dense(
          3 * self.inner_dim,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
      )

    self.context_embedder = nn.Dense(
        self.inner_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.x_embedder = nn.Dense(
        self.inner_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.pos_embed = FluxPosEmbed(
        theta=self.theta,
        axes_dim=self.axes_dim,
        return_tuple=True,
    )

    double_blocks = []
    for _ in range(self.num_layers):
      double_block = Flux2KleinTransformerBlock(
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
      )
      double_blocks.append(double_block)
    self.double_blocks = double_blocks

    single_blocks = []
    for _ in range(self.num_single_layers):
      single_block = Flux2KleinSingleTransformerBlock(
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
      )
      single_blocks.append(single_block)
    self.single_blocks = single_blocks

    self.norm_out = AdaLayerNormContinuous(
        self.inner_dim,
        elementwise_affine=False,
        eps=1e-6,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        scale_shift_order=self.scale_shift_order,
    )

    self.proj_out = nn.Dense(
        self.in_channels,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        use_bias=self.proj_out_bias,
    )

  def timestep_embedding(self, t: jax.Array, dim: int, max_period=10000, time_factor: float = 1.0) -> jax.Array:
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
      embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
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
    intermediates = {}
    if return_intermediates:
      intermediates["double_block_outputs"] = []
      intermediates["single_block_outputs"] = []

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep * 1000.0
    if guidance is not None:
      guidance = guidance * 1000.0
    temb = self.time_text_embed(timestep, guidance, pooled_projections)
    temb = temb.astype(hidden_states.dtype)

    if self.use_global_modulation:
      temb_silu = nn.silu(temb)
      double_stream_mod_img = self.double_stream_modulation_img(temb_silu)
      double_stream_mod_txt = self.double_stream_modulation_txt(temb_silu)
      single_stream_mod = self.single_stream_modulation(temb_silu)
    else:
      double_stream_mod_img, double_stream_mod_txt, single_stream_mod = None, None, None

    if encoder_hidden_states is not None and hasattr(self, "context_embedder") and self.context_embedder is not None:
      encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if return_intermediates:
      intermediates["x_embedder"] = hidden_states
      intermediates["context_embedder"] = encoder_hidden_states
      intermediates["temb"] = temb
      intermediates["double_stream_mod_img"] = double_stream_mod_img
      intermediates["double_stream_mod_txt"] = double_stream_mod_txt

    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        jnp.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        jnp.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    if return_intermediates:
      intermediates["temb"] = temb
      intermediates["global_modulation"] = (double_stream_mod_img, double_stream_mod_txt, single_stream_mod)
      intermediates["double_block_inputs"] = []
      intermediates["double_block_outputs"] = []
      intermediates["single_block_outputs"] = []

    if return_intermediates:
      intermediates["norm_hidden_states"] = []
      intermediates["norm_encoder_hidden_states"] = []

    for double_block in self.double_blocks:
      if return_intermediates:
        intermediates["double_block_inputs"].append((hidden_states, encoder_hidden_states))
        encoder_hidden_states, hidden_states, norm_h, norm_enc = double_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=concat_rotary_emb,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
            return_intermediates=True,
        )
        intermediates["norm_hidden_states"].append(norm_h)
        intermediates["norm_encoder_hidden_states"].append(norm_enc)
      else:
        encoder_hidden_states, hidden_states = double_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=concat_rotary_emb,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
        )
      if return_intermediates:
        intermediates["double_block_outputs"].append((hidden_states, encoder_hidden_states))

    num_txt_tokens = encoder_hidden_states.shape[1]
    hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

    for single_block in self.single_blocks:
      hidden_states = single_block(
          hidden_states=hidden_states,
          temb=temb,
          image_rotary_emb=concat_rotary_emb,
          temb_mod=single_stream_mod,
      )
      if return_intermediates:
        intermediates["single_block_outputs"].append(hidden_states)

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_intermediates:
      return output, intermediates

    if not return_dict:
      return (output,)

    return Transformer2DModelOutput(sample=output)


# =============================================================================
# FLAX NNX MODEL IMPLEMENTATIONS FOR FLUX.2-KLEIN
# =============================================================================


class NNXFluxDoubleAttention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      query_dim: int,
      heads: int,
      dim_head: int,
      qkv_bias: bool = False,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.query_dim = query_dim
    self.heads = heads
    self.dim_head = dim_head
    inner_dim = heads * dim_head

    self.qkv = nnx.Linear(
        in_features=query_dim,
        out_features=inner_dim * 3,
        use_bias=qkv_bias,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.encoder_qkv = nnx.Linear(
        in_features=query_dim,
        out_features=inner_dim * 3,
        use_bias=qkv_bias,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.proj_attn = nnx.Linear(
        in_features=inner_dim,
        out_features=query_dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.encoder_proj_attn = nnx.Linear(
        in_features=inner_dim,
        out_features=query_dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.query_norm = nnx.RMSNorm(
        num_features=dim_head,
        epsilon=1e-6,
        scale_init=nnx.with_partitioning(nnx.initializers.ones, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.key_norm = nnx.RMSNorm(
        num_features=dim_head,
        epsilon=1e-6,
        scale_init=nnx.with_partitioning(nnx.initializers.ones, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      image_rotary_emb: Tuple[jax.Array, jax.Array],
  ) -> Tuple[jax.Array, jax.Array]:
    batch_size, img_len, _ = hidden_states.shape
    txt_len = encoder_hidden_states.shape[1]

    qkv_img = self.qkv(hidden_states)
    qkv_txt = self.encoder_qkv(encoder_hidden_states)

    q_img, k_img, v_img = jnp.split(qkv_img, 3, axis=-1)
    q_txt, k_txt, v_txt = jnp.split(qkv_txt, 3, axis=-1)

    q_img = rearrange(q_img, "b l (h d) -> b l h d", h=self.heads)
    k_img = rearrange(k_img, "b l (h d) -> b l h d", h=self.heads)
    v_img = rearrange(v_img, "b l (h d) -> b l h d", h=self.heads)

    q_txt = rearrange(q_txt, "b l (h d) -> b l h d", h=self.heads)
    k_txt = rearrange(k_txt, "b l (h d) -> b l h d", h=self.heads)
    v_txt = rearrange(v_txt, "b l (h d) -> b l h d", h=self.heads)

    q_img = self.query_norm(q_img)
    k_img = self.key_norm(k_img)
    q_txt = self.query_norm(q_txt)
    k_txt = self.key_norm(k_txt)

    q = jnp.concatenate([q_txt, q_img], axis=1)
    k = jnp.concatenate([k_txt, k_img], axis=1)
    v = jnp.concatenate([v_txt, v_img], axis=1)

    if image_rotary_emb is not None:
      q, k = apply_rope(q, k, image_rotary_emb)

    scale = self.dim_head**-0.5
    attn_weights = jnp.einsum("b q h d, b k h d -> b h q k", q, k, precision=None) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    out = jnp.einsum("b h q k, b k h d -> b q h d", attn_weights, v, precision=None)

    out = rearrange(out, "b l h d -> b l (h d)")

    out_txt = out[:, :txt_len, :]
    out_img = out[:, txt_len:, :]

    out_img = self.proj_attn(out_img)
    out_txt = self.encoder_proj_attn(out_txt)

    return out_img, out_txt


class NNXFluxSingleAttention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.dim = dim
    self.heads = num_attention_heads
    self.dim_head = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim

    self.to_qkv_mlp_proj = nnx.Linear(
        in_features=dim,
        out_features=inner_dim * 3 + int(dim * 4.0),
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.to_out = nnx.Linear(
        in_features=inner_dim + int(dim * 4.0),
        out_features=dim,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.norm_q = nnx.RMSNorm(
        num_features=attention_head_dim,
        epsilon=1e-6,
        scale_init=nnx.with_partitioning(nnx.initializers.ones, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.norm_k = nnx.RMSNorm(
        num_features=attention_head_dim,
        epsilon=1e-6,
        scale_init=nnx.with_partitioning(nnx.initializers.ones, ("heads",)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      image_rotary_emb: Tuple[jax.Array, jax.Array],
  ) -> jax.Array:
    batch_size, seq_len, _ = hidden_states.shape
    inner_dim = self.heads * self.dim_head

    qkv_mlp = self.to_qkv_mlp_proj(hidden_states)
    qkv, mlp = jnp.split(qkv_mlp, [inner_dim * 3], axis=-1)

    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = rearrange(q, "b l (h d) -> b l h d", h=self.heads)
    k = rearrange(k, "b l (h d) -> b l h d", h=self.heads)
    v = rearrange(v, "b l (h d) -> b l h d", h=self.heads)

    q = self.norm_q(q)
    k = self.norm_k(k)

    if image_rotary_emb is not None:
      q, k = apply_rope(q, k, image_rotary_emb)

    scale = self.dim_head**-0.5
    attn_weights = jnp.einsum("b q h d, b k h d -> b h q k", q, k, precision=None) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    attn_out = jnp.einsum("b h q k, b k h d -> b q h d", attn_weights, v, precision=None)
    attn_out = rearrange(attn_out, "b l h d -> b l (h d)")

    mlp_act = jax.nn.gelu(mlp, approximate=True)
    attn_mlp = jnp.concatenate([attn_out, mlp_act], axis=-1)

    out = self.to_out(attn_mlp)
    return out


class NNXFluxDoubleTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      mlp_ratio: float = 4.0,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.dim = dim
    self.num_heads = num_attention_heads
    self.head_dim = attention_head_dim
    mlp_hidden_dim = int(dim * mlp_ratio)

    self.img_norm1 = NNXAdaLayerNormZero(dim, dtype=dtype, weights_dtype=weights_dtype)
    self.txt_norm1 = NNXAdaLayerNormZero(dim, dtype=dtype, weights_dtype=weights_dtype)

    self.attn = NNXFluxDoubleAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    self.img_mlp = nnx.Linear(
        in_features=dim,
        out_features=mlp_hidden_dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.img_mlp_out = nnx.Linear(
        in_features=mlp_hidden_dim,
        out_features=dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.txt_mlp = nnx.Linear(
        in_features=dim,
        out_features=mlp_hidden_dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.txt_mlp_out = nnx.Linear(
        in_features=mlp_hidden_dim,
        out_features=dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      image_rotary_emb: Tuple[jax.Array, jax.Array],
      temb_mod_img: Optional[jax.Array] = None,
      temb_mod_txt: Optional[jax.Array] = None,
  ) -> Tuple[jax.Array, jax.Array]:
    norm_h, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.img_norm1(hidden_states, emb=temb_mod_img)
    norm_enc, c_gate_msa_txt, c_shift_mlp_txt, c_scale_mlp_txt, c_gate_mlp_txt = self.txt_norm1(
        encoder_hidden_states, emb=temb_mod_txt
    )

    attn_img, attn_txt = self.attn(
        hidden_states=norm_h,
        encoder_hidden_states=norm_enc,
        image_rotary_emb=image_rotary_emb,
    )

    hidden_states = hidden_states + c_gate_msa * attn_img
    encoder_hidden_states = encoder_hidden_states + c_gate_msa_txt * attn_txt

    norm_h_mlp = norm_h * (1.0 + c_scale_mlp) + c_shift_mlp
    norm_enc_mlp = norm_enc * (1.0 + c_scale_mlp_txt) + c_shift_mlp_txt

    img_ff = self.img_mlp_out(jax.nn.gelu(self.img_mlp(norm_h_mlp), approximate=True))
    txt_ff = self.txt_mlp_out(jax.nn.gelu(self.txt_mlp(norm_enc_mlp), approximate=True))

    hidden_states = hidden_states + c_gate_mlp * img_ff
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp_txt * txt_ff

    return encoder_hidden_states, hidden_states


class NNXFluxSingleTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.dim = dim
    self.norm = NNXAdaLayerNormZeroSingle(dim, dtype=dtype, weights_dtype=weights_dtype)
    self.attn = NNXFluxSingleAttention(
        rngs=rngs,
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      temb: jax.Array,
      image_rotary_emb: Tuple[jax.Array, jax.Array],
      temb_mod: Optional[jax.Array] = None,
  ) -> jax.Array:
    norm_hidden_states, gate_msa = self.norm(hidden_states, emb=temb_mod)
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = hidden_states + gate_msa * attn_output
    return hidden_states


class NNXFluxTransformer2DModel(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      patch_size: int = 1,
      in_channels: int = 64,
      num_layers: int = 5,
      num_single_layers: int = 20,
      attention_head_dim: int = 128,
      num_attention_heads: int = 24,
      joint_attention_dim: int = 4096,
      pooled_projection_dim: int = 768,
      guidance_embeds: bool = True,
      axes_dim: Tuple[int, ...] = (16, 56, 56),
      theta: float = 10000.0,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.in_channels = in_channels
    self.out_channels = in_channels
    self.patch_size = patch_size
    self.num_layers = num_layers
    self.num_single_layers = num_single_layers
    self.attention_head_dim = attention_head_dim
    self.num_attention_heads = num_attention_heads
    self.inner_dim = num_attention_heads * attention_head_dim
    self.dtype = dtype

    self.pos_embed = NNXFluxPosEmbed(axes_dim=axes_dim, theta=theta, return_tuple=True)
    self.time_text_embed = NNXCombinedTimestepGuidanceTextProjEmbeddings(
        rngs=rngs,
        embedding_dim=self.inner_dim,
        pooled_projection_dim=pooled_projection_dim,
        guidance_embeds=guidance_embeds,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    self.double_stream_modulation_img = nnx.Linear(
        in_features=self.inner_dim,
        out_features=6 * self.inner_dim,
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.double_stream_modulation_txt = nnx.Linear(
        in_features=self.inner_dim,
        out_features=6 * self.inner_dim,
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.single_stream_modulation = nnx.Linear(
        in_features=self.inner_dim,
        out_features=3 * self.inner_dim,
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

    self.x_embedder = nnx.Linear(
        in_features=in_channels,
        out_features=self.inner_dim,
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.context_embedder = nnx.Linear(
        in_features=joint_attention_dim,
        out_features=self.inner_dim,
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

    self.double_blocks = nnx.List(
        [
            NNXFluxDoubleTransformerBlock(
                rngs=rngs,
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
            for _ in range(num_layers)
        ]
    )

    self.single_blocks = nnx.List(
        [
            NNXFluxSingleTransformerBlock(
                rngs=rngs,
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
            for _ in range(num_single_layers)
        ]
    )

    self.norm_out = NNXAdaLayerNormContinuous(
        rngs=rngs,
        embedding_dim=self.inner_dim,
        eps=1e-6,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    self.proj_out = nnx.Linear(
        in_features=self.inner_dim,
        out_features=in_channels,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      pooled_projections: jax.Array,
      timestep: jax.Array,
      img_ids: jax.Array,
      txt_ids: jax.Array,
      guidance: Optional[jax.Array] = None,
  ) -> jax.Array:
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep * 1000.0
    if guidance is not None:
      guidance = guidance * 1000.0
    temb = self.time_text_embed(timestep, guidance, pooled_projections)
    temb = temb.astype(hidden_states.dtype)

    temb_silu = jax.nn.silu(temb)
    double_stream_mod_img = self.double_stream_modulation_img(temb_silu)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb_silu)
    single_stream_mod = self.single_stream_modulation(temb_silu)

    if encoder_hidden_states is not None:
      encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        jnp.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        jnp.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    for double_block in self.double_blocks:
      encoder_hidden_states, hidden_states = double_block(
          hidden_states=hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          temb=temb,
          image_rotary_emb=concat_rotary_emb,
          temb_mod_img=double_stream_mod_img,
          temb_mod_txt=double_stream_mod_txt,
      )

    num_txt_tokens = encoder_hidden_states.shape[1]
    hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

    for single_block in self.single_blocks:
      hidden_states = single_block(
          hidden_states=hidden_states,
          temb=temb,
          image_rotary_emb=concat_rotary_emb,
          temb_mod=single_stream_mod,
      )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    return output
