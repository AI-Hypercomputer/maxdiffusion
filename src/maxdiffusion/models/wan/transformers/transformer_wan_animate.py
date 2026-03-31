"""
Copyright 2026 Google LLC

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

from typing import Tuple, Optional, Dict, Union, Any
import contextlib
import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
from .... import common_types
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin, register_to_config
from ...normalization_flax import FP32LayerNorm
from ...gradient_checkpoint import GradientCheckpointType
from .transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)

BlockSizes = common_types.BlockSizes

WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


class FlaxFusedLeakyReLU(nnx.Module):
  """
  Fused LeakyRelu with scale factor and channel-wise bias.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      negative_slope: float = 0.2,
      scale: float = 2**0.5,
      bias_channels: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.negative_slope = negative_slope
    self.scale = scale
    self.channels = bias_channels
    self.dtype = dtype
    self.weights_dtype = weights_dtype

    if self.channels is not None:
      self.bias = nnx.Param(jnp.zeros((self.channels,), dtype=self.weights_dtype))
    else:
      self.bias = None

  def __call__(self, x: jax.Array, channel_dim: int = 1) -> jax.Array:
    if self.bias is not None:
      # Expand self.bias to have all singleton dims except at channel_dim
      expanded_shape = [1] * x.ndim
      expanded_shape[channel_dim] = self.channels
      bias = jnp.reshape(self.bias, expanded_shape)
      x = x + bias
    x = jax.nn.leaky_relu(x, self.negative_slope) * self.scale
    return x


class FlaxMotionConv2d(nnx.Module):
  """2-D convolution with EqualizedLR scaling and optional FusedLeakyReLU.

  Weights are stored in PyTorch OIHW format (out, in, k, k) as raw nnx.Param
  so that the weight-loading code in wan_utils.py can map them without
  transposing.  No sharding annotations are applied because this module is
  part of the small motion encoder network.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
      bias: bool = True,
      blur_kernel: Optional[Tuple[int, ...]] = None,
      blur_upsample_factor: int = 1,
      use_activation: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.use_activation = use_activation
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding_size = padding
    self.dtype = dtype
    self.weights_dtype = weights_dtype

    self.blur = False
    if blur_kernel is not None:
      p = (len(blur_kernel) - stride) + (kernel_size - 1)
      self.blur_padding = ((p + 1) // 2, p // 2)

      kernel = np.asarray(blur_kernel, dtype=np.float32)
      if kernel.ndim == 1:
        kernel = np.expand_dims(kernel, 0) * np.expand_dims(kernel, 1)
      kernel = kernel / kernel.sum()

      if blur_upsample_factor > 1:
        kernel = kernel * (blur_upsample_factor**2)

      self.blur_kernel = nnx.static(tuple(tuple(float(v) for v in row) for row in kernel))
      self.blur = True
    else:
      self.blur_kernel = nnx.static(None)

    key = rngs.params()
    # Shape: (out_channels, in_channels, kernel, kernel) — PyTorch OIHW format.
    self.weight = nnx.Param(
        jax.random.normal(key, (out_channels, in_channels, kernel_size, kernel_size), dtype=weights_dtype)
    )
    self.scale = 1.0 / math.sqrt(in_channels * kernel_size**2)

    if bias and not self.use_activation:
      self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=weights_dtype))
    else:
      self.bias = None

    if self.use_activation:
      self.act_fn = FlaxFusedLeakyReLU(
          rngs=rngs, bias_channels=out_channels, dtype=dtype, weights_dtype=weights_dtype
      )
    else:
      self.act_fn = None

  def __call__(self, x: jax.Array, channel_dim: int = 1) -> jax.Array:
    # 1. Blur Pass (Depthwise)
    if self.blur:
      blur_kernel = jnp.asarray(self.blur_kernel, dtype=jnp.float32)
      expanded_kernel = jnp.expand_dims(jnp.expand_dims(blur_kernel, 0), 0)
      expanded_kernel = jnp.broadcast_to(
          expanded_kernel,
          (
              self.in_channels,
              1,
              expanded_kernel.shape[2],
              expanded_kernel.shape[3],
          ),
      )
      x = x.astype(expanded_kernel.dtype)

      pad_h, pad_w = self.blur_padding
      x = jax.lax.conv_general_dilated(
          x,
          expanded_kernel,
          window_strides=(1, 1),
          padding=[(pad_h, pad_h), (pad_w, pad_w)],
          dimension_numbers=("NCHW", "OIHW", "NCHW"),
          feature_group_count=self.in_channels,
      )

    # 2. Main Convolution Pass
    x = x.astype(self.weight.dtype)
    conv_weight = self.weight * self.scale
    x = jax.lax.conv_general_dilated(
        x,
        conv_weight,
        window_strides=(self.stride, self.stride),
        padding=[
            (self.padding_size, self.padding_size),
            (self.padding_size, self.padding_size),
        ],
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

    # 3. Bias and Activation
    if self.bias is not None:
      b = jnp.reshape(self.bias, (1, self.out_channels, 1, 1))
      x = x + b

    if self.use_activation:
      x = self.act_fn(x, channel_dim=channel_dim)

    return x


class FlaxMotionLinear(nnx.Module):
  """Equalized-LR linear layer with optional FusedLeakyReLU.

  Weights are stored in PyTorch (out, in) format as raw nnx.Param — same
  reason as FlaxMotionConv2d.  No sharding annotations needed (small layer).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_dim: int,
      out_dim: int,
      bias: bool = True,
      use_activation: bool = False,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.use_activation = use_activation
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.dtype = dtype
    self.weights_dtype = weights_dtype

    key = rngs.params()
    self.weight = nnx.Param(jax.random.normal(key, (out_dim, in_dim), dtype=weights_dtype))
    self.scale = 1.0 / math.sqrt(in_dim)

    if bias and not self.use_activation:
      self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=weights_dtype))
    else:
      self.bias = None

    if self.use_activation:
      self.act_fn = FlaxFusedLeakyReLU(rngs=rngs, bias_channels=out_dim, dtype=dtype, weights_dtype=weights_dtype)
    else:
      self.act_fn = None

  def __call__(self, inputs: jax.Array, channel_dim: int = 1) -> jax.Array:
    inputs = inputs.astype(self.weight.dtype)
    # Transpose to (in_dim, out_dim) and apply scale
    w = self.weight.T * self.scale

    out = inputs @ w

    if self.bias is not None:
      out = out + self.bias

    if self.use_activation:
      out = self.act_fn(out, channel_dim=channel_dim)

    return out


class FlaxMotionEncoderResBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      kernel_size_skip: int = 1,
      blur_kernel: Tuple[int, ...] = (1, 3, 3, 1),
      downsample_factor: int = 2,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.downsample_factor = downsample_factor
    self.dtype = dtype

    # 3 X 3 Conv + fused leaky ReLU
    self.conv1 = FlaxMotionConv2d(
        rngs,
        in_channels,
        in_channels,
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        use_activation=True,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    # 3 X 3 Conv + downsample 2x + fused leaky ReLU
    self.conv2 = FlaxMotionConv2d(
        rngs,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=self.downsample_factor,
        padding=0,
        blur_kernel=blur_kernel,
        use_activation=True,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    # 1 X 1 Conv + downsample 2x in skip connection
    self.conv_skip = FlaxMotionConv2d(
        rngs,
        in_channels,
        out_channels,
        kernel_size=kernel_size_skip,
        stride=self.downsample_factor,
        padding=0,
        bias=False,
        blur_kernel=blur_kernel,
        use_activation=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

  def __call__(self, x: jax.Array, channel_dim: int = 1) -> jax.Array:
    x_out = self.conv1(x, channel_dim=channel_dim)
    x_out = self.conv2(x_out, channel_dim=channel_dim)

    x_skip = self.conv_skip(x, channel_dim=channel_dim)

    x_out = (x_out + x_skip) / math.sqrt(2.0)
    return x_out


class FlaxWanAnimateMotionEncoder(nnx.Module):
  """Encodes a face video frame into a motion vector.

  All weights in this network are small (the largest is 32×512→16) so
  sharding annotations are not applied.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      size: int = 512,
      style_dim: int = 512,
      motion_dim: int = 20,
      out_dim: int = 512,
      motion_blocks: int = 5,
      channels: Optional[Dict[str, int]] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.size = size
    self.dtype = dtype
    self.weights_dtype = weights_dtype

    if channels is None:
      channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

    self.conv_in = FlaxMotionConv2d(
        rngs, 3, channels[str(size)], 1, use_activation=True, dtype=dtype, weights_dtype=weights_dtype
    )

    res_blocks = []
    in_channels = channels[str(size)]
    log_size = int(math.log(size, 2))
    for i in range(log_size, 2, -1):
      out_channels = channels[str(2 ** (i - 1))]
      res_blocks.append(
          FlaxMotionEncoderResBlock(rngs, in_channels, out_channels, dtype=dtype, weights_dtype=weights_dtype)
      )
      in_channels = out_channels
    self.res_blocks = nnx.List(res_blocks)

    self.conv_out = FlaxMotionConv2d(
        rngs,
        in_channels,
        style_dim,
        4,
        padding=0,
        bias=False,
        use_activation=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    linears = []
    for _ in range(motion_blocks - 1):
      linears.append(FlaxMotionLinear(rngs, style_dim, style_dim, dtype=dtype, weights_dtype=weights_dtype))

    linears.append(FlaxMotionLinear(rngs, style_dim, motion_dim, dtype=dtype, weights_dtype=weights_dtype))
    self.motion_network = nnx.List(linears)

    key = rngs.params()
    self.motion_synthesis_weight = nnx.Param(jax.random.normal(key, (out_dim, motion_dim), dtype=weights_dtype))

  def __call__(self, face_image: jax.Array, channel_dim: int = 1) -> jax.Array:
    if face_image.shape[-2] != self.size or face_image.shape[-1] != self.size:
      raise ValueError(f"Expected {self.size} got {face_image.shape[-1]}")

    x = self.conv_in(face_image, channel_dim=channel_dim)
    for block in self.res_blocks:
      x = block(x, channel_dim=channel_dim)
    x = self.conv_out(x, channel_dim=channel_dim)

    motion_feat = jnp.squeeze(x, axis=(-1, -2))

    for linear_layer in self.motion_network:
      motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

    weight = self.motion_synthesis_weight[...] + 1e-8

    original_dtype = motion_feat.dtype
    motion_feat_fp32 = motion_feat.astype(jnp.float32)
    weight_fp32 = weight.astype(jnp.float32)

    Q, _ = jnp.linalg.qr(weight_fp32)

    motion_vec = jnp.matmul(motion_feat_fp32, jnp.transpose(Q, (1, 0)))

    return motion_vec.astype(original_dtype)


class FlaxWanAnimateFaceEncoder(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_dim: int,
      out_dim: int,
      hidden_dim: int = 1024,
      num_heads: int = 4,
      kernel_size: int = 3,
      eps: float = 1e-6,
      pad_mode: str = "edge",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.num_heads = num_heads
    self.kernel_size = kernel_size
    self.pad_mode = pad_mode
    self.out_dim = out_dim
    self.dtype = dtype

    self.act = jax.nn.silu

    self.conv1_local = nnx.Conv(
        in_dim,
        hidden_dim * num_heads,
        kernel_size=(kernel_size,),
        strides=(1,),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
    )
    self.conv2 = nnx.Conv(
        hidden_dim,
        hidden_dim,
        kernel_size=(kernel_size,),
        strides=(2,),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
    )
    self.conv3 = nnx.Conv(
        hidden_dim,
        hidden_dim,
        kernel_size=(kernel_size,),
        strides=(2,),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
    )

    self.norm1 = nnx.LayerNorm(
        hidden_dim,
        epsilon=eps,
        use_bias=False,
        use_scale=False,
        rngs=rngs,
        dtype=dtype,
    )
    self.norm2 = nnx.LayerNorm(
        hidden_dim,
        epsilon=eps,
        use_bias=False,
        use_scale=False,
        rngs=rngs,
        dtype=dtype,
    )
    self.norm3 = nnx.LayerNorm(
        hidden_dim,
        epsilon=eps,
        use_bias=False,
        use_scale=False,
        rngs=rngs,
        dtype=dtype,
    )

    # hidden_dim (mlp) → out_dim (embed): ("mlp", "embed")
    self.out_proj = nnx.Linear(
        hidden_dim,
        out_dim,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
    )

    self.padding_tokens = nnx.Param(jnp.zeros((1, 1, 1, out_dim), dtype=weights_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    batch_size = x.shape[0]

    # Local attention via causal convolution
    x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)), mode=self.pad_mode)
    x = self.conv1_local(x)

    x = jnp.reshape(x, (batch_size, x.shape[1], self.num_heads, -1))
    x = jnp.transpose(x, (0, 2, 1, 3))
    x = jnp.reshape(x, (batch_size * self.num_heads, x.shape[2], x.shape[3]))

    x = self.norm1(x)
    x = self.act(x)

    x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)), mode=self.pad_mode)
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.act(x)

    x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)), mode=self.pad_mode)
    x = self.conv3(x)
    x = self.norm3(x)
    x = self.act(x)

    x = self.out_proj(x)

    x = jnp.reshape(x, (batch_size, self.num_heads, x.shape[1], x.shape[2]))
    x = jnp.transpose(x, (0, 2, 1, 3))

    padding = jnp.broadcast_to(self.padding_tokens[...], (batch_size, x.shape[1], 1, self.out_dim))
    x = jnp.concatenate([x, padding], axis=2)

    return x


class FlaxWanAnimateFaceBlockCrossAttention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      heads: int = 8,
      dim_head: int = 64,
      eps: float = 1e-6,
      cross_attention_dim_head: Optional[int] = None,
      use_bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.heads = heads
    self.inner_dim = dim_head * heads
    self.cross_attention_dim_head = cross_attention_dim_head
    self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
    self.dtype = dtype

    self.pre_norm_q = nnx.LayerNorm(dim, epsilon=eps, use_bias=False, use_scale=False, rngs=rngs, dtype=dtype)
    self.pre_norm_kv = nnx.LayerNorm(dim, epsilon=eps, use_bias=False, use_scale=False, rngs=rngs, dtype=dtype)

    # embed → heads
    self.to_q = nnx.Linear(
        dim,
        self.inner_dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
    )
    self.to_k = nnx.Linear(
        dim,
        self.kv_inner_dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
    )
    self.to_v = nnx.Linear(
        dim,
        self.kv_inner_dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
    )

    # heads → embed
    self.to_out = nnx.Linear(
        self.inner_dim,
        dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
    )

    self.norm_q = nnx.RMSNorm(dim_head, epsilon=eps, use_scale=True, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
    self.norm_k = nnx.RMSNorm(dim_head, epsilon=eps, use_scale=True, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      attention_mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    hidden_states = self.pre_norm_q(hidden_states)
    encoder_hidden_states = self.pre_norm_kv(encoder_hidden_states)

    B, T, N, C = encoder_hidden_states.shape

    query = self.to_q(hidden_states)
    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    # Reshape to extract heads
    query = jnp.reshape(query, (query.shape[0], query.shape[1], self.heads, -1))
    key = jnp.reshape(key, (B, T, N, self.heads, -1))
    value = jnp.reshape(value, (B, T, N, self.heads, -1))

    query = self.norm_q(query)
    key = self.norm_k(key)

    query_S = query.shape[1]

    # Fold Time into the Batch dimension for attention
    query = jnp.reshape(query, (B * T, query_S // T, self.heads, -1))
    key = jnp.reshape(key, (B * T, N, self.heads, -1))
    value = jnp.reshape(value, (B * T, N, self.heads, -1))

    attn_output = jax.nn.dot_product_attention(query, key, value)

    # Restore (Batch, Total Sequence, Dim)
    attn_output = jnp.reshape(attn_output, (B, query_S, -1))

    hidden_states = self.to_out(attn_output)

    if attention_mask is not None:
      attention_mask = jnp.reshape(attention_mask, (attention_mask.shape[0], -1))
      hidden_states = hidden_states * jnp.expand_dims(attention_mask, axis=-1)

    return hidden_states


class NNXWanAnimateTransformer3DModel(nnx.Module, FlaxModelMixin, ConfigMixin):

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size: Tuple[int, int, int] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 36,
      latent_channels: int = 16,
      out_channels: Optional[int] = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      eps: float = 1e-6,
      image_dim: Optional[int] = 1280,
      added_kv_proj_dim: Optional[int] = None,
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
      motion_encoder_channel_sizes: Optional[Dict[str, int]] = None,
      motion_encoder_size: int = 512,
      motion_style_dim: int = 512,
      motion_dim: int = 20,
      motion_encoder_dim: int = 512,
      face_encoder_hidden_dim: int = 1024,
      face_encoder_num_heads: int = 4,
      inject_face_latents_blocks: int = 5,
      motion_encoder_batch_size: int = 8,
  ):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or latent_channels

    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.patch_size = patch_size
    self.inject_face_latents_blocks = inject_face_latents_blocks
    self.motion_encoder_batch_size = motion_encoder_batch_size
    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)

    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

    # Patch embeddings — shard output (conv_out) axis across model parallelism.
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        kernel_size=patch_size,
        strides=patch_size,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )
    self.pose_patch_embedding = nnx.Conv(
        latent_channels,
        inner_dim,
        kernel_size=patch_size,
        strides=patch_size,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
        flash_min_seq_length=flash_min_seq_length,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    self.motion_encoder = FlaxWanAnimateMotionEncoder(
        rngs=rngs,
        size=motion_encoder_size,
        style_dim=motion_style_dim,
        motion_dim=motion_dim,
        out_dim=motion_encoder_dim,
        channels=motion_encoder_channel_sizes,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    self.face_encoder = FlaxWanAnimateFaceEncoder(
        rngs=rngs,
        in_dim=motion_encoder_dim,
        out_dim=inner_dim,
        hidden_dim=face_encoder_hidden_dim,
        num_heads=face_encoder_num_heads,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

    blocks = []
    for _ in range(num_layers):
      block = WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          added_kv_proj_dim=added_kv_proj_dim,
          image_seq_len=image_seq_len,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          enable_jax_named_scopes=enable_jax_named_scopes,
      )
      blocks.append(block)
    self.blocks = nnx.List(blocks)

    face_adapters = []
    num_face_adapters = num_layers // inject_face_latents_blocks
    for _ in range(num_face_adapters):
      fa = FlaxWanAnimateFaceBlockCrossAttention(
          rngs=rngs,
          dim=inner_dim,
          heads=num_attention_heads,
          dim_head=inner_dim // num_attention_heads,
          eps=eps,
          cross_attention_dim_head=inner_dim // num_attention_heads,
          dtype=dtype,
          weights_dtype=weights_dtype,
      )
      face_adapters.append(fa)
    self.face_adapter = nnx.List(face_adapters)

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)

    # Final projection — embed → output tokens.
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
    )

    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim), dtype=weights_dtype) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "embed")),
    )

  def conditional_named_scope(self, name: str):
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  @jax.named_scope("WanAnimateTransformer3DModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      pose_hidden_states: Optional[jax.Array] = None,
      face_pixel_values: Optional[jax.Array] = None,
      motion_encode_batch_size: Optional[int] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> Union[jax.Array, Dict[str, jax.Array]]:
    if pose_hidden_states is not None and pose_hidden_states.shape[2] + 1 != hidden_states.shape[2]:
      raise ValueError(
          f"Pose frames + 1 ({pose_hidden_states.shape[2]} + 1) must equal hidden_states frames ({hidden_states.shape[2]})"
      )

    # Constrain input to batch-sharded layout before any computation.
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 1 & 2. Rotary Position & Patch Embedding
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    rotary_emb = self.rope(hidden_states)
    hidden_states = self.patch_embedding(hidden_states)

    pose_hidden_states = jnp.transpose(pose_hidden_states, (0, 2, 3, 4, 1))
    pose_hidden_states = self.pose_patch_embedding(pose_hidden_states)
    pose_pad = jnp.zeros(
        (
            batch_size,
            1,
            pose_hidden_states.shape[2],
            pose_hidden_states.shape[3],
            pose_hidden_states.shape[4],
        ),
        dtype=hidden_states.dtype,
    )
    pose_pad = jnp.concatenate([pose_pad, pose_hidden_states], axis=1)
    hidden_states = hidden_states + pose_pad

    hidden_states = jnp.reshape(hidden_states, (batch_size, -1, hidden_states.shape[-1]))

    # 3. Condition Embeddings
    (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        _,
    ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
    timestep_proj = timestep_proj.reshape(batch_size, 6, -1)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)

    # 4. Batched Face & Motion Encoding
    _, face_channels, num_face_frames, face_height, face_width = face_pixel_values.shape

    # Rearrange from (B, C, T, H, W) to (B*T, C, H, W)
    face_pixel_values = jnp.transpose(face_pixel_values, (0, 2, 1, 3, 4))
    face_pixel_values = jnp.reshape(face_pixel_values, (-1, face_channels, face_height, face_width))

    total_face_frames = face_pixel_values.shape[0]
    motion_encode_batch_size = motion_encode_batch_size or self.motion_encoder_batch_size

    # Pad sequence if it doesn't divide evenly by encode_bs
    pad_len = (motion_encode_batch_size - (total_face_frames % motion_encode_batch_size)) % motion_encode_batch_size
    if pad_len > 0:
      pad_tensor = jnp.zeros(
          (pad_len, face_channels, face_height, face_width),
          dtype=face_pixel_values.dtype,
      )
      face_pixel_values = jnp.concatenate([face_pixel_values, pad_tensor], axis=0)

    # Reshape into chunks for scan
    num_chunks = face_pixel_values.shape[0] // motion_encode_batch_size
    face_chunks = jnp.reshape(
        face_pixel_values,
        (
            num_chunks,
            motion_encode_batch_size,
            face_channels,
            face_height,
            face_width,
        ),
    )

    # Use jax.lax.scan to iterate over chunks to save memory
    def encode_chunk_fn(carry, chunk):
      encoded_chunk = self.motion_encoder(chunk)
      return carry, encoded_chunk

    _, motion_vec_chunks = jax.lax.scan(encode_chunk_fn, None, face_chunks)
    motion_vec = jnp.reshape(motion_vec_chunks, (-1, motion_vec_chunks.shape[-1]))

    # Remove padding if added
    if pad_len > 0:
      motion_vec = motion_vec[:-pad_len]

    motion_vec = jnp.reshape(motion_vec, (batch_size, num_face_frames, -1))

    # Apply face encoder
    motion_vec = self.face_encoder(motion_vec)
    pad_face = jnp.zeros_like(motion_vec[:, :1])
    motion_vec = jnp.concatenate([pad_face, motion_vec], axis=1)

    # 5. Transformer Blocks
    for block_idx, block in enumerate(self.blocks):
      hidden_states = block(
          hidden_states,
          encoder_hidden_states,
          timestep_proj,
          rotary_emb,
          deterministic,
          rngs,
      )

      # Face adapter integration: apply after every inject_face_latents_blocks-th block
      if motion_vec is not None and block_idx % self.inject_face_latents_blocks == 0:
        face_adapter_block_idx = block_idx // self.inject_face_latents_blocks
        face_adapter_output = self.face_adapter[face_adapter_block_idx](hidden_states, motion_vec)
        hidden_states = hidden_states + face_adapter_output

    # 6. Output Norm & Projection
    shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = jnp.reshape(
        hidden_states,
        (
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        ),
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = jnp.reshape(hidden_states, (batch_size, -1, num_frames, height, width))

    if not return_dict:
      return (hidden_states,)
    return {"sample": hidden_states}
