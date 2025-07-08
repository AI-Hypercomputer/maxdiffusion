# Copyright 2025 Lightricks Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Lightricks/LTX-Video/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the Torch version available at:
# https://github.com/Lightricks/LTX-Video/tree/main
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from maxdiffusion.models.ltx_video.linear import DenseGeneral
from maxdiffusion.models.ltx_video.transformers.adaln import AdaLayerNormSingle
from maxdiffusion.models.ltx_video.transformers.attention import BasicTransformerBlock
from maxdiffusion.models.ltx_video.transformers.caption_projection import CaptionProjection
from maxdiffusion.models.ltx_video.gradient_checkpoint import GradientCheckpointType
from maxdiffusion.models.ltx_video.repeatable_layer import RepeatableLayer


class Transformer3DModel(nn.Module):
  num_attention_heads: int = 16
  attention_head_dim: int = 88
  out_channels: int = 128
  num_layers: int = 1
  dropout: float = 0.0
  cross_attention_dim: Optional[int] = None
  attention_bias: bool = False
  activation_fn: str = "geglu"
  num_embeds_ada_norm: Optional[int] = None
  only_cross_attention: bool = False
  double_self_attention: bool = False
  upcast_attention: bool = False
  adaptive_norm: str = "single_scale_shift"  # 'single_scale_shift' or 'single_scale'
  standardization_norm: str = "layer_norm"  # 'layer_norm' or 'rms_norm'
  norm_elementwise_affine: bool = True
  norm_eps: float = 1e-5
  attention_type: str = "default"
  caption_channels: int = None
  use_tpu_flash_attention: bool = True  # if True uses the TPU attention offload ('flash attention')
  qk_norm: Optional[str] = None
  positional_embedding_type: str = "rope"
  positional_embedding_theta: Optional[float] = None
  positional_embedding_max_pos: Optional[List[int]] = None
  timestep_scale_multiplier: Optional[float] = None
  ffn_dim_mult: Optional[int] = 4
  output_scale: Optional[float] = None
  attention_op: Optional[nn.Module] = None
  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"
  sharding_mesh: Optional[jax.sharding.Mesh] = None
  param_scan_axis: int = 0
  gradient_checkpointing: Optional[str] = None

  def setup(self):
    assert self.out_channels is not None, "out channels must be specified in model config."
    self.inner_dim = self.num_attention_heads * self.attention_head_dim
    self.patchify_proj = DenseGeneral(
        self.inner_dim,
        use_bias=True,
        kernel_axes=(None, "embed"),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="patchify_proj",
    )
    self.freq_cis_pre_computer = FreqsCisPrecomputer(
        self.positional_embedding_max_pos, self.positional_embedding_theta, self.inner_dim
    )
    self.adaln_single = AdaLayerNormSingle(
        self.inner_dim,
        embedding_coefficient=4 if self.adaptive_norm == "single_scale" else 6,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
    )

    def scale_shift_table_init(key):
      return jax.random.normal(key, (2, self.inner_dim)) / self.inner_dim**0.5

    self.scale_shift_table = self.param(
        "scale_shift_table",  # Trainable parameter name
        nn.with_logical_partitioning(scale_shift_table_init, ("ada", "embed")),
    )
    self.norm_out = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
    self.proj_out = DenseGeneral(
        self.out_channels,
        use_bias=True,
        kernel_axes=("embed", None),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="proj_out",
    )
    self.use_rope = self.positional_embedding_type == "rope"
    if self.num_layers > 0:
      RemattedBasicTransformerBlock = GradientCheckpointType.from_str(self.gradient_checkpointing).apply(
          BasicTransformerBlock
      )

      self.transformer_blocks = RepeatableLayer(
          RemattedBasicTransformerBlock,
          num_layers=self.num_layers,
          module_init_kwargs=dict(  # noqa C408
              dim=self.inner_dim,
              num_attention_heads=self.num_attention_heads,
              attention_head_dim=self.attention_head_dim,
              dropout=self.dropout,
              cross_attention_dim=self.cross_attention_dim,
              activation_fn=self.activation_fn,
              num_embeds_ada_norm=self.num_embeds_ada_norm,
              attention_bias=self.attention_bias,
              only_cross_attention=self.only_cross_attention,
              double_self_attention=self.double_self_attention,
              upcast_attention=self.upcast_attention,
              adaptive_norm=self.adaptive_norm,
              standardization_norm=self.standardization_norm,
              norm_elementwise_affine=self.norm_elementwise_affine,
              norm_eps=self.norm_eps,
              attention_type=self.attention_type,
              use_tpu_flash_attention=self.use_tpu_flash_attention,
              qk_norm=self.qk_norm,
              use_rope=self.use_rope,
              ffn_dim_mult=self.ffn_dim_mult,
              attention_op=self.attention_op,
              dtype=self.dtype,
              weight_dtype=self.weight_dtype,
              matmul_precision=self.matmul_precision,
              sharding_mesh=self.sharding_mesh,
              name="CheckpointBasicTransformerBlock_0",
          ),
          pspec_name="layers",
          param_scan_axis=self.param_scan_axis,
      )

    if self.caption_channels is not None:
      self.caption_projection = CaptionProjection(
          in_features=self.caption_channels,
          hidden_size=self.inner_dim,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          matmul_precision=self.matmul_precision,
      )

  def init_weights(self, in_channels, key, caption_channels, eval_only=True):
    example_inputs = {}
    batch_size, num_tokens = 4, 256
    input_shapes = {
        "hidden_states": (batch_size, num_tokens, in_channels),
        "indices_grid": (batch_size, 3, num_tokens),
        "encoder_hidden_states": (batch_size, 128, caption_channels),
        "timestep": (batch_size, 256),
        "segment_ids": (batch_size, 256),
        "encoder_attention_segment_ids": (batch_size, 128),
    }
    for name, shape in input_shapes.items():
      example_inputs[name] = jnp.ones(
          shape, dtype=jnp.float32 if name not in ["attention_mask", "encoder_attention_mask"] else jnp.bool
      )

    if eval_only:
      return jax.eval_shape(
          self.init,
          key,
          **example_inputs,
      )["params"]
    else:
      return self.init(key, **example_inputs)["params"]

  def __call__(
      self,
      hidden_states,
      indices_grid,
      encoder_hidden_states=None,
      timestep=None,
      class_labels=None,
      cross_attention_kwargs=None,
      segment_ids=None,
      encoder_attention_segment_ids=None,
      return_dict=True,
  ):
    hidden_states = self.patchify_proj(hidden_states)
    freqs_cis = self.freq_cis_pre_computer(indices_grid)

    if self.timestep_scale_multiplier:
      timestep = self.timestep_scale_multiplier * timestep

    batch_size = hidden_states.shape[0]

    timestep, embedded_timestep = self.adaln_single(
        timestep,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )

    if self.caption_projection is not None:
      encoder_hidden_states = self.caption_projection(encoder_hidden_states)

    if self.num_layers > 0:
      hidden_states = self.transformer_blocks(
          hidden_states,
          freqs_cis,
          segment_ids,
          encoder_hidden_states,
          encoder_attention_segment_ids,
          timestep,
          cross_attention_kwargs,
          class_labels,
      )
    # Output processing

    scale_shift_values = self.scale_shift_table[jnp.newaxis, jnp.newaxis, :, :] + embedded_timestep[:, :, jnp.newaxis]
    scale_shift_values = nn.with_logical_constraint(
        scale_shift_values, ("activation_batch", "activation_length", "activation_ada", "activation_embed")
    )
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.proj_out(hidden_states)
    if self.output_scale:
      hidden_states = hidden_states / self.output_scale

    return hidden_states


def log_base(x: jax.Array, base: jax.Array) -> jax.Array:
  """
  Computes log of x with defined base.

  Args:
      x (jax.Array): log value
      base (jax.Array):  base of the log

  Returns:
      jax.Array: log(x)[base]
  """
  return jnp.log(x) / jnp.log(base)


class FreqsCisPrecomputer(nn.Module):
  """
  computes frequency components (cosine and sine embeddings) for positional encodings based on fractional positions.
  This is commonly used in rotary embeddings (RoPE) for transformers.
  """

  positional_embedding_max_pos: List[int]
  positional_embedding_theta: float
  inner_dim: int

  def get_fractional_positions(self, indices_grid: jax.Array) -> jax.Array:
    fractional_positions = jnp.stack(
        [indices_grid[:, i] / self.positional_embedding_max_pos[i] for i in range(3)],
        axis=-1,
    )
    return fractional_positions

  @nn.compact
  def __call__(self, indices_grid: jax.Array) -> Tuple[jax.Array, jax.Array]:
    source_dtype = indices_grid.dtype
    dtype = jnp.float32  # We need full precision in the freqs_cis computation.
    dim = self.inner_dim
    theta = self.positional_embedding_theta

    fractional_positions = self.get_fractional_positions(indices_grid)

    start = 1
    end = theta
    indices = jnp.power(
        theta,
        jnp.linspace(
            log_base(start, theta),
            log_base(end, theta),
            dim // 6,
            dtype=dtype,
        ),
    )
    indices = indices.astype(dtype)

    indices = indices * jnp.pi / 2

    freqs = (indices * (jnp.expand_dims(fractional_positions, axis=-1) * 2 - 1)).swapaxes(-1, -2)
    freqs = freqs.reshape(freqs.shape[0], freqs.shape[1], -1)  # Flatten along axis 2

    cos_freq = jnp.cos(freqs).repeat(2, axis=-1)
    sin_freq = jnp.sin(freqs).repeat(2, axis=-1)

    if dim % 6 != 0:
      cos_padding = jnp.ones_like(cos_freq[:, :, : dim % 6])
      sin_padding = jnp.zeros_like(sin_freq[:, :, : dim % 6])

      cos_freq = jnp.concatenate([cos_padding, cos_freq], axis=-1)
      sin_freq = jnp.concatenate([sin_padding, sin_freq], axis=-1)
    return cos_freq.astype(source_dtype), sin_freq.astype(source_dtype)
