from functools import partial
import math
from typing import Any, Dict, Optional, Tuple
from enum import Enum, auto

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as jax_flash_attention,
    SegmentIds,
    BlockSizes,
)

from flax import linen as nn

from maxdiffusion.models.ltx_video.linear import DenseGeneral, Initializer
from maxdiffusion.models.ltx_video.transformers.activations import (
    GELU,
    GEGLU,
    ApproximateGELU,
)


class SkipLayerStrategy(Enum):
  AttentionSkip = auto()
  AttentionValues = auto()
  Residual = auto()
  TransformerBlock = auto()


class Identity(nn.Module):

  def __call__(self, x):
    return x


class BasicTransformerBlock(nn.Module):
  dim: int
  num_attention_heads: int
  attention_head_dim: int
  dropout: float = 0.0
  cross_attention_dim: Optional[int] = None
  activation_fn: str = "geglu"
  num_embeds_ada_norm: Optional[int] = None
  attention_bias: bool = False
  only_cross_attention: bool = False
  double_self_attention: bool = False
  upcast_attention: bool = False
  norm_elementwise_affine: bool = True
  adaptive_norm: str = "single_scale_shift"
  standardization_norm: str = "layer_norm"
  norm_eps: float = 1e-5
  qk_norm: str = None
  final_dropout: bool = False
  attention_type: str = ("default",)  # pylint: disable=unused-argument
  ff_inner_dim: Optional[int] = None
  ff_bias: bool = True
  attention_out_bias: bool = True
  use_tpu_flash_attention: bool = True
  use_rope: bool = False
  ffn_dim_mult: Optional[int] = 4
  attention_op: Optional[nn.Module] = None
  sharding_mesh: Optional[jax.sharding.Mesh] = None

  dtype: jax.numpy.dtype = jnp.float32
  weight_dtype: jax.numpy.dtype = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    assert self.standardization_norm in ["layer_norm", "rms_norm"]
    assert self.adaptive_norm in ["single_scale_shift", "single_scale", "none"]
    assert self.use_tpu_flash_attention, "Jax version only use tpu_flash attention."

    if self.standardization_norm == "layer_norm":
      make_norm_layer = partial(
          nn.LayerNorm,
          epsilon=self.norm_eps,
          param_dtype=self.weight_dtype,
          dtype=self.dtype,
      )
    else:
      make_norm_layer = partial(
          RMSNorm,
          epsilon=self.norm_eps,
          elementwise_affine=self.norm_elementwise_affine,
          weight_dtype=self.weight_dtype,
          dtype=self.dtype,
          kernel_axes=("norm",),
      )

    # 1. Self-Attn
    self.norm1 = make_norm_layer(name="norm1")
    self.attn1 = Attention(
        query_dim=self.dim,
        heads=self.num_attention_heads,
        dim_head=self.attention_head_dim,
        dropout=self.dropout,
        bias=self.attention_bias,
        cross_attention_dim=self.cross_attention_dim if self.only_cross_attention else None,
        upcast_attention=self.upcast_attention,
        out_bias=self.attention_out_bias,
        use_tpu_flash_attention=self.use_tpu_flash_attention,
        qk_norm=self.qk_norm,
        use_rope=self.use_rope,
        attention_op=self.attention_op,
        name="attn1",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
    )

    # 2. Cross-Attn
    if self.cross_attention_dim is not None or self.double_self_attention:
      self.attn2 = Attention(
          query_dim=self.dim,
          cross_attention_dim=self.cross_attention_dim if not self.double_self_attention else None,
          heads=self.num_attention_heads,
          dim_head=self.attention_head_dim,
          dropout=self.dropout,
          bias=self.attention_bias,
          upcast_attention=self.upcast_attention,
          out_bias=self.attention_out_bias,
          use_tpu_flash_attention=self.use_tpu_flash_attention,
          qk_norm=self.qk_norm,
          use_rope=self.use_rope,
          attention_op=self.attention_op,
          name="attn2",
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
      )
      if self.adaptive_norm == "none":
        self.attn2_norm = make_norm_layer()
    else:
      self.attn2 = None
      self.attn2_norm = None

    self.norm2 = make_norm_layer(name="norm2")
    # 3. Feed-forward
    self.ff = FeedForward(
        self.dim,
        dropout=self.dropout,
        activation_fn=self.activation_fn,
        final_dropout=self.final_dropout,
        inner_dim=self.ff_inner_dim,
        bias=self.ff_bias,
        mult=self.ffn_dim_mult,
        name="ff",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
    )

    # 4. Scale-Shift
    if self.adaptive_norm != "none":
      num_ada_params = 4 if self.adaptive_norm == "single_scale" else 6

      def ada_initalizer(key):
        return jax.random.normal(key, (num_ada_params, self.dim), dtype=self.weight_dtype) / self.dim**0.5

      self.scale_shift_table = self.param(
          "scale_shift_table",  # Trainable parameter name
          nn.with_logical_partitioning(ada_initalizer, ("ada", "embed")),
      )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      freqs_cis: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
      segment_ids: Optional[jnp.ndarray] = None,
      encoder_hidden_states: Optional[jnp.ndarray] = None,
      encoder_attention_segment_ids: Optional[jnp.ndarray] = None,
      timestep: Optional[jnp.ndarray] = None,
      cross_attention_kwargs: Dict[str, Any] = None,
      class_labels: Optional[jnp.ndarray] = None,
      skip_layer_mask: Optional[jnp.ndarray] = None,
      skip_layer_strategy: Optional[SkipLayerStrategy] = None,
  ) -> jnp.ndarray:
    if cross_attention_kwargs is not None:
      if cross_attention_kwargs.get("scale", None) is not None:
        print("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

    hidden_states = nn.with_logical_constraint(
        hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    hidden_states = checkpoint_name(hidden_states, "basic_transformer_block hidden_states")

    batch_size = hidden_states.shape[0]

    # 0. Self-Attention
    norm_hidden_states = self.norm1(hidden_states)

    norm_hidden_states = nn.with_logical_constraint(
        norm_hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    # Adaptive Norm
    if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
      # [batch, 1 or num_tokens, embedding_dim]
      assert timestep.ndim == 3
      num_ada_params = self.scale_shift_table.shape[0]
      ada_values = self.scale_shift_table[None, None].astype(self.weight_dtype) + timestep.reshape(
          batch_size, timestep.shape[1], num_ada_params, -1
      )
      # Moving ada values to computation dtype to prevent dtype promotion
      ada_values = ada_values.astype(self.dtype)
      ada_values = nn.with_logical_constraint(
          ada_values, ("activation_batch", "activation_norm_length", "activation_ada", "activation_embed")
      )

      if self.adaptive_norm == "single_scale_shift":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.squeeze(arr, axis=2) for arr in jnp.split(ada_values, 6, axis=2)
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
      else:
        scale_msa, gate_msa, scale_mlp, gate_mlp = (jnp.squeeze(arr, axis=2) for arr in jnp.split(ada_values, 4, axis=2))
        norm_hidden_states = norm_hidden_states * (1 + scale_msa)
    elif self.adaptive_norm == "none":
      scale_msa, gate_msa, scale_mlp, gate_mlp = None, None, None, None
    else:
      raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

    if norm_hidden_states.shape[1] == 1:
      norm_hidden_states = jnp.squeeze(norm_hidden_states, axis=1)

    # 1. Self-Attention
    attn_output = self.attn1(
        norm_hidden_states,
        freqs_cis=freqs_cis,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        segment_ids=segment_ids,
        kv_attention_segment_ids=encoder_attention_segment_ids if self.only_cross_attention else segment_ids,
        sharding_mesh=self.sharding_mesh,
        skip_layer_mask=skip_layer_mask,
        skip_layer_strategy=skip_layer_strategy,
        **(cross_attention_kwargs or {}),
    )

    attn_output = nn.with_logical_constraint(attn_output, ("activation_batch", "activation_norm_length", "activation_embed"))

    if gate_msa is not None:
      attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
      hidden_states = jnp.squeeze(hidden_states, axis=1)

    # 3. Cross-Attention
    if self.attn2 is not None:
      attn_input = self.attn2_norm(hidden_states) if self.adaptive_norm == "none" else hidden_states
      attn_input = nn.with_logical_constraint(attn_input, ("activation_batch", "activation_norm_length", "activation_embed"))
      attn_output = self.attn2(
          attn_input,
          freqs_cis=freqs_cis,
          encoder_hidden_states=encoder_hidden_states,
          segment_ids=segment_ids,
          kv_attention_segment_ids=encoder_attention_segment_ids,
          sharding_mesh=self.sharding_mesh,
          **(cross_attention_kwargs or {}),
      )
      hidden_states = attn_output + hidden_states

    # 4. Feed-Forward
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = nn.with_logical_constraint(
        norm_hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    if self.adaptive_norm == "single_scale_shift":
      norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    elif self.adaptive_norm == "single_scale":
      norm_hidden_states = norm_hidden_states * (1 + scale_mlp)
    elif self.adaptive_norm == "none":
      pass
    else:
      raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

    ff_output = self.ff(norm_hidden_states)
    ff_output = nn.with_logical_constraint(ff_output, ("activation_batch", "activation_norm_length", "activation_embed"))
    if gate_mlp is not None:
      ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
      hidden_states = jnp.squeeze(hidden_states, axis=1)
    hidden_states = nn.with_logical_constraint(
        hidden_states,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )
    return hidden_states


class Attention(nn.Module):
  query_dim: int
  cross_attention_dim: Optional[int] = None
  heads: int = 8
  dim_head: int = 64
  dropout: float = 0.0
  bias: bool = False
  upcast_attention: bool = False
  upcast_softmax: bool = False
  cross_attention_norm: Optional[str] = None
  added_kv_proj_dim: Optional[int] = None
  out_bias: bool = True
  scale_qk: bool = True
  qk_norm: Optional[str] = None
  only_cross_attention: bool = False
  eps: float = 1e-5
  rescale_output_factor: float = 1.0
  residual_connection: bool = False
  out_dim: Optional[int] = None
  use_tpu_flash_attention: bool = True
  use_rope: bool = False
  attention_op: Optional[nn.Module] = None

  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    """Initialize layers in Flax `setup()`."""
    self.inner_dim = self.out_dim if self.out_dim is not None else self.dim_head * self.heads
    self.use_bias = self.bias
    self.is_cross_attention = self.cross_attention_dim is not None
    self.fused_projections = False
    out_dim = self.out_dim if self.out_dim is not None else self.query_dim
    self.scale = self.dim_head**-0.5 if self.scale_qk else 1.0

    # Query and Key Normalization
    if self.qk_norm is None:
      self.q_norm = Identity()
      self.k_norm = Identity()
    elif self.qk_norm == "rms_norm":
      self.q_norm = RMSNorm(epsilon=self.eps, kernel_axes=("norm",))
      self.k_norm = RMSNorm(epsilon=self.eps, kernel_axes=("norm",))
    elif self.qk_norm == "layer_norm":
      self.q_norm = nn.LayerNorm(epsilon=self.eps)
      self.k_norm = nn.LayerNorm(epsilon=self.eps)
    else:
      raise ValueError(f"Unsupported qk_norm method: {self.qk_norm}")

    if out_dim is not None:
      self.heads_count = out_dim // self.dim_head

    # Validate parameters
    if self.added_kv_proj_dim is None and self.only_cross_attention:
      raise ValueError(
          "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. "
          "Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
      )

    if self.cross_attention_norm is None:
      self.norm_cross = None
    elif self.cross_attention_norm == "layer_norm":
      self.norm_cross = nn.LayerNorm(epsilon=self.eps)
    else:
      raise ValueError(
          f"Unknown cross_attention_norm: {self.cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'."
      )

    # Linear layers for queries, keys, values
    self.to_q = DenseGeneral(
        features=(self.inner_dim,),
        use_bias=self.bias,
        name="to_q",
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        kernel_axes=("embed", "kv"),
        axis=-1,
    )

    if not self.only_cross_attention:
      self.to_k = DenseGeneral(
          features=(self.inner_dim,),
          use_bias=self.bias,
          name="to_k",
          matmul_precision=self.matmul_precision,
          weight_dtype=self.weight_dtype,
          dtype=self.dtype,
          kernel_axes=("embed", "kv_head_dim"),
          axis=-1,
      )
      self.to_v = DenseGeneral(
          features=(self.inner_dim,),
          use_bias=self.bias,
          name="to_v",
          matmul_precision=self.matmul_precision,
          weight_dtype=self.weight_dtype,
          dtype=self.dtype,
          kernel_axes=("embed", "kv_head_dim"),
          axis=-1,
      )
    else:
      self.to_k = None
      self.to_v = None

    if self.added_kv_proj_dim is not None:
      self.add_k_proj = nn.Dense(self.inner_dim, name="add_k_proj")
      self.add_v_proj = nn.Dense(self.inner_dim, name="add_v_proj")

    self.to_out = [
        DenseGeneral(
            features=(out_dim,),
            use_bias=self.out_bias,
            axis=-1,
            kernel_axes=("kv", "embed"),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            name="to_out.0",
            matmul_precision=self.matmul_precision,
        ),
        nn.Dropout(self.dropout),
    ]

    if self.attention_op is not None:
      self.attention = self.attention_op
    else:
      _tpu_available = any(device.platform == "tpu" for device in jax.devices())
      self.attention = AttentionOp() if _tpu_available else ExplicitAttention()
      if not _tpu_available:
        print("Warning: Running with explicit attention since tpu is not available.")

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      freqs_cis: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
      encoder_hidden_states: Optional[jnp.ndarray] = None,
      segment_ids: Optional[jnp.ndarray] = None,
      kv_attention_segment_ids: Optional[jnp.ndarray] = None,
      sharding_mesh: Optional[jax.sharding.Mesh] = None,
      skip_layer_mask: Optional[jnp.ndarray] = None,
      skip_layer_strategy: Optional[str] = None,
      temb: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
      **cross_attention_kwargs,
  ) -> jnp.ndarray:
    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}  # noqa F821
    assert cross_attention_kwargs.get("scale", None) is None, "Not supported"

    input_axis_names = ("activation_batch", "activation_length", "activation_embed")
    hidden_states = nn.with_logical_constraint(hidden_states, input_axis_names)
    if encoder_hidden_states is not None:
      encoder_hidden_states = nn.with_logical_constraint(encoder_hidden_states, input_axis_names)

    residual = hidden_states
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
      batch_size, channel, height, width = hidden_states.shape
      hidden_states = jnp.reshape(hidden_states, (batch_size, channel, height * width))
      hidden_states = jnp.swapaxes(hidden_states, 1, 2)

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    if skip_layer_mask is not None:
      skip_layer_mask = jnp.reshape(skip_layer_mask, (batch_size, 1, 1))

    query = self.to_q(hidden_states)
    query = self.q_norm(query)

    if encoder_hidden_states is not None:
      if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
      key = self.to_k(encoder_hidden_states)
      key = self.k_norm(key)
    else:
      encoder_hidden_states = hidden_states
      key = self.to_k(hidden_states)
      key = self.k_norm(key)
      if self.use_rope:
        key = apply_rotary_emb(key, freqs_cis)
        query = apply_rotary_emb(query, freqs_cis)

    value = self.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads

    query = jnp.reshape(query, (batch_size, -1, self.heads, head_dim))
    query = jnp.swapaxes(query, 1, 2)
    query = nn.with_logical_constraint(
        query, ("activation_kv_batch", "activation_kv_heads", "activation_length", "activation_kv_head_dim")
    )
    query = checkpoint_name(query, "attention query")

    key = jnp.reshape(key, (batch_size, -1, self.heads, head_dim))
    key = jnp.swapaxes(key, 1, 2)
    key = nn.with_logical_constraint(
        key, ("activation_kv_batch", "activation_kv_heads", "activation_length", "activation_kv_head_dim")
    )
    key = checkpoint_name(key, "attention key")

    value = jnp.reshape(value, (batch_size, -1, self.heads, head_dim))
    value = jnp.swapaxes(value, 1, 2)
    value = nn.with_logical_constraint(
        value, ("activation_kv_batch", "activation_kv_heads", "activation_length", "activation_kv_head_dim")
    )
    value = checkpoint_name(value, "attention value")

    assert self.use_tpu_flash_attention, "JAX only support `use_tpu_flash_attention`"

    q_segment_ids = segment_ids
    if q_segment_ids is not None:
      q_segment_ids = q_segment_ids.astype(jnp.float32)

    if kv_attention_segment_ids is not None and q_segment_ids is None:
      q_segment_ids = jnp.ones((batch_size, query.shape[2]), dtype=jnp.float32)

    hidden_states_a = self.attention(query, key, value, q_segment_ids, kv_attention_segment_ids, sharding_mesh, self.dtype)

    hidden_states_a: jax.Array = nn.with_logical_constraint(
        hidden_states_a, ("activation_kv_batch", "activation_heads", "activation_length", "activation_kv")
    )

    hidden_states_a = jnp.reshape(jnp.swapaxes(hidden_states_a, 1, 2), (batch_size, -1, self.heads * head_dim))
    if skip_layer_mask is not None and skip_layer_strategy == SkipLayerStrategy.AttentionSkip:
      hidden_states = hidden_states_a * skip_layer_mask + hidden_states * (1.0 - skip_layer_mask)
    else:
      hidden_states = hidden_states_a

    hidden_states = self.to_out[0](hidden_states)
    hidden_states = self.to_out[1](hidden_states, deterministic=deterministic)  # Dropout

    if input_ndim == 4:
      hidden_states = jnp.reshape(jnp.swapaxes(hidden_states, -1, -2), (batch_size, channel, height, width))
      if skip_layer_mask is not None and skip_layer_strategy == SkipLayerStrategy.Residual:
        skip_layer_mask = jnp.reshape(skip_layer_mask, (batch_size, 1, 1, 1))

    if self.residual_connection:
      if skip_layer_mask is not None and skip_layer_strategy == SkipLayerStrategy.Residual:
        hidden_states = hidden_states + residual * skip_layer_mask
      else:
        hidden_states = hidden_states + residual

    if self.rescale_output_factor != 1.0:
      hidden_states = hidden_states / self.rescale_output_factor
    hidden_states = checkpoint_name(hidden_states, "attention_output")

    return hidden_states

  def prepare_attention_mask(
      self, attention_mask: jnp.ndarray, target_length: int, batch_size: int, out_dim: int = 3
  ) -> jnp.ndarray:
    head_size = self.heads_count
    if attention_mask is None:
      return attention_mask

    current_length = attention_mask.shape[-1]
    if current_length != target_length:
      remaining_length = target_length - current_length
      attention_mask = jnp.pad(attention_mask, ((0, 0), (0, remaining_length)), constant_values=0.0)

    if out_dim == 3:
      if attention_mask.shape[0] < batch_size * head_size:
        attention_mask = jnp.repeat(attention_mask, head_size, axis=0)
    elif out_dim == 4:
      attention_mask = jnp.expand_dims(attention_mask, axis=1)
      attention_mask = jnp.repeat(attention_mask, head_size, axis=1)

    return attention_mask

  def norm_encoder_hidden_states(self, encoder_hidden_states: jnp.ndarray) -> jnp.ndarray:
    assert self.norm_cross is not None, "self.norm_cross must be defined to call norm_encoder_hidden_states."

    if isinstance(self.norm_cross, nn.LayerNorm):
      encoder_hidden_states = self.norm_cross(encoder_hidden_states)
    elif isinstance(self.norm_cross, nn.GroupNorm):
      encoder_hidden_states = jnp.swapaxes(encoder_hidden_states, 1, 2)
      encoder_hidden_states = self.norm_cross(encoder_hidden_states)
      encoder_hidden_states = jnp.swapaxes(encoder_hidden_states, 1, 2)
    else:
      raise ValueError("Unknown normalization type for cross-attention.")

    return encoder_hidden_states


class AttentionOp(nn.Module):

  @nn.compact
  def __call__(
      self,
      q: jax.Array,  # [batch_size, heads, q_tokens, hidden_dim]
      k: jax.Array,  # [batch_size, heads, kv_tokens, hidden_dim]
      v: jax.Array,  # [batch_size, heads, kv_tokens, hidden_dim]
      q_segment_ids: jax.Array,  # [batch_size, q_tokens]
      kv_segment_ids: jax.Array,  # [batch_size, kv_tokens]
      sharding_mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
      block_sizes: Optional[BlockSizes] = None,
  ):
    if block_sizes is None:
      block_sizes = self.default_block_sizes(q, k, dtype)

    scale_factor = 1 / math.sqrt(q.shape[-1])

    def partial_flash_attention(q, k, v, q_segment_ids, kv_segment_ids):
      s = (
          # flash attention expects segment ids to be float32
          SegmentIds(q_segment_ids.astype(jnp.float32), kv_segment_ids.astype(jnp.float32))
          if q_segment_ids is not None and kv_segment_ids is not None
          else None
      )
      output = jax_flash_attention(
          q,
          k,
          v,
          None,
          s,
          sm_scale=scale_factor,
          block_sizes=block_sizes,
      )
      return output

    if sharding_mesh is not None:
      if q.ndim != 4:
        raise ValueError(f"Expected input with 4 dims, got {q.ndim}.")
      if q_segment_ids is not None and q_segment_ids.ndim != 2:
        raise ValueError(f"Expected mask with 2 dims, got {q_segment_ids.ndim}.")
      # Based on: ("activation_kv_batch", "activation_kv_heads", "activation_length", "activation_kv_head_dim")
      # Computation of the spec based on the logical constraints can be found in logical_axes_to_spec.py.
      # qkvo_sharding_spec = jax.sharding.PartitionSpec(
      #     ("data", "fsdp", "fsdp_transpose", "expert"),
      #     ("tensor", "tensor_transpose", "sequence", "tensor_sequence"),
      #     None,
      #     None,
      # )
      qkvo_sharding_spec = jax.sharding.PartitionSpec(
          None,
          ("data", "fsdp", "tensor"),
          None,
          None,
      )
      # Based on: ("activation_kv_batch", "activation_length")
      qkv_segment_ids_spec = jax.sharding.PartitionSpec("fsdp", None)
      # qkv_segment_ids_spec = jax.sharding.PartitionSpec(None, None)
      wrapped_flash_attention = shard_map(
          partial_flash_attention,
          mesh=sharding_mesh,
          in_specs=(
              qkvo_sharding_spec,
              qkvo_sharding_spec,
              qkvo_sharding_spec,
              qkv_segment_ids_spec,
              qkv_segment_ids_spec,
          ),
          out_specs=qkvo_sharding_spec,
          check_rep=False,
      )
    else:
      wrapped_flash_attention = partial_flash_attention

    return wrapped_flash_attention(
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
    )

  def default_block_sizes(self, q: jax.Array, k: jax.Array, dtype: jnp.dtype = jnp.float32) -> BlockSizes:
    """
    Default block sizes for Flash Attention.

    TPU kernel ops runs in grids, the bigger the grid - the more data that is loaded on the SRAM
    we want to utilize the SRAM the best we can

    too big grids will cuase cache misses and slow down the computation while the faster SRAM retrieves the other block data
    from the slower HBRAM

    a certain balance has to be met to get the best performance
    imho, that balance must be computed with the combination of the information supplied by q and k (which will supply query sequence and key/value sequence lengths)
    along with the SRAM cache size

    ** SRAM cache size for TPU
    V5P - 1MB SRAM per core

    Args:
        q (jax.Array): Query tensor to be used
        k (jax.Array): Key tensor to be used

    Returns:
        BlockSizes: Grid block sizes
    """
    max_block_size = 1024 if dtype == jnp.bfloat16 else 512
    return BlockSizes(
        block_q=min(max_block_size, q.shape[-2]),
        block_k_major=min(max_block_size, k.shape[-2]),
        block_k=min(max_block_size, k.shape[-2]),
        block_b=min(1, q.shape[0]),
        block_q_major_dkv=min(max_block_size, q.shape[-2]),
        block_k_major_dkv=min(max_block_size, k.shape[-2]),
        block_q_dkv=min(max_block_size, q.shape[-2]),
        block_k_dkv=min(max_block_size, k.shape[-2]),
        block_q_dq=min(max_block_size, q.shape[-2]),
        block_k_dq=min(512, k.shape[-2]),
        block_k_major_dq=min(max_block_size, k.shape[-2]),
    )


class ExplicitAttention(nn.Module):

  def __call__(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array,
      kv_segment_ids: jax.Array,
      sharding_mesh: Optional[jax.sharding.Mesh] = None,
      dtype: jnp.dtype = jnp.float32,
  ):
    assert sharding_mesh is None, "Explicit attention does not support sharding mesh."
    attn_mask = None
    if kv_segment_ids is not None:
      q_segment_ids_expanded = q_segment_ids[:, None, :, None]
      kv_segment_ids_expanded = kv_segment_ids[:, None, None, :]
      attn_mask = q_segment_ids_expanded == kv_segment_ids_expanded

    scale_factor = 1 / jnp.sqrt(q.shape[-1])
    attn_bias = jnp.zeros((q.shape[-2], k.shape[-2]), dtype=q.dtype)

    if attn_mask is not None:
      if attn_mask.dtype == jnp.bool_:
        attn_bias = jnp.where(attn_mask, attn_bias, float("-inf"))
      else:
        attn_bias += attn_mask

    attn_weight = q @ k.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = jnn.softmax(attn_weight, axis=-1)

    return attn_weight @ v


class RMSNorm(nn.Module):
  """
  RMSNorm is a normalization layer that normalizes the input using the root mean square.
  """

  epsilon: float
  dtype: jnp.dtype = jnp.float32
  elementwise_affine: bool = True
  weight_dtype: jnp.dtype = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    """
    Forward pass of the RMSNorm layer.

    First we compute the variance (mean of the square of the input)
    and then normalize the input using the root mean square.

    NOTE: if weight is in mixed precision, the operand should be in the same precision.
    Args:
        hidden_states (jax.Array): Input data

    Returns:
        jax.Array: Normed data
    """

    # dim = (self.dim,) if isinstance(self.dim, numbers.Integral) else self.dim
    dim = hidden_states.shape[-1]
    if self.elementwise_affine:
      scale = self.param(
          "scale",
          nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
          (dim,),
          self.weight_dtype,
      )
    else:
      scale = None

    input_dtype = hidden_states.dtype
    variance = jnp.mean(jnp.square(hidden_states.astype(jnp.float32)), axis=-1, keepdims=True)
    hidden_states: jax.Array = hidden_states * jax.lax.rsqrt(variance + self.epsilon)

    if self.elementwise_affine:
      # convert into half-precision if necessary
      hidden_states = (hidden_states.astype(self.dtype) * scale.astype(self.dtype)).astype(input_dtype)
    else:
      hidden_states = hidden_states.astype(input_dtype)

    return hidden_states


class FeedForward(nn.Module):
  r"""
  A feed-forward layer.

  Parameters:
      dim (`int`): The number of channels in the input.
      dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
      mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
      dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
      activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
      final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
      bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
  """

  dim_out: Optional[int] = None
  mult: int = 4
  dropout: float = 0.0
  activation_fn: str = "gelu"
  final_dropout: bool = False
  bias: bool = True
  inner_dim: Optional[int] = None
  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  @nn.compact
  def __call__(self, hidden_states: jax.Array, scale: float = 1.0, deterministic: bool = False) -> jax.Array:
    dim = hidden_states.shape[-1]
    if self.inner_dim is None:
      inner_dim = dim * self.mult
      if inner_dim < 256:
        raise ValueError("inner_dim must be at least 256")
      # round to nearest multiple of 256
      inner_dim = round(inner_dim / 256) * 256
    else:
      inner_dim = self.inner_dim

    dim_out = self.dim_out if self.dim_out is not None else dim

    act_kwargs = {
        "name": "net.0",
        "bias": self.bias,
        "kernel_axes": ("embed", "mlp"),
        "matmul_precision": self.matmul_precision,
        "weight_dtype": self.weight_dtype,
        "dtype": self.dtype,
    }
    match self.activation_fn:
      case "gelu":
        act_fn = GELU(dim, inner_dim, **act_kwargs)
      case "gelu-approximate":
        act_fn = GELU(dim, inner_dim, approximate="tanh", **act_kwargs)
      case "geglu":
        act_fn = GEGLU(dim, inner_dim, **act_kwargs)
      case "geglu-approximate":
        act_fn = ApproximateGELU(dim, inner_dim, **act_kwargs)
      case _:
        raise ValueError(f"activation function {self.activation_fn} not supported")

    if isinstance(act_fn, GEGLU):
      hidden_states = act_fn(hidden_states, scale)
    else:
      hidden_states = act_fn(hidden_states)

    hidden_states = checkpoint_name(hidden_states, "FFN - activation")
    hidden_states = nn.Dropout(self.dropout)(hidden_states, deterministic=deterministic)

    hidden_states = DenseGeneral(
        dim_out,
        use_bias=self.bias,
        kernel_axes=("mlp", "embed"),
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="net.2",
    )(hidden_states)
    hidden_states = checkpoint_name(hidden_states, "FFN - Reprojection")
    if self.final_dropout:
      # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
      hidden_states = nn.Dropout(self.dropout)(hidden_states, deterministic=deterministic)

    return hidden_states


def apply_rotary_emb(input_tensor: jax.Array, freqs_cis: Tuple[jax.Array, jax.Array]) -> jax.Array:
  """
  Integrates positional information into input tensors using RoPE.

  Args:
      input_tensor (jax.Array): Input_tensor (from QKV of attention mechanism)
      freqs_cis (Tuple[jax.Array, jax.Array]): The sine and cosine frequencies

  Returns:
      jax.Array: Tensor where positional information has been integrated into the original input tensor
  """
  if len(freqs_cis) != 2:
    raise ValueError("freqs_cis must be a tuple of 2 elements")

  cos_freqs, sin_freqs = freqs_cis

  t_dup = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
  t1, t2 = jnp.split(t_dup, 2, axis=-1)
  t_dup = jnp.concatenate([-t2, t1], axis=-1)
  input_tensor_rot = t_dup.reshape(*input_tensor.shape)

  # Apply rotary embeddings
  out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

  return out
