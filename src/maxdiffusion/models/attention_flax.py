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

import functools
import math
from typing import Optional, Callable, Tuple
import flax.linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from einops import rearrange
from .. import common_types, max_logging

from . import quantizations


Array = common_types.Array
Mesh = common_types.Mesh
DType = common_types.DType
BlockSizes = common_types.BlockSizes


AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED
Quant = quantizations.AqtQuantization


def _maybe_aqt_einsum(quant: Quant):
  return jnp.einsum if quant is None else quant.einsum()


def _check_attention_inputs(query: Array, key: Array, value: Array) -> None:
  """Check attention inputs."""

  assert key.ndim == value.ndim, "k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
  assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."


def _reshape_data_from_cudnn_flash(tensor):
  # reshapes from [b, s, h, d] back to [b, s, h * d]
  return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


def _reshape_data_for_cudnn_flash(tensor, heads):
  # reshapes from [b, s, h * d] to [b, s, h, d] (input format to flash format)
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  return tensor


def _reshape_batch_dim_to_heads(tensor, heads):
  batch_size, seq_len, dim = tensor.shape
  head_size = heads
  tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  reshaped_tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
  return jax.lax.with_sharding_constraint(reshaped_tensor, PartitionSpec("data", "fsdp", "tensor"))


def _reshape_heads_to_batch_dim(tensor, heads):
  if tensor.ndim == 3:
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
  else:
    batch_size, head_size, seq_len, head_dim = tensor.shape
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, head_dim)

  return jax.lax.with_sharding_constraint(reshaped_tensor, PartitionSpec("data", "fsdp", "tensor"))


def _reshape_heads_to_head_dim(tensor):
  # takes a tensor of shape [b, h, s, d] and reshapes to [b, s, h * d]
  # This is used to transform the output of flash attention back into the format of other attention outputs
  b, h, s, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  reshaped_tensor = jnp.reshape(tensor, (b, -1, h * d))
  return jax.lax.with_sharding_constraint(reshaped_tensor, PartitionSpec("data", "fsdp", "tensor"))


def _unflatten_heads(tensor, heads):
  # reshapes from [b, s, h * d] to [b, h, s, d] (input format to flash format)
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  # Transpose to ('batch', 'heads', 'length', 'kv')
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _reshape_data_for_flash(tensor, heads):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  if tensor.ndim != 4:
    tensor = _unflatten_heads(tensor, heads)
  return tensor


def _pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  tensor = _reshape_data_for_flash(tensor, heads)

  # Pad head_dim to 128 if less than that.
  kv_size = tensor.shape[-1]
  head_dim_pad = 0
  if kv_size < 128:
    head_dim_pad = 128 - kv_size

  # Pad seq_len with sharding constraints.
  seq_len = tensor.shape[2]

  # 1. First, pad seq_len to be a multiple of flash_block_size
  rem = seq_len % flash_block_size
  if rem != 0:
    seq_len_padded_pre = seq_len + (flash_block_size - rem)
  else:
    seq_len_padded_pre = seq_len

  # 2. Ensure num_blocks is divisible by num_shards
  num_blocks = seq_len_padded_pre // flash_block_size
  if num_blocks % num_shards != 0:
    num_blocks += num_shards - (num_blocks % num_shards)

  final_padded_len = num_blocks * flash_block_size
  seq_len_pad = final_padded_len - seq_len

  if kv_size < 128 or seq_len_pad != 0:
    npad = ((0, 0), (0, 0), (0, seq_len_pad), (0, head_dim_pad))
    tensor = jnp.pad(tensor, npad)

  return tensor, kv_size, seq_len


def _tpu_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    attention_kernel: str = "flash",
) -> jax.Array:
  """TPU Flash Attention"""

  q_max_block_size = 1024 if dtype == jnp.bfloat16 else 512
  # This is the case for cross-attn.
  if key.shape[1] != query.shape[1]:
    assert key.shape[1] % 128 == 0
    kv_max_block_size = key.shape[1]
  else:
    kv_max_block_size = q_max_block_size
  if flash_block_sizes:
    block_sizes = flash_block_sizes
  else:
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(q_max_block_size, query.shape[2]),
        block_kv_compute=min(kv_max_block_size, key.shape[2]),
        block_kv=min(kv_max_block_size, key.shape[2]),
        block_q_dkv=min(q_max_block_size, query.shape[2]),
        block_kv_dkv=min(kv_max_block_size, key.shape[2]),
        block_kv_dkv_compute=min(kv_max_block_size, query.shape[2]),
        block_q_dq=min(q_max_block_size, query.shape[2]),
        block_kv_dq=min(kv_max_block_size, query.shape[2]),
    )
  num_fsdp_shards = mesh.shape["fsdp"]
  query = _reshape_data_for_flash(query, heads)
  key = _reshape_data_for_flash(key, heads)
  value = _reshape_data_for_flash(value, heads)
  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)

  @functools.partial(
      shard_map.shard_map,
      mesh=mesh,
      in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
      out_specs=q_axis_names,
      check_rep=False,
  )
  def wrap_flash_attention(query, key, value):

    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_sizes.block_q)
    key, _, key_seq_len = _pad_data_for_flash(key, heads, block_sizes.block_kv)
    value, _, _ = _pad_data_for_flash(value, heads, block_sizes.block_kv)

    mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

    q_padded_len = query.shape[2]
    q_indices = jax.lax.broadcasted_iota(jnp.int32, (q_padded_len,), 0)
    q_segment_ids = (q_indices < query_seq_len).astype(jnp.int32)

    kv_padded_len = key.shape[2]
    kv_indices = jax.lax.broadcasted_iota(jnp.int32, (kv_padded_len,), 0)
    kv_segment_ids = (kv_indices < key_seq_len).astype(jnp.int32)
    segment_ids = splash_attention_kernel.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)

    # make_splash_mha is wrapped around shardmap and seq and head is already
    # sharded based on in_specs, therefore setting head_shards=1 and q_seq_shards=1.
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,  # the sizes of the axis is sharding over heads
        q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
        block_sizes=block_sizes,
        save_residuals=True if attention_kernel == "ring" else False,
    )
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))

    if attention_kernel == "flash":
      attention_output = vmapped_splash(query, key, value, segment_ids)
    else:
      if num_fsdp_shards > 1:
        out, (lse,) = vmapped_splash(query, key, value, segment_ids)
        m = lse.astype(jnp.float32)
        l = jnp.exp(lse - m)
        o = out.astype(jnp.float32) * l[..., None]

        perm = [(j, (j + 1) % num_fsdp_shards) for j in range(num_fsdp_shards)]

        k1 = jax.lax.ppermute(key, axis_name="fsdp", perm=perm)
        v1 = jax.lax.ppermute(value, axis_name="fsdp", perm=perm)

        def ring_scan_body(carry, _):
          m, l, o, k_current, v_current = carry
          k_next = jax.lax.ppermute(k_current, axis_name="fsdp", perm=perm)
          v_next = jax.lax.ppermute(v_current, axis_name="fsdp", perm=perm)

          out_chunk, (lse_chunk,) = vmapped_splash(query, k_current, v_current, segment_ids)

          m_chunk = lse_chunk.astype(jnp.float32)
          m_old = m
          m = jnp.maximum(m_old, m_chunk)

          exp_m_diff = jnp.exp(m_old - m)
          exp_m_chunk_diff = jnp.exp(m_chunk - m)

          l = l * exp_m_diff + jnp.exp(lse_chunk - m)
          o = o * exp_m_diff[..., None]
          o += exp_m_chunk_diff[..., None] * out_chunk.astype(jnp.float32)

          # Return the updated state for the next iteration
          return (m, l, o, k_next, v_next), None

        initial_carry = (m, l, o, k1, v1)
        (m_final, l_final, o_final, _, _), _ = jax.lax.scan(ring_scan_body, initial_carry, None, length=num_fsdp_shards - 1)

        attention_output = o_final / l_final[..., None]

    return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

  devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
  # This warning might show up when doing model eval for example, when calculating model flops
  # and that is expected.
  if not (query.shape[0] / devices_in_data_fsdp).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_data_fsdp: {devices_in_data_fsdp}"
    )
  x = wrap_flash_attention(query, key, value)
  x = _reshape_heads_to_head_dim(x)

  return x


def _apply_attention_dot(
    query: Array,
    key: Array,
    value: Array,
    dtype: jnp.dtype,
    heads: int,
    dim_head: int,
    scale: float,
    split_head_dim: bool,
    float32_qk_product: bool,
    use_memory_efficient_attention: bool,
):
  """Apply Attention."""
  if split_head_dim:
    b = key.shape[0]
    query_states = jnp.reshape(query, (b, -1, heads, dim_head))
    key_states = jnp.reshape(key, (b, -1, heads, dim_head))
    value_states = jnp.reshape(value, (b, -1, heads, dim_head))
  else:
    query_states = _reshape_heads_to_batch_dim(query, heads)
    key_states = _reshape_heads_to_batch_dim(key, heads)
    value_states = _reshape_heads_to_batch_dim(value, heads)

  if float32_qk_product:
    query_states = query_states.astype(jnp.float32)
    key_states = key_states.astype(jnp.float32)

  if use_memory_efficient_attention:
    query_states = query_states.transpose(1, 0, 2)
    key_states = key_states.transpose(1, 0, 2)
    value_states = value_states.transpose(1, 0, 2)

    # this if statement create a chunk size for each layer of the unet
    # the chunk size is equal to the query_length dimension of the deepest layer of the unet

    flatten_latent_dim = query_states.shape[-3]
    if flatten_latent_dim % 64 == 0:
      query_chunk_size = int(flatten_latent_dim / 64)
    elif flatten_latent_dim % 16 == 0:
      query_chunk_size = int(flatten_latent_dim / 16)
    elif flatten_latent_dim % 4 == 0:
      query_chunk_size = int(flatten_latent_dim / 4)
    else:
      query_chunk_size = int(flatten_latent_dim)

    hidden_states = jax_memory_efficient_attention(
        query_states, key_states, value_states, query_chunk_size=query_chunk_size, key_chunk_size=4096 * 4
    )

    hidden_states = hidden_states.transpose(1, 0, 2)
  else:
    if split_head_dim:
      attention_scores = jnp.einsum("b t n h, b f n h -> b n f t", key_states, query_states)
    else:
      attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)

    attention_scores = attention_scores * scale
    attention_probs = nn.softmax(attention_scores, axis=-1 if split_head_dim else 2)

    attention_probs = attention_probs.astype(dtype)

    # attend to values
    if split_head_dim:
      hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value_states)
      b = hidden_states.shape[0]
      hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    else:
      hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
      hidden_states = _reshape_batch_dim_to_heads(hidden_states, heads)

  return hidden_states


def _cudnn_flash_attention(query: Array, key: Array, value: Array, heads: int, mesh: Mesh, dpa_layer: Callable) -> Array:
  """CUDNN Flash Attention with Transformer Engine.
  1. Stable API, supports GQA
  2. Supports head_dim till 128; head_dim=256 support will be added soon
  """
  # These imports are only meant to work in a GPU build.
  # copied from tpu_flash_attention
  query = _reshape_data_for_cudnn_flash(query, heads)
  key = _reshape_data_for_cudnn_flash(key, heads)
  value = _reshape_data_for_cudnn_flash(value, heads)

  cudnn_flash_axis_names = (BATCH, LENGTH, HEAD, D_KV)
  axis_names = nn.logical_to_mesh_axes(cudnn_flash_axis_names)

  query = nn.with_logical_constraint(query, axis_names)
  key = nn.with_logical_constraint(key, axis_names)
  value = nn.with_logical_constraint(value, axis_names)

  @functools.partial(
      shard_map.shard_map,
      mesh=mesh,
      in_specs=(axis_names, axis_names, axis_names),
      out_specs=axis_names,
      check_rep=False,
  )
  def wrap_flash_attention(query, key, value):
    return jax.vmap(dpa_layer)(query, key, value, mask=None)

  out = wrap_flash_attention(query, key, value)
  return _reshape_data_from_cudnn_flash(out)


def _apply_attention(
    query: Array,
    key: Array,
    value: Array,
    heads: int,
    dim_head: int,
    split_head_dim: bool,
    float32_qk_product: bool,
    attention_kernel: str,
    flash_min_seq_length: int,
    use_memory_efficient_attention: bool,
    scale: float,
    dtype: jnp.dtype,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dpa_layer: Callable,
):
  """Routes to different attention kernels."""
  _check_attention_inputs(query, key, value)
  seq_len_idx = 1
  if query.ndim == 4:
    seq_len_idx = 2
  if attention_kernel == "flash":
    can_use_flash_attention = (
        query.shape[seq_len_idx] >= flash_min_seq_length
        and key.shape[seq_len_idx] >= flash_min_seq_length
        and value.shape[seq_len_idx] >= flash_min_seq_length
    )
  else:
    can_use_flash_attention = True
  if attention_kernel == "dot_product" or use_memory_efficient_attention or not can_use_flash_attention:
    return _apply_attention_dot(
        query, key, value, dtype, heads, dim_head, scale, split_head_dim, float32_qk_product, use_memory_efficient_attention
    )
  elif attention_kernel == "flash":
    return _tpu_flash_attention(
        query, key * scale, value, heads, mesh, axis_names_q, axis_names_kv, flash_block_sizes, dtype
    )
  elif attention_kernel == "ring":
    return _tpu_flash_attention(
        query, key * scale, value, heads, mesh, axis_names_q, axis_names_kv, flash_block_sizes, dtype, attention_kernel
    )
  elif attention_kernel == "cudnn_flash_te":
    return _cudnn_flash_attention(query, key, value, heads, mesh, dpa_layer)
  else:
    raise ValueError(f"Unexpected attention kernel {attention_kernel=}.")


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
  """Multi-head dot product attention with a limited number of queries."""
  num_kv, num_heads, k_features = key.shape[-3:]
  v_features = value.shape[-1]
  key_chunk_size = min(key_chunk_size, num_kv)
  query = query / jnp.sqrt(k_features)

  @functools.partial(jax.checkpoint, prevent_cse=False)
  def summarize_chunk(query, key, value):
    attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

    max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    max_score = jax.lax.stop_gradient(max_score)
    exp_weights = jnp.exp(attn_weights - max_score)

    exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
    max_score = jnp.einsum("...qhk->...qh", max_score)

    return (exp_values, exp_weights.sum(axis=-1), max_score)

  def chunk_scanner(chunk_idx):
    # julienne key array
    key_chunk = jax.lax.dynamic_slice(
        operand=key,
        start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
        slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
    )

    # julienne value array
    value_chunk = jax.lax.dynamic_slice(
        operand=value,
        start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
        slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
    )

    return summarize_chunk(query, key_chunk, value_chunk)

  chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

  global_max = jnp.max(chunk_max, axis=0, keepdims=True)
  max_diffs = jnp.exp(chunk_max - global_max)

  chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
  chunk_weights *= max_diffs

  all_values = chunk_values.sum(axis=0)
  all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

  return all_values / all_weights


def jax_memory_efficient_attention(
    query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size: int = 1024, key_chunk_size: int = 4096
):
  r"""
  Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
  https://github.com/AminRezaei0x443/memory-efficient-attention

  Args:
      query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
      key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
      value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
      precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
          numerical precision for computation
      query_chunk_size (`int`, *optional*, defaults to 1024):
          chunk size to divide query array value must divide query_length equally without remainder
      key_chunk_size (`int`, *optional*, defaults to 4096):
          chunk size to divide key and value array value must divide key_value_length equally without remainder

  Returns:
      (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
  """
  num_q, num_heads, q_features = query.shape[-3:]

  def chunk_scanner(chunk_idx, _):
    # julienne query array
    query_chunk = jax.lax.dynamic_slice(
        operand=query,
        start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
        slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
    )

    return (
        chunk_idx + query_chunk_size,  # unused ignore it
        _query_chunk_attention(query=query_chunk, key=key, value=value, precision=precision, key_chunk_size=key_chunk_size),
    )

  _, res = jax.lax.scan(
      f=chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)  # start counter  # stop counter
  )

  return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


def apply_rope(xq: Array, xk: Array, freqs_cis: Array) -> tuple[Array, Array]:
  xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

  return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)


class NNXAttentionOp(nnx.Module):

  def __init__(
      self,
      mesh: Mesh,
      attention_kernel: str,
      scale: int,
      heads: int,
      dim_head: int,
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = True,
      float32_qk_product: bool = True,
      axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV),
      axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV),
      # Uses splash attention on cross attention.
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      dtype: DType = jnp.float32,
      quant: Quant = None,
  ):
    self.dpa_layer = None
    if attention_kernel == "cudnn_flash_te":
      raise NotImplementedError(f"{self} has not been tested with {attention_kernel}")

    self.mesh = mesh
    self.scale = scale
    self.heads = heads
    self.dim_head = dim_head
    self.attention_kernel = attention_kernel
    self.use_memory_efficient_attention = use_memory_efficient_attention
    self.split_head_dim = split_head_dim
    self.float32_qk_product = float32_qk_product
    self.axis_names_q = axis_names_q
    self.axis_names_kv = axis_names_kv
    self.flash_min_seq_length = flash_min_seq_length
    self.flash_block_sizes = flash_block_sizes
    self.dtype = dtype
    self.quant = quant

  def apply_attention(self, query: Array, key: Array, value: Array):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        dpa_layer=self.dpa_layer,
    )


class AttentionOp(nn.Module):
  mesh: Mesh
  attention_kernel: str
  scale: int
  heads: int
  dim_head: int
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  float32_qk_product: bool = True
  axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV)
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  dtype: DType = jnp.float32
  quant: Quant = None

  def setup(self):
    self.dpa_layer = None
    if self.attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      self.dpa_layer = DotProductAttention(
          head_dim=self.dim_head,
          num_attention_heads=self.heads,
          num_gqa_groups=self.heads,
          attn_mask_type="no_mask",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
          attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
          # attention_dropout=self.dropout_rate,
          dropout_rng_name="aqt",
          dtype=self.dtype,
          # float32_logits=self.float32_logits,
          qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
          scale_factor=self.scale,
          transpose_batch_sequence=False,
      )

  def apply_attention(self, query: Array, key: Array, value: Array):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        dpa_layer=self.dpa_layer,
    )


class FlaxWanAttention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      query_dim: int,
      cross_attention_dim: Optional[int] = None,
      heads: int = 8,
      dim_head: int = 64,
      dropout: float = 0.0,
      eps: float = 1e-6,
      qk_norm: str = "rms_norm_across_heads",
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = False,
      attention_kernel: str = "flash",
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      key_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      value_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      out_axis_names: AxisNames = (BATCH, LENGTH, EMBED),
      precision: jax.lax.Precision = None,
      qkv_bias: bool = False,
      quant: Quant = None,
  ):
    if attention_kernel == "cudnn_flash_te":
      raise NotImplementedError(f"Wan 2.1 has not been tested with {attention_kernel}")

    if attention_kernel in {"flash", "cudnn_flash_te"} and mesh is None:
      raise ValueError(f"The flash attention kernel requires a value for mesh, but mesh is {self.mesh}")
    self.dim_head = dim_head
    self.heads = heads
    self.inner_dim = dim_head * heads
    scale = dim_head**-0.5
    self.qk_norm = qk_norm
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.out_axis_names = out_axis_names

    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=scale,
        heads=heads,
        dim_head=dim_head,
        use_memory_efficient_attention=use_memory_efficient_attention,
        split_head_dim=split_head_dim,
        float32_qk_product=False,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        quant=quant,
    )
    # None axes corresponds to the stacked weights across all blocks
    # because of the use of nnx.vmap and nnx.scan.
    # Dims are [num_blocks, embed, heads]
    kernel_axes = ("embed", "heads")
    qkv_init_kernel = nnx.with_partitioning(nnx.initializers.lecun_normal(), kernel_axes)

    self.query = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("embed",),
        ),
    )

    self.key = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("embed",),
        ),
    )

    self.value = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("embed",),
        ),
    )

    self.proj_attn = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.drop_out = nnx.Dropout(dropout)

    self.norm_q = nnx.data(None)
    self.norm_k = nnx.data(None)
    if qk_norm is not None:
      self.norm_q = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

      self.norm_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

  def _apply_rope(self, xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array) -> Tuple[jax.Array, jax.Array]:
    dtype = xq.dtype
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    xq_out_complex = xq_ * freqs_cis
    xk_out_complex = xk_ * freqs_cis

    xq_out = jnp.stack([jnp.real(xq_out_complex), jnp.imag(xq_out_complex)], axis=-1).reshape(xq.shape).astype(dtype)
    xk_out = jnp.stack([jnp.real(xk_out_complex), jnp.imag(xk_out_complex)], axis=-1).reshape(xk.shape).astype(dtype)

    return xq_out, xk_out

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array = None,
      rotary_emb: Optional[jax.Array] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    hidden_states = jax.lax.with_sharding_constraint(hidden_states, PartitionSpec("data", "fsdp", "tensor"))
    encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, PartitionSpec("data", "fsdp", "tensor"))
    dtype = hidden_states.dtype
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    query_proj = self.query(hidden_states)
    key_proj = self.key(encoder_hidden_states)
    value_proj = self.value(encoder_hidden_states)

    if self.qk_norm:
      query_proj = self.norm_q(query_proj)
      key_proj = self.norm_k(key_proj)
    if rotary_emb is not None:
      query_proj = _unflatten_heads(query_proj, self.heads)
      key_proj = _unflatten_heads(key_proj, self.heads)
      value_proj = _unflatten_heads(value_proj, self.heads)
      # output of _unflatten_heads Batch, heads, seq_len, head_dim
      query_proj, key_proj = self._apply_rope(query_proj, key_proj, rotary_emb)

    query_proj = checkpoint_name(query_proj, "query_proj")
    key_proj = checkpoint_name(key_proj, "key_proj")
    value_proj = checkpoint_name(value_proj, "value_proj")
    attn_output = self.attention_op.apply_attention(query_proj, key_proj, value_proj)

    attn_output = attn_output.astype(dtype=dtype)
    attn_output = checkpoint_name(attn_output, "attn_output")
    hidden_states = self.proj_attn(attn_output)
    hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    return hidden_states


class FlaxFluxAttention(nn.Module):
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
  out_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  precision: jax.lax.Precision = None
  qkv_bias: bool = False

  def setup(self):
    if self.attention_kernel in {"flash", "cudnn_flash_te"} and self.mesh is None:
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
        float32_qk_product=False,
    )

    kernel_axes = ("embed", "heads")
    qkv_init_kernel = nn.with_logical_partitioning(nn.initializers.lecun_normal(), kernel_axes)

    self.qkv = nn.Dense(
        inner_dim * 3,
        kernel_init=qkv_init_kernel,
        use_bias=self.qkv_bias,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="i_qkv",
        precision=self.precision,
    )

    self.encoder_qkv = nn.Dense(
        inner_dim * 3,
        kernel_init=qkv_init_kernel,
        use_bias=self.qkv_bias,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="e_qkv",
        precision=self.precision,
    )

    self.proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), kernel_axes),
        use_bias=True,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="i_proj",
        precision=self.precision,
    )

    self.encoder_proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), kernel_axes),
        use_bias=True,
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("heads",)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="e_proj",
        precision=self.precision,
    )

    self.query_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )
    self.key_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )

    self.encoder_query_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )
    self.encoder_key_norm = nn.RMSNorm(
        dtype=self.dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
        param_dtype=self.weights_dtype,
    )

  def __call__(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):

    qkv_proj = self.qkv(hidden_states)
    B, L = hidden_states.shape[:2]
    H, D, K = self.heads, qkv_proj.shape[-1] // (self.heads * 3), 3
    qkv_proj = qkv_proj.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
    query_proj, key_proj, value_proj = qkv_proj

    query_proj = self.query_norm(query_proj)

    key_proj = self.key_norm(key_proj)

    if encoder_hidden_states is not None:

      encoder_qkv_proj = self.encoder_qkv(encoder_hidden_states)
      B, L = encoder_hidden_states.shape[:2]
      H, D, K = self.heads, encoder_qkv_proj.shape[-1] // (self.heads * 3), 3
      encoder_qkv_proj = encoder_qkv_proj.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
      encoder_query_proj, encoder_key_proj, encoder_value_proj = encoder_qkv_proj

      encoder_query_proj = self.encoder_query_norm(encoder_query_proj)

      encoder_key_proj = self.encoder_key_norm(encoder_key_proj)

      query_proj = jnp.concatenate((encoder_query_proj, query_proj), axis=2)
      key_proj = jnp.concatenate((encoder_key_proj, key_proj), axis=2)
      value_proj = jnp.concatenate((encoder_value_proj, value_proj), axis=2)

      query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
      key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
      value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

    image_rotary_emb = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)
    query_proj, key_proj = apply_rope(query_proj, key_proj, image_rotary_emb)

    query_proj = query_proj.transpose(0, 2, 1, 3).reshape(query_proj.shape[0], query_proj.shape[2], -1)
    key_proj = key_proj.transpose(0, 2, 1, 3).reshape(key_proj.shape[0], key_proj.shape[2], -1)
    value_proj = value_proj.transpose(0, 2, 1, 3).reshape(value_proj.shape[0], value_proj.shape[2], -1)

    attn_output = self.attention_op.apply_attention(query_proj, key_proj, value_proj)
    context_attn_output = None

    if encoder_hidden_states is not None:
      context_attn_output, attn_output = (
          attn_output[:, : encoder_hidden_states.shape[1]],
          attn_output[:, encoder_hidden_states.shape[1] :],
      )

      attn_output = self.proj_attn(attn_output)

      context_attn_output = self.encoder_proj_attn(context_attn_output)

    return attn_output, context_attn_output


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

  def __call__(self, hidden_states, context=None, deterministic=True, cross_attention_kwargs=None):
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
        dim=self.dim, dropout=self.dropout, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision
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
          self.norm1(hidden_states), context, deterministic=deterministic, cross_attention_kwargs=cross_attention_kwargs
      )
    else:
      hidden_states = self.attn1(
          self.norm1(hidden_states), deterministic=deterministic, cross_attention_kwargs=cross_attention_kwargs
      )

    hidden_states = hidden_states + residual

    # cross attention
    residual = hidden_states
    hidden_states = self.attn2(
        self.norm2(hidden_states), context, deterministic=deterministic, cross_attention_kwargs=cross_attention_kwargs
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
    self.norm = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)

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
          hidden_states, context, deterministic=deterministic, cross_attention_kwargs=cross_attention_kwargs
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
    self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype, self.weights_dtype, precision=self.precision)
    self.net_2 = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.weights_dtype, precision=self.precision)

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
    self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, param_dtype=self.weights_dtype, precision=self.precision)
    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.proj(hidden_states)
    hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
    return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)
