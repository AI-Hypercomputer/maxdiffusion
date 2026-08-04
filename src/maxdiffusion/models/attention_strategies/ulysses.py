# Copyright 2026 Google LLC / The HuggingFace Team. All rights reserved.
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

"""UlyssesStrategy for sequence/head-parallel distributed attention orchestration."""

import functools
import math
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxdiffusion import common_types
from maxdiffusion.models.attention_strategies.protocol import AttentionBackend
from maxdiffusion.models.attention_utils import (
    INTERNAL_RING_AXIS,
    INTERNAL_ULYSSES_AXIS,
    _create_internal_ulysses_ring_mesh,
    _extract_custom_block_sizes,
    _replace_mesh_axis_names,
    _reshape_data_for_flash,
    _reshape_heads_to_head_dim,
    _run_chunked_ulysses_attention,
)

LOG2E = math.log2(math.e)
CONTEXT = common_types.CONTEXT


class UlyssesStrategy:
  """Orchestrates Ulysses sequence-parallel attention and optional inner ring strategies."""

  def __init__(
      self,
      ulysses_shards: int,
      num_ring_shards: int = 1,
      ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
      ring_axis: str = INTERNAL_RING_AXIS,
      ulysses_attention_chunks: int = 1,
      use_base2_exp: bool = True,
      use_experimental_scheduler: bool = False,
      use_fixed_m: bool = False,
      bidirectional: bool = False,
      inner_strategy: Optional[Any] = None,
      scale: float = 1.0,
  ):
    self.ulysses_shards = ulysses_shards
    self.num_ring_shards = num_ring_shards
    self.ulysses_axis = ulysses_axis
    self.ring_axis = ring_axis
    self.ulysses_attention_chunks = ulysses_attention_chunks
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.use_fixed_m = use_fixed_m
    self.bidirectional = bidirectional
    if inner_strategy is None:
      from maxdiffusion.models.attention_strategies.ring import RingAttentionStrategy
      from maxdiffusion.models.attention_strategies.protocol import AttentionBackend

      inner_strategy = RingAttentionStrategy(
          block_sizes=None,
          ring_axis=self.ring_axis,
          ulysses_axis=self.ulysses_axis,
          num_ring_shards=self.num_ring_shards,
          num_ulysses_shards=self.ulysses_shards,
          use_base2_exp=self.use_base2_exp,
          use_experimental_scheduler=self.use_experimental_scheduler,
          use_fixed_m=self.use_fixed_m,
          bidirectional=self.bidirectional,
          backend=AttentionBackend.CUSTOM,
      )
    self.inner_strategy = inner_strategy
    self.scale = scale

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      heads: int,
      mesh: Mesh,
      axis_names_q: Any,
      axis_names_kv: Any,
      flash_block_sizes: Any,
      dtype: jnp.dtype = jnp.float32,
      attention_mask: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Executes Ulysses sequence-parallel attention over context mesh."""
    axis_name = CONTEXT
    num_context_shards = mesh.shape[axis_name] if axis_name in mesh.shape else 1
    num_ulysses_shards = self.ulysses_shards
    if num_ulysses_shards <= 0:
      raise ValueError("UlyssesStrategy requires ulysses_shards > 0.")
    if num_context_shards % num_ulysses_shards != 0:
      raise ValueError(
          f"UlyssesStrategy requires ulysses_shards to divide context_shards, "
          f"got context_shards={num_context_shards} and ulysses_shards={num_ulysses_shards}."
      )
    num_ring_shards = num_context_shards // num_ulysses_shards

    query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
    key, orig_kv_seq_len = _reshape_data_for_flash(key, heads, num_context_shards)
    value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
    num_heads = query.shape[1]
    if num_heads % num_ulysses_shards != 0:
      raise ValueError(
          f"UlyssesStrategy requires heads to be divisible by ulysses_shards (or context_shards), "
          f"got heads={num_heads} and context_shards={num_context_shards} (heads={num_heads} and ulysses_shards={num_ulysses_shards})."
      )

    custom_bs = _extract_custom_block_sizes(flash_block_sizes)
    heads_per_tile = custom_bs.heads_per_tile
    if heads_per_tile > 1:
      raise NotImplementedError("UlyssesStrategy currently supports heads_per_tile == 1 only.")

    q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
    kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
    if num_ring_shards > 1 or self.ring_axis != self.ulysses_axis:
      internal_mesh = _create_internal_ulysses_ring_mesh(
          mesh, num_ring_shards, num_ulysses_shards, self.ring_axis, self.ulysses_axis
      )
      internal_q_axis_names = _replace_mesh_axis_names(q_axis_names, axis_name, (self.ring_axis, self.ulysses_axis))
      internal_kv_axis_names = _replace_mesh_axis_names(kv_axis_names, axis_name, (self.ring_axis, self.ulysses_axis))
    else:
      internal_mesh = mesh
      internal_q_axis_names = q_axis_names
      internal_kv_axis_names = kv_axis_names

    if attention_mask is not None and (
        self.inner_strategy is None
        or getattr(self.inner_strategy, "backend", AttentionBackend.CUSTOM) == AttentionBackend.CUSTOM
    ):
      raise NotImplementedError(
          "Ulysses custom splash kernels do not support attention_mask (padding is handled via orig_seq_len)."
      )

    mask_needs_ulysses_gather = False
    if attention_mask is not None:
      mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))
      if num_ring_shards > 1 or self.ring_axis != self.ulysses_axis:
        internal_mask_axis_names = _replace_mesh_axis_names(mask_axis_names, axis_name, (self.ring_axis, self.ulysses_axis))
      else:
        internal_mask_axis_names = mask_axis_names
      mask_needs_ulysses_gather = self.ulysses_axis in internal_mask_axis_names[1:]

    def wrap_ulysses_ring_attention(query, key, value, mask=None):
      fixed_m_norms = None
      if self.inner_strategy is not None:
        fixed_m_norms = self.inner_strategy.pre_all_to_all_hook(query, key, value, {})

      a2a = functools.partial(jax.lax.all_to_all, axis_name=self.ulysses_axis, tiled=True)
      query = a2a(query, split_axis=1, concat_axis=2)
      key = a2a(key, split_axis=1, concat_axis=2)
      value = a2a(value, split_axis=1, concat_axis=2)
      if mask is not None and mask_needs_ulysses_gather:
        mask = jax.lax.all_gather(mask, axis_name=self.ulysses_axis, axis=1, tiled=True)

      attention_output = self.inner_strategy(
          query,
          key,
          value,
          q_seq_len=orig_q_seq_len // num_ring_shards,
          kv_seq_len=orig_kv_seq_len // num_ring_shards,
          heads_per_tile=heads_per_tile,
          mk_arr=fixed_m_norms,
          attention_mask=mask,
          swap_ds_axes=False,
          block_sizes=getattr(self.inner_strategy, "block_sizes", None) or custom_bs,
      )
      attention_output = a2a(attention_output, split_axis=2, concat_axis=1)
      return attention_output

    if attention_mask is None:
      sharded_fn = jax.shard_map(
          lambda q, k, v: wrap_ulysses_ring_attention(q, k, v, None),
          mesh=internal_mesh,
          in_specs=(internal_q_axis_names, internal_kv_axis_names, internal_kv_axis_names),
          out_specs=internal_q_axis_names,
          check_vma=False,
      )

      def run_fn(q, k, v):
        return sharded_fn(q, k, v)

    else:
      sharded_fn = jax.shard_map(
          wrap_ulysses_ring_attention,
          mesh=internal_mesh,
          in_specs=(internal_q_axis_names, internal_kv_axis_names, internal_kv_axis_names, internal_mask_axis_names),
          out_specs=internal_q_axis_names,
          check_vma=False,
      )

      def run_fn(q, k, v):
        return sharded_fn(q, k, v, attention_mask)

    x = _run_chunked_ulysses_attention(
        query,
        key,
        value,
        num_heads,
        num_ulysses_shards,
        self.ulysses_attention_chunks,
        run_fn,
    )
    x = jax.lax.with_sharding_constraint(x, q_axis_names)
    x = x[:, :, :orig_q_seq_len, :]
    x = _reshape_heads_to_head_dim(x)
    return x
