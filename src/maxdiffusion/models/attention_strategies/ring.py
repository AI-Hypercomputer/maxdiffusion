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

"""RingAttentionStrategy and BidirectionalRingStrategy for distributed attention."""

import math
from typing import Any, Optional, Tuple
import jax

from maxdiffusion.models.attention_utils import (
    _max_row_norm_per_head,
    _build_padding_segment_ids,
    _pad_data_for_flash,
    _extract_custom_block_sizes,
    convert_to_tokamax_splash_config,
)
from maxdiffusion.models.attention_strategies import custom_ring
from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import ring_attention_kernel
from maxdiffusion.kernels.splash_attention import base as tokamax_splash_base
from maxdiffusion.kernels.splash_attention import splash_attention_mask as tokamax_splash_attention_mask
from maxdiffusion.models.attention_strategies.protocol import AttentionBackend

LOG2E = math.log2(math.e)


class RingAttentionStrategy:
  """Ring attention across a mesh axis using standard or custom Pallas kernels."""

  def __init__(
      self,
      block_sizes: Any,
      ring_axis: str = "ring",
      ulysses_axis: str = "ulysses",
      num_ring_shards: int = 1,
      num_ulysses_shards: int = 1,
      use_base2_exp: bool = True,
      use_experimental_scheduler: bool = False,
      vmem_limit_bytes: Optional[int] = None,
      use_fixed_m: bool = False,
      bidirectional: bool = False,
      backend: AttentionBackend = AttentionBackend.CUSTOM,
      mask_value: float = -1e30,
  ):
    self.block_sizes = block_sizes
    self.ring_axis = ring_axis
    self.ulysses_axis = ulysses_axis
    self.num_ring_shards = num_ring_shards
    self.num_ulysses_shards = num_ulysses_shards
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.vmem_limit_bytes = vmem_limit_bytes
    self.use_fixed_m = use_fixed_m
    self.bidirectional = bidirectional
    self.backend = backend
    self.mask_value = mask_value

  def pre_all_to_all_hook(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      context: Optional[dict] = None,
  ) -> Optional[Tuple[jax.Array, jax.Array]]:
    """Computes Cauchy-Schwarz fixed-m norm bounds before Ulysses all-to-all."""
    if not self.use_fixed_m or self.num_ring_shards <= 1:
      return None

    # Reduce on PRE-a2a activation so reduction overlaps all_to_all transfer.
    # The optimization barrier prevents XLA from duplicating the producer chain
    # into the norm fusion.
    query, key = jax.lax.optimization_barrier((q, k))
    qn_local = _max_row_norm_per_head(query)
    kn_local = _max_row_norm_per_head(key)
    if self.use_base2_exp:
      qn_local = qn_local * LOG2E

    env = jax._src.core.get_axis_env()
    axes_to_pmax_q = []
    if env.axis_exists(self.ring_axis):
      axes_to_pmax_q.append(self.ring_axis)
    if env.axis_exists(self.ulysses_axis):
      axes_to_pmax_q.append(self.ulysses_axis)

    qn_all = qn_local
    if axes_to_pmax_q:
      qn_all = jax.lax.pmax(qn_all, tuple(axes_to_pmax_q))

    mk_all = kn_local
    if env.axis_exists(self.ulysses_axis):
      mk_all = jax.lax.pmax(mk_all, self.ulysses_axis)
      heads_per_dev = qn_all.shape[0] // self.num_ulysses_shards
      start_head = jax.lax.axis_index(self.ulysses_axis) * heads_per_dev
      return (
          jax.lax.dynamic_slice_in_dim(qn_all, start_head, heads_per_dev),
          jax.lax.dynamic_slice_in_dim(mk_all, start_head, heads_per_dev),
      )
    else:
      return (qn_all, mk_all)

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      q_seq_len: int,
      kv_seq_len: int,
      heads_per_tile: int = 1,
      mk_arr: Optional[Tuple[jax.Array, jax.Array]] = None,
      attention_mask: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Executes ring attention scan across self.ring_axis."""
    if self.backend == AttentionBackend.CUSTOM:
      if attention_mask is not None:
        raise NotImplementedError(
            "Ulysses custom splash kernels do not support attention_mask (padding is handled via orig_seq_len)."
        )
      block_sizes = kwargs.get("block_sizes", self.block_sizes)
      custom_bs = _extract_custom_block_sizes(block_sizes)
      bsizes = custom_splash._BlockSizes(
          block_q=custom_bs.bq,
          block_kv=custom_bs.bkv,
          block_kv_compute=custom_bs.bkv_compute,
          block_kv_compute_in=custom_bs.bkv_compute_in,
      )

      # Pad tensors to block boundaries for custom splash kernel
      query_pad, kv_size, query_seq_len = _pad_data_for_flash(query, query.shape[1], custom_bs.bq)
      key_pad, _, key_seq_len = _pad_data_for_flash(key, key.shape[1], custom_bs.bkv)
      value_pad, _, _ = _pad_data_for_flash(value, value.shape[1], custom_bs.bkv)

      ring_kernel = custom_ring.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          use_base2_exp=self.use_base2_exp,
          use_experimental_scheduler=self.use_experimental_scheduler,
          vmem_limit_bytes=self.vmem_limit_bytes,
          ring_axis=self.ring_axis,
          ring_size=self.num_ring_shards,
          bidirectional=self.bidirectional,
          use_fixed_m=self.use_fixed_m,
          fixed_m_norms=mk_arr,
      )
      q_in = query_pad * LOG2E if self.use_base2_exp else query_pad
      attention_output = jax.vmap(ring_kernel, in_axes=(0, 0, 0))(q_in, key_pad, value_pad)
      return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)
    elif self.backend == AttentionBackend.TOKAMAX:
      q_padded_len = query.shape[2]
      kv_padded_len = key.shape[2]
      total_kv_len = kv_padded_len * self.num_ring_shards

      mask = tokamax_splash_attention_mask.FullMask(_shape=(q_padded_len, total_kv_len))
      ring_kernel = ring_attention_kernel.make_ring_attention(
          mask=mask,
          is_mqa=False,
          config=convert_to_tokamax_splash_config(
              self.block_sizes,
              use_base2_exp=self.use_base2_exp,
              use_experimental_scheduler=self.use_experimental_scheduler,
          ),
          save_residuals=False,
          ring_axis=self.ring_axis,
          kv_seq_shards=self.num_ring_shards,
          rotate_segment_ids=attention_mask is not None,
      )
      segment_ids = _build_padding_segment_ids(
          q_seq_len,
          q_padded_len,
          kv_seq_len,
          kv_padded_len,
          attention_mask,
          tokamax_splash_base.SegmentIds,
      )
      segment_ids_in_axes = 0 if attention_mask is not None else None
      vmapped_splash = jax.vmap(ring_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
      attention_output = vmapped_splash(query, key, value, segment_ids)
      return attention_output
    else:
      raise ValueError(f"Unsupported backend: {self.backend}. Expected one of: 'custom', 'tokamax'.")
