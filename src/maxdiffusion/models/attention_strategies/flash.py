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

"""FlashAttentionStrategy for TPU and GPU flash attention kernels."""

import math
from typing import Any, Optional
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from maxdiffusion.kernels.splash_attention import splash_attention_kernel as tokamax_splash_attention_kernel
from maxdiffusion.kernels.splash_attention import ring_attention_kernel as tokamax_ring_attention_kernel
from maxdiffusion.kernels.splash_attention import splash_attention_mask as tokamax_splash_attention_mask
from maxdiffusion.kernels.splash_attention import base as tokamax_splash_base
from maxdiffusion.models import attention_utils
from maxdiffusion.models.attention_strategies.ring import RingAttentionStrategy
from maxdiffusion.models.attention_strategies.protocol import AttentionBackend

LOG2E = math.log2(math.e)
CONTEXT = "context"


class FlashAttentionStrategy:
  """Orchestrates flash attention across TokaMax, custom splash, or Pallas backends."""

  def __init__(
      self,
      backend: str = "tokamax",
      block_sizes: Any = None,
      use_base2_exp: bool = True,
      use_experimental_scheduler: bool = False,
      vmem_limit_bytes: Optional[int] = None,
      use_fixed_m: bool = False,
      mask_padding_tokens: bool = True,
      residual_checkpoint_name: Optional[str] = None,
      num_context_shards: int = 1,
      scale: float = 1.0,
  ):
    self.backend = backend
    self.block_sizes = block_sizes
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.vmem_limit_bytes = vmem_limit_bytes
    self.use_fixed_m = use_fixed_m
    self.mask_padding_tokens = mask_padding_tokens
    self.residual_checkpoint_name = residual_checkpoint_name
    self.num_context_shards = num_context_shards
    self.scale = scale

  def pre_all_to_all_hook(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      context: Optional[dict[str, Any]] = None,
  ) -> Optional[tuple[jax.Array, jax.Array]]:
    """No pre-all-to-all Cauchy-Schwarz bounds needed for standard flash attention."""
    return None

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      q_seq_len: int,
      kv_seq_len: int,
      attention_mask: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Executes flash attention using the specified backend."""
    if self.backend == AttentionBackend.TOKAMAX_RING_CUSTOM:
      strategy = RingAttentionStrategy(
          backend=AttentionBackend.CUSTOM,
          block_sizes=self.block_sizes,
          use_base2_exp=self.use_base2_exp,
          use_experimental_scheduler=self.use_experimental_scheduler,
          vmem_limit_bytes=self.vmem_limit_bytes,
          use_fixed_m=self.use_fixed_m,
      )
      heads = query.shape[1]
      custom_bs = attention_utils._extract_custom_block_sizes(self.block_sizes)
      if custom_bs.heads_per_tile > 1:
        raise NotImplementedError("tokamax_ring_custom currently supports heads_per_tile == 1 only.")
      q_in = query * LOG2E if self.use_base2_exp else query
      query_local, kv_size, query_seq_len = attention_utils._pad_data_for_flash(q_in, heads, custom_bs.bq)
      key_local, _, key_seq_len = attention_utils._pad_data_for_flash(key, heads, custom_bs.bkv)
      value_local, _, _ = attention_utils._pad_data_for_flash(value, heads, custom_bs.bkv)
      out = strategy(
          query_local,
          key_local,
          value_local,
          q_seq_len=query_seq_len,
          kv_seq_len=key_seq_len,
          attention_mask=attention_mask,
      )
      return out[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    heads = query.shape[1]
    uses_fused_kernel = self.block_sizes.use_fused_bwd_kernel
    block_q_sizes = (self.block_sizes.block_q, self.block_sizes.block_q_dkv)
    block_kv_sizes = (self.block_sizes.block_kv, self.block_sizes.block_kv_dkv)
    if uses_fused_kernel:
      block_q_sizes += (self.block_sizes.block_q_dkv,)
      block_kv_sizes += (self.block_sizes.block_kv_dkv,)
    else:
      block_q_sizes += (self.block_sizes.block_q_dq,)
      block_kv_sizes += (self.block_sizes.block_kv_dq,)

    block_q = max(s for s in block_q_sizes if s is not None)
    query_pad, kv_size, query_seq_len = attention_utils._pad_data_for_flash(query, heads, block_q)
    block_kv = max(s for s in block_kv_sizes if s is not None)
    key_pad, _, key_seq_len = attention_utils._pad_data_for_flash(key, heads, block_kv)
    value_pad, _, _ = attention_utils._pad_data_for_flash(value, heads, block_kv)

    mask = splash_attention_mask.FullMask(_shape=(query_pad.shape[2], key_pad.shape[2]))
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query_pad.shape[1])

    segment_ids_cls = (
        tokamax_splash_base.SegmentIds
        if self.backend == AttentionBackend.TOKAMAX_RING
        else splash_attention_kernel.SegmentIds
    )
    segment_ids = attention_utils._build_padding_segment_ids(
        query_seq_len,
        query_pad.shape[2],
        key_seq_len,
        key_pad.shape[2],
        attention_mask,
        segment_ids_cls,
    )

    if self.backend == AttentionBackend.TOKAMAX_FLASH:
      mask = tokamax_splash_attention_mask.FullMask(
          _shape=(query_pad.shape[2], key_pad.shape[2]),
      )
      splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
          mask=mask,
          q_seq_shards=1,
          config=attention_utils.convert_to_tokamax_splash_config(
              self.block_sizes,
              residual_checkpoint_name=self.residual_checkpoint_name,
              use_base2_exp=self.use_base2_exp,
              use_experimental_scheduler=self.use_experimental_scheduler,
          ),
          save_residuals=False,
      )
    elif self.backend == AttentionBackend.TOKAMAX_RING:
      mask = tokamax_splash_attention_mask.FullMask(
          _shape=(query_pad.shape[2], key_pad.shape[2]),
      )
      splash_kernel = tokamax_ring_attention_kernel.make_ring_attention(
          mask=mask,
          is_mqa=False,
          config=attention_utils.convert_to_tokamax_splash_config(
              self.block_sizes,
              residual_checkpoint_name=self.residual_checkpoint_name,
              use_base2_exp=self.use_base2_exp,
              use_experimental_scheduler=self.use_experimental_scheduler,
          ),
          save_residuals=False,
          ring_axis=CONTEXT,
          rotate_segment_ids=False,
      )
    else:
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=self.block_sizes,
          save_residuals=True if "ring" in self.backend else False,
          residual_checkpoint_name=self.residual_checkpoint_name,
      )

    segment_ids_in_axes = 0 if attention_mask is not None else None
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
    if not self.mask_padding_tokens:
      segment_ids = None

    attention_output = vmapped_splash(query_pad, key_pad, value_pad, segment_ids)
    return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)
