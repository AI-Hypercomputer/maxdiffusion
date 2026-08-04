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

"""SingleShardStrategy for single-device or intra-shard attention compute."""

import inspect
from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp
from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import splash_attention_kernel
from maxdiffusion.models.attention_utils import (
    _build_padding_segment_ids,
    _pad_data_for_flash,
    _extract_custom_block_sizes,
    LOG2E,
)
from maxdiffusion.models.attention_strategies.protocol import AttentionBackend


class SingleShardStrategy:
  """Executes single-shard / local attention across the batch dimension."""

  def __init__(
      self,
      local_kernel: Callable,
      block_sizes: Any,
      use_base2_exp: bool = True,
      use_experimental_scheduler: bool = False,
      vmem_limit_bytes: Optional[int] = None,
      use_fixed_m: bool = False,
      backend: AttentionBackend = AttentionBackend.TOKAMAX,
  ):
    self.local_kernel = local_kernel
    self.block_sizes = block_sizes
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.vmem_limit_bytes = vmem_limit_bytes
    self.use_fixed_m = use_fixed_m
    self.backend = backend
    if callable(local_kernel):
      sig = inspect.signature(local_kernel)
      self._kernel_params = set(sig.parameters.keys())
      self._accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    else:
      self._kernel_params = None
      self._accepts_var_kwargs = True

  def pre_all_to_all_hook(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      context: Optional[dict[str, Any]] = None,
  ) -> Optional[tuple[jax.Array, jax.Array]]:
    """No pre-all-to-all Cauchy-Schwarz bounds needed for single shard."""
    return None

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      q_seq_len: int,
      kv_seq_len: int,
      heads_per_tile: int = 1,
      mk_arr: Optional[jax.Array] = None,
      swap_ds_axes: bool = True,
      attention_mask: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Invokes the local attention kernel vmapped over the batch dimension."""
    if self.backend == AttentionBackend.CUSTOM:
      if attention_mask is not None:
        raise NotImplementedError(
            "The custom dense splash kernel does not support attention_mask "
            "(padding is handled via orig_seq_len); got a non-None attention_mask."
        )
      # 1. K-smoothing precondition for single-shard / R=1 Fixed-M
      if self.use_fixed_m:
        key = key - jnp.mean(key, axis=2, keepdims=True)

      custom_bs = _extract_custom_block_sizes(self.block_sizes or kwargs.get("block_sizes"))
      bsizes = custom_splash._BlockSizes(
          block_q=custom_bs.bq,
          block_kv=custom_bs.bkv,
          block_kv_compute=custom_bs.bkv_compute,
          block_kv_compute_in=custom_bs.bkv_compute_in,
      )

      query_pad, kv_size, query_seq_len = _pad_data_for_flash(query, query.shape[1], custom_bs.bq)
      key_pad, _, key_seq_len = _pad_data_for_flash(key, key.shape[1], custom_bs.bkv)
      value_pad, _, _ = _pad_data_for_flash(value, value.shape[1], custom_bs.bkv)

      if self.use_base2_exp:
        query_pad = query_pad * LOG2E

      # 2. Local Cauchy-Schwarz norm calculation for Fixed-M gate
      if mk_arr is None and self.use_fixed_m:
        qf = query_pad.astype(jnp.float32)
        kf = key_pad.astype(jnp.float32)
        qn_max = jnp.sqrt((qf * qf).sum(-1)).max(axis=(0, 2))  # (local_heads,)
        mk_h = jnp.sqrt((kf * kf).sum(-1)).max(axis=(0, 2))  # (local_heads,)
        fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
        mk_arr = jnp.stack([mk_h, fixed_ok])  # (2, local_heads)

      splash_kernel = self.local_kernel(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          heads_per_tile=heads_per_tile,
          use_base2_exp=self.use_base2_exp,
          use_experimental_scheduler=self.use_experimental_scheduler,
          vmem_limit_bytes=self.vmem_limit_bytes,
          use_fixed_m=self.use_fixed_m,
      )

      if self.use_fixed_m:
        vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))
        attention_output = vmapped_splash(query_pad, key_pad, value_pad, mk_arr)
      else:
        vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0))
        attention_output = vmapped_splash(query_pad, key_pad, value_pad)

      attention_output = jnp.swapaxes(attention_output, 2, 3)
      attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)
      return attention_output

    else:
      # Tokamax / standard splash kernel
      candidate_kwargs = {
          "block_sizes": self.block_sizes,
          "orig_q_seq_len": q_seq_len,
          "orig_kv_seq_len": kv_seq_len,
          "heads_per_tile": heads_per_tile,
          "use_base2_exp": self.use_base2_exp,
          "use_experimental_scheduler": self.use_experimental_scheduler,
          "vmem_limit_bytes": self.vmem_limit_bytes,
          "use_fixed_m": self.use_fixed_m,
      }
      candidate_kwargs.update(kwargs)
      if self._kernel_params is not None and not self._accepts_var_kwargs:
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in self._kernel_params}
      else:
        filtered_kwargs = candidate_kwargs

      kernel_fn = self.local_kernel(**filtered_kwargs)

      block_q = getattr(self.block_sizes, "block_q", 512)
      block_kv = getattr(self.block_sizes, "block_kv", 512)
      query_pad, kv_size, query_seq_len = _pad_data_for_flash(query, query.shape[1], block_q)
      key_pad, _, key_seq_len = _pad_data_for_flash(key, key.shape[1], block_kv)
      value_pad, _, _ = _pad_data_for_flash(value, value.shape[1], block_kv)

      segment_ids = _build_padding_segment_ids(
          query_seq_len,
          query_pad.shape[2],
          key_seq_len,
          key_pad.shape[2],
          attention_mask,
          splash_attention_kernel.SegmentIds,
      )
      vmapped = jax.vmap(kernel_fn, in_axes=(0, 0, 0, None))
      out = vmapped(query_pad, key_pad, value_pad, segment_ids)
      out = out[:, :, :query_seq_len, :kv_size].astype(query.dtype)
      if (
          swap_ds_axes
          and out.ndim == 4
          and out.shape[2] == query.shape[3]
          and out.shape[3] == query.shape[2]
          and out.shape[2] != out.shape[3]
      ):
        out = jnp.swapaxes(out, 2, 3)
      return out
