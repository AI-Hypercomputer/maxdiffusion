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

"""Protocols and accumulator data structures for attention kernels and strategies."""

from enum import Enum
from typing import Any, Optional, Protocol
import flax.struct
import jax


class AttentionBackend(str, Enum):
  """Supported attention kernel backends."""

  DOT_PRODUCT = "dot_product"
  FLASH = "flash"
  TOKAMAX_FLASH = "tokamax_flash"
  TOKAMAX_RING = "tokamax_ring"
  TOKAMAX_RING_CUSTOM = "tokamax_ring_custom"
  CUSTOM = "custom"
  TOKAMAX = "tokamax"
  CUDNN_FLASH_TE = "cudnn_flash_te"
  ULYSSES = "ulysses"
  ULYSSES_CUSTOM = "ulysses_custom"
  ULYSSES_RING = "ulysses_ring"
  ULYSSES_RING_CUSTOM = "ulysses_ring_custom"
  ULYSSES_CUSTOM_FIXED_M = "ulysses_custom_fixed_m"
  ULYSSES_RING_CUSTOM_FIXED_M = "ulysses_ring_custom_fixed_m"
  ULYSSES_RING_CUSTOM_BIDIR = "ulysses_ring_custom_bidir"


@flax.struct.dataclass
class SplashAccumulators:
  """Un-normalized accumulator contract for Ring attention in LSE or linear space.

  Attributes:
    numerator: Un-normalized attention output `(num_heads, seq_len, head_dim)`.
    max_logit: Per-row maximum logit `(num_heads, seq_len)`.
    denominator: Linear softmax denominator `(num_heads, seq_len)`.
  """

  numerator: jax.Array
  max_logit: jax.Array
  denominator: jax.Array


class LocalAttentionKernel(Protocol):
  """Protocol for single-device or local-shard attention kernels returning normalized output."""

  def __call__(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      *,
      block_sizes: Any,
      q_seq_len: int,
      kv_seq_len: int,
      use_base2_exp: bool = True,
      mk: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Computes attention for a local shard returning normalized output `[H, S_q, D_v]`."""
    ...


class LocalAccumulatorKernel(Protocol):
  """Protocol for kernels that return unnormalized accumulators for ring merging."""

  def call_with_accumulators(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      *,
      block_sizes: Any,
      q_seq_len: int,
      kv_seq_len: int,
      use_base2_exp: bool = True,
      mk: Optional[jax.Array] = None,
      **kwargs,
  ) -> SplashAccumulators:
    """Computes attention accumulators for ring merging."""
    ...


class AttentionStrategy(Protocol):
  """Protocol for distributed attention orchestration strategies."""

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
    """Computes distributed or local attention."""
    ...

  def pre_all_to_all_hook(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      context: Optional[dict[str, Any]] = None,
  ) -> Optional[tuple[jax.Array, jax.Array]]:
    """Computes optional Cauchy-Schwarz local norms before sequence all-to-all communication."""
    ...
