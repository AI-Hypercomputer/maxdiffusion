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

"""Attention communication strategies and protocols for maxdiffusion."""

from maxdiffusion.models.attention_utils import CustomBlockSizes
from maxdiffusion.models.attention_strategies.protocol import (
    AttentionBackend,
    AttentionStrategy,
    LocalAccumulatorKernel,
    LocalAttentionKernel,
    SplashAccumulators,
)
from maxdiffusion.models.attention_strategies.single_shard import SingleShardStrategy
from maxdiffusion.models.attention_strategies.ring import RingAttentionStrategy
from maxdiffusion.models.attention_strategies.ulysses import UlyssesStrategy
from maxdiffusion.models.attention_strategies.dot_product import DotProductAttentionStrategy
from maxdiffusion.models.attention_strategies.flash import FlashAttentionStrategy
from maxdiffusion.models.attention_strategies.custom_ring import make_custom_ring_attention

__all__ = [
    "AttentionBackend",
    "AttentionStrategy",
    "CustomBlockSizes",
    "LocalAccumulatorKernel",
    "LocalAttentionKernel",
    "SplashAccumulators",
    "SingleShardStrategy",
    "RingAttentionStrategy",
    "UlyssesStrategy",
    "DotProductAttentionStrategy",
    "FlashAttentionStrategy",
    "make_custom_ring_attention",
]
