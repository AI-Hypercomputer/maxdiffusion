#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Common types."""

from typing import Any, Sequence

from flax.linen import partitioning
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

Config = Any

Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]

Mesh = jax.sharding.Mesh
ScanIn = partitioning.ScanIn
BlockSizes = splash_attention_kernel.BlockSizes

AxisNames = tuple[str, ...]
# Physical axis names for device meshes.
DATA = "data"
FSDP = "fsdp"
CONTEXT = "context"
TENSOR = "tensor"
# Logical axis names for model parameters and activations.
BATCH = "activation_batch"
LENGTH = "activation_length"
KV_LENGTH = "activation_kv_length"
EMBED = "activation_embed"
HEAD = "activation_heads"
D_KV = "activation_kv"
KEEP_1 = "activation_keep_1"
KEEP_2 = "activation_keep_2"
CONV_OUT = "activation_conv_out_channels"

WAN2_1 = "wan2.1"
WAN2_2 = "wan2.2"

WAN_MODEL = WAN2_1

# For setting self/cross attention independently in splash kernel
SELF_ATTN_HEAD = "activation_self_attn_heads"
SELF_ATTN_Q_LENGTH = "activation_self_attn_q_length"
SELF_ATTN_KV_LENGTH = "activation_self_attn_kv_length"
CROSS_ATTN_HEAD = "activation_cross_attn_heads"
CROSS_ATTN_Q_LENGTH = "activation_cross_attn_q_length"
CROSS_ATTN_KV_LENGTH = "activation_cross_attn_kv_length"


WAN_MODEL = "Wan2.1"

### Common axis rules for ring attention ###
RING_ATTENTION_AXIS_RULES = [
    [SELF_ATTN_HEAD, None],
    [SELF_ATTN_Q_LENGTH, CONTEXT],
    [SELF_ATTN_KV_LENGTH, CONTEXT],
    [CROSS_ATTN_HEAD, None],
    [CROSS_ATTN_Q_LENGTH, CONTEXT],
    [CROSS_ATTN_KV_LENGTH, CONTEXT],
]

SEQUENCE_PARALLEL_AXIS_RULES = [
    [SELF_ATTN_HEAD, None],
    [SELF_ATTN_Q_LENGTH, CONTEXT],
    [SELF_ATTN_KV_LENGTH, None],
    [CROSS_ATTN_HEAD, None],
    [CROSS_ATTN_Q_LENGTH, CONTEXT],
    [CROSS_ATTN_KV_LENGTH, None],
]
