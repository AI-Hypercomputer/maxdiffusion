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
from typing import Any, Callable, Iterable, Tuple, Union


import flax.linen as nn
import jax
import jax.numpy as jnp
# Not sure which initializer to use, ruff was complaining, so added an ignore
# from jax.nn import initializers # noqa: F811


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
Dtype = Any  # this could be a real type?
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            kernel_init = nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ('keep_1', 'keep_2', 'conv_in', 'conv_out')
            )
        )

    @nn.compact
    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )

        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('conv_batch', 'height', 'keep_2', 'out_channels')
        )

        hidden_states = self.conv(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('conv_batch', 'height', 'keep_2', 'out_channels')
        )
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # padding="VALID",
            dtype=self.dtype,
            kernel_init = nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ('keep_1', 'keep_2', 'conv_in', 'conv_out')
            )
        )
    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('conv_batch', 'height', 'keep_2', 'out_channels')
        )
        return hidden_states


class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5)

        self.norm2 = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5)
        self.dropout = nn.Dropout(self.dropout_prob)

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
                kernel_init = nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ('keep_1', 'keep_2', 'conv_in', 'conv_out')
            )
            )
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            kernel_init = nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ('keep_1', 'keep_2', 'conv_in', 'conv_out')
            )
        )

        self.time_emb_proj = nn.Dense(
           out_channels,
           dtype=self.dtype,
           )
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            kernel_init = nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ('keep_1', 'keep_2', 'conv_in', 'conv_out')
            )
        )

    def __call__(self, hidden_states, temb, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('batch', None, None, 'out_channels')
        )

        temb = self.time_emb_proj(nn.swish(temb))
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('conv_batch', 'height', 'keep_2', 'out_channels')
        )

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual
