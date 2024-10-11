"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os

from typing import Union, Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn

class BaseLoRALayer():
  """
  Base LoRA layer class for all LoRA layer implementation
  """
  pass

class LoRALinearLayer(nn.Module, BaseLoRALayer):
  """
  Implements LoRA linear layer
  """
  out_features: int
  rank: int = 0
  network_alpha: float = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  axis_names=('embed', 'heads') #default for qkv and to_out, set it otherwise.

  @nn.compact
  def __call__(self, h, hidden_states):
    if self.rank > self.out_features:
      raise ValueError(f"LoRA rank {self.rank} must be less or equal to {min(self.in_features, self.out_features)}")
    
    down_hidden_states = nn.Dense(
      features=self.rank,
      use_bias=False,
      kernel_init=nn.initializers.kaiming_uniform(),
      # kernel_init=nn.with_logical_partitioning(
      #   nn.initializers.kaiming_uniform(),
      #   self.axis_names
      # ),
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      name="down"
    )(hidden_states)
    up_hidden_states = nn.Dense(
      features=self.out_features,
      use_bias=False,
      kernel_init=nn.initializers.zeros_init(),
      # kernel_init=nn.with_logical_partitioning(
      #   nn.initializers.zeros_init(),
      #   self.axis_names
      # ),
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      name="up"
    )(down_hidden_states)
    if self.network_alpha:
      up_hidden_states *= self.network_alpha / self.rank
    
    return h + up_hidden_states

class LoRAConv2DLayer(nn.Module):
  """
  Implements LoRA Conv layer
  """
  out_features: int
  rank: int = 4
  kernel_size: Union[int, Tuple[int, int]] = (1,1)
  strides: Union[int, Tuple[int, int]] = (1, 1)
  padding: Union[int, Tuple[int, int], str] = 0
  network_alpha: Optional[float] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, h, hidden_states):
    down_hidden_states = nn.Conv(
      self.rank,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      kernel_init=nn.initializers.kaiming_uniform(),
      # kernel_init=nn.with_logical_partitioning(
      #   nn.initializers.kaiming_uniform(),,
      #   ("keep_1", "keep_2", "conv_in", "conv_out")
      # ),
      precision=self.precision,
      name="down"
    )(hidden_states)

    up_hidden_states = nn.Conv(
      self.out_features,
      use_bias=False,
      kernel_size=(1, 1),
      strides=(1, 1),
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      kernel_init=nn.initializers.zeros_init(),
      # kernel_init=nn.with_logical_partitioning(
      #   nn.initializers.zeros_init(),
      #   ("keep_1", "keep_2", "conv_in", "conv_out")
      # ),
      precision=self.precision,
      name="up"
    )(down_hidden_states)

    if self.network_alpha:
      up_hidden_states *= self.network_alpha / self.rank
    
    return  h + up_hidden_states