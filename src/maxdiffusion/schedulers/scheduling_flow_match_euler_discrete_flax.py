"""
 Copyright 2025 Google LLC

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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    CommonSchedulerState,
    # FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    broadcast_to_shape_from_left,
)


@flax.struct.dataclass
class FlowMatchEulerDiscreteSchedulerState:
  common: CommonSchedulerState


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
  state: FlowMatchEulerDiscreteSchedulerState


class FlowMatchEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
  # _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

  dtype: jnp.dtype

  @property
  def has_state(self):
    return True

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      shift: float = 1.0,
      use_dynamic_shifting: bool = False,
      base_shift: Optional[float] = 0.5,
      max_shift: Optional[float] = 1.15,
      base_image_seq_len: Optional[int] = 256,
      max_image_seq_len: Optional[int] = 4096,
      invert_sigmas: bool = False,
      shift_terminal: Optional[float] = None,
      use_karras_sigmas: Optional[bool] = False,
      use_exponential_sigmas: Optional[bool] = False,
      use_beta_sigmas: Optional[bool] = False,
      time_shift_type: str = "exponential",
      dtype: jnp.dtype = jnp.float32,
  ):
    self.dtype = dtype

  def create_state(self, common: Optional[CommonSchedulerState] = None) -> FlowMatchEulerDiscreteSchedulerState:
    if common is None:
      common = CommonSchedulerState.create(self)
