# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantization calibration methods."""

import abc
from typing import Union, Sequence
from aqt.jax.v2 import utils
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class Calibration(abc.ABC):
  """Abstract class for calibration."""

  @abc.abstractmethod
  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None
  ) -> jnp.ndarray:
    pass


@utils.flax_slots_kw_only_dataclass
class ConstantCalibration(Calibration):
  """Calibration with a constant value."""

  bound: Union[jnp.ndarray, float]

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    """Calibration."""
    del shared_axes, context
    assert self.bound > 0, 'Bound should be positive.'
    # TODO(yichizh): hardcode bf16 for the scales, subject to quality evaluation
    return jnp.asarray(self.bound).reshape((1,) * len(x.shape)).astype(x.dtype)


@utils.flax_slots_kw_only_dataclass
class AbsMaxCalibration(Calibration):
  """Simple max(abs(x)) calibration.

  Attributes:
    scale: Set it to something like 0.3, 0.1, 0.03. If scale < 1.0, setting
      IntNumerics.clip_gradient=True is likely to be important.
  """

  scale: float | None = None

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    """Calibration."""
    del context

    msg = (
        'Perhaps you are using DequantMode.THIS_INPUT (fake_quant) and forgot'
        ' to set them.'
    )
    assert shared_axes is not None, msg

    # NOTE: If you want to clip, consider using clip and clip_gradient in
    # int_numerics.IntNumerics.
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    # TODO(yichizh): the zero filtering is not needed anymore because inf is
    # filtered when calculating the reciprocal of scaline factor
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    if self.scale is not None:
      abs_max = abs_max * self.scale
    return abs_max.astype(x.dtype)
