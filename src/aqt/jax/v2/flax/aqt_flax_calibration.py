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
"""Static range calibration."""

from typing import Sequence

from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
import flax.linen as nn
from jax import numpy as jnp


_SUM_OF_ONES = "sum_of_ones"
_SUM_OF_VALS = "sum_of_vals"
_MAX_OF_ABS_VALS = "max_of_abs_vals"
_SUM_OF_L1_VALS = "sum_of_l1_vals"
_SUM_OF_LP_VALS = "sum_of_lp_vals"


@utils.flax_slots_kw_only_dataclass
class MeanOfAbsMaxCalibration(calibration.Calibration, nn.Module):
  """State for static range AQT/PTQ/QAT."""

  quant_collection: str

  @nn.compact
  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    quant_mode = context.quant_mode if context else utils.QuantMode.TRAIN

    # Initialize the max_stat.value as 0; does not update the max_stat value
    # during initialization.
    is_initializing = not self.has_variable(self.quant_collection, "sum_of_max")

    sum_of_max = self.variable(
        self.quant_collection,
        "sum_of_max",
        jnp.zeros,
        abs_max.shape,
        abs_max.dtype,
    )
    count = self.variable(
        self.quant_collection, "count", jnp.zeros, (), jnp.int32
    )

    if quant_mode == utils.QuantMode.CALIBRATE and not is_initializing:
      sum_of_max.value = sum_of_max.value + abs_max
      count.value = count.value + 1

    # TODO(dhchoi): Introduce checker for count.value.
    # This assertion should be done on jitted graph, and this is not
    # straightforward (or does not have a good option).
    # One option is jax.experimental.checkify
    # (https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html), but
    # we need to transform the whole function with jax.experimental.checkify to
    # use this feature.
    # Maybe wait for the JAX language upgrade to have a better support for this?
    return sum_of_max.value / count.value


# TODO: b/335764538 - Check the math correctness of the module.
class WeightedStatsCalibration(calibration.Calibration, nn.Module):
  """Migration of AQTv1 calibration to AQTv2."""

  quant_collection: str

  # Calibration coefficients
  l1_dev_coeff: float = 0.0
  lp_dev_coeff: float = 0.0
  max_dev_coeff: float = 0.0
  const_bound_coeff: float = 0.0

  # Priors
  update_count_prior: float = 1.0
  max_dev_prior: float = 0.0
  mean_prior: float = 0.0
  l1_dev_prior: float = 0.0
  lp_dev_prior: float = 0.0

  lp_order: int = 2
  safe_divide: bool = False
  filter_zeros: bool = True
  tpu_cross_replica_sum: bool = False
  ema_update_count: float = 100

  def _get_value(self, name: str):
    return self.get_variable(self.quant_collection, name)

  def _mean(self) -> jnp.ndarray:
    sum_of_vals = self._get_value(_SUM_OF_VALS)
    sum_of_ones = self._get_value(_SUM_OF_ONES)
    return self._divide(sum_of_vals, sum_of_ones)

  def _max_dev(self) -> jnp.ndarray:
    return self._get_value(_MAX_OF_ABS_VALS)

  def _l1_dev(self) -> jnp.ndarray:
    sum_of_l1_vals = self._get_value(_SUM_OF_L1_VALS)
    sum_of_ones = self._get_value(_SUM_OF_ONES)
    return self._divide(sum_of_l1_vals, sum_of_ones)

  def _lp_dev(self) -> jnp.ndarray:
    sum_of_lp_vals = self._get_value(_SUM_OF_LP_VALS)
    sum_of_ones = self._get_value(_SUM_OF_ONES)
    if self.lp_order == 2:
      # sqrt() is numerically more accurate
      return jnp.sqrt(self._divide(sum_of_lp_vals, sum_of_ones))
    else:
      # TODO(b/205769820): Make sure if the output of pow op below is
      # numerically valid.
      return self._divide(sum_of_lp_vals, sum_of_ones) ** (1.0 / self.lp_order)

  def _update_var(
      self,
      var: nn.Variable,
      s: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      weight: jnp.ndarray | None = None,
      reduce_max: bool = False,
  ):
    """Updates the given Flax variable."""
    if weight is not None:
      if reduce_max:
        # A maximum is a maximum, regardless of its nonzero weight.
        s = jnp.where(weight > 0, s, 0)
      else:
        s = s * weight

    reduce_fn = jnp.max if reduce_max else jnp.sum
    s = reduce_fn(s, axis=shared_axes, keepdims=True)
    if self.tpu_cross_replica_sum:
      raise NotImplementedError(
          "support for tpu_cross_replica_sum=True is not implemented"
      )
    rate = 1.0 / self.ema_update_count
    var.value = (1.0 - rate) * var.value + rate * s

  def _divide(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    if self.safe_divide:
      # Safe-divide without division-by-zero.
      res = jnp.where(y != 0, x / y, jnp.zeros_like(x))
      return res

    return jnp.divide(x, y)

  @nn.compact
  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    if self.lp_order > 30:
      raise NotImplementedError("For higher norms we should add stabilization.")

    quant_mode = context.quant_mode if context else utils.QuantMode.TRAIN
    stats_shape = list(x.shape)
    if shared_axes is not None:
      for axis in shared_axes:
        stats_shape[axis] = 1

    def _make_var(name: str, init_val: float) -> nn.Variable:
      def init_var_fn(init_val: float) -> jnp.ndarray:
        return jnp.full(stats_shape, init_val, dtype=jnp.float32)

      return self.variable(self.quant_collection, name, init_var_fn, init_val)

    sum_of_ones = _make_var(_SUM_OF_ONES, self.update_count_prior)
    sum_of_vals = _make_var(
        _SUM_OF_VALS, self.mean_prior * self.update_count_prior
    )
    max_of_abs_vals = _make_var(_MAX_OF_ABS_VALS, self.max_dev_prior)
    sum_of_l1_vals = _make_var(
        _SUM_OF_L1_VALS, self.l1_dev_prior * self.update_count_prior
    )
    sum_of_lp_vals = _make_var(
        _SUM_OF_LP_VALS,
        self.lp_dev_prior**self.lp_order * self.update_count_prior,
    )

    if quant_mode in [utils.QuantMode.TRAIN, utils.QuantMode.CALIBRATE]:
      # Update variables.

      # Layers such as ReLU emit zeros often. In such cases, we can model
      # the non-sparse distribution of weights separately, resulting in
      # unbiased estimation of non-sparse mean l1 and lp.
      # This clips away less of the distribution of inputs.
      if self.filter_zeros:
        ones = jnp.where(x != 0, 1.0, 0.0)
      else:
        ones = jnp.ones_like(x)

      px = x if self.lp_order % 2 == 0 else jnp.abs(x)

      self._update_var(sum_of_ones, ones, shared_axes)
      self._update_var(sum_of_vals, x, shared_axes)
      self._update_var(
          max_of_abs_vals, jnp.abs(x), shared_axes, reduce_max=True
      )
      self._update_var(sum_of_l1_vals, jnp.abs(x), shared_axes)
      self._update_var(sum_of_lp_vals, px**self.lp_order, shared_axes)

    return (
        self.l1_dev_coeff * self._l1_dev()
        + self.lp_dev_coeff * self._lp_dev()
        + self.max_dev_coeff * self._max_dev()
        + self.const_bound_coeff
    )
