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
"""DotGeneralQuantizer for GPTQ."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - hinv - Inverse of the hessian

from typing import Sequence
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import int_numerics
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from rigl.projects.bigsparse import sharded_gptc


def _get_quant_mode(context: utils.Context | None) -> utils.QuantMode:
  return utils.QuantMode.TRAIN if context is None else context.quant_mode


def _get_divisible_blocksize(dim: int, blocksize_top: int) -> int:
  """Returns the blocksize which could divide the given dimension.

  Returns the largest blocksize which is smaller than or equal to
  blocksize_top.

  Args:
    dim: Target dimension which should be divided.
    blocksize_top: Maximal dimension size.

  Returns:
    The largest blocksize which is smaller than blocksize_top, and which could
    divide the given dim.
  """
  blocksize = blocksize_top
  while dim % blocksize != 0:
    blocksize -= 1

  return blocksize


def _reshape_kernel_for_gptq(
    kernel: jnp.ndarray,
    ca: Sequence[utils.AxisIdx],
    sharding_axes: str | None,
    act_order: bool,
    perm: Sequence[utils.AxisIdx] | None,
    blocksize: int,
) -> tuple[jnp.ndarray, Sequence[int]]:
  """Reshapes kernel to (features / blocksize, blocksize, -1) and potentially reshard."""
  # TODO(dhchoi): Consider using kernel[XXX:] form, check its efficiency.
  for new_axis_idx, original_axis_idx in enumerate(sorted(ca)):
    kernel = jnp.moveaxis(kernel, original_axis_idx, new_axis_idx)

  kernel_feature_grouped_shape = kernel.shape
  kernel = kernel.reshape((np.prod(kernel.shape[: len(ca)]), -1))
  if act_order:
    assert perm is not None
    kernel = kernel[perm, :]

  kernel = sharded_gptc.maybe_shard(
      kernel,
      jax.sharding.PartitionSpec(sharding_axes, None),
      sharding_axes,
  )
  kernel = jnp.float32(kernel)  # convert to higher precision after sharding

  return (
      kernel.reshape((-1, blocksize, kernel.shape[-1])),
      kernel_feature_grouped_shape,
  )


def _recover_kernel_from_gptq_result(
    kernel: jnp.ndarray,
    ca: Sequence[utils.AxisIdx],
    sharding_axes: str | None,
    act_order: bool,
    perm: Sequence[utils.AxisIdx] | None,
    kernel_dtype: jnp.dtype,
    kernel_feature_grouped_shape: Sequence[int]
) -> jnp.ndarray:
  """Recovers original kernel shape."""
  kernel = kernel.astype(kernel_dtype)
  if act_order:
    assert perm is not None
    invperm = jnp.argsort(perm)  # pytype: disable=wrong-arg-types
    kernel = kernel.reshape((-1, kernel.shape[-1]))
    kernel = kernel[invperm, :]
  kernel = kernel.reshape(kernel_feature_grouped_shape)

  for original_axis_idx, new_axis_idx in list(enumerate(ca))[::-1]:
    kernel = jnp.moveaxis(kernel, original_axis_idx, new_axis_idx)

  kernel = sharded_gptc.maybe_shard(kernel, None, sharding_axes)
  return kernel


def _init_hinv_for_calibration(inputs, perc_damp=0.01):
  """Initializes hinv with a damping term."""
  features = np.prod(inputs.shape[2:])
  diag = jnp.mean(jnp.float32(inputs) ** 2, (0, 1)).flatten()
  damp = perc_damp * jnp.mean(diag)
  return jnp.diag(jnp.ones(features) / damp)


class GptqHinvCollector(nn.Module):
  """GPTQ hinv collector module.

  Written as a separated module since if we make the GptqDotGeneralQuantizer as
  a module, we cannot inject its instance (module cannot be initialized outside
  of flax.init / flax.apply). If we inject the type directly, then we cannot
  use its member utility functions (ex. swap_lhs_and_rhs) during configuration.
  """
  quant_collection: str

  sharding_axes: str | None

  # Percentage of damping during hinv initialization.
  perc_damp: float = 0.01

  # If act_order is set, weights with large corresponding activation values are
  # updated first.
  act_order: bool = False

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      ca: Sequence[utils.AxisIdx],
      quant_mode: utils.QuantMode,
  ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Collects Inverse of the hessian.

    Args:
      x: Activation using which the hessian will be collected.
      ca: Contracting axis.
      quant_mode: Quantization mode.
    Returns:
      Collected hessian inverse.
    """
    # The GPTQ assumes that 2+ dimensions are all the reducing dimension.
    # We should reshape so that the shape is something like BS X S X D, where
    # D is the contracting dimension.
    transpose = [axis for axis in range(x.ndim) if axis not in ca] + list(ca)
    x = x.transpose(transpose)
    x = x.reshape((-1, np.prod(x.shape[-len(ca) :])))
    x = sharded_gptc.maybe_shard(
        x,
        jax.sharding.PartitionSpec(None, self.sharding_axes),
        self.sharding_axes,
    )

    blocksize = _get_divisible_blocksize(x.shape[0], sharded_gptc.BLOCKSIZE)
    x = x.reshape((-1, blocksize, x.shape[-1]))

    is_initializing = not self.has_variable(
        self.quant_collection, "num_calibrated_batches"
    )
    num_calibrated_batches = self.variable(
        self.quant_collection,
        "num_calibrated_batches",
        jnp.zeros,
        (),
        jnp.int32,
    )

    hinv = None
    if not is_initializing:
      # Collect hinv from the previous calibration step.
      hinv = self.get_variable(self.quant_collection, "collected_hinv")
      hinv = jax.lax.cond(
          num_calibrated_batches.value == 0,
          lambda: _init_hinv_for_calibration(x, self.perc_damp),
          lambda: hinv
      )

    # This function, when called with hinv = None, calculates
    # (D + (X^T @ X) / bs)^-1, where bs = batch size and D is damping term.
    # When called with hinv != None, it aggregates the impact of additional
    # (X^T @ X) / bs on the hinv. This is possible since the the impact of each
    # data instance is accumulated using Woodbury formula on the hinv.
    hinv, perm = sharded_gptc.compute_hinv(
        x,
        perc_damp=self.perc_damp,
        sharding_axes=self.sharding_axes,
        act_order=self.act_order,
        blocksize=blocksize,
        hinv=hinv,
    )

    collected_hinv = self.variable(
        self.quant_collection,
        "collected_hinv",
        jnp.zeros,
        hinv.shape,
        hinv.dtype,
    )

    if quant_mode == utils.QuantMode.CALIBRATE and not is_initializing:
      collected_hinv.value = hinv
      num_calibrated_batches.value = num_calibrated_batches.value + 1

    # The collected hinv will be (D + (C^T @ C) / bs)^-1, where C is the whole
    # calibration data with instance number = bs * bn (bn is batch num).
    # We multiply the collected hinv by bn. By doing so, the result will be
    # bn * (D + (C^T @ C) / bs)^-1 = (D / bn + (C^T @ C) / (bs * bn))^-1).
    # Althogh the damping term is divided by bn, the values calculated using the
    # calibration data remains the same with the case when the whole calibration
    # data is used at once to calculate the hinv. In addition, we can manually
    # configure perc_damp (divide it by the number of calibration batches) to
    # approximate it more precise.
    return collected_hinv.value * num_calibrated_batches.value, perm


@utils.flax_slots_kw_only_dataclass
class GptqDotGeneralQuantizer(aqt_dot_general.DefaultDotGeneralQuantizer):
  """GPTQ dot_general quantizer."""

  sharding_axes: str | None = utils.static_field()

  quant_collection: str = utils.static_field()

  # Percentage of damping during hinv initialization.
  perc_damp: float = utils.static_field(default=0.01)

  # If act_order is set, weights with large corresponding activation values are
  # updated first.
  act_order: bool = utils.static_field(default=False)

  # Boolean flag to see which side of argument is the kernel.
  is_rhs_kernel: bool = utils.static_field(default=True)

  def calibrate(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[int]],
      rhs_quantization_info: tuple[jax.Array, Sequence[int]],
  ) -> tuple[
      tuple[jax.Array, aqt_tensor.QTensor],
      tuple[jax.Array, aqt_tensor.QTensor]
  ]:
    """GPTQ calibration.

    Majority of the codes are copied from sharded_gptc.gptc_skeleton. we copied
    the code snippet to enable the collection of hinv statistics.

    Args:
      lhs_quantization_info: left argument and its contracting axis.
      rhs_quantization_info: right argument and its contracting axis:
    Returns:
      Updated weight and incomplete QTensor for lhs and rhs.
    """
    # Follow the quantization mode and num_bits of the kernel.
    if self.is_rhs_kernel:
      quant_mode = _get_quant_mode(self.rhs.context)
      assert isinstance(self.rhs.numerics, int_numerics.IntNumerics)
      num_bits = self.rhs.numerics.bits
    else:
      quant_mode = _get_quant_mode(self.lhs.context)
      assert isinstance(self.lhs.numerics, int_numerics.IntNumerics)
      num_bits = self.lhs.numerics.bits

    if quant_mode == utils.QuantMode.TRAIN:
      # During training, we should not allow collecting hinvs and updating
      # weights using it.
      return super(GptqDotGeneralQuantizer, self).calibrate(  # pytype: disable=attribute-error
          lhs_quantization_info, rhs_quantization_info
      )

    if not self.is_rhs_kernel:
      # Swap so that rhs is kernel.
      lhs_quantization_info, rhs_quantization_info = (
          rhs_quantization_info,
          lhs_quantization_info,
      )

    lhs, lhs_ca = lhs_quantization_info
    rhs, rhs_ca = rhs_quantization_info

    hinv_collector = GptqHinvCollector(
        quant_collection=self.quant_collection,
        sharding_axes=self.sharding_axes,
        perc_damp=self.perc_damp,
        act_order=self.act_order,
    )
    hinv, perm = hinv_collector(lhs, lhs_ca, quant_mode)

    # Cholesky decomposition.
    blocksize = _get_divisible_blocksize(hinv.shape[0], sharded_gptc.BLOCKSIZE)
    hinv = sharded_gptc.cholesky(
        hinv, sharding_axes=self.sharding_axes, blocksize=blocksize
    )

    # Reshape rhs to (features, -1) and potentially reshard
    rhs_dtype = rhs.dtype

    rhs, rhs_feature_grouped_shape = _reshape_kernel_for_gptq(
        rhs, rhs_ca, self.sharding_axes, self.act_order, perm, blocksize
    )
    hinv = hinv.reshape((-1, blocksize, hinv.shape[-1]))

    def find_qparams(w, solve=None):
      # Fake-Quant calculator used by the GPTQ.
      # W is expected to have the shape C X D, while C is the contracting.
      del solve

      # Add a smoothing term to prevent it from breaking by having max value 0
      # in some block. (Ex: when w is identity + subchannel)
      return sharded_gptc.absmax_params(w + 1e-7, bits=num_bits)

    # The GPTQ:
    # 1. Fake-quant the weight.
    # 2. Calculate quantization error of the weight.
    # 3. Outer-product hessian inverse and quantization error; update W using
    #    the product value.
    # 4. Weight is updated sequentially.
    rhs, _ = sharded_gptc.gptq(rhs, hinv, find_qparams)

    # Recover original rhs shape.
    rhs = _recover_kernel_from_gptq_result(
        rhs,
        rhs_ca,
        self.sharding_axes,
        self.act_order,
        perm,
        rhs_dtype,
        rhs_feature_grouped_shape,
    )

    # Restore lhs and rhs position.
    if not self.is_rhs_kernel:
      lhs, rhs = rhs, lhs
      lhs_ca, rhs_ca = rhs_ca, lhs_ca

    # Retrieve the scales using the updated lhs and rhs.
    return super(GptqDotGeneralQuantizer, self).calibrate(  # pytype: disable=attribute-error
        (lhs, lhs_ca), (rhs, rhs_ca)
    )

  def swap_lhs_and_rhs(self) -> None:
    """Swaps lhs and rhs configuration."""
    self.lhs, self.rhs = self.rhs, self.lhs
    self.is_rhs_kernel = not self.is_rhs_kernel
