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
"""Quantized conv_general."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=protected-access

from typing import Optional, Tuple
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
import jax
from jax import lax
import jax.numpy as jnp


def _postprocess_qtensor(
    input_qtensor: Optional[aqt_tensor.QTensor],
    calculated_qtensor: aqt_tensor.QTensor,
) -> aqt_tensor.QTensor:
  """Compute qtensor from input or input_qtensor."""
  # TODO(lew): moving this if out of DotGeneralRaw into v2.flax.DotGeneral
  # would induce huge code simplification here.
  # E.g. this file would not have to know about DotGeneralQuantizer.
  if input_qtensor is not None:
    output_qtensor = input_qtensor
  else:
    output_qtensor = calculated_qtensor

  return output_qtensor


def make_conv_general_dilated(cfg: aqt_dot_general.DotGeneralRaw):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific cfg.
  assert cfg is not None, "Missing config for make_conv_general_dilated"

  def my_conv_general_dilated(
      lhs,
      rhs,
      lhs_qt: aqt_tensor.QTensor | None,
      rhs_qt: aqt_tensor.QTensor | None,
      window_strides,
      padding,
      lhs_dilation=None,
      rhs_dilation=None,
      dimension_numbers=None,
      feature_group_count=1,
      batch_group_count=1,
      precision=None,
      preferred_element_type=None,
  ) -> Tuple[jax.Array, tuple[aqt_tensor.QTensor, aqt_tensor.QTensor]]:
    msg1 = """
To simplify the code, we currently assume a Flax-particular layout of the data.
This makes sense, because this is the main use-case of this function.
However if there is any other use, we will drop that assumption."""
    rank = len(lhs.shape)
    assert len(rhs.shape) == rank
    assert dimension_numbers is not None, msg1
    assert dimension_numbers.lhs_spec[0:2] == (0, rank - 1), msg1
    assert dimension_numbers.rhs_spec[0:2] == (rank - 1, rank - 2), msg1
    assert dimension_numbers.out_spec[0:2] == (0, rank - 1), msg1
    # In Flax, lhs is the inputs, rhs is the kernel.
    # lhs layout is B, spatials..., Ci
    # rhs layout is: spatials..., Ci, Co
    # out layous it: B, spatials..., Co
    #
    # we need to share these axes: lhs[1:] , rhs[:-1]
    # we have a scale/invscale per: lhs[0] / out[0] and rhs[-1] / out[-1]

    # Flax assumptions.
    msg = (
        "Convolution formula does not follow flax assumption. This could be"
        " because spatial dimensions was incorrectly set during DotGeneralRaw"
        " creation. Please double check the parameter in"
        " #conv_general_dilated_make()."
    )
    # TODO(lew): Perhaps we should rely only on passing  passing calib shared
    # axes value instead of setting it in config. (we pass None below)
    cfg.dg_quantizer.assert_calib_shared_axes_value(
        list(range(1, rank)), list(range(0, rank - 1)), msg
    )

    (lhs, lhs_incomplete_qt), (rhs, rhs_incomplete_qt) = (
        cfg.dg_quantizer.calibrate((lhs, None), (rhs, None))
    )
    if lhs_qt is not None and not lhs_qt.is_full():
      # Incomplete QTensor is provided as lhs_qt.
      lhs_incomplete_qt = lhs_qt
      lhs_qt = None

    if rhs_qt is not None and not rhs_qt.is_full():
      # Incomplete QTensor is provided as rhs_qt.
      rhs_incomplete_qt = rhs_qt
      rhs_qt = None

    lhs_quantized, rhs_quantized = cfg.dg_quantizer.calculate_qvalue(
        lhs, lhs_incomplete_qt, rhs, rhs_incomplete_qt
    )
    lhs_qt_calculated, _ = lhs_quantized
    rhs_qt_calculated, _ = rhs_quantized

    lhs_qt = _postprocess_qtensor(lhs_qt, lhs_qt_calculated)
    rhs_qt = _postprocess_qtensor(rhs_qt, rhs_qt_calculated)

    # Therefore, cast qvalue back to its original data dtype.
    # Delete the following two lines when the constraint is lifted.
    lhs_qt = lhs_qt.qvalue_astype(lhs.dtype)
    rhs_qt = rhs_qt.qvalue_astype(rhs.dtype)

    out = lax.conv_general_dilated(
        lhs=lhs_qt.qvalue,
        rhs=rhs_qt.qvalue,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # It seems lucky that original scale has shape suitable for output
    # scaling without any transposition.
    out = aqt_tensor.QTensor(
        qvalue=out,
        scale=[],
        scale_t=None,
        dequant_dtype=jnp.promote_types(lhs, rhs),
    )
    assert out.scale is not None  # pytype help
    out.scale.extend(lhs_qt.scale)
    out.scale.extend(rhs_qt.scale)
    out = out.dequant()

    # # Future scale granularity optimization.
    # In 1x1 conv, each pixel (spatial location) can have different scales
    # in 1xN (rows x colums) conv each row can have different scale, but
    # columns need to share the scales ,  because we are adding pixels across.
    #
    # For patch convs we could have separate scales per patch.
    # We don't do that optimization, because there is a  Flax op: ConvLocal
    # using lax.conv_general_dilated_local which uses lax.dot_general.
    #
    # Dilations: If a dilation of LHS is bigger than the total spatial size of
    # RHS, we could use separe (per LHS pixel) scales.
    # The same applies to dilated RHS.
    # We don't do that optimization yet.
    #
    # We can have different scales across different groups.
    # This applies to both feature and batch.
    return out, (lhs_qt, rhs_qt)

  return my_conv_general_dilated


def conv_general_dilated_make(
    spatial_dimensions: int,
    lhs_bits: int | None = None,
    rhs_bits: int | None = None,
    initialize_calibration: bool = True,
) -> aqt_dot_general.DotGeneralRaw:
  """Create quantization config conv_general_dilated.

  Args:
    spatial_dimensions: The number of dimensions of the base area that the
      convolutional window moves across.
    lhs_bits: The precision for quantization for lhs
    rhs_bits: The precision for quantization for rhs
    initialize_calibration: If set, calibrators are initialized. Set False if
      you want to run static range quantization.

  Returns:
    DotGeneralRaw object to be injected into nn.Conv as conv_general_dilated.
  """
  config = aqt_dot_general.dot_general_raw_make(
      lhs_bits, rhs_bits, initialize_calibration=initialize_calibration)
  # Hardcoding flax assumptions.
  lhs_calib_shared_axes = (
      list(range(1, spatial_dimensions + 2)) if config.lhs else None
  )
  rhs_calib_shared_axes = (
      list(range(0, spatial_dimensions + 2 - 1)) if config.rhs else None
  )

  assert isinstance(
      config.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  config.dg_quantizer.lhs.calib_shared_axes = lhs_calib_shared_axes
  config.dg_quantizer.rhs.calib_shared_axes = rhs_calib_shared_axes

  return config
