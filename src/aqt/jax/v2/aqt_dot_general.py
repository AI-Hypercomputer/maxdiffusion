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
"""Quantized dot_general."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=g-explicit-bool-comparison
# pylint: disable=g-explicit-length-test

import abc
import enum
from typing import Any, Optional, Sequence, Union

from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import transpose
from aqt.jax.v2 import utils
import jax
from jax import lax
from jax._src.numpy import lax_numpy
import jax.numpy as jnp
import numpy as onp
from typing_extensions import Self  # for python version < 3.11


dtypes_allowed_for_int32_accum = [jnp.int4, jnp.int8]


class CalibrationMode(enum.Enum):
  """Calibration axis modes."""

  CONTRACTING_AXIS = 1
  REMAINING_AXIS = 2


class DequantMode(enum.Enum):
  """Dequant modes."""

  # Multiply output of dot_general by the transposed scale
  # Compatible with CONTRACTING_AXIS.
  OUTPUT = 1
  # Multiply QTensor.qvalue by untransposed QTensor.scale before
  # dot_general (a.k.a. FakeQuant )
  # Compatible with both CalibrationMode.
  THIS_INPUT = 2
  # Multiply other argument of dot general by appropriately transposed scale.
  # Compatible with REMAINING_AXIS.
  OTHER_INPUT = 3


@utils.flax_slots_kw_only_dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  # Controls at what value of input tensor should be used.
  # Setting it to True, but not quantizing fwd pass will assert-fail.
  use_fwd_quant: bool = utils.static_field(default=False)
  # Dequantization mode.
  dequant_mode: DequantMode = utils.static_field(default=DequantMode.OUTPUT)
  # Calibration axis mode.
  calibration_mode: CalibrationMode = utils.static_field(
      default=CalibrationMode.CONTRACTING_AXIS
  )


@utils.flax_slots_kw_only_dataclass
class LocalAqt:
  contraction_axis_shard_count: int | None = utils.static_field(default=None)
  contraction_axis_shard_size: int | None = utils.static_field(default=None)
  # tile_largest_shape=True will apply the factor_reshape_largest function,
  # where shard_size is required and shard_count should be None.
  # If it is False, the original factor_reshape function will be applied,
  # where shard_count is required and shard_size should be None.
  tile_largest_shape: bool = utils.static_field(default=False)


def dot_general_raw_make(
    lhs_bits=None,
    rhs_bits=None,
    local_aqt=None,
    jax_scope_name='aqt',
    initialize_calibration=True,
) -> 'DotGeneralRaw':
  """Create quantization configs for input matrices to a matmul."""
  # TODO: b/343490088 - Move all the parameters to dataclass defaults,
  #   provide setters to modify the configuration.
  lhs_cfg = Tensor()
  rhs_cfg = Tensor()

  # TODO(lew): Binary uses 0.5 right now, it should use -1 and 1.
  if (
      lhs_bits is not None
      and rhs_bits is not None
      and 2 <= lhs_bits <= 8
      and 2 <= rhs_bits <= 8
  ):
    dg_accumulator_dtype = jnp.int32
  else:
    dg_accumulator_dtype = None

  # DotGeneralRaw should create a this quantizer on defualt.
  # Then setter can change it.

  # initialize_calibration=False because that has to be delayed to be called
  # *inside* of flax.nn.custom_vjp
  lhs = aqt_quantizer.quantizer_make(
      lhs_bits, initialize_calibration=initialize_calibration
  )
  rhs = aqt_quantizer.quantizer_make(
      rhs_bits, initialize_calibration=initialize_calibration
  )
  dg_quantizer = DefaultDotGeneralQuantizer(lhs=lhs, rhs=rhs)

  return DotGeneralRaw(
      lhs=lhs_cfg,
      rhs=rhs_cfg,
      dg_quantizer=dg_quantizer,
      dg_accumulator_dtype=dg_accumulator_dtype,
      local_aqt=local_aqt,
      jax_scope_name=jax_scope_name,
  )


# TODO: b/343490088 - Move all the parameters to dataclass defaults,
#   provide setters to modify the configuration.
def dot_general_make(
    lhs_bits: Optional[int] = None,
    rhs_bits: Optional[int] = None,
    bwd_bits: Optional[int] = None,
    use_fwd_quant: bool = True,
    dlhs_local_aqt=None,
    drhs_local_aqt=None,
) -> 'DotGeneral':
  """Create quantization configs for input matrices to a matmul."""
  fwd = dot_general_raw_make(
      lhs_bits,
      rhs_bits,
      jax_scope_name='aqt_fwd',
      initialize_calibration=False,
  )
  dlhs = dot_general_raw_make(
      bwd_bits,
      bwd_bits,
      local_aqt=dlhs_local_aqt,
      jax_scope_name='aqt_dlhs',
      initialize_calibration=False,
  )
  drhs = dot_general_raw_make(
      bwd_bits,
      bwd_bits,
      local_aqt=drhs_local_aqt,
      jax_scope_name='aqt_drhs',
      initialize_calibration=False,
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  if lhs_bits is not None:
    # Only rhs is accepting MultiTensor.
    cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
  if rhs_bits is not None:
    cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


# ------------------------------------------------------------------------------


@utils.flax_slots_kw_only_dataclass
class MultiTensor:
  x: jnp.ndarray
  qx: aqt_tensor.QTensor


@utils.flax_slots_kw_only_dataclass
class TensorRes:
  """All the things we pass from the forward pass to the backward pass."""

  mt: MultiTensor
  quant_grad: aqt_tensor.GradientFn


@utils.flax_slots_kw_only_dataclass
class DotGeneralRes:
  lhs: TensorRes
  rhs: TensorRes


def einsum(eqn: str, lhs: jnp.ndarray, rhs: jnp.ndarray, dg=lax.dot_general):
  """A copy of jnp.einsum but without the default jit so as to be injectable."""
  operands, contractions = lax_numpy._default_poly_einsum_handler(  # pylint: disable=protected-access
      eqn, lhs, rhs, einsum_call=True, use_blas=True, optimize='optimal'
  )
  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
  return jax.named_call(lax_numpy._einsum, name=eqn)(  # pylint: disable=protected-access
      operands,
      contractions,
      precision=None,
      preferred_element_type=None,
      _dot_general=dg,
  )


# TODO(lew): Inline and simplify, perhaps using list comprehension.
def _get_scale_t(
    qt: aqt_tensor.QTensor,
    transpose_fn: Any,
    dimension_numbers: lax.DotDimensionNumbers,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> Sequence[jnp.ndarray]:
  list_scale_t = []
  for scale in qt.scale:
    scale_t = transpose_fn(scale, dimension_numbers, lhs_shape, rhs_shape)
    list_scale_t.append(scale_t)
  return list_scale_t


def _apply_local_aqt(
    local_aqt: LocalAqt,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.lax.DotDimensionNumbers]:
  """Applies local AQT if the configuration is given.

  Args:
    local_aqt: Local AQT configuration.
    lhs: Left hand side of the dot_general.
    rhs: Right hand side of the dot_general.
    dimension_numbers: Dot_general dimension numbers.

  Returns:
    A tuple of (lhs, rhs, dimension_numbers) with local AQT applied.
  """

  def factor_reshape_largest(x, ca, ba):
    # Tile the contraction axis with largest shape using tile_size.
    # This function can be a replacement for the original factor_reshape.
    # In the spirit of backward compatibility, we keep the original
    # factor_reshape as well.
    msg = 'When tile_largest_shape is True, must set shard_size.'
    assert local_aqt.contraction_axis_shard_size is not None, msg
    assert local_aqt.contraction_axis_shard_count is None, msg
    shard_size = local_aqt.contraction_axis_shard_size
    if not ca:
      return x, ca, ba
    shape = list(x.shape)
    max_ca_shape = max(shape[axis] for axis in ca)
    ax = shape.index(max_ca_shape)
    ax_ca_idx = ca.index(ax)
    assert max_ca_shape % shard_size == 0
    shape[ax] = max_ca_shape // shard_size
    shape.insert(ax + 1, shard_size)
    new_ca = [(b + int(b >= ax)) for b in ca]
    assert new_ca[ax_ca_idx] == ax + 1
    new_ba = [ax] + [(b + int(b > ax)) for b in ba]
    return x.reshape(shape), new_ca, new_ba

  def factor_reshape(x, ca, ba):
    # Original tiling function. Only support configuring the tile count.
    msg = 'You are using the original local aqt. Please only set shard_count.'
    assert local_aqt.contraction_axis_shard_count is not None, msg
    assert local_aqt.contraction_axis_shard_size is None, msg
    factor = local_aqt.contraction_axis_shard_count
    if not ca:
      return x, ca, ba
    shape = list(x.shape)
    ax = ca[0]
    orig_size = shape[ax]
    assert orig_size % factor == 0
    shape[ax] = factor
    shape.insert(ax + 1, orig_size // factor)
    new_ca = [(b + int(b >= ax)) for b in ca]
    assert new_ca[0] == ax + 1
    new_ba = [ax] + [(b + int(b > ax)) for b in ba]
    return x.reshape(shape), new_ca, new_ba

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if local_aqt.tile_largest_shape:
    factor_fn = factor_reshape_largest
  else:
    factor_fn = factor_reshape
  lhs, lhs_ca, lhs_ba = factor_fn(lhs, lhs_ca, lhs_ba)
  rhs, rhs_ca, rhs_ba = factor_fn(rhs, rhs_ca, rhs_ba)

  dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)
  return lhs, rhs, dimension_numbers


@utils.flax_slots_kw_only_dataclass
class DotGeneralQuantizer(abc.ABC):
  """Abstract class for dot_general quantizer."""

  def __call__(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[utils.AxisIdx]],
      rhs_quantization_info: tuple[jax.Array, Sequence[utils.AxisIdx]],
  ) -> tuple[
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
  ]:
    (lhs, lhs_qt), (rhs, rhs_qt) = self.calibrate(
        lhs_quantization_info, rhs_quantization_info
    )
    return self.calculate_qvalue(lhs, lhs_qt, rhs, rhs_qt)

  @abc.abstractmethod
  def init_calibration(self):
    pass

  @abc.abstractmethod
  def calibrate(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[int]],
      rhs_quantization_info: tuple[jax.Array, Sequence[int]],
  ) -> tuple[
      tuple[jax.Array, aqt_tensor.QTensor],
      tuple[jax.Array, aqt_tensor.QTensor]
  ]:
    """Calculates incomplete QTensor from the given inputs.

    The function calculates incomplete QTensor from the given inputs. It also
    returns the updated lhs and rhs during calibration.

    Args:
      lhs_quantization_info: left argument and its contracting axis.
      rhs_quantization_info: right argument and its contracting axis:
    Returns:
      A tuple of (lhs, lhs_qt), (rhs, rhs_qt) where lhs and rhs are updated
      arguments and lhs_qt and rhs_qt are incomplete QTensor.
    """
    pass

  # TODO(lew): There is only one meaningful implementation of
  # calculate_qvalue. Does not have to be an overridable method.
  @abc.abstractmethod
  def calculate_qvalue(
      self,
      lhs: jax.Array,
      lhs_qt: aqt_tensor.QTensor,
      rhs: jax.Array,
      rhs_qt: aqt_tensor.QTensor,
  ) -> tuple[
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
  ]:
    """Calculates qvalues from the given inputs."""
    pass

  @abc.abstractmethod
  def swap_lhs_and_rhs(self) -> None:
    """Swaps lhs and rhs configuration."""
    pass

  @abc.abstractmethod
  def assert_calib_shared_axes_value(
      self,
      lhs_val: Sequence[utils.AxisIdx] | None,
      rhs_val: Sequence[utils.AxisIdx] | None,
      msg: str,
  ) -> None:
    """Asserts if calib_shared_axes have certain values."""
    pass

  @abc.abstractmethod
  def set_context(
      self,
      lhs_context: utils.Context,
      rhs_context: utils.Context,
  ) -> None:
    """Sets context for lhs and rhs."""
    pass


# TODO(lew): Find a better name instead of Default* (Factored?, Separated?)
@utils.flax_slots_kw_only_dataclass
class DefaultDotGeneralQuantizer(DotGeneralQuantizer):
  """Default dot_general quantizer."""

  lhs: aqt_quantizer.Quantizer
  rhs: aqt_quantizer.Quantizer

  def init_calibration(self):
    self.lhs.init_calibration()
    self.rhs.init_calibration()

  def calibrate(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[int]],
      rhs_quantization_info: tuple[jax.Array, Sequence[int]],
  ) -> tuple[
      tuple[jax.Array, aqt_tensor.QTensor],
      tuple[jax.Array, aqt_tensor.QTensor]
  ]:
    lhs_input, lhs_ca = lhs_quantization_info
    rhs_input, rhs_ca = rhs_quantization_info
    lhs_qt = self.lhs.calibrate(lhs_input, calibration_axes=lhs_ca)
    rhs_qt = self.rhs.calibrate(rhs_input, calibration_axes=rhs_ca)
    return ((lhs_input, lhs_qt), (rhs_input, rhs_qt))

  def calculate_qvalue(
      self,
      lhs: jax.Array,
      lhs_qt: aqt_tensor.QTensor,
      rhs: jax.Array,
      rhs_qt: aqt_tensor.QTensor,
  ) -> tuple[
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
  ]:
    """Calculates qvalues from the given inputs."""
    lhs_qt, lhs_grad = self.lhs.calculate_qvalue(lhs, lhs_qt)
    rhs_qt, rhs_grad = self.rhs.calculate_qvalue(rhs, rhs_qt)

    return (lhs_qt, lhs_grad), (rhs_qt, rhs_grad)

  def swap_lhs_and_rhs(self) -> None:
    self.lhs, self.rhs = self.rhs, self.lhs

  def assert_calib_shared_axes_value(
      self,
      lhs_val: Sequence[utils.AxisIdx] | None,
      rhs_val: Sequence[utils.AxisIdx] | None,
      msg: str,
  ) -> None:
    assert self.lhs.calib_shared_axes == lhs_val, msg
    assert self.rhs.calib_shared_axes == rhs_val, msg

  def set_context(
      self,
      lhs_context: utils.Context,
      rhs_context: utils.Context,
  ) -> None:
    self.lhs.context = lhs_context
    self.rhs.context = rhs_context


def quant(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lhs_qt: aqt_tensor.QTensor | None,
    rhs_qt: aqt_tensor.QTensor | None,
    dg_quantizer: DotGeneralQuantizer,
    lhs_cfg: Tensor,
    rhs_cfg: Tensor,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    allow_dummy_gradient_into_qtensor: bool
) -> tuple[
    tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
    tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
]:
  """Quantizes the given lhs and rhs using dg_quantizer."""
  def _get_calibration_axes(
      tensor_cfg: Tensor,
      ndim: int,
      ca: Sequence[utils.AxisIdx],
      ba: Sequence[utils.AxisIdx],
  ) -> Sequence[utils.AxisIdx]:
    """Computes calibration axes for the given Tensor."""
    match tensor_cfg.calibration_mode:
      case CalibrationMode.REMAINING_AXIS:
        calibration_axes = utils.get_remaining_axes(ndim, ca, ba)
      case CalibrationMode.CONTRACTING_AXIS:
        calibration_axes = ca
      case _:
        raise ValueError(
            f'Unknown calibration mode: {tensor_cfg.calibration_mode}'
        )
    return calibration_axes

  def _postprocess_qtensor(
      input_qtensor: Optional[aqt_tensor.QTensor],
      calculated_qtensor: aqt_tensor.QTensor,
      quant_grad: aqt_tensor.GradientFn,
  ) -> tuple[aqt_tensor.QTensor, str | aqt_tensor.GradientFn]:
    """Compute qtensor from input or input_qtensor."""
    # TODO(lew): moving this if out of DotGeneralRaw into v2.flax.DotGeneral
    # would induce huge code simplification here.
    # E.g. this file would not have to know about DotGeneralQuantizer.
    if input_qtensor is not None:
      if not allow_dummy_gradient_into_qtensor:
        quant_grad = (
            'Poison. '
            + 'Gradients are not generally expected in serving. '
            + 'Please set allow_dummy_gradient_into_qtensor to True '
            + 'if this is the intended behavior.'
        )
      output_qtensor = input_qtensor
    else:
      output_qtensor = calculated_qtensor

    return output_qtensor, quant_grad

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  lhs_calib_axes = _get_calibration_axes(lhs_cfg, lhs.ndim, lhs_ca, lhs_ba)
  rhs_calib_axes = _get_calibration_axes(rhs_cfg, rhs.ndim, rhs_ca, rhs_ba)

  (lhs, lhs_incomplete_qt), (rhs, rhs_incomplete_qt) = dg_quantizer.calibrate(
      (lhs, lhs_calib_axes), (rhs, rhs_calib_axes)
  )
  if lhs_qt is not None and not lhs_qt.is_full():
    # Incomplete QTensor is provided as lhs_qt.
    lhs_incomplete_qt = lhs_qt
    lhs_qt = None

  if rhs_qt is not None and not rhs_qt.is_full():
    # Incomplete QTensor is provided as rhs_qt.
    rhs_incomplete_qt = rhs_qt
    rhs_qt = None

  lhs_quantized, rhs_quantized = dg_quantizer.calculate_qvalue(
      lhs, lhs_incomplete_qt, rhs, rhs_incomplete_qt
  )
  lhs_qt_calculated, lhs_quant_grad = lhs_quantized
  rhs_qt_calculated, rhs_quant_grad = rhs_quantized

  lhs_qt, lhs_quant_grad = _postprocess_qtensor(
      lhs_qt,
      lhs_qt_calculated,
      lhs_quant_grad,
  )

  rhs_qt, rhs_quant_grad = _postprocess_qtensor(
      rhs_qt,
      rhs_qt_calculated,
      rhs_quant_grad,
  )
  return (lhs_qt, lhs_quant_grad), (rhs_qt, rhs_quant_grad)


def _maybe_use_fwd_quant(
    lhs: jnp.ndarray,
    rhs: MultiTensor,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    use_fwd_quant: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Applies already quantized value for backpropagation, if the flag is set.

  Note that we cannot directly use the quantized rhs for gradient calculation,
  since the dimension numbers are changed.

  Args:
    lhs: Left hand size of gradient dot_general.
    rhs: Right hand size of gradient dot_general.
    dimension_numbers: Dot_general dimension numbers.
    use_fwd_quant: Flag to use forward quantization.
  Returns:
    A tuple of updated (lhs, rhs). If use_fwd_quant is True, lhs is multiplied
    with rhs scale, while rhs is set to the original rhs's qvalue.
  """
  fwd_quantized = rhs.qx.scale is not None and len(rhs.qx.scale) == 1
  msg = (
      f'Found use_fwd_quant is {use_fwd_quant} in bwd, but fwd is not'
      ' quantized.'
  )
  if use_fwd_quant:
    assert fwd_quantized, msg
    scale_t = transpose.rhs_scale_transpose_for_lhs_input(
        rhs.qx.scale[0], dimension_numbers, lhs.shape
    )

    # Cast rhs scales to lhs dtype when multiplying with lhs. This is to
    # avoid an unexpected upcast when rhs is float32 but lhs is float16.
    lhs = lhs * scale_t.astype(lhs.dtype)

    # rhs qvalue may be integer. It will be quantized again later, so cast
    # its dtype back to dequant dtype.
    # TODO(yichizh): avoid double quantization and evaluate model quality.
    rhs = rhs.qx.qvalue.astype(rhs.qx.dequant_dtype)
  else:
    rhs = rhs.x

  return lhs, rhs


@utils.flax_slots_kw_only_dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  dg_quantizer: DotGeneralQuantizer
  dg_accumulator_dtype: Optional[jnp.dtype] = utils.static_field()
  local_aqt: Optional[LocalAqt] = utils.static_field()
  jax_scope_name: str = utils.static_field()

  # Set it to true in order to train with non-None lhs_qt or rhs_qt.
  # Gradient will still flow into lhs_qt and/or rhs_qt, but it may be incorrect.
  # It is a caller responsibility to NOT update these QTensors
  allow_dummy_gradient_into_qtensor: bool = utils.static_field(default=False)
  dot_general: utils.DotGeneralT = utils.static_field(
      default=jax.lax.dot_general
  )

  # TODO(lew): Remove this function.
  @classmethod
  def make(cls, *args, **kwargs) -> Self:
    return dot_general_raw_make(*args, **kwargs)

  # TODO(lew): Can we remove MutliTensor and pass rhs_qt instead?
  def __call__(
      self,
      lhs: jnp.ndarray,
      rhs: Union[jnp.ndarray, MultiTensor],
      # xhs_qt are used in serving.
      lhs_qt: Optional[aqt_tensor.QTensor],
      rhs_qt: Optional[aqt_tensor.QTensor],
      dimension_numbers: jax.lax.DotDimensionNumbers,
  ):
    """A quantized dot_general function without custom gradient."""
    with jax.named_scope(self.jax_scope_name):
      # TODO(lew):
      #  - Use qx.value with the int type.
      #  - Handle qx.value with the int type in an optimized way.
      #  - Add a "FQ" case we multiply qx.value*qx.value_scale (not transposed).
      #  - Can we carry untransposed scale and transpose here?
      if isinstance(rhs, MultiTensor):
        lhs, rhs = _maybe_use_fwd_quant(
            lhs, rhs, dimension_numbers, self.rhs.use_fwd_quant
        )
      assert isinstance(rhs, jnp.ndarray)

      # TODO(lew): Define cutsom_vjp on tiled_dot_general and replace local_aqt.
      if self.local_aqt is not None:
        msg = 'Custom calib_shared_axes not implemented for local AQT.'
        if isinstance(self.dg_quantizer, DefaultDotGeneralQuantizer):
          self.dg_quantizer.assert_calib_shared_axes_value(None, None, msg)

        lhs, rhs, dimension_numbers = _apply_local_aqt(
            self.local_aqt,  # pytype: disable=attribute-error
            lhs,
            rhs,
            dimension_numbers,
        )

      (lhs_qt, lhs_quant_grad), (rhs_qt, rhs_quant_grad) = quant(
          lhs,
          rhs,
          lhs_qt,
          rhs_qt,
          self.dg_quantizer,
          self.lhs,
          self.rhs,
          dimension_numbers,
          self.allow_dummy_gradient_into_qtensor
      )

      lhs_mt = MultiTensor(x=lhs, qx=lhs_qt)
      lhs_res = TensorRes(mt=lhs_mt, quant_grad=lhs_quant_grad)

      rhs_mt = MultiTensor(x=rhs, qx=rhs_qt)
      rhs_res = TensorRes(mt=rhs_mt, quant_grad=rhs_quant_grad)

      # TODO(lew): mt.x above should be clipped for clipping calibrations
      out = _qtensor_dot_general(
          lhs_qt, rhs_qt, dimension_numbers, self, jnp.promote_types(lhs, rhs)
      )

      out = out.dequant()

      res = DotGeneralRes(lhs=lhs_res, rhs=rhs_res)
      if self.local_aqt is not None:
        (lhs_ca, rhs_ca), _ = dimension_numbers
        assert len(lhs_ca) == len(rhs_ca)
        if len(lhs_ca) > 0:
          out = jnp.sum(out, axis=0)
        # We are not supporting local AQT in fwd pass, so no res needed.
        res = None
      return out, res


def _qtensor_dot_general(
    lhs_qt: aqt_tensor.QTensor,
    rhs_qt: aqt_tensor.QTensor,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    cfg: ...,  # DotGeneralRaw,
    # dequant_dtype: DType,
    dequant_dtype: jnp.dtype,
) -> aqt_tensor.QTensor:
  """QTensor lax.dot_general replacement."""

  def _maybe_dequant(
      input_qtensor: aqt_tensor.QTensor, tensor_cfg: Tensor
  ) -> jnp.ndarray:
    if tensor_cfg.dequant_mode == DequantMode.THIS_INPUT:
      output = input_qtensor.dequant()
    else:
      output = input_qtensor.qvalue
    return output

  # Dequantize before the lax dg call if in fake quant mode
  lhs_qin = _maybe_dequant(lhs_qt, cfg.lhs)
  rhs_qin = _maybe_dequant(rhs_qt, cfg.rhs)

  dtype_ms = (
      f'Found {cfg.dg_accumulator_dtype=}, {lhs_qin.dtype=} and'
      f' {rhs_qin.dtype=}. Dot general accumulator dtype can only be'
      ' jnp.int32 when both inputs are int8. Otherwise it is recommended to'
      ' be None to let lax.dot_general automatically decide it.'
  )
  if cfg.dg_accumulator_dtype == jnp.int32:
    assert (
        lhs_qin.dtype in dtypes_allowed_for_int32_accum
        and rhs_qin.dtype in dtypes_allowed_for_int32_accum
    ), dtype_ms

  dtypes_can_be_scaled = [jnp.bfloat16, jnp.float32, jnp.float64]

  if cfg.lhs.dequant_mode == DequantMode.OTHER_INPUT:
    assert rhs_qin.dtype in dtypes_can_be_scaled
    for scale in lhs_qt.scale:
      transposed_scale = transpose.lhs_scale_transpose_for_rhs_input(
          scale, dimension_numbers, rhs_qt.shape
      )
      assert isinstance(transposed_scale, jnp.ndarray)  # make pytype quiet
      rhs_qin = rhs_qin * transposed_scale.astype(rhs_qin.dtype)

  if cfg.rhs.dequant_mode == DequantMode.OTHER_INPUT:
    assert lhs_qin.dtype in dtypes_can_be_scaled
    for scale in rhs_qt.scale:
      transposed_scale = transpose.rhs_scale_transpose_for_lhs_input(
          scale, dimension_numbers, lhs_qt.shape
      )
      assert isinstance(transposed_scale, jnp.ndarray)  # make pytype quiet
      lhs_qin = lhs_qin * transposed_scale.astype(lhs_qin.dtype)

  if jax.local_devices()[0].platform == 'cpu':
    # needed bet lax.dot_general(int4, int4) is illegal on cpu.
    # TODO(aqt): Remove this platform check once
    # https://github.com/google/jax/issues/19682 is fixed.
    # TODO(yichizh): It's better to assert False here with the following msg
    # msg = (
    #     'lax.dot_general(int4, int4) is illegal on cpu:'
    #     ' https://github.com/google/jax/issues/19682. The simple workaround'
    #     ' is to upcast to int8, but in that case please directly set the'
    #     ' numerics bits to int8. Please contact the AQT team if you believe'
    #     ' the workaround is needed.'
    # )
    if lhs_qin.dtype == jnp.int4 and rhs_qin.dtype == jnp.int4:
      lhs_qin = lhs_qin.astype(jnp.int8)
      rhs_qin = rhs_qin.astype(jnp.int8)

  out = cfg.dot_general(
      lhs_qin,
      rhs_qin,
      dimension_numbers=dimension_numbers,
      preferred_element_type=cfg.dg_accumulator_dtype,
      precision=lax.Precision.DEFAULT,
  )
  # TODO(lew): Do we have a correct precision above?
  #   Relevant: https://github.com/google/jax/issues/14022
  out = aqt_tensor.QTensor(
      qvalue=out,
      scale=[],
      scale_t=None,
      dequant_dtype=dequant_dtype,
  )
  assert out.scale is not None  # pytype help

  if cfg.lhs.dequant_mode == DequantMode.OUTPUT:
    extend_scale = _get_scale_t(
        qt=lhs_qt,
        transpose_fn=transpose.lhs_scale_transpose_to_output,
        dimension_numbers=dimension_numbers,
        lhs_shape=lhs_qin.shape,
        rhs_shape=rhs_qin.shape,
    )

    out.scale.extend(extend_scale)
  if cfg.rhs.dequant_mode == DequantMode.OUTPUT:
    extend_scale = _get_scale_t(
        qt=rhs_qt,
        transpose_fn=transpose.rhs_scale_transpose_to_output,
        dimension_numbers=dimension_numbers,
        lhs_shape=lhs_qin.shape,
        rhs_shape=rhs_qin.shape,
    )
    out.scale.extend(extend_scale)
  return out


@utils.flax_slots_kw_only_dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""

  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw

  apply_custom_vjp_on_jax: bool = utils.static_field(default=True)

  @classmethod
  def make(cls, *args, **kwargs) -> Self:
    return dot_general_make(*args, **kwargs)

  def dg_core(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      lhs_qt: Optional[aqt_tensor.QTensor],
      rhs_qt: Optional[aqt_tensor.QTensor],
      dimension_numbers: lax.DotDimensionNumbers,
  ):
    """dot_general function with expanded API."""
    msg = 'AQT is not yet optimized to accept quantized types directly. '
    msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
    assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
    assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

    self.assert_config_validity()
    # TODO(dhchoi): Refactor this to make both branches to have the similar
    # functionality of applying custom_vjp.
    if self.apply_custom_vjp_on_jax:
      dg_core_with_custom_grad = jax.custom_vjp(_dg_core, nondiff_argnums=(4,))
      dg_core_with_custom_grad.defvjp(dg_core_vjp_fwd, dg_core_vjp_bwd)
      out, res = dg_core_with_custom_grad(
          lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, self
      )
    else:
      out, res = _dg_core(lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, self)  # pytype: disable=wrong-arg-types
    return out, res

  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
  ):
    del preferred_element_type
    assert (
        precision is None
    ), f'Precision {precision} requested together with quantization.'

    out, _ = self.dg_core(
        lhs=lhs,
        rhs=rhs,
        lhs_qt=None,
        rhs_qt=None,
        dimension_numbers=dimension_numbers,
    )
    return out

  def assert_config_validity(self: Self):
    """Asserts if configuration is valid."""

    # use_fwd_quant is not enabled when calibration_axis = remaining_axis.
    # TODO: b/336198483 - Enable use_fwd_quant flag in the case.
    expected_fwd_quant = False
    msg_fwd_quant = (
        f'use_fwd_quant should be set to {expected_fwd_quant} when remaining'
        ' axis are used for calibration axis.'
    )

    if self.fwd.rhs.calibration_mode == CalibrationMode.REMAINING_AXIS:
      msg_fwd_quant += (
          f'rhs.calibration_mode: {self.fwd.rhs.calibration_mode},'
          f' dlhs use_fwd_quant: {self.dlhs.rhs.use_fwd_quant}'
      )
      assert self.dlhs.rhs.use_fwd_quant is expected_fwd_quant, msg_fwd_quant

    if self.fwd.lhs.calibration_mode == CalibrationMode.REMAINING_AXIS:
      msg_fwd_quant += (
          f'lhs.calibration_mode: {self.fwd.lhs.calibration_mode},'
          f' drhs use_fwd_quant: {self.drhs.rhs.use_fwd_quant}'
      )
      assert self.drhs.rhs.use_fwd_quant is expected_fwd_quant, msg_fwd_quant

    # Check valid combination between calibration_mode and dequant_mode
    unsupported_calibration_dequant_pairs = [
        (DequantMode.OUTPUT, CalibrationMode.REMAINING_AXIS),
        (DequantMode.OTHER_INPUT, CalibrationMode.CONTRACTING_AXIS),
    ]
    msg_mode_mismatch = (
        'Unsupported calibration mode - dequant mode combination '
    )
    for (
        dequant_mode,
        calibration_mode,
    ) in unsupported_calibration_dequant_pairs:
      assert not (
          self.fwd.lhs.calibration_mode == calibration_mode
          and self.fwd.lhs.dequant_mode == dequant_mode
      ), (
          msg_mode_mismatch
          + ' for lhs. calibration_mode:'
          f' {self.fwd.lhs.calibration_mode}, dequant_mode:'
          f' {self.fwd.lhs.dequant_mode}'
      )
      assert not (
          self.fwd.rhs.calibration_mode == calibration_mode
          and self.fwd.rhs.dequant_mode == dequant_mode
      ), (
          msg_mode_mismatch
          + ' for rhs. calibration_mode:'
          f' {self.fwd.rhs.calibration_mode}, dequant_mode:'
          f' {self.fwd.rhs.dequant_mode}'
      )


def _dg_core(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lhs_qt: Optional[aqt_tensor.QTensor],
    rhs_qt: Optional[aqt_tensor.QTensor],
    dimension_numbers: lax.DotDimensionNumbers,
    cfg: DotGeneral,
):
  out, _ = dg_core_vjp_fwd(lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, cfg)
  return out


# When defining a vjp, all traceable variables must be input arguments of
# both the fwd and bwd function.
# The cfg (DotGeneral) contains the key used for stochastic rounding,
# which are traceable dynamic variables. It needs to be an input argument
# to prevent the jax side effect.
def dg_core_vjp_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lhs_qt: Optional[aqt_tensor.QTensor],
    rhs_qt: Optional[aqt_tensor.QTensor],
    dimension_numbers: lax.DotDimensionNumbers,
    cfg: DotGeneral,
):
  """custom_vjp fwd pass."""
  assert (
      lhs.dtype == rhs.dtype
  ), f'Unmatched lhs and rhs dtype: {lhs.dtype} vs {rhs.dtype}'
  cfg.fwd.dg_quantizer.init_calibration()
  cfg.dlhs.dg_quantizer.init_calibration()
  cfg.drhs.dg_quantizer.init_calibration()
  ret, res = cfg.fwd(
      lhs,
      rhs,
      lhs_qt,
      rhs_qt,
      dimension_numbers,
  )
  ret = ret.astype(lhs.dtype)
  # We return these values to allow for materialization.
  assert res is not None, 'res cannot be None in fwd pass.'
  qret = (res.lhs.mt.qx, res.rhs.mt.qx)
  return ((ret, qret), (res, cfg))


def _update_dimension_numbers_for_backward(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    y_is_lhs: bool,
    gradient_rank: int,
    y_rank: int,
) -> tuple[jax.lax.DotDimensionNumbers, tuple[int, ...]]:
  """Generates a new dimension number for backward pass.

  For dot_general(lhs, rhs), the gradients for lhs and rhs are calculated by
  dot_general(g, rhs) and dot_general(g, lhs). This function generates proper
  dimension numbers for the dot_generals used for gradient calculation.

  Args:
    fwd_dimension_numbers: Dimension number used during forward pass
    y_is_lhs: If set, the function calculates dimension numbers for dlhs.
    gradient_rank: Rank of the gradient.
    y_rank: Rank of the other side input.
  Returns:
    A tuple of (dimension numbers for gradient dot_general, transpose axes to be
    applied on the gradient dot_generals output to match with the original
    argument dimension).
  """
  def ranges_like(*xs):
    start = 0
    for x in xs:
      yield tuple(range(start, start + len(x)))
      start += len(x)

  (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
  if y_is_lhs:
    (y_ca, x_ca) = (x_ca, y_ca)
    (y_ba, x_ba) = (x_ba, y_ba)

  gradient_rank = gradient_rank - y_rank + len(x_ba) + 2 * len(x_ca)
  x_ra = tuple(utils.get_remaining_axes(gradient_rank, x_ca, x_ba))
  y_ra = tuple(utils.get_remaining_axes(y_rank, y_ca, y_ba))
  if y_is_lhs:
    g_ba, g_ca, _ = ranges_like(x_ba, y_ra, x_ra)
  else:
    g_ba, _, g_ca = ranges_like(x_ba, x_ra, y_ra)
  dims = ((g_ca, y_ra), (g_ba, y_ba))

  x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
  out_transpose_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))

  return dims, out_transpose_axes


def dg_core_vjp_bwd(
    fwd_dimension_numbers: lax.DotDimensionNumbers,
    res: tuple[Optional[DotGeneralRes], DotGeneral],
    g,
):
  """custom_vjp bwd pass."""
  dg_res, cfg = res
  msg = 'dg_res can only be None in 2nd derivative. It is not yet supported.'
  assert dg_res is not None, msg
  g = g[0]  # g[1] is gradient with respect to qret which we are ignoring.

  def grad_dot_general(
      y_res: TensorRes,
      quant_grad: aqt_tensor.GradientFn,
      dg_raw: DotGeneralRaw,
      y_is_lhs,
  ):
    dims, out_transpose_axes = _update_dimension_numbers_for_backward(
        fwd_dimension_numbers, y_is_lhs, g.ndim, y_res.mt.x.ndim
    )

    out, _ = dg_raw(g, y_res.mt, None, None, dims)

    transposed_out = jax.lax.transpose(out, out_transpose_axes)
    if quant_grad is not None:
      transposed_out = quant_grad(transposed_out)[0]
    return transposed_out

  dlhs = grad_dot_general(
      dg_res.rhs,
      dg_res.lhs.quant_grad,
      cfg.dlhs,
      False,
  )
  drhs = grad_dot_general(
      dg_res.lhs,
      dg_res.rhs.quant_grad,
      cfg.drhs,
      True,
  )
  # fwd_dimension_numbers are marked as nondiff_argnums instead of returning
  # None as grad to it. This is because it is a tuple of Python integers
  # that cannot be traced by Jax.
  return (dlhs, drhs, None, None, None)
