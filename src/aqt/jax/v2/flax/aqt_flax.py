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
"""Flax layer for AQT injection."""

# pylint: disable=unnecessary-lambda
# pylint: disable=g-importing-member
import copy
import enum
import functools
from typing import Callable, Iterable, Optional, Sequence, Union
from aqt.jax.v2 import aqt_conv_general
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import transpose
from aqt.jax.v2 import utils
from aqt.jax.v2.flax import aqt_flax_dg_core
from aqt.jax.v2.flax import freezer as general_freezer
from aqt.jax.v2.utils import QuantMode
import flax.core.meta as nn_meta
import flax.linen as nn
import jax
import jax.numpy as jnp


NoShardingAxes = Sequence[utils.AxisIdx]
AxisMetadataWrapper = Callable[
    [jnp.ndarray, tiled_dot_general.AqtTileMap | None, NoShardingAxes],
    nn_meta.AxisMetadata,
]
DotGeneralTilingFn = Callable[
    [jnp.ndarray, jnp.ndarray, jax.lax.DotDimensionNumbers],
    tiled_dot_general.Cfg
]
EinsumTilingFn = Callable[
    [tiled_dot_general.EinsumEqnLetter, jnp.ndarray, jnp.ndarray],
    tiled_dot_general.Cfg
]


_QUANT_TO_FREEZER_MODE = {
    QuantMode.TRAIN: general_freezer.FreezerMode.NONE,
    QuantMode.CALIBRATE: general_freezer.FreezerMode.NONE,
    QuantMode.CONVERT: general_freezer.FreezerMode.WRITE,
    QuantMode.SERVE: general_freezer.FreezerMode.READ,
}


def _freezer_qtensor_init_wrapper(
    qt: aqt_tensor.QTensor,
    contracting_axis: Sequence[utils.AxisIdx],
    axis_metadata_wrapper: Optional[AxisMetadataWrapper],
    tile_map: tiled_dot_general.AqtTileMap,
):
  """QTensor initialization wrapper function for the new freezer."""
  if axis_metadata_wrapper is None:
    return qt

  scale_non_shard_axis_all = list(range(qt.ndim))
  scale_non_shard_axis_contracting = list(contracting_axis)

  qt = qt.replace(
      qvalue=axis_metadata_wrapper(qt.qvalue, tile_map, []),
      scale=jax.tree.map(
          lambda x: axis_metadata_wrapper(
              x, tile_map, scale_non_shard_axis_contracting
          ),
          qt.scale,
      ),
      # Passing scale_non_shard_axis_contracting would be incorrect due to
      # scale transposition. scale_t is being removed from QTensor anyway
      # so we just pass scale_non_shard_axis_all.
      scale_t=jax.tree.map(
          lambda x: axis_metadata_wrapper(
              x, tile_map, scale_non_shard_axis_all
          ),
          qt.scale_t,
      ),
  )
  return qt


class FreezerMode(enum.Enum):
  NONE = 1
  CALIBRATION = 2
  CALIBRATION_AND_VALUE = 3


class Freezer(nn.Module):
  """Identity function that can freeze its input.

  On default it is an identity function that saves the input in a variable.
  In 'quant_mode=QuantMode.Serve' mode, ignores the input and returns the frozen
  value. It is usefult to implement 'constant folding' and put quantized weights
  and scales in the checkpoint for serving. Specifically:

  self.get() returns None when quant_mode=TRAIN or CONVERT, returns variable
  when quant_mode=SERVE.
  self.set() does nothing when quant_mode=TRAIN or SERVE, creates and stores
  quantized tensor when quant_mode=CONVERT.
  """

  quant_collection: str
  quant_mode: QuantMode
  q_shape: Iterable[int]
  q_dtype: jnp.dtype
  q_init: nn.initializers.Initializer
  s_shape: Iterable[int]
  s_init: nn.initializers.Initializer

  def setup(self):
    mode = self.quant_mode
    if mode == QuantMode.SERVE or mode == QuantMode.CONVERT:
      collection = self.quant_collection
      q_init = self.q_init
      q_shape = self.q_shape
      q_dtype = self.q_dtype
      s_init = self.s_init
      s_shape = self.s_shape
      # TODO(lew): Store whole QTensor?
      # We could have created one self.variable whose value is a QTensor,
      # but we are unsure how this would complicate the init function,
      # which could potentially be used by adding metadata such as
      # sharding axises, etc.
      self.qvalue = self.variable(collection, 'value', q_init, q_shape, q_dtype)
      self.scale_t = self.variable(collection, 'scale', s_init, s_shape)

  def get(self) -> Optional[aqt_tensor.QTensor]:
    if self.quant_mode == QuantMode.TRAIN:
      return None
    elif self.quant_mode == QuantMode.CALIBRATE:
      return None
    elif self.quant_mode == QuantMode.CONVERT:
      return None
    elif self.quant_mode == QuantMode.SERVE:
      return aqt_tensor.QTensor(
          qvalue=self.qvalue.value,
          scale=None,
          scale_t=[self.scale_t.value],
          # TODO(lew): Ideal solution: To find out this dequant_dtype one should
          # use the dtype of inputs of the quant function. We should store it as
          # a dtype of small-sized scale tensor.
          dequant_dtype=self.scale_t.value.dtype,
      )
    else:
      assert False, 'Unknown quant mode.'

  def set(self, inputs: aqt_tensor.QTensor) -> None:
    # TODO(b/325626080): Uncommenting the assert will fail.
    # assert inputs.qvalue.dtype == self.q_dtype, (
    #     f'Freezer got a QTensor of type {inputs.qvalue.dtype} but expected'
    #     f' {self.q_dtype}.'
    # )
    if self.quant_mode == QuantMode.TRAIN:
      pass
    elif self.quant_mode == QuantMode.CALIBRATE:
      pass
    elif self.quant_mode == QuantMode.CONVERT:
      self.qvalue.value = inputs.qvalue
      assert inputs.scale_t is not None and len(inputs.scale_t) == 1
      self.scale_t.value = inputs.scale_t[0]
    elif self.quant_mode == QuantMode.SERVE:
      # TODO(lew): Optionally compare stored and served value.
      pass
    else:
      assert False, 'Unknown quant mode.'
    return None


def _maybe_recover_scale_from_scale_t(
    qt: aqt_tensor.QTensor | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    is_rhs: bool,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> aqt_tensor.QTensor:
  """Recovers scale from scale_t if necessary."""
  if qt is None or qt.scale is not None or qt.scale_t is None:
    return qt

  transpose_fn = transpose.lhs_recover_scale_from_scale_t
  if is_rhs:
    transpose_fn = transpose.rhs_recover_scale_from_scale_t

  return qt.replace(
      scale=[
          transpose_fn(scale_t, dimension_numbers, lhs_shape, rhs_shape)
          for scale_t in qt.scale_t
      ],
      scale_t=None,
  )


def _populate_scale_t(
    qt: aqt_tensor.QTensor,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    is_rhs: bool,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> aqt_tensor.QTensor:
  """Populates scale_t from scale."""
  if qt.scale is None:
    return qt

  transpose_fn = transpose.lhs_scale_transpose_to_output
  if is_rhs:
    transpose_fn = transpose.rhs_scale_transpose_to_output

  return qt.replace(
      scale_t=[
          transpose_fn(scale, dimension_numbers, lhs_shape, rhs_shape)
          for scale in qt.scale
      ]
  )


class AqtDotGeneral(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

  cfg: Optional[aqt_dot_general.DotGeneral] = None
  prng_name: Optional[str] = 'params'

  # TODO(lew): split out separate class for each side.
  # Quant mode determines whether flax variables are created to store quantized
  # inputs. Refer to the Freezer doc str to see variable creation in each mode.
  lhs_quant_mode: QuantMode = QuantMode.TRAIN
  # apply_quant_mode determines if using Freezer in cfg.get/set_tensor
  lhs_apply_quant_mode: bool = True
  lhs_var_name: str = 'qlhs'
  lhs_qtensor: Optional[aqt_tensor.QTensor] = None

  rhs_quant_mode: QuantMode = QuantMode.TRAIN
  rhs_apply_quant_mode: bool = True
  rhs_var_name: str = 'qrhs'
  rhs_qtensor: Optional[aqt_tensor.QTensor] = None

  # Variables only for the legacy Freezer.
  lhs_init: nn.initializers.Initializer = jnp.zeros
  lhs_scale_init: nn.initializers.Initializer = jnp.zeros

  rhs_init: nn.initializers.Initializer = jnp.zeros
  rhs_scale_init: nn.initializers.Initializer = jnp.zeros

  # Variables only for the new Freezer.
  lhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None
  rhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None

  # Freeze mode. Set as FreezerMode.CALIBRATION to store only scales; set as
  # CALIBRATION_AND_VALUE to store both scales and quantized values.
  lhs_freeze_mode: FreezerMode = FreezerMode.NONE
  rhs_freeze_mode: FreezerMode = FreezerMode.NONE

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  quant_collection: str = 'aqt'

  # Tiling configs. tilling_fn is valid only when tiling_cfg is None.
  tiling_cfg: Optional[tiled_dot_general.Cfg] = None
  tiling_fn: Optional[DotGeneralTilingFn] = None

  # If set to True, use the current Freezer. Otherwise, use the new
  # QuantFreezer.
  use_legacy_freezer: bool = True

  def make_aqt_dg(
      self,
      lhs_shape,
      rhs_shape,
      dimension_numbers: tuple[Iterable[int], Iterable[int]],
      lhs_tile_map: tiled_dot_general.AqtTileMap | None = None,
      rhs_tile_map: tiled_dot_general.AqtTileMap | None = None,
  ):
    if self.cfg is None:
      return jax.lax.dot_general

    cfg = copy.deepcopy(self.cfg)
    lhs_scale_shape = list(lhs_shape)
    rhs_scale_shape = list(rhs_shape)
    (contr, _) = dimension_numbers
    for li, ri in zip(*contr):
      lhs_scale_shape[li] = 1
      rhs_scale_shape[ri] = 1
    lhs_scale = transpose.lhs_scale_transpose_to_output(
        jnp.zeros(lhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert lhs_scale is not None
    lhs_scale_shape = lhs_scale.shape
    rhs_scale = transpose.rhs_scale_transpose_to_output(
        jnp.zeros(rhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert rhs_scale is not None
    rhs_scale_shape = rhs_scale.shape
    rhs_qm = self.rhs_quant_mode
    lhs_qm = self.lhs_quant_mode

    assert isinstance(
        cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
    )
    lhs_q_dtype = cfg.fwd.dg_quantizer.lhs.numerics.get_dtype()
    rhs_q_dtype = cfg.fwd.dg_quantizer.rhs.numerics.get_dtype()

    if self.use_legacy_freezer:
      lhs_freezer = Freezer(
          name=self.lhs_var_name,
          quant_mode=lhs_qm,
          q_shape=lhs_shape,
          q_dtype=lhs_q_dtype,
          q_init=self.lhs_init,
          s_shape=lhs_scale_shape,
          s_init=self.lhs_scale_init,
          quant_collection=self.quant_collection,
      )

      rhs_freezer = Freezer(
          name=self.rhs_var_name,
          quant_mode=rhs_qm,
          q_shape=rhs_shape,
          q_dtype=rhs_q_dtype,
          q_init=self.rhs_init,
          s_shape=rhs_scale_shape,
          s_init=self.rhs_scale_init,
          quant_collection=self.quant_collection,
      )
    else:
      quant_to_freezer_mode = {
          QuantMode.TRAIN: general_freezer.FreezerMode.NONE,
          QuantMode.CALIBRATE: general_freezer.FreezerMode.NONE,
          QuantMode.CONVERT: general_freezer.FreezerMode.WRITE,
          QuantMode.SERVE: general_freezer.FreezerMode.READ,
      }

      def init_wrapper(
          qt: aqt_tensor.QTensor,
          contracting_axis: Sequence[utils.AxisIdx],
          axis_metadata_wrapper: Optional[AxisMetadataWrapper],
          tile_map: tiled_dot_general.AqtTileMap,
      ):
        if axis_metadata_wrapper is None:
          return qt

        scale_non_shard_axis_all = list(range(qt.ndim))
        scale_non_shard_axis_contracting = list(contracting_axis)

        qt = qt.replace(
            qvalue=axis_metadata_wrapper(
                qt.qvalue,
                tile_map,
                [],
            ),
            scale=jax.tree.map(
                lambda x: axis_metadata_wrapper(
                    x, tile_map, scale_non_shard_axis_contracting
                ),
                qt.scale,
            ),
            # Passing scale_non_shard_axis_contracting would be incorrect due to
            # scale transposition. scale_t is being removed from QTensor anyway
            # so we just pass scale_non_shard_axis_all.
            scale_t=jax.tree.map(
                lambda x: axis_metadata_wrapper(
                    x, tile_map, scale_non_shard_axis_all
                ),
                qt.scale_t,
            ),
        )
        return qt

      lhs_ca, rhs_ca = contr
      lhs_init_wrapper = functools.partial(
          init_wrapper,
          contracting_axis=lhs_ca,
          axis_metadata_wrapper=self.lhs_axis_metadata_wrapper,
          tile_map=lhs_tile_map,
      )
      rhs_init_wrapper = functools.partial(
          init_wrapper,
          contracting_axis=rhs_ca,
          axis_metadata_wrapper=self.rhs_axis_metadata_wrapper,
          tile_map=rhs_tile_map,
      )

      lhs_freezer = general_freezer.Freezer(
          name=self.lhs_var_name,
          mode=quant_to_freezer_mode[lhs_qm],
          collection=self.quant_collection,
          axis_metadata_wrapper=lhs_init_wrapper,
      )

      rhs_freezer = general_freezer.Freezer(
          name=self.rhs_var_name,
          mode=quant_to_freezer_mode[rhs_qm],
          collection=self.quant_collection,
          axis_metadata_wrapper=rhs_init_wrapper,
      )

    prng_name = self.prng_name
    key = self.make_rng(prng_name) if prng_name is not None else None
    cfg = config.set_context(
        cfg,
        key,
        train_step=None,
        lhs_quant_mode=self.lhs_quant_mode,
        rhs_quant_mode=self.rhs_quant_mode,
    )

    def ret_dg(
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

      # TODO(yichizh): asserting xhs dtype only when apply_quant_mode=False
      # and cfg.get_qtensor() is None
      msg = 'AQT is not yet optimized to accept quantized types directly. '
      msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
      assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
      assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

      cfg.assert_config_validity()

      # Getter
      lhs_apply_quant_mode = self.lhs_apply_quant_mode
      rhs_apply_quant_mode = self.rhs_apply_quant_mode
      lhs_qt = lhs_freezer.get() if lhs_apply_quant_mode else self.lhs_qtensor
      rhs_qt = rhs_freezer.get() if rhs_apply_quant_mode else self.rhs_qtensor

      # Recover scale from scale_t, if necessary.
      # The quantized tensor loaded from the legacy freezer has only scale_t.
      lhs_qt = _maybe_recover_scale_from_scale_t(
          lhs_qt, dimension_numbers, False, lhs_shape, rhs_shape
      )
      rhs_qt = _maybe_recover_scale_from_scale_t(
          rhs_qt, dimension_numbers, True, lhs_shape, rhs_shape
      )

      cfg.apply_custom_vjp_on_jax = False
      out, (out_lhs_qt, out_rhs_qt) = aqt_flax_dg_core.dg_core_flax_lifted(
          lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, self, cfg
      )

      # Remove qvalue of the activation to not to store it in Freezer.
      match self.lhs_freeze_mode:
        case FreezerMode.NONE:
          if self.lhs_apply_quant_mode and self.lhs_quant_mode in {
              QuantMode.CONVERT,
              QuantMode.SERVE,
          }:
            raise ValueError('Freezer is used with Freezer mode NONE.')
        case FreezerMode.CALIBRATION:
          out_lhs_qt = out_lhs_qt.without_qvalue()
        case FreezerMode.CALIBRATION_AND_VALUE:
          pass
        case _:
          raise ValueError('Unknown freeze mode: %s' % self.lhs_freeze_mode)

      match self.rhs_freeze_mode:
        case FreezerMode.NONE:
          if self.rhs_apply_quant_mode and self.rhs_quant_mode in {
              QuantMode.CONVERT,
              QuantMode.SERVE,
          }:
            raise ValueError('Freezer is used with Freezer mode NONE.')
        case FreezerMode.CALIBRATION:
          out_rhs_qt = out_rhs_qt.without_qvalue()
        case FreezerMode.CALIBRATION_AND_VALUE:
          pass
        case _:
          raise ValueError('Unknown freeze mode: %s' % self.rhs_freeze_mode)

      # Setter
      calib_contracting_axis = aqt_dot_general.CalibrationMode.CONTRACTING_AXIS
      if self.use_legacy_freezer:
        # We need to populate the stored QTensor with scale_t.
        if cfg.fwd.lhs.calibration_mode == calib_contracting_axis:
          out_lhs_qt = _populate_scale_t(
              out_lhs_qt, dimension_numbers, False, lhs_shape, rhs_shape
          )
        if cfg.fwd.rhs.calibration_mode == calib_contracting_axis:
          out_rhs_qt = _populate_scale_t(
              out_rhs_qt, dimension_numbers, True, lhs_shape, rhs_shape
          )

      if self.lhs_apply_quant_mode:
        lhs_freezer.set(out_lhs_qt)
      if self.rhs_apply_quant_mode:
        rhs_freezer.set(out_rhs_qt)

      return out

    return ret_dg

  @nn.compact
  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision,
      preferred_element_type=None,
  ):
    tiling_cfg = self.tiling_cfg
    if tiling_cfg is None and self.tiling_fn is not None:
      tiling_cfg = self.tiling_fn(lhs, rhs, dimension_numbers)

    if tiling_cfg is not None:
      xlhs, xrhs = tiled_dot_general.generate_tiling_states_for_dot_general(
          tiling_cfg, lhs, rhs, dimension_numbers
      )
      # Extract tiled input shapes and dimension numbers from jaxpr
      def dummy_tiled_dg(lhs_in, rhs_in):
        return tiled_dot_general.tiled_dot_general(
            tiling_cfg, lhs_in, rhs_in, dimension_numbers
        )

      tiled_dg_jaxpr = jax.make_jaxpr(dummy_tiled_dg)(lhs, rhs)
      dg_eqn = [eqn for eqn in tiled_dg_jaxpr.eqns if 'dot_general' in str(eqn)]
      assert len(dg_eqn) == 1, 'Multiple dg calls are found in tiled dg jaxpr.'
      lhs_invar, rhs_invar = dg_eqn[0].invars
      tiled_lhs_shape = lhs_invar.aval.shape
      tiled_rhs_shape = rhs_invar.aval.shape
      tiled_dimension_numbers = dg_eqn[0].params['dimension_numbers']
      # Use tiled input shapes and dimension numbers to create aqt_dg that
      # will be injected to tiled_dot_general
      aqt_dg = self.make_aqt_dg(
          tiled_lhs_shape,
          tiled_rhs_shape,
          tiled_dimension_numbers,
          lhs_tile_map=xlhs.tile_map,
          rhs_tile_map=xrhs.tile_map,
      )
      # We integrate tiling here and not on Jax level, so that the Freezers
      # observe tiled shapes.
      ret_dg = functools.partial(
          tiled_dot_general.tiled_dot_general,
          tiling_cfg,
          dot_general=aqt_dg,
      )
    else:
      ret_dg = self.make_aqt_dg(lhs.shape, rhs.shape, dimension_numbers)
    return ret_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
    )


class AqtEinsum(nn.Module):
  """Quantized Einsum class for model injection."""

  cfg: Optional[config.DotGeneral] = None
  prng_name: Optional[str] = 'params'

  # TODO(lew): split out separate class for each side.
  lhs_quant_mode: QuantMode = QuantMode.TRAIN
  lhs_var_name: str = 'qlhs'

  rhs_quant_mode: QuantMode = QuantMode.TRAIN
  rhs_var_name: str = 'qrhs'

  # Variables only for the legacy Freezer.
  lhs_init: nn.initializers.Initializer = jnp.zeros
  lhs_scale_init: nn.initializers.Initializer = jnp.zeros

  rhs_init: nn.initializers.Initializer = jnp.zeros
  rhs_scale_init: nn.initializers.Initializer = jnp.zeros

  # Variables only for the new Freezer.
  lhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None
  rhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None

  # Freeze mode. Set as FreezerMode.CALIBRATION to store only scales; set as
  # CALIBRATION_AND_VALUE to store both scales and quantized values.
  lhs_freeze_mode: FreezerMode = FreezerMode.NONE
  rhs_freeze_mode: FreezerMode = FreezerMode.NONE

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  quant_collection: str = 'aqt'

  assert_eqn: Optional[str] = None
  assert_lhs_shape: Optional[utils.ShapeTemplate] = None
  assert_rhs_shape: Optional[utils.ShapeTemplate] = None
  tile_sizes: Optional[tiled_dot_general.EinsumTileSizes] = None
  tiling_fn: Optional[EinsumTilingFn] = None

  # If set to True, use the current Freezer. Otherwise, use the new
  # QTensorFreezer.
  use_legacy_freezer: bool = True

  @nn.compact
  def __call__(
      self,
      eqn,
      lhs_g: Union[jnp.ndarray, aqt_tensor.QTensor],
      rhs_g: Union[jnp.ndarray, aqt_tensor.QTensor],
  ):
    if self.assert_eqn is not None:
      utils.assert_eq(eqn, self.assert_eqn, 'einsum_eqn')
    if self.assert_lhs_shape is not None:
      utils.assert_shape(lhs_g.shape, self.assert_lhs_shape, 'lhs.shape')
    if self.assert_rhs_shape is not None:
      utils.assert_shape(rhs_g.shape, self.assert_rhs_shape, 'rhs.shape')

    cfg = self.cfg
    lhs_is_qt = isinstance(lhs_g, aqt_tensor.QTensor)
    rhs_is_qt = isinstance(rhs_g, aqt_tensor.QTensor)
    msg = 'Aqt config is None but inputs to AqtEinsum are QTensor.'
    assert not ((lhs_is_qt or rhs_is_qt) and cfg is None), msg
    # when inputs are qtensor, xhs_in is a dummy input that will be consumed by
    # lax einsum API, but it is not used for computation in aqt_dg because it
    # will be overwritten by get_tensor()
    # TODO(lew): We can pass QTensor to lax_numpy._einsum if we add some
    # specific methods to QTensor.
    lhs_in = jnp.zeros_like(lhs_g.qvalue) if lhs_is_qt else lhs_g
    rhs_in = jnp.zeros_like(rhs_g.qvalue) if rhs_is_qt else rhs_g

    # Set the types of dummy input to the same as original input, to prevent it
    # from being rejected by assertions in aqt_dot_general.py, line 522-526 and
    # 414.
    # TODO: b/322111904 - Handle this in more proper way.
    # We hand-hold int4 because promote_dtype(int4, x) fails.
    # (To avoid unintended promotion, 4-bit integers do not support
    # implicit promotion.)
    if lhs_in.dtype == jnp.int4:
      lhs_in = jnp.float32(lhs_in)
    if rhs_in.dtype == jnp.int4:
      rhs_in = jnp.float32(rhs_in)
    if lhs_in.dtype != jnp.int4 and rhs_in.dtype != jnp.int4:
      lhs_in, rhs_in = nn.dtypes.promote_dtype(lhs_in, rhs_in)

    # yes_swap = whether einsum swaps [lhs,rhs] when passing them to dot_general
    einsum = functools.partial(aqt_dot_general.einsum, eqn=eqn)
    a = jax.make_jaxpr(einsum)(lhs=lhs_in, rhs=rhs_in)
    [lhs_g_id, rhs_g_id] = a.eqns[0].invars
    [lhs_l_id, rhs_l_id] = a.jaxpr.invars
    not_swap = lhs_g_id == lhs_l_id and rhs_g_id == rhs_l_id
    yes_swap = lhs_g_id == rhs_l_id and rhs_g_id == lhs_l_id
    assert not_swap != yes_swap

    prng_name = self.prng_name

    lhs_quant_mode = self.lhs_quant_mode
    lhs_init = self.lhs_init
    lhs_axis_metadata_wrapper = self.lhs_axis_metadata_wrapper
    lhs_scale_init = self.lhs_scale_init
    lhs_var_name = self.lhs_var_name
    lhs_qtensor = lhs_g if lhs_is_qt else None

    rhs_quant_mode = self.rhs_quant_mode
    rhs_init = self.rhs_init
    rhs_axis_metadata_wrapper = self.rhs_axis_metadata_wrapper
    rhs_scale_init = self.rhs_scale_init
    rhs_var_name = self.rhs_var_name
    rhs_qtensor = rhs_g if rhs_is_qt else None

    lhs_freeze_mode = self.lhs_freeze_mode
    rhs_freeze_mode = self.rhs_freeze_mode

    quant_collection = self.quant_collection
    tiling_config = None
    if self.tile_sizes is not None:
      tiling_config = tiled_dot_general.Cfg.from_einsum(eqn, self.tile_sizes)
    elif self.tiling_fn is not None:
      tiling_config = self.tiling_fn(eqn, lhs_in, rhs_in)

    if yes_swap:
      if cfg is not None:
        cfg = copy.deepcopy(cfg)
        cfg.fwd.lhs, cfg.fwd.rhs = cfg.fwd.rhs, cfg.fwd.lhs
        cfg.fwd.dg_quantizer.swap_lhs_and_rhs()
        cfg.dlhs, cfg.drhs = cfg.drhs, cfg.dlhs
      lhs_quant_mode, rhs_quant_mode = rhs_quant_mode, lhs_quant_mode
      lhs_init, rhs_init = rhs_init, lhs_init
      lhs_scale_init, rhs_scale_init = rhs_scale_init, lhs_scale_init
      lhs_var_name, rhs_var_name = rhs_var_name, lhs_var_name
      lhs_is_qt, rhs_is_qt = rhs_is_qt, lhs_is_qt
      lhs_qtensor, rhs_qtensor = rhs_qtensor, lhs_qtensor
      lhs_axis_metadata_wrapper, rhs_axis_metadata_wrapper = (
          rhs_axis_metadata_wrapper,
          lhs_axis_metadata_wrapper,
      )
      lhs_freeze_mode, rhs_freeze_mode = rhs_freeze_mode, lhs_freeze_mode
      if tiling_config is not None:
        tiling_config = tiled_dot_general.Cfg(
            lhs=tiling_config.rhs, rhs=tiling_config.lhs
        )

    aqt_dg = AqtDotGeneral(
        cfg=cfg,
        prng_name=prng_name,
        lhs_quant_mode=lhs_quant_mode,
        # when passing pre-computed qtensor as inputs, apply_quant_mode flag
        # should be set to False so that Freezer will not be set to overwrite
        # the qtensor passed to dg.
        lhs_apply_quant_mode=not lhs_is_qt,  # Freezer not used if lhs is qt
        lhs_init=lhs_init,
        lhs_axis_metadata_wrapper=lhs_axis_metadata_wrapper,
        lhs_scale_init=lhs_scale_init,
        lhs_var_name=lhs_var_name,
        lhs_qtensor=lhs_qtensor,
        rhs_quant_mode=rhs_quant_mode,
        rhs_apply_quant_mode=not rhs_is_qt,  # Freezer not used if rhs is qt
        rhs_init=rhs_init,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        rhs_scale_init=rhs_scale_init,
        rhs_var_name=rhs_var_name,
        rhs_qtensor=rhs_qtensor,
        quant_collection=quant_collection,
        tiling_cfg=tiling_config,
        use_legacy_freezer=self.use_legacy_freezer,
        lhs_freeze_mode=lhs_freeze_mode,
        rhs_freeze_mode=rhs_freeze_mode,
    )
    return einsum(lhs=lhs_in, rhs=rhs_in, dg=aqt_dg)


class AqtConvGeneralDilated(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

  cfg: Optional[aqt_dot_general.DotGeneral] = None
  prng_name: Optional[str] = 'params'

  # TODO(lew): split out separate class for each side.
  # Quant mode determines whether flax variables are created to store quantized
  # inputs. Refer to the Freezer doc str to see variable creation in each mode.
  lhs_quant_mode: QuantMode = QuantMode.TRAIN
  # apply_quant_mode determines if using Freezer in cfg.get/set_tensor
  lhs_apply_quant_mode: bool = True
  lhs_var_name: str = 'qlhs'
  lhs_qtensor: Optional[aqt_tensor.QTensor] = None

  rhs_quant_mode: QuantMode = QuantMode.TRAIN
  rhs_apply_quant_mode: bool = True
  rhs_var_name: str = 'qrhs'
  rhs_qtensor: Optional[aqt_tensor.QTensor] = None

  # Variables only for the new Freezer.
  lhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None
  rhs_axis_metadata_wrapper: Optional[AxisMetadataWrapper] = None

  # Freeze mode. Set as FreezerMode.CALIBRATION to store only scales; set as
  # CALIBRATION_AND_VALUE to store both scales and quantized values.
  lhs_freeze_mode: FreezerMode = FreezerMode.NONE
  rhs_freeze_mode: FreezerMode = FreezerMode.NONE

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  quant_collection: str = 'aqt'

  def make_aqt_cg(
      self,
  ):
    if self.cfg is None:
      return jax.lax.conv_general_dilated

    cfg = copy.deepcopy(self.cfg)
    rhs_qm = self.rhs_quant_mode
    lhs_qm = self.lhs_quant_mode

    
    assert isinstance(
        cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
    )
    lhs_init_wrapper = functools.partial(
        _freezer_qtensor_init_wrapper,
        contracting_axis=[],
        axis_metadata_wrapper=self.lhs_axis_metadata_wrapper,
        tile_map=None,
    )
    rhs_init_wrapper = functools.partial(
        _freezer_qtensor_init_wrapper,
        contracting_axis=[],
        axis_metadata_wrapper=self.rhs_axis_metadata_wrapper,
        tile_map=None,
    )
    lhs_freezer = general_freezer.Freezer(
        name=self.lhs_var_name,
        mode=_QUANT_TO_FREEZER_MODE[lhs_qm],
        collection=self.quant_collection,
        axis_metadata_wrapper=lhs_init_wrapper,
    )
    rhs_freezer = general_freezer.Freezer(
        name=self.rhs_var_name,
        mode=_QUANT_TO_FREEZER_MODE[rhs_qm],
        collection=self.quant_collection,
        axis_metadata_wrapper=rhs_init_wrapper,
    )

    prng_name = self.prng_name
    key = self.make_rng(prng_name) if prng_name is not None else None

    cfg = config.set_context(
        cfg,
        key,
        train_step=None,
        lhs_quant_mode=self.lhs_quant_mode,
        rhs_quant_mode=self.rhs_quant_mode,
    )

    def ret_cg(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type,
    ):
      # TODO(yichizh): asserting xhs dtype only when apply_quant_mode=False
      # and cfg.get_qtensor() is None
      msg = 'AQT is not yet optimized to accept quantized types directly. '
      msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
      assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
      assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

      cfg.assert_config_validity()

      # Getter
      lhs_apply_quant_mode = self.lhs_apply_quant_mode
      rhs_apply_quant_mode = self.rhs_apply_quant_mode
      lhs_qt = lhs_freezer.get() if lhs_apply_quant_mode else self.lhs_qtensor
      rhs_qt = rhs_freezer.get() if rhs_apply_quant_mode else self.rhs_qtensor

      cfg.apply_custom_vjp_on_jax = False
      aqt_conv = aqt_conv_general.make_conv_general_dilated(cfg.fwd)
      out, (out_lhs_qt, out_rhs_qt) = aqt_conv(
          lhs,
          rhs,
          lhs_qt,
          rhs_qt,
          window_strides,
          padding,
          lhs_dilation,
          rhs_dilation,
          dimension_numbers,
          feature_group_count,
          batch_group_count,
          precision,
          preferred_element_type
          )

      # Remove qvalue of the activation to not to store it in Freezer.
      match self.lhs_freeze_mode:
        case FreezerMode.NONE:
          if self.lhs_apply_quant_mode and self.lhs_quant_mode in {
              QuantMode.CONVERT,
              QuantMode.SERVE,
          }:
            raise ValueError('Freezer is used with Freezer mode NONE.')
        case FreezerMode.CALIBRATION:
          out_lhs_qt = out_lhs_qt.without_qvalue()
        case FreezerMode.CALIBRATION_AND_VALUE:
          pass
        case _:
          raise ValueError('Unknown freeze mode: %s' % self.lhs_freeze_mode)

      match self.rhs_freeze_mode:
        case FreezerMode.NONE:
          if self.rhs_apply_quant_mode and self.rhs_quant_mode in {
              QuantMode.CONVERT,
              QuantMode.SERVE,
          }:
            raise ValueError('Freezer is used with Freezer mode NONE.')
        case FreezerMode.CALIBRATION:
          out_rhs_qt = out_rhs_qt.without_qvalue()
        case FreezerMode.CALIBRATION_AND_VALUE:
          pass
        case _:
          raise ValueError('Unknown freeze mode: %s' % self.rhs_freeze_mode)

      if self.lhs_apply_quant_mode:
        lhs_freezer.set(out_lhs_qt)
      if self.rhs_apply_quant_mode:
        rhs_freezer.set(out_rhs_qt)

      return out

    return ret_cg

  @nn.compact
  def __call__(
      self,
      lhs,
      rhs,
      window_strides: Sequence[int],
      padding: str | Sequence[tuple[int, int]],
      lhs_dilation: Sequence[int] | None = None,
      rhs_dilation: Sequence[int] | None = None,
      dimension_numbers: (
          jax.lax.ConvGeneralDilatedDimensionNumbers
          | tuple[str, str, str]
          | None
      ) = None,
      feature_group_count: int = 1,
      batch_group_count: int = 1,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type=None,
  ):
    ret_cg = self.make_aqt_cg()

    return ret_cg(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type,
    )


