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
"""Configuration dataclasses."""


import copy
import functools
from typing import Literal, Optional, TypeAlias, Union
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils

# Temporary re-export from aqt.jax.v2.aqt_dot_general
# TODO(lew): Remove these imports, use setters instead
# pylint: disable=g-importing-member
# pylint: disable=unused-import
from aqt.jax.v2.aqt_conv_general import conv_general_dilated_make
from aqt.jax.v2.aqt_dot_general import CalibrationMode
from aqt.jax.v2.aqt_dot_general import DequantMode
from aqt.jax.v2.aqt_dot_general import dot_general_make
from aqt.jax.v2.aqt_dot_general import dot_general_raw_make
from aqt.jax.v2.aqt_dot_general import DotGeneral
from aqt.jax.v2.aqt_dot_general import DotGeneralRaw
from aqt.jax.v2.aqt_dot_general import dtypes_allowed_for_int32_accum
from aqt.jax.v2.aqt_dot_general import LocalAqt
from aqt.jax.v2.aqt_dot_general import Tensor

from aqt.jax.v2.aqt_quantizer import quantizer_make

from aqt.jax.v2.numerics import fp8_numerics
from aqt.jax.v2.numerics import fp_numerics
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp

################################################################################
# Functions below are auxiliary config attribute setters.

# SKIP can be used as an argument to some set_xyz functions.
# It signals that set_xyz should make no changes to that option.
SKIP = 'skip'
SkipT: TypeAlias = Literal[SKIP]


def _split_key(key: Optional[jax.Array], num_splits: int):
  default = (None,) * num_splits
  return default if key is None else jax.random.split(key, num_splits)


def set_context(
    cfg: DotGeneral,
    key: Optional[jax.Array],
    train_step: Optional[int],
    lhs_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN,
    rhs_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN,
):
  """Set context with prng keys and train_steps for dot_general config."""

  def set_dg_raw_context(cfg_raw: DotGeneralRaw, key: Optional[jax.Array]):
    key1, key2 = _split_key(key, num_splits=2)
    lhs_context = utils.Context(
        key=key1, train_step=train_step, quant_mode=lhs_quant_mode
    )
    rhs_context = utils.Context(
        key=key2, train_step=train_step, quant_mode=rhs_quant_mode
    )
    cfg_raw.dg_quantizer.set_context(lhs_context, rhs_context)

  key_fwd, key_dlhs, key_drhs = _split_key(key, num_splits=3)
  ret_cfg = copy.deepcopy(cfg)
  set_dg_raw_context(ret_cfg.fwd, key_fwd)
  set_dg_raw_context(ret_cfg.dlhs, key_dlhs)
  set_dg_raw_context(ret_cfg.drhs, key_drhs)
  return ret_cfg


def set_fwd_dequant_mode(
    cfg: DotGeneral,
    *,
    lhs_dequant_mode: Optional[DequantMode] = None,
    rhs_dequant_mode: Optional[DequantMode] = None,
):
  if lhs_dequant_mode is not None:
    cfg.fwd.lhs.dequant_mode = lhs_dequant_mode
  if rhs_dequant_mode is not None:
    cfg.fwd.rhs.dequant_mode = rhs_dequant_mode


def set_fwd_calibration_mode(
    cfg: DotGeneral,
    *,
    lhs_calibration_mode: CalibrationMode | SkipT = SKIP,
    rhs_calibration_mode: CalibrationMode | SkipT = SKIP,
):
  if lhs_calibration_mode != SKIP:
    cfg.fwd.lhs.calibration_mode = lhs_calibration_mode
  if rhs_calibration_mode != SKIP:
    cfg.fwd.rhs.calibration_mode = rhs_calibration_mode


def set_numerics(
    cfg: DotGeneralRaw,
    lhs_numerics: numerics.AqtNumerics,
    rhs_numerics: numerics.AqtNumerics,
):
  """Set numerics for DotGeneralRaw config."""
  assert isinstance(
      cfg.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  cfg.dg_quantizer.lhs.numerics = lhs_numerics
  cfg.dg_quantizer.rhs.numerics = rhs_numerics
  if (
      lhs_numerics.get_dtype() in dtypes_allowed_for_int32_accum
      and rhs_numerics.get_dtype() in dtypes_allowed_for_int32_accum
  ):
    cfg.dg_accumulator_dtype = jnp.int32
  elif (
      lhs_numerics.get_dtype() in fp8_numerics.fp8_map.values()
      or rhs_numerics.get_dtype() in fp8_numerics.fp8_map.values()
  ):
    cfg.dg_accumulator_dtype = jnp.float32
  else:
    cfg.dg_accumulator_dtype = None


def set_accumulator_dtype(
    cfg: DotGeneral,
    fwd_dtype: Union[jnp.dtype, None, SkipT],
    dlhs_dtype: Union[jnp.dtype, None, SkipT],
    drhs_dtype: Union[jnp.dtype, None, SkipT],
):
  if fwd_dtype != SKIP:
    cfg.fwd.dg_accumulator_dtype = fwd_dtype
  if dlhs_dtype != SKIP:
    cfg.dlhs.dg_accumulator_dtype = dlhs_dtype
  if drhs_dtype != SKIP:
    cfg.drhs.dg_accumulator_dtype = drhs_dtype


def set_stochastic_rounding(
    cfg: DotGeneral,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: bool,
    vjp_rhs_stochastic_rounding: bool,
    implementation: str,
):
  """Configure stochastic rounding implementation."""
  noise_implementations = {
      'jax.uniform': stochastic_rounding.JaxUniform(),
      'custom-1': stochastic_rounding.RandomCenteredUniform(),
  }
  msg = f'{implementation} not supported.'
  assert implementation in noise_implementations.keys(), msg
  noise_fn = noise_implementations[implementation]
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  def _set_numerics_noise_fn(
      dg_quantizer: aqt_dot_general.DefaultDotGeneralQuantizer,
      is_lhs_sr: bool,
      is_rhs_sr: bool,
  ) -> None:
    # set stochastic noise for each dg_quantizer
    def _set_noise_fn(
        quantizer: aqt_quantizer.Quantizer,
        is_sr: bool,
    ) -> None:
      # set stochastic noise for each side of a dg_quantizer
      if isinstance(quantizer.numerics, fp8_numerics.Fp8Numerics):
        quantizer.numerics.stochastic_rounding = is_sr
      else:
        quantizer.numerics.noise_fn = noise_fn if is_sr else None

    _set_noise_fn(dg_quantizer.lhs, is_lhs_sr)
    _set_noise_fn(dg_quantizer.rhs, is_rhs_sr)

  is_lhs_sr = vjp_lhs_stochastic_rounding
  is_rhs_sr = vjp_rhs_stochastic_rounding
  _set_numerics_noise_fn(cfg.dlhs.dg_quantizer, is_lhs_sr, is_rhs_sr)
  _set_numerics_noise_fn(cfg.drhs.dg_quantizer, is_lhs_sr, is_rhs_sr)


def set_static_bound(cfg: DotGeneral, bound: float = 1.0):
  """Sets the static bound for calibration."""
  calibration_cls = functools.partial(
      calibration.ConstantCalibration, bound=bound
  )

  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  cfg.fwd.dg_quantizer.lhs.calibration = calibration_cls
  cfg.fwd.dg_quantizer.rhs.calibration = calibration_cls
  cfg.dlhs.dg_quantizer.lhs.calibration = calibration_cls
  cfg.dlhs.dg_quantizer.rhs.calibration = calibration_cls
  cfg.drhs.dg_quantizer.lhs.calibration = calibration_cls
  cfg.drhs.dg_quantizer.rhs.calibration = calibration_cls


def set_local_aqt(
    cfg: DotGeneral,
    fwd_local_aqt: Union[SkipT, LocalAqt, None],
    dlhs_local_aqt: Union[SkipT, LocalAqt, None],
    drhs_local_aqt: Union[SkipT, LocalAqt, None],
):
  if fwd_local_aqt != SKIP:
    cfg.fwd.local_aqt = fwd_local_aqt
  if dlhs_local_aqt != SKIP:
    cfg.dlhs.local_aqt = dlhs_local_aqt
  if drhs_local_aqt != SKIP:
    cfg.drhs.local_aqt = drhs_local_aqt


def set_use_fwd_quant(
    cfg: DotGeneral,
    dlhs_use_fwd_quant: Union[bool, None, SkipT],
    drhs_use_fwd_quant: Union[bool, None, SkipT],
):
  if dlhs_use_fwd_quant != SKIP:
    cfg.dlhs.rhs.use_fwd_quant = dlhs_use_fwd_quant
  if drhs_use_fwd_quant != SKIP:
    cfg.drhs.rhs.use_fwd_quant = drhs_use_fwd_quant


def set_int_numerics_preserve_zero(cfg: DotGeneral, preserve_zero: bool):
  """Set preserve_zero for int_numerics."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  for dot_general_raw in [cfg.fwd, cfg.dlhs, cfg.drhs]:
    dg_quantizer = dot_general_raw.dg_quantizer
    for q_numerics in [dg_quantizer.lhs.numerics, dg_quantizer.rhs.numerics]:
      if isinstance(q_numerics, int_numerics.IntNumerics):
        q_numerics.preserve_zero = preserve_zero
        updated_dtype = (
            utils.infer_dtype_from_bits(q_numerics.bits)  # pytype: disable=attribute-error
            if preserve_zero
            else None
        )
        q_numerics.dtype = updated_dtype


def set_absmax_calib_scale(cfg: DotGeneral, scale: float):
  """Set AbsMaxCalibration scale and update clip_gradient accordingly."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  for dot_general_raw in [cfg.fwd, cfg.dlhs, cfg.drhs]:
    dg_quantizer = dot_general_raw.dg_quantizer
    for quantizer in [dg_quantizer.lhs, dg_quantizer.rhs]:
      calibration_cls = quantizer.calibration
      if isinstance(calibration_cls, functools.partial):
        calibration_cls = calibration_cls.func
      assert calibration_cls == calibration.AbsMaxCalibration, (
          'scale is only available in AbsMaxCalibration, while'
          f' {quantizer.calibration} is used in current config.'
      )
      quantizer.calibration = functools.partial(
          calibration.AbsMaxCalibration, scale=scale
      )
      if scale < 1.0 and isinstance(
          quantizer.numerics, int_numerics.IntNumerics
      ):
        quantizer.numerics.clip_gradient = True


def set_bits(
    cfg: DotGeneral,
    fwd_lhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
    fwd_rhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
    dlhs_lhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
    dlhs_rhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
    drhs_lhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
    drhs_rhs_bit: Union[int, None, fp8_numerics.FP8Dtype],
) -> DotGeneral:
  """Set quantization bits for dot_general config."""

  def get_numerics(bits):
    if bits is None:
      effective_numerics = no_numerics.NoNumerics()
    elif bits in fp8_numerics.fp8_map.keys():
      exponent_bits, mantissa_bits = int(bits[1]), int(bits[3])
      effective_numerics = fp8_numerics.Fp8Numerics(
          exponent_bits=exponent_bits,
          mantissa_bits=mantissa_bits,
          dtype=fp8_numerics.fp8_map[bits],
      )
    else:
      pz = False if bits == 1 else True
      dtype = utils.infer_dtype_from_bits(bits) if pz else None
      effective_numerics = int_numerics.IntNumerics(
          bits=bits,
          preserve_zero=pz,
          preserve_max_val=False,
          clip=True,
          round=True,
          noise_fn=None,
          clip_gradient=False,  # Can be disabled when using abs-max scaling.
          dtype=dtype,
      )
    return effective_numerics

  set_numerics(cfg.fwd, get_numerics(fwd_lhs_bit), get_numerics(fwd_rhs_bit))
  set_numerics(cfg.dlhs, get_numerics(dlhs_lhs_bit), get_numerics(dlhs_rhs_bit))
  set_numerics(cfg.drhs, get_numerics(drhs_lhs_bit), get_numerics(drhs_rhs_bit))
  # use_fwd_quant is by default set to False if fwd pass is quantized.
  # This is to make the configuration logically correct,
  # i.e., use_fwd_quant cannot be None when fwd is quantized.
  # It is user's responsibility to further choose between False and True.
  dlhs_use_fwd_quant = False if fwd_rhs_bit is not None else SKIP
  drhs_use_fwd_quant = False if fwd_lhs_bit is not None else SKIP
  set_use_fwd_quant(cfg, dlhs_use_fwd_quant, drhs_use_fwd_quant)
  return cfg


def set_scale_dtype(cfg: DotGeneral, scale_dtype: jnp.dtype):
  """Set the dtype for all scales in the given DotGeneral config."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  cfg.fwd.dg_quantizer.lhs.scale_dtype = scale_dtype
  cfg.fwd.dg_quantizer.rhs.scale_dtype = scale_dtype
  cfg.dlhs.dg_quantizer.lhs.scale_dtype = scale_dtype
  cfg.dlhs.dg_quantizer.rhs.scale_dtype = scale_dtype
  cfg.drhs.dg_quantizer.lhs.scale_dtype = scale_dtype
  cfg.drhs.dg_quantizer.rhs.scale_dtype = scale_dtype


################################################################################
# Functions below are auxiliary config creators.


def default_unquantized_config() -> DotGeneral:
  """Aqt config for floating-point dot general."""

  def tensor_cfg() -> Tensor:
    cfg = Tensor(
        use_fwd_quant=False,
        dequant_mode=DequantMode.OUTPUT,
        calibration_mode=CalibrationMode.CONTRACTING_AXIS,
    )
    return cfg

  def quantizer() -> aqt_quantizer.Quantizer:
    return aqt_quantizer.Quantizer(
        numerics=no_numerics.NoNumerics(),
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=calibration.AbsMaxCalibration,
        po2_scale=False,
        context=utils.Context(key=None, train_step=None),
    )

  def dg_raw_cfg(jax_scope_name: str) -> DotGeneralRaw:
    return DotGeneralRaw(
        lhs=tensor_cfg(),
        rhs=tensor_cfg(),
        dg_quantizer=aqt_dot_general.DefaultDotGeneralQuantizer(
            lhs=quantizer(), rhs=quantizer()
        ),
        dg_accumulator_dtype=None,
        local_aqt=None,
        jax_scope_name=jax_scope_name,
    )

  dg_cfg = DotGeneral(
      fwd=dg_raw_cfg('aqt_fwd'),
      dlhs=dg_raw_cfg('aqt_dlhs'),
      drhs=dg_raw_cfg('aqt_drhs'),
  )
  return dg_cfg


def fully_quantized(
    *,
    fwd_bits: Optional[int] = 8,
    bwd_bits: Optional[int] = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: Optional[bool] = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: Optional[bool] = None,
    vjp_rhs_stochastic_rounding: Optional[bool] = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  cfg = dot_general_make(
      lhs_bits=fwd_bits,
      rhs_bits=fwd_bits,
      bwd_bits=bwd_bits,
      use_fwd_quant=use_fwd_quant,
      dlhs_local_aqt=dlhs_local_aqt,
      drhs_local_aqt=drhs_local_aqt,
  )

  # Stochastic Rounding
  # These 3 variables are used to ensure we don't mix
  # old and new style of SR configuration.
  old_style_sr_config = use_stochastic_rounding is not None
  new_style_sr_config_lhs = vjp_lhs_stochastic_rounding is not None
  new_style_sr_config_rhs = vjp_rhs_stochastic_rounding is not None
  assert new_style_sr_config_lhs == new_style_sr_config_rhs, (
      'if you use new style SR config (vjp_xhs_stochastic_rounding), do pass'
      ' both lhs and rhs explicitely.'
  )
  assert new_style_sr_config_lhs != old_style_sr_config

  true = True  # A crude way to get around g-explicit-bool-comparison warning

  assert not (vjp_lhs_stochastic_rounding and vjp_rhs_stochastic_rounding), (
      'This config is buggy when you set both to True. Contact lew@ or use'
      ' config_v3'
  )

  # By default use jax.uniform for stochastic rounding
  if use_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, True, 'jax.uniform')

  if vjp_lhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, False, 'jax.uniform')

  if vjp_rhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, False, True, 'jax.uniform')

  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)

  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'

  return cfg


def config_v3(
    *,
    fwd_bits: Optional[int] = 8,
    dlhs_bits: Optional[int] = 8,
    drhs_bits: Optional[int] = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
    fwd_accumulator_dtype: ... = jnp.int32,
    dlhs_accumulator_dtype: ... = jnp.int32,
    drhs_accumulator_dtype: ... = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  fwd = dot_general_raw_make(
      fwd_bits,
      fwd_bits,
      jax_scope_name='aqt_fwd',
      initialize_calibration=False,
  )
  dlhs = dot_general_raw_make(
      dlhs_bits,
      dlhs_bits,
      local_aqt=dlhs_local_aqt,
      jax_scope_name='aqt_dlhs',
      initialize_calibration=False,
  )
  drhs = dot_general_raw_make(
      drhs_bits,
      drhs_bits,
      local_aqt=drhs_local_aqt,
      jax_scope_name='aqt_drhs',
      initialize_calibration=False,
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  cfg.dlhs.rhs.use_fwd_quant = False
  cfg.drhs.rhs.use_fwd_quant = False

  # Typically we have (but I don't know if it is guraranteed):
  # - vjp_lhs_stochastic_rounding is referring to the gradient and
  # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
  set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )

  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)

  set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def config_v4(
    *,
    fwd_bits: Union[int, None, fp8_numerics.FP8Dtype] = 8,
    dlhs_bits: Union[int, None, fp8_numerics.FP8Dtype] = 8,
    drhs_bits: Union[int, None, fp8_numerics.FP8Dtype] = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
    # accumulator dtype by default is automatically set in set_bits,
    # but users can still configure a special dtype such as jnp.int16, etc.
    fwd_accumulator_dtype: Union[jnp.dtype, None, SkipT] = SKIP,
    dlhs_accumulator_dtype: Union[jnp.dtype, None, SkipT] = SKIP,
    drhs_accumulator_dtype: Union[jnp.dtype, None, SkipT] = SKIP,
    dlhs_use_fwd_quant: Union[bool, None, SkipT] = SKIP,
    drhs_use_fwd_quant: Union[bool, None, SkipT] = SKIP,
) -> DotGeneral:
  """Version 4 of user-visible AQT config."""
  cfg = default_unquantized_config()
  set_bits(
      cfg,
      fwd_lhs_bit=fwd_bits,
      fwd_rhs_bit=fwd_bits,
      dlhs_lhs_bit=dlhs_bits,
      dlhs_rhs_bit=dlhs_bits,
      drhs_lhs_bit=drhs_bits,
      drhs_rhs_bit=drhs_bits,
  )
  set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )
  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)
  set_local_aqt(
      cfg,
      fwd_local_aqt=SKIP,
      dlhs_local_aqt=dlhs_local_aqt,
      drhs_local_aqt=drhs_local_aqt,
  )
  set_use_fwd_quant(
      cfg,
      dlhs_use_fwd_quant=dlhs_use_fwd_quant,
      drhs_use_fwd_quant=drhs_use_fwd_quant,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def config_fwd_fp8(fwd_bits: fp8_numerics.FP8Dtype = 'e4m3') -> DotGeneral:
  """Configs for FP8 forward pass."""
  assert (
      fwd_bits in fp8_numerics.fp8_map.keys()
  ), 'FP8 only supports 4 or 5 exponent bits'
  cfg = config_v4(fwd_bits=8, dlhs_bits=None, drhs_bits=None)
  set_bits(
      cfg,
      fwd_lhs_bit=fwd_bits,
      fwd_rhs_bit=fwd_bits,
      dlhs_lhs_bit=None,
      dlhs_rhs_bit=None,
      drhs_lhs_bit=None,
      drhs_rhs_bit=None,
  )
  set_stochastic_rounding(cfg, False, False, 'jax.uniform')
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def set_fwd_calibration(
    cfg: DotGeneral,
    calibration_factory
) -> DotGeneral:
  """Updates aqt_cfg for static range calibration."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  cfg.fwd.dg_quantizer.lhs.calibration = calibration_factory
  cfg.fwd.dg_quantizer.rhs.calibration = calibration_factory
