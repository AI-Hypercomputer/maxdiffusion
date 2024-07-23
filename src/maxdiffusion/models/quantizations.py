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

import functools

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_conv_general
from aqt.jax.v2 import utils
from aqt.jax.v2.flax import aqt_flax

from ..common_types import Config
from dataclasses import dataclass
import jax.numpy as jnp
import copy

def create_weight_only_cfg(w_bits):
  # return aqt_config.dot_general_make(
  #     lhs_bits=None, rhs_bits=8
  # )
  cfg = aqt_config.default_unquantized_config()
  aqt_config.set_bits(
      cfg,
      fwd_lhs_bit=8,
      fwd_rhs_bit=8,
      dlhs_lhs_bit=None,
      dlhs_rhs_bit=None,
      drhs_lhs_bit=None,
      drhs_rhs_bit=None,
  )
  return cfg

@dataclass
class AqtQuantization:
  """ Configures AQT quantization github.com/google/aqt. """
  quant_dg: aqt_config.DotGeneral
  quant_dg_conv: aqt_config.DotGeneralRaw
  lhs_quant_mode: aqt_flax.QuantMode
  rhs_quant_mode: aqt_flax.QuantMode
  activation_quant_mode: utils.QuantMode = utils.QuantMode.CONVERT
  weights_quant_mode: utils.QuantMode = utils.QuantMode.CONVERT

  def dot_general_cls(self):
    """ Returns dot_general configured with aqt params. """
    aqt_dg_cls = functools.partial(
      aqt_flax.AqtDotGeneral,
      self.quant_dg,
      lhs_quant_mode=self.lhs_quant_mode,
      rhs_quant_mode=self.rhs_quant_mode,
      lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
      rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
      )
    return None
    # return aqt_dg_cls
    #return None

  def conv_general(self):
      """ Returns dot_general configured with aqt params. """
      aqt_cfg_conv = copy.deepcopy(self.quant_dg)
      aqt_cfg_conv = aqt_cfg_conv.replace(fwd=self.quant_dg_conv)
      # breakpoint()
      aqt_dg_cls = functools.partial(
              aqt_flax.AqtConvGeneralDilated,
              aqt_cfg_conv,
              lhs_quant_mode=self.activation_quant_mode,
              rhs_quant_mode=self.weights_quant_mode,
              lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
              rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,)
      return aqt_dg_cls

  def einsum(self):
    """ Returns einsum configured with aqt params """
    aqt_einsum = functools.partial(aqt_flax.AqtEinsum(
      cfg=self.quant_dg,
      lhs_quant_mode=self.lhs_quant_mode,
      rhs_quant_mode=self.rhs_quant_mode,
      lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
      rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
      )
    )
    return aqt_einsum

  # def conv_general_dialated(self):
  #   conv_general = functools.partial(aqt_conv_general.make_conv_general_dilated(
  #      aqt_config.conv_general_dilated_make(lhs_bits=8, rhs_bits=8, spatial_dimensions=2, window_strides=(1,1), padding=((1, 1), (1, 1)))))
  #   return conv_general


def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)


def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


def get_quant_mode(quant_mode_str: str = 'train'):
  """ Set quant mode."""
  if quant_mode_str == 'train':
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == 'serve':
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == 'convert':
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f'Invalid quantization mode {quant_mode_str}.')
  return None


def configure_quantization(
    config: Config,
    lhs_quant_mode=aqt_flax.QuantMode.TRAIN,
    rhs_quant_mode=aqt_flax.QuantMode.TRAIN,
    activation_quant_mode=aqt_flax.QuantMode.TRAIN,
    weights_quant_mode=aqt_flax.QuantMode.TRAIN
    ):
  """ Configure quantization based on user config and quant mode."""
  # quant_cfg = _get_quant_config(config)
  quant_cfg = create_weight_only_cfg(config)
  quant_dg_conv = aqt_config.conv_general_dilated_make(
      2, lhs_bits=8, rhs_bits=8, initialize_calibration=False)
  if quant_cfg:
    return AqtQuantization(
      quant_dg=quant_cfg,
      quant_dg_conv=quant_dg_conv,
      lhs_quant_mode=lhs_quant_mode,
      rhs_quant_mode=rhs_quant_mode,
      activation_quant_mode=activation_quant_mode,
      weights_quant_mode=weights_quant_mode
      )
  return None

