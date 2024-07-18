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

from aqt.jax.v2.flax import aqt_flax
from ..common_types import Config
from dataclasses import dataclass
import jax.numpy as jnp
import jax
from jax.tree_util import tree_flatten_with_path, tree_unflatten


def _get_aqt_key_paths(aqt_vars):
  """Generate a list of paths which have aqt state"""
  aqt_tree_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_vars)
  aqt_key_paths = []
  for k, _ in aqt_tree_flat:
    pruned_keys = []
    for d in list(k):
      if "AqtDotGeneral" in d.key:
        pruned_keys.append(jax.tree_util.DictKey(key="kernel"))
        break
      else:
        assert "Aqt" not in d.key, f"Unexpected Aqt op {d.key} in {k}."
        pruned_keys.append(d)
    aqt_key_paths.append(tuple(pruned_keys))
  return aqt_key_paths


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  aqt_paths = _get_aqt_key_paths(aqt_vars)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in aqt_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)

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
  lhs_quant_mode: aqt_flax.QuantMode
  rhs_quant_mode: aqt_flax.QuantMode

  def dot_general_cls(self):
    """ Returns dot_general configured with aqt params. """
    # return None
    aqt_dg_cls = functools.partial(
      aqt_flax.AqtDotGeneral,
      self.quant_dg,
      lhs_quant_mode=self.lhs_quant_mode,
      rhs_quant_mode=self.rhs_quant_mode,
      lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
      rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
      )
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
  def conv_general_dialated(self):
    # return None
    conv_general = functools.partial(aqt_conv_general.make_conv_general_dilated(
       aqt_config.conv_general_dilated_make(lhs_bits=8, rhs_bits=8)))
    return conv_general
  def get_quant_mode(self, quant_mode_str: str = 'train'):
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
  
def _get_quant_config(config):
  return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=8,
      rng_type='jax.uniform',
      dlhs_local_aqt=None,
      drhs_local_aqt=None,
      fwd_accumulator_dtype=jnp.float32,
      dlhs_accumulator_dtype=jnp.float32,
      drhs_accumulator_dtype=jnp.float32,
    )

def _get_quant_config_old(config):
  if not config.quantization or config.quantization == '':
    return None
  elif config.quantization == "int8":
    if config.quantization_local_shard_count == 0:
      drhs_bits = None
      drhs_accumulator_dtype = None
      drhs_local_aqt=None
    else:
      drhs_bits = 8
      drhs_accumulator_dtype = jnp.int32
      print(config.quantization_local_shard_count) # -1
      drhs_local_aqt = aqt_config.LocalAqt(contraction_axis_shard_count=config.quantization_local_shard_count)
    return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type='jax.uniform',
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
    )
  else:
    raise ValueError(f'Invalid value configured for quantization {config.quantization}.')

def in_convert_mode(quant):
  return quant and (quant.rhs_quant_mode == aqt_flax.QuantMode.CONVERT)

def in_serve_mode(quant):
  return quant and (quant.rhs_quant_mode == aqt_flax.QuantMode.SERVE)

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

def configure_quantization(config: Config, lhs_quant_mode=aqt_flax.QuantMode.TRAIN, rhs_quant_mode=aqt_flax.QuantMode.TRAIN):
  """ Configure quantization based on user config and quant mode."""
  # quant_cfg = _get_quant_config(config)
  quant_cfg = create_weight_only_cfg(config)
  if quant_cfg:
    return AqtQuantization(quant_dg=quant_cfg, lhs_quant_mode=lhs_quant_mode, rhs_quant_mode=rhs_quant_mode)
  return None

def configure_quantizatio_old(config: Config, lhs_quant_mode=aqt_flax.QuantMode.TRAIN, rhs_quant_mode=aqt_flax.QuantMode.TRAIN):
  """ Configure quantization based on user config and quant mode."""

  if not config:
    return None
  # quant_cfg = _get_quant_config(config)
  quant_cfg = create_weight_only_cfg(8)
  if quant_cfg:
    return AqtQuantization(quant_dg=quant_cfg, lhs_quant_mode=lhs_quant_mode, rhs_quant_mode=rhs_quant_mode)
  return None

# @dataclass
# class AqtQuantization:
#   """ Configures AQT quantization github.com/google/aqt. """
#   quant_dg: aqt_config.DotGeneral
#   quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN




#   def dot_general_cls_aqt(self, aqt_cfg, lhs_quant_mode, rhs_quant_mode):
#     """ Returns dot_general configured with aqt params. """
#     aqt_dg_cls = functools.partial(
#       aqt_flax.AqtDotGeneral,
#       aqt_cfg,
#       lhs_quant_mode=lhs_quant_mode,
#       rhs_quant_mode=rhs_quant_mode,
#       lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
#       rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
#       )
#     return aqt_dg_cls

#   def einsum_aqt(self, aqt_cfg, lhs_quant_mode, rhs_quant_mode):
#     return functools.partial(
#       aqt_flax.AqtEinsum,
#       aqt_cfg,
#       lhs_quant_mode=lhs_quant_mode,
#       rhs_quant_mode=rhs_quant_mode,
#       lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
#       rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
#     )
  