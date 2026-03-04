#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.flax import aqt_flax
from .. import common_types
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, tree_unflatten
from typing import Tuple, Sequence

# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config

MAX_INT8 = 127.5
MAX_INT4 = 7.5
Array = common_types.Array
DType = common_types.DType


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    pass

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""
    pass


def _rhs_axis_metadata_wrapper(
    x: jnp.ndarray,
    tile_map,
    no_sharding_axis: Sequence[int],
    mesh_axes: Tuple[str, ...],
    is_tiled: bool,
    replicate_scale: bool = False,
):
  if replicate_scale:
    # Temporarily using the shape to identify the scale.
    # TODO: remove the replication once the 2d sharding quantization
    # works as expected.
    if len(x.shape) == 1:
      return nn.with_logical_partitioning((lambda: x), tuple([None for _ in mesh_axes]))()

  mesh_axes = list(mesh_axes)
  if is_tiled:
    # tile_map is a mapping between original rank and a list of new, tiled rank.
    if len(mesh_axes) < len(tile_map):
      mesh_axes = [None] * (len(tile_map) - len(mesh_axes)) + mesh_axes
    new_mesh_axes = [None] * len(x.shape)
    for orig_rank, new_rank in tile_map.items():
      assert new_rank
      assert len(new_rank) <= 2
      new_mesh_axes[new_rank[-1]] = mesh_axes[orig_rank]
    mesh_axes = new_mesh_axes

  if mesh_axes is not None and len(mesh_axes) > 0:
    for no_shard_idx in no_sharding_axis:
      if no_shard_idx < len(mesh_axes):
        mesh_axes[no_shard_idx] = None

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()


@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN
  replicate_scale: bool = False

  def _get_rhs_axis_metadata_wrapper(
      self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False, replicate_scale: bool = False
  ):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled, replicate_scale=replicate_scale
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    # module_path = "/".join(nn.module._context.module_stack[-1].path)
    # print(f"quant_dg: {quant_dg}, is_tiled: {is_tiled}, module_path: {module_path}")
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        self.quant_dg,
        lhs_quant_mode=self.lhs_quant_mode,
        rhs_quant_mode=self.rhs_quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
    )
    return aqt_dg_cls

  def einsum(self):
    """Returns einsum configured with aqt params"""
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=self.quant_dg,
            lhs_quant_mode=self.lhs_quant_mode,
            rhs_quant_mode=self.rhs_quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        )
    )
    return aqt_einsum


def _get_quant_config(config):
  if not config.quantization or config.quantization == "":
    return None
  elif config.quantization == "int8":
    if config.quantization_local_shard_count == 0:
      drhs_bits = None
      drhs_accumulator_dtype = None
      drhs_local_aqt = None
    else:
      drhs_bits = 8
      drhs_accumulator_dtype = jnp.int32
      print(config.quantization_local_shard_count)  # -1
      drhs_local_aqt = aqt_config.LocalAqt(contraction_axis_shard_count=config.quantization_local_shard_count)
    return aqt_config.config_v4(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=drhs_bits,
        rng_type="jax.uniform",
        dlhs_local_aqt=None,
        drhs_local_aqt=drhs_local_aqt,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=drhs_accumulator_dtype,
    )


def _get_int8_quant_config(config):
  drhs_bits = None
  drhs_accumulator_dtype = None
  drhs_local_aqt = None
  if config.quantization_local_shard_count != 0:
    drhs_bits = 8
    drhs_accumulator_dtype = jnp.int32
    drhs_local_aqt = aqt_config.LocalAqt(contraction_axis_shard_count=config.quantization_local_shard_count)
  return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type="jax.uniform",
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
  )


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization == "int8":
    return _get_int8_quant_config(config)
  raise ValueError(f"Invalid value configured for quantization {config.quantization}.")


def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)


def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


def get_quant_mode(quant_mode_str: str = "train"):
  """Set quant mode."""
  if quant_mode_str == "train":
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == "serve":
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == "convert":
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f"Invalid quantization mode {quant_mode_str}.")
  return None


def configure_quantization(config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    quant_mode = get_quant_mode(quant_mode_str)
    replicate_scale = config.replicate_quant_scale if config.replicate_quant_scale else False
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode, replicate_scale=replicate_scale)
  return None


def match_aqt_and_unquantized_param(aqt_params, params):
  aqt_param_flat, aqt_tree_def = jax.tree_util.tree_flatten_with_path(
      aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor)
  )
  param_tree_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  aqt_paths = []
  # Original path of quantized AQT param path.
  param_paths = []

  for aqt_k, _ in aqt_param_flat:
    for index, (k, _) in enumerate(param_tree_flat):
      path_depth = len(k)
      # every quantized parameter has AQT.. as the leaf node
      # AqtDotGeneral and AqtEinsum replace leaf node.
      # Therefore, leaf node should be ignored for path matching
      if k[: path_depth - 1] == aqt_k[: path_depth - 1]:
        aqt_paths.append(aqt_k)
        param_paths.append(k)
        break
    # since the parameter is already added, we can delete it.
    param_tree_flat.pop(index)
  return jax.tree_util.tree_unflatten(aqt_tree_def, param_paths)


def _get_aqt_key_paths(aqt_vars, params):
  """Generate a list of paths which have aqt state"""
  aqt_to_unquantized_key_path = match_aqt_and_unquantized_param(aqt_vars, params)
  aqt_key_paths, _ = jax.tree_util.tree_flatten(aqt_to_unquantized_key_path, is_leaf=lambda x: isinstance(x, tuple))
  return list(aqt_key_paths)


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  quantized_param_paths = _get_aqt_key_paths(aqt_vars, params)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in quantized_param_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)
