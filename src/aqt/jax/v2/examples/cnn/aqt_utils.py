# Copyright 2024 Google LLC
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

"""Utility functions to quantize models using AQT.

NOTE: All functions in this file assume lhs is an activation.
"""

import functools
from typing import Any

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import utils
from aqt.jax.v2.examples.cnn import model_utils
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn
from flax import typing as flax_typing
import jax
import jax.numpy as jnp

ModuleDef = Any


#########################################
# Common utility functions
#########################################
def _get_aqt_injected_model_kwargs(
    aqt_cfg: aqt_config.DotGeneral,
    lhs_quant_mode: utils.QuantMode,
    rhs_quant_mode: utils.QuantMode,
) -> dict[str, Any]:
  """Injects AQT functions to a given model class."""

  def _get_aqt_kwargs(
      aqt_cfg: aqt_config.DotGeneral,
      lhs_quant_mode: utils.QuantMode,
      rhs_quant_mode: utils.QuantMode,
  ) -> dict[str, Any]:
    """Gets AQT class init func prefilled with given configurations."""
    return dict(
        cfg=aqt_cfg,
        lhs_quant_mode=lhs_quant_mode,
        rhs_quant_mode=rhs_quant_mode,
        # For activations, we never freeze quantized values.
        lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
    )

  return dict(
      dot_general_cls=functools.partial(
          aqt_flax.AqtDotGeneral,
          **(_get_aqt_kwargs(aqt_cfg, lhs_quant_mode, rhs_quant_mode)),
      ),
      einsum_cls=functools.partial(
          aqt_flax.AqtEinsum,
          **(_get_aqt_kwargs(aqt_cfg, lhs_quant_mode, rhs_quant_mode)),
      ),
  )


#########################################
# Serving utility functions
#########################################
def _get_quantized_vars(
    model_cls: ModuleDef,
    aqt_cfg: aqt_config.DotGeneral,
    model_vars: flax_typing.FrozenVariableDict,
    act_calibrated: bool,
) -> flax_typing.FrozenVariableDict:
  """Gets quantized model variables."""
  # If activations have been calibrated, use the CONVERT mode to compute and
  # freeze scales from the statistics collected during the calibration pass.
  lhs_quant_mode = (
      utils.QuantMode.CONVERT if act_calibrated else utils.QuantMode.TRAIN
  )
  cnn_freeze = model_cls(
      bn_use_stats=False,
      **(
          _get_aqt_injected_model_kwargs(
              aqt_cfg, lhs_quant_mode, utils.QuantMode.CONVERT
          )
      ),
  )
  input_shape = (1, 28, 28, 1)
  _, quantized_param_vars = cnn_freeze.apply(
      model_vars,
      jnp.ones(input_shape),
      rngs={'params': jax.random.PRNGKey(0)},
      mutable=True,
  )
  return quantized_param_vars


def _get_serving_model(
    model_cls: ModuleDef, aqt_cfg: aqt_config.DotGeneral, act_calibrated: bool
) -> nn.Module:
  """Gets a Flax module to serve an AQT quantized model."""
  # If activations have been calibrated, use the SERVE mode to use the scales
  # frozen and stored in the model variables.
  lhs_quant_mode = (
      utils.QuantMode.SERVE if act_calibrated else utils.QuantMode.TRAIN
  )
  cnn_serve = model_cls(
      bn_use_stats=False,
      **(
          _get_aqt_injected_model_kwargs(
              aqt_cfg, lhs_quant_mode, utils.QuantMode.SERVE
          )
      ),
  )
  return cnn_serve


def serve_quantized(
    model_cls: ModuleDef,
    test_ds: dict[str, jax.Array],
    aqt_cfg: aqt_config.DotGeneral,
    model_vars: flax_typing.FrozenVariableDict,
    act_calibrated: bool,
) -> jax.Array:
  """Quantizes and tests a model with a given test dataset."""
  serve_model = _get_serving_model(
      model_cls, aqt_cfg, act_calibrated=act_calibrated
  )
  model_vars = _get_quantized_vars(
      model_cls, aqt_cfg, model_vars, act_calibrated=act_calibrated
  )
  return model_utils.serve(serve_model, model_vars, test_ds)


#########################################
# Calibration utility functions
#########################################
def _get_calibration_model(
    model_cls: ModuleDef, aqt_cfg: aqt_config.DotGeneral
) -> nn.Module:
  cnn_calibrate = model_cls(
      bn_use_stats=False,
      **(
          _get_aqt_injected_model_kwargs(
              aqt_cfg, utils.QuantMode.CALIBRATE, utils.QuantMode.CALIBRATE
          )
      ),
  )
  return cnn_calibrate


def _calibrate_epoch(
    calibration_model: nn.Module,
    model_vars: flax_typing.FrozenVariableDict,
    calibrate_ds: dict[str, jax.Array],
    batch_size: int,
    rng: jax.Array,
    calibration_steps: int,
) -> flax_typing.FrozenVariableDict:
  """Calibrates for a single epoch."""
  perms = model_utils.prepare_data_perm(
      calibrate_ds, batch_size, rng, calibration_steps
  )
  apply_fn = jax.jit(calibration_model.apply, static_argnames='mutable')
  for perm in perms:
    batch_images = calibrate_ds['image'][perm, ...]
    # Calibration simply updates model during inference. No need to compute loss
    # or gradients.
    _, model_vars = apply_fn(
        model_vars,
        batch_images,
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
  return model_vars


def calibrate(
    model_cls: ModuleDef,
    aqt_cfg: aqt_config.DotGeneral,
    state: model_utils.TrainState,
    calibration_steps: int,
    calibrate_ds: dict[str, jax.Array],
) -> model_utils.TrainState:
  """Calibrates a given model."""
  rng = jax.random.key(0)
  batch_size = 128
  calibration_model = _get_calibration_model(model_cls, aqt_cfg)
  calibrated_vars = _calibrate_epoch(
      calibration_model,
      state.model_vars,
      calibrate_ds,
      batch_size,
      rng,
      calibration_steps,
  )

  return state.replace(model_vars=calibrated_vars)
