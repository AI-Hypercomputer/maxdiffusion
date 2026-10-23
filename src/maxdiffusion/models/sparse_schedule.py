"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Step/layer warm-up schedules shared by training-free sparse attention methods
(Sparse VideoGen, Sol-Attn, ...). Most methods keep the first denoising steps
and the first transformer layers dense and sparsify the rest.
"""

from __future__ import annotations

import math
from numbers import Integral

import jax
import jax.numpy as jnp


def is_sparse_active(
    step_index: int | jax.Array | None = None,
    layer_index: int | jax.Array | None = None,
    timestep: int | float | jax.Array | None = None,
    start_step: int = -1,
    end_step: int = -1,
    start_layer: int = -1,
    end_layer: int = -1,
    dense_layer_fraction: float = 0.0,
    dense_timestep_fraction: float = 0.0,
    num_train_timesteps: int = 1000,
    num_layers: int = 40,
    name: str = "sparse attention",
) -> bool | jax.Array:
  """Evaluates whether sparse attention is active for the current step and layer.

  Explicit `[start, end)` intervals take precedence over the fraction-based
  schedules. Static Python ints resolve to a Python bool at trace time (no
  `lax.cond` is emitted); traced indices yield a boolean array.
  """
  has_start_step = start_step >= 0
  has_end_step = end_step >= 0
  if has_start_step != has_end_step:
    raise ValueError(
        f"Incomplete explicit {name} step schedule: start_step={start_step}, end_step={end_step}. "
        "Both bounds must be non-negative or both unset (< 0)."
    )
  has_explicit_step = has_start_step and has_end_step

  has_start_layer = start_layer >= 0
  has_end_layer = end_layer >= 0
  if has_start_layer != has_end_layer:
    raise ValueError(
        f"Incomplete explicit {name} layer schedule: start_layer={start_layer}, end_layer={end_layer}. "
        "Both bounds must be non-negative or both unset (< 0)."
    )
  has_explicit_layer = has_start_layer and has_end_layer

  if has_explicit_step and step_index is None:
    raise ValueError(f"Explicit {name} step schedule requires step_index.")
  if has_explicit_layer and layer_index is None:
    raise ValueError(f"Explicit {name} layer schedule requires layer_index.")

  # 1. Evaluate layer interval / fraction
  if has_explicit_layer:
    if isinstance(layer_index, Integral):
      layer_active: bool | jax.Array = bool(start_layer <= layer_index < end_layer)
    else:
      layer_arr = jnp.asarray(layer_index)
      layer_active = jnp.logical_and(layer_arr >= start_layer, layer_arr < end_layer)
  else:
    dense_layer_count = math.ceil(dense_layer_fraction * num_layers)
    if dense_layer_count > 0 and layer_index is not None:
      if isinstance(layer_index, Integral):
        layer_active = bool(layer_index >= dense_layer_count)
      else:
        layer_active = jnp.asarray(layer_index) >= dense_layer_count
    else:
      layer_active = True

  if isinstance(layer_active, bool) and not layer_active:
    return False

  # 2. Evaluate step interval / fraction
  if has_explicit_step:
    if isinstance(step_index, Integral):
      step_active: bool | jax.Array = bool(start_step <= step_index < end_step)
    else:
      step_arr = jnp.asarray(step_index)
      step_active = jnp.logical_and(step_arr >= start_step, step_arr < end_step)
  else:
    if dense_timestep_fraction > 0.0 and timestep is not None:
      first_sparse_timestep = (1.0 - dense_timestep_fraction) * num_train_timesteps
      if isinstance(timestep, (Integral, float)):
        step_active = bool(timestep < first_sparse_timestep)
      else:
        step_active = jnp.max(jnp.asarray(timestep)) < first_sparse_timestep
    else:
      step_active = True

  if isinstance(step_active, bool) and not step_active:
    return False

  if isinstance(layer_active, bool) and isinstance(step_active, bool):
    return bool(layer_active and step_active)
  if isinstance(layer_active, bool):
    return step_active
  if isinstance(step_active, bool):
    return layer_active
  return jnp.logical_and(step_active, layer_active)
