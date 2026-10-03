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

Sparse VideoGen (SVG) per-head routing, layout placement, and scheduling helpers.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

# Attention-level SVG keys and their defaults. `svg_num_layers` is model
# specific and supplied by the caller. Pipeline-level keys such as
# `svg_high_noise_density` / `svg_low_noise_density` (Wan expert selection)
# are resolved into `svg_spatial_density` before reaching attention.
SVG_ATTENTION_DEFAULTS = {
    "use_svg_attention": False,
    "svg_spatial_density": 0.25,
    "svg_sample_max_row": 10000,
    "svg_profile_query_count": 64,
    "svg_profile_seed": 0,
    "svg_dense_layer_fraction": 0.0,
    "svg_dense_timestep_fraction": 0.0,
    "svg_active_start_step": -1,
    "svg_active_end_step": -1,
    "svg_active_start_layer": -1,
    "svg_active_end_layer": -1,
    "svg_num_train_timesteps": 1000,
    "svg_include_first_frame": True,
    "svg_global_stride": 0,
    "svg_flash_block_sizes": None,
}


def init_svg_config(attention_config: Optional[Mapping[str, Any]], default_num_layers: int) -> dict[str, Any]:
  """Resolves the attention-level SVG settings from `attention_config`.

  Returns a dict keyed by the names in `SVG_ATTENTION_DEFAULTS` plus
  `svg_num_layers`; other keys are ignored. Settings that configs accept but
  the head-local SVG kernel does not implement fail loudly instead of being
  silently dropped.
  """
  attention_config = attention_config or {}
  implementation = attention_config.get("svg_implementation", "official_svg")
  if implementation != "official_svg":
    raise ValueError(f"Unsupported svg_implementation={implementation!r}; only 'official_svg' is implemented.")
  global_offset = attention_config.get("svg_global_offset", 0)
  if global_offset:
    raise ValueError(f"svg_global_offset={global_offset} is not supported by head-local SVG.")
  resolved = {**SVG_ATTENTION_DEFAULTS, "svg_num_layers": default_num_layers}
  for name in resolved:
    if name in attention_config:
      resolved[name] = attention_config[name]
  return resolved


def apply_svg_or_dense(
    svg: Any,
    run_attention: Callable[..., jax.Array],
    spatiotemporal_shape: Tuple[int, int, int],
    svg_layer_index: int | jax.Array | None,
    svg_step_index: int | jax.Array | None,
    svg_timestep: int | float | jax.Array | None,
    named_scope: Callable[[str], Any] = jax.named_scope,
) -> jax.Array:
  """Runs SVG sparse attention on active steps/layers and dense attention otherwise.

  Args:
    svg: Object exposing the attributes resolved by `init_svg_config`
      (the attention module itself).
    run_attention: Calls the attention op. Invoked with no arguments for the
      dense path, and with `spatiotemporal_shape` and `sparse_config_override`
      keyword arguments for the sparse path.
    spatiotemporal_shape: Latent token grid `(frames, height, width)`.
    svg_layer_index: Transformer layer index (static int or traced array).
    svg_step_index: Denoising step index (static int or traced array).
    svg_timestep: Diffusion timestep, used by fraction-based schedules.
    named_scope: Context manager factory used to label the dispatch.
  """
  is_active = is_svg_active(
      step_index=svg_step_index,
      layer_index=svg_layer_index,
      timestep=svg_timestep,
      start_step=svg.svg_active_start_step,
      end_step=svg.svg_active_end_step,
      start_layer=svg.svg_active_start_layer,
      end_layer=svg.svg_active_end_layer,
      dense_layer_fraction=svg.svg_dense_layer_fraction,
      dense_timestep_fraction=svg.svg_dense_timestep_fraction,
      num_train_timesteps=svg.svg_num_train_timesteps,
      num_layers=svg.svg_num_layers,
  )

  def run_dense(_):
    return run_attention()

  def run_sparse(_):
    sparse_config = {
        "use_svg_attention": True,
        "mask_type": "svg_spatial",
        "band_width": svg_execution_band_width(spatiotemporal_shape, svg.svg_spatial_density),
        "include_first_frame": svg.svg_include_first_frame,
        "global_stride": svg.svg_global_stride,
        "profile_query_count": svg.svg_profile_query_count,
        "profile_seed": svg.svg_profile_seed,
        "sample_max_row": svg.svg_sample_max_row,
        "custom_flash_block_sizes": svg.svg_flash_block_sizes,
        "svg_step_index": svg_step_index,
        "svg_layer_index": svg_layer_index,
        "svg_timestep": svg_timestep,
    }
    return run_attention(
        spatiotemporal_shape=spatiotemporal_shape,
        sparse_config_override=sparse_config,
    )

  with named_scope("apply_attention"):
    if isinstance(is_active, bool):
      return run_sparse(None) if is_active else run_dense(None)
    return jax.lax.cond(is_active, run_sparse, run_dense, operand=None)


def svg_execution_band_width(token_grid: Tuple[int, int, int], density: float) -> int:
  """Compute SVG symmetric band width from retained density with 128-token block ceiling.

  For sequence length N and retained pair density s:
    w = N * (1.0 - sqrt(1.0 - s))
  rounded up to the nearest multiple of 128.

  Edge cases:
    density == 1.0 -> N - 1 (dense band covering all token pairs)
    density <= 0.0 or density > 1.0 -> raises ValueError
  """
  sequence_length = math.prod(token_grid)
  if sequence_length <= 0:
    raise ValueError(f"SVG sequence length must be positive, got {sequence_length}")
  density = float(density)
  if density <= 0.0 or density > 1.0:
    raise ValueError(f"svg_retained_density must be in (0.0, 1.0], got {density}")
  if density >= 1.0:
    return sequence_length - 1

  width = sequence_length * (1.0 - math.sqrt(1.0 - density))
  band_width = int(math.ceil(width / 128.0)) * 128
  return min(max(band_width, 0), sequence_length - 1)


def svg_probe_masks(
    sampled_rows: jax.Array,
    token_grid: Tuple[int, int, int],
    block_size: int = 128,
) -> Tuple[jax.Array, jax.Array]:
  """Construct fixed spatial and temporal profiler masks for sampled query rows.

  Spatial mask:
    (k < frame_size) | (abs(q // 128 - k // 128) < (2 * frame_size) // 128)
  Temporal mask:
    (k_tm < frame_size) | (abs(q_tm // 128 - k_tm // 128) < (2 * frame_size) // 128)
    where q_tm = (q % frame_size) * frames + (q // frame_size)
          k_tm = (k % frame_size) * frames + (k // frame_size)
  """
  frames, height, width = token_grid
  frame_size = height * width
  sequence_length = frames * frame_size

  key_indices = jnp.arange(sequence_length, dtype=jnp.int32)
  query_indices = sampled_rows.astype(jnp.int32)

  block_thres_blocks = (2 * frame_size) // block_size

  q_blk = query_indices[:, None] // block_size
  k_blk = key_indices[None, :] // block_size

  q_tm = (query_indices[:, None] % frame_size) * frames + (query_indices[:, None] // frame_size)
  k_tm = (key_indices[None, :] % frame_size) * frames + (key_indices[None, :] // frame_size)

  q_tm_blk = q_tm // block_size
  k_tm_blk = k_tm // block_size

  spatial_mask = (key_indices[None, :] < frame_size) | (jnp.abs(q_blk - k_blk) < block_thres_blocks)
  temporal_mask = (k_tm < frame_size) | (jnp.abs(q_tm_blk - k_tm_blk) < block_thres_blocks)

  return spatial_mask, temporal_mask


def svg_profile_temporal_heads(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    token_grid: Tuple[int, int, int],
    query_count: int,
    profile_key: jax.Array,
    scale: float,
    sample_max_row: int = 10000,
) -> jax.Array:
  """Profile each head independently using sampled dense reconstruction MSE.

  `scale` is the ordinary attention logit scale. The profiler intentionally
  uses natural-exp softmax. The production base-2 kernel multiplies Q by
  log2(e) before exp2, which is mathematically equivalent to this natural-exp
  formulation, so no additional LOG2E factor belongs in the profiler.
  """
  with jax.named_scope("svg_route_profile"):
    sequence_length = query.shape[2]
    sample_pool_size = min(max(int(sample_max_row), 1), sequence_length)
    sample_count = min(max(int(query_count), 1), sample_pool_size)

    if sample_count >= sample_pool_size:
      sampled_rows = jnp.arange(sample_count, dtype=jnp.int32)
    else:
      sampled_rows = jax.random.randint(
          profile_key,
          (sample_count,),
          minval=0,
          maxval=sample_pool_size,
          dtype=jnp.int32,
      )

    sampled_query = jnp.take(query, sampled_rows, axis=2)
    spatial_mask, temporal_mask = svg_probe_masks(sampled_rows, token_grid)

    sampled_qk = (
        jnp.einsum(
            "bhqd,bhkd->bhqk",
            sampled_query.astype(jnp.float32),
            key.astype(jnp.float32),
        )
        * scale
    )

    dense_weights = jax.nn.softmax(sampled_qk, axis=-1)
    dense_output = jnp.einsum("bhqk,bhkd->bhqd", dense_weights, value.astype(jnp.float32))

    spatial_logits = jnp.where(spatial_mask[None, None, :, :], sampled_qk, -1e9)
    spatial_weights = jax.nn.softmax(spatial_logits, axis=-1)
    spatial_output = jnp.einsum("bhqk,bhkd->bhqd", spatial_weights, value.astype(jnp.float32))

    temporal_logits = jnp.where(temporal_mask[None, None, :, :], sampled_qk, -1e9)
    temporal_weights = jax.nn.softmax(temporal_logits, axis=-1)
    temporal_output = jnp.einsum("bhqk,bhkd->bhqd", temporal_weights, value.astype(jnp.float32))

    spatial_error = jnp.mean(jnp.square(spatial_output - dense_output), axis=(-2, -1))
    temporal_error = jnp.mean(jnp.square(temporal_output - dense_output), axis=(-2, -1))
    return temporal_error < spatial_error


def svg_token_major_indices(token_grid: Tuple[int, int, int]) -> Tuple[jax.Array, jax.Array]:
  """Return forward (frame-major -> token-major) and inverse permutation index arrays."""
  frames, height, width = token_grid
  frame_size = height * width
  sequence_length = frames * frame_size

  j = jnp.arange(sequence_length, dtype=jnp.int32)
  forward_gather_idx = (j % frames) * frame_size + (j // frames)
  inverse_gather_idx = (j % frame_size) * frames + (j // frame_size)
  return forward_gather_idx, inverse_gather_idx


def svg_placement_permute(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    is_temporal: jax.Array,
    token_grid: Tuple[int, int, int],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Place temporal heads into token-major order, leaving spatial heads unchanged."""
  with jax.named_scope("svg_layout_place"):
    frames, height, width = (int(v) for v in token_grid)
    frame_size = height * width
    b, heads, n, d = query.shape

    q_tm = query.reshape(b, heads, frames, frame_size, d).transpose(0, 1, 3, 2, 4).reshape(b, heads, n, d)
    k_tm = key.reshape(b, heads, frames, frame_size, d).transpose(0, 1, 3, 2, 4).reshape(b, heads, n, d)
    v_tm = value.reshape(b, heads, frames, frame_size, d).transpose(0, 1, 3, 2, 4).reshape(b, heads, n, d)

    cond = is_temporal[:, :, None, None]
    q_out = jnp.where(cond, q_tm, query)
    k_out = jnp.where(cond, k_tm, key)
    v_out = jnp.where(cond, v_tm, value)
    return q_out, k_out, v_out


def svg_placement_unpermute(
    output: jax.Array,
    is_temporal: jax.Array,
    token_grid: Tuple[int, int, int],
) -> jax.Array:
  """Restore temporal-head output from token-major to frame-major order."""
  with jax.named_scope("svg_layout_restore"):
    frames, height, width = (int(v) for v in token_grid)
    frame_size = height * width
    b, heads, n, d = output.shape
    out_fm = output.reshape(b, heads, frame_size, frames, d).transpose(0, 1, 3, 2, 4).reshape(b, heads, n, d)
    return jnp.where(is_temporal[:, :, None, None], out_fm, output)


def is_svg_active(
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
) -> bool | jax.Array:
  """Evaluate whether SVG attention is active for the current step and layer."""
  has_start_step = start_step >= 0
  has_end_step = end_step >= 0
  if has_start_step != has_end_step:
    raise ValueError(
        f"Incomplete explicit SVG step schedule: start_step={start_step}, end_step={end_step}. "
        "Both bounds must be non-negative or both unset (< 0)."
    )
  has_explicit_step = has_start_step and has_end_step

  has_start_layer = start_layer >= 0
  has_end_layer = end_layer >= 0
  if has_start_layer != has_end_layer:
    raise ValueError(
        f"Incomplete explicit SVG layer schedule: start_layer={start_layer}, end_layer={end_layer}. "
        "Both bounds must be non-negative or both unset (< 0)."
    )
  has_explicit_layer = has_start_layer and has_end_layer

  if has_explicit_step and step_index is None:
    raise ValueError("Explicit SVG step schedule requires step_index.")
  if has_explicit_layer and layer_index is None:
    raise ValueError("Explicit SVG layer schedule requires layer_index.")

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


def place_sequence_for_mask(
    tensor: jax.Array,
    is_temporal: jax.Array,
    token_grid: Tuple[int, int, int],
) -> jax.Array:
  """Place a single tensor (query, key, or value) into per-head layout."""
  frames, height, width = (int(v) for v in token_grid)
  frame_size = height * width
  b, heads, n, d = tensor.shape
  t_tm = tensor.reshape(b, heads, frames, frame_size, d).transpose(0, 1, 3, 2, 4).reshape(b, heads, n, d)
  return jnp.where(is_temporal[:, :, None, None], t_tm, tensor)


def unplace_sequence_for_mask(
    tensor: jax.Array,
    is_temporal: jax.Array,
    token_grid: Tuple[int, int, int],
) -> jax.Array:
  """Unplace a single tensor from per-head layout back to frame-major."""
  return svg_placement_unpermute(tensor, is_temporal, token_grid)


class SVGHeadRouting:

  def __init__(self, is_temporal: jax.Array):
    self.is_temporal = is_temporal


def route_heads(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    token_grid: Tuple[int, int, int],
    spatial_density: float = 0.25,
    temporal_density: float = 0.25,
    sample_query_count: int = 64,
    sample_max_row: int = 10000,
    profile_seed: int = 0,
    scale: float = 1.0,
    include_first_frame: bool = True,
    global_stride: int = 0,
    global_offset: int = 0,
    layer_index: int | jax.Array | None = None,
    step_index: int | jax.Array | None = None,
    timestep: int | float | jax.Array | None = None,
) -> SVGHeadRouting:
  """Profile and route heads between spatial and temporal attention layouts."""
  del spatial_density, temporal_density, include_first_frame, global_stride, global_offset
  profile_key = jax.random.PRNGKey(profile_seed)
  if layer_index is not None:
    profile_key = jax.random.fold_in(profile_key, jnp.asarray(layer_index, dtype=jnp.uint32))
  if step_index is not None:
    profile_key = jax.random.fold_in(profile_key, jnp.asarray(step_index, dtype=jnp.uint32))
  elif timestep is not None:
    profile_key = jax.random.fold_in(profile_key, jnp.max(jnp.asarray(timestep)).astype(jnp.uint32))

  is_temporal = svg_profile_temporal_heads(
      query,
      key,
      value,
      token_grid,
      sample_query_count,
      profile_key,
      scale,
      sample_max_row=sample_max_row,
  )
  return SVGHeadRouting(is_temporal=is_temporal)
