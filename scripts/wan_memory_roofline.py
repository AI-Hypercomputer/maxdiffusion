#!/usr/bin/env python3
"""
Shape-only WAN memory report with sharding-aware per-device estimates.

This script uses abstract evaluation on CPU to trace:
- WAN Animate transformer forward
- WAN VAE encode
- WAN VAE decode

It reports, per module:
- input/output shapes
- dtypes
- logical tensor bytes
- estimated per-device bytes from intended sharding
- own parameter bytes

It also writes a static HTML report you can open in a browser.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import html
import inspect
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx
from flax.nnx import graphlib, variablelib
from jax.sharding import Mesh

from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.models.wan.transformers.transformer_wan_animate import NNXWanAnimateTransformer3DModel

DEFAULT_TRANSFORMER_AXIS_RULES = (
    ("batch", "data"),
    ("activation_batch", "data"),
    ("activation_self_attn_heads", ("context", "tensor")),
    ("activation_self_attn_q_length", "context"),
    ("activation_self_attn_kv_length", None),
    ("activation_cross_attn_q_length", "context"),
    ("activation_cross_attn_heads", "tensor"),
    ("activation_cross_attn_kv_length", None),
    ("activation_length", "context"),
    ("activation_heads", "tensor"),
    ("activation_kv", "tensor"),
    ("activation_kv_length", "context"),
    ("mlp", "tensor"),
    ("embed", "fsdp"),
    ("heads", "tensor"),
    ("norm", "tensor"),
    ("conv_batch", ("data", "context")),
    ("out_channels", "tensor"),
    ("conv_out", "context"),
    ("layers_per_stage", None),
)

DEFAULT_VAE_AXIS_RULES = (
    ("activation_batch", "redundant"),
    ("activation_length", None),
    ("conv_batch", "redundant"),
    ("conv_in", None),
    ("out_channels", "vae_spatial"),
    ("conv_out", "vae_spatial"),
    ("embed", None),
    ("heads", None),
    ("norm", None),
)


@dataclasses.dataclass
class ArrayStat:
  path: str
  shape: tuple[int, ...]
  dtype: str
  numel: int
  bytes: int
  per_device_bytes: int
  sharding: str
  shard_factor: int


@dataclasses.dataclass
class ParamStats:
  own_by_dtype: dict[str, int]
  own_per_device_by_dtype: dict[str, int]
  subtree_by_dtype: dict[str, int]
  subtree_per_device_by_dtype: dict[str, int]
  own_shardings: list[str]


@dataclasses.dataclass
class CallRecord:
  trace_id: int
  path: tuple[Any, ...]
  module_type: str
  method: str
  input_arrays: list[ArrayStat]
  output_arrays: list[ArrayStat]


def _dtype_name(dtype: Any) -> str:
  return str(jnp.dtype(dtype))


def _short_dtype_name(dtype: str) -> str:
  mapping = {
      "bfloat16": "bf16",
      "float32": "fp32",
      "float16": "fp16",
      "int32": "i32",
      "int64": "i64",
      "uint8": "u8",
      "bool": "bool",
  }
  return mapping.get(dtype, dtype)


def _numel(shape: Iterable[int]) -> int:
  out = 1
  for dim in shape:
    out *= int(dim)
  return out


def _human_bytes(num_bytes: int) -> str:
  if num_bytes == 0:
    return "0 B"
  value = float(num_bytes)
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if value < 1024.0 or unit == "TiB":
      return f"{value:.2f} {unit}"
    value /= 1024.0
  return f"{num_bytes} B"


def _lists_to_tuples(obj: Any) -> Any:
  if isinstance(obj, list):
    return tuple(_lists_to_tuples(x) for x in obj)
  if isinstance(obj, dict):
    return {k: _lists_to_tuples(v) for k, v in obj.items()}
  if obj == "None":
    return None
  return obj


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
  with config_path.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  raw = _lists_to_tuples(raw)
  raw["weights_dtype"] = jnp.dtype(raw["weights_dtype"])
  raw["activations_dtype"] = jnp.dtype(raw["activations_dtype"])
  return raw


def _coerce_override(raw_value: str, current_value: Any) -> Any:
  if raw_value == "None":
    return None
  if isinstance(current_value, bool):
    return raw_value.lower() == "true"
  if isinstance(current_value, int) and not isinstance(current_value, bool):
    return int(raw_value)
  if isinstance(current_value, float):
    return float(raw_value)
  if isinstance(current_value, tuple):
    return _lists_to_tuples(yaml.safe_load(raw_value))
  return raw_value


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
  updated = dict(config)
  for override in overrides:
    if "=" not in override:
      raise ValueError(f"Expected key=value override, got {override!r}")
    key, raw_value = override.split("=", 1)
    if key not in updated:
      raise ValueError(f"Unknown config override {key!r}")
    updated[key] = _coerce_override(raw_value, updated[key])
  return updated


def _maybe_load_local_subconfig(model_cls: type, model_id: str, subfolder: str) -> dict[str, Any]:
  try:
    return dict(model_cls.load_config(model_id, subfolder=subfolder, local_files_only=True))
  except Exception:
    return {}


def _transformer_axis_sizes(config: dict[str, Any]) -> dict[str, int]:
  axis_sizes: dict[str, int] = {}
  for axis in config.get("mesh_axes", ("data", "fsdp", "context", "tensor")):
    ici = int(config.get(f"ici_{axis}_parallelism", 1))
    dcn = int(config.get(f"dcn_{axis}_parallelism", 1))
    if ici <= 0:
      ici = 1
    if dcn <= 0:
      dcn = 1
    axis_sizes[axis] = ici * dcn
  return axis_sizes


def _vae_axis_sizes(config: dict[str, Any]) -> dict[str, int]:
  transformer_axis_sizes = _transformer_axis_sizes(config)
  total_devices = max(1, math.prod(transformer_axis_sizes.values()))
  vae_spatial = int(config.get("vae_spatial", -1))
  if vae_spatial <= 0:
    vae_spatial = total_devices
  return {
      "redundant": max(1, total_devices // vae_spatial),
      "vae_spatial": max(1, vae_spatial),
  }


def _build_transformer_mesh() -> Mesh:
  return Mesh(np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


def _build_vae_mesh() -> Mesh:
  return Mesh(np.array(jax.devices()).reshape(1, 1), ("redundant", "vae_spatial"))


def _collect_sharding_axes(spec_part: Any) -> list[str]:
  if spec_part is None:
    return []
  if isinstance(spec_part, str):
    return [spec_part]
  if isinstance(spec_part, tuple):
    out: list[str] = []
    for item in spec_part:
      out.extend(_collect_sharding_axes(item))
    return out
  return []


def _describe_sharding(sharding: Any, axis_sizes: dict[str, int]) -> tuple[str, int]:
  spec = getattr(sharding, "spec", None)
  if spec is None:
    return "replicated", 1
  axes: list[str] = []
  for part in tuple(spec):
    axes.extend(_collect_sharding_axes(part))
  unique_axes: list[str] = []
  for axis in axes:
    if axis not in unique_axes:
      unique_axes.append(axis)
  if not unique_axes:
    return "replicated", 1
  shard_factor = 1
  for axis in unique_axes:
    shard_factor *= max(1, int(axis_sizes.get(axis, 1)))
  return "+".join(unique_axes), max(1, shard_factor)


def _array_stat_from_leaf(path: str, leaf: Any, axis_sizes: dict[str, int]) -> ArrayStat | None:
  if isinstance(leaf, jax.ShapeDtypeStruct):
    shape = tuple(int(x) for x in leaf.shape)
    dtype = jnp.dtype(leaf.dtype)
    sharding_obj = getattr(leaf, "sharding", None)
  elif isinstance(leaf, (jax.Array, np.ndarray)):
    shape = tuple(int(x) for x in leaf.shape)
    dtype = jnp.dtype(leaf.dtype)
    sharding_obj = getattr(leaf, "sharding", None)
  else:
    return None

  logical_bytes = _numel(shape) * dtype.itemsize
  sharding, shard_factor = _describe_sharding(sharding_obj, axis_sizes)
  local_bytes = math.ceil(logical_bytes / shard_factor)
  return ArrayStat(
      path=path,
      shape=shape,
      dtype=_dtype_name(dtype),
      numel=_numel(shape),
      bytes=logical_bytes,
      per_device_bytes=local_bytes,
      sharding=sharding,
      shard_factor=shard_factor,
  )


def _collect_array_stats(tree: Any, axis_sizes: dict[str, int], prefix: str = "") -> list[ArrayStat]:
  results: list[ArrayStat] = []

  def _walk(node: Any, path: str):
    stat = _array_stat_from_leaf(path or "value", node, axis_sizes)
    if stat is not None:
      results.append(stat)
      return
    if isinstance(node, dict):
      for key, value in node.items():
        _walk(value, f"{path}.{key}" if path else str(key))
      return
    if isinstance(node, (tuple, list)):
      for idx, value in enumerate(node):
        _walk(value, f"{path}[{idx}]" if path else f"[{idx}]")

  _walk(tree, prefix)
  return results


def _sum_by_dtype(stats: Iterable[ArrayStat], attr: str) -> dict[str, int]:
  totals: dict[str, int] = collections.defaultdict(int)
  for stat in stats:
    totals[stat.dtype] += int(getattr(stat, attr))
  return dict(sorted(totals.items()))


def _unique_in_order(values: Iterable[str]) -> list[str]:
  out: list[str] = []
  for value in values:
    if value not in out:
      out.append(value)
  return out


def _format_dtype_bytes(dtype_bytes: dict[str, int]) -> str:
  if not dtype_bytes:
    return "-"
  return ", ".join(f"{_short_dtype_name(dtype)}={_human_bytes(num_bytes)}" for dtype, num_bytes in dtype_bytes.items())


def _format_array_dtypes(arrays: list[dict[str, Any]] | list[ArrayStat]) -> str:
  values = [arr["dtype"] if isinstance(arr, dict) else arr.dtype for arr in arrays]
  return ", ".join(_short_dtype_name(v) for v in _unique_in_order(values)) or "-"


def _format_array_shardings(arrays: list[dict[str, Any]] | list[ArrayStat]) -> str:
  values = [arr["sharding"] if isinstance(arr, dict) else arr.sharding for arr in arrays]
  return ", ".join(_unique_in_order(values)) or "-"


def _format_shapes(arrays: list[dict[str, Any]] | list[ArrayStat], limit: int = 3) -> str:
  formatted: list[str] = []
  for arr in arrays[:limit]:
    shape = arr["shape"] if isinstance(arr, dict) else list(arr.shape)
    formatted.append("x".join(str(x) for x in shape))
  if len(arrays) > limit:
    formatted.append(f"+{len(arrays) - limit} more")
  return ", ".join(formatted) if formatted else "-"


def _path_to_string(path: tuple[Any, ...]) -> str:
  return "<root>" if not path else "/".join(str(p) for p in path)


def _cast_leaf_dtype(x: Any, dtype_to_cast: jnp.dtype) -> Any:
  if isinstance(x, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(x.shape, dtype_to_cast, sharding=getattr(x, "sharding", None))
  if hasattr(x, "astype"):
    return x.astype(dtype_to_cast)
  raise TypeError(f"Unsupported leaf type {type(x)!r}")


def _cast_with_exclusion_shape(path: tuple[Any, ...], x: Any, dtype_to_cast: jnp.dtype) -> Any:
  exclusion_keywords = ("norm", "condition_embedder", "scale_shift_table")
  path_str = ".".join(str(part) for part in path)
  target_dtype = jnp.float32 if any(keyword in path_str.lower() for keyword in exclusion_keywords) else dtype_to_cast
  return _cast_leaf_dtype(x, target_dtype)


def _mark_trace_ids(root: nnx.Module) -> tuple[dict[int, tuple[Any, ...]], dict[int, nnx.Module]]:
  path_by_trace_id: dict[int, tuple[Any, ...]] = {}
  module_by_trace_id: dict[int, nnx.Module] = {}
  for path, module in nnx.iter_modules(root):
    trace_id = id(module)
    setattr(module, "_wan_trace_id", trace_id)
    path_by_trace_id[trace_id] = path
    module_by_trace_id[trace_id] = module
  return path_by_trace_id, module_by_trace_id


def _cast_param_state_for_transformer(model: nnx.Module, weights_dtype: jnp.dtype) -> nnx.Module:
  graphdef, state, rest = nnx.split(model, nnx.Param, ...)
  flat_state = dict(nnx.to_flat_state(state))
  for path, variable_state in flat_state.items():
    variable_state.value = _cast_with_exclusion_shape(path, variable_state.value, dtype_to_cast=weights_dtype)
  return nnx.merge(graphdef, nnx.from_flat_state(flat_state), rest)


def _direct_param_array_stats(module: nnx.Module, axis_sizes: dict[str, int]) -> list[ArrayStat]:
  stats: list[ArrayStat] = []
  node_impl = graphlib.get_node_impl(module)
  assert node_impl is not None
  for name, value in node_impl.node_dict(module).items():
    if isinstance(value, variablelib.Variable):
      stat = _array_stat_from_leaf(str(name), value.value, axis_sizes)
      if stat is not None:
        stats.append(stat)
  return stats


def _subtree_param_array_stats(module: nnx.Module, axis_sizes: dict[str, int]) -> list[ArrayStat]:
  stats: list[ArrayStat] = []
  state = nnx.state(module, nnx.Param)
  for path, variable_state in nnx.to_flat_state(state):
    stat = _array_stat_from_leaf(".".join(str(p) for p in path), variable_state.value, axis_sizes)
    if stat is not None:
      stats.append(stat)
  return stats


def _collect_param_stats(root: nnx.Module, axis_sizes: dict[str, int]) -> dict[int, ParamStats]:
  out: dict[int, ParamStats] = {}
  for _, module in nnx.iter_modules(root):
    own_stats = _direct_param_array_stats(module, axis_sizes)
    subtree_stats = _subtree_param_array_stats(module, axis_sizes)
    out[id(module)] = ParamStats(
        own_by_dtype=_sum_by_dtype(own_stats, "bytes"),
        own_per_device_by_dtype=_sum_by_dtype(own_stats, "per_device_bytes"),
        subtree_by_dtype=_sum_by_dtype(subtree_stats, "bytes"),
        subtree_per_device_by_dtype=_sum_by_dtype(subtree_stats, "per_device_bytes"),
        own_shardings=_unique_in_order(stat.sharding for stat in own_stats),
    )
  return out


def _trace_module_calls(
    root: nnx.Module,
    method: str,
    method_args: tuple[Any, ...],
    method_kwargs: dict[str, Any],
    mesh: Mesh,
    axis_rules: tuple[Any, ...],
    axis_sizes: dict[str, int],
) -> list[CallRecord]:
  path_by_trace_id, _ = _mark_trace_ids(root)
  object_types = {type(module) for _, module in nnx.iter_modules(root)}
  originals: list[tuple[type, str, Any]] = []
  records: list[CallRecord] = []
  seen: set[tuple[int, str, tuple[tuple[str, tuple[int, ...], str], ...]]] = set()

  def _wrap(fn, method_name: str):
    def wrapper(obj, *args, **kwargs):
      trace_id = getattr(obj, "_wan_trace_id", None)
      if trace_id is None or trace_id not in path_by_trace_id:
        return fn(obj, *args, **kwargs)
      input_arrays = _collect_array_stats({"args": args, "kwargs": kwargs}, axis_sizes)
      signature = tuple((stat.path, stat.shape, stat.dtype) for stat in input_arrays)
      key = (trace_id, method_name, signature)
      output = fn(obj, *args, **kwargs)
      if key not in seen:
        seen.add(key)
        records.append(
            CallRecord(
                trace_id=trace_id,
                path=path_by_trace_id[trace_id],
                module_type=type(obj).__name__,
                method=method_name,
                input_arrays=input_arrays,
                output_arrays=_collect_array_stats(output, axis_sizes),
            )
        )
      return output

    return wrapper

  for obj_type in object_types:
    if hasattr(obj_type, "__call__") and inspect.isfunction(obj_type.__call__):
      originals.append((obj_type, "__call__", obj_type.__call__))
      setattr(obj_type, "__call__", _wrap(obj_type.__call__, "__call__"))

  if method != "__call__":
    root_type = type(root)
    if hasattr(root_type, method) and inspect.isfunction(getattr(root_type, method)):
      originals.append((root_type, method, getattr(root_type, method)))
      setattr(root_type, method, _wrap(getattr(root_type, method), method))

  try:
    with mesh, nn.partitioning.axis_rules(axis_rules):
      nnx.eval_shape(lambda mdl, *run_args: getattr(mdl, method)(*run_args, **method_kwargs), root, *method_args)
  finally:
    for obj_type, method_name, original in reversed(originals):
      setattr(obj_type, method_name, original)

  records.sort(key=lambda r: (len(r.path), _path_to_string(r.path), r.method))
  return records


def _shape_to_list(shape: tuple[int, ...]) -> list[int]:
  return [int(x) for x in shape]


def _array_stats_to_json(stats: list[ArrayStat]) -> list[dict[str, Any]]:
  return [
      {
          "path": stat.path,
          "shape": _shape_to_list(stat.shape),
          "dtype": stat.dtype,
          "bytes": stat.bytes,
          "per_device_bytes": stat.per_device_bytes,
          "sharding": stat.sharding,
          "shard_factor": stat.shard_factor,
      }
      for stat in stats
  ]


def _record_to_json(record: CallRecord, param_stats: ParamStats) -> dict[str, Any]:
  input_logical = _sum_by_dtype(record.input_arrays, "bytes")
  input_local = _sum_by_dtype(record.input_arrays, "per_device_bytes")
  output_logical = _sum_by_dtype(record.output_arrays, "bytes")
  output_local = _sum_by_dtype(record.output_arrays, "per_device_bytes")
  logical_act_peak = max(sum(input_logical.values()), sum(output_logical.values()))
  local_act_peak = max(sum(input_local.values()), sum(output_local.values()))
  logical_peak_estimate = sum(param_stats.subtree_by_dtype.values()) + logical_act_peak
  local_peak_estimate = sum(param_stats.subtree_per_device_by_dtype.values()) + local_act_peak

  return {
      "path": _path_to_string(record.path),
      "method": record.method,
      "module_type": record.module_type,
      "params": {
          "own_by_dtype": param_stats.own_by_dtype,
          "own_per_device_by_dtype": param_stats.own_per_device_by_dtype,
          "subtree_by_dtype": param_stats.subtree_by_dtype,
          "subtree_per_device_by_dtype": param_stats.subtree_per_device_by_dtype,
          "own_total_bytes": sum(param_stats.own_by_dtype.values()),
          "own_per_device_total_bytes": sum(param_stats.own_per_device_by_dtype.values()),
          "subtree_total_bytes": sum(param_stats.subtree_by_dtype.values()),
          "subtree_per_device_total_bytes": sum(param_stats.subtree_per_device_by_dtype.values()),
          "own_shardings": param_stats.own_shardings,
      },
      "activations": {
          "inputs": _array_stats_to_json(record.input_arrays),
          "outputs": _array_stats_to_json(record.output_arrays),
          "input_by_dtype": input_logical,
          "input_per_device_by_dtype": input_local,
          "output_by_dtype": output_logical,
          "output_per_device_by_dtype": output_local,
          "input_total_bytes": sum(input_logical.values()),
          "input_per_device_total_bytes": sum(input_local.values()),
          "output_total_bytes": sum(output_logical.values()),
          "output_per_device_total_bytes": sum(output_local.values()),
          "activation_peak_bytes": logical_act_peak,
          "activation_peak_per_device_bytes": local_act_peak,
      },
      "peak_estimate_bytes": logical_peak_estimate,
      "peak_estimate_per_device_bytes": local_peak_estimate,
  }


def _make_transformer_inputs(config: dict[str, Any], model: NNXWanAnimateTransformer3DModel, batch_size: int) -> tuple[Any, ...]:
  segment_frames = int(config.get("segment_frame_length", 77))
  height = int(config.get("height", 720))
  width = int(config.get("width", 1280))
  latent_frames = (segment_frames - 1) // 4 + 1
  latent_h = height // 8
  latent_w = width // 8
  max_sequence_length = int(config.get("max_sequence_length", 512))
  image_seq_len = int(config.get("image_seq_len", 257))
  face_size = int(model.motion_encoder.size)
  dtype = config["activations_dtype"]

  return (
      jax.ShapeDtypeStruct((batch_size, model.config.in_channels, latent_frames + 1, latent_h, latent_w), dtype),
      jax.ShapeDtypeStruct((batch_size,), jnp.int32),
      jax.ShapeDtypeStruct((batch_size, max_sequence_length, model.config.text_dim), dtype),
      jax.ShapeDtypeStruct((batch_size, image_seq_len, model.config.image_dim), dtype),
      jax.ShapeDtypeStruct((batch_size, model.config.latent_channels, latent_frames, latent_h, latent_w), dtype),
      jax.ShapeDtypeStruct((batch_size, 3, segment_frames, face_size, face_size), dtype),
  )


def _make_vae_encode_inputs(config: dict[str, Any], batch_size: int) -> tuple[Any, ...]:
  segment_frames = int(config.get("segment_frame_length", 77))
  height = int(config.get("height", 720))
  width = int(config.get("width", 1280))
  return (jax.ShapeDtypeStruct((batch_size, 3, segment_frames, height, width), jnp.float32),)


def _make_vae_decode_inputs(config: dict[str, Any], batch_size: int) -> tuple[Any, ...]:
  segment_frames = int(config.get("segment_frame_length", 77))
  height = int(config.get("height", 720))
  width = int(config.get("width", 1280))
  latent_frames = (segment_frames - 1) // 4 + 1
  return (jax.ShapeDtypeStruct((batch_size, 16, latent_frames, height // 8, width // 8), jnp.float32),)


def _build_animate_transformer(config: dict[str, Any]) -> tuple[nnx.Module, Mesh, tuple[Any, ...], dict[str, int]]:
  mesh = _build_transformer_mesh()
  axis_sizes = _transformer_axis_sizes(config)
  local_cfg = _maybe_load_local_subconfig(NNXWanAnimateTransformer3DModel, config["pretrained_model_name_or_path"], "transformer")
  local_cfg.update(
      {
          "mesh": mesh,
          "dtype": config["activations_dtype"],
          "weights_dtype": config["weights_dtype"],
          "attention": config.get("attention", "flash"),
          "flash_min_seq_length": config.get("flash_min_seq_length", 4096),
          "dropout": config.get("dropout", 0.0),
          "scan_layers": config.get("scan_layers", False),
          "mask_padding_tokens": config.get("mask_padding_tokens", True),
      }
  )

  def build(rngs):
    return NNXWanAnimateTransformer3DModel(rngs=rngs, **local_cfg)

  with mesh, nn.partitioning.axis_rules(DEFAULT_TRANSFORMER_AXIS_RULES):
    model = nnx.eval_shape(build, rngs=nnx.Rngs(int(config.get("seed", 0))))
  model = _cast_param_state_for_transformer(model, config["weights_dtype"])
  return model, mesh, DEFAULT_TRANSFORMER_AXIS_RULES, axis_sizes


def _build_vae(config: dict[str, Any]) -> tuple[nnx.Module, Mesh, tuple[Any, ...], dict[str, int]]:
  mesh = _build_vae_mesh()
  axis_sizes = _vae_axis_sizes(config)

  def build(rngs):
    return AutoencoderKLWan(rngs=rngs, mesh=mesh, dtype=jnp.float32, weights_dtype=config["weights_dtype"])

  with mesh, nn.partitioning.axis_rules(DEFAULT_VAE_AXIS_RULES):
    model = nnx.eval_shape(build, rngs=nnx.Rngs(int(config.get("seed", 0))))
  return model, mesh, DEFAULT_VAE_AXIS_RULES, axis_sizes


def _run_component(name: str, config: dict[str, Any], batch_size: int) -> dict[str, Any]:
  if name == "animate_transformer":
    model, mesh, axis_rules, axis_sizes = _build_animate_transformer(config)
    _, module_by_trace_id = _mark_trace_ids(model)
    param_stats = _collect_param_stats(model, axis_sizes)
    call_records = _trace_module_calls(
        model,
        "__call__",
        _make_transformer_inputs(config, model, batch_size),
        {"return_dict": False},
        mesh,
        axis_rules,
        axis_sizes,
    )
    method = "__call__"
  elif name == "vae_encode":
    model, mesh, axis_rules, axis_sizes = _build_vae(config)
    _, module_by_trace_id = _mark_trace_ids(model)
    param_stats = _collect_param_stats(model, axis_sizes)
    cache = AutoencoderKLWanCache(model)
    call_records = _trace_module_calls(
        model,
        "encode",
        _make_vae_encode_inputs(config, batch_size) + (cache,),
        {"return_dict": False},
        mesh,
        axis_rules,
        axis_sizes,
    )
    method = "encode"
  elif name == "vae_decode":
    model, mesh, axis_rules, axis_sizes = _build_vae(config)
    _, module_by_trace_id = _mark_trace_ids(model)
    param_stats = _collect_param_stats(model, axis_sizes)
    cache = AutoencoderKLWanCache(model)
    call_records = _trace_module_calls(
        model,
        "decode",
        _make_vae_decode_inputs(config, batch_size) + (cache,),
        {"return_dict": False},
        mesh,
        axis_rules,
        axis_sizes,
    )
    method = "decode"
  else:
    raise ValueError(f"Unsupported component {name}")

  rows = [_record_to_json(record, param_stats[id(module_by_trace_id[record.trace_id])]) for record in call_records]
  rows.sort(key=lambda row: row["activations"]["activation_peak_per_device_bytes"], reverse=True)
  param_rows = sorted(rows, key=lambda row: row["params"]["own_per_device_total_bytes"], reverse=True)

  root_module = next(module for path, module in nnx.iter_modules(model) if path == ())
  root_param_stats = param_stats[id(root_module)]
  return {
      "component": name,
      "root_method": method,
      "mesh_axis_sizes": axis_sizes,
      "model_size": {
          "logical_by_dtype": root_param_stats.subtree_by_dtype,
          "logical_total_bytes": sum(root_param_stats.subtree_by_dtype.values()),
          "per_device_by_dtype": root_param_stats.subtree_per_device_by_dtype,
          "per_device_total_bytes": sum(root_param_stats.subtree_per_device_by_dtype.values()),
      },
      "rows": rows,
      "param_rows": param_rows,
  }


def _format_console_row(row: dict[str, Any]) -> str:
  in_info = row["activations"]
  params = row["params"]
  return (
      f"{row['path']:<48} "
      f"{row['module_type']:<32} "
      f"in={_human_bytes(in_info['input_total_bytes'])}/{_human_bytes(in_info['input_per_device_total_bytes'])} "
      f"out={_human_bytes(in_info['output_total_bytes'])}/{_human_bytes(in_info['output_per_device_total_bytes'])} "
      f"act={_human_bytes(in_info['activation_peak_bytes'])}/{_human_bytes(in_info['activation_peak_per_device_bytes'])} "
      f"params={_human_bytes(params['own_total_bytes'])}/{_human_bytes(params['own_per_device_total_bytes'])} "
      f"dtype={_format_array_dtypes(in_info['outputs'])} "
      f"shard={_format_array_shardings(in_info['outputs'])}"
  )


def _render_array_cell(arrays: list[dict[str, Any]]) -> str:
  if not arrays:
    return "<div class='muted'>-</div>"
  pieces = []
  for arr in arrays[:6]:
    pieces.append(
        "<div class='tensor'>"
        f"<div><code>{html.escape(arr['path'])}</code></div>"
        f"<div>{html.escape('x'.join(str(x) for x in arr['shape']))}</div>"
        f"<div>{html.escape(_short_dtype_name(arr['dtype']))} | {html.escape(arr['sharding'])}</div>"
        f"<div>{html.escape(_human_bytes(arr['bytes']))} logical</div>"
        f"<div>{html.escape(_human_bytes(arr['per_device_bytes']))} per-device</div>"
        "</div>"
    )
  if len(arrays) > 6:
    pieces.append(f"<div class='muted'>+{len(arrays) - 6} more</div>")
  return "".join(pieces)


def _render_model_size(component: dict[str, Any]) -> str:
  ms = component["model_size"]
  logical = _format_dtype_bytes(ms["logical_by_dtype"])
  local = _format_dtype_bytes(ms["per_device_by_dtype"])
  mesh = ", ".join(f"{k}={v}" for k, v in component["mesh_axis_sizes"].items())
  return (
      "<div class='summary-grid'>"
      f"<div class='card'><div class='label'>Mesh</div><div class='value'>{html.escape(mesh)}</div></div>"
      f"<div class='card'><div class='label'>Model Logical</div><div class='value'>{html.escape(_human_bytes(ms['logical_total_bytes']))}</div><div class='sub'>{html.escape(logical)}</div></div>"
      f"<div class='card'><div class='label'>Model Per Device</div><div class='value'>{html.escape(_human_bytes(ms['per_device_total_bytes']))}</div><div class='sub'>{html.escape(local)}</div></div>"
      "</div>"
  )


def _render_rows_table(title: str, rows: list[dict[str, Any]], limit: int, param_mode: bool = False) -> str:
  if param_mode:
    body_rows = []
    shown = 0
    for row in rows:
      if row["params"]["own_total_bytes"] == 0:
        continue
      body_rows.append(
          "<tr>"
          f"<td><code>{html.escape(row['path'])}</code></td>"
          f"<td>{html.escape(row['module_type'])}</td>"
          f"<td>{html.escape(_human_bytes(row['params']['own_total_bytes']))}</td>"
          f"<td>{html.escape(_human_bytes(row['params']['own_per_device_total_bytes']))}</td>"
          f"<td>{html.escape(_format_dtype_bytes(row['params']['own_by_dtype']))}</td>"
          f"<td>{html.escape(', '.join(row['params']['own_shardings']) if row['params']['own_shardings'] else '-')}</td>"
          "</tr>"
      )
      shown += 1
      if shown >= limit:
        break
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table><thead><tr>"
        "<th>Path</th><th>Type</th><th>Params Logical</th><th>Params Per Device</th><th>Dtypes</th><th>Sharding</th>"
        "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )

  body_rows = []
  for row in rows[:limit]:
    act = row["activations"]
    body_rows.append(
        "<tr>"
        f"<td><code>{html.escape(row['path'])}</code></td>"
        f"<td>{html.escape(row['module_type'])}</td>"
        f"<td>{_render_array_cell(act['inputs'])}</td>"
        f"<td>{html.escape(_human_bytes(act['input_total_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(act['input_per_device_total_bytes']))}</td>"
        f"<td>{_render_array_cell(act['outputs'])}</td>"
        f"<td>{html.escape(_human_bytes(act['output_total_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(act['output_per_device_total_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(act['activation_peak_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(act['activation_peak_per_device_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(row['params']['own_total_bytes']))}</td>"
        f"<td>{html.escape(_human_bytes(row['params']['own_per_device_total_bytes']))}</td>"
        "</tr>"
    )
  return (
      f"<h3>{html.escape(title)}</h3>"
      "<table><thead><tr>"
      "<th>Path</th><th>Type</th><th>Inputs</th><th>In Logical</th><th>In Per Device</th>"
      "<th>Outputs</th><th>Out Logical</th><th>Out Per Device</th>"
      "<th>Act Peak Logical</th><th>Act Peak Per Device</th>"
      "<th>Own Params Logical</th><th>Own Params Per Device</th>"
      "</tr></thead><tbody>"
      + "".join(body_rows)
      + "</tbody></table>"
  )


def _render_html(results: dict[str, Any], top: int, param_top: int) -> str:
  component_sections = []
  for component in results["components"]:
    component_sections.append(
        "<section class='component'>"
        f"<h2>{html.escape(component['component'])}</h2>"
        f"{_render_model_size(component)}"
        f"{_render_rows_table('Activation Hotspots', component['rows'], top, param_mode=False)}"
        f"{_render_rows_table('Parameter Hotspots', component['param_rows'], param_top, param_mode=True)}"
        "</section>"
    )

  config_json = html.escape(json.dumps(results["config"], indent=2))
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WAN Memory Roofline</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a2f;
      --panel-2: #17213b;
      --text: #e7ecf7;
      --muted: #9fb0cf;
      --line: #2b3658;
      --accent: #8bd3ff;
      --good: #86efac;
      --warn: #fbbf24;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #08101d 0%, #0d1528 100%);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
    }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    h3 {{ font-size: 16px; margin-top: 20px; }}
    .intro, .component, .config {{
      background: rgba(18, 26, 47, 0.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.2);
      backdrop-filter: blur(8px);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin: 14px 0 10px;
    }}
    .card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .sub {{
      color: var(--muted);
      margin-top: 6px;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #0f1730;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #16213f;
      z-index: 1;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }}
    tr:hover td {{
      background: rgba(139, 211, 255, 0.04);
    }}
    code {{
      color: var(--accent);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-word;
    }}
    .tensor {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 8px;
      padding: 8px;
      margin-bottom: 8px;
    }}
    .muted {{
      color: var(--muted);
    }}
    pre {{
      background: #0f1730;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="intro">
      <h1>WAN Memory Roofline</h1>
      <p>Logical bytes are full tensor sizes. Per-device bytes are estimated from the intended sharding axes in the current config overrides.</p>
    </section>
    <section class="config">
      <h2>Run Config</h2>
      <pre>{config_json}</pre>
    </section>
    {''.join(component_sections)}
  </main>
</body>
</html>"""


def main() -> None:
  parser = argparse.ArgumentParser(description="Shape-only WAN memory report")
  parser.add_argument("--config", default="src/maxdiffusion/configs/base_wan_animate_27b.yml")
  parser.add_argument(
      "--components",
      nargs="+",
      default=["animate_transformer", "vae_encode", "vae_decode"],
      choices=["animate_transformer", "vae_encode", "vae_decode"],
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--top", type=int, default=25)
  parser.add_argument("--param-top", type=int, default=10)
  parser.add_argument("--json-out", default="")
  parser.add_argument("--html-out", default="/tmp/wan_memory_roofline.html")
  args, overrides = parser.parse_known_args()

  config = _apply_overrides(_load_yaml_config(Path(args.config)), overrides)
  results = {
      "config": {
          "config_path": str(Path(args.config).resolve()),
          "weights_dtype": _dtype_name(config["weights_dtype"]),
          "activations_dtype": _dtype_name(config["activations_dtype"]),
          "height": int(config.get("height", 720)),
          "width": int(config.get("width", 1280)),
          "num_frames": int(config.get("num_frames", 121)),
          "segment_frame_length": int(config.get("segment_frame_length", 77)),
          "batch_size": args.batch_size,
          "transformer_mesh_axis_sizes": _transformer_axis_sizes(config),
          "vae_mesh_axis_sizes": _vae_axis_sizes(config),
          "overrides": overrides,
      },
      "components": [],
  }

  for component_name in args.components:
    results["components"].append(_run_component(component_name, config, args.batch_size))

  if args.json_out:
    Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")

  html_report = _render_html(results, args.top, args.param_top)
  Path(args.html_out).write_text(html_report, encoding="utf-8")

  for component in results["components"]:
    model_size = component["model_size"]
    print(f"\n== {component['component']} ==")
    print(
        "model_size "
        f"logical={_human_bytes(model_size['logical_total_bytes'])} "
        f"per_device={_human_bytes(model_size['per_device_total_bytes'])}"
    )
    print("activation_hotspots:")
    for row in component["rows"][: args.top]:
      print(_format_console_row(row))
    print("param_hotspots:")
    printed = 0
    for row in component["param_rows"]:
      if row["params"]["own_total_bytes"] == 0:
        continue
      print(
          f"{row['path']:<48} {row['module_type']:<32} "
          f"params={_human_bytes(row['params']['own_total_bytes'])}/{_human_bytes(row['params']['own_per_device_total_bytes'])} "
          f"dtype={_format_dtype_bytes(row['params']['own_by_dtype'])} "
          f"shard={', '.join(row['params']['own_shardings']) if row['params']['own_shardings'] else '-'}"
      )
      printed += 1
      if printed >= args.param_top:
        break

  print(f"\nhtml_report: {args.html_out}")
  if args.json_out:
    print(f"json_report: {args.json_out}")


if __name__ == "__main__":
  main()
