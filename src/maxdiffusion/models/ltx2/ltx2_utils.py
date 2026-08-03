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
"""

import json
import os
import time
import concurrent.futures
from typing import Optional, Callable
import torch
import numpy as np
import re
import ml_dtypes
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
import uuid
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from safetensors.flax import load_file, save_file
from flax.traverse_util import unflatten_dict, flatten_dict
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)


def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def try_load_converted_weights(cache_dir: str, eval_shapes: dict, cast_dtype_fn: Optional[Callable]) -> Optional[dict]:
  cache_file = cache_dir + ".safetensors"
  if not os.path.isfile(cache_file):
    return None
  try:
    expected_keys = set(flatten_dict(eval_shapes).keys())
    tensors = load_file(cache_file)
    flax_state_dict = {}

    flat_shapes = flatten_dict(eval_shapes)
    for key_str, value in tensors.items():
      flax_key = _tuple_str_to_int(tuple(key_str.split(".")))

      expected_dtype = np.dtype(cast_dtype_fn(flax_key)) if cast_dtype_fn is not None else np.dtype(value.dtype)
      if np.dtype(value.dtype) != expected_dtype:
        raise ValueError(f"dtype policy changed for {key_str}")

      expected_shape = flat_shapes[flax_key].shape
      if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"shape changed for {key_str}")

      flax_state_dict[flax_key] = value

    if set(flax_state_dict.keys()) != expected_keys:
      return None
    return unflatten_dict(flax_state_dict)
  except Exception as e:
    max_logging.log(f"Converted-weights cache unusable ({e}); reconverting")
    return None


def save_converted_weights(cache_dir: str, flat_state_dict: dict) -> None:
  cache_file = cache_dir + ".safetensors"
  tmp_file = f"{cache_file}.tmp.{uuid.uuid4().hex}"

  string_dict = {}
  for flax_key, value in flat_state_dict.items():
    key_str = ".".join(str(k) for k in flax_key)
    string_dict[key_str] = np.array(value)

  save_file(string_dict, tmp_file)
  try:
    os.replace(tmp_file, cache_file)
  except OSError:
    try:
      os.remove(tmp_file)
    except OSError:
      pass


KNOWN_UPSAMPLER_CONFIGS = {
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": {
        "spatial_upsample": True,
        "temporal_upsample": False,
        "rational_spatial_scale": None,
    },
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": {
        "spatial_upsample": True,
        "temporal_upsample": False,
        "rational_spatial_scale": None,
    },
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors": {
        "spatial_upsample": True,
        "temporal_upsample": False,
        "rational_spatial_scale": 1.5,
    },
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors": {
        "spatial_upsample": False,
        "temporal_upsample": True,
        "rational_spatial_scale": None,
    },
}


def rename_for_ltx2_transformer(key):
  """
  Renames Diffusers LTX-2 keys to MaxDiffusion Flax LTX-2 keys.
  """
  if "caption_proj" in key and "caption_projection" not in key:
    key = key.replace("caption_proj", "caption_projection")
  if "audio_caption_proj" in key and "audio_caption_projection" not in key:
    key = key.replace("audio_caption_proj", "audio_caption_projection")

  key = key.replace("patchify_proj", "proj_in")
  key = key.replace("audio_patchify_proj", "audio_proj_in")
  key = key.replace("norm_final", "norm_out")
  if "adaLN_modulation_1" in key:
    key = key.replace("adaLN_modulation_1", "scale_shift_table")

  if "caption_modulator_1" in key:
    key = key.replace("caption_modulator_1", "video_a2v_cross_attn_scale_shift_table")
  if "audio_caption_modulator_1" in key:
    key = key.replace("audio_caption_modulator_1", "audio_a2v_cross_attn_scale_shift_table")
  if "audio_norm_final" in key:
    key = key.replace("audio_norm_final", "audio_norm_out")
  if ("audio_ff" in key or "ff" in key) and "proj" in key:
    key = key.replace(".proj", "")
  if "to_out_0" in key:
    key = key.replace("to_out_0", "to_out")

  return key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48):
  block_index = None

  # Handle transformer_blocks_N (underscore) produced by rename_key
  if len(pt_tuple_key) > 0 and "transformer_blocks_" in pt_tuple_key[0]:
    m = re.match(r"transformer_blocks_(\d+)", pt_tuple_key[0])
    if m:
      block_index = int(m.group(1))
      if scan_layers:
        # Map transformer_blocks_N -> transformer_blocks
        pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[1:]
      else:
        # Map transformer_blocks_N -> transformer_blocks, index
        pt_tuple_key = ("transformer_blocks", str(block_index)) + pt_tuple_key[1:]

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  flax_key_str = [str(k) for k in flax_key]

  if "scale_shift_table" in flax_key_str:
    if flax_key_str[-1] in ["kernel", "weight"]:
      flax_key_str.pop()

  flax_key = tuple(flax_key_str)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers and block_index is not None:
    if "transformer_blocks" in flax_key:
      if flax_key in flax_state_dict:
        new_tensor = flax_state_dict[flax_key]
      else:
        new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)

      new_tensor = new_tensor.at[block_index].set(flax_tensor)
      flax_tensor = new_tensor

  return flax_key, flax_tensor


def load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=None):
  """
  Loads weights from a sharded safetensors checkpoint or a specific file.
  """
  tensors = {}

  if filename is not None:
    try:
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
      if filename.endswith(".safetensors"):
        with safe_open(ckpt_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
      else:
        loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
        for k, v in loaded_state_dict.items():
          tensors[k] = torch2jax(v)
      return tensors
    except EntryNotFoundError:
      max_logging.log(f"Warning: Specific file {filename} not found. Falling back to default logic.")

  index_file = "diffusion_pytorch_model.safetensors.index.json"
  try:
    index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=index_file)
    with open(index_path, "r") as f:
      index_data = json.load(f)
    weight_map = index_data["weight_map"]
    shards = set(weight_map.values())

    for shard_file in shards:
      shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=shard_file)
      with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
  except EntryNotFoundError:
    # Fallback to single file
    filename = "diffusion_pytorch_model.safetensors"
    try:
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
    except EntryNotFoundError:
      filename = "diffusion_pytorch_model.bin"
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)

    if filename.endswith(".safetensors"):
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
    else:
      loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
      for k, v in loaded_state_dict.items():
        tensors[k] = torch2jax(v)

  return tensors


def _torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
  if tensor.dtype == torch.bfloat16:
    return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
  return tensor.numpy()


def _get_eval_shape(value) -> tuple[int, ...]:
  if hasattr(value, "shape"):
    return tuple(value.shape)
  if hasattr(value, "value") and hasattr(value.value, "shape"):
    return tuple(value.value.shape)
  raise ValueError(f"Unable to determine the initialized shape for {type(value).__name__}")


def _get_eval_dtype(value) -> np.dtype:
  if hasattr(value, "dtype"):
    return np.dtype(value.dtype)
  if hasattr(value, "value") and hasattr(value.value, "dtype"):
    return np.dtype(value.value.dtype)
  raise ValueError(f"Unable to determine the initialized dtype for {type(value).__name__}")


def _get_scanned_layer_shapes(flattened_eval_shapes):
  scanned_layer_shapes = {}
  for key, value in flattened_eval_shapes.items():
    normalized_key = _tuple_str_to_int(tuple(str(item) for item in key))
    if not normalized_key or normalized_key[0] != "transformer_blocks":
      continue

    shape = _get_eval_shape(value)
    if not shape:
      raise ValueError(f"Scanned parameter {'.'.join(map(str, normalized_key))} has no layer axis")
    scanned_layer_shapes[normalized_key] = shape

  if not scanned_layer_shapes:
    raise ValueError("scan_layers=True, but eval_shapes contains no transformer_blocks parameters")

  layer_counts = {shape[0] for shape in scanned_layer_shapes.values()}
  if len(layer_counts) != 1:
    raise ValueError(f"Inconsistent scanned layer counts in eval_shapes: {sorted(layer_counts)}")

  scanned_num_layers = next(iter(layer_counts))
  if scanned_num_layers <= 0:
    raise ValueError(f"Invalid scanned layer count derived from eval_shapes: {scanned_num_layers}")
  return scanned_num_layers, scanned_layer_shapes


def load_transformer_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: Optional[int] = None,
    scan_layers: bool = True,
    subfolder: str = "transformer",
    cast_dtype_fn=None,
    converted_cache_dir: str = "",
):
  """Loads and converts an LTX2 transformer checkpoint into host arrays.

  When ``converted_cache_dir`` is set, the final-dtype flat tree is cached
  under a content-addressed identity derived from the resolved checkpoint,
  converter ABI, scan mode, and exact initialized parameter schema.
  """
  device = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")

  flattened_dict = flatten_dict(eval_shapes)
  random_flax_state_dict = {}
  for key in flattened_dict:
    random_flax_state_dict[tuple(str(item) for item in key)] = flattened_dict[key]

  scanned_num_layers = None
  scanned_layer_shapes = {}
  if scan_layers:
    scanned_num_layers, scanned_layer_shapes = _get_scanned_layer_shapes(flattened_dict)
    if num_layers is not None and num_layers != scanned_num_layers:
      raise ValueError(f"num_layers={num_layers} does not match the {scanned_num_layers} layers derived from eval_shapes")

  index_file = "diffusion_pytorch_model.safetensors.index.json"
  try:
    index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=index_file)
    with open(index_path, "r") as f:
      index_data = json.load(f)
    weight_map = index_data["weight_map"]
    shards = sorted(set(weight_map.values()))

    def resolve_shard_path(model_file):
      return hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=model_file)

  except EntryNotFoundError:
    shards = ["diffusion_pytorch_model.safetensors"]

    def resolve_shard_path(model_file):
      try:
        return hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=model_file)
      except EntryNotFoundError:
        return hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename="diffusion_pytorch_model.bin")

  checkpoint_files = [resolve_shard_path(model_file) for model_file in shards]

  expected_dtypes = {
      tuple(key): np.dtype(cast_dtype_fn(tuple(key))) if cast_dtype_fn is not None else _get_eval_dtype(value)
      for key, value in flattened_dict.items()
  }

  if converted_cache_dir:
    t_cache_load = time.perf_counter()
    cached = try_load_converted_weights(converted_cache_dir, unflatten_dict(flattened_dict), cast_dtype_fn)
    if cached is not None:
      max_logging.log(
          f"Loaded converted {subfolder or 'transformer'} weights from {converted_cache_dir} "
          f"in {time.perf_counter() - t_cache_load:.1f}s"
      )
      return cached

  t_start = time.perf_counter()

  flax_state_dict = {}
  populated_layer_indices = {}

  def convert_safetensors_chunk(ckpt_shard_path, chunk_keys):
    results = []
    with safe_open(ckpt_shard_path, framework="pt") as f:
      for pt_key in chunk_keys:
        tensor = _torch_tensor_to_numpy(f.get_tensor(pt_key))
        results.append(process_tensor(pt_key, tensor))
    return results

  def convert_bin(ckpt_shard_path):
    results = []
    loaded_state_dict = torch.load(ckpt_shard_path, map_location="cpu")
    for pt_key, pt_tensor in loaded_state_dict.items():
      tensor = _torch_tensor_to_numpy(pt_tensor)
      results.append(process_tensor(pt_key, tensor))
    return results

  def process_tensor(pt_key, tensor):
    renamed_pt_key = rename_key(pt_key)
    renamed_pt_key = rename_for_ltx2_transformer(renamed_pt_key)
    pt_tuple_key = tuple(renamed_pt_key.split("."))

    block_index = None
    if len(pt_tuple_key) > 0 and "transformer_blocks_" in pt_tuple_key[0]:
      m = re.match(r"transformer_blocks_(\d+)", pt_tuple_key[0])
      if m:
        if scan_layers:
          block_index = int(m.group(1))
          pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[1:]
        else:
          # For nnx.List, NNX uses string indices ('0', '1', etc.)
          pt_tuple_key = ("transformer_blocks", m.group(1)) + pt_tuple_key[1:]

    flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
    flax_key_str = [str(k) for k in flax_key]
    if "scale_shift_table" in flax_key_str and flax_key_str[-1] in ["kernel", "weight"]:
      flax_key_str.pop()
    flax_key = tuple(flax_key_str)
    flax_key = _tuple_str_to_int(flax_key)

    if block_index is not None:
      key_name = ".".join(map(str, flax_key))
      if block_index < 0 or block_index >= scanned_num_layers:
        raise ValueError(f"Scanned layer index {block_index} for {key_name} is out of range [0, {scanned_num_layers})")
      if flax_key not in scanned_layer_shapes:
        raise ValueError(f"Checkpoint tensor {pt_key!r} maps to unexpected scanned parameter {key_name}")

      expected_shape = scanned_layer_shapes[flax_key]
      if tuple(flax_tensor.shape) != expected_shape[1:]:
        raise ValueError(
            f"Shape mismatch for scanned parameter {key_name}: "
            f"expected per-layer shape {expected_shape[1:]}, got {tuple(flax_tensor.shape)}"
        )

      return (True, pt_key, flax_key, block_index, flax_tensor)
    else:
      target_dtype = (
          np.dtype(cast_dtype_fn(flax_key))
          if cast_dtype_fn is not None
          else expected_dtypes.get(flax_key, np.dtype(flax_tensor.dtype))
      )
      value = np.array(flax_tensor, dtype=target_dtype, copy=True, order="C")
      return (False, pt_key, flax_key, None, value)

  def apply_result(is_scanned, pt_key, flax_key, block_index, value):
    if is_scanned:
      key_name = ".".join(map(str, flax_key))
      populated = populated_layer_indices.setdefault(flax_key, set())
      if block_index in populated:
        raise ValueError(f"Duplicate scanned layer index {block_index} for {key_name}")

      stacked = flax_state_dict.get(flax_key)
      if stacked is None:
        expected_shape = scanned_layer_shapes[flax_key]
        target_dtype = (
            np.dtype(cast_dtype_fn(flax_key))
            if cast_dtype_fn is not None
            else expected_dtypes.get(flax_key, np.dtype(value.dtype))
        )
        stacked = np.zeros(expected_shape, dtype=target_dtype)
        flax_state_dict[flax_key] = stacked
      stacked[block_index] = value
      populated.add(block_index)
    else:
      flax_state_dict[flax_key] = value

  chunk_size = 32
  safetensors_tasks = []
  all_results = []
  for ckpt_shard_path in checkpoint_files:
    if ckpt_shard_path.endswith(".safetensors"):
      with safe_open(ckpt_shard_path, framework="pt") as f:
        shard_keys = list(f.keys())
      for i in range(0, len(shard_keys), chunk_size):
        safetensors_tasks.append((ckpt_shard_path, shard_keys[i : i + chunk_size]))
    else:
      all_results.extend(convert_bin(ckpt_shard_path))

  if safetensors_tasks:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(convert_safetensors_chunk, path, keys) for path, keys in safetensors_tasks]
      for future in concurrent.futures.as_completed(futures):
        all_results.extend(future.result())

  for result in all_results:
    apply_result(*result)

  if scan_layers:
    expected_indices = set(range(scanned_num_layers))
    missing_layers = []
    for flax_key in scanned_layer_shapes:
      missing = sorted(expected_indices - populated_layer_indices.get(flax_key, set()))
      if missing:
        missing_layers.append(f"{'.'.join(map(str, flax_key))}: {missing}")
    if missing_layers:
      raise ValueError(f"Missing scanned layer indices: {'; '.join(missing_layers)}")

  validate_flax_state_dict(eval_shapes, flax_state_dict)
  if converted_cache_dir and not os.path.isdir(converted_cache_dir):
    t_cache_save = time.perf_counter()
    if jax.process_index() == 0:
      save_converted_weights(converted_cache_dir, flax_state_dict)
      max_logging.log(
          f"Saved converted {subfolder or 'transformer'} weights to {converted_cache_dir} "
          f"in {time.perf_counter() - t_cache_save:.1f}s"
      )
  flax_state_dict = unflatten_dict(flax_state_dict)
  max_logging.log(f"Converted weights in {time.perf_counter() - t_start:.1f}s")
  return flax_state_dict


def load_vae_weights(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vae"
):
  device = jax.local_devices(backend=device)[0]

  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")

  with jax.default_device(device):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_eval = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_eval:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_eval[key]

    for pt_key, tensor in tensors.items():
      # latents_mean and latents_std are nnx.Params and will be loaded correctly.
      renamed_pt_key = rename_key(pt_key)
      renamed_pt_key = renamed_pt_key.replace("nin_shortcut", "conv_shortcut")

      pt_tuple_key = tuple(renamed_pt_key.split("."))

      pt_list = []
      resnet_index = None

      for i, part in enumerate(pt_tuple_key):
        if "_" in part and part.split("_")[-1].isdigit():
          name = "_".join(part.split("_")[:-1])
          idx = int(part.split("_")[-1])

          if name == "resnets":
            pt_list.append("resnets")
            resnet_index = idx
          elif name == "upsamplers":
            pt_list.append("upsampler")
          elif name in ["down_blocks", "up_blocks", "downsamplers"]:
            pt_list.append(name)
            pt_list.append(str(idx))
          else:
            pt_list.append(part)
        elif part == "upsampler":
          pt_list.append("upsampler")
        elif part in ["conv1", "conv2", "conv", "conv_in", "conv_out", "conv_shortcut"]:
          pt_list.append(part)
          if (
              part != "conv"
              and (i + 1 == len(pt_tuple_key) or pt_tuple_key[i + 1] != "conv")
              and (len(pt_list) < 2 or pt_list[-2] != "conv")
          ):
            pt_list.append("conv")
        else:
          pt_list.append(part)

      pt_tuple_key = tuple(pt_list)

      flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
      flax_key = _tuple_str_to_int(flax_key)

      if resnet_index is not None:
        str_flax_key = tuple([str(x) for x in flax_key])
        if str_flax_key in random_flax_state_dict:
          if flax_key not in flax_state_dict:
            target_shape = random_flax_state_dict[str_flax_key].shape
            flax_state_dict[flax_key] = jnp.zeros(target_shape, dtype=flax_tensor.dtype)
          flax_state_dict[flax_key] = flax_state_dict[flax_key].at[resnet_index].set(flax_tensor)
        else:
          flax_state_dict[flax_key] = flax_tensor
      else:
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
    filtered_eval_shapes = {
        k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
    }

    validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def rename_for_ltx2_vocoder(key):
  key = key.replace("ups.", "upsamplers.")
  key = key.replace("resblocks", "resnets")
  key = key.replace("conv_post", "conv_out")
  return key


def load_vocoder_weights(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vocoder"
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)

  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  for pt_key, tensor in tensors.items():
    key = rename_for_ltx2_vocoder(pt_key)
    parts = key.split(".")

    if parts[-1] == "weight":
      parts[-1] = "kernel"

    flax_key = _tuple_str_to_int(parts)

    # Skip filter keys as they are derived in NNX model
    if "filter" in flax_key:
      continue

    if flax_key[-1] == "kernel":
      if "upsamplers" in flax_key:
        tensor = tensor.transpose(2, 0, 1)[::-1, :, :]
      else:
        tensor = tensor.transpose(2, 1, 0)

    if "mel_stft" in flax_key and ("forward_basis" in flax_key or "inverse_basis" in flax_key):
      tensor = tensor.transpose(2, 1, 0)

    flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

  validate_flax_state_dict(eval_shapes, flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_connector(key):
  key = key.replace("video_connector", "video_embeddings_connector")
  key = key.replace("audio_connector", "audio_embeddings_connector")
  key = key.replace("text_proj_in", "feature_extractor.linear")
  key = key.replace("audio_feature_extractor.linear", "audio_text_proj_in")
  key = key.replace("video_feature_extractor.linear", "video_text_proj_in")

  if "transformer_blocks" in key:
    key = key.replace("transformer_blocks", "stacked_blocks")
    key = key.replace("ff.net.0.proj", "ff.net_0")
    key = key.replace("ff.net.2", "ff.net_2")
    key = key.replace("to_out.0", "to_out")

  if key.endswith(".weight"):
    if "norm_q" in key or "norm_k" in key:
      key = key.replace(".weight", ".scale")
    else:
      key = key.replace(".weight", ".kernel")

  return key


def load_connector_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "connectors",
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  grouped_weights = {"video_embeddings_connector": {}, "audio_embeddings_connector": {}}

  for pt_key, tensor in tensors.items():
    key = rename_for_ltx2_connector(pt_key)

    if key.endswith(".kernel"):
      if tensor.ndim == 2:
        tensor = tensor.transpose(1, 0)

    if "stacked_blocks" in key:
      parts = key.split(".")
      if "stacked_blocks" in parts:
        sb_index = parts.index("stacked_blocks")
        if sb_index + 1 < len(parts):
          layer_idx = int(parts[sb_index + 1])
          connector = parts[0]

          param_parts = parts[: sb_index + 1] + parts[sb_index + 2 :]
          param_name = tuple(param_parts)

          if connector in grouped_weights:
            if param_name not in grouped_weights[connector]:
              grouped_weights[connector][param_name] = {}
            grouped_weights[connector][param_name][layer_idx] = tensor
            continue

    key_tuple = tuple(key.split("."))
    final_key_tuple = _tuple_str_to_int(key_tuple)

    flax_state_dict[final_key_tuple] = jax.device_put(tensor, device=cpu)

  for connector, params in grouped_weights.items():
    for param_name, layers in params.items():
      sorted_layers = sorted(layers.keys())
      stacked_tensor = jnp.stack([layers[i] for i in sorted_layers], axis=0)

      flax_state_dict[_tuple_str_to_int(param_name)] = jax.device_put(stacked_tensor, device=cpu)

  del tensors
  jax.clear_caches()
  validate_flax_state_dict(eval_shapes, flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_audio_vae(key):
  if key.endswith(".weight"):
    key = key.replace(".weight", ".kernel")

  key = key.replace("mid.block_1", "mid_block1")
  key = key.replace("mid.block_2", "mid_block2")
  key = key.replace("mid.attn_1", "mid_attn")

  key = key.replace("up.", "up_stages.")
  key = key.replace("down.", "down_stages.")

  key = key.replace("block.", "blocks.")

  key = key.replace("nin_shortcut", "conv_shortcut_layer")

  if "upsample.conv.kernel" in key:
    key = key.replace("upsample.conv.kernel", "upsample.conv.conv.kernel")
  if "upsample.conv.bias" in key:
    key = key.replace("upsample.conv.bias", "upsample.conv.conv.bias")

  return key


def load_audio_vae_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "audio_vae",
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  flattened_eval = flatten_dict(eval_shapes)

  for pt_key, tensor in tensors.items():
    key = rename_for_ltx2_audio_vae(pt_key)

    if key.endswith(".kernel") and tensor.ndim == 4:
      tensor = tensor.transpose(2, 3, 1, 0)

    flax_key = _tuple_str_to_int(key.split("."))

    if "up_stages" in flax_key:
      up_stages_idx = flax_key.index("up_stages")
      if up_stages_idx + 1 < len(flax_key) and isinstance(flax_key[up_stages_idx + 1], int):
        flax_key_list = list(flax_key)
        flax_key_list[up_stages_idx + 1] = 2 - flax_key[up_stages_idx + 1]
        flax_key = tuple(flax_key_list)

    flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)
  filtered_eval_shapes = {
      k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
  }

  validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_upsampler(key):
  """
  Renames PyTorch Latent Upsampler keys to match Flax LTX2LatentUpsamplerModel.
  """
  # Map weights to Flax's kernel
  key = key.replace(".weight", ".kernel")

  # Map GroupNorm weight/kernel to Flax's scale
  if "norm" in key:
    key = key.replace(".kernel", ".scale")

  # Standardize the loop naming to lowercase
  key = key.replace("res_blocks.", "res_blocks_")
  key = key.replace("post_upsample_res_blocks.", "post_upsample_res_blocks_")

  # PyTorch Sequential upsampler uses index 0 (upsampler.0.weight)
  key = key.replace("upsampler.0.", "upsampler_0.")

  # PyTorch Rational Resampler upsampler uses self.conv (upsampler.conv.weight)
  # We don't need to replace this! It naturally maps to "upsampler/conv/kernel"

  return key


def load_upsampler_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "latent_upsampler",
    dims: int = 3,
    filename: Optional[str] = None,
):
  """
  Loads and ports PyTorch upsampler weights to Flax.
  """
  device_obj = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device_obj}")

  with jax.default_device(device_obj):
    # This native util automatically handles HF hub downloads and caching!
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device_obj, filename=filename)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]

    for pt_key, tensor in tensors.items():
      key = rename_for_ltx2_upsampler(pt_key)

      # Transpose kernels for Flax
      if key.endswith(".kernel") and tensor.ndim > 1:
        if tensor.ndim == 5:
          # 3D Conv: (Out, In, D, H, W) -> (D, H, W, In, Out)
          tensor = tensor.transpose(2, 3, 4, 1, 0)
        elif tensor.ndim == 4:
          # 2D Conv: (Out, In, H, W) -> (H, W, In, Out)
          tensor = tensor.transpose(2, 3, 1, 0)
        elif tensor.ndim == 2:
          # Linear: (Out, In) -> (In, Out)
          tensor = tensor.transpose(1, 0)

      # Convert string key to tuple for unflattening
      parts = key.split(".")
      flax_key = tuple(int(p) if p.isdigit() else p for p in parts)

      flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

    # Optional validation against model shapes
    if eval_shapes:
      try:
        validate_flax_state_dict(flatten_dict(eval_shapes), flax_state_dict)
      except ValueError as e:
        max_logging.log(f"CRITICAL: Upsampler weight shape mismatch detected: {e}")
        raise RuntimeError("Failed to validate upsampler weights against expected shapes.") from e

    del tensors
    jax.clear_caches()
    return unflatten_dict(flax_state_dict)


def adain_filter_latent(latents: jax.Array, reference_latents: jax.Array, factor: float = 1.0) -> jax.Array:
  """Scales high-res latents using global channel statistics from reference latents."""
  axes = (1, 2, 3)
  r_sd = jnp.std(reference_latents, axis=axes, keepdims=True)
  r_mean = jnp.mean(reference_latents, axis=axes, keepdims=True)

  i_sd = jnp.std(latents, axis=axes, keepdims=True)
  i_mean = jnp.mean(latents, axis=axes, keepdims=True)

  result = ((latents - i_mean) / (i_sd + 1e-5)) * r_sd + r_mean
  result = latents + factor * (result - latents)
  return result


def tone_map_latents(latents: jax.Array, compression: float) -> jax.Array:
  """Sigmoid-based compression to regularize high-variance latents."""
  scale_factor = compression * 0.75
  abs_latents = jnp.abs(latents)
  sigmoid_term = jax.nn.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
  scales = 1.0 - 0.8 * scale_factor * sigmoid_term
  return latents * scales
