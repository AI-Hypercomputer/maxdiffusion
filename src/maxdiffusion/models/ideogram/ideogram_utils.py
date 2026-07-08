# pylint: disable=missing-module-docstring, missing-function-docstring, too-many-positional-arguments, consider-using-dict-items, unused-argument, unspecified-encoding
import json
import re
from typing import Optional

import jax
import jax.numpy as jnp

from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open
import torch

from maxdiffusion import max_logging
from ..modeling_flax_pytorch_utils import (
    rename_key_and_reshape_tensor,
    torch2jax,
    validate_flax_state_dict,
)
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

_NUM_RESOLUTIONS = 4


def _rewrite_diffusers_key(key: str) -> Optional[str]:
  if key.startswith("bn."):
    return key

  if key.startswith("quant_conv."):
    return key.replace("quant_conv.", "encoder.quant_conv.", 1)
  if key.startswith("post_quant_conv."):
    return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

  if key == "encoder.conv_norm_out.weight":
    return "encoder.norm_out.weight"
  if key == "encoder.conv_norm_out.bias":
    return "encoder.norm_out.bias"
  if key == "decoder.conv_norm_out.weight":
    return "decoder.norm_out.weight"
  if key == "decoder.conv_norm_out.bias":
    return "decoder.norm_out.bias"

  m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
  if m:
    side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
    rest = rest.replace("conv_shortcut", "nin_shortcut")
    return f"{side}.mid_block_{idx + 1}.{rest}"

  m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
  if m:
    side, rest = m.group(1), m.group(2)
    rest = (
        rest.replace("group_norm.", "norm.")
        .replace("to_q.", "q.")
        .replace("to_k.", "k.")
        .replace("to_v.", "v.")
        .replace("to_out.0.", "proj_out.")
    )
    return f"{side}.mid_attn_1.{rest}"

  m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
  if m:
    level, res_idx, rest = m.group(1), m.group(2), m.group(3)
    rest = rest.replace("conv_shortcut", "nin_shortcut")
    return f"encoder.down_blocks.{level}.{res_idx}.{rest}"

  m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
  if m:
    return f"encoder.downsamples.{m.group(1)}.conv.{m.group(2)}"

  m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
  if m:
    diffusers_idx = int(m.group(1))
    res_idx = m.group(2)
    rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
    return f"decoder.up_blocks.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.{res_idx}.{rest}"

  m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
  if m:
    diffusers_idx = int(m.group(1))
    return f"decoder.upsamples.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.conv.{m.group(2)}"

  if key.startswith((
      "encoder.conv_in.",
      "encoder.conv_out.",
      "decoder.conv_in.",
      "decoder.conv_out.",
  )):
    return key

  return None


def convert_diffusers_state_dict(src: dict) -> dict:
  out = {}
  attn_substrings = (".mid_attn_1.",)
  for src_key, tensor in src.items():
    dst_key = _rewrite_diffusers_key(src_key)
    if dst_key is None:
      continue
    if dst_key.startswith("bn."):
      continue
    if any(s in dst_key for s in attn_substrings) and dst_key.endswith(".weight") and tensor.ndim == 2:
      tensor = jnp.expand_dims(tensor, axis=(-1, -2))
    out[dst_key] = tensor
  return out


def _tuple_str_to_int(tuple_key):
  return tuple(int(item) if item.isdigit() else item for item in tuple_key)


def load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=None):
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

  index_file = "model.safetensors.index.json"
  try:
    index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=index_file)
    with open(index_path, "r") as f:
      index_data = json.load(f)
    weight_map = index_data["weight_map"]
    shards = set(weight_map.values())

    for shard in shards:
      shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=shard)
      with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
  except EntryNotFoundError:
    # Fallback for non-sharded model
    try:
      filename = "model.safetensors"
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
    except EntryNotFoundError:
      filename = "diffusion_pytorch_model.safetensors"
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


def rename_for_ideogram_transformer(pt_key: str) -> str:
  # Rename rules specifically for ideogram.
  # The pytorch model uses `transformer_blocks.0.attn.qkv.weight`
  # and we map to nnx arrays. `rename_key` handles weight->kernel.
  return pt_key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers):
  block_index = None
  m = re.match(r"^transformer_blocks\.(\d+)", ".".join(pt_tuple_key))
  if m:
    block_index = int(m.group(1))
    if scan_layers:
      # Map transformer_blocks.N -> layers
      pt_tuple_key = ("layers",) + pt_tuple_key[2:]
    else:
      # Map transformer_blocks.N -> layers.index
      pt_tuple_key = ("layers", str(block_index)) + pt_tuple_key[2:]

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  flax_key_final = []
  for k in flax_key:
    if isinstance(k, str) and k.isdigit():
      flax_key_final.append(int(k))
    else:
      flax_key_final.append(k)
  flax_key = tuple(flax_key_final)

  if scan_layers and block_index is not None:
    if "layers" in flax_key:
      if flax_key in flax_state_dict:
        new_tensor = flax_state_dict[flax_key]
      else:
        new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)

      new_tensor = new_tensor.at[block_index].set(flax_tensor)
      flax_tensor = new_tensor

  return flax_key, flax_tensor


def load_transformer_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 42,
    scan_layers: bool = True,
    subfolder: str = "transformer",
):
  device = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")

  with jax.default_device(device):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_dict:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_dict[key]

    for pt_key, tensor in tensors.items():
      if pt_key.endswith(".weight") and pt_key + "_scale" in tensors:
        scale = tensors[pt_key + "_scale"]
        if scale.ndim == 1 and tensor.ndim == 2 and scale.shape[0] == tensor.shape[0]:
          scale = jnp.expand_dims(scale, axis=-1)
        tensor = tensor * scale

      renamed_pt_key = pt_key
      renamed_pt_key = rename_for_ideogram_transformer(renamed_pt_key)

      pt_tuple_key = tuple(renamed_pt_key.split("."))

      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
      )

      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def load_vae_weights(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vae"
):
  device = jax.local_devices(backend=device)[0]

  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")

  with jax.default_device(device):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
    tensors = convert_diffusers_state_dict(tensors)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_eval = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_eval:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_eval[key]

    for pt_key, tensor in tensors.items():
      if pt_key.endswith(".weight") and pt_key + "_scale" in tensors:
        scale = tensors[pt_key + "_scale"]
        if scale.ndim == 1 and tensor.ndim == 2 and scale.shape[0] == tensor.shape[0]:
          scale = jnp.expand_dims(scale, axis=-1)
        tensor = tensor * scale
      renamed_pt_key = pt_key
      pt_tuple_key = tuple(renamed_pt_key.split("."))

      flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)

      flax_key = _tuple_str_to_int(flax_key)
      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict
