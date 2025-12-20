"""
 Copyright 2025 Google LLC

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

import os
import json
import torch
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)

CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH = "lightx2v/Wan2.1-T2V-14B-CausVid"
WAN_21_FUSION_X_MODEL_NAME_OR_PATH = "vrgamedevgirl84/Wan14BT2VFusioniX"


def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def rename_for_nnx(key):
  new_key = key
  if "norm_k" in key or "norm_q" in key:
    new_key = key[:-1] + ("scale",)
  return new_key


def rename_for_custom_trasformer(key):
  renamed_pt_key = key.replace("model.diffusion_model.", "")

  renamed_pt_key = renamed_pt_key.replace("head.modulation", "scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("head.head", "proj_out")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "condition_embedder.text_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "condition_embedder.text_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "condition_embedder.time_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "condition_embedder.time_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_projection_1", "condition_embedder.time_proj")

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace("self_attn", "attn1")
  renamed_pt_key = renamed_pt_key.replace("cross_attn", "attn2")
  renamed_pt_key = renamed_pt_key.replace(".q.", ".query.")
  renamed_pt_key = renamed_pt_key.replace(".k.", ".key.")
  renamed_pt_key = renamed_pt_key.replace(".v.", ".value.")
  renamed_pt_key = renamed_pt_key.replace(".o.", ".proj_attn.")
  renamed_pt_key = renamed_pt_key.replace("ffn_0", "ffn.act_fn.proj")
  renamed_pt_key = renamed_pt_key.replace("ffn_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace(".modulation", ".scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("norm3", "norm2.layer_norm")

  return renamed_pt_key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers):
  if scan_layers:
    if "blocks" in pt_tuple_key:
      new_key = ("blocks",) + pt_tuple_key[2:]
      block_index = int(pt_tuple_key[1])
      pt_tuple_key = new_key

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)

  flax_key = rename_for_nnx(flax_key)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers:
    if "blocks" in flax_key:
      if flax_key in flax_state_dict:
        new_tensor = flax_state_dict[flax_key]
      else:
        new_tensor = jnp.zeros((40,) + flax_tensor.shape)
      flax_tensor = new_tensor.at[block_index].set(flax_tensor)
  return flax_key, flax_tensor


def load_fusionx_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, filename="Wan14BT2VFusioniX_fp16_.safetensors")
      tensors = {}
      with safe_open(ckpt_shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))

      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)

        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))

        flax_key, flax_tensor = get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_causvid_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, filename="causal_model.pt")
      loaded_state_dict = torch.load(ckpt_shard_path)

      tensors = {}
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in loaded_state_dict.items():
        tensor = torch2jax(tensor)
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
):

  if pretrained_model_name_or_path == CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH:
    return load_causvid_transformer(pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers)
  elif pretrained_model_name_or_path == WAN_21_FUSION_X_MODEL_NAME_OR_PATH:
    return load_fusionx_transformer(pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers)
  else:
    return load_base_wan_transformer(
        pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers, subfolder
    )


def load_base_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
):
  device = jax.local_devices(backend=device)[0]
  filename = "diffusion_pytorch_model.safetensors.index.json"
  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
    index_file_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    if not os.path.isfile(index_file_path):
      raise FileNotFoundError(f"File {index_file_path} not found for local directory.")
    local_files = True
  elif hf_download:
    # download the index file for sharded models.
    index_file_path = hf_hub_download(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        filename=filename,
    )
  with jax.default_device(device):
    # open the index file.
    with open(index_file_path, "r") as f:
      index_dict = json.load(f)
    model_files = set()
    for key in index_dict["weight_map"].keys():
      model_files.add(index_dict["weight_map"][key])

    model_files = list(model_files)
    tensors = {}
    for model_file in model_files:
      if local_files:
        ckpt_shard_path = os.path.join(pretrained_model_name_or_path, subfolder, model_file)
      else:
        ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=model_file)
      # now get all the filenames for the model that need downloading
      max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")

      if ckpt_shard_path is not None:
        with safe_open(ckpt_shard_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)
    # turn all block numbers to strings just for matching weights.
    # Later they will be turned back to ints.
    random_flax_state_dict = {}
    for key in flattened_dict:
      string_tuple = tuple([str(item) for item in key])
      random_flax_state_dict[string_tuple] = flattened_dict[key]
    # del flattened_dict
    norm_added_q_buffer = {}
    print(f"DEBUG: Total keys found in checkpoint: {len(tensors)}")
    for pt_key, tensor in tensors.items():
      if "norm_added_q" in pt_key and "weight" in pt_key:
           parts = pt_key.split(".")
           try:
               if "blocks" in parts:
                   block_idx_loc = parts.index("blocks") + 1
                   block_idx = int(parts[block_idx_loc])
                   tensor = tensor.T
                   norm_added_q_buffer[block_idx] = tensor
           except Exception as e:
               print(f"Warning: skipped {pt_key} due to {e}")
           continue
           
      renamed_pt_key = rename_key(pt_key)
      if "image_embedder" in renamed_pt_key:
          if "net.0" in renamed_pt_key or "net_0" in renamed_pt_key or \
             "net.2" in renamed_pt_key or "net_2" in renamed_pt_key:
              tensor = tensor.T
          if "net.0.proj" in renamed_pt_key:
              renamed_pt_key = renamed_pt_key.replace("net.0.proj", "net_0")
              renamed_pt_key = renamed_pt_key.replace("weight", "kernel")
          elif "net_0.proj" in renamed_pt_key:
              renamed_pt_key = renamed_pt_key.replace("net_0.proj", "net_0")
              renamed_pt_key = renamed_pt_key.replace("weight", "kernel")
          if "net.2" in renamed_pt_key:
              renamed_pt_key = renamed_pt_key.replace("net.2", "net_2")
              renamed_pt_key = renamed_pt_key.replace("weight", "kernel")
          elif "net_2" in renamed_pt_key:
              renamed_pt_key = renamed_pt_key.replace("weight", "kernel")
          renamed_pt_key = renamed_pt_key.replace("norm1", "norm1.layer_norm")
          if "norm1" in renamed_pt_key or "norm2" in renamed_pt_key:
              renamed_pt_key = renamed_pt_key.replace("weight", "scale")
              renamed_pt_key = renamed_pt_key.replace("kernel", "scale")
      renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
      renamed_pt_key = renamed_pt_key.replace(".scale_shift_table", ".adaln_scale_shift_table")
      renamed_pt_key = renamed_pt_key.replace("to_out_0", "proj_attn")
      renamed_pt_key = renamed_pt_key.replace("ffn.net_2", "ffn.proj_out")
      renamed_pt_key = renamed_pt_key.replace("ffn.net_0", "ffn.act_fn")
      if "norm2.layer_norm" not in renamed_pt_key:
        renamed_pt_key = renamed_pt_key.replace("norm2", "norm2.layer_norm")
      pt_tuple_key = tuple(renamed_pt_key.split("."))
      flax_key, flax_tensor = get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers)
      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
    if norm_added_q_buffer:
        sorted_keys = sorted(norm_added_q_buffer.keys())
        sorted_tensors = [norm_added_q_buffer[i] for i in sorted_keys]
        stacked_tensor = jnp.stack(sorted_tensors, axis=0)

        target_key = None

        for key_tuple in flattened_dict.keys():
            # Check if this tuple looks like what we want
            # We check if it ends with 'norm_added_q' and 'kernel'
            if len(key_tuple) >= 2 and key_tuple[-2:] == ('norm_added_q', 'kernel'):
                 target_key = key_tuple
                 break
        if target_key:
             print(f"DEBUG: Found authoritative key in eval_shapes: {target_key}")
             flax_state_dict[target_key] = jax.device_put(stacked_tensor, device=cpu)
             print(f"Successfully injected norm_added_q with shape {stacked_tensor.shape}")
        else:
             # Fallback (should typically not happen if error message was correct)
             print("CRITICAL WARNING: Could not find norm_added_q key in eval_shapes! Using manual fallback.")
             manual_key = ('blocks', 'attn2', 'norm_added_q', 'kernel')
             flax_state_dict[manual_key] = jax.device_put(stacked_tensor, device=cpu)
        
    del flattened_dict

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def load_wan_vae(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
  device = jax.devices(device)[0]
  subfolder = "vae"
  filename = "diffusion_pytorch_model.safetensors"
  if os.path.isdir(pretrained_model_name_or_path):
    ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    if not os.path.isfile(ckpt_path):
      raise FileNotFoundError(f"File {ckpt_path} not found for local directory.")
  elif hf_download:
    ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")
  with jax.default_device(device):
    if ckpt_path is not None:
      tensors = {}
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        # Order matters
        renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
        renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
        renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

        renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_in.weight", "conv_in.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv_out.bias", "conv_out.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_out.weight", "conv_out.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
        renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
        renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
        renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
        renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("time_conv.bias", "time_conv.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("time_conv.weight", "time_conv.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
        renamed_pt_key = renamed_pt_key.replace("conv_shortcut", "conv_shortcut.conv")
        if "decoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
          renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
        if "encoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
        flax_key = _tuple_str_to_int(flax_key)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
    else:
      raise FileNotFoundError(f"Path {ckpt_path} was not found")

    return flax_state_dict