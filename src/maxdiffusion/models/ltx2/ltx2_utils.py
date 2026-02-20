"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from safetensors import safe_open
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


def rename_for_nnx(key):
  new_key = key
  if "norm_k" in key or "norm_q" in key:
    new_key = key[:-1] + ("scale",)
  return new_key


def rename_for_ltx2(key):
  renamed_pt_key = key.replace("model.diffusion_model.", "")

  # Map blocks to transformer_blocks
  renamed_pt_key = renamed_pt_key.replace("blocks.", "transformer_blocks.")
  
  # Map Embeddings
  renamed_pt_key = renamed_pt_key.replace("time_embed.linear.", "time_embed.linear.") # Check if this needs specific mapping
  # Looking at transformer_ltx2.py: time_embed is LTX2AdaLayerNormSingle which has a linear layer.
  # But PyTorch might have "time_embed.linear" and LTX2AdaLayerNormSingle has "linear" attribute.
  # So "time_embed.linear" -> "time_embed.linear" is fine, but we might need to be careful with weights/bias.

  # Map Attention Layers
  renamed_pt_key = renamed_pt_key.replace("to_q", "to_q")
  renamed_pt_key = renamed_pt_key.replace("to_k", "to_k")
  renamed_pt_key = renamed_pt_key.replace("to_v", "to_v")
  renamed_pt_key = renamed_pt_key.replace("to_out.0", "to_out") # PyTorch usually has Sequential(Linear, Identity)

  # Map Norms
  # PyTorch LTX-2 often uses "norm1", "norm2", etc.
  # Check if we need to rename gamma/beta to scale/bias? 
  # modeling_flax_pytorch_utils.rename_key already handles some of this if it sees gamma/beta.
  
  # Specific LTX-2 Mappings based on transformer_ltx2.py structure
  # transformer_ltx2.py uses:
  # self.proj_in
  # self.caption_projection
  # self.time_embed
  # self.audio_proj_in
  # self.audio_caption_projection
  # self.audio_time_embed
  # self.transformer_blocks (init_block)
  
  return renamed_pt_key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=28):
  if scan_layers:
    if "transformer_blocks" in pt_tuple_key:
      # transformer_blocks.0.attn1... -> transformer_blocks.attn1... with index 0
      # but we need to stack them.
      new_key = ("transformer_blocks",) + pt_tuple_key[2:]
      block_index = int(pt_tuple_key[1])
      pt_tuple_key = new_key

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)

  flax_key = rename_for_nnx(flax_key)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers:
    if "transformer_blocks" in flax_key:
      # Initialize the stack if not exists
      # We need to look up the shape from one layer if possible or infer it.
      # But we process one tensor at a time.
      if flax_key in flax_state_dict:
         new_tensor = flax_state_dict[flax_key]
      else:
         # Create a zeros tensor with (num_layers, ...)
         new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)
      
      new_tensor = new_tensor.at[block_index].set(flax_tensor)
      flax_tensor = new_tensor
      
  return flax_key, flax_tensor


def load_ltx2_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 28, # LTX-2 default
    scan_layers: bool = True,
    subfolder: str = "",
):
  device = jax.local_devices(backend=device)[0]
  # LTX-2 usually has a specific filename like "ltx_video_transformer.safetensors" or similar in diffusers
  # But since we are pointing to Lightricks/LTX-2, it might be in a subfolder or root.
  # Diffusers usually expects "transformer" subfolder.
  filename = "diffusion_pytorch_model.safetensors"
  if subfolder == "":
      subfolder = "transformer" # Default for diffusers pipeline

  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
    index_file_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_file_path):
       local_files = True
       # If index file exists, it's sharded. But LTX-2 is likely single file or sharded.
       # If single file:
    elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, filename)):
       local_files = True
       ckpt_shard_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    else:
       # Verify if it is just a file path provided
       if os.path.isfile(pretrained_model_name_or_path):
           ckpt_shard_path = pretrained_model_name_or_path
           local_files = True
           subfolder="" # It's a file
  
  tensors = {}
  # Handle both sharded and non-sharded loading loosely for now, based on wan_utils.
  # Wan utils had specific logic for index file.
  
  if hf_download and not local_files:
      try:
          # Try downloading index first
          index_file_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="diffusion_pytorch_model.safetensors.index.json",
          )
          # If successful, load sharded
          with open(index_file_path, "r") as f:
            index_dict = json.load(f)
          model_files = set(index_dict["weight_map"].values())
          for model_file in model_files:
             path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=model_file)
             with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = torch2jax(f.get_tensor(k))
      except:
          # Fallback to single file
          try:
              ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
              with safe_open(ckpt_shard_path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = torch2jax(f.get_tensor(k))
          except Exception as e:
                # Try requesting just the file if subfolder is wrong
                raise e

  elif local_files and 'ckpt_shard_path' in locals():
      with safe_open(ckpt_shard_path, framework="pt") as f:
          for k in f.keys():
              tensors[k] = torch2jax(f.get_tensor(k))
  
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]
  flattened_dict = flatten_dict(eval_shapes)
  
  random_flax_state_dict = {}
  for key in flattened_dict:
    string_tuple = tuple([str(item) for item in key])
    random_flax_state_dict[string_tuple] = flattened_dict[key]
  
  del flattened_dict

  for pt_key, tensor in tensors.items():
      renamed_pt_key = rename_key(pt_key)
      renamed_pt_key = rename_for_ltx2(renamed_pt_key)
      
      pt_tuple_key = tuple(renamed_pt_key.split("."))
      
      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
      )
      
      # For scanned layers, flax_tensor might be the full stack updated at one index.
      # We just assign it. logic in get_key_and_value handles the update if it exists.
      # But wait, get_key_and_value retrieves existing tensor from flax_state_dict if it exists.
      # So we can just overwrite.
      
      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

  validate_flax_state_dict(eval_shapes, flax_state_dict)
  flax_state_dict = unflatten_dict(flax_state_dict)
  del tensors
  jax.clear_caches()
  return flax_state_dict
