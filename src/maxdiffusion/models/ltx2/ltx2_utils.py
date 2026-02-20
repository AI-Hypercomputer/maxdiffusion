import os
import json
import torch
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
from ..modeling_flax_pytorch_utils import (
    rename_key,
    rename_key_and_reshape_tensor,
    torch2jax,
    validate_flax_state_dict
)

def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)

def rename_for_ltx2_transformer(key):
    """
    Renames Diffusers LTX-2 keys to MaxDiffusion Flax LTX-2 keys.
    """
    key = key.replace("patchify_proj", "proj_in")
    key = key.replace("audio_patchify_proj", "audio_proj_in")

    if "caption_projection" in key:
        key = key.replace("caption_projection", "audio_caption_projection")

    # Handle audio_ff.net_0.proj -> audio_ff.net_0
    if "audio_ff" in key and "proj" in key:
        key = key.replace(".proj", "")

    # This line was redundant, keeping it as a no-op or removing it is fine.
    # The instruction implies it should be `return key` at the end.
    key = key.replace("transformer_blocks", "transformer_blocks")
    return key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48):
  block_index = None
  
  # Handle transformer_blocks_N produced by rename_key
  if scan_layers and len(pt_tuple_key) > 0 and "transformer_blocks_" in pt_tuple_key[0]:
      import re
      m = re.match(r"transformer_blocks_(\d+)", pt_tuple_key[0])
      if m:
          block_index = int(m.group(1))
          # Map transformer_blocks_N -> transformer_blocks
          pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[1:]

  if scan_layers:
    if "transformer_blocks" in pt_tuple_key:
       pass # Already handled above or matches standard format
      
  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers and block_index is not None:
    if "transformer_blocks" in flax_key:
        if flax_key in flax_state_dict:
            new_tensor = flax_state_dict[flax_key]
        else:
            # Initialize with correct shape (layers, ...)
            new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)
        
        new_tensor = new_tensor.at[block_index].set(flax_tensor)
        flax_tensor = new_tensor
        
  return flax_key, flax_tensor

def load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device):
    """
    Loads weights from a sharded safetensors checkpoint.
    """
    index_file = "diffusion_pytorch_model.safetensors.index.json"
    tensors = {}
    
    # Try to download index file
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
    except Exception:
        # Fallback to single file
        filename = "diffusion_pytorch_model.safetensors"
        try:
            ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
        except Exception:
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

def load_transformer_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 48,
    scan_layers: bool = True,
    subfolder: str = "transformer",
):
  device = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")
  
  with jax.default_device(device):
    # Support sharded loading
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
             
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)
    
    random_flax_state_dict = {}
    for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
    
    # DEBUG: Print keys to understand mapping
    print("DEBUG: Top 20 keys from Checkpoint (tensors):")
    for k in list(tensors.keys())[:20]:
        print(k)
        
    print("\nDEBUG: Top 20 keys from Flax Model (eval_shapes):")
    for k in list(random_flax_state_dict.keys())[:20]:
        print(k)
        
    for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = rename_for_ltx2_transformer(renamed_pt_key)
        
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
    pretrained_model_name_or_path: str, 
    eval_shapes: dict, 
    device: str, 
    hf_download: bool = True,
    subfolder: str = "vae"
):
  device = jax.local_devices(backend=device)[0]
  # VAE for LTX-2 is likely single file, but safe to use the helper if we wanted general robustness.
  # But `lightricks/LTX-2` VAE is single file.
  
  filename = "diffusion_pytorch_model.safetensors"
  try:
       ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
  except Exception:
       filename = "diffusion_pytorch_model.bin"
       ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)

  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")
  
  with jax.default_device(device):
      tensors = {}
      if filename.endswith(".safetensors"):
        with safe_open(ckpt_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = torch2jax(f.get_tensor(k))
      else:
        loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
        for k, v in loaded_state_dict.items():
            tensors[k] = torch2jax(v)
            
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_eval = flatten_dict(eval_shapes)
      
      random_flax_state_dict = {}
      for key in flattened_eval:
          string_tuple = tuple([str(item) for item in key])
          random_flax_state_dict[string_tuple] = flattened_eval[key]
            
      for pt_key, tensor in tensors.items():
          renamed_pt_key = rename_key(pt_key)
          if ".resnets." in renamed_pt_key:
             # pattern: resnets.0 -> resnets_0
             parts = renamed_pt_key.split(".")
             new_parts = []
             i = 0
             while i < len(parts):
                 if parts[i] == "resnets" and i+1 < len(parts) and parts[i+1].isdigit():
                     new_parts.append(f"resnets_{parts[i+1]}")
                     i += 2
                 else:
                     new_parts.append(parts[i])
                     i += 1
             renamed_pt_key = ".".join(new_parts)
             
          pt_tuple_key = tuple(renamed_pt_key.split("."))
          
          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
          flax_key = _tuple_str_to_int(flax_key)
          
          flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
          
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict

