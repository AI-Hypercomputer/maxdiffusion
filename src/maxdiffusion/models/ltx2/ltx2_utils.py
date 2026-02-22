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
    key = key.replace("norm_final", "norm_out")
    
    # Handle scale_shift_table
    # PyTorch: adaLN_modulation.1.weight/bias -> scale_shift_table
    # rename_key changes adaLN_modulation.1 -> adaLN_modulation_1
    if "adaLN_modulation_1" in key:
        key = key.replace("adaLN_modulation_1", "scale_shift_table")
        
    if "caption_modulator_1" in key:
        key = key.replace("caption_modulator_1", "video_a2v_cross_attn_scale_shift_table")

    # Audio caption modulator?
    # Checkpoint: audio_caption_modulator.1.weight (Guessing name)
    # Let's inspect checkpoint keys for clues if this guess fails.
    if "audio_caption_modulator_1" in key:
        key = key.replace("audio_caption_modulator_1", "audio_a2v_cross_attn_scale_shift_table")
    
    # Handle audio_caption_projection
    # Checkpoint: audio_caption_projection.linear_1.weight
    # Flax: audio_caption_projection.linear_1.kernel
    # rename_key_and_reshape_tensor catches 'weight' -> 'kernel', but maybe something else renaming it?
    # No explicit rename needed if it's already linear_1/linear_2 unless name mismatch.
    
    # Handle global norms (norm_out, audio_norm_out)
    # Checkpoint: norm_final -> norm_out (already handled)
    # Checkpoint also has audio_norm_final -> audio_norm_out?
    if "audio_norm_final" in key:
        key = key.replace("audio_norm_final", "audio_norm_out")
    
    # Handle time_embed/audio_time_embed
    # Checkpoint: time_embed.emb.timestep_embedder.linear_1.weight
    # Flax: time_embed.emb.timestep_embedder.linear_1.kernel
    # If checkpoint uses different name structure?
    # time_embed.emb.timestep_embedder -> time_embed.emb.timestep_embedder (seems OK)
    
    # Handle av_cross_attn...
    # These seem fine in name but verify if they are Linear or Conv? Linear.


    
    # Handle autoencoder_kl_ltx2 specific renames if any, but this is for transformer usually.
    
    # Handle audio_ff.net_0.proj -> audio_ff.net_0
    # Also handle ff.net_0.proj -> ff.net_0
    if ("audio_ff" in key or "ff" in key) and "proj" in key:
        key = key.replace(".proj", "")
    
    # Handle to_out.0 -> to_out for LTX2Attention
    # rename_key changes to_out.0 -> to_out_0
    if "to_out_0" in key:
        key = key.replace("to_out_0", "to_out")
        
    return key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48):
  block_index = None
  
  # Handle transformer_blocks_N (underscore) produced by rename_key
  if scan_layers and len(pt_tuple_key) > 0 and "transformer_blocks_" in pt_tuple_key[0]:
      import re
      m = re.match(r"transformer_blocks_(\d+)", pt_tuple_key[0])
      if m:
          block_index = int(m.group(1))
          # Map transformer_blocks_N -> transformer_blocks
          pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[1:]
          
  # Handle transformer_blocks.N (dot) from original keys if rename_key didn't underscore it
  if scan_layers and len(pt_tuple_key) > 1 and pt_tuple_key[0] == "transformer_blocks" and pt_tuple_key[1].isdigit():
       block_index = int(pt_tuple_key[1])
       pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[2:]

  if scan_layers:
    if "transformer_blocks" in pt_tuple_key:
       pass # Already handled above or matches standard format
      
  # Handle scale_shift_table keys
  if "scale_shift_table" in pt_tuple_key[-1] or "scale_shift_table" in pt_tuple_key:
       pass

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  
  # Check if we got 'kernel' but expected 'scale' (common for scanned layers where shape check fails)
  # Also check 'weight' because rename_key might not have converted it to kernel if it wasn't a known Linear
  flax_key_str = [str(k) for k in flax_key]
  
  if flax_key_str[-1] in ["kernel", "weight"]:
       # Try replacing with scale and check if it exists in random_flax_state_dict
       temp_key_str = flax_key_str[:-1] + ["scale"]
       temp_key = tuple(temp_key_str) # Tuple of strings
       
       if temp_key in random_flax_state_dict:
            flax_key_str = temp_key_str
            pass

  # RESTORE LTX-2 specific keys that rename_key_and_reshape_tensor incorrectly maps to standard Flax names
  # Fix scale_shift_table mapping if it got 'kernel' appended
  if "scale_shift_table" in flax_key_str:
      # if last is kernel/weight, remove it
      if flax_key_str[-1] in ["kernel", "weight"]:
           flax_key_str.pop()
  
  # Handle audio_norm_out / norm_out bias mapping
  # If renamed to ('audio_norm_out', 'bias') matches ('audio_norm_out', 'bias') in random_flax_state_dict?
  # Yes. But if rename_key mapped it differently?
  # Ensure norm_out/audio_norm_out are preserved.
  
  # Helper to replace last occurrence
  def replace_suffix(lst, old, new):
      if lst and lst[-1] == old:
          lst[-1] = new
      return lst

  # LTX-2 uses to_q, to_k, to_v, to_out, NOT query, key, value, proj_attn
  if "transformer_blocks" in flax_key_str:
      if flax_key_str[-1] == "query":
          flax_key_str[-1] = "to_q"
      elif flax_key_str[-1] == "key":
          flax_key_str[-1] = "to_k"
      elif flax_key_str[-1] == "value":
          flax_key_str[-1] = "to_v"
      
      if len(flax_key_str) >= 2 and flax_key_str[-2] == "proj_attn":
           # proj_attn, kernel -> to_out, kernel
           flax_key_str[-2] = "to_out"

  flax_key = tuple(flax_key_str)
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

    print("DEBUG: NON-BLOCK keys in Checkpoint:")
    for k in tensors.keys():
        if "transformer_blocks" not in k:
            print(k)
        
    print("\nDEBUG: Top 20 keys from Flax Model (eval_shapes):")
    for k in list(random_flax_state_dict.keys())[:20]:
        print(k)

    print("\nDEBUG: Transformer Block keys from Flax Model (eval_shapes):")
    for k in list(random_flax_state_dict.keys()):
        k_str = str(k)
        if "transformer_blocks" in k_str and ("attn1" in k_str or "ff" in k_str):
             print(f"EVAL_SHAPE: {k}")
        
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
      
      # DEBUG: Print keys to understand mapping
      print("DEBUG: Top 20 keys from VAE Checkpoint (tensors):")
      for k in list(tensors.keys())[:20]:
          print(k)
            
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_eval = flatten_dict(eval_shapes)
      
      random_flax_state_dict = {}
      for key in flattened_eval:
          string_tuple = tuple([str(item) for item in key])
          random_flax_state_dict[string_tuple] = flattened_eval[key]
            
      for pt_key, tensor in tensors.items():
          renamed_pt_key = rename_key(pt_key)
          
          pt_tuple_key = tuple(renamed_pt_key.split("."))
          
          pt_list = []
          resnet_index = None
          
          for i, part in enumerate(pt_tuple_key):
              # Check for name_N pattern
              if "_" in part and part.split("_")[-1].isdigit():
                  name = "_".join(part.split("_")[:-1])
                  idx = int(part.split("_")[-1])
                  
                  if name == "resnets":
                      resnet_index = idx
                      pt_list.append("resnets")
                  elif name == "upsamplers":
                      pt_list.append("upsampler")
                      # Skip the index 0 for upsampler as Flax uses singular non-list
                  elif name in ["down_blocks", "up_blocks", "downsamplers"]:
                      pt_list.append(name)
                      pt_list.append(str(idx))
                  else:
                      pt_list.append(part)
              elif part == "upsampler":
                  pt_list.append("upsampler") 
              elif part in ["conv1", "conv2", "conv"]:
                  pt_list.append(part)
                  # Inject 'conv' if it's not already there AND not just added
                  if i + 1 < len(pt_tuple_key) and pt_tuple_key[i+1] == "conv":
                      pass # already has conv
                  elif pt_list[-1] == "conv": 
                      pass # already has conv
                  elif len(pt_list) >= 2 and pt_list[-2] == "conv":
                       pass
                  elif part == "conv":
                      pass
                  else:
                      # If part is conv1/conv2, append 'conv'
                      pt_list.append("conv")
              else:
                  pt_list.append(part)
          
          pt_tuple_key = tuple(pt_list)

          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
          # _tuple_str_to_int might not be needed if we already injected ints, but it's safe
          flax_key = _tuple_str_to_int(flax_key)
          
          # Allow latents_mean/std
          
          # DEBUG
          flax_key_str = [str(x) for x in flax_key]
          if "conv" in flax_key_str or "bias" in flax_key_str:
              # print(f"DEBUG: VAE Key Map: {pt_tuple_key} -> {flax_key}")
              pass

          if resnet_index is not None:
              if flax_key in flax_state_dict:
                  current_tensor = flax_state_dict[flax_key]
              else:
                  # Initialize with correct shape from random_flax_state_dict
                  # We must use STRING tuple for lookup in random_flax_state_dict
                  str_flax_key = tuple([str(x) for x in flax_key])
                  
                  if str_flax_key in random_flax_state_dict:
                       target_shape = random_flax_state_dict[str_flax_key].shape
                       current_tensor = jnp.zeros(target_shape, dtype=flax_tensor.dtype)
                  else:
                       # Fallback if key missing (shouldn't happen with correct mapping)
                       # print(f"Warning: Key {str_flax_key} not found in random_flax_state_dict, cannot stack.")
                       current_tensor = flax_tensor # Might fail shape check later
              
              # Place the tensor at the correct index
              # flax_tensor is (..., C), target is (N_resnets, ..., C)
              
              str_flax_key = tuple([str(x) for x in flax_key])
              if str_flax_key in random_flax_state_dict: # Only stack if we have a valid target
                  current_tensor = current_tensor.at[resnet_index].set(flax_tensor)
                  flax_state_dict[flax_key] = current_tensor
              else:
                   flax_state_dict[flax_key] = flax_tensor
          else:
              flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
          
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict

