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
  flax_key_str = [str(k) for k in flax_key]
  
  if flax_key_str[-1] in ["kernel", "weight"]:
       temp_key_str = flax_key_str[:-1] + ["scale"]
       temp_key = tuple(temp_key_str) # Tuple of strings
       
       if temp_key in random_flax_state_dict:
            flax_key_str = temp_key_str
            pass
  if "scale_shift_table" in flax_key_str:
      if flax_key_str[-1] in ["kernel", "weight"]:
           flax_key_str.pop()
  
  def replace_suffix(lst, old, new):
      if lst and lst[-1] == old:
          lst[-1] = new
      return lst

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

  if flax_key[-1] == "weight":
      parent = flax_key[-2] if len(flax_key) > 1 else ""
      grandparent = flax_key[-3] if len(flax_key) > 2 else ""
      
      should_be_kernel = False
      if "linear" in parent or "proj" in parent or "proj" in grandparent:
          should_be_kernel = True
      if "time_embed" in flax_key[0] or "cross_attn" in flax_key[0]:
          if "linear" in parent or "emb" in parent:
              should_be_kernel = True
      
      if "norm" in parent:
          should_be_kernel = False

      if should_be_kernel:
          flax_key = flax_key[:-1] + ("kernel",)

  if flax_key[-1] == "weight":
      if "norm" in flax_key[-2] or "norm" in flax_key[0]:
           flax_key = flax_key[:-1] + ("scale",)

  if flax_key[-1] == "weight" and flax_key[-2] in ["to_q", "to_k", "to_v", "to_out"]:
      flax_key = flax_key[:-1] + ("kernel",)
      
  if len(flax_key) >= 2 and flax_key[-2] == "0" and flax_key[-3] == "to_out":
      flax_key = flax_key[:-3] + ("to_out", flax_key[-1])

  flax_key_str = [str(k) for k in flax_key]
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

def load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device):
    """
    Loads weights from a sharded safetensors checkpoint.
    """
    index_file = "diffusion_pytorch_model.safetensors.index.json"
    tensors = {}
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

    for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]

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
              elif part in ["conv1", "conv2", "conv", "conv_in", "conv_out"]:
                  pt_list.append(part)
                  if i + 1 < len(pt_tuple_key) and pt_tuple_key[i+1] == "conv":
                      pass
                  elif pt_list[-1] == "conv": 
                      pass
                  elif len(pt_list) >= 2 and pt_list[-2] == "conv":
                       pass
                  elif part == "conv":
                      pass
                  else:
                      pt_list.append("conv")
              else:
                  pt_list.append(part)
          
          pt_tuple_key = tuple(pt_list)

          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
          flax_key = _tuple_str_to_int(flax_key)
          
          flax_key_str = [str(x) for x in flax_key]
          if "conv" in flax_key_str or "bias" in flax_key_str:
              pass

          if resnet_index is not None:
              if flax_key in flax_state_dict:
                  current_tensor = flax_state_dict[flax_key]
              else:
                  str_flax_key = tuple([str(x) for x in flax_key])
                  
                  if str_flax_key in random_flax_state_dict:
                       target_shape = random_flax_state_dict[str_flax_key].shape
                       current_tensor = jnp.zeros(target_shape, dtype=flax_tensor.dtype)
                  else:
                       current_tensor = flax_tensor
              
              str_flax_key = tuple([str(x) for x in flax_key])
              if str_flax_key in random_flax_state_dict:
                  current_tensor = current_tensor.at[resnet_index].set(flax_tensor)
                  flax_state_dict[flax_key] = current_tensor
              else:
                   flax_state_dict[flax_key] = flax_tensor
          else:
              flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
      filtered_eval_shapes = {}
      for k, v in flattened_eval.items():
          k_str = [str(x) for x in k]
          if "dropout" in k_str or "rngs" in k_str:
              continue
          filtered_eval_shapes[k] = v

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
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "vocoder"
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
  
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  for pt_key, tensor in tensors.items():
      key = rename_for_ltx2_vocoder(pt_key)
      parts = key.split(".")
      
      flax_key_parts = []
      for part in parts:
          if part.isdigit():
              flax_key_parts.append(int(part))
          else:
              flax_key_parts.append(part)
              
      if flax_key_parts[-1] == "weight":
          flax_key_parts[-1] = "kernel"
      
      flax_key = tuple(flax_key_parts)
      
      if flax_key[-1] == "kernel":
           if "upsamplers" in flax_key:
               tensor = tensor.transpose(2, 0, 1)
           else:
               tensor = tensor.transpose(2, 1, 0)
               
      flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)
      
  validate_flax_state_dict(eval_shapes, flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_connector(key):
    key = key.replace("video_connector", "video_embeddings_connector")
    key = key.replace("audio_connector", "audio_embeddings_connector")
    key = key.replace("text_proj_in", "feature_extractor.linear")
    
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
    subfolder: str = "connectors"
):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]

    grouped_weights = {
        "video_embeddings_connector": {},
        "audio_embeddings_connector": {}
    }
    
    for pt_key, tensor in tensors.items():
        key = rename_for_ltx2_connector(pt_key)
        
        if key.endswith(".kernel"):
            if tensor.ndim == 2:
                tensor = tensor.transpose(1, 0)
        
        if "stacked_blocks" in key:
            parts = key.split(".")
            try:
                sb_index = parts.index("stacked_blocks")
                layer_idx = int(parts[sb_index + 1])
                connector = parts[0]
                
                param_parts = parts[:sb_index+1] + parts[sb_index+2:]
                param_name = tuple(param_parts)
                
                if connector in grouped_weights:
                    if param_name not in grouped_weights[connector]:
                        grouped_weights[connector][param_name] = {}
                    grouped_weights[connector][param_name][layer_idx] = tensor
                    continue
            except (ValueError, IndexError):
                pass
                
        key_tuple = tuple(key.split("."))
        
        final_key_tuple = []
        for p in key_tuple:
             if p.isdigit(): final_key_tuple.append(int(p))
             else: final_key_tuple.append(p)
        final_key_tuple = tuple(final_key_tuple)

        flax_state_dict[final_key_tuple] = jax.device_put(tensor, device=cpu)
        
    for connector, params in grouped_weights.items():
        for param_name, layers in params.items():
            sorted_layers = sorted(layers.keys())
            stacked_tensor = jnp.stack([layers[i] for i in sorted_layers], axis=0)
            
            final_param_name = []
            for p in param_name:
                 if isinstance(p, str) and p.isdigit(): final_param_name.append(int(p))
                 else: final_param_name.append(p)
            final_param_name = tuple(final_param_name)
            
            flax_state_dict[final_param_name] = jax.device_put(stacked_tensor, device=cpu)
            
    del tensors
    jax.clear_caches()
    validate_flax_state_dict(eval_shapes, flax_state_dict)
    return unflatten_dict(flax_state_dict)

def rename_for_ltx2_audio_vae(key):
    # LTX-2 Audio VAE specific renaming
    
    # 1. Common renames
    if key.endswith(".weight"):
        key = key.replace(".weight", ".kernel")
        
    # 2. Structure renames
    # mid.block_1 -> mid_block1
    key = key.replace("mid.block_1", "mid_block1")
    key = key.replace("mid.block_2", "mid_block2")
    key = key.replace("mid.attn_1", "mid_attn") # Assumption, but safe
    
    # up.0 -> up_stages.0
    key = key.replace("up.", "up_stages.")
    # down.0 -> down_stages.0
    key = key.replace("down.", "down_stages.")
    
    # block.0 -> blocks.0
    key = key.replace("block.", "blocks.")
    
    # nin_shortcut -> conv_shortcut_layer
    key = key.replace("nin_shortcut", "conv_shortcut_layer")
    
    # In case upsample/downsample keys are just 'upsample' / 'downsample'
    # Check for CausalConv wrapping in upsample
    # If PT is 'upsample.conv.kernel' but Flax needs 'upsample.conv.conv.kernel'
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
    subfolder: str = "audio_vae"
):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    
    flattened_eval = flatten_dict(eval_shapes)
    random_flax_state_dict = {}
    for key in flattened_eval:
          string_tuple = tuple([str(item) for item in key])
          random_flax_state_dict[string_tuple] = flattened_eval[key]

    for pt_key, tensor in tensors.items():
        key = rename_for_ltx2_audio_vae(pt_key)
        
        # Determine if we need to transpose (Conv weights: OHWI -> HWIO)
        # Note: 1x1 convs might also be 4D. 
        # Standard Flax Conv: (H, W, I, O)
        # Standard PyTorch Conv: (O, I, H, W)
        
        should_transpose = False
        if key.endswith(".kernel"):
             if tensor.ndim == 4:
                 should_transpose = True
        
        if should_transpose:
            tensor = tensor.transpose(2, 3, 1, 0)
            
        # Convert key to tuple
        parts = key.split(".")
        flax_key_parts = []
        for part in parts:
            if part.isdigit():
                flax_key_parts.append(int(part))
            else:
                flax_key_parts.append(part)
        
        flax_key = tuple(flax_key_parts)
             
        # Reverse up_stages indices if present
        if "up_stages" in flax_key:
             # Find index of 'up_stages'
             try:
                 up_stages_idx = flax_key.index("up_stages")
                 # The integer index follows "up_stages"
                 if up_stages_idx + 1 < len(flax_key):
                     stage_idx = flax_key[up_stages_idx + 1]
                     if isinstance(stage_idx, int):
                         # Assuming 3 stages (0, 1, 2)
                         # Map 0 -> 2, 1 -> 1, 2 -> 0
                         new_stage_idx = 2 - stage_idx
                         if "upsample" in flax_key:
                              # print(f"DEBUG REVERSAL: {flax_key} -> stage_idx={stage_idx} -> new={new_stage_idx}")
                              pass
                         flax_key_parts[up_stages_idx + 1] = new_stage_idx
                         flax_key = tuple(flax_key_parts)
             except ValueError:
                 pass

        flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

    # Filter eval shapes to remove rngs/dropout
    filtered_eval_shapes = {}
    for k, v in flattened_eval.items():
          k_str = [str(x) for x in k]
          is_stat = False
          for ks in k_str:
              if "dropout" in ks or "rngs" in ks:
                  is_stat = True
                  break
          if is_stat:
              continue
          filtered_eval_shapes[k] = v

    validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
    return unflatten_dict(flax_state_dict)
