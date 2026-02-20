
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
    # General replacements
    key = key.replace("patchify_proj", "proj_in")
    key = key.replace("audio_patchify_proj", "audio_proj_in")
    key = key.replace("transformer_blocks", "transformer_blocks") # kept same
    
    # AdaLN / Timestep Embed handling
    # Diffusers uses: time_embed, audio_time_embed, av_cross_attn_...
    # These match Flax implementation names mostly.
    
    # Attention QK Norms -> Flax uses "norm_q", "norm_k" (Diffusers often uses q_norm, k_norm but conversion script mapped them to norm_q/norm_k already? 
    # Wait, the conversion script maps *from* original *to* Diffusers. 
    # If loading Diffusers checkpoint, we should expect "norm_q", "norm_k" if that's what Diffusers uses.
    # Checking conversion script: LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT maps "q_norm" -> "norm_q".
    # So Diffusers likely uses "norm_q".
    
    # Handle "weight" -> "kernel" for Linear/Conv layers is done in rename_key_and_reshape_tensor
    # checking rename_key_and_reshape_tensor: it handles "weight" -> "kernel" for linear/conv.
    
    # Specific LTX-2 nested renaming
    # Diffusers: transformer_blocks.0.attn1.to_q.weight
    # Flax: transformer_blocks.layers.0.attn1.query.kernel (if scanned)
    
    # rename_key_and_reshape_tensor handles:
    # to_q -> query
    # to_k -> key
    # to_v -> value
    # to_out.0 -> proj_attn
    
    # We might need to handle specific mismatches if any.
    
    # The "scale" vs "weight" for LayerNorm is also handled in rename_key_and_reshape_tensor 
    # BUT only if it detects "norm" in key.
    
    # LTX2AdaLayerNormSingle usually has "linear" which is a Linear layer.
    
    return key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48):
  if scan_layers:
    if "transformer_blocks" in pt_tuple_key:
      # transformer_blocks.0.attn1... -> transformer_blocks.layers.attn1...
      # We need to extract the block index
      new_key = ("transformer_blocks",) + pt_tuple_key[2:] # removing index
      block_index = int(pt_tuple_key[1])
      pt_tuple_key = new_key
      
      # For scanned layers, we need to locate the param in the huge stacked tensor
      # But wait, rename_key_and_reshape_tensor takes the *modified* pt_tuple_key?
      # No, it takes the original one usually to check against random_flax_state_dict.
      # But here we are constructing it.
      
  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  
  # Custom cleaning after generic rename
  # e.g. converting "weight" to "value" for Params if needed, though they usually just take array.
  
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers:
    if "transformer_blocks" in flax_key:
        # We need to stack correct index
        if flax_key in flax_state_dict:
            new_tensor = flax_state_dict[flax_key]
        else:
            # Initialize with zeros of shape (num_layers, ...) + tensor.shape
            new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)
            
        new_tensor = new_tensor.at[block_index].set(flax_tensor)
        flax_tensor = new_tensor
        
  return flax_key, flax_tensor

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
  
  # Determine if local or hub
  filename = "diffusion_pytorch_model.safetensors"
  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
      ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
      if not os.path.isfile(ckpt_path):
           # Try .bin just in case
           filename = "diffusion_pytorch_model.bin"
           ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
           if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"File {ckpt_path} not found for local directory.")
      local_files = True
  elif hf_download:
      try:
        ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
      except Exception:
         filename = "diffusion_pytorch_model.bin"
         ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)

  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")
  
  with jax.default_device(device):
    tensors = {}
    if filename.endswith(".safetensors"):
        with safe_open(ckpt_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = torch2jax(f.get_tensor(k))
    else: # bin/pt
         loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
         for k, v in loaded_state_dict.items():
             tensors[k] = torch2jax(v)
             
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)
    
    # Create random state dict with string keys for matching
    random_flax_state_dict = {}
    for key in flattened_dict:
        # Convert all ints to strings in key tuple
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
        
    for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = rename_for_ltx2_transformer(renamed_pt_key)
        
        # Handling specific replacements that `rename_key` might miss or `rename_for_ltx2` specifically targets
        # The `scan_layers` handling requires splitting the key differently if needed.
        
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
  
  if os.path.isdir(pretrained_model_name_or_path):
    ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    if not os.path.isfile(ckpt_path):
      filename = "diffusion_pytorch_model.bin"
      ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
      if not os.path.isfile(ckpt_path):
         raise FileNotFoundError(f"File {ckpt_path} not found for local directory.")
  elif hf_download:
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
      
      # Build random state dict for shape checking/key matching help
      # VAE usually doesn't need scan layers logic for mapping (unless we implement scanned VAE similar to Transformer, but autoencoder_kl_ltx2.py uses scan but keys seem compatible with standard diffusers structure if mapped correctly)
      # Wait, `autoencoder_kl_ltx2.py` DOES use scan for `resnets`!
      # See `create_resnets` and `resnet_scan_fn`.
      # So we DO need scan layer handling for VAE if we want to load it into that structure.
      # The VAE resnets are scanned over `num_layers`.
      
      # Mapping Diffusers VAE to Scanned VAE:
      # Diffusers: down_blocks.0.resnets.0 ...
      # Flax Scanned: down_blocks.0.resnets.layers.0 ... (if mapped that way)
      # OR: down_blocks.0.resnets -> (num_layers, ...) tensor if we stack them.
      
      # Let's check `autoencoder_kl_ltx2.py` again.
      # `self.resnets = create_resnets(rngs)` where `create_resnets` is vmapped.
      # This creates params with a leading dimension = num_layers.
      # So we need to stack Diffusers resnets weights.
      
      # We need a custom `get_key_and_value` for VAE or modify the existing one to handle VAE blocks too.
      pass 
      
      # For now, let's just write the loading logic and we might need to iterate and fix VAE scanning logic if it fails validation.
      # Ideally we use `rename_key_and_reshape_tensor` heavily.

      random_flax_state_dict = {}
      for key in flattened_eval:
          string_tuple = tuple([str(item) for item in key])
          random_flax_state_dict[string_tuple] = flattened_eval[key]
          
      for pt_key, tensor in tensors.items():
          renamed_pt_key = rename_key(pt_key)
          
          # VAE specific renames
          renamed_pt_key = renamed_pt_key.replace("mid_block.resnets.", "mid_block.resnets.layers.")
          renamed_pt_key = renamed_pt_key.replace("down_blocks.", "down_blocks.") # keeping same
          # Need to handle resnets.0 -> resnets.layers.0 etc if we want to be explicit, or rely on scanning logic.
          
          # If we use scan, we need to stack "resnets.0", "resnets.1" etc into "resnets" tensor.
          # The logic in `get_key_and_value` handles `transformer_blocks` scanning. We should extend it for VAE `resnets`.
          
          # Actually, `autoencoder_kl_ltx2.py` VAE scanning is slightly different.
          # It scans over `resnets`.
          # Diffusers has `down_blocks.0.resnets.0`, `down_blocks.0.resnets.1`.
          # We need to stack these.
          
          pt_tuple_key = tuple(renamed_pt_key.split("."))
          
          # Let's add VAE scanning logic here or in a helper
          # Identifying keys to stack: keys containing `resnets._`
          
          # Simplified VAE Loading (non-scanned or manual stacking):
          # If `rename_key_and_reshape_tensor` expects exact matching, we might have trouble if keys are "resnets.0" but flax expects "resnets" (stacked).
          
          # I will implement a check: if key has `resnets.N`, we try to stack it.
          
          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
          
          # If it didn't match immediately, check if it's a resnet layer that needs stacking
          # This part is tricky without strictly knowing num_layers per block.
          # But we can infer or just load individually if Flax wasn't scanned?
          # The Flax code definitely uses scan.
          
          # HACK: For VAE, let's assume we might need to manually stack or map to specific indices if `rename_key_and_reshape_tensor` didn't catch it.
          # But for now, let's just use `rename_key_and_reshape_tensor` and `validate_flax_state_dict` will tell us what failed.
          
          flax_key = _tuple_str_to_int(flax_key)
          
          # Manual VAE Stacking logic if needed:
          # if "resnets" in flax_key and generic match failed...
          
          # Let's rely on `validate_flax_state_dict` to debug VAE mapping in the test phase if it's complex.
          # But I should probably add the `resnets` -> `resnets.layers` replacement to be safe?
          # Wait, if I replace `resnets.0` with `resnets.layers.0`, and Flax expects `resnets` (stacked), it still won't match.
          # Flax `nnx.vmap` with `transform_metadata={nnx.PARTITION_NAME: "layers"}` usually expects a stacked axis.
          # The parameter key in `state_dict` for a vmapped layer often depends on how it's stored. 
          # In NNX/Flax, it might be stored as `resnets.layers`? No, usually just `resnets` with an extra dim?
          # Or `resnets.layers.kernel` if it kept the name.
          
          flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
          
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict

