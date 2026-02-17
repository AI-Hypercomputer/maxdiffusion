import os
import sys
import torch
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion import pyconfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from flax import nnx
from flax import traverse_util
import shutil

def convert_ltx2_vae(hf_repo, output_path):
    # Ensure output path is absolute
    output_path = os.path.abspath(output_path)
    
    # Load weights directly from Safetensors
    print(f"Downloading/Loading weights from {hf_repo}...")
    try:
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename="vae/diffusion_pytorch_model.safetensors")
    except Exception:
        # Fallback for if it's in the root or named differently
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename="diffusion_pytorch_model.safetensors")
    
    print(f"Loading weights from {ckpt_path}...")
    pt_state_dict = load_file(ckpt_path)

    # Initialize MaxDiffusion model
    print("Initializing MaxDiffusion model...")
    config = pyconfig.initialize([None, "src/maxdiffusion/configs/ltx2_video.yml"])
    
    # Create abstract instance to get the structure
    dummy_input = jnp.zeros((1, 9, 128, 128, 3))
    rng = jax.random.PRNGKey(0)
    
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        # Corrected Decoder Config based on PyTorch weights
        decoder_block_out_channels=(256, 512, 1024), # 3 blocks
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(5, 5, 5, 5), # Mid + 3 Up, 5 layers each
        spatio_temporal_scaling=(True, True, True, True),
        decoder_spatio_temporal_scaling=(True, True, True),
        decoder_inject_noise=(False, False, False, False),
        upsample_factor=(2, 2, 2),
        upsample_residual=(False, False, False),
        dtype=jnp.float32,
        rngs=nnx.Rngs(0)
    )
    
    print("\nMapping weights...")
    graphdef, state = nnx.split(model)
    params = state.filter(nnx.Param)
    
    params_dict = params.to_pure_dict()
    flat_params = traverse_util.flatten_dict(params_dict)
    
    new_params = {}
    
    mapped_count = 0
    for key_tuple, value in flat_params.items():
        # Skip Rngs if any leak through
        if "rngs" in key_tuple or "count" in key_tuple or "key" in key_tuple:
            continue
            
        # Construct PyTorch key
        pt_key_parts = []
        for p in key_tuple:
            if isinstance(p, int):
                pt_key_parts.append(str(p))
            else:
                pt_key_parts.append(p)
        
        # Adjust property names from MaxDiffusion to Diffusers
        if pt_key_parts[-1] == "kernel":
            pt_key_parts[-1] = "weight"
        elif pt_key_parts[-1] == "scale":
            pt_key_parts[-1] = "weight"
            
        pt_key = ".".join(pt_key_parts)
        
        # Check if key exists in PT dict
        if pt_key not in pt_state_dict:
            # Check for specific mismatches
            # Example: MaxDiffusion uses 'scale' for RMSNorm, PT uses 'weight'
            # (Handled above)
            print(f"Warning: {pt_key} not found in PyTorch state dict.")
            continue
            
        pt_tensor = pt_state_dict[pt_key]
        
        # Handle BFloat16
        if pt_tensor.dtype == torch.bfloat16:
            pt_tensor = pt_tensor.float()
            
        np_array = pt_tensor.numpy()
        
        # Handle shape mismatch (Transpose Conv3d weights)
        is_conv_weight = "conv" in pt_key and "weight" in pt_key and len(np_array.shape) == 5
        
        if is_conv_weight:
             # PyTorch Conv3d: (Out, In, T, H, W)
             # JAX Conv: (T, H, W, In, Out)
             # Permutation: 0, 1, 2, 3, 4 -> 2, 3, 4, 1, 0
             np_array = np_array.transpose(2, 3, 4, 1, 0)
        
        # Verify shape
        if np_array.shape != value.shape:
            # Handle singleton dimensions if they match in total size or one dim
            if np_array.shape == (1,) + value.shape:
                 np_array = np_array.squeeze(0)
            elif value.shape == (1,) + np_array.shape:
                 np_array = np_array[None]
            else:
                 print(f"Shape mismatch for {pt_key}: PT {np_array.shape} vs Max {value.shape}")
                 continue

        new_params[key_tuple] = jnp.array(np_array)
        mapped_count += 1

    print(f"Mapped {mapped_count} out of {len(flat_params)} parameters.")

    # Reconstruct nested dictionary
    params_nested = traverse_util.unflatten_dict(new_params)
    
    # Save checkpoint
    print(f"Saving converted weights to {output_path}...")
    
    if os.path.exists(output_path):
        print(f"Removing existing checkpoint at {output_path}...")
        shutil.rmtree(output_path)
        
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    save_args = orbax_utils.save_args_from_target(params_nested)
    checkpointer.save(output_path, params_nested, save_args=save_args)
    print("Done!")

if __name__ == "__main__":
    convert_ltx2_vae("Lightricks/LTX-2", "ltx2_vae_checkpoint")
