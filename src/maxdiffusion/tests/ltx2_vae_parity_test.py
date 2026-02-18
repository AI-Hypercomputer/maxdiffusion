"""
This is a test file used for ensuring numerical parity between pytorch and jax implementation of LTX2.
This is to be ignored and will not be pushed when commiting to main branch.
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add maxdiffusion/src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from builtins import Exception
from flax import traverse_util

def load_and_convert_pytorch_weights(pth_path, maxdiffusion_model):
    import torch
    print(f"Loading PyTorch state dict from {pth_path}...")
    pytorch_state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    
    print("Converting weights to MaxDiffusion format...")
    # Get the state graph from the initialized model
    _, state_graph = nnx.split(maxdiffusion_model)
    params = state_graph.filter(nnx.Param)
    flax_state_dict = params.to_pure_dict()
    
    # Inline conversion logic
    flat_params = traverse_util.flatten_dict(flax_state_dict)
    
    mapped_weights = {}
    missing_keys = []
    
    for key_tuple, value in flat_params.items():
        if "rngs" in key_tuple or "count" in key_tuple or "key" in key_tuple:
            continue
            
        pt_key_parts = [str(p) if isinstance(p, int) else p for p in key_tuple]
        
        # MaxDiff to PT key mapping rules
        if pt_key_parts[-1] == "kernel":
            pt_key_parts[-1] = "weight"
        elif pt_key_parts[-1] == "scale":
            pt_key_parts[-1] = "weight"
            
        pt_key = ".".join(pt_key_parts)
        
        if pt_key not in pytorch_state_dict:
            missing_keys.append(pt_key)
            continue
            
        pt_tensor = pytorch_state_dict[pt_key]
        if pt_tensor.dtype == torch.bfloat16:
            pt_tensor = pt_tensor.float()
            
        np_array = pt_tensor.numpy()
        
        # Transpose conv weights (Out, In, T, H, W) -> (T, H, W, In, Out)
        if "conv" in pt_key and "weight" in pt_key and len(np_array.shape) == 5:
             np_array = np_array.transpose(2, 3, 4, 1, 0)
        
        # Squeeze/Unsqueeze singleton logic
        if np_array.shape != value.shape:
            if np_array.shape == (1,) + value.shape:
                 np_array = np_array.squeeze(0)
            elif value.shape == (1,) + np_array.shape:
                 np_array = np_array[None]
                 
        mapped_weights[key_tuple] = jnp.array(np_array)

    for k in missing_keys:
        print(f"Warning: {k} not found in PyTorch state dict.")
        
    print(f"Mapped {len(mapped_weights)} parameters out of {len(flat_params)}.")
    params_nested = traverse_util.unflatten_dict(mapped_weights)
    return params_nested

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ltx2_parity_data"))
    
    print("Initializing MaxDiffusion LTX-2 VAE...")
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        decoder_block_out_channels=(256, 512, 1024),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(5, 5, 5, 5),
        rngs=nnx.Rngs(0)
    )
    
    # Load converted weights
    pth_path = os.path.join(data_dir, "pytorch_model.bin")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"PyTorch weights not found at {pth_path}. Run diffusers script first.")
        
    state_graph = load_and_convert_pytorch_weights(pth_path, model)
    nnx.update(model, state_graph)

    # Load inputs
    print("Loading PyTorch input...")
    pt_input = np.load(os.path.join(data_dir, "input.npy"))
    # PT Shape: (B, C, T, H, W) -> JAX Shape: (B, T, H, W, C)
    jax_input = jnp.transpose(pt_input, (0, 2, 3, 4, 1))
    
    print(f"\n--- Input ---")
    print(f"JAX Shape: {jax_input.shape}")
    print(f"Mean: {jax_input.mean():.6f}, Std: {jax_input.std():.6f}")
    
    print("\nRunning Encoder...")
    posterior = model.encode(jax_input).latent_dist
    jax_latents = posterior.mode()
    
    print(f"\n--- Encoder Latents ---")
    print(f"JAX Shape: {jax_latents.shape}")
    print(f"Mean: {jax_latents.mean():.6f}, Std: {jax_latents.std():.6f}")
    
    print("\nRunning Decoder...")
    # VAE decode output gives FlaxDecoderOutput with .sample
    jax_recon = model.decode(jax_latents).sample
    
    print(f"\n--- Decoder Output ---")
    print(f"JAX Shape: {jax_recon.shape}")
    print(f"Mean: {jax_recon.mean():.6f}, Std: {jax_recon.std():.6f}")

    # Compare with stored Diffusers outputs
    print("\n--- Parity Check ---")
    pt_latents = np.load(os.path.join(data_dir, "latents.npy"))
    pt_latents_transposed = np.transpose(pt_latents, (0, 2, 3, 4, 1))
    
    pt_recon = np.load(os.path.join(data_dir, "reconstruction.npy"))
    pt_recon_transposed = np.transpose(pt_recon, (0, 2, 3, 4, 1))
    
    latent_diff = np.abs(jax_latents - pt_latents_transposed)
    print(f"Max Latent Absolute Difference: {latent_diff.max():.8f}")
    
    recon_diff = np.abs(jax_recon - pt_recon_transposed)
    print(f"Max Reconstruction Absolute Difference: {recon_diff.max():.8f}")
    
    print("Done!")

if __name__ == "__main__":
    main()
