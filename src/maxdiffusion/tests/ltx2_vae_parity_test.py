import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add maxdiffusion/src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.scripts.convert_ltx2_vae_weights import convert_weights, ParamDict

def load_and_convert_pytorch_weights(pth_path, maxdiffusion_model):
    import torch
    print(f"Loading PyTorch state dict from {pth_path}...")
    pytorch_state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    
    print("Converting weights to MaxDiffusion format...")
    # Get the state graph from the initialized model
    _, state_graph = nnx.split(maxdiffusion_model)
    flax_state_dict = nnx.state.to_state_dict(state_graph)
    
    # Use the conversion utility
    mapped_weights, missing_keys, unexpected_keys = convert_weights(pytorch_state_dict, flax_state_dict, ParamDict())
    
    for k in missing_keys:
        print(f"Warning: {k} not found in PyTorch state dict.")
    for k in unexpected_keys:
        print(f"Warning: Unexpected key {k} in PyTorch state dict.")
        
    print(f"Mapped {len(mapped_weights)} parameters.")
    return mapped_weights

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
