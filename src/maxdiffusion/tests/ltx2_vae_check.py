
import os
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL

def print_stats(name, x):
    if isinstance(x, jax.Array):
        x = np.array(x)
    print(f"{name} stats:")
    print(f"  Shape: {x.shape}")
    print(f"  Min: {np.min(x):.4f}")
    print(f"  Max: {np.max(x):.4f}")
    print(f"  Mean: {np.mean(x):.4f}")
    print(f"  Std: {np.std(x):.4f}")
    print("-" * 20)

def main():
    print("Initializing LTX2VideoAutoencoderKL...")
    rngs = nnx.Rngs(0)
    
    # Initialize model with default config or specific for testing
    # Using small config to ensure it runs quickly/fits in memory if running locally
    # But user might want real config. Default is huge?
    # Default block_out_channels is (256, 512, 1024, 2048). That's big.
    # I'll use default to be safe on correctness vs parity, but maybe small input.
    
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        rngs=rngs
    )
    
    # Create dummy input: (B, T, H, W, C)
    # LTX2 expects channel last in MaxDiffusion implementation
    B, T, H, W, C = 1, 8, 128, 128, 3
    print(f"Generating random input with shape {(B, T, H, W, C)}...")
    
    key = jax.random.PRNGKey(42)
    key_input, key_model = jax.random.split(key)
    
    # Input in range [-1, 1] usually for VAEs? or [0, 1]? 
    # Diffusers VAEs usually expect [-1, 1].
    sample = jax.random.uniform(key_input, (B, T, H, W, C), minval=-1.0, maxval=1.0)
    
    print_stats("Input sample", sample)
    
    # Run Encode
    print("Running Encode...")
    # encode returns FlaxAutoencoderKLOutput which has .latent_dist
    posterior = model.encode(sample, return_dict=True, key=key_model).latent_dist
    
    # Sample from posterior
    print("Sampling from posterior...")
    key_sample = jax.random.fold_in(key_model, 1)
    latents = posterior.sample(key=key_sample)
    
    print_stats("Latents", latents)
    
    # Run Decode
    print("Running Decode...")
    # decode returns FlaxDecoderOutput which has .sample
    decoded = model.decode(latents, return_dict=True, generator=key_model).sample
    
    print_stats("Decoded sample", decoded)
    
    print("Done.")

if __name__ == "__main__":
    main()
