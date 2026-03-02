import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from diffusers import LTXPipeline
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.ltx2_utils import load_vae_weights

def prepare_maxdiffusion_vae():
    # Load PyTorch original model
    print("Loading PyTorch VAE...")
    pt_pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
    pt_vae = pt_pipe.vae.to("cpu")

    # Load MaxDiffusion VAE
    print("Loading MaxDiffusion VAE...")
    jax_vae = LTX2VideoAutoencoderKL(
        in_channels=3, out_channels=3, 
        latent_channels=128, 
        block_out_channels=(128, 256, 512, 512),
        decoder_block_out_channels=(128, 256, 512, 512),
        layers_per_block=(4, 3, 3, 3, 4),
        decoder_layers_per_block=(4, 3, 3, 3, 4),
        spatio_temporal_scaling=(True, True, True, False),
        decoder_spatio_temporal_scaling=(True, True, True, False),
        patch_size=4, patch_size_t=1,
        dtype=jnp.float32,
        encoder_causal=True, decoder_causal=False,
        rngs=nnx.Rngs(params=0)
    )

    print("Loading weights from PyTorch into MaxDiffusion...")
    state_dict = pt_vae.state_dict()
    # Apply weight conversion
    flax_dict = load_vae_weights(state_dict)
    
    # Check if there are missing or extra keys
    from maxdiffusion.models.ltx2.ltx2_utils import validate_flax_state_dict
    validate_flax_state_dict(jax_vae, flax_dict)
    
    from flax import nnx
    nnx.update(jax_vae, flax_dict)
    
    return pt_vae, jax_vae

def test_vae_parity():
    pt_vae, jax_vae = prepare_maxdiffusion_vae()

    B, C, T, H, W = 1, 3, 9, 32, 32
    print(f"\n=======================")
    print(f"Testing input shape: [B={B}, C={C}, T={T}, H={H}, W={W}]")

    # Generate random pixels, normalized typical format
    torch.manual_seed(42)
    pixel_input_pt = torch.randn(B, C, T, H, W, dtype=torch.float32)
    pixel_input_jax = jnp.array(pixel_input_pt.numpy()).transpose(0, 2, 3, 4, 1)  # B, T, H, W, C

    # 1. Test Encoder
    print("Testing Encoder...")
    with torch.no_grad():
        pt_latent_dist = pt_vae.encode(pixel_input_pt).latent_dist
        pt_latent = pt_latent_dist.mode()

    jax_latent_dist = jax_vae.encode(pixel_input_jax)
    jax_latent = jax_latent_dist.mean

    # Convert JAX latent (B, T', H', W', C) back to (B, C, T', H', W') for comparison
    jax_latent_converted = torch.from_numpy(np.array(jax_latent).transpose(0, 4, 1, 2, 3))
    
    encoder_diff = torch.abs(pt_latent - jax_latent_converted)
    max_enc_diff = encoder_diff.max().item()
    mean_enc_diff = encoder_diff.mean().item()
    print(f"Encoder Parity -> Max Diff: {max_enc_diff:.6f}, Mean Diff: {mean_enc_diff:.6f}")
    if max_enc_diff > 1e-4:
        print("-> ENCODER FAILED PARITY CHECK.")
        # Find exactly where the difference is large
        mask = encoder_diff > 1e-3
        print(f"Number of mismatches > 1e-3: {mask.sum().item()} out of {mask.numel()}")
        
    # 2. Test Decoder (Decoding the exact same PT latents)
    print("\nTesting Decoder...")
    with torch.no_grad():
        pt_decoded = pt_vae.decode(pt_latent).sample

    # Pass the PYTORCH latents into JAX decoder to isolate decoder bugs
    pt_latent_for_jax = jnp.array(pt_latent.numpy()).transpose(0, 2, 3, 4, 1)
    jax_decoded = jax_vae.decode(pt_latent_for_jax).sample

    # Convert JAX decoded (B, T, H, W, C) back to (B, C, T, H, W) for comparison
    jax_decoded_converted = torch.from_numpy(np.array(jax_decoded).transpose(0, 4, 1, 2, 3))

    decoder_diff = torch.abs(pt_decoded - jax_decoded_converted)
    max_dec_diff = decoder_diff.max().item()
    mean_dec_diff = decoder_diff.mean().item()
    print(f"Decoder Parity -> Max Diff: {max_dec_diff:.6f}, Mean Diff: {mean_dec_diff:.6f}")
    if max_dec_diff > 1e-4:
        print("-> DECODER FAILED PARITY CHECK.")

if __name__ == "__main__":
    test_vae_parity()
