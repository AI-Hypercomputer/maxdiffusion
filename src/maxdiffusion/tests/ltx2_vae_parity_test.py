
import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import orbax_utils
import orbax.checkpoint
from diffusers import AutoencoderKLLTXVideo
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion import pyconfig
from maxdiffusion import max_utils

def test_ltx2_vae_parity():
    # 1. Load PyTorch Model
    print("Loading PyTorch model...")
    pt_model = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-2", 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    pt_model.eval()

    # 2. Load Flax Model
    print("Loading Flax model...")
    # Initialize with same config as conversion
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        decoder_block_out_channels=(256, 512, 1024),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(5, 5, 5, 5),
        spatio_temporal_scaling=(True, True, True, True),
        decoder_spatio_temporal_scaling=(True, True, True),
        decoder_inject_noise=(False, False, False, False),
        upsample_factor=(2, 2, 2),
        upsample_residual=(False, False, False),
        dtype=jnp.float32,
        rngs=nnx.Rngs(0)
    )
    
    # Load checkpoint
    ckpt_path = os.path.abspath("ltx2_vae_checkpoint")
    print(f"Loading checkpoint from {ckpt_path}...")
    
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    
    # recreate split to get structure
    graphdef, state = nnx.split(model)
    params = state.filter(nnx.Param)
    
    # Load into structure
    loaded_params = checkpointer.restore(ckpt_path, item=params)
    
    # Merge back
    nnx.update(model, loaded_params)

    # 3. Create Inputs
    # Shape: (Batch, Channels, Frames, Height, Width)
    # LTX-2 uses (B, C, F, H, W) for PT
    # MaxDiffusion uses (B, F, H, W, C) for JAX
    
    B, C, F, H, W = 1, 3, 17, 64, 64 # Small input for speed (F=1 + 16 for patching?) 
    # F should be compatible with temporal patch (1) and scaling. 
    # PT model expects specific structure?
    
    torch.manual_seed(42)
    pt_input = torch.randn(B, C, F, H, W, dtype=torch.float32)
    
    # 4. Run PyTorch
    print("Running PyTorch forward pass...")
    with torch.no_grad():
        pt_output = pt_model(pt_input, sample_posterior=True).latent_dist.mode() # Compare encoded latents? Or full round trip?
        # Let's compare ENCODER first as valid middle ground, then full decode
        
        pt_enc_dist = pt_model.encode(pt_input).latent_dist
        pt_latents = pt_enc_dist.mode()
        
        pt_recon = pt_model.decode(pt_latents).sample

    # 5. Run Flax
    print("Running Flax forward pass...")
    # Convert input to JAX format: (B, F, H, W, C)
    jax_input = jnp.array(pt_input.permute(0, 2, 3, 4, 1).numpy())
    
    # Encode
    # AutoencoderKL usually returns distribution or sample
    # LTX2VideoAutoencoderKL.encode returns (params, rngs) -> but we are calling methods directly if split?
    # No, we called nnx.merge or update. model is stateful.
    
    # model.encode(sample, return_dict=False) -> (mean, logvar) ??
    # Checking implementation of encode in autoencoder_kl_ltx2.py check...
    
    rngs = nnx.Rngs(0)
    # We need to call it appropriately.
    # The class has __call__ which does encode -> decode (round trip)
    
    # Round trip match
    jax_recon = model(jax_input, sample_posterior=False, deterministic=True) # mode() equivalent?
    
    # Check encode separately if possible, but __call__ is easiest for verified end-to-end
    
    # For fair comparison with mode(), we need to tell JAX to sample mode.
    # If sample_posterior=True, it samples.
    # If sample_posterior=False, it returns mode (usually).
    
    # 6. Compare Outputs
    print("Comparing outputs...")
    
    # JAX output: (B, F, H, W, C) -> PT: (B, C, F, H, W)
    jax_recon_pt = torch.tensor(np.array(jax_recon)).permute(0, 4, 1, 2, 3)
    
    diff = (pt_recon - jax_recon_pt).abs()
    mae = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Max Difference: {max_diff}")
    
    if max_diff > 1e-3: # Loose tolerance initially
        print("❌ Parity Check FAILED")
    else:
        print("✅ Parity Check PASSED")
        
    # Also Check Encoder Latents
    print("\nComparing Encoder Latents...")
    # Flax Encode
    # model.encode returns diagonal_gaussian_distribution
    posterior = model.encode(jax_input)
    jax_latents = posterior.mode()
    
    jax_latents_pt = torch.tensor(np.array(jax_latents)).permute(0, 4, 1, 2, 3)
    diff_latents = (pt_latents - jax_latents_pt).abs()
    print(f"Latents MAE: {diff_latents.mean().item()}")
    print(f"Latents Max Diff: {diff_latents.max().item()}")

if __name__ == "__main__":
    test_ltx2_vae_parity()
