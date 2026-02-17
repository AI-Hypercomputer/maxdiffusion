
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import orbax_utils
import orbax.checkpoint
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL

def test_ltx2_vae_parity():
    # 1. Load Flax Model
    print("Initializing MaxDiffusion model...")
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
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint path {ckpt_path} does not exist.")
        return

    loaded_params = checkpointer.restore(ckpt_path, item=params)
    
    # Merge back
    nnx.update(model, loaded_params)

    # 3. Create Inputs
    print("Creating deterministic input...")
    # Shape: (Batch, Frames, Height, Width, Channels) for JAX
    # Using fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    B, F, H, W, C = 1, 17, 64, 64, 3
    
    jax_input = jax.random.normal(key, (B, F, H, W, C), dtype=jnp.float32)
    
    print(f"Input Shape: {jax_input.shape}")
    print(f"Input Stats: Mean={jax_input.mean():.6f}, Std={jax_input.std():.6f}, Min={jax_input.min():.6f}, Max={jax_input.max():.6f}")

    # 4. Run Flax
    print("Running Flax forward pass...")
    # model(sample, sample_posterior=False) -> should return reconstructed image
    
    # We use valid key for potential noise injection (though disabled in config)
    rngs = nnx.Rngs(0)
    
    # Call the model
    # Note: default deterministic=True, causal=True/False depending on init
    jax_recon = model(jax_input, sample_posterior=False, deterministic=True)
    
    # 5. Print Output Stats
    print("\nOutput Stats:")
    print(f"Output Shape: {jax_recon.shape}")
    print(f"Output Mean: {jax_recon.mean():.6f}")
    print(f"Output Std:  {jax_recon.std():.6f}")
    print(f"Output Min:  {jax_recon.min():.6f}")
    print(f"Output Max:  {jax_recon.max():.6f}")
    
    # Also Check Encoder Latents
    print("\nEncoder Latents Stats:")
    posterior = model.encode(jax_input)
    # posterior is DiagonalGaussianDistribution
    # Check mode
    latents = posterior.mode()
    print(f"Latents Shape: {latents.shape}")
    print(f"Latents Mean: {latents.mean():.6f}")
    print(f"Latents Std:  {latents.std():.6f}")

if __name__ == "__main__":
    test_ltx2_vae_parity()
