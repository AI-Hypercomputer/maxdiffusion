
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax import traverse_util
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
    
    # Load without 'item' to avoid structure mismatch errors with State vs Dict
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint path {ckpt_path} does not exist.")
        return

    loaded_params = checkpointer.restore(ckpt_path)
    
    # Debug: Print structure of loaded_params
    print("Loaded params type:", type(loaded_params))
    if isinstance(loaded_params, dict):
        print("Loaded keys sample:", list(loaded_params.keys())[:5])
        # Check encoder down_blocks if present
        if 'encoder' in loaded_params and 'down_blocks' in loaded_params['encoder']:
             print("Encoder down_blocks keys:", list(loaded_params['encoder']['down_blocks'].keys()))
             first_key = next(iter(loaded_params['encoder']['down_blocks']))
             print(f"Key type: {type(first_key)}")

    # Merge back
    try:
        nnx.update(model, loaded_params)
    except KeyError as e:
        print(f"Caught KeyError during update: {e}")
        print("Attempting to fix integer keys...")
        
        def fix_keys(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    v = fix_keys(v)
                
                # Check if key is a string digit
                if isinstance(k, str) and k.isdigit():
                    new_k = int(k)
                else:
                    new_k = k
                new_d[new_k] = v
            return new_d

        fixed_params = fix_keys(loaded_params)
        print("Retrying update with fixed keys...")
        nnx.update(model, fixed_params)

    # Debug: Check Model Weights Shapes
    print("\n--- Model Weights Debug ---")
    try:
        if hasattr(model, 'encoder'):
            conv_in_kernel = model.encoder.conv_in.conv.kernel.value
            print(f"Encoder conv_in kernel shape: {conv_in_kernel.shape}")
            
            # Check first resnet
            if len(model.encoder.down_blocks) > 0:
                resnet0 = model.encoder.down_blocks[0].resnets[0]
                conv1_kernel = resnet0.conv1.conv.kernel.value
                print(f"Encoder down_blocks[0].resnets[0].conv1 kernel shape: {conv1_kernel.shape}")
    except Exception as e:
        print(f"Could not inspect weights: {e}")
    print("---------------------------\n")

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
    # Call the model
    # Note: default deterministic=True, causal=True/False depending on init
    jax_recon = model(jax_input, sample_posterior=False)
    
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
