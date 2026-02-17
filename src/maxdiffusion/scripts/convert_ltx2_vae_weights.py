import torch
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
from diffusers import AutoencoderKLLTXVideo
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion import pyconfig
import os

def convert_ltx2_vae(hf_repo, output_path):
    # Load Diffusers model
    print(f"Loading Diffusers model from {hf_repo}...")
    pt_model = AutoencoderKLLTXVideo.from_pretrained(hf_repo, subfolder="vae")
    pt_model.eval()

    # Initialize MaxDiffusion model
    print("Initializing MaxDiffusion model...")
    config = pyconfig.initialize([None, "src/maxdiffusion/configs/base_2.yml"])
    
    # Create abstract instance to get the structure
    dummy_input = jnp.zeros((1, 9, 128, 128, 3))
    rng = jax.random.PRNGKey(0)
    
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        decoder_block_out_channels=(2048, 1024, 512, 256),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(2, 2, 6, 6, 4),
        spatio_temporal_scaling=(True, True, True, True),
        decoder_spatio_temporal_scaling=(True, True, True, True),
        decoder_inject_noise=(False, False, False, False, False),
        upsample_factor=2,
        dtype=jnp.float32
    )

    # Get PyTorch state dict
    pt_state_dict = pt_model.state_dict()
    
    # Define mapping
    # We will need to map PT keys to Flax keys
    # Helper to print PT keys
    # for k, v in pt_state_dict.items():
    #    print(k, v.shape)

    params = {}
    
    # TODO: Implement the mapping logic here
    # This acts as a template for now
    
    print(f"Saving converted weights to {output_path}...")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(output_path, params, save_args=save_args)
    print("Done!")

if __name__ == "__main__":
    convert_ltx2_vae("Lightricks/LTX-Video", "ltx2_vae_checkpoint")
