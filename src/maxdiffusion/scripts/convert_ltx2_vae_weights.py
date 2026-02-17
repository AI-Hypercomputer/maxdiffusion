import os
import sys
# Add local diffusers to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../diffusers/src")))

import torch
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
from diffusers import AutoencoderKLLTXVideo
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion import pyconfig

def convert_ltx2_vae(hf_repo, output_path):
    # Load Diffusers model
    print(f"Loading Diffusers model from {hf_repo}...")
    pt_model = AutoencoderKLLTXVideo.from_pretrained(hf_repo, subfolder="vae")
    pt_model.eval()

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
        decoder_block_out_channels=(2048, 1024, 512, 256),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(2, 2, 6, 6, 4),
        spatio_temporal_scaling=(True, True, True, True),
        decoder_spatio_temporal_scaling=(True, True, True, True),
        decoder_inject_noise=(False, False, False, False, False),
        upsample_factor=2,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0)
    )

    # Get PyTorch state dict
    pt_state_dict = pt_model.state_dict()
    
    # Define mapping
    # We will need to map PT keys to Flax keys
    # Helper to print PT keys
    from flax import nnx
    print("PyTorch Keys:")
    sorted_pt_keys = sorted(pt_state_dict.keys())
    for k in sorted_pt_keys:
        v = pt_state_dict[k]
        print(f"{k}: {v.shape}")
    
    print("\nMaxDiffusion Keys (initialization):")
    # Get MaxDiffusion keys from initialized model
    # We need to run a dummy forward or init to get parameters if they are lazy, 
    # but nnx.Module usually has them after init if shape is provided? 
    # Wait, nnx modules need to be split to see params.
    graphdef, params = nnx.split(model, nnx.Param)
    flat_params = nnx.traverse_util.flatten_dict(params)
    sorted_flat_keys = sorted(flat_params.keys())
    for k in sorted_flat_keys:
        v = flat_params[k]
        print(f"{k}: {v.shape}")

    params = {}
    
    # TODO: Implement the mapping logic here
    # This acts as a template for now
    
    print(f"Saving converted weights to {output_path}...")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(output_path, params, save_args=save_args)
    print("Done!")

if __name__ == "__main__":
    convert_ltx2_vae("Lightricks/LTX-2", "ltx2_vae_checkpoint")
