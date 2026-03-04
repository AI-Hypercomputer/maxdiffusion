import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2_audio import FlaxAutoencoderKLLTX2Audio
from maxdiffusion.models.ltx2.ltx2_utils import load_audio_vae_weights
import os

os.environ["JAX_PLATFORMS"] = "cpu"

config = {
    "base_channels": 128,
    "ch_mult": (1, 2, 4),
    "double_z": True,
    "dropout": 0.0,
    "in_channels": 2,
    "latent_channels": 8,
    "mel_bins": 64,
    "mel_hop_length": 160,
    "mid_block_add_attention": False,
    "norm_type": "pixel",
    "num_res_blocks": 2,
    "output_channels": 2,
    "resolution": 256,
    "sample_rate": 16000,
    "rngs": nnx.Rngs(0)
}

print("Initializing model...")
model = FlaxAutoencoderKLLTX2Audio(**config)

state = nnx.state(model)
eval_shapes = state.to_pure_dict()

print("Loading weights...")
weights = load_audio_vae_weights(
    pretrained_model_name_or_path="Lightricks/LTX-2",
    eval_shapes=eval_shapes,
    device="cpu",
    hf_download=True
)

print("\n--- RESULTS ---")
print("latents_mean in weights?", "latents_mean" in weights)
if "latents_mean" in weights:
    val = weights["latents_mean"]["value"] if "value" in weights["latents_mean"] else weights["latents_mean"]
    print("latents_mean value (first 5):", val[:5])
    print("latents_mean sum:", float(val.sum()))

print("latents_std in weights?", "latents_std" in weights)
if "latents_std" in weights:
    val = weights["latents_std"]["value"] if "value" in weights["latents_std"] else weights["latents_std"]
    print("latents_std sum:", float(val.sum()))

EOF
