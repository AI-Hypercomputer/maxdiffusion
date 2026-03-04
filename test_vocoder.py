import jax
import jax.numpy as jnp
import os
import torch
from maxdiffusion.pyconfig import HyperParameters
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline

os.environ["JAX_PLATFORMS"] = "cpu"

config = HyperParameters(
    pretrained_model_name_or_path="Lightricks/LTX-Video",
    fps=24,
    use_qwix_quantization=False,
    logical_axis_rules=[],
    mesh_axes=[]
)

pipe, _ = LTX2Pipeline._load_and_init(config, None, vae_only=False, load_transformer=False)

# Dummy Mel Spectrogram conforming to Vocoder expected shape
# (Batch, Channels, Time, MelBins)
# e.g., (1, 100, 64) -> Let's check what it expects!
# The vocoder receives (Batch, Channels, Time, MelBins)
mel_specs = jnp.zeros((1, 1, 100, 64))

try:
    audio = pipe.vocoder(mel_specs)
    print("Audio shape:", audio.shape)
    
except Exception as e:
    import traceback
    traceback.print_exc()
