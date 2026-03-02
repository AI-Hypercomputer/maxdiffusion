import jax
import sys
import maxdiffusion.pyconfig as pyconfig
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline

argv = ["", "src/maxdiffusion/configs/ltx2_video.yml"]
pyconfig.initialize(argv)

pipeline = LTX2Pipeline.from_pretrained(pyconfig.config, load_transformer=False)
print("latents_mean (Video VAE):", pipeline.vae.latents_mean.value[:5])
print("latents_std (Video VAE):", pipeline.vae.latents_std.value[:5])
print("latents_mean (Audio VAE):", pipeline.audio_vae.latents_mean.value[:5])
print("latents_std (Audio VAE):", pipeline.audio_vae.latents_std.value[:5])
