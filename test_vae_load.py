import jax
import sys
import maxdiffusion.pyconfig as pyconfig
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline

argv = ["", "src/maxdiffusion/configs/ltx2_video.yml"]
pyconfig.initialize(argv)

pipeline = LTX2Pipeline.from_pretrained(pyconfig.config, vae_only=True)
print("latents_mean:", pipeline.vae.latents_mean.value[:10])
print("latents_std:", pipeline.vae.latents_std.value[:10])
print("audio_latents_mean:", pipeline.audio_vae.latents_mean.value[:10])
