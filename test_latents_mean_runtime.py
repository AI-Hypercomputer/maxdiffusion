from maxdiffusion.pyconfig import pyconfig
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
import os
import jax

os.environ["JAX_PLATFORMS"] = "cpu"

pyconfig.initialize([None, "configs/ltx2_video.yml", "pretrained_model_name_or_path=Lightricks/LTX-2"])
config = pyconfig.config

pipe, _ = LTX2Pipeline._load_and_init(config, None, vae_only=False, load_transformer=False)

print("Video VAE latents_mean type:", type(pipe.vae.latents_mean.value))
print("Video VAE latents_mean value (first 5):\n", pipe.vae.latents_mean.value[:5])
print("Video VAE latents_mean sum:", float(pipe.vae.latents_mean.value.sum()))

print("Audio VAE latents_mean type:", type(pipe.audio_vae.latents_mean.value))
print("Audio VAE latents_mean value (first 5):\n", pipe.audio_vae.latents_mean.value[:5])
print("Audio VAE latents_mean sum:", float(pipe.audio_vae.latents_mean.value.sum()))
