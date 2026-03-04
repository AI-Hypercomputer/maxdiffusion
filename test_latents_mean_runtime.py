from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
import os
import jax

os.environ["JAX_PLATFORMS"] = "cpu"

class MockConfig:
    pass

config = MockConfig()
config.pretrained_model_name_or_path = "Lightricks/LTX-2"
config.fps = 24
config.use_qwix_quantization = False
config.logical_axis_rules = []
config.mesh_axes = []
config.activations_dtype = "bfloat16"
config.weights_dtype = "bfloat16"

pipe, _ = LTX2Pipeline._load_and_init(config, None, vae_only=False, load_transformer=False)

print("Video VAE latents_mean type:", type(pipe.vae.latents_mean.value))
print("Video VAE latents_mean sum:", float(pipe.vae.latents_mean.value.sum()))

print("Audio VAE latents_mean type:", type(pipe.audio_vae.latents_mean.value))
print("Audio VAE latents_mean sum:", float(pipe.audio_vae.latents_mean.value.sum()))
