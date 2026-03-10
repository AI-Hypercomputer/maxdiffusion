import os
import sys

# Ensure we use the local maxdiffusion src directory so we pull from the repository
sys.path.insert(0, os.path.abspath("src"))

# Must patch jax.random.normal before importing anything that uses it
import jax
import jax.numpy as jnp
import numpy as np

orig_normal = jax.random.normal
video_noise = None
audio_noise = None

def load_noises():
    global video_noise, audio_noise
    video_noise_path = "video_noise.npy"
    audio_noise_path = "audio_noise.npy"
    if os.path.exists(video_noise_path):
        video_noise = np.load(video_noise_path)
    else:
        print(f"Warning: {video_noise_path} not found")

    if os.path.exists(audio_noise_path):
        audio_noise = np.load(audio_noise_path)
    else:
        print(f"Warning: {audio_noise_path} not found")

def custom_normal(key, shape, dtype=None, **kwargs):
    if len(shape) == 5 and video_noise is not None:
         return jnp.array(video_noise, dtype=dtype)
    if len(shape) == 4 and audio_noise is not None:
         return jnp.array(audio_noise, dtype=dtype)
    return orig_normal(key, shape, dtype=dtype, **kwargs)

jax.random.normal = custom_normal

from maxdiffusion import pyconfig
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
import maxdiffusion.pipelines.ltx2.ltx2_pipeline as pipe_module
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2_audio import FlaxAutoencoderKLLTX2Audio
from maxdiffusion.models.ltx2.vocoder_ltx2 import LTX2Vocoder
import transformers

# Patch pipeline module's random normal if needed
pipe_module.jax.random.normal = custom_normal

def _print_stat_impl(name, t):
    if hasattr(t, "cpu"):
        t = t.detach().cpu().float().numpy()
    t_np = np.array(t, dtype=np.float32)
    print(f"[{name}] min: {t_np.min():.5f}, max: {t_np.max():.5f}, mean: {t_np.mean():.5f}, std: {t_np.std():.5f}")

def print_stat(name, t):
    if isinstance(t, jax.core.Tracer):
        jax.debug.callback(_print_stat_impl, name, t)
    else:
        _print_stat_impl(name, t)

# Patch transformer forward pass
orig_transformer_forward_pass = pipe_module.transformer_forward_pass
def patched_transformer_forward_pass(*args, **kwargs):
    noise_pred, noise_pred_audio = orig_transformer_forward_pass(*args, **kwargs)
    print_stat("transformer_video", noise_pred)
    print_stat("transformer_audio", noise_pred_audio)
    return noise_pred, noise_pred_audio
pipe_module.transformer_forward_pass = patched_transformer_forward_pass

# Patch Gemma
orig_gemma_call = transformers.Gemma3ForConditionalGeneration.forward
def patched_gemma_call(self, *args, **kwargs):
    out = orig_gemma_call(self, *args, **kwargs)
    if hasattr(out, "hidden_states") and out.hidden_states:
        print_stat("text_encoder", out.hidden_states[-1])
    elif isinstance(out, (list, tuple)):
        print_stat("text_encoder", out[0])
    return out
transformers.Gemma3ForConditionalGeneration.forward = patched_gemma_call

from maxdiffusion.models.ltx2.text_encoders.feature_extractor_ltx2 import LTX2GemmaFeatureExtractor, _norm_and_concat_padded_batch

# Patch Feature Extractor
orig_fe_call = LTX2GemmaFeatureExtractor.__call__
def patched_fe_call(self, hidden_states, attention_mask):
    if isinstance(hidden_states, (tuple, list)):
        x = jnp.stack(hidden_states, axis=-1)
    else:
        x = hidden_states
    x_norm = _norm_and_concat_padded_batch(x, attention_mask)
    print_stat("packed_text_embeds", x_norm)
    out = self.linear(x_norm)
    print_stat("text_proj_out", out)
    return out
LTX2GemmaFeatureExtractor.__call__ = patched_fe_call

# Patch VAE Decoder
orig_vae_decode = LTX2VideoAutoencoderKL.decode
def patched_vae_decode(self, *args, **kwargs):
    out = orig_vae_decode(self, *args, **kwargs)
    if isinstance(out, (tuple, list)):
         print_stat("vae_decoder", out[0])
    else:
         print_stat("vae_decoder", out)
    return out
LTX2VideoAutoencoderKL.decode = patched_vae_decode

# Patch Audio VAE Decoder
orig_audio_vae_decode = FlaxAutoencoderKLLTX2Audio.decode
def patched_audio_vae_decode(self, *args, **kwargs):
    out = orig_audio_vae_decode(self, *args, **kwargs)
    if isinstance(out, (tuple, list)):
         print_stat("audio_vae_decoder", out[0])
    else:
         print_stat("audio_vae_decoder", out)
    return out
FlaxAutoencoderKLLTX2Audio.decode = patched_audio_vae_decode

# Patch Vocoder
orig_vocoder_call = LTX2Vocoder.__call__
def patched_vocoder_call(self, *args, **kwargs):
    out = orig_vocoder_call(self, *args, **kwargs)
    print_stat("vocoder", out)
    return out
LTX2Vocoder.__call__ = patched_vocoder_call


def main():
    load_noises()

    # Init pyconfig, this assumes the user runs: python parity_ltx2_maxdiffusion.py src/maxdiffusion/configs/ltx2_video.yml
    if len(sys.argv) < 2:
        print("Please provide the path to ltx2_video.yml")
        sys.exit(1)

    pyconfig.initialize(sys.argv)
    config = pyconfig.config

    # Create the pipeline
    pipe = LTX2Pipeline.from_pretrained(config)

    prompt = getattr(config, "prompt", "A man in a brightly lit room...")
    if not isinstance(prompt, str):
         if isinstance(prompt, (list, tuple)):
              prompt = ", ".join(str(p) for p in prompt)
         else:
              prompt = str(prompt)
    prompt = [prompt] # Pass as list to avoid pipeline encode_prompt type validation bug

    negative_prompt = getattr(config, "negative_prompt", "shaky, glitchy, low quality...")
    if not isinstance(negative_prompt, str):
         if isinstance(negative_prompt, (list, tuple)):
              # Re-assemble commas if it was parsed as a list by commas
              negative_prompt = ", ".join(str(p) for p in negative_prompt)
         else:
              negative_prompt = str(negative_prompt)
    negative_prompt = [negative_prompt] # Pass as list to avoid pipeline encode_prompt type validation bug
    height = getattr(config, "height", 512)
    width = getattr(config, "width", 768)
    num_frames = getattr(config, "num_frames", 121)
    frame_rate = 24.0

    print("Running MaxDiffusion pipeline...")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=getattr(config, "num_inference_steps", 40),
        guidance_scale=getattr(config, "guidance_scale", 3.0),
        output_type="np", # ensures VAE is called
        return_dict=False
    )
    print("Done")

if __name__ == '__main__':
    main()
