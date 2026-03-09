import os
import sys

# Ensure we use the local maxdiffusion src directory so we pull from the repository
sys.path.insert(0, os.path.abspath("src"))

import jax
import jax.numpy as jnp
import numpy as np

orig_normal = jax.random.normal
video_noise = None
audio_noise = None

def load_noises():
    global video_noise, audio_noise
    video_noise_path = "video_noise.npy"
    if os.path.exists(video_noise_path):
        video_noise = np.load(video_noise_path)
    else:
        print(f"Warning: {video_noise_path} not found")
        
    audio_noise_path = "audio_noise.npy"
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
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder

pipe_module.jax.random.normal = custom_normal

def _print_stat_impl(name, t):
    if hasattr(t, "cpu"):
        t = t.detach().cpu().float().numpy()
    t_np = np.array(t, dtype=np.float32)
    print(f"[{name}] shape: {t_np.shape}, min: {t_np.min():.5f}, max: {t_np.max():.5f}, mean: {t_np.mean():.5f}, std: {t_np.std():.5f}")

def print_stat(name, t):
    if isinstance(t, jax.core.Tracer):
        jax.debug.callback(_print_stat_impl, name, t)
    else:
        _print_stat_impl(name, t)

from maxdiffusion.models.ltx2.text_encoders.feature_extractor_ltx2 import LTX2GemmaFeatureExtractor, _norm_and_concat_padded_batch

# Patch Feature Extractor
orig_fe_call = LTX2GemmaFeatureExtractor.__call__
def patched_fe_call(self, hidden_states, attention_mask):
    if isinstance(hidden_states, (tuple, list)):
        x = jnp.stack(hidden_states, axis=-1)
    else:
        x = hidden_states
    x_norm = _norm_and_concat_padded_batch(x, attention_mask)
    print("\n=== FEATURE EXTRACTOR / TEXT PROJ OUTPUTS ===")
    print_stat("packed_text_embeds", x_norm)
    out = self.linear(x_norm)
    print_stat("text_proj_out", out)
    return out
LTX2GemmaFeatureExtractor.__call__ = patched_fe_call

from maxdiffusion.models.ltx2.text_encoders.embeddings_connector_ltx2 import Embeddings1DConnector

orig_replace = Embeddings1DConnector._replace_padded_with_learnable_registers
def patched_replace(self, hidden_states, attention_mask):
    if attention_mask.ndim == 2:
        mask = attention_mask
    else:
        mask = attention_mask.squeeze(-1) # [B, T]
    curr_mask = (mask > 0.5).astype(jnp.int32)
    
    jax.debug.print("\n[MAXDIFFUSION] Mask Debug:")
    jax.debug.print("  Input Attn Mask min/max: {} / {}", jnp.min(attention_mask), jnp.max(attention_mask))
    jax.debug.print("  Curr Mask sum: {} (valid tokens)", jnp.sum(curr_mask))
    jax.debug.print("  Curr Mask start 20 elements: {}", curr_mask[0, :20])
    
    flipped = jnp.flip(curr_mask, axis=[1])
    jax.debug.print("  Flipped Mask Fwd logic sum: {} (first 20 elements: {})", jnp.sum(flipped), flipped[0, :20])

    regs = self.learnable_registers.value
    jax.debug.print("  [MAXDIFFUSION] Connector Registers std: {std}, mean: {mean}, min: {min}", 
                    std=jnp.std(regs), mean=jnp.mean(regs), min=jnp.min(regs))
    
    return orig_replace(self, hidden_states, attention_mask)
    
Embeddings1DConnector._replace_padded_with_learnable_registers = patched_replace

# Patch Transformer forward pass to intercept inputs and EXIT EARLY
orig_transformer_forward_pass = pipe_module.transformer_forward_pass
def patched_transformer_forward_pass(*args, **kwargs):
    print("\n=== TRANSFORMER INPUTS (MAXDIFFUSION) ===")
    
    # In Maxdiffusion, args are usually (hidden_states, encoder_hidden_states, timestep, ...)
    if "hidden_states" in kwargs:
         print_stat("transformer_input_video_latents", kwargs["hidden_states"])
    elif len(args) > 2 and args[2] is not None:
         print_stat("transformer_input_video_latents", args[2])
         
    if "encoder_hidden_states" in kwargs:
         print_stat("transformers_encoder_hidden_states", kwargs["encoder_hidden_states"])
    elif len(args) > 5 and args[5] is not None:
         print_stat("transformers_encoder_hidden_states", args[5])
         
    if "timestep" in kwargs:
         print_stat("transformer_timestep", kwargs["timestep"])
    elif len(args) > 4 and args[4] is not None:
         print_stat("transformer_timestep", args[4])
         
    if "audio_hidden_states" in kwargs:
         print_stat("transformer_input_audio_latents", kwargs["audio_hidden_states"])
    elif len(args) > 3 and args[3] is not None:
         print_stat("transformer_input_audio_latents", args[3])
         
    if "audio_encoder_hidden_states" in kwargs:
         print_stat("transformers_audio_encoder_hidden_states", kwargs["audio_encoder_hidden_states"])
    elif len(args) > 6 and args[6] is not None:
         print_stat("transformers_audio_encoder_hidden_states", args[6])

    print("\n[SUCCESS] Captured all inputs up to Transformer logic. Exiting early to save compute.\n")
    import os
    os._exit(0)
pipe_module.transformer_forward_pass = patched_transformer_forward_pass

def main():
    load_noises()
    
    # Init pyconfig, this assumes the user runs: python before_transformer_parity_maxdiffusion.py src/maxdiffusion/configs/ltx2_video.yml
    if len(sys.argv) < 2:
        print("Please provide the path to ltx2_video.yml")
        sys.exit(1)
        
    pyconfig.initialize(sys.argv)
    config = pyconfig.config
    
    # Create the pipeline
    pipe = LTX2Pipeline.from_pretrained(config)

    prompt = getattr(config, "prompt", "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie.")
    if not isinstance(prompt, str):
         if isinstance(prompt, (list, tuple)):
              prompt = ", ".join(str(p) for p in prompt)
         else:
              prompt = str(prompt)
    prompt = [prompt] # Pass as list to avoid pipeline encode_prompt type validation bug

    negative_prompt = getattr(config, "negative_prompt", "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.")
    if not isinstance(negative_prompt, str):
         if isinstance(negative_prompt, (list, tuple)):
              negative_prompt = ", ".join(str(p) for p in negative_prompt)
         else:
              negative_prompt = str(negative_prompt)
    negative_prompt = [negative_prompt]
    
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
