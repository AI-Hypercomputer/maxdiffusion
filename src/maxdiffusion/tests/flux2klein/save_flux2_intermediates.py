import os
import torch
import numpy as np
from diffusers import Flux2KleinPipeline

# Unified dictionary to hold every diagnostic array
saved_data = {}

# Global state tracking for CFG and Multi-step
current_step = 0
transformer_call_count = 0

def get_prefix():
    pass_name = "cond" if transformer_call_count == 0 else "uncond"
    return f"step_{current_step}_{pass_name}_"

# ==========================================
# 1. Load Model (Strictly CPU & Float32)
# ==========================================
print("Loading FLUX.2-klein-4B pipeline on CPU...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.float32,
)
pipe.to("cpu")

transformer = pipe.transformer
vae = pipe.vae

# ==========================================
# 2. Register Hooks for FLUX.2 Core Elements
# ==========================================
print("Registering updated hooks for FLUX.2 conditioning layouts...")

# --- Target 1) Text Embeddings Sequence ---
def hook_context_embedder(module, args, output):
    # Only save this once (it is the same across steps, but let's save it under cond)
    if "sequence_text_emb" not in saved_data:
        print("\n[HOOK DEBUG] hook_context_embedder triggered!")
        print(f"[HOOK DEBUG]   args type: {type(args)}, len: {len(args)}")
        saved_data["sequence_text_emb"] = output.detach().cpu().numpy()
        try:
            saved_data["raw_sequence_text_emb"] = args[0].detach().cpu().numpy()
            print("[HOOK DEBUG]   Successfully saved raw_sequence_text_emb to saved_data!")
        except Exception as e:
            print(f"[HOOK DEBUG]   FAILED to save raw_sequence_text_emb: {e}")

transformer.context_embedder.register_forward_hook(hook_context_embedder)

# --- Target 2) & 3) Time/Guidance Conditioning Vector ---
# Capture this per step and pass!
def hook_time_guidance_embed(module, args, output):
    prefix = get_prefix()
    saved_data[f"{prefix}joint_time_guidance_conditioning_vector"] = output.detach().cpu().numpy()

if hasattr(transformer, "time_guidance_embed"):
    transformer.time_guidance_embed.register_forward_hook(hook_time_guidance_embed)
    
    # Capture pure sinusoidal time embedding before guidance mix
    if hasattr(transformer.time_guidance_embed, "timestep_embedder"):
        def hook_timestep_embedder(module, args, output):
            prefix = get_prefix()
            saved_data[f"{prefix}pure_time_embedding"] = output.detach().cpu().numpy()
        transformer.time_guidance_embed.timestep_embedder.register_forward_hook(hook_timestep_embedder)

# --- Target 4) Global Shift/Scale Parameter Generators ---
# Temporary pre-hook to inspect transformer input shapes
def hook_inspect_inputs(module, args, kwargs):
    print("\n[INPUT INSPECT] Transformer forward inputs:")
    for k, v in kwargs.items():
        print(f"[INPUT INSPECT]   {k}: shape {v.shape if hasattr(v, 'shape') else type(v)}")
    if len(args) > 0:
        for i, v in enumerate(args):
            print(f"[INPUT INSPECT]   arg_{i}: shape {v.shape if hasattr(v, 'shape') else type(v)}")

transformer.register_forward_pre_hook(hook_inspect_inputs, with_kwargs=True)

# Capture these per step and pass!
# Capture these per step and pass!
def make_modulation_hook(name):
    return lambda m, inp, out: saved_data.update({f"{get_prefix()}{name}": out.detach().cpu().numpy()})

if hasattr(transformer, "double_stream_modulation_img"):
    transformer.double_stream_modulation_img.register_forward_hook(
        make_modulation_hook("global_double_img_modulation_params")
    )
if hasattr(transformer, "double_stream_modulation_txt"):
    transformer.double_stream_modulation_txt.register_forward_hook(
        make_modulation_hook("global_double_txt_modulation_params")
    )
if hasattr(transformer, "single_stream_modulation"):
    transformer.single_stream_modulation.register_forward_hook(
        make_modulation_hook("global_single_joint_modulation_params")
    )

# --- Target 5) Top-level Latent Entry & Exit ---
def transformer_top_pre_hook(module, args, kwargs):
    prefix = get_prefix()
    h = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
    if h is not None:
        saved_data[f"{prefix}transformer_input_latents"] = h.detach().cpu().numpy()
        
    # Capture RoPE IDs once on step 0
    if current_step == 0:
        if "txt_ids" not in saved_data:
            txt_ids = kwargs.get("txt_ids", args[4] if len(args) > 4 else None)
            if txt_ids is not None:
                saved_data["txt_ids"] = txt_ids.detach().cpu().numpy()
        if "img_ids" not in saved_data:
            img_ids = kwargs.get("img_ids", args[5] if len(args) > 5 else None)
            if img_ids is not None:
                saved_data["img_ids"] = img_ids.detach().cpu().numpy()

def transformer_top_post_hook(module, args, output):
    global transformer_call_count
    prefix = get_prefix()
    
    # Capture top-level output exiting the transformer at every step
    if isinstance(output, tuple):
        saved_data[f"{prefix}transformer_output_latents"] = output[0].detach().cpu().numpy()
    else:
        saved_data[f"{prefix}transformer_output_latents"] = output.detach().cpu().numpy()
        
    transformer_call_count += 1

transformer.register_forward_pre_hook(transformer_top_pre_hook, with_kwargs=True)
transformer.register_forward_hook(transformer_top_post_hook)


# ==========================================
# 3. Register Block-by-Block Hooks (Step 0 Only!)
# ==========================================
print("Registering deep hooks inside individual transformer layers (active on Step 0 only)...")

# --- Hook Double Stream Transformer Blocks ---
if hasattr(transformer, "transformer_blocks"):
    for i, block in enumerate(transformer.transformer_blocks):
        
        def make_double_pre_hook(block_idx):
            def pre_hook(module, args, kwargs):
                if current_step > 0:
                    return
                prefix = get_prefix()
                h = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                ctx = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
                if h is not None:
                    saved_data[f"{prefix}double_block_{block_idx}_input_image_latents"] = h.detach().cpu().numpy()
                if ctx is not None:
                    saved_data[f"{prefix}double_block_{block_idx}_input_text_latents"] = ctx.detach().cpu().numpy()
            return pre_hook
        block.register_forward_pre_hook(make_double_pre_hook(i), with_kwargs=True)

        # Capture modulated activations
        def make_double_norm1_hook(block_idx):
            def hook(m, inp, out):
                if current_step > 0:
                    return
                saved_data[f"{get_prefix()}double_block_{block_idx}_modulated_image_latents"] = out.detach().cpu().numpy()
            return hook
        if hasattr(block, "norm1"):
            block.norm1.register_forward_hook(make_double_norm1_hook(i))
            
        def make_double_norm1_context_hook(block_idx):
            def hook(m, inp, out):
                if current_step > 0:
                    return
                saved_data[f"{get_prefix()}double_block_{block_idx}_modulated_text_latents"] = out.detach().cpu().numpy()
            return hook
        if hasattr(block, "norm1_context"):
            block.norm1_context.register_forward_hook(make_double_norm1_context_hook(i))

        # Capture outputs exiting the block
        def make_double_post_hook(block_idx):
            def post_hook(module, args, output):
                if current_step > 0:
                    return
                prefix = get_prefix()
                if isinstance(output, tuple):
                    saved_data[f"{prefix}double_block_{block_idx}_output_image_latents"] = output[0].detach().cpu().numpy()
                    if len(output) > 1 and output[1] is not None:
                        saved_data[f"{prefix}double_block_{block_idx}_output_text_latents"] = output[1].detach().cpu().numpy()
            return post_hook
        block.register_forward_hook(make_double_post_hook(i))

# --- Hook Single Stream Transformer Blocks ---
if hasattr(transformer, "single_transformer_blocks"):
    for i, block in enumerate(transformer.single_transformer_blocks):
        
        def make_single_pre_hook(block_idx):
            def pre_hook(module, args, kwargs):
                if current_step > 0:
                    return
                prefix = get_prefix()
                h = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                if h is not None:
                    saved_data[f"{prefix}single_block_{block_idx}_input_latents"] = h.detach().cpu().numpy()
            return pre_hook
        block.register_forward_pre_hook(make_single_pre_hook(i), with_kwargs=True)
        
        def make_single_norm_hook(block_idx):
            def hook(m, inp, out):
                if current_step > 0:
                    return
                saved_data[f"{get_prefix()}single_block_{block_idx}_modulated_latents"] = out.detach().cpu().numpy()
            return hook
        if hasattr(block, "norm"):
            block.norm.register_forward_hook(make_single_norm_hook(i))
            
        def make_single_post_hook(block_idx):
            def hook(m, inp, out):
                if current_step > 0:
                    return
                saved_data[f"{get_prefix()}single_block_{block_idx}_output_latents"] = out.detach().cpu().numpy()
            return hook
        block.register_forward_hook(make_single_post_hook(i))


# ==========================================
# 4. Register VAE Decoder Hooks
# ==========================================
print("Registering hooks across VAE Decoder layers...")

# Monkey-patch vae.decode to reliably capture its input latents (direct method calls bypass standard module hooks)
old_decode = vae.decode
def debug_decode(latents, *args, **kwargs):
    saved_data["vae_input_unpacked_scaled_latents"] = latents.detach().cpu().numpy()
    return old_decode(latents, *args, **kwargs)
vae.decode = debug_decode

if hasattr(vae, "decoder"):
    decoder = vae.decoder
    if hasattr(decoder, "conv_in"):
        decoder.conv_in.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_conv_in_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "mid_block") and decoder.mid_block is not None:
        decoder.mid_block.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_mid_block_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "up_blocks"):
        for i, up_block in enumerate(decoder.up_blocks):
            up_block.register_forward_hook(lambda m, inp, out, idx=i: saved_data.update({f"vae_decoder_up_block_{idx}_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "conv_out"):
        decoder.conv_out.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_conv_out_output": out.detach().cpu().numpy()}))


# ==========================================
# 5. Execute Pipeline Pass (4 Steps, CFG 4.0)
# ==========================================
def callback_on_step_end(pipe, step, timestep, callback_kwargs):
    global current_step, transformer_call_count
    
    # Save the updated latents exiting this step (which is the input to the next step)
    saved_data[f"step_{step}_output_latents"] = callback_kwargs["latents"].detach().cpu().numpy()
    saved_data[f"step_{step}_timestep"] = timestep.item()
    
    # Reset call count and increment step
    transformer_call_count = 0
    current_step += 1
    
    return callback_kwargs

prompt = "A detailed vector illustration of a robotic hummingbird"
generator = torch.Generator(device="cpu").manual_seed(42)

print("Executing 4-step forward pass with guidance_scale=4.0...")
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=4,
        generator=generator,
        guidance_scale=4.0,
        callback_on_step_end=callback_on_step_end
    )

# ==========================================
# 6. Save Visual Target Image (PNG)
# ==========================================
if hasattr(output, "images") and len(output.images) > 0:
    image = output.images[0]
    target_image_path = "src/maxdiffusion/tests/flux2_klein_4step_cfg4_target.png"
    image.save(target_image_path)
    print(f"\nSaved visual target image to: {os.path.abspath(target_image_path)}")
    saved_data["output_image"] = np.array(image) / 255.0

# ==========================================
# 7. Save Bundle to Disk
# ==========================================
output_filename = "src/maxdiffusion/tests/flux2_klein_complete_diagnostic_bundle.npz"
np.savez_compressed(output_filename, **saved_data)

print(f"\n=======================================================")
print(f"SUCCESS! Harvested {len(saved_data)} clean tracking arrays across 4 steps.")
print(f"File Saved: {os.path.abspath(output_filename)}")
print(f"=======================================================")
