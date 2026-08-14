"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import gc
import json
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline

def generate_reference_images(pipe, output_dir, height=512, width=512):
    """Generates 4 distinct reference images using FLUX.2-Klein 4B text-to-image."""
    ref_prompts = [
        "a fluffy brown dog sitting happily in the grass",
        "a sleek blue sports car parked on a scenic coastal road",
        "a majestic snow-capped mountain peak under a clear blue sky",
        "a rustic wooden bowl filled with fresh colorful fruits on a table"
    ]
    ref_images = []
    os.makedirs(os.path.join(output_dir, "ref_images"), exist_ok=True)
    for idx, p in enumerate(ref_prompts):
        img_path = os.path.join(output_dir, "ref_images", f"ref_image_{idx}.png")
        if os.path.exists(img_path):
            print(f" -> Found existing reference image {idx + 1}/4 at {img_path}")
            ref_images.append(Image.open(img_path).convert("RGB"))
            continue
        print(f"🎨 Generating reference image {idx + 1}/4: '{p}'...")
        gen = torch.Generator(device="cpu").manual_seed(100 + idx)
        with torch.no_grad():
            out = pipe(
                prompt=p,
                height=height,
                width=width,
                num_inference_steps=4,
                generator=gen,
                guidance_scale=1.0,
            )
        img = out.images[0]
        img.save(img_path)
        ref_images.append(img)
        print(f" -> Saved {img_path}")
    return ref_images


def run_diffusers_image_edit_and_dump_golden(
    model_path,
    output_dir="/mnt/data/golden_image_edit_data",
    height=512,
    width=512,
    num_steps=4,
    seed=42,
):
    """Runs official Diffusers Flux2KleinPipeline on 4 reference images and captures all golden states."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading Diffusers Flux2KleinPipeline from: {model_path}...")
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cpu")

    # 1. Generate / Load 4 reference images
    ref_images = generate_reference_images(pipe, output_dir, height=height, width=width)

    edit_prompt = "a vibrant artistic painting combining the dog, car, mountain, and fruit bowl in surreal neon lighting"
    print(f"\n🚀 Running Diffusers Image Editing Ground Truth with prompt: '{edit_prompt}'...")

    with torch.no_grad():
        # 2. Extract Prompt Embeddings & txt_ids
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=edit_prompt,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27)
        )
        np.save(os.path.join(output_dir, "prompt_embeds.npy"), prompt_embeds.detach().float().cpu().numpy())
        np.save(os.path.join(output_dir, "text_ids.npy"), text_ids.detach().cpu().numpy())
        print(f" -> Saved prompt_embeds shape: {prompt_embeds.shape}, text_ids shape: {text_ids.shape}")

        # 3. Process & Encode Reference Images
        condition_images = []
        for idx, img in enumerate(ref_images):
            w, h = img.size
            multiple_of = pipe.vae_scale_factor * 2  # 16
            w = (w // multiple_of) * multiple_of
            h = (h // multiple_of) * multiple_of
            preprocessed = pipe.image_processor.preprocess(img, height=h, width=w, resize_mode="crop")
            condition_images.append(preprocessed)
            np.save(os.path.join(output_dir, f"preprocessed_image_{idx}.npy"), preprocessed.detach().float().cpu().numpy())

        # VAE Encode each reference image
        encoded_ref_latents_raw = []
        encoded_ref_latents_patchified = []
        encoded_ref_latents_norm = []
        encoded_ref_latents_packed = []

        for idx, cond_img in enumerate(condition_images):
            cond_img = cond_img.to(device="cpu", dtype=torch.bfloat16)
            raw_latents = pipe.vae.encode(cond_img).latent_dist.mode() # (1, 32, 64, 64)
            encoded_ref_latents_raw.append(raw_latents.detach().float().cpu().numpy())
            
            patchified = pipe._patchify_latents(raw_latents) # (1, 128, 32, 32)
            encoded_ref_latents_patchified.append(patchified.detach().float().cpu().numpy())
            
            bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patchified.device, patchified.dtype)
            bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
                patchified.device, patchified.dtype
            )
            norm_latent = (patchified - bn_mean) / bn_std
            encoded_ref_latents_norm.append(norm_latent.detach().float().cpu().numpy())
            
            packed = pipe._pack_latents(norm_latent).squeeze(0) # (1024, 128)
            encoded_ref_latents_packed.append(packed.detach().float().cpu().numpy())

        np.save(os.path.join(output_dir, "ref_latents_raw.npy"), np.stack(encoded_ref_latents_raw))
        np.save(os.path.join(output_dir, "ref_latents_patchified.npy"), np.stack(encoded_ref_latents_patchified))
        np.save(os.path.join(output_dir, "ref_latents_normalized.npy"), np.stack(encoded_ref_latents_norm))
        np.save(os.path.join(output_dir, "ref_latents_packed.npy"), np.stack(encoded_ref_latents_packed))

        # All ref latents concatenated along sequence dimension: (1, 4096, 128)
        image_latents_all = torch.cat([torch.tensor(p) for p in encoded_ref_latents_packed], dim=0).unsqueeze(0).to(dtype=torch.bfloat16)
        np.save(os.path.join(output_dir, "image_latents_concat.npy"), image_latents_all.detach().float().cpu().numpy())

        # Reference image position IDs:
        # 4 images, scale=10 -> T = 10, 20, 30, 40
        image_latent_ids = pipe._prepare_image_ids(
            [torch.tensor(p).to(dtype=torch.bfloat16) for p in encoded_ref_latents_norm],
            scale=10
        )
        np.save(os.path.join(output_dir, "image_latent_ids.npy"), image_latent_ids.detach().cpu().numpy())
        print(f" -> Saved image_latents_concat shape: {image_latents_all.shape}, image_latent_ids shape: {image_latent_ids.shape}")

        # 4. Prepare Initial Generation Latents
        gen_gen = torch.Generator(device="cpu").manual_seed(seed)
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=torch.bfloat16,
            device="cpu",
            generator=gen_gen,
        )
        np.save(os.path.join(output_dir, "initial_noise_latents.npy"), latents.detach().float().cpu().numpy())
        np.save(os.path.join(output_dir, "gen_latent_ids.npy"), latent_ids.detach().cpu().numpy())
        print(f" -> Saved initial_noise_latents shape: {latents.shape}, gen_latent_ids shape: {latent_ids.shape}")

        # 5. Timesteps
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = latents.shape[1]
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
        timesteps, num_steps = retrieve_timesteps(
            pipe.scheduler,
            num_steps,
            "cpu",
            sigmas=sigmas,
            mu=mu,
        )
        np.save(os.path.join(output_dir, "timesteps.npy"), np.array(timesteps))
        print(f" -> Timesteps: {timesteps}")

        # 6. Step-by-Step Denoising Loop with golden capture
        current_latents = latents.clone()
        step_noise_preds = []
        step_latents = []

        for i, t in enumerate(timesteps):
            print(f" -> Denoising step {i + 1}/{len(timesteps)} (t={t:.4f})...")
            timestep = t.expand(current_latents.shape[0]).to(current_latents.dtype)

            # Concatenate noisy latents (1024 tokens) and reference image latents (4096 tokens) -> (5120 tokens)
            latent_model_input = torch.cat([current_latents, image_latents_all], dim=1).to(pipe.transformer.dtype)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            if i == 0:
                np.save(os.path.join(output_dir, "step0_joint_latent_input.npy"), latent_model_input.detach().float().cpu().numpy())
                np.save(os.path.join(output_dir, "joint_image_ids.npy"), latent_image_ids.detach().cpu().numpy())

            noise_pred_full = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000.0,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # Slice out generation tokens only!
            noise_pred = noise_pred_full[:, :current_latents.size(1), :]
            step_noise_preds.append(noise_pred.detach().float().cpu().numpy())

            # Euler step
            current_latents = pipe.scheduler.step(noise_pred, t, current_latents, return_dict=False)[0]
            step_latents.append(current_latents.detach().float().cpu().numpy())

        np.save(os.path.join(output_dir, "step_noise_preds.npy"), np.stack(step_noise_preds))
        np.save(os.path.join(output_dir, "step_latents.npy"), np.stack(step_latents))

        # 7. Unpack & VAE Decode
        latent_height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
        unpacked_latents = pipe._unpack_latents_with_ids(
            current_latents, latent_ids, latent_height // 2, latent_width // 2
        )
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked_latents.device, unpacked_latents.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            unpacked_latents.device, unpacked_latents.dtype
        )
        unpacked_latents = unpacked_latents * bn_std + bn_mean
        unpatchified_latents = pipe._unpatchify_latents(unpacked_latents)
        np.save(os.path.join(output_dir, "final_unpatchified_latents.npy"), unpatchified_latents.detach().float().cpu().numpy())

        decoded_image = pipe.vae.decode(unpatchified_latents, return_dict=False)[0]
        pil_image = pipe.image_processor.postprocess(decoded_image, output_type="pil")[0]
        final_img_path = os.path.join(output_dir, "golden_edited_image.png")
        pil_image.save(final_img_path)
        print(f"\n🎉 Successfully finished Diffusers golden state generation!")
        print(f" -> Golden edited image saved to: {final_img_path}")


if __name__ == "__main__":
    import glob
    model_dir = None
    candidates = [
        "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        "/mnt/hyperdisk_weights/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"),
    ]
    for c in candidates:
        if os.path.exists(c):
            snaps = os.listdir(c)
            if snaps:
                model_dir = os.path.join(c, snaps[0])
                break
    if model_dir is None:
        model_dir = "black-forest-labs/FLUX.2-klein-4B"
    
    out_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data") else "golden_image_edit_data"
    run_diffusers_image_edit_and_dump_golden(model_dir, output_dir=out_dir, height=512, width=512, num_steps=4, seed=42)
