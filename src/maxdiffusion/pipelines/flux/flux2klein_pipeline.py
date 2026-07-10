# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import time
from typing import List, Union, Optional, Any
from PIL import Image

import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.linen import partitioning as nn_partitioning

from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ...models.flux.transformers.transformer_flux2klein_flax import Flux2KleinTransformer2DModel
from ...models.vae_flax import FlaxAutoencoderKL
from ...models.qwen3_flax import FlaxQwen3Model
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, compute_empirical_mu

from ...models.flux.util import (
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
    prepare_text_ids,
)

class FlaxFlux2KleinPipeline(FlaxDiffusionPipeline):
    """
    Unified end-to-end inference pipeline for Flux.2-klein-4B and 9B models on JAX+TPU.
    Supports dynamic parameter offloading to Host CPU to optimize HBM footprint.
    """

    def __init__(
        self,
        transformer: Flux2KleinTransformer2DModel,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxQwen3Model,
        tokenizer,
        scheduler: FlaxFlowMatchScheduler,
        config,
        mesh,
        **kwargs
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        self._config = config
        self.mesh = mesh
        
        # JIT compilation cache
        self._jitted_qwen3_forward = None
        self._jitted_transformer_step = None
        self._jitted_vae_decode = None

    def _setup_jit_functions(self):
        if self._jitted_qwen3_forward is not None:
            return
            
        @jax.jit
        def qwen3_forward(q_params, ids, mask):
            return self.text_encoder.apply({"params": q_params}, input_ids=ids, attention_mask=mask)
            
        @jax.jit
        def transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
            return self.transformer.apply(
                {"params": t_params},
                hidden_states=latents,
                img_ids=img_ids,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                pooled_projections=vec,
                timestep=timestep,
                guidance=guidance,
            )
            
        @jax.jit
        def vae_decode(v_params, latents_unpatched):
            return self.vae.apply({"params": v_params}, latents=latents_unpatched, method=self.vae.decode)
            
        self._jitted_qwen3_forward = qwen3_forward
        self._jitted_transformer_step = transformer_step
        self._jitted_vae_decode = vae_decode

    def _prepare_latents(self, config, batch_size, height, width):
        num_channels_latents = 32
        latent_height = height // 8
        latent_width = width // 8
        latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)

        seed_val = getattr(config, 'seed', None)
        if seed_val is None:
            seed_val = int(time.time()) & 0x7fffffff
        print(f"Generating random gaussian noise in unpacked space (32 channels) with seed: {seed_val} and shape: {latent_shape}...")
        np.random.seed(seed_val)
        latents_unpacked = np.random.randn(*latent_shape).astype(np.float32)
        
        # Pack/patchify noise exactly like PyTorch:
        # (batch, 32, H/16, 2, W/16, 2) -> permute(0, 1, 3, 5, 2, 4) -> reshape(batch, 128, H/16, W/16)
        B, C, H, W = latents_unpacked.shape
        latents_packed = latents_unpacked.reshape(B, C, H // 2, 2, W // 2, 2)
        latents_packed = np.transpose(latents_packed, (0, 1, 3, 5, 2, 4))
        latents_packed = latents_packed.reshape(B, 128, H // 2, W // 2)
        
        return latents_packed

    def __call__(
        self,
        prompt: Union[str, List[str]],
        params,
        vae_params,
        qwen3_params,
        vae_bn_mean,
        vae_bn_std,
        transformer_shardings,
        vae_shardings,
        qwen3_shardings,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        batch_size: int = 1,
        offload_encoders: bool = False,
        use_latents: bool = False,
        latents: Optional[Any] = None,
        measure_time: bool = False,
        output_dir: str = "output/",
        output_name: str = "flux2klein_generated_image.png"
    ):
        # 1. Setup JIT functions
        self._setup_jit_functions()

        # 2. Setup prompts and inputs
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        else:
            prompts = prompt
            
        seq_len_img = (height // 16) * (width // 16)
        seq_len_txt = self._config.max_sequence_length

        # Load or generate latents
        if use_latents and latents is not None:
            latents_jax = jnp.array(latents)
            if latents_jax.ndim == 4:
                B, C, H, W = latents_jax.shape
                if C == 32:
                    print("  [PIPELINE] Unpacked 32-channel latents detected. Packing using pack_latents...")
                    latents_jax = pack_latents(latents_jax)
                else:
                    latents_jax = jnp.transpose(jnp.reshape(latents_jax, (B, C, H * W)), (0, 2, 1))
        else:
            latents_numpy = self._prepare_latents(self._config, batch_size, height, width)
            print(f"DEBUG PIPELINE: latents_numpy sum = {latents_numpy.sum():.6f} | mean = {latents_numpy.mean():.6f}")
            B, C, H, W = latents_numpy.shape
            latents_jax = jnp.transpose(jnp.reshape(latents_numpy, (B, C, H * W)), (0, 2, 1))

        # RoPE position IDs
        txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
        img_ids_val = prepare_latent_image_ids(batch_size, height // 16, width // 16)

        # Scheduler
        mu = compute_empirical_mu(seq_len_img, num_inference_steps)
        scheduler_state = self.scheduler.create_state()
        sigmas_custom = jnp.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=jnp.float32)
        scheduler_state = self.scheduler.set_timesteps_ltx2(
            state=scheduler_state,
            num_inference_steps=num_inference_steps,
            shift=mu,
            sigmas=sigmas_custom,
        )

        trace = {}

        with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
            # ---------------------------------------------------------------------
            # PHASE A: Encode Prompt (Qwen3)
            # ---------------------------------------------------------------------
            print(f"[PHASE A] Encoding {len(prompts)} prompt(s) using JAX Qwen3 on TPU...")
            t0 = time.perf_counter()
            
            # Resolve tokenizer path from config
            tokenizer_path = self._config.tokenizer_model_name_or_path
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            repo_cache = os.path.join(hf_home, "hub", f"models--{self._config.pretrained_model_name_or_path.replace('/', '--')}", "snapshots")
            if os.path.exists(repo_cache) and os.listdir(repo_cache):
                tokenizer_path = os.path.join(repo_cache, os.listdir(repo_cache)[0])
                
            from transformers import Qwen2TokenizerFast
            try:
                tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
            except Exception:
                tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, subfolder="tokenizer", local_files_only=True)
            
            # Tokenize
            messages = [{"role": "user", "content": p} for p in prompts]
            # In batch execution, we format and tokenize prompts
            templated_texts = [
                tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for msg in messages
            ]
            print(f"DEBUG: templated_texts[0] = {repr(templated_texts[0])}")
            inputs = tokenizer(templated_texts, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt)
            prompt_ids = jnp.array(inputs["input_ids"])
            prompt_mask = jnp.array(inputs["attention_mask"])
            print(f"DEBUG: prompt_ids[0][:15] = {prompt_ids[0][:15]}")

            # Dynamically move Qwen3 parameters to TPU
            if offload_encoders:
                print("  Moving Qwen3 parameters to TPU HBM...")
                with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
                    qwen3_params_tpu = jax.device_put(qwen3_params, qwen3_shardings)
            else:
                qwen3_params_tpu = qwen3_params

            # Run Text Encoding
            hidden_states, all_hidden_states = self._jitted_qwen3_forward(qwen3_params_tpu, prompt_ids, prompt_mask)
            
            # Stack layers 9, 18, 27 to form prompt embeddings
            h_9 = all_hidden_states[9]
            h_18 = all_hidden_states[18]
            h_27 = all_hidden_states[27]
            out = jnp.stack([h_9, h_18, h_27], axis=1)
            # Transpose shape to [B, seq_len, 3*hidden_size]
            prompt_embeds_jax = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len_txt, -1))
            prompt_embeds_jax.block_until_ready()
            print(f"DEBUG: prompt_embeds_jax min={float(prompt_embeds_jax.min()):.4f}, max={float(prompt_embeds_jax.max()):.4f}, mean={float(prompt_embeds_jax.mean()):.4f}, sum={float(prompt_embeds_jax.sum()):.4f}")
            
            trace["prompt_encoding"] = time.perf_counter() - t0
            print(f" -> [TIMING] Prompt Encoding (Qwen3): {trace['prompt_encoding']:.4f} seconds ⏱️")

            # ---------------------------------------------------------------------
            # PHASE B: Denoising Loop (Flux Transformer)
            # ---------------------------------------------------------------------
            print(f"[PHASE B] Running {num_inference_steps}-step E2E Denoising Loop on a batch of {batch_size} images...")
            t0 = time.perf_counter()

            guidance_vec_val = None
            vec_val = None

            for step_idx in range(num_inference_steps):
                timestep = scheduler_state.timesteps[step_idx]
                t_vec = jnp.array([timestep / 1000.0] * batch_size)
                
                # Execute transformer forward pass step
                model_output = self._jitted_transformer_step(
                    params,
                    latents_jax,
                    img_ids_val,
                    prompt_embeds_jax,
                    txt_ids_val,
                    vec_val,
                    t_vec,
                    guidance_vec_val
                )
                
                # Update latents using FlowMatch step
                latents_jax = self.scheduler.step(
                    state=scheduler_state,
                    model_output=model_output.sample,
                    timestep=scheduler_state.timesteps[step_idx],
                    sample=latents_jax,
                ).prev_sample
                
                # Print progress
                sigma_val = scheduler_state.sigmas[step_idx]
                print(f" -> Step {step_idx}: Timestep = {scheduler_state.timesteps[step_idx]:.4f}, Sigma = {sigma_val:.4f}")

            latents_jax.block_until_ready()

        trace["denoise_loop"] = time.perf_counter() - t0
        print(f" -> [TIMING] Denoising Loop (Flux): {trace['denoise_loop']:.4f} seconds ⏱️")

        # ---------------------------------------------------------------------
        # PHASE C: Decode Latents (VAE Decoder)
        # ---------------------------------------------------------------------
        print("[PHASE C] Decoding final latents to RGB image using JAX VAE decoder on TPU...")
        t0 = time.perf_counter()

        # Apply Channel-wise Batch Normalization Scaling in packed sequence format (denormalize)
        vae_bn_mean_seq = vae_bn_mean.reshape(1, 1, 128)
        vae_bn_std_seq = vae_bn_std.reshape(1, 1, 128)
        latents_bn = latents_jax * vae_bn_std_seq + vae_bn_mean_seq

        # Unpack packed latents back to spatial grid
        latents_unpacked = unpack_latents(latents_bn, batch_size, 32, height, width)

        # Decode VAE latents to RGB pixels
        decoded_out = self._jitted_vae_decode(vae_params, latents_unpacked)
        # VAE output is in decoded_out.sample
        images_rgb = decoded_out.sample
        images_rgb.block_until_ready()

        trace["vae_decode"] = time.perf_counter() - t0
        print(f" -> [TIMING] VAE Decoding: {trace['vae_decode']:.4f} seconds ⏱️")

        if offload_encoders:
            print("  Releasing VAE parameters from TPU HBM...")
            del vae_params_tpu
            gc.collect()

        # ---------------------------------------------------------------------
        # POST-PROCESS: Format and Save Outputs
        # ---------------------------------------------------------------------
        print("Postprocessing and saving generated images...")
        saved_paths = []
        # Clamp pixels and scale to [0, 255]
        images_rgb = jnp.clip((images_rgb + 1.0) / 2.0, 0.0, 1.0)
        images_numpy = np.array(images_rgb)

        for b_idx in range(batch_size):
            image_np = np.array(images_numpy[b_idx] * 255.0, dtype=np.uint8)
            # Transpose channel dimension if shape is (C, H, W) instead of (H, W, C)
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
                
            img = Image.fromarray(image_np)
            
            # Formulate output filename for this batch index
            if batch_size > 1:
                batch_output_name = output_name.replace(".png", f"_b{b_idx}.png")
            else:
                batch_output_name = output_name
                
            output_png_path = os.path.join(output_dir, batch_output_name)
            img.save(output_png_path)
            print(f" -> Saved image: {output_png_path} | Prompt: '{prompts[b_idx]}'")
            saved_paths.append(output_png_path)

        return saved_paths, trace
