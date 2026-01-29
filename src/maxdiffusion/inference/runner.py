"""
Copyright 2025 Google LLC

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

import functools
import time
import math
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from flax.linen import partitioning as nn_partitioning
from einops import rearrange
from contextlib import ExitStack
import flax.linen as nn

from maxdiffusion import max_logging
from maxdiffusion.maxdiffusion_utils import rescale_noise_cfg
from maxdiffusion.common_types import WAN2_1, WAN2_2

class DiffusionRunner:
    """Core Inference Runner executing the denoising loop on TPU."""

    def __init__(self, loaded_model, config):
        """
        Args:
            loaded_model: Dict returned by InferenceLoader.load()
            config: Configuration object
        """
        self.loaded = loaded_model
        self.config = config
        self.mesh = self.loaded["mesh"]
        self.model_name = config.model_name.lower()
        
        self.compiled_step = None

    def run(self, **kwargs):
        """
        Main entry point for generation.
        Kwargs can override config defaults (prompt, height, width, etc.)
        """
        if "wan" in self.model_name:
            return self._run_wan(**kwargs)
        elif "flux" in self.model_name:
            return self._run_flux(**kwargs)
        elif "sdxl" in self.model_name:
            return self._run_sdxl(**kwargs)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _run_wan(self, prompt=None, negative_prompt=None, **kwargs):
        # Wan Pipeline encapsulates the loop. We just call it.
        pipeline = self.loaded["pipeline"]
        
        # Merge kwargs with config
        # WanPipeline signature:
        # __call__(prompt, image=None, negative_prompt, height, width, num_frames, num_inference_steps, guidance_scale, ...)
        
        run_config = {
            "prompt": prompt or self.config.prompt,
            "negative_prompt": negative_prompt or self.config.negative_prompt,
            "height": kwargs.get("height", self.config.height),
            "width": kwargs.get("width", self.config.width),
            "num_frames": kwargs.get("num_frames", self.config.num_frames),
            "num_inference_steps": kwargs.get("num_inference_steps", self.config.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.config.guidance_scale),
        }
        
        # Load image for I2V
        if self.config.model_type == "I2V":
             # Logic to load image from config.image_url if not passed in kwargs?
             # For now assume pipeline handles it or it's passed as 'image'
             if "image" in kwargs:
                 run_config["image"] = kwargs["image"]
             elif hasattr(self.config, "image_url"):
                 from maxdiffusion.utils.loading_utils import load_image
                 run_config["image"] = load_image(self.config.image_url)

        # Wan pipeline returns frames
        return pipeline(**run_config)

    def _run_flux(self, prompt=None, **kwargs):
        # Extract components
        comps = self.loaded["pipeline"]
        states = self.loaded["states"]
        transformer = comps["transformer"]
        vae = comps["vae"]
        clip_tokenizer = comps["clip_tokenizer"]
        clip_text_encoder = comps["clip_text_encoder"]
        t5_tokenizer = comps["t5_tokenizer"]
        t5_encoder = comps["t5_encoder"]
        lora_interceptors = self.loaded["lora_interceptors"]

        prompt = prompt or self.config.prompt
        
        # Prepare inputs (from flux_utils)
        from maxdiffusion.inference.flux_utils import encode_prompt, prepare_latents, get_lin_function, time_shift, run_inference as run_inference_flux
        
        rng = jax.random.key(self.config.seed)
        global_batch_size = self.config.per_device_batch_size * jax.local_device_count()
        
        # Encoding
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            prompt=prompt,
            prompt_2=self.config.prompt_2,
            clip_tokenizer=clip_tokenizer,
            clip_text_encoder=clip_text_encoder,
            t5_tokenizer=t5_tokenizer,
            t5_text_encoder=t5_encoder,
            num_images_per_prompt=global_batch_size,
            max_sequence_length=self.config.max_sequence_length,
        )
        
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        num_channels_latents = transformer.in_channels // 4
        latents, latent_image_ids = prepare_latents(
            batch_size=global_batch_size,
            num_channels_latents=num_channels_latents,
            height=self.config.resolution,
            width=self.config.resolution,
            vae_scale_factor=vae_scale_factor,
            dtype=jnp.bfloat16,
            rng=rng,
        )
        
        guidance = jnp.asarray([self.config.guidance_scale] * global_batch_size, dtype=jnp.bfloat16)
        
        # Sharding
        data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*self.config.data_sharding))
        latents = jax.device_put(latents, data_sharding)
        latent_image_ids = jax.device_put(latent_image_ids)
        prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
        text_ids = jax.device_put(text_ids)
        guidance = jax.device_put(guidance, data_sharding)
        pooled_prompt_embeds = jax.device_put(pooled_prompt_embeds, data_sharding)
        
        # Timesteps
        timesteps = jnp.linspace(1, 0, self.config.num_inference_steps + 1)
        if self.config.time_shift:
            lin_function = get_lin_function(x1=self.config.max_sequence_length, y1=self.config.base_shift, y2=self.config.max_shift)
            mu = lin_function(latents.shape[1])
            timesteps = time_shift(mu, 1.0, timesteps)
        c_ts = timesteps[:-1]
        p_ts = timesteps[1:]

        # Compile if not ready
        if not self.compiled_step:
            
            p_run_inference = jax.jit(
                functools.partial(
                    run_inference_flux,
                    transformer=transformer,
                    vae=vae,
                    config=self.config,
                    mesh=self.mesh,
                    latents=latents,
                    latent_image_ids=latent_image_ids,
                    prompt_embeds=prompt_embeds,
                    txt_ids=text_ids,
                    vec=pooled_prompt_embeds,
                    guidance_vec=guidance,
                    c_ts=c_ts,
                    p_ts=p_ts,
                ),
                in_shardings=(self.loaded["shardings"],),
                out_shardings=None,
            )
            self.compiled_step = p_run_inference

        # Run
        with ExitStack() as stack:
             _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
             images = self.compiled_step(states).block_until_ready()
             
        # Post-process
        images = jax.experimental.multihost_utils.process_allgather(images, tiled=True)
        return self._postprocess_flux(images)

    def _postprocess_flux(self, imgs):
        imgs = np.array(imgs)
        imgs = (imgs * 0.5 + 0.5).clip(0, 1)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        imgs = np.uint8(imgs * 255)
        pil_images = [Image.fromarray(image) for image in imgs]
        return pil_images

    def _run_sdxl(self, prompt=None, **kwargs):
        pipeline = self.loaded["pipeline"]
        params = self.loaded["params"]
        states = self.loaded["states"]
        state_shardings = self.loaded["shardings"]
        lora_interceptors = self.loaded["lora_interceptors"]
        
        # Inputs from sdxl_utils
        from maxdiffusion.inference.sdxl_utils import get_unet_inputs, run_inference as run_inference_sdxl
        
        rng = jax.random.key(self.config.seed)
        global_batch_size = self.config.total_train_batch_size # Or infer from per_device * devices
        
        # Override config prompt if passed
        config_override = self.config
        if prompt:
             # Hack: modify config object or pass params?
             # Ideally get_unet_inputs should take prompt arg. 
             # For now, let's assume config is modified or we pass prompt down.
             pass

        (latents, prompt_embeds, added_cond_kwargs, guidance_scale, guidance_rescale, scheduler_state) = get_unet_inputs(
            pipeline, params, states, self.config, rng, self.mesh, global_batch_size
        )
        
        if not self.compiled_step:
            p_run_inference = jax.jit(
                functools.partial(
                    run_inference_sdxl,
                    pipeline=pipeline,
                    params=params,
                    config=self.config,
                    rng=rng,
                    mesh=self.mesh,
                    batch_size=global_batch_size,
                ),
                in_shardings=(state_shardings,),
                out_shardings=None,
            )
            self.compiled_step = p_run_inference
            
        with ExitStack() as stack:
            _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
            images = self.compiled_step(states).block_until_ready()
            
        images = jax.experimental.multihost_utils.process_allgather(images, tiled=True)
        # Post process SDXL
        from maxdiffusion.image_processor import VaeImageProcessor
        numpy_images = np.array(images)
        pil_images = VaeImageProcessor.numpy_to_pil(numpy_images)
        return pil_images
