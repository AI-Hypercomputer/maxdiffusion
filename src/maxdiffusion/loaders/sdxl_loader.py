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

import os
import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from contextlib import ExitStack
from contextlib import nullcontext

from maxdiffusion import (
    max_utils,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
    FlaxAutoencoderKL,
    max_logging,
)
from transformers import (CLIPTokenizer, FlaxCLIPTextModel, CLIPTextConfig, FlaxCLIPTextModelWithProjection)
from maxdiffusion.checkpointing.checkpointing_utils import (
    create_orbax_checkpoint_manager,
    load_stable_diffusion_configs,
    load_params_from_path
)
from maxdiffusion.maxdiffusion_utils import (
    load_sdxllightning_unet,
    maybe_load_sdxl_lora,
    create_scheduler,
)
import flax.linen as nn

STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"

class SDXLLoader:
    """Loads SDXL models for inference without training overhead."""

    @staticmethod
    def load(config, mesh):
        rng = jax.random.key(config.seed)
        
        # 1. Try Loading from Orbax
        checkpoint_manager = create_orbax_checkpoint_manager(
            config.checkpoint_dir,
            enable_checkpointing=True,
            save_interval_steps=1,
            checkpoint_type=STABLE_DIFFUSION_XL_CHECKPOINT,
            dataset_type=config.dataset_type,
        )
        
        model_configs = load_stable_diffusion_configs(config, checkpoint_manager, STABLE_DIFFUSION_XL_CHECKPOINT, None)
        
        pipeline = None
        params = {}
        
        if model_configs:
            # Reconstruct pipeline from configs
            max_logging.log("Loading SDXL from Orbax configs...")
            precision = max_utils.get_precision(config)
            flash_block_sizes = max_utils.get_flash_block_sizes(config)
            
            unet = FlaxUNet2DConditionModel.from_config(
                model_configs[0]["unet_config"],
                dtype=config.activations_dtype,
                weights_dtype=config.weights_dtype,
                from_pt=config.from_pt,
                split_head_dim=config.split_head_dim,
                norm_num_groups=config.norm_num_groups,
                attention_kernel=config.attention,
                flash_block_sizes=flash_block_sizes,
                mesh=mesh,
                precision=precision,
            )

            vae = FlaxAutoencoderKL.from_config(
                model_configs[0]["vae_config"],
                dtype=config.activations_dtype,
                weights_dtype=config.weights_dtype,
                from_pt=config.from_pt,
            )

            tokenizer_path = model_configs[0]["tokenizer_config"]["path"]
            if "gs://" in tokenizer_path:
                if "tokenizer" not in tokenizer_path:
                    tokenizer_path = os.path.join(tokenizer_path, "tokenizer")
                tokenizer_path = max_utils.download_blobs(tokenizer_path, "/tmp")
                
            tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                subfolder="tokenizer",
                dtype=config.activations_dtype,
            )

            te_pretrained_config = CLIPTextConfig(**model_configs[0]["text_encoder_config"])
            text_encoder = FlaxCLIPTextModel(
                te_pretrained_config,
                seed=config.seed,
                dtype=config.activations_dtype,
                _do_init=False,
            )
            
            te_pretrained_2_config = CLIPTextConfig(**model_configs[0]["text_encoder_2_config"])
            text_encoder_2 = FlaxCLIPTextModelWithProjection(
                te_pretrained_2_config, seed=config.seed, dtype=config.activations_dtype, _do_init=False
            )

            pipeline = FlaxStableDiffusionXLPipeline(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer, # Shared tokenizer for SDXL
                scheduler=None # Will be created later
            )
        else:
            # Fallback to Diffusers loading
            max_logging.log(f"Loading SDXL from Diffusers checkpoint: {config.pretrained_model_name_or_path}")
            pipeline, params = SDXLLoader._load_diffusers_checkpoint(config, mesh)

        # 2. Setup Scheduler
        noise_scheduler, noise_scheduler_state = create_scheduler(pipeline.scheduler.config, config)
        pipeline.scheduler = noise_scheduler
        params["scheduler"] = noise_scheduler_state

        # 3. Load UNet Params (if from Orbax specific path)
        weights_init_fn = functools.partial(pipeline.unet.init_weights, rng=rng)
        unboxed_abstract_state, _, _ = max_utils.get_abstract_state(
            pipeline.unet, None, config, mesh, weights_init_fn, False
        )
        
        unet_params = load_params_from_path(
            config, checkpoint_manager, unboxed_abstract_state.params, "unet_state"
        )
        if unet_params:
            params["unet"] = unet_params

        # 4. LoRA
        params, lora_interceptors = maybe_load_sdxl_lora(config, pipeline, params)

        # 5. Lightning
        if config.lightning_repo:
            pipeline, params = load_sdxllightning_unet(config, pipeline, params)

        # 6. Create States (Sharding)
        with ExitStack() as stack:
            _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
            unet_state, unet_state_shardings = max_utils.setup_initial_state(
                model=pipeline.unet,
                tx=None,
                config=config,
                mesh=mesh,
                weights_init_fn=weights_init_fn,
                model_params=None,
                training=False,
            )
            # Inject params
            unet_state = unet_state.replace(params=params.get("unet", None))
            unet_state = jax.device_put(unet_state, unet_state_shardings)

        # Create other states
        # VAE
        vae_weights_init_fn = functools.partial(pipeline.vae.init_weights, rng=rng)
        vae_state, vae_state_shardings = max_utils.setup_initial_state(
            model=pipeline.vae, tx=None, config=config, mesh=mesh,
            weights_init_fn=vae_weights_init_fn, model_params=params.get("vae", None), training=False
        )

        # Text Encoders
        with ExitStack() as stack:
            _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
            
            # TE 1
            te_init_fn = functools.partial(
                pipeline.text_encoder.init_weights, rng=rng,
                input_shape=(config.total_train_batch_size, pipeline.tokenizer.model_max_length)
            )
            text_encoder_state, text_encoder_state_shardings = max_utils.setup_initial_state(
                model=pipeline.text_encoder, tx=None, config=config, mesh=mesh,
                weights_init_fn=te_init_fn, model_params=params.get("text_encoder", None), training=False
            )

            # TE 2
            te2_init_fn = functools.partial(
                pipeline.text_encoder_2.init_weights, rng=rng,
                input_shape=(config.total_train_batch_size, pipeline.tokenizer.model_max_length)
            )
            text_encoder_2_state, text_encoder_2_state_shardings = max_utils.setup_initial_state(
                model=pipeline.text_encoder_2, tx=None, config=config, mesh=mesh,
                weights_init_fn=te2_init_fn, model_params=params.get("text_encoder_2", None), training=False
            )

        states = {
            "unet_state": unet_state,
            "vae_state": vae_state,
            "text_encoder_state": text_encoder_state,
            "text_encoder_2_state": text_encoder_2_state
        }
        
        state_shardings = {
            "unet_state": unet_state_shardings,
            "vae_state": vae_state_shardings,
            "text_encoder_state": text_encoder_state_shardings,
            "text_encoder_2_state": text_encoder_2_state_shardings
        }

        return pipeline, params, states, state_shardings, lora_interceptors

    @staticmethod
    def _load_diffusers_checkpoint(config, mesh):
        pipeline_class = FlaxStableDiffusionXLPipeline
        precision = max_utils.get_precision(config)
        flash_block_sizes = max_utils.get_flash_block_sizes(config)

        if jax.device_count() == jax.local_device_count():
            context = jax.default_device(jax.devices("cpu")[0])
        else:
            context = nullcontext()
            
        with context:
            pipeline, params = pipeline_class.from_pretrained(
                config.pretrained_model_name_or_path,
                revision=config.revision,
                dtype=config.activations_dtype,
                weights_dtype=config.weights_dtype,
                safety_checker=None,
                feature_extractor=None,
                from_pt=config.from_pt,
                split_head_dim=config.split_head_dim,
                norm_num_groups=config.norm_num_groups,
                attention_kernel=config.attention,
                flash_block_sizes=flash_block_sizes,
                mesh=mesh,
                precision=precision,
            )
            
            # Helper to load UNet separately if specified
            if len(config.unet_checkpoint) > 0:
                unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
                    config.unet_checkpoint,
                    split_head_dim=config.split_head_dim,
                    norm_num_groups=config.norm_num_groups,
                    attention_kernel=config.attention,
                    flash_block_sizes=flash_block_sizes,
                    dtype=config.activations_dtype,
                    weights_dtype=config.weights_dtype,
                    mesh=mesh,
                )
                params["unet"] = unet_params
                pipeline.unet = unet
            
            params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
            
        return pipeline, params
