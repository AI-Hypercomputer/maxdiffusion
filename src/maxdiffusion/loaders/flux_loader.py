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
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from contextlib import ExitStack
from transformers import (CLIPTokenizer, FlaxCLIPTextModel, FlaxT5EncoderModel, AutoTokenizer)

from maxdiffusion import FlaxAutoencoderKL, max_utils
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.models.flux.util import load_flow_model
from maxdiffusion.loaders.flux_lora_pipeline import FluxLoraLoaderMixin
from flax import linen as nn

class FluxLoader:
    """Loads Flux models for inference without training overhead."""

    @staticmethod
    def load(config, mesh):
        """
        Loads the Flux pipeline components and parameters.
        Returns: (pipeline_components_dict, params_dict)
        """
        rng = jax.random.key(config.seed)
        
        # 1. Load VAE
        with mesh:
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, 
                subfolder="vae", 
                from_pt=True, 
                use_safetensors=True, 
                dtype="bfloat16"
            )

            weights_init_fn = functools.partial(vae.init_weights, rng=rng)
            vae_state, vae_state_shardings = max_utils.setup_initial_state(
                model=vae,
                tx=None,
                config=config,
                mesh=mesh,
                weights_init_fn=weights_init_fn,
                model_params=vae_params,
                training=False,
            )

            # 2. Load Transformer
            flash_block_sizes = max_utils.get_flash_block_sizes(config)
            transformer = FluxTransformer2DModel.from_config(
                config.pretrained_model_name_or_path,
                subfolder="transformer",
                mesh=mesh,
                split_head_dim=config.split_head_dim,
                attention_kernel=config.attention,
                flash_block_sizes=flash_block_sizes,
                dtype=config.activations_dtype,
                weights_dtype=config.weights_dtype,
                precision=max_utils.get_precision(config),
            )

            # 3. Load Text Encoders
            clip_text_encoder = FlaxCLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="text_encoder", from_pt=True, dtype=config.weights_dtype
            )
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="tokenizer", dtype=config.weights_dtype
            )

            t5_encoder = FlaxT5EncoderModel.from_pretrained(config.t5xxl_model_name_or_path, dtype=config.weights_dtype)
            t5_tokenizer = AutoTokenizer.from_pretrained(
                config.t5xxl_model_name_or_path, max_length=config.max_sequence_length, use_fast=True
            )

            # Shard Text Encoders
            encoders_sharding = NamedSharding(mesh, P())
            partial_device_put_replicated = functools.partial(max_utils.device_put_replicated, sharding=encoders_sharding)
            
            clip_text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), clip_text_encoder.params)
            clip_text_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, clip_text_encoder.params)
            
            t5_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), t5_encoder.params)
            t5_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, t5_encoder.params)

            if config.offload_encoders:
                cpus = jax.devices("cpu")
                t5_encoder.params = jax.device_put(t5_encoder.params, device=cpus[0])

            # Load Transformer Weights
            max_utils.get_memory_allocations()
            transformer_eval_params = transformer.init_weights(
                rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=True
            )

            transformer_params = load_flow_model(config.flux_name, transformer_eval_params, "cpu")
            params = {"transformer": transformer_params}

            # Maybe Load LoRA
            lora_loader = FluxLoraLoaderMixin()
            params, lora_interceptors = FluxLoader._maybe_load_lora(config, lora_loader, params)
            transformer_params = params["transformer"]

            # Create Transformer State
            weights_init_fn = functools.partial(
                transformer.init_weights, rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=False
            )
            
            with ExitStack() as stack:
                _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
                transformer_state, transformer_state_shardings = max_utils.setup_initial_state(
                    model=transformer,
                    tx=None,
                    config=config,
                    mesh=mesh,
                    weights_init_fn=weights_init_fn,
                    model_params=None,
                    training=False,
                )
                transformer_state = transformer_state.replace(params=transformer_params)
                transformer_state = jax.device_put(transformer_state, transformer_state_shardings)
            
            max_utils.get_memory_allocations()

            components = {
                "vae": vae,
                "transformer": transformer,
                "clip_tokenizer": clip_tokenizer,
                "clip_text_encoder": clip_text_encoder,
                "t5_tokenizer": t5_tokenizer,
                "t5_encoder": t5_encoder,
                "lora_interceptors": lora_interceptors
            }
            
            states = {
                "vae": vae_state,
                "transformer": transformer_state,
            }
            
            shardings = {
                "vae": vae_state_shardings,
                "transformer": transformer_state_shardings
            }

            return components, states, shardings

    @staticmethod
    def _maybe_load_lora(config, lora_loader, params):
        def _noop_interceptor(next_fn, args, kwargs, context):
            return next_fn(*args, **kwargs)

        lora_config = config.lora_config
        interceptors = [_noop_interceptor]
        if hasattr(config, "lora_config") and len(lora_config.get("lora_model_name_or_path", [])) > 0:
            interceptors = []
            for i in range(len(lora_config["lora_model_name_or_path"])):
                params, rank, network_alphas = lora_loader.load_lora_weights(
                    config,
                    lora_config["lora_model_name_or_path"][i],
                    weight_name=lora_config["weight_name"][i],
                    params=params,
                    adapter_name=lora_config["adapter_name"][i],
                )
                interceptor = lora_loader.make_lora_interceptor(params, rank, network_alphas, lora_config["adapter_name"][i])
                interceptors.append(interceptor)
        return params, interceptors
