"""
 Copyright 2024 Google LLC

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

import jax
import jax.numpy as jnp

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from maxdiffusion import max_utils

def load_sdxllightning_unet(config, pipeline, params):
  """Load lightning """
  if not config.lightning_from_pt:
    raise ValueError("Only loading lightning models from Pytorch is currently supported.")
  unet_lightning_state_dict = load_file(hf_hub_download(config.lightning_repo, config.lightning_ckpt), device="cpu")
  flax_unet_dict = convert_pytorch_state_dict_to_flax(unet_lightning_state_dict, pipeline.unet)
  params["unet"] = flax_unet_dict
  return pipeline, params

def get_dummy_unet_inputs(config, pipeline):
  vae_scale_factor = 2 ** (len(pipeline.vae.config['block_out_channels']) -1)
  batch_size = config.per_device_batch_size
  dtype=max_utils.get_dtype(config)
  input_shape = (batch_size,
                  pipeline.unet.config['in_channels'],
                  config.resolution // vae_scale_factor,
                  config.resolution // vae_scale_factor)

  latents = jax.random.normal(jax.random.PRNGKey(0),
                              shape=input_shape,
                              dtype=dtype
                              )
  timesteps = jnp.ones((latents.shape[0],))
  encoder_hidden_states_shape = (latents.shape[0],
                                  pipeline.text_encoder.config.max_position_embeddings,
                                  pipeline.unet.cross_attention_dim)
  encoder_hidden_states = jnp.zeros(encoder_hidden_states_shape)
  added_cond_kwargs = None
  if pipeline.unet.addition_embed_type == "text_time":
    unet_config = pipeline.unet.config
    is_refiner = (
      5 * unet_config.addition_time_embed_dim + unet_config.cross_attention_dim
      == unet_config.projection_class_embeddings_input_dim
    )
    num_micro_conditions = 5 if is_refiner else 6

    text_embeds_dim = unet_config.projection_class_embeddings_input_dim - (
      num_micro_conditions * unet_config.addition_time_embed_dim
    )
    time_ids_channels = pipeline.unet.projection_class_embeddings_input_dim - text_embeds_dim
    time_ids_dims = time_ids_channels // pipeline.unet.addition_time_embed_dim
    added_cond_kwargs = {
      "text_embeds": jnp.zeros((batch_size, text_embeds_dim), dtype=dtype),
      "time_ids": jnp.zeros((batch_size, time_ids_dims), dtype=dtype),
    }
  return (latents, timesteps, encoder_hidden_states, added_cond_kwargs)

def calculate_unet_tflops(config, pipeline, rngs, train):
  """Calculates unet tflops."""

  (latents, timesteps,
    encoder_hidden_states, added_cond_kwargs) = get_dummy_unet_inputs(config, pipeline)
  return max_utils.calculate_model_tflops(
    pipeline.unet,
    rngs,
    train,
    sample=latents,
    timesteps=timesteps,
    encoder_hidden_states=encoder_hidden_states,
    added_cond_kwargs=added_cond_kwargs)