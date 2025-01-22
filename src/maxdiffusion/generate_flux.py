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

from typing import Any, Callable, Dict, List, Optional, Union, Sequence
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from chex import Array
from transformers import (
  CLIPTokenizer,
  FlaxCLIPTextModel,
  T5TokenizerFast,
  T5EncoderModel
)

from maxdiffusion import FlaxAutoencoderKL
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel

from maxdiffusion import pyconfig

def prepare_latent_image_ids(height, width):
  latent_image_ids = jnp.zeros((height, width, 3))
  latent_image_ids = latent_image_ids.at[..., 1].set(
    latent_image_ids[..., 1] + jnp.arange(height)[:, None]
  )
  latent_image_ids = latent_image_ids.at[..., 2].set(
    latent_image_ids[..., 2] + jnp.arange(width)[None, :]
  )

  latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

  latent_image_ids = latent_image_ids.reshape(
    latent_image_id_height * latent_image_id_width, latent_image_id_channels
  )

  return latent_image_ids.astype(jnp.bfloat16)

def pack_latents(
  latents: Array,
  batch_size: int,
  num_channels_latents: int,
  height: int,
  width: int,
):
  latents = jnp.reshape(latents, (batch_size, num_channels_latents, height // 2, 2, width // 2, 2))
  latents = jnp.permute_dims(latents, (0, 2, 4, 1, 3, 5))
  latents = jnp.reshape(latents, (batch_size, (height // 2) * (width // 2), num_channels_latents * 4))

  return latents

def prepare_latents(
  batch_size: int,
  num_channels_latents: int,
  height: int,
  width: int,
  vae_scale_factor: int,
  dtype: jnp.dtype,
  rng: Array
):

  # VAE applies 8x compression on images but we must also account for packing which
  # requires latent height and width to be divisibly by 2.
  height = 2 * (height // (vae_scale_factor * 2))
  width = 2 * (width // (vae_scale_factor * 2))

  shape = (batch_size, num_channels_latents, height, width)

  latents = jax.random.normal(rng, shape=shape, dtype=jnp.bfloat16)
  # pack latents
  latents = pack_latents(latents, batch_size, num_channels_latents, height, width)

  latent_image_ids = prepare_latent_image_ids(height // 2, width // 2)

  return latents, latent_image_ids

def get_clip_prompt_embeds(
  prompt: Union[str, List[str]],
  num_images_per_prompt : int,
  tokenizer: CLIPTokenizer,
  text_encoder : FlaxCLIPTextModel
):
  prompt = [prompt] if isinstance(prompt, str) else prompt
  batch_size = len(prompt)
  text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_overflowing_tokens=False,
    return_length=False,
    return_tensors="np"
  )

  text_input_ids = text_inputs.input_ids

  prompt_embeds = text_encoder(text_input_ids, params=text_encoder.params, train=False)
  prompt_embeds = prompt_embeds.pooler_output
  prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=-1)
  prompt_embeds = np.reshape(prompt_embeds, (batch_size * num_images_per_prompt, -1))
  return prompt_embeds

def get_t5_prompt_embeds(
  prompt: Union[str, List[str]],
  num_images_per_prompt: int,
  tokenizer: T5TokenizerFast,
  text_encoder: T5EncoderModel,
  max_sequence_length: int = 512
):

  prompt = [prompt] if isinstance(prompt, str) else prompt
  batch_size = len(prompt)

  text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=max_sequence_length,
    truncation=True,
    return_length=False,
    return_overflowing_tokens=False,
    return_tensors="pt"
  )
  text_input_ids = text_inputs.input_ids

  prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)[0]
  dtype = text_encoder.dtype
  prompt_embeds = prompt_embeds.to(dtype=dtype)

  _, seq_len, _ = prompt_embeds.shape

  # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
  prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
  prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

  return prompt_embeds


def encode_prompt(
  prompt: Union[str, List[str]],
  prompt_2: Union[str, List[str]],
  clip_tokenizer: CLIPTokenizer,
  clip_text_encoder: FlaxCLIPTextModel,
  t5_tokenizer: T5TokenizerFast,
  t5_text_encoder: T5EncoderModel,
  num_images_per_prompt: int = 1,
  max_sequence_length: int = 512
):
  
  prompt = [prompt] if isinstance(prompt, str) else prompt
  prompt_2 = prompt or prompt_2
  prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

  pooled_prompt_embeds = get_clip_prompt_embeds(
    prompt=prompt,
    num_images_per_prompt=num_images_per_prompt,
    tokenizer=clip_tokenizer,
    text_encoder=clip_text_encoder
  )

  prompt_embeds = get_t5_prompt_embeds(
    prompt=prompt_2,
    num_images_per_prompt=num_images_per_prompt,
    tokenizer=t5_tokenizer,
    text_encoder=t5_text_encoder
  )
  prompt_embeds = jnp.asarray(prompt_embeds.detach().numpy())

  text_ids = jnp.zeros((prompt_embeds.shape[1], 3)).astype(jnp.bfloat16)
  return prompt_embeds, pooled_prompt_embeds, text_ids

def run(config):
  from maxdiffusion.models.flux.util import load_flow_model

  rng = jax.random.PRNGKey(config.seed)

  per_host_number_of_images = config.per_device_batch_size * jax.local_device_count()

  # LOAD VAE

  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="vae",
    from_pt=True,
    use_safetensors=True,
    dtype="bfloat16"
  )
  vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

  # LOAD UNET

  transformer = FluxTransformer2DModel.from_config(
    config.pretrained_model_name_or_path,
    subfolder="transformer"
  )
  
  num_channels_latents = transformer.in_channels // 4
  latents, latent_image_ids = prepare_latents(
    batch_size=per_host_number_of_images,
    num_channels_latents=num_channels_latents,
    height=config.resolution,
    width=config.resolution,
    dtype=jnp.bfloat16,
    vae_scale_factor=vae_scale_factor,
    rng=rng
  )

  #load_flow_model("flux-dev", "cpu")

  # transformer, params = FluxTransformer2DModel.from_pretrained(
  #   config.pretrained_model_name_or_path,
  #   subfolder="text_encoder_2",
  #   from_pt=True,
  #   dtype=config.weights_dtype
  # )


  # LOAD TEXT ENCODERS - t5 on cpu
  clip_text_encoder = FlaxCLIPTextModel.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="text_encoder",
    from_pt=True,
    dtype=config.weights_dtype
  )
  clip_tokenizer = CLIPTokenizer.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="tokenizer",
    dtype=config.weights_dtype
  )

  t5_encoder_pt = T5EncoderModel.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="text_encoder_2",
  )

  t5_tokenizer = T5TokenizerFast.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
  )

  prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
    prompt=config.prompt,
    prompt_2=config.prompt_2,
    clip_tokenizer=clip_tokenizer,
    clip_text_encoder=clip_text_encoder,
    t5_tokenizer=t5_tokenizer,
    t5_text_encoder=t5_encoder_pt,
  )

  transformer_params = transformer.init(
    {"params" : rng},
    img=latents,
    img_ids=latent_image_ids,
    txt=prompt_embeds,
    txt_ids=text_ids,
    timesteps=[1.0],
    y=pooled_prompt_embeds
  )["params"]
  breakpoint()



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)