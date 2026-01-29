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

import math
import functools
import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange
from typing import List, Union, Callable
from flax.linen import partitioning as nn_partitioning
from transformers import CLIPTokenizer, FlaxCLIPTextModel, T5EncoderModel, AutoTokenizer

def unpack(x: Array, height: int, width: int) -> Array:
  return rearrange(
      x,
      "b (h w) (c ph pw) -> b c (h ph) (w pw)",
      h=math.ceil(height / 16),
      w=math.ceil(width / 16),
      ph=2,
      pw=2,
  )

def vae_decode(latents, vae, state, config):
  img = unpack(x=latents.astype(jnp.float32), height=config.resolution, width=config.resolution)
  img = img / vae.config.scaling_factor + vae.config.shift_factor
  img = vae.apply({"params": state.params}, img, deterministic=True, method=vae.decode).sample
  return img

def loop_body(
    step,
    args,
    transformer,
    latent_image_ids,
    prompt_embeds,
    txt_ids,
    vec,
    guidance_vec,
):
  latents, state, c_ts, p_ts = args
  latents_dtype = latents.dtype
  t_curr = c_ts[step]
  t_prev = p_ts[step]
  t_vec = jnp.full((latents.shape[0],), t_curr, dtype=latents.dtype)
  pred = transformer.apply(
      {"params": state.params},
      hidden_states=latents,
      img_ids=latent_image_ids,
      encoder_hidden_states=prompt_embeds,
      txt_ids=txt_ids,
      timestep=t_vec,
      guidance=guidance_vec,
      pooled_projections=vec,
  ).sample
  latents = latents + (t_prev - t_curr) * pred
  latents = jnp.array(latents, dtype=latents_dtype)
  return latents, state, c_ts, p_ts

def prepare_latent_image_ids(height, width):
  latent_image_ids = jnp.zeros((height, width, 3))
  latent_image_ids = latent_image_ids.at[..., 1].set(jnp.arange(height)[:, None])
  latent_image_ids = latent_image_ids.at[..., 2].set(jnp.arange(width)[None, :])

  latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

  latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)
  return latent_image_ids.astype(jnp.bfloat16)

def time_shift(mu: float, sigma: float, t: Array):
  return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
  m = (y2 - y1) / (x2 - x1)
  b = y1 - m * x1
  return lambda x: m * x + b

def run_inference(
    states, transformer, vae, config, mesh, latents, latent_image_ids, prompt_embeds, txt_ids, vec, guidance_vec, c_ts, p_ts
):
  transformer_state = states["transformer"]
  vae_state = states["vae"]

  loop_body_p = functools.partial(
      loop_body,
      transformer=transformer,
      latent_image_ids=latent_image_ids,
      prompt_embeds=prompt_embeds,
      txt_ids=txt_ids,
      vec=vec,
      guidance_vec=guidance_vec,
  )
  vae_decode_p = functools.partial(vae_decode, vae=vae, state=vae_state, config=config)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    latents, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, transformer_state, c_ts, p_ts))
  image = vae_decode_p(latents)
  return image

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
    batch_size: int, num_channels_latents: int, height: int, width: int, vae_scale_factor: int, dtype: jnp.dtype, rng: Array
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
  latent_image_ids = jnp.tile(latent_image_ids, (batch_size, 1, 1))

  return latents, latent_image_ids

def get_clip_prompt_embeds(
    prompt: Union[str, List[str]], num_images_per_prompt: int, tokenizer: CLIPTokenizer, text_encoder: FlaxCLIPTextModel
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
      return_tensors="np",
  )

  text_input_ids = text_inputs.input_ids

  prompt_embeds = text_encoder(text_input_ids, params=text_encoder.params, train=False)
  prompt_embeds = prompt_embeds.pooler_output
  prompt_embeds = jnp.tile(prompt_embeds, (batch_size * num_images_per_prompt, 1))
  return prompt_embeds

def get_t5_prompt_embeds(
    prompt: Union[str, List[str]],
    num_images_per_prompt: int,
    tokenizer: AutoTokenizer,
    text_encoder: T5EncoderModel,
    max_sequence_length: int = 512,
):
  prompt = [prompt] if isinstance(prompt, str) else prompt
  batch_size = len(prompt)
  text_inputs = tokenizer(
      prompt,
      truncation=True,
      max_length=max_sequence_length,
      return_length=False,
      return_overflowing_tokens=False,
      padding="max_length",
      return_tensors="np",
  )
  text_input_ids = text_inputs.input_ids
  prompt_embeds = text_encoder(text_input_ids, attention_mask=None, output_hidden_states=False)["last_hidden_state"]
  dtype = text_encoder.dtype
  prompt_embeds = prompt_embeds.astype(dtype)
  _, seq_len, _ = prompt_embeds.shape
  # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
  prompt_embeds = jnp.tile(prompt_embeds, (1, num_images_per_prompt, 1))
  prompt_embeds = jnp.reshape(prompt_embeds, (batch_size * num_images_per_prompt, seq_len, -1))
  return prompt_embeds

def encode_prompt(
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    clip_tokenizer: CLIPTokenizer,
    clip_text_encoder: FlaxCLIPTextModel,
    t5_tokenizer: AutoTokenizer,
    t5_text_encoder: T5EncoderModel,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
):
  prompt = [prompt] if isinstance(prompt, str) else prompt
  prompt_2 = prompt or prompt_2
  prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

  pooled_prompt_embeds = get_clip_prompt_embeds(
      prompt=prompt, num_images_per_prompt=num_images_per_prompt, tokenizer=clip_tokenizer, text_encoder=clip_text_encoder
  )

  prompt_embeds = get_t5_prompt_embeds(
      prompt=prompt_2,
      num_images_per_prompt=num_images_per_prompt,
      tokenizer=t5_tokenizer,
      text_encoder=t5_text_encoder,
      max_sequence_length=max_sequence_length,
  )

  text_ids = jnp.zeros((prompt_embeds.shape[1], 3)).astype(jnp.bfloat16)
  return prompt_embeds, pooled_prompt_embeds, text_ids
