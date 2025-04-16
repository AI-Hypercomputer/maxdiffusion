# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from functools import partial
from typing import List, Optional, Union, Callable

import jax
import jax.numpy as jnp
import math
from transformers import (CLIPTokenizer, FlaxCLIPTextModel, FlaxT5EncoderModel, AutoTokenizer)
from einops import rearrange
from jax.typing import DTypeLike
from chex import Array

from flax.linen import partitioning as nn_partitioning

from maxdiffusion.utils import logging

from ...models import FlaxAutoencoderKL
from ...schedulers import (FlaxEulerDiscreteScheduler)
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class FluxPipeline(FlaxDiffusionPipeline):

  def __init__(
      self,
      t5_encoder: FlaxCLIPTextModel,
      clip_encoder: FlaxCLIPTextModel,
      vae: FlaxAutoencoderKL,
      t5_tokenizer: FlaxT5EncoderModel,
      clip_tokenizer: CLIPTokenizer,
      flux: FluxTransformer2DModel,
      scheduler: FlaxEulerDiscreteScheduler,
      dtype: jnp.dtype = jnp.float32,
      mesh: Optional = None,
      config: Optional = None,
      rng: Optional = None,
  ):
    super().__init__()
    self.dtype = dtype
    self.register_modules(
        vae=vae,
        t5_encoder=t5_encoder,
        clip_encoder=clip_encoder,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
        flux=flux,
        scheduler=scheduler,
    )
    self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    self.mesh = mesh
    self._config = config
    self.rng = rng

  def create_noise(
      self,
      num_samples: int,
      height: int,
      width: int,
      dtype: DTypeLike,
      seed: jax.random.PRNGKey,
  ):
    return jax.random.normal(
        key=seed,
        shape=(num_samples, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16)),
        dtype=dtype,
    )

  def unpack(self, x: Array, height: int, width: int) -> Array:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

  def vae_decode(self, latents, vae, state, config):
    img = self.unpack(x=latents, height=config.resolution, width=config.resolution)
    img = img / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.apply({"params": state["params"]}, img, deterministic=True, method=vae.decode).sample
    return img

  def vae_encode(self, latents, vae, state):
    img = vae.apply({"params": state["params"]}, latents, deterministic=True, method=vae.encode).latent_dist.mode()
    img = vae.config.scaling_factor * (img - vae.config.shift_factor)
    return img

  # this is the reverse of the unpack function
  def pack_latents(
      self,
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
      self,
      batch_size: int,
      num_channels_latents: int,
      height: int,
      width: int,
      vae_scale_factor: int,
      dtype: jnp.dtype,
      rng: Array,
  ):

    # VAE applies 8x compression on images but we must also account for packing which
    # requires latent height and width to be divisibly by 2.
    height = 2 * (height // (vae_scale_factor * 2))
    width = 2 * (width // (vae_scale_factor * 2))

    shape = (batch_size, num_channels_latents, height, width)

    latents = jax.random.normal(rng, shape=shape, dtype=jnp.bfloat16)
    # pack latents
    latents = self.pack_latents(latents, batch_size, num_channels_latents, height, width)

    latent_image_ids = self.prepare_latent_image_ids(height // 2, width // 2)
    latent_image_ids = jnp.tile(latent_image_ids, (batch_size, 1, 1))

    return latents, latent_image_ids

  def prepare_latent_image_ids(self, height, width):
    latent_image_ids = jnp.zeros((height, width, 3))
    latent_image_ids = latent_image_ids.at[..., 1].set(jnp.arange(height)[:, None])
    latent_image_ids = latent_image_ids.at[..., 2].set(jnp.arange(width)[None, :])

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

    return latent_image_ids.astype(jnp.bfloat16)

  def get_clip_prompt_embeds(
      self,
      prompt: Union[str, List[str]],
      num_images_per_prompt: int,
      tokenizer: CLIPTokenizer,
      text_encoder: FlaxCLIPTextModel,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
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
    prompt_embeds = jnp.tile(prompt_embeds, (num_images_per_prompt, 1))
    return prompt_embeds

  def get_t5_prompt_embeds(
      self,
      prompt: Union[str, List[str]],
      num_images_per_prompt: int,
      tokenizer: AutoTokenizer,
      text_encoder: FlaxT5EncoderModel,
      max_sequence_length: int = 512,
      encode_in_batches=False,
      encode_batch_size=None,
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
    if encode_in_batches:
      prompt_embeds = None
      for i in range(0, text_input_ids.shape[0], encode_batch_size):
        batch_prompt_embeds = text_encoder(
            text_input_ids[i : i + encode_batch_size], attention_mask=None, output_hidden_states=False
        )["last_hidden_state"]
        if prompt_embeds is None:
          prompt_embeds = batch_prompt_embeds
        else:
          prompt_embeds = jnp.concatenate([prompt_embeds, batch_prompt_embeds])
    else:
      prompt_embeds = text_encoder(text_input_ids, attention_mask=None, output_hidden_states=False)["last_hidden_state"]
      _, seq_len, _ = prompt_embeds.shape
      # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
      prompt_embeds = jnp.tile(prompt_embeds, (1, num_images_per_prompt, 1))
      prompt_embeds = jnp.reshape(prompt_embeds, (batch_size * num_images_per_prompt, seq_len, -1))

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.astype(dtype)

    return prompt_embeds

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      prompt_2: Union[str, List[str]],
      clip_tokenizer: CLIPTokenizer,
      clip_text_encoder: FlaxCLIPTextModel,
      t5_tokenizer: AutoTokenizer,
      t5_text_encoder: FlaxT5EncoderModel,
      num_images_per_prompt: int = 1,
      max_sequence_length: int = 512,
      encode_in_batches: bool = False,
      encode_batch_size: int = None,
  ):

    if encode_in_batches:
      assert encode_in_batches is not None

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = prompt or prompt_2
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    pooled_prompt_embeds = self.get_clip_prompt_embeds(
        prompt=prompt, num_images_per_prompt=num_images_per_prompt, tokenizer=clip_tokenizer, text_encoder=clip_text_encoder
    )

    prompt_embeds = self.get_t5_prompt_embeds(
        prompt=prompt_2,
        num_images_per_prompt=num_images_per_prompt,
        tokenizer=t5_tokenizer,
        text_encoder=t5_text_encoder,
        max_sequence_length=max_sequence_length,
        encode_in_batches=encode_in_batches,
        encode_batch_size=encode_batch_size,
    )

    text_ids = jnp.zeros((prompt_embeds.shape[0], prompt_embeds.shape[1], 3)).astype(jnp.bfloat16)
    return prompt_embeds, pooled_prompt_embeds, text_ids

  def _generate(
      self, flux_params, vae_params, latents, latent_image_ids, prompt_embeds, txt_ids, vec, guidance_vec, c_ts, p_ts
  ):

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
          {"params": state["params"]},
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

    loop_body_p = partial(
        loop_body,
        transformer=self.flux,
        latent_image_ids=latent_image_ids,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        vec=vec,
        guidance_vec=guidance_vec,
    )

    vae_decode_p = partial(self.vae_decode, vae=self.vae, state=vae_params, config=self._config)

    with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
      latents, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, flux_params, c_ts, p_ts))
    image = vae_decode_p(latents)
    return image

  def do_time_shift(self, mu: float, sigma: float, t: Array):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

  def get_lin_function(
      self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
  ) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

  def time_shift(self, latents, timesteps):
    # estimate mu based on linear estimation between two points
    lin_function = self.get_lin_function(
        x1=self._config.max_sequence_length, y1=self._config.base_shift, y2=self._config.max_shift
    )
    mu = lin_function(latents.shape[1])
    timesteps = self.do_time_shift(mu, 1.0, timesteps)
    return timesteps

  def __call__(self, timesteps: int, flux_params, vae_params):
    r"""
    The call function to the pipeline for generation.

    Args:
      txt: jnp.array,
      txt_ids: jnp.array,
      vec: jnp.array,
      num_inference_steps: int,
      height: int,
      width: int,
      guidance_scale: float,
      img: Optional[jnp.ndarray] = None,
      shift: bool = False,
      jit (`bool`, defaults to `False`):

    Examples:

    """

    if isinstance(timesteps, int):
      timesteps = jnp.linspace(1, 0, timesteps + 1)

    global_batch_size = 1 * jax.local_device_count()

    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=self._config.prompt,
        prompt_2=self._config.prompt_2,
        clip_tokenizer=self.clip_tokenizer,
        clip_text_encoder=self.clip_encoder,
        t5_tokenizer=self.t5_tokenizer,
        t5_text_encoder=self.t5_encoder,
        num_images_per_prompt=global_batch_size,
        max_sequence_length=self._config.max_sequence_length,
    )

    num_channels_latents = self.flux.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size=global_batch_size,
        num_channels_latents=num_channels_latents,
        height=self._config.resolution,
        width=self._config.resolution,
        dtype=jnp.bfloat16,
        vae_scale_factor=self.vae_scale_factor,
        rng=self.rng,
    )

    if self._config.time_shift:
      timesteps = self.time_shift(latents, timesteps)
    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    guidance = jnp.asarray([self._config.guidance_scale] * global_batch_size, dtype=jnp.bfloat16)

    images = self._generate(
        flux_params,
        vae_params,
        latents,
        latent_image_ids,
        prompt_embeds,
        text_ids,
        pooled_prompt_embeds,
        guidance,
        c_ts,
        p_ts,
    )

    images = images
    return images
