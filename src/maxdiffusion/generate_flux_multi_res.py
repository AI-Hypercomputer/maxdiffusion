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

from typing import List, Union, Sequence
from absl import app
from contextlib import ExitStack
import functools
import math
import time
import numpy as np
from PIL import Image
import jax
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P
import jax.numpy as jnp
import flax.linen as nn
from chex import Array
from flax.linen import partitioning as nn_partitioning
from transformers import (CLIPTokenizer, FlaxCLIPTextModel, T5EncoderModel, FlaxT5EncoderModel, AutoTokenizer)

from maxdiffusion import FlaxAutoencoderKL, pyconfig, max_logging
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.max_utils import (
    device_put_replicated,
    get_memory_allocations,
    create_device_mesh,
    get_flash_block_sizes,
    get_precision,
    setup_initial_state,
)
from maxdiffusion.loaders.flux_lora_pipeline import FluxLoraLoaderMixin


def maybe_load_flux_lora(config, lora_loader, params):
  def _noop_interceptor(next_fn, args, kwargs, context):
    return next_fn(*args, **kwargs)

  lora_config = config.lora_config
  interceptors = [_noop_interceptor]
  if len(lora_config["lora_model_name_or_path"]) > 0:
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


def unpack(x: Array, height: int, width: int, vae_scale_factor: int) -> Array:
  batch_size, num_patches, channels = x.shape
  height = 2 * (int(height) // (vae_scale_factor * 2))
  width = int(2 * (width // (vae_scale_factor * 2)))

  x = jnp.reshape(x, (batch_size, height // 2, width // 2, channels // 4, 2, 2))
  x = jnp.transpose(x, (0, 3, 1, 4, 2, 5))
  x = jnp.reshape(x, (batch_size, channels // (2 * 2), height, width))

  return x


def vae_decode(latents, vae, state, vae_scale_factor, resolution):
  img = unpack(x=latents.astype(jnp.float32), height=resolution[0], width=resolution[1], vae_scale_factor=vae_scale_factor)
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


def calculate_shift(
    image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5, max_shift: float = 1.16
):
  m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
  b = base_shift - m * base_seq_len
  mu = image_seq_len * m + b
  return mu


def run_inference(
    states,
    transformer,
    vae,
    config,
    resolution,
    mesh,
    latents,
    latent_image_ids,
    prompt_embeds,
    txt_ids,
    vec,
    guidance_vec,
    c_ts,
    p_ts,
    vae_scale_factor,
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
  vae_decode_p = functools.partial(
      vae_decode, vae=vae, state=vae_state, vae_scale_factor=vae_scale_factor, resolution=resolution
  )

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

  latents = jax.random.normal(rng, shape=shape, dtype=dtype)
  # pack latents
  latents = pack_latents(latents, batch_size, num_channels_latents, height, width)

  latent_image_ids = prepare_latent_image_ids(height // 2, width // 2)
  latent_image_ids = jnp.tile(latent_image_ids, (batch_size, 1, 1))

  return latents, latent_image_ids


def tokenize_clip(prompt: Union[str, List[str]], tokenizer: CLIPTokenizer):
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
  return text_inputs.input_ids


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


def tokenize_t5(prompt: Union[str, List[str]], tokenizer: AutoTokenizer, max_sequence_length: int = 512):
  prompt = [prompt] if isinstance(prompt, str) else prompt
  text_inputs = tokenizer(
      prompt,
      truncation=True,
      max_length=max_sequence_length,
      return_length=False,
      return_overflowing_tokens=False,
      padding="max_length",
      return_tensors="np",
  )
  return text_inputs.input_ids


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


def run(config):
  from maxdiffusion.models.flux.util import load_flow_model

  rng = jax.random.key(config.seed)
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  global_batch_size = config.per_device_batch_size * jax.local_device_count()

  # LOAD VAE

  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      config.pretrained_model_name_or_path, subfolder="vae", from_pt=True, use_safetensors=True, dtype="bfloat16"
  )

  weights_init_fn = functools.partial(vae.init_weights, rng=rng)
  vae_state, vae_state_shardings = setup_initial_state(
      model=vae,
      tx=None,
      config=config,
      mesh=mesh,
      weights_init_fn=weights_init_fn,
      model_params=vae_params,
      training=False,
  )

  vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

  # LOAD TRANSFORMER
  flash_block_sizes = get_flash_block_sizes(config)
  transformer = FluxTransformer2DModel.from_config(
      config.pretrained_model_name_or_path,
      subfolder="transformer",
      mesh=mesh,
      split_head_dim=config.split_head_dim,
      attention_kernel=config.attention,
      flash_block_sizes=flash_block_sizes,
      dtype=config.activations_dtype,
      weights_dtype=config.weights_dtype,
      precision=get_precision(config),
  )

  num_channels_latents = transformer.in_channels // 4

  # LOAD TEXT ENCODERS
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

  encoders_sharding = PositionalSharding(devices_array).replicate()
  partial_device_put_replicated = functools.partial(device_put_replicated, sharding=encoders_sharding)
  clip_text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), clip_text_encoder.params)
  clip_text_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, clip_text_encoder.params)
  t5_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), t5_encoder.params)
  t5_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, t5_encoder.params)

  def validate_inputs(latents, latent_image_ids, prompt_embeds, text_ids, timesteps, guidance, pooled_prompt_embeds):
    print("latents.shape: ", latents.shape, latents.dtype)
    print("latent_image_ids.shape: ", latent_image_ids.shape, latent_image_ids.dtype)
    print("text_ids.shape: ", text_ids.shape, text_ids.dtype)
    print("prompt_embeds: ", prompt_embeds.shape, prompt_embeds.dtype)
    print("timesteps.shape: ", timesteps.shape, timesteps.dtype)
    print("guidance.shape: ", guidance.shape, guidance.dtype)
    print("pooled_prompt_embeds.shape: ", pooled_prompt_embeds.shape, pooled_prompt_embeds.dtype)

  guidance = jnp.asarray([config.guidance_scale] * global_batch_size, dtype=jnp.bfloat16)

  get_memory_allocations()
  # evaluate shapes
  transformer_eval_params = transformer.init_weights(
      rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=True
  )

  # loads pretrained weights
  transformer_params = load_flow_model(config.flux_name, transformer_eval_params, "cpu")
  params = {}
  params["transformer"] = transformer_params
  # maybe load lora and create interceptor
  lora_loader = FluxLoraLoaderMixin()
  params, lora_interceptors = maybe_load_flux_lora(config, lora_loader, params)
  transformer_params = params["transformer"]
  # create transformer state
  weights_init_fn = functools.partial(
      transformer.init_weights, rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=False
  )
  with ExitStack() as stack:
    _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
    transformer_state, transformer_state_shardings = setup_initial_state(
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
  get_memory_allocations()

  states = {}
  state_shardings = {}

  state_shardings["transformer"] = transformer_state_shardings
  state_shardings["vae"] = vae_state_shardings

  states["transformer"] = transformer_state
  states["vae"] = vae_state
  # some resolutions from https://www.reddit.com/r/StableDiffusion/comments/1enxdga/flux_recommended_resolutions_from_01_to_20/
  resolutions = [
      (768, 768),
      (768, 1024),
      (1024, 768),
      (1024, 1024),
      (1408, 1408),
      (1728, 1152),
      (1152, 1728),
      (1664, 1216),
      (1216, 1664),
      (1920, 1088),
      (1088, 1920),
      (2176, 960),
      (960, 2176),
  ]
  p_jitted = {}
  recorded_times = {}
  text_encoding_time_final = 0
  for resolution in resolutions:
    max_logging.log(f"Resolutions: {resolution}")
    for _ in range(2):
      s0 = time.perf_counter()
      if config.offload_encoders:
        t5_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, t5_encoder.params)
        max_logging.log(f"Moving encoder to TPU time: {(time.perf_counter() - s0)}")
      prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
          prompt=config.prompt,
          prompt_2=config.prompt_2,
          clip_tokenizer=clip_tokenizer,
          clip_text_encoder=clip_text_encoder,
          t5_tokenizer=t5_tokenizer,
          t5_text_encoder=t5_encoder,
          num_images_per_prompt=global_batch_size,
          max_sequence_length=config.max_sequence_length,
      )
      if config.offload_encoders:
        s1 = time.perf_counter()
        cpus = jax.devices("cpu")
        t5_encoder.params = jax.device_put(t5_encoder.params, device=cpus[0])
        max_logging.log(f"Text encoding offload time: {(time.perf_counter() - s1)}")
      text_encoding_time_final = time.perf_counter() - s0
      max_logging.log(f"text encoding time: {text_encoding_time_final}")
      latents, latent_image_ids = prepare_latents(
          batch_size=global_batch_size,
          num_channels_latents=num_channels_latents,
          height=resolution[0],
          width=resolution[1],
          dtype=jnp.bfloat16,
          vae_scale_factor=vae_scale_factor,
          rng=rng,
      )

      # move inputs to device and shard
      s0 = time.perf_counter()
      data_sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
      prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
      text_ids = jax.device_put(text_ids)
      guidance = jax.device_put(guidance, data_sharding)
      pooled_prompt_embeds = jax.device_put(pooled_prompt_embeds, data_sharding)
      latents = jax.device_put(latents, data_sharding)
      latent_image_ids = jax.device_put(latent_image_ids)
      max_logging.log(f"Moving to device time: {(time.perf_counter() - s0)}")

      # Setup timesteps
      timesteps = jnp.linspace(1, 0, config.num_inference_steps + 1)
      # shifting the schedule to favor high timesteps for higher signal images
      if config.time_shift:
        # estimate mu based on linear estimation between two points
        # lin_function = get_lin_function(x1=config.max_sequence_length, y1=config.base_shift, y2=config.max_shift)
        # mu = lin_function(latents.shape[1])
        mu = calculate_shift(latents.shape[1], base_shift=config.base_shift, max_shift=config.max_shift)
        timesteps = time_shift(mu, 1.0, timesteps)
      c_ts = timesteps[:-1]
      p_ts = timesteps[1:]
      # validate_inputs(latents, latent_image_ids, prompt_embeds, text_ids, timesteps, guidance, pooled_prompt_embeds)
      p_run_inference = p_jitted.get(resolution, None)
      if p_run_inference is None:
        print("FN not found, compiling...")
        p_run_inference = jax.jit(
            functools.partial(
                run_inference,
                transformer=transformer,
                vae=vae,
                config=config,
                resolution=resolution,
                mesh=mesh,
                latents=latents,
                latent_image_ids=latent_image_ids,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                vec=pooled_prompt_embeds,
                guidance_vec=guidance,
                c_ts=c_ts,
                p_ts=p_ts,
                vae_scale_factor=vae_scale_factor,
            ),
        )
        p_jitted[resolution] = p_run_inference
      with ExitStack() as stack:
        _ = [stack.enter_context(nn.intercept_methods(interceptor)) for interceptor in lora_interceptors]
        s0 = time.perf_counter()
        imgs = p_run_inference(
            states,
            latents=latents,
            latent_image_ids=latent_image_ids,
            prompt_embeds=prompt_embeds,
            txt_ids=text_ids,
            vec=pooled_prompt_embeds,
        ).block_until_ready()
        recorded_times[resolution] = time.perf_counter() - s0
        max_logging.log(f"inference time: {recorded_times[resolution]}")
        s0 = time.perf_counter()
        imgs = jax.experimental.multihost_utils.process_allgather(imgs, tiled=True)
        max_logging.log(f"Gathering all time: {(time.perf_counter() - s0)}")
        s0 = time.perf_counter()
        imgs = np.array(imgs)
        imgs = (imgs * 0.5 + 0.5).clip(0, 1)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        imgs = np.uint8(imgs * 255)
        for i, image in enumerate(imgs):
          Image.fromarray(image).save(f"flux_{resolution[0]}_{resolution[1]}_{i}.png")
        max_logging.log(f"Saving images time: {(time.perf_counter() - s0)}")
        get_memory_allocations()

  max_logging.log("***************RESULTS***************")
  for key in recorded_times.keys():
    max_logging.log(f"Resolution: {key}, last inference time: {recorded_times[key]}")
  max_logging.log(f"\nText encoding time: {text_encoding_time_final}")

  return imgs


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
