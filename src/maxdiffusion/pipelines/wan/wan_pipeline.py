# Copyright 2025 Google LLC
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

from typing import List, Union, Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax
import flax.linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...max_utils import get_flash_block_sizes, get_precision
from ...models.wan.wan_utils import load_wan_transformer, load_wan_vae
from ...models.wan.transformers.transformer_wan import WanModel
from ...models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.video_processor import VideoProcessor
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState
from transformers import AutoTokenizer, UMT5EncoderModel
from maxdiffusion.utils.import_utils import is_ftfy_available
import html
import re
import torch


def basic_clean(text):
  if is_ftfy_available():
    import ftfy
    text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def prompt_clean(text):
  text = whitespace_clean(basic_clean(text))
  return text


def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs


# For some reason, jitting this function increases the memory significantly, so instead manually move weights to device.
def create_sharded_logical_transformer(devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):

  def create_model(rngs: nnx.Rngs, wan_config: dict):
    wan_transformer = WanModel(**wan_config, rngs=rngs)
    return wan_transformer

  # 1. Load config.
  wan_config = WanModel.load_config(config.pretrained_model_name_or_path, subfolder="transformer")
  wan_config["mesh"] = mesh
  wan_config["dtype"] = config.activations_dtype
  wan_config["weights_dtype"] = config.weights_dtype
  wan_config["attention"] = config.attention
  wan_config["precision"] = get_precision(config)
  wan_config["flash_block_sizes"] = get_flash_block_sizes(config)

  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, wan_config=wan_config)
  wan_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_transformer, nnx.Param, ...)

  # 3. retrieve the state shardings, mapping logical names to mesh axis names.
  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  # 4. Load pretrained weights and move them to device using the state shardings from (3) above.
  # This helps with loading sharded weights directly into the accelerators without fist copying them
  # all to one device and then distributing them, thus using low HBM memory.
  params = load_wan_transformer(config.pretrained_model_name_or_path, params, "cpu")
  params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
  for path, val in flax.traverse_util.flatten_dict(params).items():
    sharding = logical_state_sharding[path].value
    state[path].value = jax.device_put(val, sharding)
  state = nnx.from_flat_state(state)

  wan_transformer = nnx.merge(graphdef, state, rest_of_state)
  return wan_transformer


@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model


class WanPipeline:
  r"""
  Pipeline for text-to-video generation using Wan.

  tokenizer ([`T5Tokenizer`]):
      Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
      specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  text_encoder ([`T5EncoderModel`]):
      [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
      the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  transformer ([`WanModel`]):
      Conditional Transformer to denoise the input latents.
  scheduler ([`FlaxUniPCMultistepScheduler`]):
      A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
  vae ([`AutoencoderKLWan`]):
      Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
  """

  def __init__(
      self,
      tokenizer: AutoTokenizer,
      text_encoder: UMT5EncoderModel,
      transformer: WanModel,
      vae: AutoencoderKLWan,
      vae_cache: AutoencoderKLWanCache,
      scheduler: FlaxUniPCMultistepScheduler,
      scheduler_state: UniPCMultistepSchedulerState,
      devices_array: np.array,
      mesh: Mesh,
      config: HyperParameters,
  ):
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.transformer = transformer
    self.vae = vae
    self.vae_cache = vae_cache
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config

    self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
    self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    self.p_run_inference = None

  @classmethod
  def load_text_encoder(cls, config: HyperParameters):
    text_encoder = UMT5EncoderModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    return text_encoder

  @classmethod
  def load_tokenizer(cls, config: HyperParameters):
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    return tokenizer

  @classmethod
  def load_vae(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
    wan_vae = AutoencoderKLWan.from_config(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        rngs=rngs,
        mesh=mesh,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
    )
    vae_cache = AutoencoderKLWanCache(wan_vae)

    graphdef, state = nnx.split(wan_vae, nnx.Param)
    params = state.to_pure_dict()
    # This replaces random params with the model.
    params = load_wan_vae(config.pretrained_model_name_or_path, params, "cpu")
    params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
    params = jax.device_put(params, NamedSharding(mesh, P()))
    wan_vae = nnx.merge(graphdef, params)
    p_create_sharded_logical_model = partial(create_sharded_logical_model, logical_axis_rules=config.logical_axis_rules)
    # Shard
    with mesh:
      wan_vae = p_create_sharded_logical_model(model=wan_vae)
    return wan_vae, vae_cache

  @classmethod
  def load_transformer(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
    with mesh:
      wan_transformer = create_sharded_logical_transformer(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)
    return wan_transformer

  @classmethod
  def load_scheduler(cls, config):
    scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
        flow_shift=config.flow_shift,  # 5.0 for 720p, 3.0 for 480p
    )
    return scheduler, scheduler_state

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False):
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)
    transformer = None
    tokenizer = None
    scheduler = None
    scheduler_state = None
    text_encoder = None
    if not vae_only:
      with mesh:
        transformer = cls.load_transformer(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)

      text_encoder = cls.load_text_encoder(config=config)
      tokenizer = cls.load_tokenizer(config=config)

      scheduler, scheduler_state = cls.load_scheduler(config=config)

    with mesh:
      wan_vae, vae_cache = cls.load_vae(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)

    return WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=wan_vae,
        vae_cache=vae_cache,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
        devices_array=devices_array,
        mesh=mesh,
        config=config,
    )

  def _get_t5_prompt_embeds(
      self,
      prompt: Union[str, List[str]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 226,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 226,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if prompt_embeds is None:
      prompt_embeds = self._get_t5_prompt_embeds(
          prompt=prompt,
          num_videos_per_prompt=num_videos_per_prompt,
          max_sequence_length=max_sequence_length,
      )
      prompt_embeds = jnp.array(prompt_embeds.detach().numpy(), dtype=self.config.weights_dtype)

    if negative_prompt_embeds is None:
      negative_prompt = negative_prompt or ""
      negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
      negative_prompt_embeds = self._get_t5_prompt_embeds(
          prompt=negative_prompt,
          num_videos_per_prompt=num_videos_per_prompt,
          max_sequence_length=max_sequence_length,
      )
      negative_prompt_embeds = jnp.array(negative_prompt_embeds.detach().numpy(), dtype=self.config.weights_dtype)

    return prompt_embeds, negative_prompt_embeds

  def prepare_latents(
      self,
      batch_size: int,
      vae_scale_factor_temporal: int,
      vae_scale_factor_spatial: int,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_channels_latents: int = 16,
  ):
    rng = jax.random.key(self.config.seed)
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    shape = (
        batch_size,
        num_channels_latents,
        num_latent_frames,
        int(height) // vae_scale_factor_spatial,
        int(width) // vae_scale_factor_spatial,
    )
    latents = jax.random.normal(rng, shape=shape, dtype=self.config.weights_dtype)

    return latents

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: jax.Array = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
      vae_only: bool = False,
      slg_layers: List[int] = None,
      slg_start: float = 0.0,
      slg_end: float = 1.0,
  ):
    if not vae_only:
      if num_frames % self.vae_scale_factor_temporal != 1:
        max_logging.log(
            f"`num_frames -1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
      num_frames = max(num_frames, 1)

      # 2. Define call parameters
      if prompt is not None and isinstance(prompt, str):
        prompt = [prompt]

      batch_size = len(prompt)

      prompt_embeds, negative_prompt_embeds = self.encode_prompt(
          prompt=prompt,
          negative_prompt=negative_prompt,
          max_sequence_length=max_sequence_length,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )

      num_channel_latents = self.transformer.config.in_channels
      if latents is None:
        latents = self.prepare_latents(
            batch_size=batch_size,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            height=height,
            width=width,
            num_frames=num_frames,
            num_channels_latents=num_channel_latents,
        )

      data_sharding = NamedSharding(self.mesh, P())
      if len(prompt) % jax.device_count() == 0:
        data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

      latents = jax.device_put(latents, data_sharding)
      prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
      negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)

      scheduler_state = self.scheduler.set_timesteps(
          self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
      )

      graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

      p_run_inference = partial(
          run_inference,
          guidance_scale=guidance_scale,
          num_inference_steps=num_inference_steps,
          scheduler=self.scheduler,
          scheduler_state=scheduler_state,
          slg_layers=slg_layers,
          slg_start=slg_start,
          slg_end=slg_end,
          num_transformer_layers=self.transformer.config.num_layers,
      )

      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        latents = p_run_inference(
            graphdef=graphdef,
            sharded_state=state,
            rest_of_state=rest_of_state,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
      latents_mean = jnp.array(self.vae.latents_mean).reshape(1, self.vae.z_dim, 1, 1, 1)
      latents_std = 1.0 / jnp.array(self.vae.latents_std).reshape(1, self.vae.z_dim, 1, 1, 1)
      latents = latents / latents_std + latents_mean
      latents = latents.astype(self.config.weights_dtype)

    video = self.vae.decode(latents, self.vae_cache)[0]

    video = jnp.transpose(video, (0, 4, 1, 2, 3))
    video = torch.from_numpy(np.array(video.astype(dtype=jnp.float32))).to(dtype=torch.bfloat16)
    video = self.video_processor.postprocess_video(video, output_type="np")
    return video


@jax.jit
def transformer_forward_pass(graphdef, sharded_state, rest_of_state, latents, timestep, prompt_embeds, is_uncond, slg_mask):
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  return wan_transformer(
      hidden_states=latents, timestep=timestep, encoder_hidden_states=prompt_embeds, is_uncond=is_uncond, slg_mask=slg_mask
  )


def run_inference(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    num_transformer_layers: int,
    scheduler_state,
    slg_layers: List[int] = None,
    slg_start: float = 0.0,
    slg_end: float = 1.0,
):
  do_classifier_free_guidance = guidance_scale > 1.0
  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
  for step in range(num_inference_steps):
    slg_mask = jnp.zeros(num_transformer_layers, dtype=jnp.bool_)
    if slg_layers and int(slg_start * num_inference_steps) <= step < int(slg_end * num_inference_steps):
      slg_mask = slg_mask.at[jnp.array(slg_layers)].set(True)
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    # get original batch size before concat in case of cfg.
    bsz = latents.shape[0]
    if do_classifier_free_guidance:
      latents = jnp.concatenate([latents] * 2)
    timestep = jnp.broadcast_to(t, latents.shape[0])

    noise_pred = transformer_forward_pass(
        graphdef,
        sharded_state,
        rest_of_state,
        latents,
        timestep,
        prompt_embeds,
        is_uncond=jnp.array(True, dtype=jnp.bool_),
        slg_mask=slg_mask,
    )

    if do_classifier_free_guidance:
      noise_uncond = noise_pred[bsz:]
      noise_pred = noise_pred[:bsz]
      noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
      latents = latents[:bsz]
    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents
