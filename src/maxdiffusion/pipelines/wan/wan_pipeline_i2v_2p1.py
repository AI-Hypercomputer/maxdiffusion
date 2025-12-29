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

from maxdiffusion import max_logging
from maxdiffusion.image_processor import PipelineImageInput
from .wan_pipeline import WanPipeline, transformer_forward_pass
from ...models.wan.transformers.transformer_wan import WanModel
from typing import List, Union, Optional, Tuple
from ...pyconfig import HyperParameters
from functools import partial
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler
from ...max_utils import randn_tensor

class WanPipelineI2V_2_1(WanPipeline):
  """Pipeline for WAN 2.1 Image-to-Video."""
  def __init__(self, config: HyperParameters, transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only, i2v=True)
    transformer = None
    if not vae_only:
      if load_transformer:
        transformer = super().load_transformer(
            devices_array=common_components["devices_array"],
            mesh=common_components["mesh"],
            rngs=common_components["rngs"],
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer",
            use_real=True
        )

    pipeline = cls(
      tokenizer=common_components["tokenizer"],
      text_encoder=common_components["text_encoder"],
      image_processor=common_components["image_processor"],
      image_encoder=common_components["image_encoder"],
      transformer=transformer,
      vae=common_components["vae"],
      vae_cache=common_components["vae_cache"],
      scheduler=common_components["scheduler"],
      scheduler_state=common_components["scheduler_state"],
      devices_array=common_components["devices_array"],
      mesh=common_components["mesh"],
      config=config,
    )
    return pipeline, transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline , _ = cls._load_and_init(config, None, vae_only, load_transformer)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  def prepare_latents(
      self,
      image: jax.Array,
      batch_size: int,
      height: int,
      width: int,
      num_frames: int,
      dtype: jnp.dtype,
      rng: jax.Array,
      latents: Optional[jax.Array] = None,
      last_image: Optional[jax.Array] = None,
      num_videos_per_prompt: int = 1,
  ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = jnp.array(image)

        if last_image is not None:
            if hasattr(last_image, "detach"):
                last_image = last_image.detach().cpu().numpy()
            last_image = jnp.array(last_image)
        
        if num_videos_per_prompt > 1:
           image = jnp.repeat(image, num_videos_per_prompt, axis=0)
           if last_image is not None:
              last_image = jnp.repeat(last_image, num_videos_per_prompt, axis=0)
        
        num_channels_latents = self.vae.z_dim
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_latent_frames, latent_height, latent_width, num_channels_latents)

        if latents is None:
            latents = randn_tensor(shape, rng, self.config, dtype)
        else:
            latents = latents.astype(dtype)
        latent_condition, _ = self.prepare_latents_i2v_base(image, num_frames, dtype, last_image)
        mask_lat_size = jnp.ones((batch_size, 1, num_frames, latent_height, latent_width), dtype=dtype)
        if last_image is None:
            mask_lat_size = mask_lat_size.at[:, :, 1:, :, :].set(0)
        else:
            mask_lat_size = mask_lat_size.at[:, :, 1:-1, :, :].set(0)     
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = jnp.repeat(first_frame_mask, self.vae_scale_factor_temporal, axis=2)
        mask_lat_size = jnp.concatenate([first_frame_mask, mask_lat_size[:, :, 1:]], axis=2)
        mask_lat_size = mask_lat_size.reshape(
          batch_size, 
          1,
          num_latent_frames, 
          self.vae_scale_factor_temporal, 
          latent_height, 
          latent_width
        )
        mask_lat_size = jnp.transpose(mask_lat_size, (0, 2, 4, 5, 3, 1)).squeeze(-1)
        condition = jnp.concatenate([mask_lat_size, latent_condition], axis=-1)

        return latents, condition, None


  def __call__(
    self,
    prompt: Union[str, List[str]],
    image: PipelineImageInput,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    latents: Optional[jax.Array] = None,
    prompt_embeds: Optional[jax.Array] = None,
    negative_prompt_embeds: Optional[jax.Array] = None,
    image_embeds: Optional[jax.Array] = None,
    last_image: Optional[PipelineImageInput] = None,
    output_type: Optional[str] = "np",
    rng: Optional[jax.Array] = None,
  ):
    
    if num_videos_per_prompt == 1:
      n_devices = jax.device_count()
      if n_devices > 1:
        num_videos_per_prompt = n_devices
    
    height = height or self.config.height
    width = width or self.config.width
    num_frames = num_frames or self.config.num_frames

    prompt_embeds, negative_prompt_embeds, image_embeds, effective_batch_size = self._prepare_model_inputs_i2v(
        prompt, image, negative_prompt, num_videos_per_prompt, max_sequence_length,
        prompt_embeds, negative_prompt_embeds, image_embeds, last_image
    )

    image_tensor = self.video_processor.preprocess(image, height=height, width=width)
    last_image_tensor = None
    if last_image:
        last_image_tensor = self.video_processor.preprocess(last_image, height=height, width=width)

    if rng is None:
        rng = jax.random.key(self.config.seed)
    latents_rng, inference_rng = jax.random.split(rng)

    latents, condition, first_frame_mask = self.prepare_latents(
        image=image_tensor,
        batch_size=effective_batch_size,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=image_embeds.dtype,
        rng=latents_rng,
        latents=latents,
        last_image=last_image_tensor,
        num_videos_per_prompt=num_videos_per_prompt,
    )

    scheduler_state = self.scheduler.set_timesteps(
        self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
    )

    if self.scheduler_state.last_sample is None or self.scheduler_state.step_index is None:
      max_logging.log("[DEBUG] Priming scheduler state...")
      t0 = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[0]
      dummy_noise = jnp.zeros_like(latents)
      # This call initializes the internal state arrays
      step_output = self.scheduler.step(scheduler_state, dummy_noise, t0, latents)
      max_logging.log(f"[DEBUG] scheduler.step output type: {type(step_output)}")
      max_logging.log(f"[DEBUG] scheduler.step output value: {step_output}")
      _, scheduler_state = step_output
      max_logging.log(f"[DEBUG] After prime step: scheduler_state type: {type(scheduler_state)}")
      max_logging.log(f"[DEBUG] After prime step: scheduler_state value: {scheduler_state}")
      max_logging.log(f"[DEBUG] Scheduler state primed: step_index={scheduler_state.step_index is not None}, last_sample={scheduler_state.last_sample is not None}")

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)
    data_sharding = NamedSharding(self.mesh, P(*self.config.data_sharding))
    latents = jax.device_put(latents, data_sharding)
    condition = jax.device_put(condition, data_sharding)
    if first_frame_mask is not None:
        first_frame_mask = jax.device_put(first_frame_mask, data_sharding)

    p_run_inference = partial(
        run_inference_2_1_i2v,
        graphdef=graphdef,
        sharded_state=state,
        rest_of_state=rest_of_state,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        expand_timesteps=self.config.expand_timesteps
    )

    
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      latents = p_run_inference(
          latents=latents,
          condition=condition,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
          image_embeds=image_embeds,
          first_frame_mask=first_frame_mask,
          scheduler_state=scheduler_state,
          rng=inference_rng,
      )
      if self.config.expand_timesteps:
         latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
      latents = self._denormalize_latents(latents)

    if output_type == "latent":
      return latents
    return self._decode_latents_to_video(latents)

def run_inference_2_1_i2v(
    graphdef, sharded_state, rest_of_state,
    latents: jnp.array,
    condition: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    image_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    rng: jax.Array,
    expand_timesteps: bool,
    first_frame_mask: Optional[jnp.array],
):
  do_classifier_free_guidance = guidance_scale > 1.0

  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
    image_embeds = jnp.concatenate([image_embeds, image_embeds], axis=0)
    if  expand_timesteps:
        condition = jnp.concatenate([condition] * 2)
        first_frame_mask = jnp.concatenate([first_frame_mask] * 2)
    else:
        condition = jnp.concatenate([condition] * 2)


  def loop_body(step, vals):
    latents, scheduler_state, rng = vals
    rng, timestep_rng = jax.random.split(rng)
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

    latents_input = latents
    if do_classifier_free_guidance:
        latents_input = jnp.concatenate([latents, latents], axis=0)

    latent_model_input = jnp.concatenate([latents_input, condition], axis=-1)
    timestep = jnp.broadcast_to(t, latents_input.shape[0])
    latent_model_input = jnp.transpose(latent_model_input, (0, 4, 1, 2, 3))

    prompt_embeds_input = prompt_embeds
    image_embeds_input = image_embeds

    noise_pred, _ = transformer_forward_pass(
        graphdef, sharded_state, rest_of_state,
        latent_model_input, timestep, prompt_embeds_input,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
        encoder_hidden_states_image=image_embeds_input,
    )
    noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    return latents, scheduler_state, rng

  latents, _, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state, rng))
  return latents
