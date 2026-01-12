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

from maxdiffusion.image_processor import PipelineImageInput
from maxdiffusion import max_logging
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

class WanPipelineI2V_2_2(WanPipeline):
  """Pipeline for WAN 2.2 Image-to-Video."""
  def __init__(self, config: HyperParameters, low_noise_transformer: Optional[WanModel], high_noise_transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.low_noise_transformer = low_noise_transformer
    self.high_noise_transformer = high_noise_transformer
    self.boundary_ratio = config.boundary_ratio

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only, i2v=True)
    low_noise_transformer, high_noise_transformer = None, None
    if not vae_only:
        if load_transformer:
            high_noise_transformer = super().load_transformer(
                devices_array=common_components["devices_array"], mesh=common_components["mesh"],
                rngs=common_components["rngs"], config=config, restored_checkpoint=restored_checkpoint,
                subfolder="transformer"
            )
            low_noise_transformer = super().load_transformer(
                devices_array=common_components["devices_array"], mesh=common_components["mesh"],
                rngs=common_components["rngs"], config=config, restored_checkpoint=restored_checkpoint,
                subfolder="transformer_2"
            )

    pipeline = cls(
      tokenizer=common_components["tokenizer"], text_encoder=common_components["text_encoder"],
      image_processor=common_components["image_processor"], image_encoder=common_components["image_encoder"],
      low_noise_transformer=low_noise_transformer, high_noise_transformer=high_noise_transformer,
      vae=common_components["vae"], vae_cache=common_components["vae_cache"],
      scheduler=common_components["scheduler"], scheduler_state=common_components["scheduler_state"],
      devices_array=common_components["devices_array"], mesh=common_components["mesh"],
      config=config,
    )
    return pipeline, low_noise_transformer, high_noise_transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline, low_noise_transformer, high_noise_transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    pipeline.low_noise_transformer = cls.quantize_transformer(config, low_noise_transformer, pipeline, pipeline.mesh)
    pipeline.high_noise_transformer = cls.quantize_transformer(config, high_noise_transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
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
        latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
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
        batch_size, 1, num_latent_frames, self.vae_scale_factor_temporal, latent_height, latent_width
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
    guidance_scale_low: float = 3.0,
    guidance_scale_high: float = 4.0,
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
    height = height or self.config.height
    width = width or self.config.width
    num_frames = num_frames or self.config.num_frames

    if num_frames % self.vae_scale_factor_temporal != 1:
        max_logging.log(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
            f"Rounding {num_frames} to the nearest valid number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        max_logging.log(f"Adjusted num_frames to: {num_frames}")
    num_frames = max(num_frames, 1)

    prompt_embeds, negative_prompt_embeds, image_embeds, effective_batch_size = self._prepare_model_inputs_i2v(
        prompt, image, negative_prompt, num_videos_per_prompt, max_sequence_length,
        prompt_embeds, negative_prompt_embeds, image_embeds, last_image
    )

    image_tensor = self.video_processor.preprocess(image, height=height, width=width)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor[None, ...] 
    last_image_tensor = None
    if last_image:
        last_image_tensor = self.video_processor.preprocess(last_image, height=height, width=width)
        if last_image_tensor.ndim == 3:
            last_image_tensor = last_image_tensor[None, ...] # Add batch dimension
    
    if effective_batch_size > 1:
        image_tensor = jnp.repeat(image_tensor, effective_batch_size, axis=0)
        if last_image_tensor is not None:
            last_image_tensor = jnp.repeat(last_image_tensor, effective_batch_size, axis=0)



    if rng is None:
        rng = jax.random.key(self.config.seed)
    latents_rng, inference_rng = jax.random.split(rng)

    # For WAN 2.2, image_embeds may be None (no CLIP image encoder)
    # Use prompt_embeds dtype as fallback
    latents_dtype = image_embeds.dtype if image_embeds is not None else prompt_embeds.dtype

    latents, condition, first_frame_mask = self.prepare_latents(
        image=image_tensor,
        batch_size=effective_batch_size,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=latents_dtype,
        rng=latents_rng,
        latents=latents,
        last_image=last_image_tensor,
    )

    scheduler_state = self.scheduler.set_timesteps(
        self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
    )

    low_noise_graphdef, low_noise_state, low_noise_rest = nnx.split(self.low_noise_transformer, nnx.Param, ...)
    high_noise_graphdef, high_noise_state, high_noise_rest = nnx.split(self.high_noise_transformer, nnx.Param, ...)
    data_sharding = NamedSharding(self.mesh, P())
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
        data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
    latents = jax.device_put(latents, data_sharding)
    condition = jax.device_put(condition, data_sharding)
    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)
    # WAN 2.2 I2V doesn't use image_embeds (it's None), but we still need to pass it to the function
    if image_embeds is not None:
        image_embeds = jax.device_put(image_embeds, data_sharding)
    if first_frame_mask is not None:
        first_frame_mask = jax.device_put(first_frame_mask, data_sharding)


    boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps

    p_run_inference = partial(
        run_inference_2_2_i2v,
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=boundary_timestep,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        image_embeds=image_embeds,
        first_frame_mask=first_frame_mask,
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      latents = p_run_inference(
          low_noise_graphdef=low_noise_graphdef, low_noise_state=low_noise_state, low_noise_rest=low_noise_rest,
          high_noise_graphdef=high_noise_graphdef, high_noise_state=high_noise_state, high_noise_rest=high_noise_rest,
          latents=latents, condition=condition,
          prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
          scheduler_state=scheduler_state, rng=inference_rng,
      )
      latents = jnp.transpose(latents, (0, 4, 1, 2, 3))
      latents = self._denormalize_latents(latents)

    if output_type == "latent":
      return latents
    return self._decode_latents_to_video(latents)

def run_inference_2_2_i2v(
    low_noise_graphdef, low_noise_state, low_noise_rest,
    high_noise_graphdef, high_noise_state, high_noise_rest,
    latents: jnp.array,
    condition: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    image_embeds: jnp.array,
    first_frame_mask: Optional[jnp.array],
    guidance_scale_low: float,
    guidance_scale_high: float,
    boundary: int,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    rng: jax.Array,
):
    do_classifier_free_guidance = guidance_scale_low > 1.0 or guidance_scale_high > 1.0
    def high_noise_branch(operands):
        latents_input, ts_input, pe_input, ie_input = operands
        latents_input = jnp.transpose(latents_input, (0, 4, 1, 2, 3))
        noise_pred, latents_out = transformer_forward_pass(
            high_noise_graphdef, high_noise_state, high_noise_rest,
            latents_input, ts_input, pe_input,
            do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale_high,
            encoder_hidden_states_image=ie_input
        )
        return noise_pred, latents_out

    def low_noise_branch(operands):
        latents_input, ts_input, pe_input, ie_input = operands
        latents_input = jnp.transpose(latents_input, (0, 4, 1, 2, 3))
        noise_pred, latents_out = transformer_forward_pass(
            low_noise_graphdef, low_noise_state, low_noise_rest,
            latents_input, ts_input, pe_input,
            do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale_low,
            encoder_hidden_states_image=ie_input
        )
        return noise_pred, latents_out

    if do_classifier_free_guidance:
        prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
        # WAN 2.2 I2V: image_embeds may be None since it doesn't use CLIP image encoder
        if image_embeds is not None:
            image_embeds = jnp.concatenate([image_embeds, image_embeds], axis=0)
        condition = jnp.concatenate([condition] * 2)

    for step in range(num_inference_steps):
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        latents_input = latents
        if do_classifier_free_guidance:
            latents_input = jnp.concatenate([latents, latents], axis=0)
        latent_model_input = jnp.concatenate([latents_input, condition], axis=-1)
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
            
        use_high_noise = jnp.greater_equal(t, boundary)
        noise_pred, _ = jax.lax.cond(
        use_high_noise,
        high_noise_branch,
        low_noise_branch,
        (latent_model_input, timestep, prompt_embeds, image_embeds)
        )
        noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    return latents