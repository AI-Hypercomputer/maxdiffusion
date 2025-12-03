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

from .wan_pipeline import WanPipeline, transformer_forward_pass
from ...models.wan.transformers.transformer_wan import WanModel
from typing import List, Union, Optional
from ...pyconfig import HyperParameters
from functools import partial
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler

class WanPipeline2_1(WanPipeline):
  """Pipeline for WAN 2.1 with a single transformer."""
  def __init__(self, config: HyperParameters, transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only:
      if load_transformer:
        transformer = super().load_transformer(
            devices_array=common_components["devices_array"],
            mesh=common_components["mesh"],
            rngs=common_components["rngs"],
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer"
        )

        pipeline = cls(
          tokenizer=common_components["tokenizer"],
          text_encoder=common_components["text_encoder"],
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
    pipeline , transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  def _get_num_channel_latents(self) -> int:
    return self.transformer.config.in_channels

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
    latents: Optional[jax.Array] = None,
    prompt_embeds: Optional[jax.Array] = None,
    negative_prompt_embeds: Optional[jax.Array] = None,
    vae_only: bool = False,
  ):
    latents, prompt_embeds, negative_prompt_embeds, scheduler_state, num_frames = self._prepare_model_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        num_videos_per_prompt,
        max_sequence_length,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        vae_only,
    )

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    p_run_inference = partial(
        run_inference_2_1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
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
      latents = self._denormalize_latents(latents)
    return self._decode_latents_to_video(latents)

def run_inference_2_1(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
):
  do_classifier_free_guidance = guidance_scale > 1.0
  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    if do_classifier_free_guidance:
      latents = jnp.concatenate([latents] * 2)
    timestep = jnp.broadcast_to(t, latents.shape[0])

    noise_pred, latents = transformer_forward_pass(
        graphdef,
        sharded_state,
        rest_of_state,
        latents,
        timestep,
        prompt_embeds,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
    )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents