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
import numpy as np
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler
import time

class WanPipeline2_2(WanPipeline):
  """Pipeline for WAN 2.2 with dual transformers."""

  def __init__(
      self,
      config: HyperParameters,
      low_noise_transformer: Optional[WanModel],
      high_noise_transformer: Optional[WanModel],
      **kwargs
  ):
    super().__init__(config=config, **kwargs)
    self.low_noise_transformer = low_noise_transformer
    self.high_noise_transformer = high_noise_transformer
    self.boundary_ratio = config.boundary_ratio

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    low_noise_transformer, high_noise_transformer = None, None
    if not vae_only and load_transformer:
      low_noise_transformer = super().load_transformer(
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          rngs=common_components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder="transformer_2",
      )
      high_noise_transformer = super().load_transformer(
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          rngs=common_components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder="transformer",
      )

      pipeline = cls(
          tokenizer=common_components["tokenizer"],
          text_encoder=common_components["text_encoder"],
          low_noise_transformer=low_noise_transformer,
          high_noise_transformer=high_noise_transformer,
          vae=common_components["vae"],
          vae_cache=common_components["vae_cache"],
          scheduler=common_components["scheduler"],
          scheduler_state=common_components["scheduler_state"],
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
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
    pipeline, low_noise_transformer, high_noise_transformer = cls._load_and_init(
        config, restored_checkpoint, vae_only, load_transformer
    )
    return pipeline

  def _get_num_channel_latents(self) -> int:
    return self.low_noise_transformer.config.in_channels

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      guidance_scale_low: float = 3.0,
      guidance_scale_high: float = 4.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: jax.Array = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
      vae_only: bool = False,
  ):
    print("Starting WAN 2.2 inference...")
    print("Starting Text Encoder inference...")
    start = time.time()
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
    print("Text Encoder inference time:", time.time() - start)

    low_noise_graphdef, low_noise_state, low_noise_rest = nnx.split(self.low_noise_transformer, nnx.Param, ...)
    high_noise_graphdef, high_noise_state, high_noise_rest = nnx.split(self.high_noise_transformer, nnx.Param, ...)

    boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps

    p_run_inference = partial(
        run_inference_2_2,
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=boundary_timestep,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      print("Starting Transformer inference...")  
      start = time.time()  
      latents = p_run_inference(
          low_noise_graphdef=low_noise_graphdef,
          low_noise_state=low_noise_state,
          low_noise_rest=low_noise_rest,
          high_noise_graphdef=high_noise_graphdef,
          high_noise_state=high_noise_state,
          high_noise_rest=high_noise_rest,
          latents=latents,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )
      latents.block_until_ready()  # Ensure all computations are finished before timing
      print("Transformer inference time:", time.time() - start)
      latents = self._denormalize_latents(latents)
    print("Starting Decoder inference...")
    start = time.time()
    output = self._decode_latents_to_video(latents)
    print("Decoder inference time:", time.time() - start)
    return output

def run_inference_2_2(
    low_noise_graphdef,
    low_noise_state,
    low_noise_rest,
    high_noise_graphdef,
    high_noise_state,
    high_noise_rest,
    latents: jax.Array,
    prompt_embeds: jax.Array,
    negative_prompt_embeds: jax.Array,
    guidance_scale_low: float,
    guidance_scale_high: float,
    boundary: int,
    num_inference_steps: int,
    scheduler,
    scheduler_state,
):
    do_classifier_free_guidance = guidance_scale_low > 1.0 or guidance_scale_high > 1.0
    
    if do_classifier_free_guidance:
        prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

    # ==========================================================
    # THE SCAN BODY
    # Compiles the models EXACTLY ONCE, eliminating the OOM.
    # ==========================================================
    def scan_body(carry, t):
        current_latents, current_scheduler_state = carry

        # 1. Expand latents to Batch=2 locally inside the step
        if do_classifier_free_guidance:
            model_latents = jnp.concatenate([current_latents] * 2)
        else:
            model_latents = current_latents
            
        timestep = jnp.broadcast_to(t, model_latents.shape[0])
        use_high_noise = jnp.greater_equal(t, boundary)

        # 2. Branch logic (Operands are passed directly via closure)
        def high_branch(_):
            return transformer_forward_pass(
                high_noise_graphdef, high_noise_state, high_noise_rest, 
                model_latents, timestep, prompt_embeds, 
                do_classifier_free_guidance, guidance_scale_high
            )

        def low_branch(_):
            return transformer_forward_pass(
                low_noise_graphdef, low_noise_state, low_noise_rest, 
                model_latents, timestep, prompt_embeds, 
                do_classifier_free_guidance, guidance_scale_low
            )

        # 3. Execute the correct model
        noise_pred, latents_out = jax.lax.cond(use_high_noise, high_branch, low_branch, operand=None)

        # 4. Step the scheduler (Collapses back to Batch=1)
        new_latents, new_scheduler_state = scheduler.step(
            current_scheduler_state, noise_pred, t, latents_out
        ).to_tuple()

        # Carry must perfectly match the shape of the input carry!
        return (new_latents, new_scheduler_state), None

    # ==========================================================
    # EXECUTE THE SCAN
    # ==========================================================
    timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)
    
    scheduler_state = scheduler_state.replace(
        last_sample=jnp.zeros_like(latents),
        step_index=jnp.array(0, dtype=jnp.int32)
      )
    
    initial_carry = (latents, scheduler_state)
    
    final_carry, _ = jax.lax.scan(scan_body, initial_carry, timesteps)
    
    final_latents, _ = final_carry
    return final_latents