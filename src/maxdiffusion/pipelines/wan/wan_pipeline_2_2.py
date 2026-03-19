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

from .wan_pipeline import WanPipeline, transformer_forward_pass, transformer_forward_pass_full_cfg, transformer_forward_pass_cfg_cache
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


class WanPipeline2_2(WanPipeline):
  """Pipeline for WAN 2.2 with dual transformers."""

  def __init__(
      self,
      config: HyperParameters,
      low_noise_transformer: Optional[WanModel],
      high_noise_transformer: Optional[WanModel],
      **kwargs,
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
          vae_mesh=common_components["vae_mesh"],
          vae_logical_axis_rules=common_components["vae_logical_axis_rules"],
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
      use_cfg_cache: bool = False,
  ):
    if use_cfg_cache and (guidance_scale_low <= 1.0 or guidance_scale_high <= 1.0):
      raise ValueError(
          f"use_cfg_cache=True requires both guidance_scale_low > 1.0 and guidance_scale_high > 1.0 "
          f"(got {guidance_scale_low}, {guidance_scale_high}). "
          "CFG cache accelerates classifier-free guidance, which must be enabled for both transformer phases."
      )

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
        use_cfg_cache=use_cfg_cache,
        height=height,
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
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
      latents = self._denormalize_latents(latents)
    return self._decode_latents_to_video(latents)


def run_inference_2_2(
    low_noise_graphdef,
    low_noise_state,
    low_noise_rest,
    high_noise_graphdef,
    high_noise_state,
    high_noise_rest,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale_low: float,
    guidance_scale_high: float,
    boundary: int,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    use_cfg_cache: bool = False,
    height: int = 480,
):
  """Denoising loop for WAN 2.2 T2V with optional FasterCache CFG-Cache.

  Dual-transformer CFG-Cache strategy (enabled via use_cfg_cache=True):
  - High-noise phase (t >= boundary): always full CFG — short phase, critical
    for establishing video structure.
  - Low-noise phase (t < boundary): FasterCache alternation — full CFG every N
    steps, FFT frequency-domain compensation on cache steps (batch×1).
  - Boundary transition: mandatory full CFG step to populate cache for the
    low-noise transformer.
  - FFT compensation identical to WAN 2.1 (Lv et al., ICLR 2025).
  """
  do_classifier_free_guidance = guidance_scale_low > 1.0 or guidance_scale_high > 1.0
  bsz = latents.shape[0]

  # ── CFG cache path ──
  if use_cfg_cache and do_classifier_free_guidance:
    # Get timesteps as numpy for Python-level scheduling decisions
    timesteps_np = np.array(scheduler_state.timesteps, dtype=np.int32)
    step_uses_high = [bool(timesteps_np[s] >= boundary) for s in range(num_inference_steps)]

    # Resolution-dependent CFG cache config — adapted for Wan 2.2.
    if height >= 720:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = int(num_inference_steps * 0.9)
      cfg_cache_alpha = 0.2
    else:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = num_inference_steps - 1
      cfg_cache_alpha = 0.2

    # Pre-split embeds once
    prompt_cond_embeds = prompt_embeds
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

    # Determine the first low-noise step (boundary transition).
    # In Wan 2.2 the boundary IS the structural→detail transition, so
    # all low-noise cache steps should emphasise high-frequency correction.
    first_low_step = next(
        (s for s in range(num_inference_steps) if not step_uses_high[s]),
        num_inference_steps,
    )
    t0_step = first_low_step  # all cache steps get high-freq boost

    # Pre-compute cache schedule and phase-dependent weights.
    first_full_in_low_seen = False
    step_is_cache = []
    step_w1w2 = []
    for s in range(num_inference_steps):
      if step_uses_high[s]:
        # Never cache high-noise transformer steps
        step_is_cache.append(False)
      else:
        is_cache = (
            first_full_in_low_seen
            and s >= cfg_cache_start_step
            and s < cfg_cache_end_step
            and (s - cfg_cache_start_step) % cfg_cache_interval != 0
        )
        step_is_cache.append(is_cache)
        if not is_cache:
          first_full_in_low_seen = True

      # Phase-dependent weights: w = 1 + α·I(condition)
      if s < t0_step:
        step_w1w2.append((1.0 + cfg_cache_alpha, 1.0))  # high-noise: boost low-freq
      else:
        step_w1w2.append((1.0, 1.0 + cfg_cache_alpha))  # low-noise: boost high-freq

    # Cache tensors (on-device JAX arrays, initialised to None).
    cached_noise_cond = None
    cached_noise_uncond = None

    for step in range(num_inference_steps):
      t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
      is_cache_step = step_is_cache[step]

      # Select transformer and guidance scale based on precomputed schedule
      if step_uses_high[step]:
        graphdef, state, rest = high_noise_graphdef, high_noise_state, high_noise_rest
        guidance_scale = guidance_scale_high
      else:
        graphdef, state, rest = low_noise_graphdef, low_noise_state, low_noise_rest
        guidance_scale = guidance_scale_low

      if is_cache_step:
        # ── Cache step: cond-only forward + FFT frequency compensation ──
        w1, w2 = step_w1w2[step]
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred, cached_noise_cond = transformer_forward_pass_cfg_cache(
            graphdef,
            state,
            rest,
            latents,
            timestep,
            prompt_cond_embeds,
            cached_noise_cond,
            cached_noise_uncond,
            guidance_scale=guidance_scale,
            w1=jnp.float32(w1),
            w2=jnp.float32(w2),
        )
      else:
        # ── Full CFG step: doubled batch, store raw cond/uncond for cache ──
        latents_doubled = jnp.concatenate([latents] * 2)
        timestep = jnp.broadcast_to(t, bsz * 2)
        noise_pred, cached_noise_cond, cached_noise_uncond = transformer_forward_pass_full_cfg(
            graphdef,
            state,
            rest,
            latents_doubled,
            timestep,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
        )

      latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    return latents

  # ── Original non-cache path ──
  # Uses same Python-level if/else transformer selection as the cache path
  # so both paths compile to identical XLA graphs (critical for bfloat16
  # reproducibility in the PSNR comparison).
  timesteps_np = np.array(scheduler_state.timesteps, dtype=np.int32)
  step_uses_high = [bool(timesteps_np[s] >= boundary) for s in range(num_inference_steps)]

  prompt_embeds_combined = (
      jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0) if do_classifier_free_guidance else prompt_embeds
  )

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

    if step_uses_high[step]:
      graphdef, state, rest = high_noise_graphdef, high_noise_state, high_noise_rest
      guidance_scale = guidance_scale_high
    else:
      graphdef, state, rest = low_noise_graphdef, low_noise_state, low_noise_rest
      guidance_scale = guidance_scale_low

    if do_classifier_free_guidance:
      latents_doubled = jnp.concatenate([latents] * 2)
      timestep = jnp.broadcast_to(t, bsz * 2)
      noise_pred, _, _ = transformer_forward_pass_full_cfg(
          graphdef,
          state,
          rest,
          latents_doubled,
          timestep,
          prompt_embeds_combined,
          guidance_scale=guidance_scale,
      )
    else:
      timestep = jnp.broadcast_to(t, bsz)
      noise_pred, latents = transformer_forward_pass(
          graphdef,
          state,
          rest,
          latents,
          timestep,
          prompt_embeds,
          do_classifier_free_guidance,
          guidance_scale,
      )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents
