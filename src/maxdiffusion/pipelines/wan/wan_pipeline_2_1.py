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

from .wan_pipeline import WanPipeline, transformer_forward_pass, transformer_forward_pass_full_cfg, transformer_forward_pass_cfg_cache, init_magcache, magcache_step
from ...schedulers.scheduling_unipc_multistep_flax import UniPCMultistepSchedulerState
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
            subfolder="transformer",
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
    pipeline, transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    pipeline.transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
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
      use_cfg_cache: bool = False,
      use_magcache: bool = False,
      magcache_thresh: Optional[float] = None,
      magcache_K: Optional[int] = None,
      retention_ratio: Optional[float] = None,
  ):
    config = getattr(self, "config", None)
    if magcache_thresh is None:
      magcache_thresh = getattr(config, "magcache_thresh", 0.12)
    if magcache_K is None:
      magcache_K = getattr(config, "magcache_K", 2)
    if retention_ratio is None:
      retention_ratio = getattr(config, "retention_ratio", 0.2)

    if use_cfg_cache and guidance_scale <= 1.0:
      raise ValueError(
          f"use_cfg_cache=True requires guidance_scale > 1.0 (got {guidance_scale}). "
          "CFG cache accelerates classifier-free guidance, which is disabled when guidance_scale <= 1.0."
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

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    # Use fori_loop path when no caching is enabled for reduced dispatch overhead.
    use_fori = not use_cfg_cache and not use_magcache

    if use_fori:
      do_cfg = guidance_scale > 1.0
      p_run_inference = partial(
          run_inference_fori_2_1,
          do_cfg=do_cfg,
          guidance_scale=guidance_scale,
          num_inference_steps=num_inference_steps,
          scheduler=self.scheduler,
      )
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        latents = p_run_inference(
            graphdef=graphdef,
            sharded_state=state,
            rest_of_state=rest_of_state,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            scheduler_state=scheduler_state,
        )
        latents = self._denormalize_latents(latents)
      return self._decode_latents_to_video(latents)

    p_run_inference = partial(
        run_inference_2_1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
        use_cfg_cache=use_cfg_cache,
        use_magcache=use_magcache,
        magcache_thresh=magcache_thresh,
        magcache_K=magcache_K,
        retention_ratio=retention_ratio,
        height=height,
        mag_ratios_base=getattr(config, "mag_ratios_base", None),
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


@partial(jax.jit, static_argnames=("do_cfg", "guidance_scale", "num_inference_steps", "scheduler"))
def run_inference_fori_2_1(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    scheduler_state: UniPCMultistepSchedulerState,
    do_cfg: bool,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
):
  """Denoising loop for WAN 2.1 T2V using jax.lax.fori_loop.

  The entire denoising loop runs as a single XLA program, eliminating
  per-step Python dispatch overhead. This path is used when no caching
  (CFG-Cache, MagCache) is enabled.
  """
  bsz = latents.shape[0]

  # Pre-combine embeddings for CFG (static at trace time).
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
  else:
    prompt_embeds_combined = prompt_embeds

  # Pre-initialize scheduler state with concrete values so the pytree
  # structure is consistent across all fori_loop iterations.
  # - step_index: must be int (not None) so scheduler.step skips _init_step_index
  # - last_sample: must be array (not None) for consistent pytree; the
  #   corrector is still skipped on step 0 because step_index > 0 is False
  # - begin_index: set to 0 so it's a concrete int rather than None
  scheduler_state = scheduler_state.replace(
      step_index=0,
      last_sample=jnp.zeros_like(latents),
      begin_index=0,
  )

  def body_fn(step, carry):
    latents, sched_state = carry
    t = sched_state.timesteps[step]

    wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)

    if do_cfg:
      latents_input = jnp.concatenate([latents] * 2)
      timestep = jnp.broadcast_to(t, (bsz * 2,))
    else:
      latents_input = latents
      timestep = jnp.broadcast_to(t, (bsz,))

    noise_pred = wan_transformer(
        hidden_states=latents_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds_combined,
    )

    if do_cfg:
      noise_cond = noise_pred[:bsz]
      noise_uncond = noise_pred[bsz:]
      noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    latents, sched_state = scheduler.step(sched_state, noise_pred, t, latents).to_tuple()
    return latents, sched_state

  latents, _ = jax.lax.fori_loop(0, num_inference_steps, body_fn, (latents, scheduler_state))
  return latents


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
    use_cfg_cache: bool = False,
    use_magcache: bool = False,
    magcache_thresh: float = 0.12,
    magcache_K: int = 2,
    retention_ratio: float = 0.2,
    height: int = 480,
    mag_ratios_base: Optional[List[float]] = None,
):
  """Denoising loop for WAN 2.1 T2V with FasterCache CFG-Cache.

  CFG-Cache strategy (Lv et al., ICLR 2025, enabled via use_cfg_cache=True):
  - Full CFG steps  : run transformer on [cond, uncond] batch (batch×2).
                      Cache raw noise_cond and noise_uncond for FFT bias.
  - Cache steps     : run transformer on cond batch only (batch×1).
                      Estimate uncond via FFT frequency-domain compensation:
                        ΔF = FFT(cached_uncond) - FFT(cached_cond)
                        Split ΔF into low-freq (ΔLF) and high-freq (ΔHF).
                        uncond_approx = IFFT(FFT(new_cond) + w1*ΔLF + w2*ΔHF)
                      Phase-dependent weights (α=0.2):
                        Early (high noise): w1=1.2, w2=1.0 (boost low-freq)
                        Late  (low noise):  w1=1.0, w2=1.2 (boost high-freq)
  - Schedule        : full CFG for the first 1/3 of steps, then
                      full CFG every 5 steps, cache the rest.

  Two separately-compiled JAX-jitted functions handle full and cache steps so
  XLA sees static shapes throughout — the key requirement for TPU efficiency.
  """
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]

  # Resolution-dependent CFG cache config (FasterCache / MixCache guidance)
  if height >= 720:
    # 720p: conservative — protect last 40%, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = int(num_inference_steps * 0.9)
    cfg_cache_alpha = 0.2
  else:
    # 480p: moderate — protect last 2 steps, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = num_inference_steps - 2
    cfg_cache_alpha = 0.2

  # Pre-split embeds once, outside the loop.
  prompt_cond_embeds = prompt_embeds
  prompt_embeds_combined = None
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

  # Pre-compute cache schedule and phase-dependent weights.
  # t₀ = midpoint step; before t₀ boost low-freq, after boost high-freq.
  t0_step = num_inference_steps // 2
  first_full_step_seen = False
  step_is_cache = []
  step_w1w2 = []
  for s in range(num_inference_steps):
    is_cache = (
        use_cfg_cache
        and do_cfg
        and first_full_step_seen
        and s >= cfg_cache_start_step
        and s < cfg_cache_end_step
        and (s - cfg_cache_start_step) % cfg_cache_interval != 0
    )
    step_is_cache.append(is_cache)
    if not is_cache:
      first_full_step_seen = True
    # Phase-dependent weights: w = 1 + α·I(condition)
    if s < t0_step:
      step_w1w2.append((1.0 + cfg_cache_alpha, 1.0))  # early: boost low-freq
    else:
      step_w1w2.append((1.0, 1.0 + cfg_cache_alpha))  # late: boost high-freq

  # Cache tensors (on-device JAX arrays, initialised to None).
  cached_noise_cond = None
  cached_noise_uncond = None

  if use_magcache and do_cfg:
    magcache_init = init_magcache(num_inference_steps, retention_ratio, mag_ratios_base)
    accumulated_state = magcache_init[:6]
    cached_residual = magcache_init[6]
    skip_warmup = magcache_init[7]
    mag_ratios = magcache_init[8]

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

    if use_magcache and do_cfg:
      timestep = jnp.broadcast_to(t, bsz * 2 if do_cfg else bsz)

      skip_blocks, accumulated_state = magcache_step(
          step, mag_ratios, accumulated_state, magcache_thresh, magcache_K, skip_warmup
      )

      noise_pred, latents, residual_x_cur = transformer_forward_pass(
          graphdef,
          sharded_state,
          rest_of_state,
          jnp.concatenate([latents] * 2) if do_cfg else latents,
          timestep,
          prompt_embeds_combined if do_cfg else prompt_cond_embeds,
          do_classifier_free_guidance=do_cfg,
          guidance_scale=guidance_scale,
          skip_blocks=bool(skip_blocks),
          cached_residual=cached_residual,
          return_residual=True,
      )

      if not skip_blocks:
        cached_residual = residual_x_cur

    else:
      is_cache_step = step_is_cache[step]

      if is_cache_step:
        w1, w2 = step_w1w2[step]
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred, cached_noise_cond = transformer_forward_pass_cfg_cache(
            graphdef,
            sharded_state,
            rest_of_state,
            latents,
            timestep,
            prompt_cond_embeds,
            cached_noise_cond,
            cached_noise_uncond,
            guidance_scale=guidance_scale,
            w1=jnp.float32(w1),
            w2=jnp.float32(w2),
        )

      elif do_cfg:
        latents_doubled = jnp.concatenate([latents] * 2)
        timestep = jnp.broadcast_to(t, bsz * 2)
        noise_pred, cached_noise_cond, cached_noise_uncond = transformer_forward_pass_full_cfg(
            graphdef,
            sharded_state,
            rest_of_state,
            latents_doubled,
            timestep,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
        )

      else:
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred, latents = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            latents,
            timestep,
            prompt_cond_embeds,
            do_classifier_free_guidance=False,
            guidance_scale=guidance_scale,
        )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

  return latents
