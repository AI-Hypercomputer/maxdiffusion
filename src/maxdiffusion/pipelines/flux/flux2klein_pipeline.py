"""
Copyright 2026 Google LLC

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

import os
import time
from typing import List, Union, Optional, Any
from PIL import Image

import sys
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
import numpy as np
from flax.linen import partitioning as nn_partitioning

from flax import nnx
from maxdiffusion import max_logging
from maxdiffusion.max_utils import device_put_replicated
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ...models.flux.transformers.transformer_flux_flax import (
    Flux2KleinTransformer2DModel,
    NNXFlux2KleinTransformer2DModel,
)
from ...models.vae_flax import FlaxAutoencoderKL, FlaxDecoderOutput
from ...models.qwen3_flax import FlaxQwen3Model
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, compute_empirical_mu

from ...models.flux.util import (
    pack_latents,
    patchify_latents,
    prepare_latent_image_ids,
    prepare_multi_image_ids,
    prepare_text_ids,
)


class FlaxFlux2KleinPipeline(FlaxDiffusionPipeline):
  """
  Unified end-to-end inference pipeline for Flux.2-klein-4B and 9B models on JAX+TPU.
  Supports dynamic parameter offloading to Host CPU to optimize HBM footprint.
  """

  def __init__(
      self,
      transformer: Union[Flux2KleinTransformer2DModel, NNXFlux2KleinTransformer2DModel],
      vae: FlaxAutoencoderKL,
      text_encoder: FlaxQwen3Model,
      tokenizer,
      scheduler: FlaxFlowMatchScheduler,
      config,
      mesh,
      **kwargs,
  ):
    super().__init__()
    self.register_modules(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    self._config = config
    max_layer = getattr(config, "text_encoder_max_layer", 27)
    if max_layer is not None and max_layer < 27:
      raise ValueError(
          f"Invalid configuration `text_encoder_max_layer={max_layer}`. "
          f"FLUX.2-Klein requires extracting intermediate prompt embeddings from Qwen3 layers 9, 18, and 27, "
          f"so `text_encoder_max_layer` must be >= 27."
      )
    self.mesh = mesh
    self.tokenizer = tokenizer
    if self.tokenizer is None:
      tokenizer_path = getattr(config, "tokenizer_model_name_or_path", None) or getattr(
          config, "pretrained_model_name_or_path", ""
      )
      hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
      repo_cache = os.path.join(
          hf_home,
          "hub",
          f"models--{getattr(config, 'pretrained_model_name_or_path', '').replace('/', '--')}",
          "snapshots",
      )
      if os.path.exists(repo_cache) and os.listdir(repo_cache):
        tokenizer_path = os.path.join(repo_cache, os.listdir(repo_cache)[0])

      from transformers import Qwen2TokenizerFast

      try:
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
      except Exception:
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, subfolder="tokenizer", local_files_only=True)

    # JIT compilation cache
    self._jitted_qwen3_forward = None
    self._jitted_transformer_step = None
    self._jitted_vae_encode = None
    self._jitted_vae_decode = None

  def _setup_jit_functions(self):
    if self._jitted_qwen3_forward is not None:
      return

    @jax.jit
    def qwen3_forward(q_params, ids, mask):
      _, all_hidden_states = self.text_encoder.apply({"params": q_params}, input_ids=ids, attention_mask=mask)
      h_9 = all_hidden_states[9]
      h_18 = all_hidden_states[18]
      h_27 = all_hidden_states[27]
      out = jnp.stack([h_9, h_18, h_27], axis=1)
      prompt_embeds = jnp.transpose(out, (0, 2, 1, 3)).reshape((ids.shape[0], ids.shape[1], -1))
      context_spec = P(None, "context") if "context" in self.mesh.axis_names and self.mesh.shape["context"] > 1 else P()
      prompt_embeds = jax.lax.with_sharding_constraint(prompt_embeds, jax.sharding.NamedSharding(self.mesh, context_spec))
      return prompt_embeds

    if isinstance(self.vae, nnx.Module):
      v_graph, _, v_rest = nnx.split(self.vae, nnx.Param, ...)

      @jax.jit
      def vae_encode(v_params, img):
        merged = nnx.merge(v_graph, v_params, v_rest)
        return merged.encode(img)

      @jax.jit(static_argnums=(4, 5), donate_argnums=(1,))
      def vae_decode(v_params, latents_packed, vae_bn_mean, vae_bn_std, height, width):
        batch_size_val = latents_packed.shape[0]
        h_latent = height // 8
        w_latent = width // 8

        vae_bn_mean_seq = vae_bn_mean.reshape(1, 1, 128)
        vae_bn_std_seq = vae_bn_std.reshape(1, 1, 128)

        latents_bn = latents_packed * vae_bn_std_seq + vae_bn_mean_seq
        latents_unpacked = jnp.reshape(latents_bn, (batch_size_val, h_latent // 2, w_latent // 2, 32, 2, 2))
        latents_unpacked = jnp.transpose(latents_unpacked, (0, 3, 1, 4, 2, 5))
        latents_unpacked = jnp.reshape(latents_unpacked, (batch_size_val, 32, h_latent, w_latent))

        merged = nnx.merge(v_graph, v_params, v_rest)
        res = merged.decode(latents_unpacked)
        return FlaxDecoderOutput(sample=res)

    else:

      @jax.jit
      def vae_encode(v_params, img):
        # FlaxAutoencoderKL expects (B, 3, H, W)
        res = self.vae.apply({"params": v_params}, sample=img, method=self.vae.encode)
        moments = res.latent_dist.mode()
        return jnp.transpose(moments, (0, 3, 1, 2))

      @jax.jit(static_argnums=(4, 5), donate_argnums=(1,))
      def vae_decode(v_params, latents_packed, vae_bn_mean, vae_bn_std, height, width):
        batch_size_val = latents_packed.shape[0]
        h_latent = height // 8
        w_latent = width // 8

        vae_bn_mean_seq = vae_bn_mean.reshape(1, 1, 128)
        vae_bn_std_seq = vae_bn_std.reshape(1, 1, 128)

        latents_bn = latents_packed * vae_bn_std_seq + vae_bn_mean_seq
        latents_unpacked = jnp.reshape(latents_bn, (batch_size_val, h_latent // 2, w_latent // 2, 32, 2, 2))
        latents_unpacked = jnp.transpose(latents_unpacked, (0, 3, 1, 4, 2, 5))
        latents_unpacked = jnp.reshape(latents_unpacked, (batch_size_val, 32, h_latent, w_latent))

        res = self.vae.apply({"params": v_params}, latents=latents_unpacked, method=self.vae.decode)
        return FlaxDecoderOutput(sample=res.sample)

    if isinstance(self.transformer, nnx.Module):
      g, nnx_state, r = nnx.split(self.transformer, nnx.Param, ...)

      @jax.jit
      def transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
        nnx_merged = nnx.merge(g, t_params, r)
        return nnx_merged(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=vec,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=True,
        )

      @jax.jit(static_argnums=(9,))
      def fused_denoise_loop(
          t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timesteps, sigmas, guidance, target_len=None
      ):
        sigmas_padded = jnp.concatenate([sigmas, jnp.array([0.0], dtype=sigmas.dtype)])
        nnx_merged = nnx.merge(g, t_params, r)

        def scan_body(cur_latents, step_idx):
          t_val = timesteps[step_idx]
          t_vec = jnp.broadcast_to(t_val / 1000.0, (cur_latents.shape[0],))
          model_output = nnx_merged(
              hidden_states=cur_latents,
              img_ids=img_ids,
              encoder_hidden_states=prompt_embeds,
              txt_ids=txt_ids,
              pooled_projections=vec,
              timestep=t_vec,
              guidance=guidance,
              return_dict=True,
          )
          sigma = sigmas_padded[step_idx]
          sigma_next = sigmas_padded[step_idx + 1]
          dt = sigma_next - sigma
          v = model_output.sample
          if target_len is not None and cur_latents.shape[1] > target_len:
            target_latents = cur_latents[:, :target_len, :]
            v_target = v[:, :target_len, :]
            next_target = target_latents + v_target * dt
            prev_sample = jnp.concatenate([next_target, cur_latents[:, target_len:, :]], axis=1)
          else:
            prev_sample = cur_latents + v * dt
          return prev_sample, None

        steps = jnp.arange(timesteps.shape[0])
        final_latents, _ = jax.lax.scan(scan_body, latents, steps)
        return final_latents

      @jax.jit(static_argnums=(10, 11))
      def fused_kv_denoise_loop(
          t_params,
          target_latents,
          ref_latents,
          target_img_ids,
          ref_img_ids,
          prompt_embeds,
          txt_ids,
          vec,
          timesteps,
          sigmas,
          guidance=None,
          num_ref_tokens=0,
      ):
        sigmas_padded = jnp.concatenate([sigmas, jnp.array([0.0], dtype=sigmas.dtype)])
        nnx_merged = nnx.merge(g, t_params, r)

        step0_latents = jnp.concatenate([ref_latents, target_latents], axis=1)
        step0_img_ids = jnp.concatenate([ref_img_ids, target_img_ids], axis=1)
        t0_val = timesteps[0]
        t0_vec = jnp.broadcast_to(t0_val / 1000.0, (target_latents.shape[0],))

        out0, kv_cache = nnx_merged(
            hidden_states=step0_latents,
            img_ids=step0_img_ids,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            pooled_projections=vec,
            timestep=t0_vec,
            guidance=guidance,
            return_dict=True,
            kv_cache_mode="extract",
            num_ref_tokens=num_ref_tokens,
        )
        dt0 = sigmas_padded[1] - sigmas_padded[0]
        v0 = out0.sample
        latents_step1 = target_latents + v0 * dt0

        def scan_body(cur_latents, step_idx):
          t_val = timesteps[step_idx]
          t_vec = jnp.broadcast_to(t_val / 1000.0, (cur_latents.shape[0],))
          model_output = nnx_merged(
              hidden_states=cur_latents,
              img_ids=target_img_ids,
              encoder_hidden_states=prompt_embeds,
              txt_ids=txt_ids,
              pooled_projections=vec,
              timestep=t_vec,
              guidance=guidance,
              return_dict=True,
              kv_cache=kv_cache,
              kv_cache_mode="cached",
              num_ref_tokens=num_ref_tokens,
          )
          sigma = sigmas_padded[step_idx]
          sigma_next = sigmas_padded[step_idx + 1]
          dt = sigma_next - sigma
          v = model_output.sample
          next_latents = cur_latents + v * dt
          return next_latents, None

        steps = jnp.arange(1, timesteps.shape[0])
        final_latents, _ = jax.lax.scan(scan_body, latents_step1, steps)
        return final_latents

    else:

      @jax.jit
      def transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
        return self.transformer.apply(
            {"params": t_params},
            hidden_states=latents,
            img_ids=img_ids,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            pooled_projections=vec,
            timestep=timestep,
            guidance=guidance,
        )

      @jax.jit(static_argnums=(9,))
      def fused_denoise_loop(
          t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timesteps, sigmas, guidance, target_len=None
      ):
        sigmas_padded = jnp.concatenate([sigmas, jnp.array([0.0], dtype=sigmas.dtype)])

        def scan_body(cur_latents, step_idx):
          t_val = timesteps[step_idx]
          t_vec = jnp.broadcast_to(t_val / 1000.0, (cur_latents.shape[0],))
          model_output = self.transformer.apply(
              {"params": t_params},
              hidden_states=cur_latents,
              img_ids=img_ids,
              encoder_hidden_states=prompt_embeds,
              txt_ids=txt_ids,
              pooled_projections=vec,
              timestep=t_vec,
              guidance=guidance,
          )
          sigma = sigmas_padded[step_idx]
          sigma_next = sigmas_padded[step_idx + 1]
          dt = sigma_next - sigma
          v = model_output.sample
          if target_len is not None and cur_latents.shape[1] > target_len:
            target_latents = cur_latents[:, :target_len, :]
            v_target = v[:, :target_len, :]
            next_target = target_latents + v_target * dt
            prev_sample = jnp.concatenate([next_target, cur_latents[:, target_len:, :]], axis=1)
          else:
            prev_sample = cur_latents + v * dt
          return prev_sample, None

        steps = jnp.arange(timesteps.shape[0])
        final_latents, _ = jax.lax.scan(scan_body, latents, steps)
        return final_latents

      fused_kv_denoise_loop = fused_denoise_loop

    self._jitted_qwen3_forward = qwen3_forward
    self._jitted_transformer_step = transformer_step
    self._jitted_fused_denoise_loop = fused_denoise_loop
    self._jitted_fused_kv_denoise_loop = fused_kv_denoise_loop
    self._jitted_vae_encode = vae_encode
    self._jitted_vae_decode = vae_decode

  def _get_dynamic_batch_sharding(self):
    """Dynamically infers the batch dimension sharding specification from self.mesh."""
    batch_axes = [axis for axis in ("data", "fsdp") if axis in self.mesh.axis_names and self.mesh.shape[axis] > 1]
    spec = P(tuple(batch_axes)) if batch_axes else P(None)
    return jax.sharding.NamedSharding(self.mesh, spec)

  def compile_aot_async(
      self,
      params,
      vae_params,
      qwen3_params,
      vae_bn_mean,
      vae_bn_std,
      batch_size=1,
      height=1024,
      width=1024,
      images=None,
      image=None,
      num_conditioning_images=0,
      use_kv=None,
  ):
    """Triggers AOT compilation for Qwen3, Flux Transformer, and VAE concurrently using ThreadPoolExecutor."""
    self._setup_jit_functions()
    max_logging.log("🚀 Pre-compiling XLA graphs for Qwen3, Flux Transformer, and VAE concurrently...")
    from concurrent.futures import ThreadPoolExecutor

    if images is None and image is not None:
      images = image if isinstance(image, list) else [image]

    if images is not None and len(images) > 0:
      num_conditioning_images = len(images)

    seq_len_img = (height // 16) * (width // 16)
    total_img_len = (1 + num_conditioning_images) * seq_len_img
    seq_len_txt = self._config.max_sequence_length

    dummy_ids = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
    dummy_mask = jnp.ones((batch_size, seq_len_txt), dtype=jnp.int32)

    dummy_latents = jnp.zeros((batch_size, total_img_len, 128), dtype=jnp.float32)
    dummy_img_ids = jnp.zeros((batch_size, total_img_len, 4), dtype=jnp.int32)
    dummy_prompt_embeds = jnp.zeros((batch_size, seq_len_txt, self.transformer.joint_attention_dim), dtype=jnp.bfloat16)
    dummy_txt_ids = jnp.zeros((batch_size, seq_len_txt, 4), dtype=jnp.float32)
    dummy_t_vec = jnp.zeros((batch_size,), dtype=jnp.float32)

    dummy_target_latents = jnp.zeros((batch_size, seq_len_img, 128), dtype=jnp.float32)
    dummy_bn_mean = jnp.array(vae_bn_mean, dtype=jnp.float32)
    dummy_bn_std = jnp.array(vae_bn_std, dtype=jnp.float32)

    data_sharding = self._get_dynamic_batch_sharding()
    replicated_sharding = jax.sharding.NamedSharding(self.mesh, P())
    context_sharding = jax.sharding.NamedSharding(self.mesh, P(None, "context"))

    def put_data_on_devices(x, sharding):
      if isinstance(x, jax.Array) and hasattr(x, "sharding") and not x.sharding.is_fully_addressable:
        return x
      if hasattr(sharding, "is_fully_addressable") and sharding.is_fully_addressable:
        return jax.device_put(x, sharding)
      return device_put_replicated(x, sharding)

    dummy_ids = put_data_on_devices(dummy_ids, data_sharding)
    dummy_mask = put_data_on_devices(dummy_mask, data_sharding)
    dummy_latents = put_data_on_devices(dummy_latents, data_sharding)
    dummy_img_ids = put_data_on_devices(dummy_img_ids, data_sharding)
    dummy_prompt_embeds = put_data_on_devices(dummy_prompt_embeds, context_sharding)
    dummy_txt_ids = put_data_on_devices(dummy_txt_ids, data_sharding)
    dummy_t_vec = put_data_on_devices(dummy_t_vec, data_sharding)
    dummy_target_latents = put_data_on_devices(dummy_target_latents, data_sharding)
    dummy_bn_mean = put_data_on_devices(dummy_bn_mean, replicated_sharding)
    dummy_bn_std = put_data_on_devices(dummy_bn_std, replicated_sharding)

    def compile_qwen3():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_qwen3_forward.lower(qwen3_params, dummy_ids, dummy_mask).compile()
      max_logging.log(f" -> [AOT COMPILED] Qwen3 Text Encoder in {time.perf_counter() - t0:.2f}s")

    num_steps = getattr(self._config, "num_inference_steps", 4)
    dummy_timesteps = put_data_on_devices(jnp.zeros((num_steps,), dtype=jnp.float32), replicated_sharding)
    dummy_sigmas = put_data_on_devices(jnp.zeros((num_steps + 1,), dtype=jnp.float32), replicated_sharding)

    use_kv = getattr(self._config, "use_kv", False) if use_kv is None else use_kv
    if images is not None and len(images) > 0:
      total_ref_tokens = sum(
          (img.size[1] // 16) * (img.size[0] // 16) if hasattr(img, "size") else seq_len_img for img in images
      )
    else:
      total_ref_tokens = num_conditioning_images * seq_len_img

    dummy_ref_latents = (
        put_data_on_devices(jnp.zeros((batch_size, total_ref_tokens, 128), dtype=jnp.float32), data_sharding)
        if total_ref_tokens > 0
        else None
    )
    dummy_ref_img_ids = (
        put_data_on_devices(jnp.zeros((batch_size, total_ref_tokens, 4), dtype=jnp.int32), data_sharding)
        if total_ref_tokens > 0
        else None
    )
    dummy_target_img_ids = put_data_on_devices(jnp.zeros((batch_size, seq_len_img, 4), dtype=jnp.int32), data_sharding)

    def compile_transformer():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        if use_kv and total_ref_tokens > 0 and self._jitted_fused_kv_denoise_loop is not None:
          self._jitted_fused_kv_denoise_loop.lower(
              params,
              dummy_target_latents,
              dummy_ref_latents,
              dummy_target_img_ids,
              dummy_ref_img_ids,
              dummy_prompt_embeds,
              dummy_txt_ids,
              None,
              dummy_timesteps,
              dummy_sigmas,
              None,
              num_ref_tokens=total_ref_tokens,
          ).compile()
          max_logging.log(f" -> [AOT COMPILED] Fused Flux Transformer KV Denoise Scan in {time.perf_counter() - t0:.2f}s")
        else:
          self._jitted_fused_denoise_loop.lower(
              params,
              dummy_latents,
              dummy_img_ids,
              dummy_prompt_embeds,
              dummy_txt_ids,
              None,
              dummy_timesteps,
              dummy_sigmas,
              None,
              seq_len_img,
          ).compile()
          max_logging.log(f" -> [AOT COMPILED] Fused Flux Transformer Denoise Scan in {time.perf_counter() - t0:.2f}s")

    def compile_vae():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_vae_decode.lower(vae_params, dummy_target_latents, dummy_bn_mean, dummy_bn_std, height, width).compile()
      max_logging.log(f" -> [AOT COMPILED] VAE Decoder in {time.perf_counter() - t0:.2f}s")

    def compile_vae_encode():
      t0 = time.perf_counter()
      dummy_rgb = jnp.zeros((1, 3, height, width), dtype=jnp.float32)
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_vae_encode.lower(vae_params, dummy_rgb).compile()
      max_logging.log(f" -> [AOT COMPILED] VAE Encoder in {time.perf_counter() - t0:.2f}s")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [
          executor.submit(compile_qwen3),
          executor.submit(compile_transformer),
          executor.submit(compile_vae),
      ]
      if num_conditioning_images > 0 or (images is not None and len(images) > 0):
        futures.append(executor.submit(compile_vae_encode))
      for future in futures:
        future.result()
    aot_duration = time.perf_counter() - t_start
    max_logging.log(f"⚡ [AOT CONCURRENT COMPILATION COMPLETE] Total AOT compile time: {aot_duration:.2f}s")
    return aot_duration

  def _prepare_latents(self, config, batch_size, height, width):
    num_channels_latents = 32
    latent_height = height // 8
    latent_width = width // 8
    latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)

    seed_val = getattr(config, "seed", None)
    if seed_val is None:
      seed_val = int(time.time()) & 0x7FFFFFFF
    max_logging.log(
        f"Generating random gaussian noise in unpacked space (32 channels) with seed: {seed_val} and shape: {latent_shape}..."
    )
    np.random.seed(seed_val)
    latents_unpacked = np.random.randn(*latent_shape).astype(np.float32)

    # Pack/patchify noise exactly like PyTorch:
    # (batch, 32, H/16, 2, W/16, 2) -> permute(0, 1, 3, 5, 2, 4) -> reshape(batch, 128, H/16, W/16)
    B, C, H, W = latents_unpacked.shape
    latents_packed = latents_unpacked.reshape(B, C, H // 2, 2, W // 2, 2)
    latents_packed = np.transpose(latents_packed, (0, 1, 3, 5, 2, 4))
    latents_packed = latents_packed.reshape(B, 128, H // 2, W // 2)

    return latents_packed

  def __call__(
      self,
      prompt: Union[str, List[str]],
      params,
      vae_params,
      qwen3_params,
      vae_bn_mean,
      vae_bn_std,
      transformer_shardings,
      vae_shardings,
      qwen3_shardings,
      height: int = 1024,
      width: int = 1024,
      num_inference_steps: int = 4,
      batch_size: int = 1,
      images: Optional[List[Any]] = None,
      image: Optional[Union[Any, List[Any]]] = None,
      use_latents: bool = False,
      latents: Optional[Any] = None,
      measure_time: bool = False,
      warmup: bool = False,
      output_dir: str = "output/",
      output_name: str = "flux2klein_generated_image.png",
      profile_target: Optional[str] = None,
      use_kv: Optional[bool] = None,
  ):
    # 1. Setup JIT functions
    self._setup_jit_functions()

    if images is None and image is not None:
      images = image if isinstance(image, list) else [image]

    # 2. Setup prompts and inputs
    if isinstance(prompt, str):
      prompts = [prompt] * batch_size
    else:
      prompts = prompt

    seq_len_img = (height // 16) * (width // 16)
    seq_len_txt = self._config.max_sequence_length

    # Load or generate latents
    if use_latents and latents is not None:
      latents_jax = jnp.array(latents)
      if latents_jax.ndim == 4:
        B, C, H, W = latents_jax.shape
        if C == 32:
          max_logging.log("  [PIPELINE] Unpacked 32-channel latents detected. Packing using pack_latents...")
          latents_jax = pack_latents(latents_jax)
        elif C == 128:
          latents_jax = jnp.transpose(jnp.reshape(latents_jax, (B, C, H * W)), (0, 2, 1))
        else:
          latents_jax = jnp.transpose(jnp.reshape(latents_jax, (B, C, H * W)), (0, 2, 1))
    else:
      latents_numpy = self._prepare_latents(self._config, batch_size, height, width)
      B, C, H, W = latents_numpy.shape
      latents_jax = jnp.transpose(jnp.reshape(latents_numpy, (B, C, H * W)), (0, 2, 1))

    # RoPE position IDs
    txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
    target_img_ids_val = prepare_latent_image_ids(batch_size, height // 16, width // 16)
    t_pipeline_start = time.perf_counter()
    trace = {}

    # Scheduler
    mu = compute_empirical_mu(seq_len_img, num_inference_steps)
    scheduler_state = self.scheduler.create_state()
    sigmas_custom = jnp.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=jnp.float32)
    scheduler_state = self.scheduler.set_timesteps_ltx2(
        state=scheduler_state,
        num_inference_steps=num_inference_steps,
        shift=mu,
        sigmas=sigmas_custom,
    )

    with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
      proc_id = jax.process_index()
      proc_cnt = jax.process_count()
      host_prefix = f"[HOST {proc_id}/{proc_cnt}] "

      # Shard pipeline batch inputs across data axis ("data") for SPMD multi-host execution
      data_sharding = jax.sharding.NamedSharding(self.mesh, P("data"))

      def put_data_on_devices(x, sharding):
        if isinstance(x, jax.Array) and hasattr(x, "sharding") and not x.sharding.is_fully_addressable:
          return x
        if hasattr(sharding, "is_fully_addressable") and sharding.is_fully_addressable:
          return jax.device_put(x, sharding)
        return device_put_replicated(x, sharding)

      # ---------------------------------------------------------------------
      # PHASE 0: Encode Reference Images (VAE)
      # ---------------------------------------------------------------------
      if images is not None and len(images) > 0:
        t0_vae_enc_start = time.perf_counter()
        trace["start_to_vae_encode"] = t0_vae_enc_start - t_pipeline_start
        max_logging.log(f"{host_prefix} [PHASE 0] Encoding {len(images)} reference image(s) using JAX VAE encoder on TPU...")
        norm_ref_latents = []
        packed_ref_latents = []
        bn_mean_arr = jnp.array(vae_bn_mean, dtype=jnp.float32)
        bn_std_arr = jnp.array(vae_bn_std, dtype=jnp.float32)

        for img in images:
          if isinstance(img, Image.Image):
            img = img.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            arr = np.transpose(arr, (2, 0, 1))
            img_tensor = jnp.expand_dims(jnp.array(arr), axis=0)
          elif isinstance(img, np.ndarray):
            if img.ndim == 3:
              img = np.expand_dims(img, axis=0)
            if img.shape[-1] == 3:
              img = np.transpose(img, (0, 3, 1, 2))
            if np.issubdtype(img.dtype, np.integer):
              img = img.astype(np.float32) / 127.5 - 1.0
            elif np.issubdtype(img.dtype, np.floating):
              if img.max() > 1.0:
                img = img / 127.5 - 1.0
              elif img.min() >= 0.0:
                img = img * 2.0 - 1.0
            img_tensor = jnp.array(img, dtype=np.float32)
          elif isinstance(img, jnp.ndarray):
            if img.ndim == 3:
              img = jnp.expand_dims(img, axis=0)
            if img.shape[-1] == 3:
              img = jnp.transpose(img, (0, 3, 1, 2))
            if jnp.issubdtype(img.dtype, jnp.integer):
              img = img.astype(jnp.float32) / 127.5 - 1.0
            elif jnp.issubdtype(img.dtype, jnp.floating):
              if img.max() > 1.0:
                img = img / 127.5 - 1.0
              elif img.min() >= 0.0:
                img = img * 2.0 - 1.0
            img_tensor = img
          else:
            raise ValueError(f"Unsupported image type: {type(img)}")

          raw_ref_latents = self._jitted_vae_encode(vae_params, img_tensor)
          raw_ref_latents.block_until_ready()
          patchified_ref = patchify_latents(raw_ref_latents)
          normalized_ref = (patchified_ref - bn_mean_arr) / bn_std_arr
          norm_ref_latents.append(normalized_ref)

          packed = jnp.transpose(
              jnp.reshape(normalized_ref, (normalized_ref.shape[0], normalized_ref.shape[1], -1)), (0, 2, 1)
          )
          if packed.shape[0] == 1 and batch_size > 1:
            packed = jnp.repeat(packed, batch_size, axis=0)
          packed_ref_latents.append(packed)

        ref_img_ids_val = prepare_multi_image_ids(norm_ref_latents, scale=10)
        if ref_img_ids_val.shape[0] == 1 and batch_size > 1:
          ref_img_ids_val = jnp.repeat(ref_img_ids_val, batch_size, axis=0)
        ref_latents_jax = jnp.concatenate(packed_ref_latents, axis=1)
        num_ref_tokens = ref_latents_jax.shape[1]
        img_ids_val = jnp.concatenate([target_img_ids_val, ref_img_ids_val], axis=1)
        latents_jax = jnp.concatenate([latents_jax] + packed_ref_latents, axis=1)
        max_logging.log(f"  [PIPELINE] Joint latents shape: {latents_jax.shape}, Joint img_ids shape: {img_ids_val.shape}")

        t0_vae_enc_end = time.perf_counter()
        trace["vae_encode"] = t0_vae_enc_end - t0_vae_enc_start
        trace["image_encoding"] = trace["vae_encode"]
        max_logging.log(f" -> [TIMING] Reference Image Encoding (VAE): {trace['vae_encode']:.4f} seconds ⏱️")
      else:
        img_ids_val = target_img_ids_val
        ref_latents_jax = None
        num_ref_tokens = 0
        trace["vae_encode"] = 0.0
        trace["image_encoding"] = 0.0

      t0_qwen3_start = time.perf_counter()
      if trace.get("vae_encode", 0.0) > 0:
        trace["vae_encode_to_qwen3"] = t0_qwen3_start - t0_vae_enc_end
        max_logging.log(f" -> [TIMING] VAE Encode to Qwen3 Overhead: {trace['vae_encode_to_qwen3']:.4f} seconds ⏱️")
      else:
        trace["start_to_qwen3"] = t0_qwen3_start - t_pipeline_start
        max_logging.log(f" -> [TIMING] Start to Qwen3: {trace['start_to_qwen3']:.4f} seconds ⏱️")

      # ---------------------------------------------------------------------
      # PHASE A: Encode Prompt (Qwen3)
      # ---------------------------------------------------------------------
      if not prompts:
        raise ValueError("Prompt must be provided to FlaxFlux2KleinPipeline")
      if isinstance(prompts, str):
        prompts = [prompts]

      max_logging.log(f"{host_prefix} [PHASE A] Encoding {len(prompts)} prompt(s) using JAX Qwen3 on TPU...")

      try:
        # Tokenize using deterministic explicit template string (version-agnostic across transformers versions)
        templated_texts = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for p in prompts
        ]
        inputs = self.tokenizer(
            templated_texts, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt
        )
        prompt_ids = jnp.array(inputs["input_ids"])
        prompt_mask = jnp.array(inputs["attention_mask"])

        # Run Text Encoding with sharded input arrays matching compile_aot_async
        prompt_ids = put_data_on_devices(prompt_ids, data_sharding)
        prompt_mask = put_data_on_devices(prompt_mask, data_sharding)
        do_prof_qwen3 = profile_target in ("all", "qwen3")
        if do_prof_qwen3:
          tb_dir = getattr(self._config, "tensorboard_dir", "/tmp")
          jax.profiler.start_trace(os.path.join(tb_dir, "profile_qwen3"))
        with jax.named_scope("qwen3_text_encoder"):
          prompt_embeds_jax = self._jitted_qwen3_forward(qwen3_params, prompt_ids, prompt_mask)
        prompt_embeds_jax.block_until_ready()
        if do_prof_qwen3:
          jax.profiler.stop_trace()
      except Exception as e:
        max_logging.log(f"❌ {host_prefix} EXCEPTION IN PHASE A (QWEN3 ENCODING): {e}")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise e

      t0_qwen3_end = time.perf_counter()
      trace["qwen3_encoding"] = t0_qwen3_end - t0_qwen3_start
      trace["prompt_encoding"] = trace["qwen3_encoding"]
      max_logging.log(f" -> [TIMING] Prompt Encoding (Qwen3): {trace['qwen3_encoding']:.4f} seconds ⏱️")

      proc_id = jax.process_index()
      proc_cnt = jax.process_count()
      host_prefix = f"[HOST {proc_id}/{proc_cnt}] "

      # Stage Sync 1: Phase A Complete
      multihost_utils.sync_global_devices("phase_a_complete")
      max_logging.log(f"{host_prefix} Passed Phase A Sync Barrier (phase_a_complete) successfully! ✅")

      latents_jax = put_data_on_devices(latents_jax, data_sharding)
      context_spec = P(None, "context") if "context" in self.mesh.axis_names and self.mesh.shape["context"] > 1 else P()
      context_sharding = jax.sharding.NamedSharding(self.mesh, context_spec)
      prompt_embeds_jax = put_data_on_devices(prompt_embeds_jax, context_sharding)
      txt_ids_val = put_data_on_devices(txt_ids_val, data_sharding)
      img_ids_val = put_data_on_devices(img_ids_val, data_sharding)

      max_logging.log(
          f"{host_prefix} DIAGNOSTIC TENSORS BEFORE PHASE B:\n"
          f"  latents_jax: shape={latents_jax.shape}, dtype={latents_jax.dtype}, sharding={getattr(latents_jax, 'sharding', None)}\n"
          f"  prompt_embeds_jax: shape={prompt_embeds_jax.shape}, dtype={prompt_embeds_jax.dtype}, sharding={getattr(prompt_embeds_jax, 'sharding', None)}\n"
          f"  txt_ids_val: shape={txt_ids_val.shape}, dtype={txt_ids_val.dtype}, sharding={getattr(txt_ids_val, 'sharding', None)}\n"
          f"  img_ids_val: shape={img_ids_val.shape}, dtype={img_ids_val.dtype}, sharding={getattr(img_ids_val, 'sharding', None)}"
      )

      # Stage Sync 2: Pre-Phase B Start
      multihost_utils.sync_global_devices("pre_phase_b_start")
      max_logging.log(f"{host_prefix} Passed Pre-Phase B Sync Barrier (pre_phase_b_start) successfully! ✅")

      t0_denoise_start = time.perf_counter()
      trace["qwen3_to_denoise"] = t0_denoise_start - t0_qwen3_end
      max_logging.log(f" -> [TIMING] Qwen3 to Denoising Overhead: {trace['qwen3_to_denoise']:.4f} seconds ⏱️")

      # ---------------------------------------------------------------------
      # PHASE B: Denoising Loop (Flux Transformer - Standalone Step JIT)
      # ---------------------------------------------------------------------
      steps_to_run = num_inference_steps
      max_logging.log(
          f"{host_prefix} [PHASE B] Running fused {steps_to_run}-step E2E Denoising Loop Scan on a batch of {batch_size} images (warmup={warmup})..."
      )

      try:
        guidance_vec_val = None
        vec_val = None
        replicated_sharding = jax.sharding.NamedSharding(self.mesh, P())
        timesteps_device = put_data_on_devices(scheduler_state.timesteps, replicated_sharding)
        sigmas_device = put_data_on_devices(scheduler_state.sigmas, replicated_sharding)

        do_prof_denoise = profile_target in ("all", "denoise")
        if do_prof_denoise:
          tb_dir = getattr(self._config, "tensorboard_dir", "/tmp")
          jax.profiler.start_trace(os.path.join(tb_dir, "profile_denoise"))

        use_kv = getattr(self._config, "use_kv", False) if use_kv is None else use_kv
        if use_kv and len(packed_ref_latents) > 0:
          ref_latents_device = put_data_on_devices(ref_latents_jax, data_sharding)
          ref_img_ids_device = put_data_on_devices(ref_img_ids_val, data_sharding)
          target_img_ids_device = put_data_on_devices(target_img_ids_val, data_sharding)
          target_latents_device = put_data_on_devices(latents_jax[:, :seq_len_img, :], data_sharding)

          with jax.named_scope("fused_flux_kv_denoise_loop"):
            latents_jax = self._jitted_fused_kv_denoise_loop(
                params,
                target_latents_device,
                ref_latents_device,
                target_img_ids_device,
                ref_img_ids_device,
                prompt_embeds_jax,
                txt_ids_val,
                vec_val,
                timesteps_device,
                sigmas_device,
                guidance_vec_val,
                num_ref_tokens,
            )
            latents_jax.block_until_ready()
        else:
          with jax.named_scope("fused_flux_denoise_loop"):
            latents_jax = self._jitted_fused_denoise_loop(
                params,
                latents_jax,
                img_ids_val,
                prompt_embeds_jax,
                txt_ids_val,
                vec_val,
                timesteps_device,
                sigmas_device,
                guidance_vec_val,
                seq_len_img,
            )
            latents_jax.block_until_ready()
        if do_prof_denoise:
          jax.profiler.stop_trace()

      except Exception as e:
        max_logging.log(f"❌ {host_prefix} EXCEPTION IN DENOISE LOOP: {e}")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise e

      # Stage Sync 3: Phase B Complete
      multihost_utils.sync_global_devices("phase_b_complete")
      max_logging.log(f"{host_prefix} Passed Phase B Sync Barrier (phase_b_complete) successfully! ✅")

      t0_denoise_end = time.perf_counter()
      trace["denoise_loop"] = t0_denoise_end - t0_denoise_start
      max_logging.log(f" -> [TIMING] Denoising Loop (Flux): {trace['denoise_loop']:.4f} seconds ⏱️")

    # ---------------------------------------------------------------------
    # PHASE C: Decode Latents (VAE Decoder)
    # ---------------------------------------------------------------------
    max_logging.log("[PHASE C] Decoding final latents to RGB image using JAX VAE decoder on TPU...")

    # Slice target latents from joint latents if reference images were present
    if latents_jax.shape[1] > seq_len_img:
      latents_jax = latents_jax[:, :seq_len_img, :]

    # Decode VAE latents to RGB pixels using fused JIT vae_decode
    data_sharding = self._get_dynamic_batch_sharding()
    replicated_sharding = jax.sharding.NamedSharding(self.mesh, P())
    latents_jax = put_data_on_devices(latents_jax, data_sharding)
    vae_bn_mean_jax = put_data_on_devices(jnp.array(vae_bn_mean, dtype=jnp.float32), replicated_sharding)
    vae_bn_std_jax = put_data_on_devices(jnp.array(vae_bn_std, dtype=jnp.float32), replicated_sharding)

    t0_vae_start = time.perf_counter()
    trace["denoise_to_vae"] = t0_vae_start - t0_denoise_end
    max_logging.log(f" -> [TIMING] Denoising to VAE Overhead: {trace['denoise_to_vae']:.4f} seconds ⏱️")

    do_prof_vae = profile_target in ("all", "vae")
    if do_prof_vae:
      tb_dir = getattr(self._config, "tensorboard_dir", "/tmp")
      jax.profiler.start_trace(os.path.join(tb_dir, "profile_vae"))
    with jax.named_scope("vae_decoder"):
      decoded_out = self._jitted_vae_decode(vae_params, latents_jax, vae_bn_mean_jax, vae_bn_std_jax, height, width)
    images_rgb = decoded_out.sample
    images_rgb.block_until_ready()
    if do_prof_vae:
      jax.profiler.stop_trace()

    t0_vae_end = time.perf_counter()
    trace["vae_decode"] = t0_vae_end - t0_vae_start
    max_logging.log(f" -> [TIMING] VAE Decoding: {trace['vae_decode']:.4f} seconds ⏱️")

    # ---------------------------------------------------------------------
    # POST-PROCESS: Format and Save Outputs
    # ---------------------------------------------------------------------
    max_logging.log("Postprocessing and saving generated images...")
    saved_paths = []
    # Perform pixel scaling, clamping, and uint8 conversion directly on TPU hardware
    images_uint8 = jnp.clip((images_rgb + 1.0) * 127.5, 0.0, 255.0).astype(jnp.uint8)
    if jax.process_count() > 1:
      images_numpy = multihost_utils.process_allgather(images_uint8, tiled=True)
    else:
      images_numpy = np.array(images_uint8)

    for b_idx in range(batch_size):
      image_np = np.array(images_numpy[b_idx])
      # Transpose channel dimension if shape is (C, H, W) instead of (H, W, C)
      if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)

      img = Image.fromarray(image_np)

      # Formulate output filename for this batch index
      if batch_size > 1:
        batch_output_name = output_name.replace(".png", f"_b{b_idx}.png")
      else:
        batch_output_name = output_name

      output_png_path = os.path.join(output_dir, batch_output_name)
      img.save(output_png_path, format="PNG", compress_level=1)
      max_logging.log(f" -> Saved image: {output_png_path} | Prompt: '{prompts[b_idx]}'")
      saved_paths.append(output_png_path)

    t0_save_end = time.perf_counter()
    trace["image_saving"] = t0_save_end - t0_vae_end
    trace["e2e_pipeline_total"] = t0_save_end - t_pipeline_start

    max_logging.log(f" -> [TIMING] Image Saving: {trace['image_saving']:.4f} seconds ⏱️")
    max_logging.log(f" -> [TIMING] E2E Pipeline Total: {trace['e2e_pipeline_total']:.4f} seconds ⏱️")

    return saved_paths, trace
