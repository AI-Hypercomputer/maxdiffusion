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

from maxdiffusion import max_logging
from maxdiffusion.max_utils import device_put_replicated
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ...models.flux.transformers.transformer_flux_flax import Flux2KleinTransformer2DModel
from ...models.vae_flax import FlaxAutoencoderKL
from ...models.qwen3_flax import FlaxQwen3Model
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, compute_empirical_mu

from ...models.flux.util import (
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
    prepare_text_ids,
)


class FlaxFlux2KleinPipeline(FlaxDiffusionPipeline):
  """
  Unified end-to-end inference pipeline for Flux.2-klein-4B and 9B models on JAX+TPU.
  Supports dynamic parameter offloading to Host CPU to optimize HBM footprint.
  """

  def __init__(
      self,
      transformer: Flux2KleinTransformer2DModel,
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
    self.mesh = mesh

    # JIT compilation cache
    self._jitted_qwen3_forward = None
    self._jitted_transformer_step = None
    self._jitted_vae_decode = None

  def _setup_jit_functions(self):
    if self._jitted_qwen3_forward is not None:
      return

    @jax.jit
    def qwen3_forward(q_params, ids, mask):
      return self.text_encoder.apply({"params": q_params}, input_ids=ids, attention_mask=mask)

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

    @jax.jit(donate_argnums=(1,))
    def vae_decode(v_params, latents_unpatched):
      return self.vae.apply({"params": v_params}, latents=latents_unpatched, method=self.vae.decode)

    self._jitted_qwen3_forward = qwen3_forward
    self._jitted_transformer_step = transformer_step
    self._jitted_vae_decode = vae_decode

  def compile_aot_async(self, params, vae_params, qwen3_params, batch_size=1, height=1024, width=1024):
    """Triggers AOT compilation for Qwen3, Flux Transformer, and VAE concurrently using ThreadPoolExecutor."""
    self._setup_jit_functions()
    max_logging.log("🚀 Pre-compiling XLA graphs for Qwen3, Flux Transformer, and VAE concurrently...")
    from concurrent.futures import ThreadPoolExecutor

    seq_len_img = (height // 16) * (width // 16)
    seq_len_txt = self._config.max_sequence_length

    dummy_ids = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
    dummy_mask = jnp.ones((batch_size, seq_len_txt), dtype=jnp.int32)

    dummy_latents = jnp.zeros((batch_size, seq_len_img, 128), dtype=jnp.float32)
    dummy_img_ids = jnp.zeros((batch_size, seq_len_img, 4), dtype=jnp.int32)
    dummy_prompt_embeds = jnp.zeros((batch_size, seq_len_txt, 12288), dtype=jnp.bfloat16)
    dummy_txt_ids = jnp.zeros((batch_size, seq_len_txt, 4), dtype=jnp.float32)
    dummy_t_vec = jnp.zeros((batch_size,), dtype=jnp.float32)

    dummy_unpacked_latents = jnp.zeros((batch_size, 32, height // 8, width // 8), dtype=jnp.float32)

    def compile_qwen3():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_qwen3_forward.lower(qwen3_params, dummy_ids, dummy_mask).compile()
      max_logging.log(f" -> [AOT COMPILED] Qwen3 Text Encoder in {time.perf_counter() - t0:.2f}s")

    def compile_transformer():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_transformer_step.lower(
            params, dummy_latents, dummy_img_ids, prompt_embeds=dummy_prompt_embeds, txt_ids=dummy_txt_ids, vec=None, timestep=dummy_t_vec, guidance=None
        ).compile()
      max_logging.log(f" -> [AOT COMPILED] Flux Transformer Step in {time.perf_counter() - t0:.2f}s")

    def compile_vae():
      t0 = time.perf_counter()
      with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
        self._jitted_vae_decode.lower(vae_params, dummy_unpacked_latents).compile()
      max_logging.log(f" -> [AOT COMPILED] VAE Decoder in {time.perf_counter() - t0:.2f}s")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
      futures = [
          executor.submit(compile_qwen3),
          executor.submit(compile_transformer),
          executor.submit(compile_vae),
      ]
      for future in futures:
        future.result()
    max_logging.log(f"⚡ [AOT CONCURRENT COMPILATION COMPLETE] Total AOT compile time: {time.perf_counter() - t_start:.2f}s")

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
      use_latents: bool = False,
      latents: Optional[Any] = None,
      measure_time: bool = False,
      warmup: bool = False,
      output_dir: str = "output/",
      output_name: str = "flux2klein_generated_image.png",
  ):
    # 1. Setup JIT functions
    self._setup_jit_functions()

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
        else:
          latents_jax = jnp.transpose(jnp.reshape(latents_jax, (B, C, H * W)), (0, 2, 1))
    else:
      latents_numpy = self._prepare_latents(self._config, batch_size, height, width)
      B, C, H, W = latents_numpy.shape
      latents_jax = jnp.transpose(jnp.reshape(latents_numpy, (B, C, H * W)), (0, 2, 1))

    # RoPE position IDs
    txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
    img_ids_val = prepare_latent_image_ids(batch_size, height // 16, width // 16)

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

    trace = {}

    with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
      proc_id = jax.process_index()
      proc_cnt = jax.process_count()
      host_prefix = f"[HOST {proc_id}/{proc_cnt}] "

      # ---------------------------------------------------------------------
      # PHASE A: Encode Prompt (Qwen3)
      # ---------------------------------------------------------------------
      print(f"{host_prefix} [PHASE A] Encoding {len(prompts)} prompt(s) using JAX Qwen3 on TPU...", flush=True)
      t0 = time.perf_counter()

      try:
        # Resolve tokenizer path from config
        tokenizer_path = self._config.tokenizer_model_name_or_path
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        repo_cache = os.path.join(
            hf_home, "hub", f"models--{self._config.pretrained_model_name_or_path.replace('/', '--')}", "snapshots"
        )
        if os.path.exists(repo_cache) and os.listdir(repo_cache):
          tokenizer_path = os.path.join(repo_cache, os.listdir(repo_cache)[0])

        from transformers import Qwen2TokenizerFast

        try:
          tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception:
          tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, subfolder="tokenizer", local_files_only=True)

        # Tokenize using deterministic explicit template string (version-agnostic across transformers versions)
        templated_texts = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for p in prompts
        ]
        inputs = tokenizer(
            templated_texts, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt
        )
        prompt_ids = jnp.array(inputs["input_ids"])
        prompt_mask = jnp.array(inputs["attention_mask"])

        # Run Text Encoding
        hidden_states, all_hidden_states = self._jitted_qwen3_forward(qwen3_params, prompt_ids, prompt_mask)

        # Stack layers 9, 18, 27 to form prompt embeddings
        h_9 = all_hidden_states[9]
        h_18 = all_hidden_states[18]
        h_27 = all_hidden_states[27]
        out = jnp.stack([h_9, h_18, h_27], axis=1)
        # Transpose shape to [B, seq_len, 3*hidden_size]
        prompt_embeds_jax = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len_txt, -1))
        prompt_embeds_jax.block_until_ready()
      except Exception as e:
        print(f"❌ {host_prefix} EXCEPTION IN PHASE A (QWEN3 ENCODING): {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise e

      trace["prompt_encoding"] = time.perf_counter() - t0
      max_logging.log(f" -> [TIMING] Prompt Encoding (Qwen3): {trace['prompt_encoding']:.4f} seconds ⏱️")

      proc_id = jax.process_index()
      proc_cnt = jax.process_count()
      host_prefix = f"[HOST {proc_id}/{proc_cnt}] "

      # Stage Sync 1: Phase A Complete
      multihost_utils.sync_global_devices("phase_a_complete")
      print(f"{host_prefix} Passed Phase A Sync Barrier (phase_a_complete) successfully! ✅", flush=True)

      # Shard pipeline batch inputs across data axis ("data") for SPMD multi-host execution
      data_sharding = jax.sharding.NamedSharding(self.mesh, P("data"))

      def put_data_on_devices(x, sharding):
        if isinstance(x, jax.Array) and hasattr(x, "sharding") and not x.sharding.is_fully_addressable:
          return x
        if hasattr(sharding, "is_fully_addressable") and sharding.is_fully_addressable:
          return jax.device_put(x, sharding)
        return device_put_replicated(x, sharding)

      latents_jax = put_data_on_devices(latents_jax, data_sharding)
      prompt_embeds_jax = put_data_on_devices(prompt_embeds_jax, data_sharding)
      txt_ids_val = put_data_on_devices(txt_ids_val, data_sharding)
      img_ids_val = put_data_on_devices(img_ids_val, data_sharding)

      print(
          f"{host_prefix} DIAGNOSTIC TENSORS BEFORE PHASE B:\n"
          f"  latents_jax: shape={latents_jax.shape}, dtype={latents_jax.dtype}, sharding={getattr(latents_jax, 'sharding', None)}\n"
          f"  prompt_embeds_jax: shape={prompt_embeds_jax.shape}, dtype={prompt_embeds_jax.dtype}, sharding={getattr(prompt_embeds_jax, 'sharding', None)}\n"
          f"  txt_ids_val: shape={txt_ids_val.shape}, dtype={txt_ids_val.dtype}, sharding={getattr(txt_ids_val, 'sharding', None)}\n"
          f"  img_ids_val: shape={img_ids_val.shape}, dtype={img_ids_val.dtype}, sharding={getattr(img_ids_val, 'sharding', None)}",
          flush=True,
      )

      # Stage Sync 2: Pre-Phase B Start
      multihost_utils.sync_global_devices("pre_phase_b_start")
      print(f"{host_prefix} Passed Pre-Phase B Sync Barrier (pre_phase_b_start) successfully! ✅", flush=True)

      # ---------------------------------------------------------------------
      # PHASE B: Denoising Loop (Flux Transformer - Standalone Step JIT)
      # ---------------------------------------------------------------------
      steps_to_run = 1 if warmup else num_inference_steps
      print(
          f"{host_prefix} [PHASE B] Running {steps_to_run}-step E2E Denoising Loop on a batch of {batch_size} images (warmup={warmup})...",
          flush=True,
      )
      t0 = time.perf_counter()

      try:
        guidance_vec_val = None
        vec_val = None

        for step_idx in range(steps_to_run):
          timestep = scheduler_state.timesteps[step_idx]
          t_vec = jnp.full((batch_size,), timestep / 1000.0, dtype=latents_jax.dtype)

          model_output = self._jitted_transformer_step(
              params, latents_jax, img_ids_val, prompt_embeds_jax, txt_ids_val, vec_val, t_vec, guidance_vec_val
          )

          prev_sample, _ = self.scheduler.step(
              state=scheduler_state,
              model_output=model_output.sample,
              timestep=scheduler_state.timesteps[step_idx],
              sample=latents_jax,
              return_dict=False,
          )
          latents_jax = prev_sample

        latents_jax.block_until_ready()
      except Exception as e:
        print(f"❌ {host_prefix} EXCEPTION IN DENOISE LOOP: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise e

      # Stage Sync 3: Phase B Complete
      multihost_utils.sync_global_devices("phase_b_complete")
      print(f"{host_prefix} Passed Phase B Sync Barrier (phase_b_complete) successfully! ✅", flush=True)

      trace["denoise_loop"] = time.perf_counter() - t0
      max_logging.log(f" -> [TIMING] Denoising Loop (Flux): {trace['denoise_loop']:.4f} seconds ⏱️")

    # ---------------------------------------------------------------------
    # PHASE C: Decode Latents (VAE Decoder)
    # ---------------------------------------------------------------------
    max_logging.log("[PHASE C] Decoding final latents to RGB image using JAX VAE decoder on TPU...")
    t0 = time.perf_counter()

    # Apply Channel-wise Batch Normalization Scaling in packed sequence format (denormalize)
    vae_bn_mean_seq = vae_bn_mean.reshape(1, 1, 128)
    vae_bn_std_seq = vae_bn_std.reshape(1, 1, 128)
    latents_bn = latents_jax * vae_bn_std_seq + vae_bn_mean_seq

    # Unpack packed latents back to spatial grid
    latents_unpacked = unpack_latents(latents_bn, batch_size, 32, height, width)

    # Decode VAE latents to RGB pixels
    decoded_out = self._jitted_vae_decode(vae_params, latents_unpacked)
    # VAE output is in decoded_out.sample
    images_rgb = decoded_out.sample
    images_rgb.block_until_ready()

    trace["vae_decode"] = time.perf_counter() - t0
    max_logging.log(f" -> [TIMING] VAE Decoding: {trace['vae_decode']:.4f} seconds ⏱️")

    # ---------------------------------------------------------------------
    # POST-PROCESS: Format and Save Outputs
    # ---------------------------------------------------------------------
    max_logging.log("Postprocessing and saving generated images...")
    saved_paths = []
    # Clamp pixels and scale to [0, 255]
    images_rgb = jnp.clip((images_rgb + 1.0) / 2.0, 0.0, 1.0)
    if jax.process_count() > 1:
      images_numpy = multihost_utils.process_allgather(images_rgb, tiled=True)
    else:
      images_numpy = np.array(images_rgb)

    for b_idx in range(batch_size):
      image_np = np.array(images_numpy[b_idx] * 255.0, dtype=np.uint8)
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
      img.save(output_png_path)
      max_logging.log(f" -> Saved image: {output_png_path} | Prompt: '{prompts[b_idx]}'")
      saved_paths.append(output_png_path)

    return saved_paths, trace
