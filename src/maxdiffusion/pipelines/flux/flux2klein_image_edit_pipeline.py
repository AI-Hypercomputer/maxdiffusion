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
from typing import List, Union, Optional, Tuple, Any
from PIL import Image

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from flax import nnx
from einops import rearrange

from maxdiffusion import max_logging
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ...models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from ...models.flux.vae.autoencoder_kl_flux2_nnx import NNXAutoencoderKLFlux2
from ...models.qwen3_flax import FlaxQwen3Model
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, compute_empirical_mu
from ...models.flux.util import (
    pack_latents,
    patchify_latents,
    unpatchify_latents,
    prepare_latent_image_ids,
    prepare_multi_image_ids,
    prepare_text_ids,
    prepare_image_latents,
)


class FlaxFlux2KleinImageEditPipeline(FlaxDiffusionPipeline):
  """
  Unified end-to-end multi-image editing pipeline for FLUX.2-Klein (4B and 9B) in pure Flax NNX.
  Supports conditioning on N arbitrary reference images and generating edited target images on JAX/TPU.
  """

  def __init__(
      self,
      transformer: NNXFlux2KleinTransformer2DModel,
      vae: NNXAutoencoderKLFlux2,
      text_encoder: FlaxQwen3Model,
      tokenizer: Any,
      scheduler: FlaxFlowMatchScheduler,
      config: Any,
      mesh: Optional[jax.sharding.Mesh] = None,
      vae_bn_mean: Optional[jnp.ndarray] = None,
      vae_bn_std: Optional[jnp.ndarray] = None,
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
    self.tokenizer = tokenizer
    self.vae_bn_mean = vae_bn_mean
    self.vae_bn_std = vae_bn_std

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

    # JIT cache
    self._jitted_qwen3_forward = None
    self._jitted_transformer_step = None
    self._jitted_vae_encode = None
    self._jitted_vae_decode = None

  def _setup_jit_functions(self):
    if self._jitted_qwen3_forward is not None:
      return

    # 1. Qwen3 forward
    @jax.jit
    def qwen3_forward(q_params, ids, mask):
      _, all_hidden_states = self.text_encoder.apply({"params": q_params}, input_ids=ids, attention_mask=mask)
      h_9 = all_hidden_states[9]
      h_18 = all_hidden_states[18]
      h_27 = all_hidden_states[27]
      out = jnp.stack([h_9, h_18, h_27], axis=1)
      prompt_embeds = jnp.transpose(out, (0, 2, 1, 3)).reshape((ids.shape[0], ids.shape[1], -1))
      return prompt_embeds

    self._jitted_qwen3_forward = qwen3_forward

    # 2. NNX Transformer Step
    t_graph, _, t_rest = nnx.split(self.transformer, nnx.Param, ...)

    @jax.jit
    def transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, timestep, guidance=None):
      merged = nnx.merge(t_graph, t_params, t_rest)
      return merged(
          hidden_states=latents,
          encoder_hidden_states=prompt_embeds,
          timestep=timestep,
          img_ids=img_ids,
          txt_ids=txt_ids,
          guidance=guidance,
      ).sample

    self._jitted_transformer_step = transformer_step

    # 3. NNX VAE Encode & Decode
    v_graph, _, v_rest = nnx.split(self.vae, nnx.Param, ...)

    @jax.jit
    def vae_encode(v_params, img):
      merged = nnx.merge(v_graph, v_params, v_rest)
      return merged.encode(img)

    @jax.jit
    def vae_decode(v_params, latents):
      merged = nnx.merge(v_graph, v_params, v_rest)
      return merged.decode(latents)

    self._jitted_vae_encode = vae_encode
    self._jitted_vae_decode = vae_decode

  def preprocess_image(self, image: Union[Image.Image, np.ndarray, jnp.ndarray], height: int = 512, width: int = 512) -> jnp.ndarray:
    """Preprocesses a single image to normalized (1, 3, H, W) in [-1, 1]."""
    if isinstance(image, Image.Image):
      image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
      arr = np.array(image, dtype=np.float32) / 127.5 - 1.0  # (H, W, 3)
      arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
      return jnp.expand_dims(jnp.array(arr), axis=0)
    elif isinstance(image, np.ndarray):
      if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
      if image.shape[-1] == 3:
        image = np.transpose(image, (0, 3, 1, 2))
      if image.max() > 1.0:
        image = image / 127.5 - 1.0
      return jnp.array(image, dtype=jnp.float32)
    elif isinstance(image, jnp.ndarray):
      if image.ndim == 3:
        image = jnp.expand_dims(image, axis=0)
      if image.shape[-1] == 3:
        image = jnp.transpose(image, (0, 3, 1, 2))
      return image
    else:
      raise ValueError(f"Unsupported image type: {type(image)}")

  def prepare_reference_latents(
      self,
      images: List[Union[Image.Image, np.ndarray, jnp.ndarray]],
      vae_params: Any,
      bn_mean: jnp.ndarray,
      bn_std: jnp.ndarray,
      scale: int = 10,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Encodes, patchifies, normalizes, and generates 4D RoPE IDs for reference images."""
    norm_latents = []
    for img in images:
      img_tensor = self.preprocess_image(img)
      raw_latents = self._jitted_vae_encode(vae_params, img_tensor)  # (1, 32, H/8, W/8)
      patchified = patchify_latents(raw_latents)  # (1, 128, H/16, W/16)
      normalized = (patchified - bn_mean) / bn_std
      norm_latents.append(normalized)

    image_latent_ids = prepare_multi_image_ids(norm_latents, scale=scale)

    packed_latents = []
    for latent in norm_latents:
      packed = rearrange(latent, "b c h w -> b (h w) c")
      packed_latents.append(packed)

    image_latents_concat = jnp.concatenate(packed_latents, axis=1)
    return image_latents_concat, image_latent_ids

  def __call__(
      self,
      prompt: str,
      images: List[Union[Image.Image, np.ndarray, jnp.ndarray]],
      height: int = 512,
      width: int = 512,
      num_inference_steps: int = 4,
      guidance_scale: Optional[float] = None,
      prng_key: Optional[jax.Array] = None,
      transformer_params: Optional[Any] = None,
      vae_params: Optional[Any] = None,
      text_encoder_params: Optional[Any] = None,
      output_type: str = "pil",
      return_dict: bool = True,
      **kwargs,
  ) -> Any:
    """Runs end-to-end FLUX.2-Klein multi-image editing inference."""
    self._setup_jit_functions()

    prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)
    key_noise, key_step = jax.random.split(prng_key)

    # Resolve model parameters
    t_params = transformer_params or nnx.state(self.transformer, nnx.Param)
    v_params = vae_params or nnx.state(self.vae, nnx.Param)
    q_params = text_encoder_params

    bn_mean = self.vae_bn_mean
    bn_std = self.vae_bn_std

    # 1. Encode Text Prompt
    prompt_embeds = kwargs.get("prompt_embeds", None)
    if prompt_embeds is None:
      max_logging.log(f"Encoding prompt: '{prompt}'...")
      templated_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
      tokens = self.tokenizer(
          templated_text,
          max_length=getattr(self._config, "max_sequence_length", 512),
          padding="max_length",
          truncation=True,
          return_tensors="np",
      )
      input_ids = jnp.array(tokens["input_ids"], dtype=jnp.int32)
      attention_mask = jnp.array(tokens["attention_mask"], dtype=jnp.int32)

      prompt_embeds = self._jitted_qwen3_forward(q_params, input_ids, attention_mask)
    else:
      prompt_embeds = jnp.array(prompt_embeds, dtype=jnp.bfloat16)

    text_ids = prepare_text_ids(prompt_embeds.shape[0], prompt_embeds.shape[1])

    # 2. Encode Reference Images
    max_logging.log(f"Preparing {len(images)} reference image latents...")
    ref_latents_concat, ref_latent_ids = self.prepare_reference_latents(
        images, v_params, bn_mean, bn_std, scale=10
    )

    # 3. Initialize Target Generation Noise & RoPE IDs
    h_latent = height // 8
    w_latent = width // 8
    num_gen_tokens = (h_latent // 2) * (w_latent // 2)
    target_noise = jax.random.normal(key_noise, (1, num_gen_tokens, 128), dtype=prompt_embeds.dtype)
    gen_latent_ids = prepare_latent_image_ids(1, h_latent // 2, w_latent // 2)

    # 4. Joint Sequence Setup
    joint_image_ids = jnp.concatenate([gen_latent_ids, ref_latent_ids], axis=1)
    latents_gen = target_noise

    # Compute FlowMatch Scheduler Timesteps
    image_seq_len = num_gen_tokens
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    scheduler_state = self.scheduler.create_state()
    sigmas_custom = jnp.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=jnp.float32)
    scheduler_state = self.scheduler.set_timesteps_ltx2(
        state=scheduler_state,
        num_inference_steps=num_inference_steps,
        shift=mu,
        sigmas=sigmas_custom,
    )
    timesteps = scheduler_state.timesteps

    # 5. Denoising Loop
    max_logging.log(f"Running {num_inference_steps}-step Euler denoising trajectory...")
    for i, t in enumerate(timesteps):
      t_curr = float(t)
      t_prev = float(timesteps[i + 1]) if i < len(timesteps) - 1 else 0.0

      joint_image_latents = jnp.concatenate([latents_gen, ref_latents_concat], axis=1)
      t_tensor = jnp.array([t_curr / 1000.0], dtype=latents_gen.dtype)

      out = self._jitted_transformer_step(
          t_params,
          joint_image_latents,
          joint_image_ids,
          prompt_embeds,
          text_ids,
          t_tensor,
          guidance=None,
      )

      noise_pred = out[:, :num_gen_tokens, :]
      dt = (t_prev - t_curr) / 1000.0
      latents_gen = latents_gen + dt * noise_pred

    # 6. Unpack & De-normalize
    latents_spatial = rearrange(latents_gen, "b (h w) c -> b c h w", h=h_latent // 2, w=w_latent // 2)
    latents_denorm = latents_spatial * bn_std + bn_mean
    unpatchified = unpatchify_latents(latents_denorm)

    # 7. VAE Decode
    max_logging.log("Decoding edited latents to RGB image...")
    decoded = self._jitted_vae_decode(v_params, unpatchified)

    if output_type == "pil":
      images_out = []
      for sample in np.array(decoded):
        img_hwc = np.transpose(sample, (1, 2, 0))
        img_uint8 = np.clip((img_hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        images_out.append(Image.fromarray(img_uint8))
      return images_out

    return decoded
