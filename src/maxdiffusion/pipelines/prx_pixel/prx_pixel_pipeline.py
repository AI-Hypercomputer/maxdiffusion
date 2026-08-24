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

"""End-to-End Inference Pipeline for PRXPixel in Pure Flax NNX (No VAE)."""

from functools import partial
import html
import time
from typing import Any, List, Optional, Tuple, Union
import ftfy
import numpy as np
from PIL import Image

from flax import nnx
import jax
import jax.numpy as jnp

from maxdiffusion.models.prx_pixel.transformer_prx_pixel import (
    FlaxPRXPixelConfig,
    NNXPRXPixelTransformer2DModel,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, NNXFlaxQwen3Model
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


def clean_prompt(prompt: str) -> str:
  """Exact prompt preprocessing for PRXPixel."""
  prompt = ftfy.fix_text(prompt)
  prompt = html.unescape(html.unescape(prompt))
  return prompt.strip()


class FlaxPRXPixelPipeline:
  """Full Text-to-Pixel Diffusion Pipeline for PRXPixel."""

  def __init__(
      self,
      transformer: NNXPRXPixelTransformer2DModel,
      text_encoder: NNXFlaxQwen3Model,
      tokenizer: Any,
      scheduler: Optional[FlaxFlowMatchScheduler] = None,
      dtype: Any = jnp.bfloat16,
  ):
    self.transformer = transformer
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.dtype = dtype

    if scheduler is None:
      self.scheduler = FlaxFlowMatchScheduler(
          num_train_timesteps=1000,
          shift=3.0,
      )
    else:
      self.scheduler = scheduler

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      do_classifier_free_guidance: bool = True,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Tokenizes and encodes text prompts using Qwen3-VL text encoder."""
    if isinstance(prompt, str):
      prompt = [prompt]
    batch_size = len(prompt)

    cleaned_prompts = [clean_prompt(p) for p in prompt]

    if do_classifier_free_guidance:
      if negative_prompt is None:
        negative_prompt = [""] * batch_size
      elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
      cleaned_neg = [clean_prompt(np) for np in negative_prompt]
      all_prompts = cleaned_neg + cleaned_prompts
    else:
      all_prompts = cleaned_prompts

    tok_out = self.tokenizer(
        all_prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = jnp.asarray(tok_out["input_ids"])
    attention_mask = jnp.asarray(tok_out["attention_mask"])

    out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(out, tuple):
      prompt_embeds = out[0]
    else:
      prompt_embeds = out

    return prompt_embeds.astype(self.dtype), attention_mask

  def __call__(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 1024,
      width: int = 1024,
      num_inference_steps: int = 28,
      guidance_scale: float = 4.5,
      generator: Optional[jax.Array] = None,
      latents: Optional[jnp.ndarray] = None,
      output_type: str = "pil",
  ) -> Union[List[Image.Image], jnp.ndarray]:
    """Generates images directly in pixel space."""
    if isinstance(prompt, str):
      prompt = [prompt]
    batch_size = len(prompt)
    do_cfg = guidance_scale > 1.0

    # 1. Encode prompt
    prompt_embeds, attention_mask = self.encode_prompt(
        prompt, negative_prompt=negative_prompt, do_classifier_free_guidance=do_cfg
    )

    # 2. Prepare initial noise latents (PRXPixel initial scale is 2.0 * epsilon)
    if latents is None:
      if generator is None:
        generator = jax.random.PRNGKey(42)
      raw_noise = 2.0 * jax.random.normal(generator, shape=(batch_size, 3, height, width), dtype=jnp.float32)
      latents = raw_noise
    else:
      latents = latents.astype(jnp.float32)

    # 3. Setup Flow Matching Scheduler Timesteps
    sigmas_np = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    shift = 3.0
    sigmas_np = shift * sigmas_np / (1.0 + (shift - 1.0) * sigmas_np)
    sigmas = jnp.asarray(np.concatenate([sigmas_np, [0.0]]).astype(np.float32))
    timesteps = sigmas[:-1] * 1000.0

    # 4. Denoising Loop
    for i, t in enumerate(timesteps):
      t_cont = (t.astype(jnp.float32) / 1000.0).reshape((1,))
      if do_cfg:
        latents_in = jnp.concatenate([latents, latents], axis=0).astype(self.dtype)
        t_cont_in = jnp.broadcast_to(t_cont, (2 * batch_size,))
      else:
        latents_in = latents.astype(self.dtype)
        t_cont_in = jnp.broadcast_to(t_cont, (batch_size,))

      # Predict clean image x0
      pred_x0 = self.transformer(
          hidden_states=latents_in,
          timestep=t_cont_in,
          encoder_hidden_states=prompt_embeds,
          attention_mask=attention_mask,
      )

      # CFG in x0-space
      if do_cfg:
        x0_uncond, x0_cond = jnp.split(pred_x0, 2, axis=0)
        x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
      else:
        x0_hat = pred_x0

      # Convert x0_hat to flow velocity: v_t = (x_t - x0_hat) / max(t/1000, 0.05)
      t_norm = jnp.maximum(t.astype(jnp.float32) / 1000.0, 0.05)
      v_t = (latents - x0_hat.astype(jnp.float32)) / t_norm

      # Flow Match Euler step: x_{t-dt} = x_t + dt * v_t
      sigma = sigmas[i]
      sigma_next = sigmas[i + 1]
      dt = sigma_next - sigma
      latents = latents + dt * v_t

    # 5. Direct Postprocessing (No VAE)
    if output_type in ["raw", "latents"]:
      return latents

    # Denormalize [-1, 1] to [0, 1]
    images_np = np.asarray(latents)
    images_np = np.clip(images_np / 2.0 + 0.5, 0.0, 1.0)
    images_np = np.transpose(images_np, (0, 2, 3, 1))

    if output_type == "np":
      return images_np

    # Quantize to [0, 255] uint8 RGB PIL Images
    images_uint8 = (images_np * 255.0).round().astype(np.uint8)
    pil_images = [Image.fromarray(images_uint8[i]) for i in range(batch_size)]
    return pil_images
