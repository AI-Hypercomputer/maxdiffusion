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


import re
import urllib.parse as ul

BAD_PUNCT_REGEX = re.compile(
    r"["
    + "#®•©™&@·º½¾¿¡§~"
    + r"\)"
    + r"\("
    + r"\]"
    + r"\["
    + r"\}"
    + r"\{"
    + r"\|"
    + r"\\"
    + r"\/"
    + r"\*"
    + r"]{1,}"
)
REGEX2 = re.compile(r"(?:\-|\_)")


def clean_prompt(text: str) -> str:
  """Clean text using exact PRX / DeepFloyd text processing logic."""
  text = str(text)
  text = ul.unquote_plus(text)
  text = text.strip().lower()
  text = re.sub("<person>", "person", text)

  # Remove urls
  text = re.sub(
      r"\b((?:https?|www):(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@))",
      "",
      text,
  )

  text = re.sub(r"@[\w\d]+\b", "", text)
  text = re.sub(r"[\u31c0-\u31ef]+", "", text)
  text = re.sub(r"[\u31f0-\u31ff]+", "", text)
  text = re.sub(r"[\u3200-\u32ff]+", "", text)
  text = re.sub(r"[\u3300-\u33ff]+", "", text)
  text = re.sub(r"[\u3400-\u4dbf]+", "", text)
  text = re.sub(r"[\u4dc0-\u4dff]+", "", text)
  text = re.sub(r"[\u4e00-\u9fff]+", "", text)

  text = re.sub(
      r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
      "-",
      text,
  )

  text = re.sub(r"[`´«»" "¨]", '"', text)
  text = re.sub(r"['']", "'", text)
  text = re.sub(r"&quot;?", "", text)
  text = re.sub(r"&amp", "", text)
  text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)
  text = re.sub(r"\d:\d\d\s+$", "", text)
  text = re.sub(r"\\n", " ", text)
  text = re.sub(r"#\d{1,3}\b", "", text)
  text = re.sub(r"#\d{5,}\b", "", text)
  text = re.sub(r"\b\d{6,}\b", "", text)
  text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)

  text = re.sub(r"[\"\']{2,}", r'"', text)
  text = re.sub(r"[\.]{2,}", r" ", text)
  text = re.sub(BAD_PUNCT_REGEX, r" ", text)
  text = re.sub(r"\s+\.\s+", r" ", text)

  if len(re.findall(REGEX2, text)) > 3:
    text = re.sub(REGEX2, " ", text)

  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  text = text.strip()

  text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)
  text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)
  text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)
  text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
  text = re.sub(r"(free\s)?download(\sfree)?", "", text)
  text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
  text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
  text = re.sub(r"\bpage\s+\d+\b", "", text)
  text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)
  text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

  text = re.sub(r"\b\s+\:\s+", r": ", text)
  text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
  text = re.sub(r"\s+", " ", text)

  text = text.strip()
  text = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", text)
  text = re.sub(r"^[\'\_,\-\:;]", r"", text)
  text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
  text = re.sub(r"^\.\S+$", "", text)
  return text.strip()


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

    # 2. Prepare initial noise latents
    if latents is None:
      if generator is None:
        generator = jax.random.PRNGKey(42)
      raw_noise = jax.random.normal(generator, shape=(batch_size, 3, height, width), dtype=jnp.float32)
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

      # Predict x0
      pred = self.transformer(
          hidden_states=latents_in,
          timestep=t_cont_in,
          encoder_hidden_states=prompt_embeds,
          attention_mask=attention_mask,
      )

      if do_cfg:
        pred_uncond, pred_cond = jnp.split(pred, 2, axis=0)
        model_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
      else:
        model_output = pred

      # Flow Match Euler update
      sigma = sigmas[i]
      sigma_next = sigmas[i + 1]
      dt = sigma_next - sigma
      latents = latents + dt * model_output.astype(jnp.float32)

    # 5. Direct Postprocessing (No VAE)
    if output_type == "raw" or output_type == "latents":
      return latents

    # Denormalize [-1, 1] to [0, 255] RGB PIL Image
    images_np = np.asarray(latents)
    images_np = (images_np.clip(-1.0, 1.0) + 1.0) * 127.5
    images_np = np.transpose(images_np.astype(np.uint8), (0, 2, 3, 1))

    pil_images = [Image.fromarray(images_np[i]) for i in range(batch_size)]
    return pil_images
