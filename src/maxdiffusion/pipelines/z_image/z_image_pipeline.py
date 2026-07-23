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

"""Inference pipeline for the official Z-Image and Z-Image-Turbo checkpoints."""

from contextlib import nullcontext
from functools import partial
import time
from typing import Optional, Union

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


@jax.jit
def encode_step(graphdef, state, rest, input_ids, attention_mask):
  """Qwen3 forward, returning the layer Z-Image conditions on."""
  text_encoder = nnx.merge(graphdef, state, rest)
  _, all_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
  # `all_hidden_states` is [embeddings, layer_1, ..., layer_n], so -2 is the
  # penultimate layer's raw output -- the same tensor Transformers hands back
  # as `hidden_states[-2]`.
  return all_hidden_states[-2]


@partial(jax.jit, static_argnames=("model_dtype",))
def denoise_step(graphdef, state, rest, latents, timestep, prompt_embeds, sigma_delta, model_dtype):
  """One flow-matching Euler step, compiled once and reused every step.

  Latents carry float32 so the Euler update accumulates in full precision;
  only the denoiser's input is cast down.
  """
  transformer = nnx.merge(graphdef, state, rest)
  # Z-Image takes one array per batch element so prompts of different lengths
  # can share a step; the frame axis it expects is the `None` below.
  samples = [sample[:, None] for sample in latents.astype(model_dtype)]
  predictions = transformer(samples, timestep, prompt_embeds, return_dict=False)[0]
  prediction = jnp.stack([item[:, 0] for item in predictions])
  return latents + sigma_delta * (-prediction.astype(jnp.float32))


class ZImagePipeline:
  """A small, composable JAX denoising pipeline.

  Every component -- the Qwen3 text encoder, the VAE and the 6B denoiser --
  runs in JAX/MaxDiffusion. Loading lives in
  `maxdiffusion.checkpointing.z_image_checkpointer.ZImageCheckpointer`.
  """

  def __init__(
      self,
      transformer,
      vae,
      vae_params,
      tokenizer,
      text_encoder,
      dtype=jnp.bfloat16,
      mesh=None,
      logical_axis_rules=(),
      vae_shift_factor: Optional[float] = None,
      offload_encoders: bool = False,
  ):
    self.transformer = transformer
    self.vae = vae
    self.vae_params = vae_params
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.dtype = dtype
    self.mesh = mesh
    # The attention op constrains its activations by logical axis name, so the
    # rules must be in scope while the denoise step is traced.
    self.logical_axis_rules = logical_axis_rules
    self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # Z-Image's Flux-format VAE config carries this field, while the older
    # Flax AutoencoderKL config loader drops unknown attributes, so the
    # checkpointer reads it from the raw config and passes it in.
    self.vae_shift_factor = vae_shift_factor if vae_shift_factor is not None else 0.1159
    self._jitted_decoder = None
    # Pre-split models at initialization to avoid any runtime CPU/metadata overhead
    self._text_encoder_split = nnx.split(self.text_encoder, nnx.Param, ...)
    self._transformer_split = nnx.split(self.transformer, nnx.Param, ...)

    self.offload_encoders = offload_encoders
    if self.offload_encoders:
      # Save the original shardings of the text encoder parameters
      graphdef, state, rest = self._text_encoder_split
      self._text_encoder_shardings = jax.tree_util.tree_map(lambda x: x.sharding, state)

  def _decode_fn(self):
    """Jitted VAE decode, including the latent denormalization."""
    if self._jitted_decoder is None:

      def decode(params, latents):
        return self.vae.apply(
            {"params": params},
            latents / self.vae.config.scaling_factor + self.vae_shift_factor,
            deterministic=True,
            method=self.vae.decode,
        ).sample

      self._jitted_decoder = jax.jit(decode)
    return self._jitted_decoder

  def encode_prompt(self, prompt: Union[str, list[str]], max_sequence_length: int = 512) -> list[jax.Array]:
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    chats = [
        self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for item in prompts
    ]
    # Padding to a fixed length keeps one compiled encoder for every prompt.
    # Qwen3 is causal and the pad mask is applied, so right padding cannot
    # change the real tokens' states -- they are sliced back out below, and
    # Z-Image consumes only those unmasked states.
    inputs = self.tokenizer(
        chats, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="np"
    )
    graphdef, state, rest = self._text_encoder_split

    if self.offload_encoders:
      # Restore parameters from CPU back to original device shardings
      state = jax.tree_util.tree_map(lambda x, sharding: jax.device_put(x, sharding), state, self._text_encoder_shardings)

    with self.mesh if self.mesh is not None else nullcontext():
      embeddings = encode_step(
          graphdef,
          state,
          rest,
          jnp.asarray(inputs["input_ids"], jnp.int32),
          jnp.asarray(inputs["attention_mask"], jnp.int32),
      )
    lengths = np.asarray(inputs["attention_mask"]).sum(axis=-1)

    if self.offload_encoders:
      from maxdiffusion import max_logging

      s_offload = time.perf_counter()
      cpus = jax.devices("cpu")
      offloaded_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=cpus[0]), state)
      self._text_encoder_split = (graphdef, offloaded_state, rest)
      max_logging.log(f"Offloaded Qwen3 text encoder parameters to CPU in {(time.perf_counter() - s_offload):.4f}s.")

    return [jnp.asarray(embeddings[index, : lengths[index]], dtype=self.dtype) for index in range(len(lengths))]

  @staticmethod
  def _sigmas(num_inference_steps: int, shift: float) -> jax.Array:
    sigmas = jnp.linspace(1.0, 0.0, num_inference_steps + 1, dtype=jnp.float32)
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

  def __call__(
      self,
      prompt: Union[str, list[str]],
      height: int = 1024,
      width: int = 1024,
      num_inference_steps: int = 9,
      guidance_scale: float = 0.0,
      seed: int = 0,
      max_sequence_length: int = 512,
      vae_decode_chunk: int = 1,
      output_type: str = "pil",
      return_timings: bool = False,
  ):
    del guidance_scale  # The published Turbo configuration uses guidance_scale=0.
    vae_scale = self.vae_scale_factor * 2
    if height % vae_scale or width % vae_scale:
      raise ValueError(f"height and width must be divisible by {vae_scale}.")
    trace = {}
    stage_start = time.perf_counter()

    def stage(name, value):
      """Close out a timed stage, if the caller asked for timings.

      Dispatch is async, so a stage's time is only its own if we block at the
      boundary. That barrier costs a host-device round trip and stops the host
      running ahead to enqueue the next stage, so it is skipped entirely when
      no trace was requested.
      """
      nonlocal stage_start
      if return_timings:
        jax.block_until_ready(value)
        trace[name] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
      return value

    prompt_embeds = stage("text_encode", self.encode_prompt(prompt, max_sequence_length))
    batch_size = len(prompt_embeds)
    latent_height, latent_width = 2 * (height // vae_scale), 2 * (width // vae_scale)
    latents = jax.random.normal(
        jax.random.key(seed), (batch_size, self.transformer.in_channels, latent_height, latent_width), self.dtype
    ).astype(jnp.float32)
    # The released scheduler is FlowMatchEulerDiscreteScheduler(shift=3.0).
    sigmas = self._sigmas(num_inference_steps, shift=3.0)
    # Split once, outside the loop: every step reuses the same compiled
    # executable, so the step count only changes the Python trip count.
    graphdef, state, rest = self._transformer_split
    with self.mesh if self.mesh is not None else nullcontext(), nn_partitioning.axis_rules(self.logical_axis_rules):
      for index in range(num_inference_steps):
        # Timestep and step size stay traced arguments; making them static
        # would compile a separate executable for every step.
        latents = denoise_step(
            graphdef,
            state,
            rest,
            latents,
            jnp.full((batch_size,), 1.0 - sigmas[index], dtype=self.dtype),
            prompt_embeds,
            (sigmas[index + 1] - sigmas[index]).astype(jnp.float32),
            self.dtype,
        )
      latents = stage("denoise", latents)

      # The decoder is replicated and its activations are full-resolution, so
      # a whole batch at once needs many GB. Decode in chunks; raise
      # vae_decode_chunk to trade memory for speed.
      chunk = max(1, int(vae_decode_chunk))
      decoded = jnp.concatenate(
          [
              self._decode_fn()(self.vae_params, latents[start : start + chunk].astype(self.dtype))
              for start in range(0, batch_size, chunk)
          ]
      )
      decoded = stage("vae_decode", decoded)
    # The conversion to numpy blocks on its own, so this stage is always real.
    frames = np.asarray((decoded.transpose(0, 2, 3, 1) / 2 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
    images = [Image.fromarray(frame) for frame in frames] if output_type == "pil" else list(frames)
    stage("host_post", None)
    return (images, trace) if return_timings else images
