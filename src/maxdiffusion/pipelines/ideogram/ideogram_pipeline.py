"""
Copyright 2025 Google LLC

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

# pylint: disable=missing-class-docstring, missing-function-docstring, too-many-positional-arguments, import-outside-toplevel, redefined-outer-name
import json
import time
from functools import partial
from typing import Optional, Any, List

import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx

from ...models.ideogram.transformer_ideogram import Ideogram4Transformer, Ideogram4Config
from ...models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams
from ...models.ideogram.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    IMAGE_POSITION_OFFSET,
)
from ...models.ideogram.ideogram_utils import load_transformer_weights, load_vae_weights
from ...models.ideogram.qwen3_vl_text_encoder import Qwen3VLTextEncoder
from ...models.ideogram.latent_norm import get_latent_norm
from ...models.ideogram.scheduler import get_schedule_for_resolution, make_step_intervals
from maxdiffusion import max_logging

# Prompt lengths are rounded up to a multiple of this so that the positive
# branch's sequence length -- and therefore the compiled denoise step -- only
# changes when a prompt crosses a bucket boundary, instead of on every prompt.
TEXT_TOKEN_BUCKET = 256


@partial(jax.jit, static_argnames=("guidance_scale", "max_text_tokens"), donate_argnames=("z",))
def _denoise_step(
    cond_graphdef,
    cond_state,
    cond_rest,
    uncond_graphdef,
    uncond_state,
    uncond_rest,
    z,
    llm_pos,
    llm_neg,
    mt_curr,
    mt_next,
    pos_position_ids,
    pos_segment_ids,
    pos_indicator,
    neg_position_ids,
    neg_segment_ids,
    neg_indicator,
    guidance_scale: float,
    max_text_tokens: int,
):
  """One Euler step of the asymmetric-CFG denoise loop.

  This is deliberately a *single step* rather than the whole loop: the compiled
  artifact is then independent of ``num_steps``, so a short warmup run compiles
  exactly the executable a long run reuses. Driving it from a Python ``for``
  loop costs one dispatch per step and buys a warmup that is actually valid.
  """
  conditional_transformer = nnx.merge(cond_graphdef, cond_state, cond_rest)
  unconditional_transformer = nnx.merge(uncond_graphdef, uncond_state, uncond_rest)

  batch_size = z.shape[0]
  t = jnp.full((batch_size,), mt_curr, dtype=jnp.float32)

  # Positive branch (conditional, text + image).
  text_z_padding = jnp.zeros((batch_size, max_text_tokens, z.shape[-1]), dtype=z.dtype)
  pos_z = jnp.concatenate([text_z_padding, z], axis=1)
  pos_v = conditional_transformer(llm_pos, pos_z, t, pos_position_ids, pos_segment_ids, pos_indicator)[:, max_text_tokens:]

  # Negative branch (unconditional, image only, asymmetric CFG).
  neg_v = unconditional_transformer(llm_neg, z, t, neg_position_ids, neg_segment_ids, neg_indicator)

  # CFG: standard formula v = uncond + guidance * (cond - uncond)
  v = neg_v + guidance_scale * (pos_v - neg_v)

  # Euler step: z_next = z + v * delta_mt
  return z + v * (mt_next - mt_curr)


@partial(jax.jit, static_argnames=("grid_h", "grid_w", "patch"))
def _decode_latents(ae_graphdef, ae_state, ae_rest, z, shift, scale, grid_h, grid_w, patch):
  """Unpatch, denormalize and VAE-decode in one compiled executable.

  Previously each of these ops dispatched individually from Python.
  """
  autoencoder = nnx.merge(ae_graphdef, ae_state, ae_rest)
  batch_size = z.shape[0]
  ae_channels = z.shape[-1] // (patch * patch)

  z = z * scale + shift
  z = z.reshape((batch_size, grid_h, grid_w, patch, patch, ae_channels))
  z = jnp.transpose(z, (0, 5, 1, 3, 2, 4))
  z = z.reshape((batch_size, ae_channels, grid_h * patch, grid_w * patch))

  # Convert to NHWC for our Flax Autoencoder and cast to BF16
  z = jnp.transpose(z, (0, 2, 3, 1)).astype(jnp.bfloat16)

  images = autoencoder.decode(z)
  return jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)


def _build_mesh(config):
  """Device mesh derived from the maxdiffusion config.

  The transformer needs it at construction time: the splash kernel is a Mosaic
  custom call, which XLA refuses to auto-partition, so it has to be wrapped in a
  shard_map with a concrete mesh.
  """
  if not (hasattr(config, "mesh_axes") and hasattr(config, "ici_fsdp_parallelism")):
    return None
  from jax.sharding import Mesh
  from maxdiffusion import max_utils

  return Mesh(max_utils.create_device_mesh(config), config.mesh_axes)


def _make_transformer_config(config) -> Ideogram4Config:
  """Build the model config from the maxdiffusion config so that
  ``attention``, ``weights_dtype``, ``activations_dtype`` and
  ``flash_block_sizes`` in base_ideogram.yml actually reach the model."""
  flash_block_sizes = getattr(config, "flash_block_sizes", None) or {}
  kwargs = {}
  if "block_q" in flash_block_sizes:
    kwargs["flash_block_q"] = int(flash_block_sizes["block_q"])
  if "block_kv" in flash_block_sizes:
    kwargs["flash_block_kv"] = int(flash_block_sizes["block_kv"])
  return Ideogram4Config(
      attention=getattr(config, "attention", "dot_product"),
      weights_dtype=jnp.dtype(getattr(config, "weights_dtype", jnp.float32)),
      activations_dtype=jnp.dtype(getattr(config, "activations_dtype", jnp.float32)),
      precision=getattr(config, "precision", None),
      **kwargs,
  )


class IdeogramPipeline:

  def __init__(
      self,
      conditional_transformer: Ideogram4Transformer,
      unconditional_transformer: Ideogram4Transformer,
      autoencoder: AutoEncoder,
      text_encoder: Any,
      tokenizer: Any,
  ):
    self.conditional_transformer = conditional_transformer
    self.unconditional_transformer = unconditional_transformer
    self.autoencoder = autoencoder
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self._splits = {}

  @classmethod
  def from_pretrained(cls, config, vae_only=False, load_transformer=True):
    return cls._load_and_init(config, None, vae_only, load_transformer)

  @classmethod
  def from_checkpoint(cls, config, restored_checkpoint, vae_only=False, load_transformer=True):
    return cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint, vae_only=False, load_transformer=True):
    max_logging.log("Loading Ideogram pipeline components...")
    ae_config = AutoEncoderParams()
    rngs = nnx.Rngs(0)

    autoencoder = nnx.eval_shape(lambda rngs: AutoEncoder(rngs, ae_config), rngs)
    ae_state = nnx.state(autoencoder).to_pure_dict()

    ae_params = load_vae_weights(config.pretrained_model_name_or_path, ae_state, "cpu")
    autoencoder = AutoEncoder(rngs, ae_config)
    nnx.update(autoencoder, ae_params)

    if vae_only:
      return cls(None, None, autoencoder, None, None)

    conditional_transformer = None
    unconditional_transformer = None
    if load_transformer:
      transformer_config = _make_transformer_config(config)
      mesh = _build_mesh(config)
      max_logging.log(
          f"Ideogram transformer: attention={transformer_config.attention}, "
          f"weights_dtype={transformer_config.weights_dtype}, "
          f"activations_dtype={transformer_config.activations_dtype}"
      )

      # Both branches have identical structure, so one set of eval shapes drives
      # both loads. The shapes carry the configured weights_dtype, which
      # load_transformer_weights uses to cast the incoming fp8/fp32 tensors.
      abstract_transformer = nnx.eval_shape(lambda rngs: Ideogram4Transformer(rngs, transformer_config, mesh=mesh), rngs)
      transformer_state = nnx.state(abstract_transformer).to_pure_dict()

      def _load_branch(subfolder, checkpoint_key):
        if restored_checkpoint:
          # This used to read "unconditional_ideogram_state" from a checkpoint
          # that only ever saved "ideogram_state", i.e. a guaranteed KeyError on
          # the from_checkpoint path. Fail with something actionable instead.
          available = list(restored_checkpoint.keys())
          if checkpoint_key not in available:
            raise KeyError(
                f"checkpoint is missing '{checkpoint_key}'; found {available}. "
                "IdeogramCheckpointer.save_checkpoint must save both transformer branches "
                "(pass unconditional_states=)."
            )
          return restored_checkpoint[checkpoint_key]
        return load_transformer_weights(
            config.pretrained_model_name_or_path,
            transformer_state,
            "cpu",
            num_layers=transformer_config.num_layers,
            scan_layers=False,
            subfolder=subfolder,
        )

      conditional_transformer = Ideogram4Transformer(rngs, transformer_config, mesh=mesh)
      nnx.update(conditional_transformer, _load_branch("transformer", "ideogram_state"))

      unconditional_transformer = Ideogram4Transformer(rngs, transformer_config, mesh=mesh)
      nnx.update(unconditional_transformer, _load_branch("unconditional_transformer", "unconditional_ideogram_state"))

    max_logging.log("Initializing Qwen3-VL text encoder...")
    text_encoder = Qwen3VLTextEncoder.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", extra_special_tokens={}
    )

    return cls(conditional_transformer, unconditional_transformer, autoencoder, text_encoder, tokenizer)

  def _split(self, name, module):
    """Cache nnx.split so the per-step jit call does not re-flatten the whole
    parameter tree on every one of the 50 iterations."""
    cached = self._splits.get(name)
    if cached is None or cached[0] is not module:
      cached = (module,) + tuple(nnx.split(module, nnx.Param, ...))
      self._splits[name] = cached
    return cached[1:]

  def _reorder_caption_keys(self, parsed: dict) -> dict:
    canonical_keys = ["high_level_description", "style_description", "compositional_deconstruction"]
    reordered = {}
    for key in canonical_keys:
      if key in parsed:
        if key == "style_description" and isinstance(parsed[key], dict):
          sd = parsed[key]
          if "art_style" in sd and "photo" not in sd:
            sd_keys = ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
          else:
            sd_keys = ["aesthetics", "lighting", "photo", "medium", "color_palette"]
          reordered_sd = {}
          for sk in sd_keys:
            if sk in sd:
              reordered_sd[sk] = sd[sk]
          for sk in sd:
            if sk not in reordered_sd:
              reordered_sd[sk] = sd[sk]
          reordered[key] = reordered_sd
        else:
          reordered[key] = parsed[key]
    for key in parsed:
      if key not in reordered:
        reordered[key] = parsed[key]
    return reordered

  def _build_inputs_cpu(self, prompts, height, width):
    batch_size = len(prompts)

    grid_h, grid_w = None, None
    num_image_tokens = None

    tokenized = []
    for prompt in prompts:
      try:
        parsed = json.loads(prompt)
        parsed = self._reorder_caption_keys(parsed)
        prompt = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
      except json.JSONDecodeError:
        pass

      if prompt != "":
        encoded = self.tokenizer(prompt, return_tensors="np", add_special_tokens=True)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
      else:
        num_text_tokens = 256
        token_ids = np.zeros((num_text_tokens,), dtype=np.int32)
      tokenized.append((token_ids, num_text_tokens))

    # Round up to a bucket so that two prompts of similar length share one
    # compiled executable instead of forcing a recompile per prompt.
    longest = max(num_text for _, num_text in tokenized)
    max_text_tokens = -(-longest // TEXT_TOKEN_BUCKET) * TEXT_TOKEN_BUCKET

    patch_size = 2
    ae_scale_factor = 8
    patch = patch_size * ae_scale_factor
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = np.broadcast_to(np.arange(grid_h).reshape(-1, 1), (grid_h, grid_w)).reshape(-1)
    w_idx = np.broadcast_to(np.arange(grid_w).reshape(1, -1), (grid_h, grid_w)).reshape(-1)
    t_idx = np.zeros_like(h_idx)
    image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

    token_ids_out = np.zeros((batch_size, total_seq_len), dtype=np.int32)
    text_position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)
    position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)

    segment_ids = np.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=np.int32)
    indicator = np.zeros((batch_size, total_seq_len), dtype=np.int32)

    for b in range(batch_size):
      toks, num_text = tokenized[b]
      pad_len = max_text_tokens - num_text
      total_unpadded = num_text + num_image_tokens
      offset = pad_len

      token_ids_out[b, offset : offset + num_text] = toks

      text_pos = np.arange(num_text)
      text_pos_3d = np.stack([text_pos, text_pos, text_pos], axis=1)
      text_position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text :] = image_pos

      indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

      segment_ids[b, offset : offset + total_unpadded] = 1

    return {
        "token_ids": token_ids_out,
        "text_position_ids": text_position_ids,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
        "num_image_tokens": num_image_tokens,
        "max_text_tokens": max_text_tokens,
        "longest_text_tokens": longest,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }

  def generate(
      self,
      prompts: List[str],
      negative_prompts: Optional[List[str]] = None,
      height: int = 1024,
      width: int = 1024,
      num_steps: int = 50,
      guidance_scale: float = 7.0,
      seed: int = 42,
  ):
    """Returns `(images, trace)`, where `trace` is a per-phase wall-clock
    breakdown in seconds (see wan_pipeline_2_2.py for the same convention).

    Each phase blocks on its own output. JAX dispatch is async, so without the
    blocks every phase but the last would measure enqueue time and the whole run
    would pile up on whichever phase happens to sync first.
    """
    trace = {}
    t_cond_start = time.perf_counter()

    if negative_prompts is None:
      negative_prompts = [""] * len(prompts)

    all_prompts = prompts + negative_prompts
    t_prep_start = time.perf_counter()
    inputs = self._build_inputs_cpu(all_prompts, height, width)
    trace["input_prep"] = time.perf_counter() - t_prep_start

    batch_size = len(prompts)
    max_text_tokens = inputs["max_text_tokens"]
    num_image_tokens = inputs["num_image_tokens"]

    # 1. Text Encoding (runs once, outside the denoise loop).
    # The transformer's text region is bucketed, but the encoder is an ~8B model
    # on CPU, so feed it only the real tokens and left-pad the features back out
    # to the bucket. Text is left-padded, so the real tokens for every batch
    # entry live in the last `longest` columns of the text region.
    longest = inputs["longest_text_tokens"]
    enc_start = max_text_tokens - longest
    llm_attention_mask = (inputs["indicator"][:, :max_text_tokens] == LLM_TOKEN_INDICATOR).astype(jnp.int32)
    t_encode_start = time.perf_counter()
    llm_features = self.text_encoder(
        inputs["token_ids"][:, enc_start:max_text_tokens],
        llm_attention_mask[:, enc_start:],
        inputs["text_position_ids"][:, enc_start:max_text_tokens, 0],  # Extract the 1D positional index for Qwen
    )
    if enc_start:
      llm_features = jnp.pad(llm_features, ((0, 0), (enc_start, 0), (0, 0)))
    # Zero out non-LLM positions (left padding)
    llm_features = llm_features * jnp.expand_dims(llm_attention_mask.astype(jnp.float32), -1)
    jax.block_until_ready(llm_features)
    trace["text_encode"] = time.perf_counter() - t_encode_start

    # Only the text prefix carries LLM features; the transformer pads the
    # projection back out to the full sequence itself.
    pos_llm_features = llm_features[:batch_size]
    # The unconditional branch has no LLM tokens, so its conditioning is
    # identically zero -- pass None rather than a zero tensor to skip the
    # 53248 -> 4608 projection entirely.
    neg_llm_features = None

    # Initialize z. The latent width is the transformer's input width
    # (ae_channels * patch_size**2); it used to be hardcoded to 128, which
    # silently mismatched any model configured with a different in_channels.
    key = jax.random.PRNGKey(seed)
    latent_dim = self.conditional_transformer.config.in_channels
    z = jax.random.normal(key, (batch_size, num_image_tokens, latent_dim), dtype=jnp.float32)

    # Precompute the scheduler timesteps. schedule_fn is vectorized, so this is
    # one dispatch rather than num_steps+1 dispatches each with a host sync.
    schedule_fn = get_schedule_for_resolution((height, width), known_mean=0.5)
    step_intervals = make_step_intervals(num_steps)
    sigmas = np.asarray(jax.jit(schedule_fn)(step_intervals), dtype=np.float32)

    # Negative branch inputs for asymmetric CFG (image tokens only).
    neg_position_ids = jnp.asarray(inputs["position_ids"][:batch_size, max_text_tokens:])
    neg_segment_ids = jnp.asarray(inputs["segment_ids"][:batch_size, max_text_tokens:])
    neg_indicator = jnp.asarray(inputs["indicator"][:batch_size, max_text_tokens:])

    # Positive branch inputs (text + image tokens).
    pos_position_ids = jnp.asarray(inputs["position_ids"][:batch_size])
    pos_segment_ids = jnp.asarray(inputs["segment_ids"][:batch_size])
    pos_indicator = jnp.asarray(inputs["indicator"][:batch_size])

    cond_graphdef, cond_state, cond_rest = self._split("cond", self.conditional_transformer)
    uncond_graphdef, uncond_state, uncond_rest = self._split("uncond", self.unconditional_transformer)

    jax.block_until_ready(z)
    trace["conditioning"] = time.perf_counter() - t_cond_start

    # 2. Denoising loop. Python for loop over a jitted single step: the compiled
    # step does not depend on num_steps, so warmup at any step count is valid.
    t_denoise_start = time.perf_counter()
    for i_fori in range(num_steps):
      i = (num_steps - 1) - i_fori
      z = _denoise_step(
          cond_graphdef,
          cond_state,
          cond_rest,
          uncond_graphdef,
          uncond_state,
          uncond_rest,
          z,
          pos_llm_features,
          neg_llm_features,
          sigmas[i + 1],
          sigmas[i],
          pos_position_ids,
          pos_segment_ids,
          pos_indicator,
          neg_position_ids,
          neg_segment_ids,
          neg_indicator,
          guidance_scale=guidance_scale,
          max_text_tokens=max_text_tokens,
      )
      # The first step carries any compile; splitting it out keeps the reported
      # per-step figure a steady-state number rather than an average polluted by
      # a one-off compile.
      if i_fori == 0:
        jax.block_until_ready(z)
        trace["denoise_first_step"] = time.perf_counter() - t_denoise_start

    jax.block_until_ready(z)
    trace["denoise_total"] = time.perf_counter() - t_denoise_start
    if num_steps > 1:
      trace["denoise_per_step"] = (trace["denoise_total"] - trace["denoise_first_step"]) / (num_steps - 1)
    else:
      trace["denoise_per_step"] = trace["denoise_total"]

    # 3. Decode
    t_decode_start = time.perf_counter()
    ae_graphdef, ae_state, ae_rest = self._split("ae", self.autoencoder)
    shift, scale = get_latent_norm()
    images = _decode_latents(
        ae_graphdef,
        ae_state,
        ae_rest,
        z,
        shift,
        scale,
        grid_h=inputs["grid_h"],
        grid_w=inputs["grid_w"],
        patch=2,
    )
    jax.block_until_ready(images)
    trace["vae_decode"] = time.perf_counter() - t_decode_start

    return images, trace
