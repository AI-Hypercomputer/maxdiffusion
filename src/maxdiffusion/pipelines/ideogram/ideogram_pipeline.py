# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring, too-many-positional-arguments, import-outside-toplevel, redefined-outer-name
from typing import Optional, Any, List

import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx

from ...models.ideogram.transformer_ideogram import Ideogram4Transformer, Ideogram4Config
from ...models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams
from ...models.ideogram.constants import LLM_TOKEN_INDICATOR
from ...models.ideogram.ideogram_utils import load_transformer_weights, load_vae_weights
from ...models.ideogram.torchax_text_encoder import TorchaxQwen3VLTextEncoder
from ...models.ideogram.latent_norm import get_latent_norm
from ...models.ideogram.scheduler import get_schedule_for_resolution, make_step_intervals
from maxdiffusion import max_logging


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
      transformer_config = Ideogram4Config()

      # Load Conditional Transformer
      conditional_transformer = nnx.eval_shape(lambda rngs: Ideogram4Transformer(rngs, transformer_config), rngs)
      transformer_state = nnx.state(conditional_transformer).to_pure_dict()

      if restored_checkpoint:
        cond_params = restored_checkpoint["ideogram_state"]
      else:
        cond_params = load_transformer_weights(
            config.pretrained_model_name_or_path,
            transformer_state,
            "cpu",
            num_layers=34,
            scan_layers=False,
            subfolder="transformer",
        )

      conditional_transformer = Ideogram4Transformer(rngs, transformer_config)
      nnx.update(conditional_transformer, cond_params)

      # Load Unconditional Transformer
      unconditional_transformer = nnx.eval_shape(lambda rngs: Ideogram4Transformer(rngs, transformer_config), rngs)
      if restored_checkpoint:
        uncond_params = restored_checkpoint["unconditional_ideogram_state"]
      else:
        uncond_params = load_transformer_weights(
            config.pretrained_model_name_or_path,
            transformer_state,
            "cpu",
            num_layers=34,
            scan_layers=False,
            subfolder="unconditional_transformer",
        )

      unconditional_transformer = Ideogram4Transformer(rngs, transformer_config)
      nnx.update(unconditional_transformer, uncond_params)

    # Skip text encoder for pure test for now unless requested
    max_logging.log("Initializing Torchax Text Encoder...")
    text_encoder_repo = config.pretrained_model_name_or_path
    subfolder = "text_encoder"
    text_encoder = TorchaxQwen3VLTextEncoder.from_pretrained(text_encoder_repo, subfolder=subfolder)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", extra_special_tokens={}
    )

    return cls(conditional_transformer, unconditional_transformer, autoencoder, text_encoder, tokenizer)

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

    max_text_tokens = 0
    grid_h, grid_w = None, None
    num_image_tokens = None

    tokenized = []
    for prompt in prompts:
      import json

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

    max_text_tokens = max(num_text for _, num_text in tokenized)

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
    IMAGE_POSITION_OFFSET = 65536
    image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

    token_ids_out = np.zeros((batch_size, total_seq_len), dtype=np.int32)
    text_position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)
    position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)

    from ...models.ideogram.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, SEQUENCE_PADDING_INDICATOR

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
    if negative_prompts is None:
      negative_prompts = [""] * len(prompts)

    all_prompts = prompts + negative_prompts
    inputs = self._build_inputs_cpu(all_prompts, height, width)

    batch_size = len(prompts)
    max_text_tokens = inputs["max_text_tokens"]
    num_image_tokens = inputs["num_image_tokens"]

    # 1. Text Encoding (using TorchAX text encoder)
    llm_attention_mask = (inputs["indicator"][:, :max_text_tokens] == LLM_TOKEN_INDICATOR).astype(jnp.int32)
    llm_features = self.text_encoder(
        inputs["token_ids"][:, :max_text_tokens],
        llm_attention_mask,
        inputs["text_position_ids"][:, :max_text_tokens, 0],  # Extract the 1D positional index for Qwen
    )
    # Zero out non-LLM positions (left padding)
    llm_features = llm_features * jnp.expand_dims(llm_attention_mask.astype(jnp.float32), -1)

    # Build padded LLM features: text features for the positive branch, zeros for image positions.
    image_llm_padding = jnp.zeros((batch_size, num_image_tokens, llm_features.shape[-1]), dtype=jnp.float32)
    # llm_features covers only the text portion; pad with zeros for the image token positions.
    pos_llm_features = jnp.concatenate([llm_features[:batch_size], image_llm_padding], axis=1)

    # Initialize z
    key = jax.random.PRNGKey(seed)
    latent_dim = 128  # 32 * patch_size * patch_size
    z = jax.random.normal(key, (batch_size, num_image_tokens, latent_dim), dtype=jnp.float32)

    # Precompute the scheduler timesteps array using the Ideogram 4 scheduler
    schedule_fn = get_schedule_for_resolution((height, width), known_mean=0.5)
    step_intervals = make_step_intervals(num_steps)
    # Compute schedule evaluated at all intervals. schedule_fn returns a continuous scalar mapping.
    sigmas = jnp.array(
        [float(schedule_fn(jnp.array([step_intervals[i]]))[0]) for i in range(num_steps + 1)], dtype=jnp.float32
    )

    # Padding for text latents
    text_z_padding = jnp.zeros((batch_size, max_text_tokens, latent_dim), dtype=jnp.float32)

    # 2. Denoising loop in JAX
    # Define the loop step
    def denoise_step(i_fori, val):
      z_curr, llm_pos, llm_neg = val

      # Euler step in model-time (mt) convention where mt increases from 0 (noisy) to 1 (clean).
      i = (num_steps - 1) - i_fori
      mt_curr = sigmas[i + 1]
      mt_next = sigmas[i]

      t = jnp.full((batch_size,), mt_curr, dtype=jnp.float32)

      # Positive branch (conditional, text + image)
      pos_z = jnp.concatenate([text_z_padding, z_curr], axis=1)
      pos_v = self.conditional_transformer(llm_pos, pos_z, t, pos_position_ids, pos_segment_ids, pos_indicator)[
          :, max_text_tokens:
      ]

      # Negative branch (unconditional, image only, asymmetric CFG)
      neg_z = z_curr
      neg_v = self.unconditional_transformer(llm_neg, neg_z, t, neg_position_ids, neg_segment_ids, neg_indicator)

      # CFG: standard formula v = uncond + guidance * (cond - uncond)
      v = neg_v + guidance_scale * (pos_v - neg_v)

      # Euler step: z_next = z + v * delta_mt
      delta_mt = mt_next - mt_curr
      z_next = z_curr + v * delta_mt

      return z_next, llm_pos, llm_neg

    # Setup negative branch inputs for asymmetric CFG (image tokens only)
    neg_llm_features = jnp.zeros((batch_size, num_image_tokens, llm_features.shape[-1]), dtype=jnp.float32)
    neg_position_ids = inputs["position_ids"][:batch_size, max_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:batch_size, max_text_tokens:]
    neg_indicator = inputs["indicator"][:batch_size, max_text_tokens:]

    # Setup positive branch inputs (text + image tokens)
    pos_position_ids = inputs["position_ids"][:batch_size]
    pos_segment_ids = inputs["segment_ids"][:batch_size]
    pos_indicator = inputs["indicator"][:batch_size]

    # Loop
    init_val = (z, pos_llm_features, neg_llm_features)
    z, _, _ = jax.lax.fori_loop(0, num_steps, denoise_step, init_val)

    # 3. Decode
    # Unpatching logic
    patch = 2
    ae_channels = z.shape[-1] // (patch * patch)

    # Apply latent scale and shift
    shift, scale = get_latent_norm()
    z = z * scale + shift

    z = z.reshape((batch_size, inputs["grid_h"], inputs["grid_w"], patch, patch, ae_channels))
    z = jnp.transpose(z, (0, 5, 1, 3, 2, 4))
    z = z.reshape((batch_size, ae_channels, inputs["grid_h"] * patch, inputs["grid_w"] * patch))

    # Convert to NHWC for our Flax Autoencoder and cast to BF16
    z = jnp.transpose(z, (0, 2, 3, 1)).astype(jnp.bfloat16)

    images = self.autoencoder.decode(z)
    images = jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)

    return images
