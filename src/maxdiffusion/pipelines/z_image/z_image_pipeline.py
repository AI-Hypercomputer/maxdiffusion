"""Inference pipeline for the official Z-Image and Z-Image-Turbo checkpoints."""

from contextlib import nullcontext
from typing import Optional

from flax import nnx
import flax.linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maxdiffusion import FlaxAutoencoderKL
from maxdiffusion.models.z_image.transformer_z_image import ZImageTransformer2DModel
from maxdiffusion.models.z_image.z_image_utils import load_z_image_transformer


def create_z_image_transformer(
    model_id: str,
    rngs: nnx.Rngs,
    attention_kernel: str = "dot_product",
    mesh: Optional[jax.sharding.Mesh] = None,
    flash_block_sizes=None,
    dtype: jnp.dtype = jnp.bfloat16,
    weights_dtype: jnp.dtype = jnp.bfloat16,
    logical_axis_rules=None,
) -> ZImageTransformer2DModel:
  """Instantiate and stream a Diffusers Z-Image checkpoint into an NNX model."""
  config = ZImageTransformer2DModel.load_config(model_id, subfolder="transformer")

  def factory(init_rngs):
    return ZImageTransformer2DModel(
        rngs=init_rngs,
        attention_kernel=attention_kernel,
        mesh=mesh,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        weights_dtype=weights_dtype,
        **config,
    )

  model_shape = nnx.eval_shape(factory, rngs)
  graphdef, state, rest = nnx.split(model_shape, nnx.Param, ...)
  target_shardings = None
  if mesh is not None and logical_axis_rules is not None:
    logical_specs = nnx.get_partition_spec(state)
    sharding_tree = nn.logical_to_mesh_sharding(logical_specs, mesh, logical_axis_rules)
    target_shardings = {path: variable.value for path, variable in nnx.to_flat_state(sharding_tree)}
  loaded_params = load_z_image_transformer(model_id, state.to_pure_dict(), target_shardings=target_shardings)
  flat_state = dict(nnx.to_flat_state(state))
  for path, value in flatten_dict(loaded_params).items():
    flat_state[path].value = value
  return nnx.merge(graphdef, nnx.from_flat_state(flat_state), rest)


class ZImagePipeline:
  """A small, composable JAX denoising pipeline.

  Text encoding stays in PyTorch because Transformers has no Flax Qwen3 model;
  the VAE and the 12B denoiser execute in JAX/MaxDiffusion.
  """

  def __init__(self, transformer, vae, vae_params, tokenizer, text_encoder, dtype=jnp.bfloat16, mesh=None):
    self.transformer = transformer
    self.vae = vae
    self.vae_params = vae_params
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.dtype = dtype
    self.mesh = mesh
    self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # Z-Image's Flux-format VAE config carries this field, while the older
    # Flax AutoencoderKL config loader drops unknown attributes.
    self.vae_shift_factor = getattr(vae.config, "shift_factor", 0.1159)

  @classmethod
  def from_pretrained(
      cls,
      model_id: str,
      rng: jax.Array,
      attention_kernel: str = "dot_product",
      mesh: Optional[jax.sharding.Mesh] = None,
      flash_block_sizes=None,
      dtype: jnp.dtype = jnp.bfloat16,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      text_device: Optional[str] = None,
      logical_axis_rules=None,
  ):
    transformer = create_z_image_transformer(
        model_id, nnx.Rngs(rng), attention_kernel, mesh, flash_block_sizes, dtype, weights_dtype, logical_axis_rules
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id, subfolder="vae", from_pt=True, use_safetensors=True, dtype=weights_dtype
    )
    if mesh is not None:
      replicated = NamedSharding(mesh, P())
      vae_params = jax.tree.map(lambda value: jax.device_put(value, replicated), vae_params)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=True)
    text_encoder = AutoModelForCausalLM.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_device = text_device or ("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(text_device).eval()
    return cls(transformer, vae, vae_params, tokenizer, text_encoder, dtype, mesh)

  def encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> list[jax.Array]:
    chat = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    # Z-Image consumes only unmasked Qwen states. Avoiding 512-token right
    # padding is materially faster for its CPU text-encoder handoff.
    inputs = self.tokenizer(chat, padding=True, max_length=max_sequence_length, truncation=True, return_tensors="pt")
    device = next(self.text_encoder.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
      embeddings = self.text_encoder(**inputs, output_hidden_states=True).hidden_states[-2]
    mask = inputs["attention_mask"].bool()
    return [
        jnp.asarray(embeddings[index][mask[index]].float().cpu().numpy(), dtype=self.dtype) for index in range(len(mask))
    ]

  @staticmethod
  def _sigmas(num_inference_steps: int, shift: float) -> jax.Array:
    sigmas = jnp.linspace(1.0, 0.0, num_inference_steps + 1, dtype=jnp.float32)
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

  def __call__(
      self,
      prompt: str,
      height: int = 1024,
      width: int = 1024,
      num_inference_steps: int = 9,
      guidance_scale: float = 0.0,
      seed: int = 0,
      max_sequence_length: int = 512,
      output_type: str = "pil",
  ):
    del guidance_scale  # The published Turbo configuration uses guidance_scale=0.
    vae_scale = self.vae_scale_factor * 2
    if height % vae_scale or width % vae_scale:
      raise ValueError(f"height and width must be divisible by {vae_scale}.")
    prompt_embeds = self.encode_prompt(prompt, max_sequence_length)
    latent_height, latent_width = 2 * (height // vae_scale), 2 * (width // vae_scale)
    latents = jax.random.normal(
        jax.random.key(seed), (1, self.transformer.in_channels, latent_height, latent_width), self.dtype
    )
    # The released scheduler is FlowMatchEulerDiscreteScheduler(shift=3.0).
    sigmas = self._sigmas(num_inference_steps, shift=3.0)
    with self.mesh if self.mesh is not None else nullcontext():
      for index in range(num_inference_steps):
        timestep = jnp.full((1,), 1.0 - sigmas[index], dtype=self.dtype)
        prediction = self.transformer([latents[0, :, None]], timestep, prompt_embeds, return_dict=False)[0][0][:, 0]
        latents = latents + (sigmas[index + 1] - sigmas[index]) * (-prediction[None].astype(latents.dtype))

      decoded = self.vae.apply(
          {"params": self.vae_params},
          latents / self.vae.config.scaling_factor + self.vae_shift_factor,
          deterministic=True,
          method=self.vae.decode,
      ).sample
    image = np.asarray((decoded[0].transpose(1, 2, 0) / 2 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
    return Image.fromarray(image) if output_type == "pil" else image
