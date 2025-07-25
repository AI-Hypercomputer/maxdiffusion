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
import math
import os
from jax import Array
from typing import Optional, List, Union, Tuple
from einops import rearrange
import torch.nn.functional as F
from maxdiffusion.models.ltx_video.autoencoders.vae_torchax import TorchaxCausalVideoAutoencoder
from transformers import (FlaxT5EncoderModel, AutoTokenizer)
from torchax import interop
from torchax import default_env
import json
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from huggingface_hub import hf_hub_download
from maxdiffusion.models.ltx_video.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from maxdiffusion.models.ltx_video.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    un_normalize_latents,
    normalize_latents,
)
from maxdiffusion.models.ltx_video.autoencoders.latent_upsampler import LatentUpsampler
from maxdiffusion.models.ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from types import NoneType
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.models.ltx_video.transformers.symmetric_patchifier import SymmetricPatchifier
from maxdiffusion.models.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ...pyconfig import HyperParameters
from ...schedulers.scheduling_rectified_flow import FlaxRectifiedFlowMultistepScheduler, RectifiedFlowSchedulerState
from ...max_utils import (create_device_mesh, setup_initial_state, get_memory_allocations)
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import functools
import orbax.checkpoint as ocp


def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, encoder_attention_segment_ids):
  # Note: reference shape annotated for first pass default inference parameters
  print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)  # (3, 256, 4096) float32
  print("fractional_coords.shape: ", fractional_coords.shape, fractional_coords.dtype)  # (3, 3, 3072) float32
  print("latents.shape: ", latents.shape, latents.dtype)  # (1, 3072, 128) float 32
  print(
      "encoder_attention_segment_ids.shape: ", encoder_attention_segment_ids.shape, encoder_attention_segment_ids.dtype
  )  # (3, 256) int32


class LTXVideoPipeline:

  def __init__(
      self,
      transformer: Transformer3DModel,
      scheduler: FlaxRectifiedFlowMultistepScheduler,
      scheduler_state: RectifiedFlowSchedulerState,
      vae: TorchaxCausalVideoAutoencoder,
      text_encoder,
      patchifier,
      tokenizer,
      prompt_enhancer_image_caption_model,
      prompt_enhancer_image_caption_processor,
      prompt_enhancer_llm_model,
      prompt_enhancer_llm_tokenizer,
      devices_array: np.array,
      mesh: Mesh,
      config: HyperParameters,
      transformer_state: Dict[Any, Any] = None,
      transformer_state_shardings: Dict[Any, Any] = NoneType,
  ):
    self.transformer = transformer
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.p_run_inference = None
    self.transformer_state = transformer_state
    self.transformer_state_shardings = transformer_state_shardings
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state
    self.vae = vae
    self.text_encoder = text_encoder
    self.patchifier = patchifier
    self.tokenizer = tokenizer
    self.prompt_enhancer_image_caption_model = prompt_enhancer_image_caption_model
    self.prompt_enhancer_image_caption_processor = prompt_enhancer_image_caption_processor
    self.prompt_enhancer_llm_model = prompt_enhancer_llm_model
    self.prompt_enhancer_llm_tokenizer = prompt_enhancer_llm_tokenizer
    self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(self.vae)

  @classmethod
  def load_scheduler(cls, ckpt_path, config):
    if config.sampler == "from_checkpoint" or not config.sampler:
      scheduler = FlaxRectifiedFlowMultistepScheduler.from_pretrained_jax(ckpt_path)
    else:
      scheduler = FlaxRectifiedFlowMultistepScheduler(
          sampler=("Uniform" if config.sampler.lower() == "uniform" else "LinearQuadratic")
      )
    scheduler_state = scheduler.create_state()

    return scheduler, scheduler_state

  @classmethod
  def load_transformer(cls, config):
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with open(config.config_path, "r") as f:
      model_config = json.load(f)

    ignored_keys = [
        "_class_name",
        "_diffusers_version",
        "_name_or_path",
        "causal_temporal_positioning",
        "in_channels",
    ]
    in_channels = model_config["in_channels"]
    for name in ignored_keys:
      if name in model_config:
        del model_config[name]
    transformer = Transformer3DModel(
        **model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh=mesh
    )
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    weights_init_fn = functools.partial(
        transformer.init_weights, in_channels, subkey, model_config["caption_channels"], eval_only=True
    )
    # loading from weight checkpoint
    models_dir = config.output_dir
    jax_weights_dir = os.path.join(models_dir, "jax_weights")
    checkpoint_manager = ocp.CheckpointManager(jax_weights_dir)
    transformer_state, transformer_state_shardings = setup_initial_state(
        model=transformer,
        tx=None,
        config=config,
        mesh=mesh,
        weights_init_fn=weights_init_fn,
        checkpoint_manager=checkpoint_manager,
        checkpoint_item=" ",
        model_params=None,
        training=False,
    )
    transformer_state = jax.device_put(transformer_state, transformer_state_shardings)
    get_memory_allocations()

    return transformer, transformer_state, transformer_state_shardings

  @classmethod
  def load_vae(cls, ckpt_path):
    torch_vae = CausalVideoAutoencoder.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    # in torchax
    with default_env():
      torch_vae = torch_vae.to("jax")
      jax_vae = TorchaxCausalVideoAutoencoder(torch_vae)
    return jax_vae

  @classmethod
  def load_text_encoder(cls, ckpt_path):
    t5_encoder = FlaxT5EncoderModel.from_pretrained(ckpt_path)
    return t5_encoder

  @classmethod
  def load_tokenizer(cls, config, ckpt_path):
    t5_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, max_length=config.max_sequence_length, use_fast=True)
    return t5_tokenizer

  @classmethod
  def load_prompt_enhancement(cls, config):
    prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
        config.prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
    )
    prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
        config.prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
    )
    prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained(
        config.prompt_enhancer_llm_model_name_or_path,
        torch_dtype="bfloat16",
    )
    prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
        config.prompt_enhancer_llm_model_name_or_path,
    )
    return (
        prompt_enhancer_image_caption_model,
        prompt_enhancer_image_caption_processor,
        prompt_enhancer_llm_model,
        prompt_enhancer_llm_tokenizer,
    )

  @classmethod
  def from_pretrained(cls, config: HyperParameters, enhance_prompt: bool = False):
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    transformer, transformer_state, transformer_state_shardings = cls.load_transformer(config)

    models_dir = config.output_dir
    ltxv_model_name_or_path = "ltxv-13b-0.9.7-dev.safetensors"
    if not os.path.isfile(ltxv_model_name_or_path):
      ltxv_model_path = hf_hub_download(
          repo_id="Lightricks/LTX-Video",
          filename=ltxv_model_name_or_path,
          local_dir=models_dir,
          repo_type="model",
      )
    else:
      ltxv_model_path = ltxv_model_name_or_path

    scheduler, scheduler_state = cls.load_scheduler(ltxv_model_path, config)

    vae = cls.load_vae(ltxv_model_path)
    text_encoder = cls.load_text_encoder(config.text_encoder_model_name_or_path)
    patchifier = SymmetricPatchifier(patch_size=1)
    tokenizer = cls.load_tokenizer(config, config.text_encoder_model_name_or_path)

    if enhance_prompt:
      (
          prompt_enhancer_image_caption_model,
          prompt_enhancer_image_caption_processor,
          prompt_enhancer_llm_model,
          prompt_enhancer_llm_tokenizer,
      ) = cls.load_prompt_enhancement(config)
    else:
      (
          prompt_enhancer_image_caption_model,
          prompt_enhancer_image_caption_processor,
          prompt_enhancer_llm_model,
          prompt_enhancer_llm_tokenizer,
      ) = (None, None, None, None)

    return LTXVideoPipeline(
        transformer=transformer,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
        vae=vae,
        text_encoder=text_encoder,
        patchifier=patchifier,
        tokenizer=tokenizer,
        prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
        prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
        prompt_enhancer_llm_model=prompt_enhancer_llm_model,
        prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
        devices_array=devices_array,
        mesh=mesh,
        config=config,
        transformer_state=transformer_state,
        transformer_state_shardings=transformer_state_shardings,
    )

  @classmethod
  def _text_preprocessing(self, text):
    if not isinstance(text, (tuple, list)):
      text = [text]

    def process(text: str):
      text = text.strip()
      return text

    return [process(t) for t in text]

  def denoising_step(
      scheduler,
      latents: Array,
      noise_pred: Array,
      current_timestep: Optional[Array],
      conditioning_mask: Optional[Array],
      t: float,
      extra_step_kwargs: Dict,
      t_eps: float = 1e-6,
      stochastic_sampling: bool = False,
  ) -> Array:
    # Denoise the latents using the scheduler
    denoised_latents = scheduler.step(
        noise_pred,
        t if current_timestep is None else current_timestep,
        latents,
        **extra_step_kwargs,
        stochastic_sampling=stochastic_sampling,
    )

    if conditioning_mask is None:
      return denoised_latents

    tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).astype(jnp.bool_)
    tokens_to_denoise_mask = jnp.expand_dims(tokens_to_denoise_mask, axis=-1)
    return jnp.where(tokens_to_denoise_mask, denoised_latents, latents)

  def retrieve_timesteps(  # currently doesn't support custom timesteps
      self,
      scheduler: FlaxRectifiedFlowMultistepScheduler,
      latent_shape,
      scheduler_state: RectifiedFlowSchedulerState,
      num_inference_steps: Optional[int] = None,
      timesteps: Optional[List[int]] = None,
      skip_initial_inference_steps: int = 0,
      skip_final_inference_steps: int = 0,
  ):
    scheduler_state = scheduler.set_timesteps(
        state=scheduler_state, samples_shape=latent_shape, num_inference_steps=num_inference_steps
    )
    timesteps = scheduler_state.timesteps
    if (
        skip_initial_inference_steps < 0
        or skip_final_inference_steps < 0
        or skip_initial_inference_steps + skip_final_inference_steps
        >= num_inference_steps  # Use the original num_inference_steps here for the check
    ):
      raise ValueError(
          "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
      )
    timesteps = timesteps[skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps]
    scheduler_state = scheduler.set_timesteps(timesteps=timesteps, samples_shape=latent_shape, state=scheduler_state)

    return scheduler_state

  def encode_prompt(  # currently only supports passing in a prompt
      self,
      prompt: Union[str, List[str]],
      do_classifier_free_guidance: bool = True,
      negative_prompt: str = "",
      num_images_per_prompt: int = 1,
      text_encoder_max_tokens: int = 256,
      **kwargs,
  ):
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)

    max_length = text_encoder_max_tokens  # TPU supports only lengths multiple of 128

    assert self.text_encoder is not None, "You should provide either prompt_embeds or self.text_encoder should not be None,"
    prompt = self._text_preprocessing(prompt)
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = jnp.array(text_inputs.input_ids)

    prompt_attention_mask = jnp.array(text_inputs.attention_mask)
    prompt_embeds = self.text_encoder(text_input_ids, attention_mask=prompt_attention_mask)
    prompt_embeds = prompt_embeds[0]

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = jnp.tile(prompt_embeds, (1, num_images_per_prompt, 1))
    prompt_embeds = jnp.reshape(prompt_embeds, (bs_embed * num_images_per_prompt, seq_len, -1))
    prompt_attention_mask = jnp.tile(prompt_attention_mask, (1, num_images_per_prompt))
    prompt_attention_mask = jnp.reshape(prompt_attention_mask, (bs_embed * num_images_per_prompt, -1))

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
      uncond_tokens = self._text_preprocessing(negative_prompt)
      uncond_tokens = uncond_tokens * batch_size
      max_length = prompt_embeds.shape[1]
      uncond_input = self.tokenizer(
          uncond_tokens,
          padding="max_length",
          max_length=max_length,
          truncation=True,
          return_attention_mask=True,
          add_special_tokens=True,
          return_tensors="pt",
      )
      negative_prompt_attention_mask = jnp.array(uncond_input.attention_mask)

      negative_prompt_embeds = self.text_encoder(
          jnp.array(uncond_input.input_ids),
          attention_mask=negative_prompt_attention_mask,
      )
      negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
      seq_len = negative_prompt_embeds.shape[1]

      negative_prompt_embeds = jnp.tile(negative_prompt_embeds, (1, num_images_per_prompt, 1))
      negative_prompt_embeds = jnp.reshape(negative_prompt_embeds, (batch_size * num_images_per_prompt, seq_len, -1))

      negative_prompt_attention_mask = jnp.tile(negative_prompt_attention_mask, (1, num_images_per_prompt))
      negative_prompt_attention_mask = jnp.reshape(negative_prompt_attention_mask, (bs_embed * num_images_per_prompt, -1))
    else:
      negative_prompt_embeds = None
      negative_prompt_attention_mask = None

    return (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    )

  def prepare_latents(  # currently no support for media item encoding, since encoder isn't tested
      self,
      latents: Optional[jnp.ndarray],
      timestep: float,
      latent_shape: Tuple[Any, ...],
      dtype: jnp.dtype,
      key: jax.random.PRNGKey,
  ) -> jnp.ndarray:
    """
    Prepares initial latents for a diffusion process, potentially encoding media items
    or adding noise
    """

    if latents is not None:
      assert latents.shape == latent_shape, f"Latents have to be of shape {latent_shape} but are {latents.shape}."

    # For backward compatibility, generate noise in the "patchified" shape and rearrange
    b, c, f, h, w = latent_shape

    # Generate noise using jax.random.normal
    noise_intermediate_shape = (b, f * h * w, c)
    noise = jax.random.normal(key, noise_intermediate_shape, dtype=dtype)

    # Rearrange "b (f h w) c -> b c f h w"
    # Step 1: Reshape to (b, f, h, w, c)
    noise = noise.reshape(b, f, h, w, c)
    # Step 2: Permute/Transpose to (b, c, f, h, w)
    noise = jnp.transpose(noise, (0, 4, 1, 2, 3))  # Old (b,f,h,w,c) -> New (b,c,f,h,w)

    if latents is None:
      latents = noise
    else:
      # Noise the latents to the required (first) timestep
      timestep_array = jnp.array(timestep, dtype=dtype)
      latents = timestep_array * noise + (1 - timestep_array) * latents

    return latents

  def prepare_conditioning(  # no support for conditioning items, conditioning mask, needs to convert to torch before patchifier
      self,
      init_latents: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    assert isinstance(self.vae, TorchaxCausalVideoAutoencoder)
    init_latents = torch.from_numpy(np.array(init_latents))
    init_latents, init_latent_coords = self.patchifier.patchify(latents=init_latents)
    init_pixel_coords = latent_to_pixel_coords(init_latent_coords, self.vae, causal_fix=True)
    return (
        jnp.array(init_latents.to(torch.float32).detach().numpy()),
        jnp.array(init_pixel_coords.to(torch.float32).detach().numpy()),
        0,
    )

  def denormalize(self, images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Borrowed from diffusers.image_processor
    Denormalize an image array to [0,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to denormalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The denormalized image array.
    """
    return (images * 0.5 + 0.5).clamp(0, 1)

  def _denormalize_conditionally(self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None) -> torch.Tensor:
    r"""
    Borrowed from diffusers.image_processor
    Denormalize a batch of images based on a condition list.

    Args:
        images (`torch.Tensor`):
            The input image tensor.
        do_denormalize (`Optional[List[bool]`, *optional*, defaults to `None`):
            A list of booleans indicating whether to denormalize each image in the batch. If `None`, will use the
            value of `do_normalize` in the `VaeImageProcessor` config.
    """
    if do_denormalize is None:
      return self.denormalize(images)

    return torch.stack([self.denormalize(images[i]) if do_denormalize[i] else images[i] for i in range(images.shape[0])])

  def postprocess_to_output_type(self, image, output_type):
    """
    Borrowed from diffusers.image_processor
    Currrently supporting output type latent and pt
    """
    if not isinstance(image, torch.Tensor):
      raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")

    if output_type not in ["latent", "pt", "np", "pil"]:
      output_type = "np"

    if output_type == "latent":
      return image
    image = self._denormalize_conditionally(image, None)
    if output_type == "pt":
      return image

  def __call__(
      self,
      config,
      height: int,
      width: int,
      num_frames: int,
      negative_prompt: str = "",
      num_images_per_prompt: Optional[int] = 1,
      frame_rate: int = 30,
      latents: Optional[jnp.ndarray] = None,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      guidance_timesteps: Optional[List[int]] = None,
      decode_timestep: Union[List[float], float] = 0.05,
      decode_noise_scale: Optional[List[float]] = 0.025,
      enhance_prompt: bool = False,
      text_encoder_max_tokens: int = 256,
      num_inference_steps: int = 50,
      guidance_scale: Union[float, List[float]] = 4.5,
      rescaling_scale: Union[float, List[float]] = 0.7,
      stg_scale: Union[float, List[float]] = 1.0,
      skip_initial_inference_steps: int = 0,
      skip_final_inference_steps: int = 0,
      cfg_star_rescale: bool = False,
      seed: int = 0,
      skip_layer_strategy: Optional[SkipLayerStrategy] = None,
      skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
      **kwargs,
  ):
    key = jax.random.PRNGKey(seed)
    prompt = self.config.prompt
    is_video = kwargs.get("is_video", False)
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)

    latent_height = height // self.vae_scale_factor
    latent_width = width // self.vae_scale_factor
    latent_num_frames = num_frames // self.video_scale_factor
    if isinstance(self.vae, TorchaxCausalVideoAutoencoder) and is_video:
      latent_num_frames += 1
    with open(config.config_path, "r") as f:
      model_config = json.load(f)
    latent_shape = (
        batch_size * num_images_per_prompt,
        model_config["in_channels"],
        latent_num_frames,
        latent_height,
        latent_width,
    )
    scheduler_state = self.retrieve_timesteps(
        self.scheduler,
        latent_shape,
        self.scheduler_state,
        num_inference_steps,
        None,
        skip_initial_inference_steps,
        skip_final_inference_steps,
    )

    # set up guidance
    guidance_mapping = []

    if guidance_timesteps:
      for timestep in scheduler_state.timesteps:
        indices = [i for i, val in enumerate(guidance_timesteps) if val <= timestep]
        guidance_mapping.append(indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1))

    if not isinstance(guidance_scale, list):
      guidance_scale = [guidance_scale] * len(scheduler_state.timesteps)
    else:
      guidance_scale = [guidance_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]

    if not isinstance(stg_scale, list):
      stg_scale = [stg_scale] * len(scheduler_state.timesteps)
    else:
      stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]

    if not isinstance(rescaling_scale, list):
      rescaling_scale = [rescaling_scale] * len(scheduler_state.timesteps)
    else:
      rescaling_scale = [rescaling_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]

    guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]
    do_classifier_free_guidance = any(x > 1.0 for x in guidance_scale)
    do_spatio_temporal_guidance = any(x > 0.0 for x in stg_scale)
    do_rescaling = any(x != 1.0 for x in rescaling_scale)

    num_conds = 1
    if do_classifier_free_guidance:
      num_conds += 1
    if do_spatio_temporal_guidance:
      num_conds += 1

    # prepare skip block list
    is_list_of_lists = bool(skip_block_list) and isinstance(skip_block_list[0], list)

    if not is_list_of_lists:
      skip_block_list = [skip_block_list] * len(scheduler_state.timesteps)
    else:
      new_skip_block_list = []
      for i in range(len(scheduler_state.timesteps)):
        new_skip_block_list.append(skip_block_list[guidance_mapping[i]])

      skip_block_list = new_skip_block_list

    if do_spatio_temporal_guidance:
      if skip_block_list is not None:
        skip_layer_masks = [
            self.transformer.create_skip_layer_mask(batch_size, num_conds, num_conds - 1, skip_blocks)
            for skip_blocks in skip_block_list
        ]

    if enhance_prompt:
      prompt = generate_cinematic_prompt(
          self.prompt_enhancer_image_caption_model,
          self.prompt_enhancer_image_caption_processor,
          self.prompt_enhancer_llm_model,
          self.prompt_enhancer_llm_tokenizer,
          prompt,
          None,  # conditioning items set to None
          max_new_tokens=text_encoder_max_tokens,
      )

    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = self.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_encoder_max_tokens=text_encoder_max_tokens,
    )
    prompt_embeds_batch = prompt_embeds
    prompt_attention_mask_batch = prompt_attention_mask

    if do_classifier_free_guidance:
      prompt_embeds_batch = jnp.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
      prompt_attention_mask_batch = jnp.concatenate([negative_prompt_attention_mask, prompt_attention_mask], axis=0)
    if do_spatio_temporal_guidance:
      prompt_embeds_batch = jnp.concatenate([prompt_embeds_batch, prompt_embeds], axis=0)
      prompt_attention_mask_batch = jnp.concatenate(
          [
              prompt_attention_mask_batch,
              prompt_attention_mask,
          ],
          axis=0,
      )

    # optionally pass in a latent here
    latents = self.prepare_latents(
        latents=latents,
        timestep=scheduler_state.timesteps[0],
        latent_shape=latent_shape,
        dtype=jnp.float32,
        key=key,
    )

    latents, pixel_coords, num_cond_latents = self.prepare_conditioning(
        init_latents=latents,
    )

    pixel_coords = jnp.concatenate([pixel_coords] * num_conds, axis=0)
    fractional_coords = pixel_coords.astype(jnp.float32)
    fractional_coords = fractional_coords.at[:, 0].set(fractional_coords[:, 0] * (1.0 / frame_rate))
    validate_transformer_inputs(prompt_embeds_batch, fractional_coords, latents, prompt_attention_mask_batch)

    p_run_inference = functools.partial(
        run_inference,
        transformer=self.transformer,
        config=self.config,
        mesh=self.mesh,
        fractional_cords=fractional_coords,
        prompt_embeds=prompt_embeds_batch,
        segment_ids=None,
        encoder_attention_segment_ids=prompt_attention_mask_batch,
        scheduler=self.scheduler,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_conds=num_conds,
        guidance_scale=guidance_scale,
        do_spatio_temporal_guidance=do_spatio_temporal_guidance,
        stg_scale=stg_scale,
        do_rescaling=do_rescaling,
        rescaling_scale=rescaling_scale,
        batch_size=batch_size,
        skip_layer_masks=skip_layer_masks,
        skip_layer_strategy=skip_layer_strategy,
        cfg_star_rescale=cfg_star_rescale,
    )

    with self.mesh:
      latents, scheduler_state = p_run_inference(
          transformer_state=self.transformer_state, latents=latents, scheduler_state=scheduler_state
      )

    latents = latents[:, num_cond_latents:]

    latents = self.patchifier.unpatchify(
        latents=latents,
        output_height=latent_height,
        output_width=latent_width,
        out_channels=model_config["in_channels"] // math.prod(self.patchifier.patch_size),
    )

    if output_type != "latent":
      if self.vae.decoder.timestep_conditioning:
        noise = jax.random.normal(key, latents.shape, dtype=latents.dtype)
        # Convert decode_timestep to a list if it's not already one
        if not isinstance(decode_timestep, (list, jnp.ndarray)):
          decode_timestep = [decode_timestep] * latents.shape[0]

        # Handle decode_noise_scale
        if decode_noise_scale is None:
          decode_noise_scale = decode_timestep
        elif not isinstance(decode_noise_scale, (list, jnp.ndarray)):
          decode_noise_scale = [decode_noise_scale] * latents.shape[0]

        decode_timestep = jnp.array(decode_timestep, dtype=jnp.float32)

        # Reshape decode_noise_scale for broadcasting
        decode_noise_scale = jnp.array(decode_noise_scale, dtype=jnp.float32)
        decode_noise_scale = jnp.reshape(decode_noise_scale, (latents.shape[0],) + (1,) * (latents.ndim - 1))

        # Apply the noise and scale
        latents = latents * (1 - decode_noise_scale) + noise * decode_noise_scale
      else:
        decode_timestep = None
      image = self.vae.decode(
          latents=jax.device_put(latents, jax.devices("tpu")[0]),
          is_video=is_video,
          vae_per_channel_normalize=kwargs.get("vae_per_channel_normalize", True),
          timestep=decode_timestep,
      )
      # convert back to torch to postprocess using the diffusers library
      image = self.postprocess_to_output_type(
          torch.from_numpy(np.asarray(image.astype(jnp.float16))), output_type=output_type
      )

    else:
      image = latents

    if not return_dict:
      return (image,)

    return image


def transformer_forward_pass(
    latents,
    state,
    noise_cond,
    transformer,
    fractional_cords,
    prompt_embeds,
    segment_ids,
    encoder_attention_segment_ids,
    skip_layer_mask,
    skip_layer_strategy,
):

  noise_pred = transformer.apply(
      {"params": state.params},
      hidden_states=latents,
      indices_grid=fractional_cords,
      encoder_hidden_states=prompt_embeds,
      timestep=noise_cond,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids,
      skip_layer_mask=skip_layer_mask,
      skip_layer_strategy=skip_layer_strategy,
  )
  return noise_pred, state


def run_inference(
    transformer_state,
    transformer,
    config,
    mesh,
    latents,
    fractional_cords,
    prompt_embeds,
    scheduler,
    segment_ids,
    encoder_attention_segment_ids,
    scheduler_state,
    do_classifier_free_guidance,
    num_conds,
    guidance_scale,
    do_spatio_temporal_guidance,
    stg_scale,
    do_rescaling,
    rescaling_scale,
    batch_size,
    skip_layer_masks,
    skip_layer_strategy,
    cfg_star_rescale,
):
  for i, t in enumerate(scheduler_state.timesteps):
    current_timestep = t
    latent_model_input = jnp.concatenate([latents] * num_conds) if num_conds > 1 else latents
    if not isinstance(current_timestep, (jnp.ndarray, jax.Array)):
      if isinstance(current_timestep, float):
        dtype = jnp.float32
      else:
        dtype = jnp.int32

      current_timestep = jnp.array(
          [current_timestep],
          dtype=dtype,
      )
    elif current_timestep.ndim == 0:
      current_timestep = jnp.expand_dims(current_timestep, axis=0)

    # Broadcast to batch dimension
    current_timestep = jnp.broadcast_to(current_timestep, (latent_model_input.shape[0], 1))

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      noise_pred, transformer_state = transformer_forward_pass(
          latent_model_input,
          transformer_state,
          current_timestep,
          transformer,
          fractional_cords,
          prompt_embeds,
          segment_ids,
          encoder_attention_segment_ids,
          skip_layer_mask=(skip_layer_masks[i] if skip_layer_masks is not None else None),
          skip_layer_strategy=skip_layer_strategy,
      )

    # perform guidance on noise prediction
    if do_spatio_temporal_guidance:
      chunks = jnp.split(noise_pred, num_conds, axis=0)
      noise_pred_text = chunks[-2]
      noise_pred_text_perturb = chunks[-1]

    if do_classifier_free_guidance:
      chunks = jnp.split(noise_pred, num_conds, axis=0)
      noise_pred_uncond = chunks[0]
      noise_pred_text = chunks[1]
      if cfg_star_rescale:
        positive_flat = noise_pred_text.reshape(batch_size, -1)
        negative_flat = noise_pred_uncond.reshape(batch_size, -1)
        dot_product = jnp.sum(positive_flat * negative_flat, axis=1, keepdims=True)
        squared_norm = jnp.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
        alpha = dot_product / squared_norm
        alpha = alpha.reshape(batch_size, 1, 1)
        noise_pred_uncond = alpha * noise_pred_uncond
      noise_pred = noise_pred_uncond + guidance_scale[i] * (noise_pred_text - noise_pred_uncond)
    elif do_spatio_temporal_guidance:
      noise_pred = noise_pred_text

    if do_spatio_temporal_guidance:
      noise_pred = noise_pred + stg_scale[i] * (noise_pred_text - noise_pred_text_perturb)
      if do_rescaling and stg_scale[i] > 0.0:
        noise_pred_text_std = jnp.std(noise_pred_text.reshape(batch_size, -1), axis=1, keepdims=True)
        noise_pred_std = jnp.std(noise_pred.reshape(batch_size, -1), axis=1, keepdims=True)

        factor = noise_pred_text_std / noise_pred_std
        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

        noise_pred = noise_pred * factor.reshape(batch_size, 1, 1)
    current_timestep = current_timestep[:1]
    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, current_timestep[0][0], latents).to_tuple()

  return latents, scheduler_state


def adain_filter_latent(latents: jnp.ndarray, reference_latents: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
  """
  Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
  statistics from a reference latent tensor, implemented in JAX.

  Args:
      latents (jax.Array): Input latents to normalize. Expected shape (B, C, F, H, W).
      reference_latents (jax.Array): The reference latents providing style statistics.
                                     Expected shape (B, C, F, H, W).
      factor (float): Blending factor between original and transformed latent.
                     Range: -10.0 to 10.0, Default: 1.0

  Returns:
      jax.Array: The transformed latent tensor.
  """
  with default_env():
    latents = jax.device_put(latents, jax.devices("tpu")[0])
    reference_latents = jax.device_put(reference_latents, jax.devices("tpu")[0])

    # Define the core AdaIN operation for a single (F, H, W) slice.
    # This function will be vmapped over batch (B) and channel (C) dimensions.
    def _adain_single_slice(latent_slice: jnp.ndarray, ref_latent_slice: jnp.ndarray) -> jnp.ndarray:
      """
      Applies AdaIN to a single latent slice (F, H, W) based on a reference slice.
      """
      r_mean = jnp.mean(ref_latent_slice)
      r_sd = jnp.std(ref_latent_slice)

      # Calculate standard deviation and mean for the input latent slice
      i_mean = jnp.mean(latent_slice)
      i_sd = jnp.std(latent_slice)
      i_sd_safe = jnp.where(i_sd < 1e-6, 1.0, i_sd)
      normalized_latent = (latent_slice - i_mean) / i_sd_safe
      transformed_latent_slice = normalized_latent * r_sd + r_mean
      return transformed_latent_slice

    transformed_latents_core = jax.vmap(
        jax.vmap(_adain_single_slice, in_axes=(0, 0)), in_axes=(0, 0)  # Vmap over batch (axis 0)
    )(latents, reference_latents)
    result_blended = latents * (1.0 - factor) + transformed_latents_core * factor

  return result_blended


class LTXMultiScalePipeline:

  @classmethod
  def load_latent_upsampler(cls, config):
    spatial_upscaler_model_name_or_path = config.spatial_upscaler_model_path

    if spatial_upscaler_model_name_or_path and not os.path.isfile(spatial_upscaler_model_name_or_path):
      spatial_upscaler_model_path = hf_hub_download(
          repo_id="Lightricks/LTX-Video",
          filename=spatial_upscaler_model_name_or_path,
          local_dir=config.output_dir,
          repo_type="model",
      )
    else:
      spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
    if not config.spatial_upscaler_model_path:
      raise ValueError(
          "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
      )
    latent_upsampler = LatentUpsampler.from_pretrained(spatial_upscaler_model_path)
    latent_upsampler.eval()
    return latent_upsampler

  def _upsample_latents(self, latest_upsampler: LatentUpsampler, latents: jnp.ndarray):
    latents = jax.device_put(latents, jax.devices("tpu")[0])
    with default_env():
      latents = un_normalize_latents(interop.torch_view(latents), self.vae, vae_per_channel_normalize=True)
      upsampled_latents = latest_upsampler(torch.from_numpy(np.array(latents)))  # converted back to torch before upsampler
      upsampled_latents = normalize_latents(
          interop.torch_view(jnp.array(upsampled_latents.detach().numpy())), self.vae, vae_per_channel_normalize=True
      )
    return upsampled_latents

  def __init__(self, video_pipeline: LTXVideoPipeline):
    self.video_pipeline = video_pipeline
    self.vae = video_pipeline.vae

  def __call__(
      self, height, width, num_frames, is_video, output_type, config, seed: int = 0, enhance_prompt: bool = False
  ) -> Any:
    # first pass
    original_output_type = output_type
    output_type = "latent"
    result = self.video_pipeline(
        config=config,
        height=height,
        width=width,
        enhance_prompt=enhance_prompt,
        num_frames=num_frames,
        is_video=is_video,
        output_type=output_type,
        seed=seed,
        guidance_scale=config.first_pass["guidance_scale"],
        stg_scale=config.first_pass["stg_scale"],
        rescaling_scale=config.first_pass["rescaling_scale"],
        skip_initial_inference_steps=config.first_pass["skip_initial_inference_steps"],
        skip_final_inference_steps=config.first_pass["skip_final_inference_steps"],
        num_inference_steps=config.first_pass["num_inference_steps"],
        guidance_timesteps=config.first_pass["guidance_timesteps"],
        cfg_star_rescale=config.first_pass["cfg_star_rescale"],
        skip_layer_strategy=None,
        skip_block_list=config.first_pass["skip_block_list"],
    )
    latents = result
    print("first pass done")
    latent_upsampler = self.load_latent_upsampler(config)
    upsampled_latents = self._upsample_latents(latent_upsampler, latents)
    upsampled_latents = adain_filter_latent(latents=upsampled_latents, reference_latents=latents)

    # second pass
    latents = upsampled_latents
    result = self.video_pipeline(
        config=config,
        height=height * 2,
        width=width * 2,
        enhance_prompt=enhance_prompt,
        num_frames=num_frames,
        is_video=is_video,
        seed=seed,
        output_type=original_output_type,
        latents=latents,
        guidance_scale=config.second_pass["guidance_scale"],
        stg_scale=config.second_pass["stg_scale"],
        rescaling_scale=config.second_pass["rescaling_scale"],
        skip_initial_inference_steps=config.second_pass["skip_initial_inference_steps"],
        skip_final_inference_steps=config.second_pass["skip_final_inference_steps"],
        num_inference_steps=config.second_pass["num_inference_steps"],
        guidance_timesteps=config.second_pass["guidance_timesteps"],
        cfg_star_rescale=config.second_pass["cfg_star_rescale"],
        skip_layer_strategy=None,
        skip_block_list=config.second_pass["skip_block_list"],
    )

    if original_output_type != "latent":
      num_frames = result.shape[2]
      videos = rearrange(result, "b c f h w -> (b f) c h w")

      videos = F.interpolate(
          videos,
          size=(height, width),
          mode="bilinear",
          align_corners=False,
      )
      videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
      result = videos

    return result
