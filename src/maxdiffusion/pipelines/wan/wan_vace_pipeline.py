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

from typing import Any, List, Union, Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax
import flax.linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...image_processor import PipelineImageInput
from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated
from ...models.wan.wan_utils import load_wan_transformer
from ...models.wan.transformers.transformer_wan_vace import WanVACEModel
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler
from ...models.modeling_flax_pytorch_utils import torch2jax
from .wan_pipeline import WanPipeline, cast_with_exclusion
import torch
import PIL


def retrieve_latents(
    encoder_output: torch.Tensor, rngs=None, sample_mode: str = "sample"
):
  """Extracts the latent codes from the encoder object.

  From https://github.com/huggingface/diffusers/blob/8d415a6f481ff1b26168c046267628419650f930/src/diffusers/pipelines/wan/pipeline_wan_vace.py#L128C1-L128C4
  """
  if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
    return encoder_output.latent_dist.sample(rngs)
  elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
    return encoder_output.latent_dist.mode()
  elif hasattr(encoder_output, "latents"):
    return encoder_output.latents
  else:
    raise AttributeError("Could not access latents of provided encoder_output")


# For some reason, jitting this function increases the memory significantly, so instead manually move weights to device.
def create_sharded_logical_transformer(
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None, subfolder: str = ""
):

  def create_model(rngs: nnx.Rngs, wan_config: dict):
    wan_vace_transformer = WanVACEModel(**wan_config, rngs=rngs)
    return wan_vace_transformer

  # 1. Load config.
  if restored_checkpoint:
    wan_config = restored_checkpoint["wan_config"]
  else:
    wan_config = WanVACEModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)
  wan_config["mesh"] = mesh
  wan_config["dtype"] = config.activations_dtype
  wan_config["weights_dtype"] = config.weights_dtype
  wan_config["attention"] = config.attention
  wan_config["precision"] = get_precision(config)
  wan_config["flash_block_sizes"] = get_flash_block_sizes(config)
  wan_config["remat_policy"] = config.remat_policy
  wan_config["names_which_can_be_saved"] = config.names_which_can_be_saved
  wan_config["names_which_can_be_offloaded"] = config.names_which_can_be_offloaded
  wan_config["flash_min_seq_length"] = config.flash_min_seq_length
  wan_config["dropout"] = config.dropout
  wan_config["scan_layers"] = False

  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, wan_config=wan_config)
  wan_vace_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_vace_transformer, nnx.Param, ...)

  # 3. retrieve the state shardings, mapping logical names to mesh axis names.
  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  # 4. Load pretrained weights and move them to device using the state shardings from (3) above.
  # This helps with loading sharded weights directly into the accelerators without fist copying them
  # all to one device and then distributing them, thus using low HBM memory.
  if restored_checkpoint:
    if "params" in restored_checkpoint["wan_state"]:  # if checkpointed with optimizer
      params = restored_checkpoint["wan_state"]["params"]
    else:  # if not checkpointed with optimizer
      params = restored_checkpoint["wan_state"]
  else:
    params = load_wan_transformer(
        config.wan_transformer_pretrained_model_name_or_path,
        eval_shapes=params,
        device="cpu",
        num_layers=wan_config["num_layers"],
        scan_layers=config.scan_layers,
        subfolder=subfolder,
    )

  params = jax.tree_util.tree_map_with_path(
      lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=config.weights_dtype), params
  )
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint:
      path = path[:-1]
    sharding = logical_state_sharding[path].value
    state[path].value = device_put_replicated(val, sharding)
  state = nnx.from_flat_state(state)

  wan_transformer = nnx.merge(graphdef, state, rest_of_state)
  return wan_transformer


class VaceWanPipeline(WanPipeline):
  r"""Pipeline for video generation using Wan + VACE.

  Currently it only supports reference image(s) + text to video generation.

  It extends `WanPipeline` to support additional conditioning signals.

  tokenizer ([`T5Tokenizer`]):
      Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
      specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  text_encoder ([`T5EncoderModel`]):
      [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
      the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  transformer ([`WanModel`]):
      Conditional Transformer to denoise the input latents.
  scheduler ([`FlaxUniPCMultistepScheduler`]):
      A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
  vae ([`AutoencoderKLWan`]):
      Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
  """

  def preprocess_conditions(
    self,
    video: Optional[PipelineImageInput] = None,
    mask: Optional[PipelineImageInput] = None,
    reference_images: Optional[PipelineImageInput] = None,
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype = None,
):
    """Prepares the conditional data for inference.

    Based on https://github.com/huggingface/diffusers/blob/17c0e79dbdf53fb6705e9c09cc1a854b84c39249/src/diffusers/pipelines/wan/pipeline_wan_vace.py#L414
    """
    if video is not None:
      raise NotImplementedError("Video support is not yet implemented.")
    else:
      video = jnp.zeros(
          (batch_size, num_frames, height, width, 3), dtype=dtype
      )
      image_size = (height, width)  # Use the height/width provider by user

    if mask is not None:
      raise NotImplementedError("Mask support is not yet implemented.")
    else:
      mask = jnp.ones_like(video)

    # Taken from
    # https://github.com/huggingface/diffusers/blob/17c0e79dbdf53fb6705e9c09cc1a854b84c39249/src/diffusers/pipelines/wan/pipeline_wan_vace.py#L464
    # Make a list of list of images where the outer list corresponds to video
    # batch size and the inner list corresponds to list of conditioning images
    # per video
    if reference_images is None or isinstance(reference_images, PIL.Image.Image):
      reference_images = [[reference_images] for _ in range(video.shape[0])]
    elif isinstance(reference_images, (list, tuple)) and isinstance(
        next(iter(reference_images)), PIL.Image.Image
    ):
      reference_images = [reference_images]
    elif (
        isinstance(reference_images, (list, tuple))
        and isinstance(next(iter(reference_images)), list)
        and isinstance(next(iter(reference_images[0])), PIL.Image.Image)
    ):
      reference_images = reference_images
    else:
      raise ValueError(
          "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
          f"`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
      )

    if video.shape[0] != len(reference_images):
      raise ValueError(
          f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
      )

    ref_images_lengths = [len(reference_images_batch) for reference_images_batch in reference_images]
    if any(l != ref_images_lengths[0] for l in ref_images_lengths):
      raise ValueError(
          f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
          "may be added in the future."
      )

    reference_images_preprocessed = []
    for reference_images_batch in reference_images:
      preprocessed_images = []
      for image in reference_images_batch:
        if image is None:
          continue
        image = self.video_processor.preprocess(image, None, None)
        img_height, img_width = image.shape[-2:]
        scale = min(image_size[0] / img_height, image_size[1] / img_width)
        new_height, new_width = int(img_height * scale), int(img_width * scale)
        # TODO: should we use jax/TF-based resizing here?
        resized_image = torch.nn.functional.interpolate(
            image, size=(new_height, new_width), mode="bilinear", align_corners=False
        ).squeeze(0)  # [C, H, W]

        top = (image_size[0] - new_height) // 2
        left = (image_size[1] - new_width) // 2
        canvas = torch.ones(3, *image_size, dtype=torch.float32)
        canvas[:, top : top + new_height, left : left + new_width] = resized_image

        canvas = canvas.permute(1, 2, 0) # Bring back to Jax
        canvas = torch2jax(canvas)

        preprocessed_images.append(canvas)
      reference_images_preprocessed.append(preprocessed_images)

    return video, mask, reference_images_preprocessed

  def prepare_masks(
      self,
      mask: torch.Tensor,
      reference_images: Optional[List[torch.Tensor]] = None,
  ):
    masks_torch = torch.Tensor(np.array(mask).transpose(0, 4, 1, 2, 3))
    mask = masks_torch
    if reference_images is None:
      # For each batch of video, we set no reference image (as one or more can
      # be passed by user)
      reference_images = [[None] for _ in range(mask.shape[0])]
    else:
      if mask.shape[0] != len(reference_images):
        raise ValueError(
            f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
        )

    if mask.shape[0] != 1:
      raise ValueError(
          "Generating with more than one video is not yet supported. This may be supported in the future."
      )

    transformer_patch_size = (
        self.transformer.config.patch_size[1]
        if self.transformer is not None
        else self.transformer_2.config.patch_size[1]
    )

    mask_list = []
    for mask_, reference_images_batch in zip(mask, reference_images):
      num_channels, num_frames, height, width = mask_.shape
      new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
      new_height = height // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
      new_width = width // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
      mask_ = mask_[0, :, :, :]
      mask_ = mask_.view(
            num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial
        )
      # TODO: should we refactor to use Jax/TF?
      mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
      mask_ = torch.nn.functional.interpolate(
            mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
        ).squeeze(0)
      num_ref_images = len(reference_images_batch)
      if num_ref_images > 0:
        mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
        mask_ = torch.cat([mask_padding, mask_], dim=1)
      mask_list.append(mask_)
    result_torch = torch.stack(mask_list)
    result_jax = jnp.array(result_torch).transpose(0, 2, 3, 4, 1)
    return result_jax

  @classmethod
  def load_transformer(
      cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None, subfolder="transformer"):
    with mesh:
      wan_transformer = create_sharded_logical_transformer(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint, subfolder=subfolder
      )
    return wan_transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)
    transformer = None
    tokenizer = None
    scheduler = None
    scheduler_state = None
    text_encoder = None
    if not vae_only:
      if load_transformer:
        with mesh:
          transformer = cls.load_transformer(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, subfolder="transformer")

      text_encoder = cls.load_text_encoder(config=config)
      tokenizer = cls.load_tokenizer(config=config)

      scheduler, scheduler_state = cls.load_scheduler(config=config)

    with mesh:
      wan_vae, vae_cache = cls.load_vae(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)

    pipeline = cls(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=wan_vae,
        vae_cache=vae_cache,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
        devices_array=devices_array,
        mesh=mesh,
        config=config,
    )

    pipeline.transformer = cls.quantize_transformer(config, pipeline.transformer, pipeline, mesh)
    return pipeline

  def check_inputs(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 480,
      width: int = 832,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      video: Optional[List[PipelineImageInput]] = None,
      mask: Optional[List[PipelineImageInput]] = None,
      reference_images: Optional[List[PipelineImageInput]] = None,
      guidance_scale_2: Optional[float] = None,
  ):
    if self.transformer is not None:
      base = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
    elif self.transformer_2 is not None:
      base = self.vae_scale_factor_spatial * self.transformer_2.config.patch_size[1]
    else:
      raise ValueError(
          "`transformer` or `transformer_2` component must be set in order to run inference with this pipeline"
      )

    if height % base != 0 or width % base != 0:
      raise ValueError(f"`height` and `width` have to be divisible by {base} but are {height} and {width}.")

    if "boundary_ratio" in self.config.get_keys() and self.config.boundary_ratio is None and guidance_scale_2 is not None:
      raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    if prompt is not None and prompt_embeds is not None:
      raise ValueError(
        f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
        " only forward one of the two."
      )
    elif negative_prompt is not None and negative_prompt_embeds is not None:
      raise ValueError(
        f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
        " only forward one of the two."
      )
    elif prompt is None and prompt_embeds is None:
      raise ValueError(
        "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
      )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    elif negative_prompt is not None and (
        not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
    ):
      raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    if video is not None:
      if mask is not None:
        if len(video) != len(mask):
          raise ValueError(
            f"Length of `video` {len(video)} and `mask` {len(mask)} do not match. Please make sure that"
            " they have the same length."
          )
      if reference_images is not None:
        is_pil_image = isinstance(reference_images, PIL.Image.Image)
        is_list_of_pil_images = isinstance(reference_images, list) and all(
          isinstance(ref_img, PIL.Image.Image) for ref_img in reference_images
        )
        is_list_of_list_of_pil_images = isinstance(reference_images, list) and all(
          isinstance(ref_img, list) and all(isinstance(ref_img_, PIL.Image.Image) for ref_img_ in ref_img)
          for ref_img in reference_images
        )
        if not (is_pil_image or is_list_of_pil_images or is_list_of_list_of_pil_images):
          raise ValueError(
            "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
            "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
          )
        if is_list_of_list_of_pil_images and len(reference_images) != 1:
          raise ValueError(
            "The pipeline only supports generating one video at a time at the moment. When passing a list "
            "of list of reference images, where the outer list corresponds to the batch size and the inner "
            "list corresponds to list of conditioning images per video, please make sure to only pass "
            "one inner list of reference images (i.e., `[[<image1>, <image2>, ...]]`"
          )
    elif mask is not None:
      raise ValueError("`mask` can only be passed if `video` is passed as well.")

  def __call__(
      self,
      video: Optional[List[PipelineImageInput]] = None,
      mask: Optional[List[PipelineImageInput]] = None,
      reference_images: Optional[List[PipelineImageInput]] = None,
      conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,

      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: jax.Array = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
      vae_only: bool = False,
  ):
    """Runs the VACE model for the given inputs.

    Args:
      video: Optional video to condition on.
      mask: Optional mask to condition on.
      reference_images: Optional reference images to condition on. Supports different formats with different value ranges.
      conditioning_scale: Conditioning scale for the VACE model (between 0 and 1).
      prompt: Prompt for text conditioning.
      negative_prompt: Negative prompt for text conditioning.
      height: Height of the video to generate.
      width: Width of the video to generate.
      num_frames: Number of frames in the video to generate.
      num_inference_steps: Number of diffusion steps to run.
      guidance_scale: Guidance scale to use when using CFG (0 to disable).
      num_videos_per_prompt: Number of videos to generate for each prompt. Currently only 1 is supported.
      max_sequence_length: Maximum sequence length of the text encoder.
      latents: Optional initial latents for the VAE.
      prompt_embeds: Optional precomputed prompt embeddings for the text encoder.
      negative_prompt_embeds: Optional precomputed negative prompt embeddings for the text encoder.
      vae_only: Whether to only run the decoder for a given latent.
    """

    self.check_inputs(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        video=video,
        mask=mask,
        reference_images=reference_images
    )
    if not vae_only:
      if num_frames % self.vae_scale_factor_temporal != 1:
        max_logging.log(
            f"`num_frames -1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
      num_frames = max(num_frames, 1)

      # 2. Define call parameters
      if prompt is not None and isinstance(prompt, str):
        prompt = [prompt]

      batch_size = len(prompt)
      if num_videos_per_prompt != 1:
        raise ValueError(
            "Generating multiple videos per prompt is not yet supported. This may be supported in the future."
        )

      prompt_embeds, negative_prompt_embeds = self.encode_prompt(
          prompt=prompt,
          negative_prompt=negative_prompt,
          max_sequence_length=max_sequence_length,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )

      transformer_dtype = self.transformer.proj_out.bias.dtype

      vace_layers = (
          self.transformer.config.vace_layers
          if self.transformer is not None
          else self.transformer_2.config.vace_layers
      )

      if isinstance(conditioning_scale, (int, float)):
        conditioning_scale = [conditioning_scale] * len(vace_layers)
      if isinstance(conditioning_scale, list):
        if len(conditioning_scale) != len(vace_layers):
          raise ValueError(
                  f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(vace_layers)}."
              )
        conditioning_scale = jnp.array(conditioning_scale)
      if isinstance(conditioning_scale, jax.Array):
        if conditioning_scale.shape[0] != len(vace_layers):
          raise ValueError(
              f"Length of `conditioning_scale` {conditioning_scale.shape[0]} does not match number of layers {len(vace_layers)}."
          )

      video, mask, reference_images = self.preprocess_conditions(
          video,
          mask,
          reference_images,
          batch_size,
          height,
          width,
          num_frames,
          dtype=jnp.float32,
      )
      num_reference_images = len(reference_images[0])
      data_sharding = NamedSharding(self.mesh, P())
      # Using global_batch_size_to_train_on so not to create more config variables
      if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
        data_sharding = NamedSharding(self.mesh, P(*self.config.data_sharding))

      conditioning_latents = self.prepare_video_latents(data_sharding=data_sharding, video=video, mask=mask, reference_images=reference_images, rngs=None)

      mask = self.prepare_masks(mask, reference_images)
      conditioning_latents = conditioning_latents.transpose(0, 4, 1, 2, 3)
      mask = mask.transpose(0, 4, 1, 2, 3)

      conditioning_latents = jnp.concatenate([conditioning_latents, mask], axis=1)
      conditioning_latents = conditioning_latents.astype(transformer_dtype)

      num_channel_latents = self.transformer.config.in_channels
      if latents is None:
        latents = self.prepare_latents(
            batch_size=batch_size,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            height=height,
            width=width,
            num_frames=num_frames + num_reference_images * self.vae_scale_factor_temporal,
            num_channels_latents=num_channel_latents,
        )

      latents = jax.device_put(latents, data_sharding)
      prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
      negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)

      conditioning_latents = jax.device_put(conditioning_latents, data_sharding)
      conditioning_scale = jax.device_put(conditioning_scale, data_sharding)

      scheduler_state = self.scheduler.set_timesteps(
          self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
      )
      if conditioning_latents.shape[2] != latents.shape[2]:
        raise ValueError(
            "The number of frames in the conditioning latents does not match"
            " the number of frames to be generated. Generation quality may be"
            f" affected.Got {conditioning_latents.shape[2]} !="
            f" {latents.shape[2]}"
        )

      graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

      p_run_inference = partial(
          run_inference,
          guidance_scale=guidance_scale,
          num_inference_steps=num_inference_steps,
          scheduler=self.scheduler,
          scheduler_state=scheduler_state,
      )

      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        latents = p_run_inference(
            graphdef=graphdef,
            sharded_state=state,
            rest_of_state=rest_of_state,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            control_hidden_states=conditioning_latents,
            control_hidden_states_scale=conditioning_scale,
        )
        latents = latents[:, :, num_reference_images:]
        latents_mean = jnp.array(self.vae.latents_mean).reshape(1, self.vae.z_dim, 1, 1, 1)
        latents_std = 1.0 / jnp.array(self.vae.latents_std).reshape(1, self.vae.z_dim, 1, 1, 1)
        latents = latents / latents_std + latents_mean
        latents = latents.astype(jnp.float32)

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      video = self.vae.decode(latents, self.vae_cache)[0]

    video = jnp.transpose(video, (0, 4, 1, 2, 3))
    video = jax.experimental.multihost_utils.process_allgather(video, tiled=True)
    video = torch.from_numpy(np.array(video.astype(dtype=jnp.float32))).to(dtype=torch.bfloat16)
    video = self.video_processor.postprocess_video(video, output_type="np")
    return video

  def prepare_video_latents(
      self,
      data_sharding: NamedSharding,
      video: torch.Tensor,
      mask: torch.Tensor,
      reference_images: Optional[List[List[torch.Tensor]]] = None,
      rngs=None,
  ) -> jax.Array:

    if reference_images is None:
      # For each batch of video, we set no re
      # ference image (as one or more can be passed by user)
      reference_images = [[None] for _ in range(video.shape[0])]
    else:
      if video.shape[0] != len(reference_images):
        raise ValueError(
            f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
        )

    if video.shape[0] != 1:
      raise ValueError(
          "Generating with more than one video is not yet supported. This may be supported in the future."
      )

    vae_dtype = self.vae.decoder.conv_in.conv.bias.dtype
    video = video.astype(dtype=vae_dtype)

    latents_mean = jnp.array(self.vae.latents_mean).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents_std = 1.0 / jnp.array(self.vae.latents_std).reshape(1, 1, 1, 1, self.vae.z_dim)

    if mask is None:
      raise NotImplementedError("b/468240950: Masks are not yet supported.")
    else:
      mask = jnp.where(mask > 0.5, 1.0, 0.0).astype(vae_dtype)
      inactive = video * (1 - mask)
      reactive = video * mask
      inactive = retrieve_latents(self.vae.encode(inactive, self.vae_cache), rngs=rngs, sample_mode="argmax")
      reactive = retrieve_latents(self.vae.encode(reactive, self.vae_cache), rngs=rngs, sample_mode="argmax")
      inactive = ((inactive.astype(jnp.float32) - latents_mean) * latents_std).astype(vae_dtype)
      reactive = ((reactive.astype(jnp.float32) - latents_mean) * latents_std).astype(vae_dtype)

      latents = jnp.concatenate([inactive, reactive], axis=-1)

    latent_list = []
    for latent, reference_images_batch in zip(latents, reference_images):
      for reference_image in reference_images_batch:
        assert reference_image.ndim == 3
        reference_image = reference_image.astype(dtype=vae_dtype)
        reference_image = jax.device_put(reference_image, data_sharding)
        reference_image = reference_image[None, None, :, :, :]  # [1, 1, H, W, C]

        reference_latent = retrieve_latents(self.vae.encode(reference_image, feat_cache=self.vae_cache), rngs=None, sample_mode="argmax")

        reference_latent = ((reference_latent.astype(jnp.float32) - latents_mean) * latents_std).astype(vae_dtype)

        reference_latent = reference_latent.squeeze(0)  # [1, H, W, C]
        reference_latent = jnp.concatenate([reference_latent, jnp.zeros_like(reference_latent)], axis=3)

        latent = jnp.concatenate([reference_latent, latent], axis=0)
      latent_list.append(latent)

    return jnp.stack(latent_list)


@partial(jax.jit, static_argnames=("do_classifier_free_guidance", "guidance_scale"))
def transformer_forward_pass(
    graphdef: nnx.graph.GraphDef,
    sharded_state: nnx.graph.GraphState,
    rest_of_state: Any,
    latents: jax.Array,
    timestep: jax.Array,
    prompt_embeds: jax.Array,
    control_hidden_states: jax.Array,
    control_hidden_states_scale: jax.Array,
    do_classifier_free_guidance: bool,
    guidance_scale: float,
):
  """Performs a forward pass on the transformer."""
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  noise_pred = wan_transformer(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds,
      control_hidden_states=control_hidden_states,
      control_hidden_states_scale=control_hidden_states_scale,
  )
  if do_classifier_free_guidance:
    bsz = latents.shape[0] // 2
    noise_uncond = noise_pred[bsz:]
    noise_pred = noise_pred[:bsz]
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
    latents = latents[:bsz]

  return noise_pred, latents


def run_inference(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,

    control_hidden_states,
    control_hidden_states_scale,
):
  do_classifier_free_guidance = guidance_scale > 1.0
  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
    control_hidden_states = jnp.concatenate([control_hidden_states] * 2)

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    if do_classifier_free_guidance:
      latents = jnp.concatenate([latents] * 2)
    timestep = jnp.broadcast_to(t, latents.shape[0])

    noise_pred, latents = transformer_forward_pass(
        graphdef,
        sharded_state,
        rest_of_state,
        latents,
        timestep,
        prompt_embeds,
        control_hidden_states=control_hidden_states,
        control_hidden_states_scale=control_hidden_states_scale,

        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
    )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents
