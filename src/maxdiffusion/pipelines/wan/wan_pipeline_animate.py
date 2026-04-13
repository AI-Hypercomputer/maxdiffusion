# Copyright 2026 Google LLC
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

"""JAX/Flax pipeline for character animation using Wan-Animate.

Ported from:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_animate.py

The pipeline supports two modes:
  - "animate": Generate a video of the reference character mimicking motion from
    pose/face videos.
  - "replace": Replace a character in a background video with the reference
    character, using pose/face videos for motion control.

Inference runs in segments of `segment_frame_length` frames (default 77), which
are stitched together with overlap conditioning from the previous segment.
"""

from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import PIL
import torch
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P
from maxdiffusion import max_logging
from maxdiffusion.image_processor import PipelineImageInput, VaeImageProcessor
from maxdiffusion.max_utils import device_put_replicated, get_flash_block_sizes, get_precision
from maxdiffusion.video_processor import VideoProcessor

from ...models.wan.transformers.transformer_wan_animate import WanAnimateTransformer3DModel
from ...models.wan.wan_utils import load_wan_animate_transformer
from ...pyconfig import HyperParameters
from .wan_pipeline import WanPipeline, cast_with_exclusion


def create_sharded_animate_transformer(
    devices_array: np.ndarray,
    mesh,
    rngs: nnx.Rngs,
    config: HyperParameters,
    restored_checkpoint=None,
    subfolder: str = "transformer",
) -> WanAnimateTransformer3DModel:
  """Creates a sharded WanAnimateTransformer3DModel on device.

  Follows the same pattern as create_sharded_logical_transformer in
  wan_pipeline.py but uses WanAnimateTransformer3DModel and the
  animate-specific weight loader.
  """

  def _create_model(rngs: nnx.Rngs, wan_config: dict):
    return WanAnimateTransformer3DModel(**wan_config, rngs=rngs)

  # 1. Load config.
  if restored_checkpoint:
    wan_config = restored_checkpoint["wan_config"]
  else:
    wan_config = WanAnimateTransformer3DModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)

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
  wan_config["mask_padding_tokens"] = config.mask_padding_tokens
  wan_config["scan_layers"] = config.scan_layers
  wan_config["enable_jax_named_scopes"] = config.enable_jax_named_scopes

  # 2. eval_shape – creates the model structure without allocating HBM.
  p_model_factory = partial(_create_model, wan_config=wan_config)
  wan_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_transformer, nnx.Param, ...)

  # 3. Retrieve logical-to-mesh sharding mappings.
  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  # 4. Load and shard pretrained weights.
  if restored_checkpoint:
    if "params" in restored_checkpoint["wan_state"]:
      params = restored_checkpoint["wan_state"]["params"]
    else:
      params = restored_checkpoint["wan_state"]
  else:
    params = load_wan_animate_transformer(
        config.wan_transformer_pretrained_model_name_or_path,
        params,
        "cpu",
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


# ---------------------------------------------------------------------------
# JIT-compiled transformer forward pass
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("motion_encode_batch_size",))
def animate_transformer_forward_pass(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.ndarray,
    reference_latents: jnp.ndarray,
    pose_latents: jnp.ndarray,
    face_video_segment: jnp.ndarray,
    timestep: jnp.ndarray,
    encoder_hidden_states: jnp.ndarray,
    encoder_hidden_states_image: jnp.ndarray,
    motion_encode_batch_size: Optional[int] = None,
) -> jnp.ndarray:
  """Single denoising step for WanAnimate.

  Args:
    latents: Noisy latents, shape (B, T_lat+1, H_lat, W_lat, z_dim), channel-last.
    reference_latents: Reference image + prev-seg conditioning,
      shape (B, T_lat+1, H_lat, W_lat, z_dim+4), channel-last.
    pose_latents: VAE-encoded pose video, shape (B, T_lat, H_lat, W_lat, z_dim),
      channel-last.
    face_video_segment: Raw face video pixels,
      shape (B, 3, T_segment, face_size, face_size), channel-first.
    encoder_hidden_states: Text embeddings.
    encoder_hidden_states_image: CLIP image embeddings.

  Returns:
    noise_pred: Predicted noise, shape (B, T_lat+1, H_lat, W_lat, z_dim),
      channel-last.
  """
  # Build the full input: cat noisy latents and reference on the channel dim.
  # latents:           (B, T+1, H, W, z_dim)
  # reference_latents: (B, T+1, H, W, z_dim+4)
  # → (B, T+1, H, W, 2*z_dim+4 = 36)
  latent_model_input = jnp.concatenate([latents, reference_latents], axis=-1)
  # Transpose to channel-first for the transformer: (B, 36, T+1, H, W)
  latent_model_input = jnp.transpose(latent_model_input, (0, 4, 1, 2, 3)).astype(encoder_hidden_states.dtype)

  # Pose latents channel-first: (B, z_dim, T_lat, H_lat, W_lat)
  pose_latents_cf = jnp.transpose(pose_latents, (0, 4, 1, 2, 3)).astype(encoder_hidden_states.dtype)

  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  output = wan_transformer(
      hidden_states=latent_model_input,
      timestep=timestep,
      encoder_hidden_states=encoder_hidden_states,
      encoder_hidden_states_image=encoder_hidden_states_image,
      pose_hidden_states=pose_latents_cf,
      face_pixel_values=face_video_segment,
      motion_encode_batch_size=motion_encode_batch_size,
      return_dict=False,
  )

  # Transpose back to channel-last: (B, T+1, H, W, z_dim)
  noise_pred = jnp.transpose(output[0], (0, 2, 3, 4, 1))
  return noise_pred


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class WanAnimatePipeline(WanPipeline):
  """JAX/Flax pipeline for Wan-Animate character animation.

  Supports two modes:
    - "animate": Animate the reference character using pose and face videos.
    - "replace": Replace a character in a background video using a mask.

  Inference is performed in temporal segments to handle arbitrary video lengths.
  Each segment denoises `segment_frame_length` frames, with overlap conditioning
  from the last few frames of the previous segment.

  Args:
    config: HyperParameters configuration.
    transformer: WanAnimateTransformer3DModel instance (may be None for
      VAE-only mode).
    **kwargs: Passed to WanPipeline.__init__ (tokenizer, text_encoder, vae, etc.)
  """

  def __init__(
      self,
      config: HyperParameters,
      transformer: Optional[WanAnimateTransformer3DModel],
      **kwargs,
  ):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer
    spatial_patch_size = self.transformer.config.patch_size[-2:] if self.transformer is not None else (2, 2)
    self.ref_image_processor = VaeImageProcessor(
        vae_scale_factor=self.vae_scale_factor_spatial,
        spatial_patch_size=spatial_patch_size,
        resample="bilinear",
        resize_mode="fill",
        fill_color=0,
    )
    self.video_processor_for_mask = VideoProcessor(
        vae_scale_factor=self.vae_scale_factor_spatial,
        do_normalize=False,
        do_convert_grayscale=True,
    )

  @classmethod
  def _needs_image_encoder(cls, config: HyperParameters, i2v: bool = False) -> bool:
    return True

  @classmethod
  def load_animate_transformer(
      cls,
      devices_array: np.ndarray,
      mesh,
      rngs: nnx.Rngs,
      config: HyperParameters,
      restored_checkpoint=None,
      subfolder: str = "transformer",
  ) -> WanAnimateTransformer3DModel:
    with mesh:
      return create_sharded_animate_transformer(
          devices_array=devices_array,
          mesh=mesh,
          rngs=rngs,
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder=subfolder,
      )

  @classmethod
  def _load_and_init(
      cls,
      config: HyperParameters,
      restored_checkpoint=None,
      vae_only: bool = False,
      load_transformer: bool = True,
  ) -> Tuple["WanAnimatePipeline", Optional[WanAnimateTransformer3DModel]]:
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only and load_transformer:
      transformer = cls.load_animate_transformer(
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          rngs=common_components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder="transformer",
      )
    pipeline = cls(
        config=config,
        transformer=transformer,
        tokenizer=common_components["tokenizer"],
        text_encoder=common_components["text_encoder"],
        image_processor=common_components["image_processor"],
        image_encoder=common_components["image_encoder"],
        vae=common_components["vae"],
        vae_cache=common_components["vae_cache"],
        scheduler=common_components["scheduler"],
        scheduler_state=common_components["scheduler_state"],
        devices_array=common_components["devices_array"],
        mesh=common_components["mesh"],
    )
    return pipeline, transformer

  @classmethod
  def from_pretrained(
      cls,
      config: HyperParameters,
      vae_only: bool = False,
      load_transformer: bool = True,
  ) -> "WanAnimatePipeline":
    pipeline, transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    pipeline.transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(
      cls,
      config: HyperParameters,
      restored_checkpoint=None,
      vae_only: bool = False,
      load_transformer: bool = True,
  ) -> "WanAnimatePipeline":
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  # ------------------------------------------------------------------
  # Abstract method implementation
  # ------------------------------------------------------------------

  def _get_num_channel_latents(self) -> int:
    return self.vae.z_dim

  # ------------------------------------------------------------------
  # Video utilities
  # ------------------------------------------------------------------

  def check_inputs(
      self,
      prompt,
      negative_prompt,
      image,
      pose_video,
      face_video,
      background_video,
      mask_video,
      height,
      width,
      prompt_embeds=None,
      negative_prompt_embeds=None,
      image_embeds=None,
      mode=None,
      prev_segment_conditioning_frames=None,
  ):
    """Validate user-facing pipeline inputs with Diffusers-compatible checks."""
    supported_image_types = (torch.Tensor, PIL.Image.Image, np.ndarray, jnp.ndarray)

    if image is not None and image_embeds is not None:
      raise ValueError(
          f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
          " only forward one of the two."
      )
    if image is None and image_embeds is None:
      raise ValueError("Provide either `image` or `image_embeds`. Cannot leave both undefined.")
    if image is not None and not isinstance(image, supported_image_types):
      raise ValueError(
          f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, or `jnp.ndarray` but is {type(image)}"
      )
    if pose_video is None:
      raise ValueError("Provide `pose_video`. Cannot leave `pose_video` undefined.")
    if face_video is None:
      raise ValueError("Provide `face_video`. Cannot leave `face_video` undefined.")
    if not isinstance(pose_video, list) or not isinstance(face_video, list):
      raise ValueError("`pose_video` and `face_video` must be lists of PIL images.")
    if len(pose_video) == 0 or len(face_video) == 0:
      raise ValueError("`pose_video` and `face_video` must contain at least one frame.")
    if mode == "replace" and (background_video is None or mask_video is None):
      raise ValueError(
          "Provide `background_video` and `mask_video`. Cannot leave both `background_video` and `mask_video`"
          " undefined when mode is `replace`."
      )
    if mode == "replace" and (not isinstance(background_video, list) or not isinstance(mask_video, list)):
      raise ValueError("`background_video` and `mask_video` must be lists of PIL images when mode is `replace`.")
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")
    if prompt is not None and prompt_embeds is not None:
      raise ValueError(
          f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
          " only forward one of the two."
      )
    if negative_prompt is not None and negative_prompt_embeds is not None:
      raise ValueError(
          f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
          " only forward one of the two."
      )
    if prompt is None and prompt_embeds is None:
      raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")
    if prompt is not None and not isinstance(prompt, (str, list)):
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
      raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
    if mode is not None and (not isinstance(mode, str) or mode not in ("animate", "replace")):
      raise ValueError(
          f"`mode` has to be of type `str` and in ('animate', 'replace') but its type is {type(mode)} and value is {mode}"
      )
    if prev_segment_conditioning_frames is not None and (
        not isinstance(prev_segment_conditioning_frames, int) or prev_segment_conditioning_frames not in (1, 5)
    ):
      raise ValueError(
          f"`prev_segment_conditioning_frames` has to be of type `int` and 1 or 5 but its type is"
          f" {type(prev_segment_conditioning_frames)} and value is {prev_segment_conditioning_frames}"
      )

  @staticmethod
  def pad_video_frames(frames: list, num_target_frames: int) -> list:
    """Pad *frames* to *num_target_frames* using a reflect-like strategy.

    Example: pad_video_frames([1,2,3,4,5], 10) → [1,2,3,4,5,4,3,2,1,2]
    """
    idx = 0
    flip = False
    target_frames = []
    while len(target_frames) < num_target_frames:
      target_frames.append(deepcopy(frames[idx]))
      if flip:
        idx -= 1
      else:
        idx += 1
      if idx == 0 or idx == len(frames) - 1:
        flip = not flip
    return target_frames

  # ------------------------------------------------------------------
  # I2V mask helpers
  # ------------------------------------------------------------------

  def get_i2v_mask(
      self,
      batch_size: int,
      latent_t: int,
      latent_h: int,
      latent_w: int,
      mask_len: int = 1,
      mask_pixel_values: Optional[jnp.ndarray] = None,
      dtype: jnp.dtype = jnp.float32,
  ) -> jnp.ndarray:
    """Construct an I2V conditioning mask in channel-last format.

    A mask value of 1 means "this frame is known/conditioned" and 0 means
    "this frame is freely generated".

    Args:
      latent_t: Number of latent temporal frames.
      mask_pixel_values: Optional pre-computed mask at pixel temporal resolution
        but latent spatial resolution, shape (B, 1, T_pixel, H_lat, W_lat).
        T_pixel = (latent_t - 1) * vae_scale_factor_temporal + 1.
      mask_len: Number of leading frames to force to 1 (known).

    Returns:
      Mask array of shape (B, latent_t, H_lat, W_lat, vae_scale_factor_temporal).
    """
    vae_scale = self.vae_scale_factor_temporal
    pixel_frames = (latent_t - 1) * vae_scale + 1

    if mask_pixel_values is None:
      mask_lat_size = jnp.zeros((batch_size, 1, pixel_frames, latent_h, latent_w), dtype=dtype)
    else:
      mask_lat_size = mask_pixel_values.astype(dtype)

    # Set the first mask_len pixel frames to 1 (conditioned).
    mask_lat_size = mask_lat_size.at[:, :, :mask_len, :, :].set(1.0)

    # Repeat the first frame vae_scale times so total frames = latent_t * vae_scale.
    first_frame = mask_lat_size[:, :, 0:1, :, :]  # (B, 1, 1, H, W)
    first_frame = jnp.repeat(first_frame, vae_scale, axis=2)  # (B, 1, vae_scale, H, W)
    mask_lat_size = jnp.concatenate([first_frame, mask_lat_size[:, :, 1:, :, :]], axis=2)
    # (B, 1, latent_t*vae_scale, H, W)

    # Reshape: (B, 1, latent_t*vae_scale, H, W) → (B, latent_t, vae_scale, H, W)
    mask_lat_size = mask_lat_size.reshape(batch_size, latent_t, vae_scale, latent_h, latent_w)
    # Transpose to (B, vae_scale, latent_t, H, W) then to (B, latent_t, H, W, vae_scale).
    mask_lat_size = jnp.transpose(mask_lat_size, (0, 2, 1, 3, 4))  # (B, vae_scale, T, H, W)
    mask_lat_size = jnp.transpose(mask_lat_size, (0, 2, 3, 4, 1))  # (B, T, H, W, vae_scale)
    return mask_lat_size

  # ------------------------------------------------------------------
  # Latent preparation helpers
  # ------------------------------------------------------------------

  def _encode_video_to_latents(
      self,
      video: jnp.ndarray,
      dtype: jnp.dtype,
  ) -> jnp.ndarray:
    """Encode a video tensor and normalize the latents.

    Args:
      video: (B, C, T, H, W) channel-first, values in [-1, 1].

    Returns:
      Normalized latents: (B, T_lat, H_lat, W_lat, z_dim) channel-last.
    """
    vae_dtype = getattr(self.vae, "dtype", jnp.float32)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      encoded = self.vae.encode(video.astype(vae_dtype), self.vae_cache)[0].mode()
    # Normalize
    mean = jnp.array(self.vae.latents_mean).reshape(1, 1, 1, 1, self.vae.z_dim)
    std = jnp.array(self.vae.latents_std).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents = (encoded - mean) / std
    return latents.astype(dtype)

  def prepare_reference_image_latents(
      self,
      image: jnp.ndarray,
      batch_size: int,
      dtype: jnp.dtype,
  ) -> jnp.ndarray:
    """Encode the reference character image and prepend an I2V mask.

    Args:
      image: (B, C, H, W) or (B, C, 1, H, W) channel-first, values in [-1, 1].

    Returns:
      (B, 1, H_lat, W_lat, z_dim + vae_scale_factor_temporal) channel-last.
    """
    if image.ndim == 4:
      image = image[:, :, jnp.newaxis, :, :]  # (B, C, 1, H, W)

    # Encode the single reference frame.
    ref_latents = self._encode_video_to_latents(image, dtype)  # (B, 1, H_lat, W_lat, z_dim)

    if ref_latents.shape[0] == 1 and batch_size > 1:
      ref_latents = jnp.broadcast_to(ref_latents, (batch_size,) + ref_latents.shape[1:])

    latent_h = ref_latents.shape[2]
    latent_w = ref_latents.shape[3]

    # Mask for the single reference frame — mark it as fully conditioned.
    ref_mask = self.get_i2v_mask(batch_size, 1, latent_h, latent_w, mask_len=1, dtype=dtype)
    # (B, 1, H_lat, W_lat, vae_scale)

    return jnp.concatenate([ref_mask, ref_latents], axis=-1)

  def _resize_mask_to_latent_spatial(
      self,
      mask: jnp.ndarray,
      latent_h: int,
      latent_w: int,
  ) -> jnp.ndarray:
    """Resize a mask from pixel spatial resolution to latent spatial resolution.

    Args:
      mask: (B, 1, T, H, W) channel-first.

    Returns:
      (B, 1, T, H_lat, W_lat) channel-first.
    """
    B, C, T, H, W = mask.shape
    if H == latent_h and W == latent_w:
      return mask
    # Match torch.nn.functional.interpolate(..., mode="nearest") exactly.
    h_indices = jnp.floor(jnp.arange(latent_h) * (H / latent_h)).astype(jnp.int32)
    w_indices = jnp.floor(jnp.arange(latent_w) * (W / latent_w)).astype(jnp.int32)
    mask = jnp.take(mask, h_indices, axis=3)
    return jnp.take(mask, w_indices, axis=4)

  def prepare_prev_segment_cond_latents(
      self,
      prev_segment_cond_video: Optional[jnp.ndarray],
      background_video: Optional[jnp.ndarray],
      mask_video: Optional[jnp.ndarray],
      batch_size: int,
      segment_frame_length: int,
      start_frame: int,
      height: int,
      width: int,
      prev_segment_cond_frames: int,
      task: str,
      dtype: jnp.dtype,
  ) -> jnp.ndarray:
    """Prepare latent conditioning from the previous segment.

    Args:
      prev_segment_cond_video: Last N decoded frames from the previous segment,
        shape (B, C, N, H, W) channel-first in [-1, 1], or None for segment 0.
      background_video: Background video segment for replace mode,
        shape (B, C, T_seg, H, W).
      mask_video: Mask video segment for replace mode (white=generate, black=preserve),
        shape (B, 1, T_seg, H, W).
      start_frame: Pixel-space start frame of the current segment.
      task: "animate" or "replace".

    Returns:
      (B, T_lat, H_lat, W_lat, z_dim + vae_scale_factor_temporal) channel-last.
    """
    vae_dtype = getattr(self.vae, "dtype", jnp.float32)
    latent_h = height // self.vae_scale_factor_spatial
    latent_w = width // self.vae_scale_factor_spatial
    num_latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1

    if prev_segment_cond_video is None:
      if task == "replace" and background_video is not None:
        prev_segment_cond_video = background_video[:, :, :prev_segment_cond_frames]
      else:
        prev_segment_cond_video = jnp.zeros((batch_size, 3, prev_segment_cond_frames, height, width), dtype=vae_dtype)

    # Build full-length cond video (prev frames + remainder).
    if task == "replace" and background_video is not None:
      remaining = background_video[:, :, prev_segment_cond_frames:]
    else:
      remaining_frames = segment_frame_length - prev_segment_cond_frames
      remaining = jnp.zeros((batch_size, 3, remaining_frames, height, width), dtype=vae_dtype)

    full_cond_video = jnp.concatenate([prev_segment_cond_video.astype(vae_dtype), remaining], axis=2)  # (B, C, T_seg, H, W)

    cond_latents = self._encode_video_to_latents(full_cond_video, dtype)
    # (B, T_lat, H_lat, W_lat, z_dim)

    # Build I2V mask.
    if task == "replace" and mask_video is not None:
      # Invert mask: white (1.0, generate) → 0.0, black (0.0, preserve) → 1.0.
      # In the I2V mask convention, 1 = known/conditioned, 0 = freely generated.
      inverted_mask = 1.0 - mask_video
      mask_pixel_values = self._resize_mask_to_latent_spatial(inverted_mask, latent_h, latent_w)
      # mask_pixel_values: (B, 1, T_seg, H_lat, W_lat) – pixel temporal resolution
    else:
      mask_pixel_values = None

    cond_mask = self.get_i2v_mask(
        batch_size,
        num_latent_frames,
        latent_h,
        latent_w,
        mask_len=prev_segment_cond_frames if start_frame > 0 else 0,
        mask_pixel_values=mask_pixel_values,
        dtype=dtype,
    )
    # (B, T_lat, H_lat, W_lat, vae_scale)

    return jnp.concatenate([cond_mask, cond_latents], axis=-1)

  def prepare_pose_latents(
      self,
      pose_video: jnp.ndarray,
      batch_size: int,
      dtype: jnp.dtype,
  ) -> jnp.ndarray:
    """Encode the pose video segment to latents.

    Args:
      pose_video: (B, C, T_seg, H, W) channel-first, values in [-1, 1].

    Returns:
      (B, T_lat, H_lat, W_lat, z_dim) channel-last.
    """
    pose_latents = self._encode_video_to_latents(pose_video, dtype)
    if pose_latents.shape[0] == 1 and batch_size > 1:
      pose_latents = jnp.broadcast_to(pose_latents, (batch_size,) + pose_latents.shape[1:])
    return pose_latents

  def prepare_segment_latents(
      self,
      batch_size: int,
      height: int,
      width: int,
      segment_frame_length: int,
      dtype: jnp.dtype,
      rng: jax.Array,
      latents: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Sample noisy latents for a denoising segment.

    The +1 accounts for the reference frame slot at index 0.

    Returns:
      (B, T_lat+1, H_lat, W_lat, z_dim) channel-last.
    """
    num_latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
    latent_h = height // self.vae_scale_factor_spatial
    latent_w = width // self.vae_scale_factor_spatial
    shape = (batch_size, num_latent_frames + 1, latent_h, latent_w, self.vae.z_dim)
    if latents is not None:
      latents = jnp.asarray(latents)
      if latents.shape != shape:
        raise ValueError(f"Unexpected latents shape {latents.shape}; expected {shape}.")
      return latents.astype(dtype)
    return jax.random.normal(rng, shape=shape, dtype=jnp.float32).astype(dtype)

  def _decode_segment_to_pixels(self, latents_cl: jnp.ndarray) -> jnp.ndarray:
    """Decode latents and return raw pixel-space frames for re-encoding.

    Args:
      latents_cl: (B, T_lat, H_lat, W_lat, z_dim) channel-last, normalised.

    Returns:
      (B, C, T, H, W) channel-first, values in [-1, 1] (VAE output range).
    """
    latents_cf = jnp.transpose(latents_cl, (0, 4, 1, 2, 3))  # (B, z_dim, T, H, W)
    latents_cf = self._denormalize_latents(latents_cf)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      video_cl = self.vae.decode(latents_cf, self.vae_cache)[0]  # (B, T, H, W, C)
    return jnp.transpose(video_cl, (0, 4, 1, 2, 3))  # (B, C, T, H, W)

  # ------------------------------------------------------------------
  # Main inference
  # ------------------------------------------------------------------

  def __call__(
      self,
      image: PipelineImageInput,
      pose_video: List[PIL.Image.Image],
      face_video: List[PIL.Image.Image],
      background_video: Optional[List[PIL.Image.Image]] = None,
      mask_video: Optional[List[PIL.Image.Image]] = None,
      prompt: Optional[Union[str, List[str]]] = None,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: Optional[int] = None,
      width: Optional[int] = None,
      segment_frame_length: int = 77,
      num_inference_steps: int = 20,
      mode: str = "animate",
      prev_segment_conditioning_frames: int = 1,
      motion_encode_batch_size: Optional[int] = None,
      guidance_scale: float = 1.0,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 512,
      latents: Optional[jnp.ndarray] = None,
      prompt_embeds: Optional[jnp.ndarray] = None,
      negative_prompt_embeds: Optional[jnp.ndarray] = None,
      image_embeds: Optional[jnp.ndarray] = None,
      output_type: Optional[str] = "np",
      rng: Optional[jax.Array] = None,
  ):
    """Run the Wan-Animate inference pipeline.

    Args:
      image: Reference character image (PIL.Image or compatible).
      pose_video: List of PIL frames representing the pose video.
      face_video: List of PIL frames representing the face video.
      background_video: (replace mode) Background video frames.
      mask_video: (replace mode) Mask frames. White=generate, black=preserve.
      prompt: Text prompt(s).
      negative_prompt: Negative prompt(s) for CFG (only used when guidance_scale > 1).
      height: Output video height in pixels.
      width: Output video width in pixels.
      segment_frame_length: Number of frames per denoising segment. Should satisfy
        (segment_frame_length - 1) % vae_scale_factor_temporal == 0.
      num_inference_steps: Denoising steps per segment.
      mode: "animate" or "replace".
      prev_segment_conditioning_frames: Overlap frames between segments (1 or 5).
      motion_encode_batch_size: Batch size for the motion encoder. Defaults to
        the transformer's configured value.
      guidance_scale: CFG scale (set > 1 to enable classifier-free guidance).
      num_videos_per_prompt: Number of videos to generate per prompt.
      rng: Optional JAX PRNG key.

    Returns:
      If output_type == "np": numpy array of shape (B, T, H, W, C) in [0, 1].
      If output_type == "latent": raw latents from the final segment.
    """
    height = height or self.config.height
    width = width or self.config.width

    self.check_inputs(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        pose_video=pose_video,
        face_video=face_video,
        background_video=background_video,
        mask_video=mask_video,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image_embeds=image_embeds,
        mode=mode,
        prev_segment_conditioning_frames=prev_segment_conditioning_frames,
    )

    # Ensure segment_frame_length satisfies the VAE temporal constraint.
    if segment_frame_length % self.vae_scale_factor_temporal != 1:
      max_logging.log(
          f"`segment_frame_length - 1` must be divisible by {self.vae_scale_factor_temporal}. "
          f"Rounding {segment_frame_length}."
      )
      segment_frame_length = segment_frame_length // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    segment_frame_length = max(segment_frame_length, 1)

    do_classifier_free_guidance = guidance_scale > 1.0

    # Determine batch size.
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None:
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]
    effective_batch_size = batch_size * num_videos_per_prompt

    # Segment arithmetic.
    cond_video_frames = len(pose_video)
    effective_segment_length = segment_frame_length - prev_segment_conditioning_frames
    last_frames = (cond_video_frames - prev_segment_conditioning_frames) % effective_segment_length
    num_padding_frames = 0 if last_frames == 0 else effective_segment_length - last_frames
    num_target_frames = cond_video_frames + num_padding_frames
    num_segments = num_target_frames // effective_segment_length

    # ---- 1. Encode prompts ----
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    transformer_dtype = self.config.activations_dtype
    latent_dtype = jnp.float32
    prompt_embeds = prompt_embeds.astype(transformer_dtype)
    if negative_prompt_embeds is not None:
      negative_prompt_embeds = negative_prompt_embeds.astype(transformer_dtype)

    # ---- 2. Encode reference image with CLIP ----
    if image_embeds is None:
      image_embeds = self.encode_image(image, num_videos_per_prompt=effective_batch_size)
    image_embeds = image_embeds.astype(transformer_dtype)

    # ---- 3. VAE-encode reference image ----
    # Use VaeImageProcessor with resize_mode="fill" so the character is letterboxed instead of cropped.
    image_tensor = self.ref_image_processor.preprocess(image, height=height, width=width)
    image_tensor = jnp.array(image_tensor.cpu().numpy())
    if image_tensor.ndim == 3:
      image_tensor = image_tensor[None]  # (1, C, H, W)
    if effective_batch_size > 1 and image_tensor.shape[0] == 1:
      image_tensor = jnp.broadcast_to(image_tensor, (effective_batch_size,) + image_tensor.shape[1:])

    reference_image_latents = self.prepare_reference_image_latents(
        image_tensor, effective_batch_size, transformer_dtype
    )  # (B, 1, H_lat, W_lat, z_dim+vae_scale)

    # ---- 4. Preprocess conditioning videos ----
    pose_video = self.pad_video_frames(pose_video, num_target_frames)
    face_video = self.pad_video_frames(face_video, num_target_frames)

    pose_video_tensor = self.video_processor.preprocess_video(pose_video, height=height, width=width)
    pose_video_tensor = jnp.array(pose_video_tensor.cpu().numpy())  # (1, C, T, H, W)

    face_size = self.transformer.motion_encoder.size
    face_video_tensor = self.video_processor.preprocess_video(face_video, height=face_size, width=face_size)
    face_video_tensor = jnp.array(face_video_tensor.cpu().numpy())  # (1, C, T, face_size, face_size)

    background_video_tensor = None
    mask_video_tensor = None
    if mode == "replace":
      if background_video is None or mask_video is None:
        raise ValueError("`background_video` and `mask_video` are required for replace mode.")
      background_video = self.pad_video_frames(background_video, num_target_frames)
      mask_video = self.pad_video_frames(mask_video, num_target_frames)

      background_video_tensor = self.video_processor.preprocess_video(background_video, height=height, width=width)
      background_video_tensor = jnp.array(background_video_tensor.cpu().numpy())
      mask_video_tensor = self.video_processor_for_mask.preprocess_video(mask_video, height=height, width=width)
      mask_video_tensor = jnp.array(mask_video_tensor.cpu().numpy())

    if rng is None:
      rng = jax.random.key(self.config.seed)

    # ---- 5. Device placement ----
    data_sharding = NamedSharding(self.mesh, P())
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    if negative_prompt_embeds is not None:
      negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)
    image_embeds = jax.device_put(image_embeds, data_sharding)

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    # ---- 6. Segment denoising loop ----
    start = 0
    end = segment_frame_length
    all_out_frames_cf = []  # list of (B, C, T, H, W) channel-first in [-1,1]
    out_frames_cf = None  # decoded output from previous segment

    for _seg in range(num_segments):
      rng, latents_rng = jax.random.split(rng)

      seg_latents = self.prepare_segment_latents(
          effective_batch_size,
          height,
          width,
          segment_frame_length,
          latent_dtype,
          latents_rng,
          latents=latents if start == 0 else None,
      )  # (B, T_lat+1, H_lat, W_lat, z_dim)

      # Extract segment slices.
      pose_seg = pose_video_tensor[:, :, start:end]  # (1, C, T_seg, H, W)
      face_seg = face_video_tensor[:, :, start:end]  # (1, C, T_seg, face_size, face_size)

      if effective_batch_size > 1:
        face_seg = jnp.broadcast_to(face_seg, (effective_batch_size,) + face_seg.shape[1:])
      face_seg = face_seg.astype(transformer_dtype)

      # Previous segment conditioning frames (pixel space, channel-first, [-1,1]).
      prev_cond_video = None
      if start > 0 and out_frames_cf is not None:
        prev_cond_video = out_frames_cf[:, :, -prev_segment_conditioning_frames:]

      # Encode pose and prepare prev-seg conditioning.
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        pose_latents = self.prepare_pose_latents(pose_seg, effective_batch_size, transformer_dtype)

      bg_seg = None
      mask_seg = None
      if mode == "replace":
        bg_seg = background_video_tensor[:, :, start:end]
        mask_seg = mask_video_tensor[:, :, start:end]
        if effective_batch_size > 1:
          bg_seg = jnp.broadcast_to(bg_seg, (effective_batch_size,) + bg_seg.shape[1:])
          mask_seg = jnp.broadcast_to(mask_seg, (effective_batch_size,) + mask_seg.shape[1:])

      prev_seg_cond_latents = self.prepare_prev_segment_cond_latents(
          prev_segment_cond_video=prev_cond_video,
          background_video=bg_seg,
          mask_video=mask_seg,
          batch_size=effective_batch_size,
          segment_frame_length=segment_frame_length,
          start_frame=start,
          height=height,
          width=width,
          prev_segment_cond_frames=prev_segment_conditioning_frames,
          task=mode,
          dtype=transformer_dtype,
      )  # (B, T_lat, H_lat, W_lat, z_dim+vae_scale)

      # Combine reference (1 frame) + prev-seg conditioning (T_lat frames).
      reference_latents = jnp.concatenate(
          [reference_image_latents, prev_seg_cond_latents], axis=1
      )  # (B, T_lat+1, H_lat, W_lat, z_dim+vae_scale)

      # Set up scheduler timesteps for this segment.
      scheduler_state = self.scheduler.set_timesteps(
          self.scheduler_state, num_inference_steps=num_inference_steps, shape=seg_latents.shape
      )

      seg_latents = jax.device_put(seg_latents, data_sharding)
      reference_latents = jax.device_put(reference_latents, data_sharding)
      pose_latents = jax.device_put(pose_latents, data_sharding)
      face_seg = jax.device_put(face_seg, data_sharding)

      # Denoising loop.
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        for step in range(num_inference_steps):
          t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
          timestep = jnp.broadcast_to(t, (seg_latents.shape[0],))

          noise_pred = animate_transformer_forward_pass(
              graphdef,
              state,
              rest_of_state,
              seg_latents,
              reference_latents,
              pose_latents,
              face_seg,
              timestep,
              prompt_embeds,
              image_embeds,
              motion_encode_batch_size=motion_encode_batch_size,
          )

          if do_classifier_free_guidance:
            # Blank face pixels (all -1) for the unconditional pass.
            face_seg_uncond = face_seg * 0 - 1
            noise_uncond = animate_transformer_forward_pass(
                graphdef,
                state,
                rest_of_state,
                seg_latents,
                reference_latents,
                pose_latents,
                face_seg_uncond,
                timestep,
                negative_prompt_embeds,
                image_embeds,
                motion_encode_batch_size=motion_encode_batch_size,
            )
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

          noise_pred = noise_pred.astype(seg_latents.dtype)
          seg_latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, seg_latents, return_dict=False)

        # Decode this segment (skip reference frame at index 0).
        out_frames_cf = self._decode_segment_to_pixels(seg_latents[:, 1:, :, :, :])
        # (B, C, T_pixel, H, W) channel-first in [-1, 1]

      if start > 0:
        # Drop overlap frames used for conditioning.
        out_frames_cf_trimmed = out_frames_cf[:, :, prev_segment_conditioning_frames:]
      else:
        out_frames_cf_trimmed = out_frames_cf

      all_out_frames_cf.append(out_frames_cf_trimmed)

      start += effective_segment_length
      end += effective_segment_length

    # ---- 7. Assemble output ----
    # Concat along the temporal dimension and trim to the original video length.
    video_cf = jnp.concatenate(all_out_frames_cf, axis=2)[:, :, :cond_video_frames]
    # (B, C, T, H, W) channel-first in [-1, 1]

    if output_type == "latent":
      return seg_latents

    # Postprocess to [0, 1] numpy.
    video_torch = torch.from_numpy(np.array(video_cf.astype(jnp.float32))).to(torch.bfloat16)
    return self.video_processor.postprocess_video(video_torch, output_type="np")
