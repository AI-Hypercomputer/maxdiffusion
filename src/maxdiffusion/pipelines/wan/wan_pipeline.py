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

from abc import abstractmethod
from typing import List, Union, Optional, Tuple
from functools import partial
from maxdiffusion.image_processor import PipelineImageInput
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
from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated
from ...models.wan.wan_utils import load_wan_transformer, load_wan_vae
from ...models.wan.transformers.transformer_wan import WanModel
from ...models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.video_processor import VideoProcessor
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState
from transformers import AutoTokenizer, UMT5EncoderModel
from maxdiffusion.utils.import_utils import is_ftfy_available
from maxdiffusion.maxdiffusion_utils import get_dummy_wan_inputs
import html
import re
import torch
import qwix
from transformers import CLIPImageProcessor
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModel
import PIL


def cast_with_exclusion(path, x, dtype_to_cast):
  """
  Casts arrays to dtype_to_cast, but keeps params from any 'norm' layer in float32.
  """

  exclusion_keywords = [
      "norm",  # For all LayerNorm/GroupNorm layers
      "condition_embedder",  # The entire time/text conditioning module
      "scale_shift_table",  # Catches both the final and the AdaLN tables
  ]

  path_str = ".".join(str(k.key) if isinstance(k, jax.tree_util.DictKey) else str(k) for k in path)

  if any(keyword in path_str.lower() for keyword in exclusion_keywords):
    print("is_norm_path: ", path)
    # Keep LayerNorm/GroupNorm weights and biases in full precision
    return x.astype(jnp.float32)
  else:
    # Cast everything else to dtype_to_cast
    return x.astype(dtype_to_cast)


def basic_clean(text):
  if is_ftfy_available():
    import ftfy

    text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def prompt_clean(text):
  text = whitespace_clean(basic_clean(text))
  return text


def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs


# For some reason, jitting this function increases the memory significantly, so instead manually move weights to device.
def create_sharded_logical_transformer(
    devices_array: np.array,
    mesh: Mesh,
    rngs: nnx.Rngs,
    config: HyperParameters,
    restored_checkpoint=None,
    subfolder: str = "",
):
  def create_model(rngs: nnx.Rngs, wan_config: dict):
    wan_transformer = WanModel(**wan_config, rngs=rngs)
    return wan_transformer

  # 1. Load config.
  if restored_checkpoint:
    wan_config = restored_checkpoint["wan_config"]
  else:
    wan_config = WanModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)
  if config.model_type == "I2V":
    # WAN 2.1 I2V uses image embeddings via CLIP encoder (image_dim and added_kv_proj_dim are set)
    # WAN 2.2 I2V uses VAE-encoded latent conditioning (image_dim and added_kv_proj_dim are None in the transformer config)
    if config.model_name == "wan2.1":
      if wan_config.get("image_seq_len") is None:
        wan_config["image_seq_len"] = 257

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

  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, wan_config=wan_config)
  wan_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_transformer, nnx.Param, ...)

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


@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model


class WanPipeline:
  r"""
  Pipeline for text-to-video generation using Wan.

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

  def __init__(
      self,
      tokenizer: AutoTokenizer,
      text_encoder: UMT5EncoderModel,
      vae: AutoencoderKLWan,
      vae_cache: AutoencoderKLWanCache,
      scheduler: FlaxUniPCMultistepScheduler,
      scheduler_state: UniPCMultistepSchedulerState,
      devices_array: np.array,
      mesh: Mesh,
      config: HyperParameters,
      image_processor: Optional[CLIPImageProcessor] = None,
      image_encoder: Optional[FlaxCLIPVisionModel] = None,
  ):
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.vae = vae
    self.vae_cache = vae_cache
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.model_name = config.model_name
    self.image_processor = image_processor
    self.image_encoder = image_encoder

    self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
    self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    self.p_run_inference = None

  @classmethod
  def load_text_encoder(cls, config: HyperParameters):
    text_encoder = UMT5EncoderModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    return text_encoder

  @classmethod
  def load_tokenizer(cls, config: HyperParameters):
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    return tokenizer

  @classmethod
  def load_image_encoder(cls, config: HyperParameters):
    image_processor = CLIPImageProcessor.from_pretrained(config.pretrained_model_name_or_path, subfolder="image_processor")
    try:
      image_encoder = FlaxCLIPVisionModel.from_pretrained(
          config.pretrained_model_name_or_path, subfolder="image_encoder", dtype=jnp.float32
      )
    except Exception as e:
      max_logging.error(f"Failed to load FlaxCLIPVisionModel: {e}")
      raise
    return image_processor, image_encoder

  @classmethod
  def load_vae(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
    def create_model(rngs: nnx.Rngs, config: HyperParameters):
      wan_vae = AutoencoderKLWan.from_config(
          config.pretrained_model_name_or_path,
          subfolder="vae",
          rngs=rngs,
          mesh=mesh,
          dtype=jnp.float32,
          weights_dtype=jnp.float32,
      )
      return wan_vae

    # 1. eval shape
    p_model_factory = partial(create_model, config=config)
    wan_vae = nnx.eval_shape(p_model_factory, rngs=rngs)
    graphdef, state = nnx.split(wan_vae, nnx.Param)

    # 2. retrieve the state shardings, mapping logical names to mesh axis names.
    logical_state_spec = nnx.get_partition_spec(state)
    logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
    logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
    params = state.to_pure_dict()
    state = dict(nnx.to_flat_state(state))

    # 4. Load pretrained weights and move them to device using the state shardings from (3) above.
    # This helps with loading sharded weights directly into the accelerators without fist copying them
    # all to one device and then distributing them, thus using low HBM memory.
    params = load_wan_vae(config.pretrained_model_name_or_path, params, "cpu")
    params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
    for path, val in flax.traverse_util.flatten_dict(params).items():
      sharding = logical_state_sharding[path].value
      if config.replicate_vae:
        sharding = NamedSharding(mesh, P())
      state[path].value = device_put_replicated(val, sharding)
    state = nnx.from_flat_state(state)

    wan_vae = nnx.merge(graphdef, state)
    vae_cache = AutoencoderKLWanCache(wan_vae)
    return wan_vae, vae_cache

  @classmethod
  def get_basic_config(cls, dtype, config: HyperParameters):
    rules = [
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=dtype,
            act_qtype=dtype,
            op_names=("dot_general", "einsum", "conv_general_dilated"),
        )
    ]
    return rules

  @classmethod
  def get_fp8_config(cls, config: HyperParameters):
    """
    fp8 config rules with per-tensor calibration.
    FLAX API (https://flax-linen.readthedocs.io/en/v0.10.6/guides/quantization/fp8_basics.html#flax-low-level-api):
    The autodiff does not automatically use E5M2 for gradients and E4M3 for activations/weights during training, which is the recommended practice.
    """
    rules = [
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=config.weight_quantization_calibration_method,
            act_calibration_method=config.act_quantization_calibration_method,
            bwd_calibration_method=config.bwd_quantization_calibration_method,
            op_names=("dot_general", "einsum"),
        ),
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=jnp.float8_e4m3fn,  # conv_general_dilated requires the same dtypes
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e4m3fn,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=config.weight_quantization_calibration_method,
            act_calibration_method=config.act_quantization_calibration_method,
            bwd_calibration_method=config.bwd_quantization_calibration_method,
            op_names=("conv_general_dilated"),
        ),
    ]
    return rules

  @classmethod
  def get_qt_provider(cls, config: HyperParameters) -> Optional[qwix.QtProvider]:
    """Get quantization rules based on the config."""
    if not getattr(config, "use_qwix_quantization", False):
      return None

    match config.quantization:
      case "int8":
        return qwix.QtProvider(cls.get_basic_config(jnp.int8, config))
      case "fp8":
        return qwix.QtProvider(cls.get_basic_config(jnp.float8_e4m3fn, config))
      case "fp8_full":
        return qwix.QtProvider(cls.get_fp8_config(config))
    return None

  @classmethod
  def quantize_transformer(cls, config: HyperParameters, model: WanModel, pipeline: "WanPipeline", mesh: Mesh):
    """Quantizes the transformer model."""
    q_rules = cls.get_qt_provider(config)
    if not q_rules:
      return model
    max_logging.log("Quantizing transformer with Qwix.")

    batch_size = config.global_batch_size_to_train_on
    latents, prompt_embeds, timesteps = get_dummy_wan_inputs(config, pipeline, batch_size)
    model_inputs = (latents, timesteps, prompt_embeds)
    with mesh:
      quantized_model = qwix.quantize_model(model, q_rules, *model_inputs)
    max_logging.log("Qwix Quantization complete.")
    return quantized_model

  @classmethod
  def load_transformer(
      cls,
      devices_array: np.array,
      mesh: Mesh,
      rngs: nnx.Rngs,
      config: HyperParameters,
      restored_checkpoint=None,
      subfolder="transformer",
  ):
    with mesh:
      wan_transformer = create_sharded_logical_transformer(
          devices_array=devices_array,
          mesh=mesh,
          rngs=rngs,
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder=subfolder,
      )
    return wan_transformer

  @classmethod
  def load_scheduler(cls, config):
    scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
        flow_shift=config.flow_shift,  # 5.0 for 720p, 3.0 for 480p
    )
    return scheduler, scheduler_state

  def encode_image(self, image: PipelineImageInput, num_videos_per_prompt: int = 1):
    if not isinstance(image, list):
      image = [image]
    image_inputs = self.image_processor(images=image, return_tensors="np")
    pixel_values = jnp.array(image_inputs.pixel_values)

    image_encoder_output = self.image_encoder(pixel_values, output_hidden_states=True)
    image_embeds = image_encoder_output.hidden_states[-2]

    image_embeds = jnp.repeat(image_embeds, num_videos_per_prompt, axis=0)
    return image_embeds

  def _get_t5_prompt_embeds(
      self,
      prompt: Union[str, List[str]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 226,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 226,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if prompt_embeds is None:
      prompt_embeds = self._get_t5_prompt_embeds(
          prompt=prompt,
          num_videos_per_prompt=num_videos_per_prompt,
          max_sequence_length=max_sequence_length,
      )
      prompt_embeds = jnp.array(prompt_embeds.detach().numpy(), dtype=jnp.float32)

    if negative_prompt_embeds is None:
      negative_prompt = negative_prompt or ""
      negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
      negative_prompt_embeds = self._get_t5_prompt_embeds(
          prompt=negative_prompt,
          num_videos_per_prompt=num_videos_per_prompt,
          max_sequence_length=max_sequence_length,
      )
      negative_prompt_embeds = jnp.array(negative_prompt_embeds.detach().numpy(), dtype=jnp.float32)

    return prompt_embeds, negative_prompt_embeds

  def prepare_latents(
      self,
      batch_size: int,
      vae_scale_factor_temporal: int,
      vae_scale_factor_spatial: int,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_channels_latents: int = 16,
  ):
    rng = jax.random.key(self.config.seed)
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    shape = (
        batch_size,
        num_channels_latents,
        num_latent_frames,
        int(height) // vae_scale_factor_spatial,
        int(width) // vae_scale_factor_spatial,
    )
    latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)

    return latents

  def prepare_latents_i2v_base(
      self,
      image: jax.Array,
      num_frames: int,
      dtype: jnp.dtype,
      last_image: Optional[jax.Array] = None,
  ) -> Tuple[jax.Array, jax.Array]:
    """
    Encodes the initial image(s) into latents to be used as conditioning.
    Returns:
        latent_condition: The VAE encoded latents of the image(s).
        video_condition: The input to the VAE.
    """
    height, width = image.shape[-2:]
    image = image[:, :, jnp.newaxis, :, :]  # [B, C, 1, H, W]

    if last_image is None:
      video_condition = jnp.concatenate(
          [image, jnp.zeros((image.shape[0], image.shape[1], num_frames - 1, height, width), dtype=image.dtype)], axis=2
      )
    else:
      last_image = last_image[:, :, jnp.newaxis, :, :]
      video_condition = jnp.concatenate(
          [image, jnp.zeros((image.shape[0], image.shape[1], num_frames - 2, height, width), dtype=image.dtype), last_image],
          axis=2,
      )

    vae_dtype = getattr(self.vae, "dtype", jnp.float32)
    video_condition = video_condition.astype(vae_dtype)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      data_mesh_size = self.mesh.shape[self.config.mesh_axes[0]]
      if video_condition.shape[0] % data_mesh_size == 0:
        sharding_spec = P(self.config.mesh_axes[0], None, None, None, None)
        video_condition = jax.lax.with_sharding_constraint(video_condition, sharding_spec)
      encoded_output = self.vae.encode(video_condition, self.vae_cache)[0].mode()

    # Normalize latents
    latents_mean = jnp.array(self.vae.latents_mean).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents_std = jnp.array(self.vae.latents_std).reshape(1, 1, 1, 1, self.vae.z_dim)
    latent_condition = encoded_output
    latent_condition = latent_condition.astype(dtype)
    latent_condition = (latent_condition - latents_mean) / latents_std

    return latent_condition, video_condition

  def _denormalize_latents(self, latents: jax.Array) -> jax.Array:
    """Denormalizes latents using VAE statistics."""
    latents_mean = jnp.array(self.vae.latents_mean).reshape(1, self.vae.z_dim, 1, 1, 1)
    latents_std = 1.0 / jnp.array(self.vae.latents_std).reshape(1, self.vae.z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    latents = latents.astype(jnp.float32)
    return latents

  def _decode_latents_to_video(self, latents: jax.Array) -> np.ndarray:
    """Decodes latents to video frames and postprocesses."""
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      video = self.vae.decode(latents, self.vae_cache)[0]

    video = jnp.transpose(video, (0, 4, 1, 2, 3))
    video = jax.experimental.multihost_utils.process_allgather(video, tiled=True)
    video = torch.from_numpy(np.array(video.astype(dtype=jnp.float32))).to(dtype=torch.bfloat16)
    return self.video_processor.postprocess_video(video, output_type="np")

  @classmethod
  def _create_common_components(cls, config, vae_only=False, i2v=False):
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)

    with mesh:
      wan_vae, vae_cache = cls.load_vae(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)

    components = {
        "vae": wan_vae,
        "vae_cache": vae_cache,
        "devices_array": devices_array,
        "rngs": rngs,
        "mesh": mesh,
        "tokenizer": None,
        "text_encoder": None,
        "scheduler": None,
        "scheduler_state": None,
        "image_processor": None,
        "image_encoder": None,
    }

    if not vae_only:
      components["tokenizer"] = cls.load_tokenizer(config=config)
      components["text_encoder"] = cls.load_text_encoder(config=config)
      components["scheduler"], components["scheduler_state"] = cls.load_scheduler(config=config)
      if i2v and config.model_name == "wan2.1":
        components["image_processor"], components["image_encoder"] = cls.load_image_encoder(config)
    return components

  @abstractmethod
  def _get_num_channel_latents(self) -> int:
    """Returns the number of input channels for the transformer."""
    pass

  def _prepare_model_inputs_i2v(
      self,
      prompt: Union[str, List[str]],
      image: Union[PIL.Image.Image, List[PIL.Image.Image]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 512,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      image_embeds: Optional[jax.Array] = None,
      last_image: Optional[PIL.Image.Image] = None,
  ):
    if prompt is not None and isinstance(prompt, str):
      prompt = [prompt]
    batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0] // num_videos_per_prompt
    effective_batch_size = batch_size * num_videos_per_prompt

    # 1. Encode Prompts
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 2. Encode Image (only for WAN 2.1 I2V which uses CLIP image embeddings)
    # WAN 2.2 I2V does not use CLIP image embeddings, it uses VAE latent conditioning instead
    transformer_dtype = self.config.activations_dtype

    if self.config.model_name == "wan2.1":
      # WAN 2.1 I2V: Use CLIP image encoder
      if image_embeds is None:
        images_to_encode = [image]
        if last_image is None:
          images_to_encode = [image]
        else:
          images_to_encode = [image, last_image]
        image_embeds = self.encode_image(images_to_encode, num_videos_per_prompt=num_videos_per_prompt)
        self.image_seq_len = image_embeds.shape[1]

      if batch_size > 1:
        image_embeds = jnp.tile(image_embeds, (batch_size, 1, 1))

      image_embeds = image_embeds.astype(transformer_dtype)
    else:
      # WAN 2.2 I2V: No CLIP image embeddings, set to None or empty tensor
      # The actual image conditioning happens via VAE latents in prepare_latents
      image_embeds = None
    prompt_embeds = prompt_embeds.astype(transformer_dtype)
    if negative_prompt_embeds is not None:
      negative_prompt_embeds = negative_prompt_embeds.astype(transformer_dtype)

    # Use same sharding logic as T2V pipeline for consistent behavior
    data_sharding = NamedSharding(self.mesh, P())
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)
    image_embeds = jax.device_put(image_embeds, data_sharding)

    return prompt_embeds, negative_prompt_embeds, image_embeds, effective_batch_size

  def _prepare_model_inputs(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: jax.Array = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
      vae_only: bool = False,
  ):
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

      with jax.named_scope("Encode-Prompt"):
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

      num_channel_latents = self._get_num_channel_latents()
      if latents is None:
        latents = self.prepare_latents(
            batch_size=batch_size,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            height=height,
            width=width,
            num_frames=num_frames,
            num_channels_latents=num_channel_latents,
        )

      data_sharding = NamedSharding(self.mesh, P())
      # Using global_batch_size_to_train_on so not to create more config variables
      if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
        data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

      latents = jax.device_put(latents, data_sharding)
      prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
      negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)

      scheduler_state = self.scheduler.set_timesteps(
          self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
      )

      return latents, prompt_embeds, negative_prompt_embeds, scheduler_state, num_frames

  @abstractmethod
  def __call__(self, **kwargs):
    """Runs the inference pipeline."""
    pass


@partial(jax.jit, static_argnames=("do_classifier_free_guidance", "guidance_scale", "return_residual", "skip_blocks"))
def transformer_forward_pass(
    graphdef,
    sharded_state,
    rest_of_state,
    latents,
    timestep,
    prompt_embeds,
    do_classifier_free_guidance,
    guidance_scale,
    encoder_hidden_states_image=None,
    skip_blocks=None,
    cached_residual=None,
    return_residual=False,
):
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  outputs = wan_transformer(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds,
      encoder_hidden_states_image=encoder_hidden_states_image,
      skip_blocks=skip_blocks,
      cached_residual=cached_residual,
      return_residual=return_residual,
  )
  
  if return_residual:
    noise_pred, residual_x = outputs
  else:
    noise_pred = outputs

  if do_classifier_free_guidance:
    bsz = latents.shape[0] // 2
    noise_cond = noise_pred[:bsz]  # First half = conditional
    noise_uncond = noise_pred[bsz:]  # Second half = unconditional
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    latents = latents[:bsz]

  if return_residual:
    return noise_pred, latents, residual_x
  return noise_pred, latents


@partial(jax.jit, static_argnames=("guidance_scale",))
def transformer_forward_pass_full_cfg(
    graphdef,
    sharded_state,
    rest_of_state,
    latents_doubled: jnp.array,
    timestep: jnp.array,
    prompt_embeds_combined: jnp.array,
    guidance_scale: float,
    encoder_hidden_states_image=None,
):
  """Full CFG forward pass.

  Accepts pre-doubled latents and pre-concatenated [cond, uncond] prompt embeds.
  Returns the merged noise_pred plus raw noise_cond and noise_uncond for
  CFG cache storage.  Keeping cond/uncond separate avoids a second forward
  pass on cache steps.
  """
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  bsz = latents_doubled.shape[0] // 2
  
  noise_pred = wan_transformer(
      hidden_states=latents_doubled,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds_combined,
      encoder_hidden_states_image=encoder_hidden_states_image,
      skip_blocks=False,
      cached_residual=None,
      return_residual=False,
  )
    
  noise_cond = noise_pred[:bsz]
  noise_uncond = noise_pred[bsz:]
  noise_pred_merged = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
  
  return noise_pred_merged, noise_cond, noise_uncond


@partial(jax.jit, static_argnames=("guidance_scale",))
def transformer_forward_pass_cfg_cache(
    graphdef,
    sharded_state,
    rest_of_state,
    latents_cond: jnp.array,
    timestep_cond: jnp.array,
    prompt_cond_embeds: jnp.array,
    cached_noise_cond: jnp.array,
    cached_noise_uncond: jnp.array,
    guidance_scale: float,
    w1: float = 1.0,
    w2: float = 1.0,
    encoder_hidden_states_image=None,
):
  """CFG-Cache forward pass with FFT frequency-domain compensation.

  FasterCache (Lv et al., ICLR 2025) CFG-Cache:
    1. Compute frequency-domain bias:  ΔF = FFT(uncond) - FFT(cond)
    2. Split into low-freq (ΔLF) and high-freq (ΔHF) via spectral mask
    3. Apply phase-dependent weights:
         F_low  = FFT(new_cond)_low  + w1 * ΔLF
         F_high = FFT(new_cond)_high + w2 * ΔHF
    4. Reconstruct:  uncond_approx = IFFT(F_low + F_high)

  w1/w2 encode the denoising phase:
    Early (high noise): w1=1+α, w2=1   → boost low-freq correction
    Late  (low noise):  w1=1,   w2=1+α → boost high-freq correction
  where α=0.2 (FasterCache default).

  On TPU this compiles to a single static XLA graph with half the batch size
  of a full CFG pass.
  """
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  noise_cond = wan_transformer(
      hidden_states=latents_cond,
      timestep=timestep_cond,
      encoder_hidden_states=prompt_cond_embeds,
      encoder_hidden_states_image=encoder_hidden_states_image,
  )

  # FFT over spatial dims (H, W) — last 2 dims of [B, C, F, H, W]
  fft_cond_cached = jnp.fft.rfft2(cached_noise_cond.astype(jnp.float32))
  fft_uncond_cached = jnp.fft.rfft2(cached_noise_uncond.astype(jnp.float32))
  fft_bias = fft_uncond_cached - fft_cond_cached

  # Build low/high frequency mask (25% cutoff)
  h = fft_bias.shape[-2]
  w_rfft = fft_bias.shape[-1]
  ch = jnp.maximum(1, h // 4)
  cw = jnp.maximum(1, w_rfft // 4)
  freq_h = jnp.arange(h)
  freq_w = jnp.arange(w_rfft)
  # Low-freq: indices near DC (0) in both dims; account for wrap-around in dim H
  low_h = (freq_h < ch) | (freq_h >= h - ch + 1)
  low_w = freq_w < cw
  low_mask = (low_h[:, None] & low_w[None, :]).astype(jnp.float32)
  high_mask = 1.0 - low_mask

  # Apply phase-dependent weights to frequency bias
  fft_bias_weighted = fft_bias * (low_mask * w1 + high_mask * w2)

  # Reconstruct unconditional output
  fft_cond_new = jnp.fft.rfft2(noise_cond.astype(jnp.float32))
  fft_uncond_approx = fft_cond_new + fft_bias_weighted
  noise_uncond_approx = jnp.fft.irfft2(fft_uncond_approx, s=noise_cond.shape[-2:]).astype(noise_cond.dtype)

  noise_pred_merged = noise_uncond_approx + guidance_scale * (noise_cond - noise_uncond_approx)
  return noise_pred_merged, noise_cond

def nearest_interp(src, target_len):
    """Nearest neighbor interpolation for ratio scaling layout."""
    src_len = len(src)
    if target_len == 1: 
        import numpy as np
        return np.array([src[-1]])
    import numpy as np
    indices = np.round(np.linspace(0, src_len - 1, target_len)).astype(np.int32)
    return src[indices]

def init_magcache(num_inference_steps, retention_ratio, mag_ratios_base, split_step=None, model_type="T2V"):
    """Initialize MagCache variables and interpolate ratios.
    
    Args:
        num_inference_steps: Number of inference steps.
        retention_ratio: Retention ratio of unchanged steps.
        mag_ratios_base: Base magnitude ratios array or list.
        split_step: Step at which model switches (e.g. high -> low noise for 2.2).
        model_type: Pipeline mode ("T2V" or "I2V").
    """
    import numpy as np
    
    accumulated_ratio_cond = 1.0
    accumulated_ratio_uncond = 1.0
    accumulated_err_cond = 0.0
    accumulated_err_uncond = 0.0
    accumulated_steps_cond = 0
    accumulated_steps_uncond = 0
    cached_residual = None

    skip_warmup = int(num_inference_steps * retention_ratio)

    mag_ratios_base = np.array(mag_ratios_base)

    if len(mag_ratios_base) != num_inference_steps * 2:
        mag_cond = nearest_interp(mag_ratios_base[0::2], num_inference_steps)
        mag_uncond = nearest_interp(mag_ratios_base[1::2], num_inference_steps)
        mag_ratios = np.concatenate([mag_cond.reshape(-1, 1), mag_uncond.reshape(-1, 1)], axis=1).reshape(-1)
    else:
        mag_ratios = mag_ratios_base

    return (
        accumulated_ratio_cond,
        accumulated_ratio_uncond,
        accumulated_err_cond,
        accumulated_err_uncond,
        accumulated_steps_cond,
        accumulated_steps_uncond,
        cached_residual,
        skip_warmup,
        mag_ratios,
        split_step,
        model_type,
    )

def magcache_step(
    step,
    mag_ratios,
    accumulated_state,
    magcache_thresh,
    magcache_K,
    skip_warmup,
    split_step=None,
    model_type="T2V",
    num_steps=None,
    retention_ratio=0.2,
):
    """Update MagCache accumulated state and decide if to skip.
    
    Args:
        step: Current inference step.
        mag_ratios: Interpolated magnitude ratios array.
        accumulated_state: Tuple containing accumulated variables.
        magcache_thresh: Error threshold.
        magcache_K: Max skip steps.
        skip_warmup: Warmup steps threshold.
        split_step: Optional step index where the model switches (e.g., from high to low noise).
        model_type: Pipeline type ("T2V" or "I2V").
        num_steps: Total inference steps, used to calculate post-split warmups.
        retention_ratio: Used to calculate post-split warmups.
    """
    import numpy as np
    
    (
        accumulated_ratio_cond,
        accumulated_ratio_uncond,
        accumulated_err_cond,
        accumulated_err_uncond,
        accumulated_steps_cond,
        accumulated_steps_uncond,
    ) = accumulated_state

    cur_mag_ratio_cond = mag_ratios[step * 2]
    cur_mag_ratio_uncond = mag_ratios[step * 2 + 1]

    use_magcache = True
    if split_step is not None:
        if model_type == "I2V":
            if step < int(split_step + (num_steps - split_step) * retention_ratio):
                use_magcache = False
        else:
            if step < int(split_step * retention_ratio) or (step <= ((num_steps - split_step) * retention_ratio + split_step) and step >= split_step):
                use_magcache = False
    else:
        if step < skip_warmup:
            use_magcache = False

    skip_blocks = False
    if use_magcache:
        new_ratio_cond = accumulated_ratio_cond * cur_mag_ratio_cond
        new_ratio_uncond = accumulated_ratio_uncond * cur_mag_ratio_uncond

        err_cond = np.abs(1.0 - new_ratio_cond)
        err_uncond = np.abs(1.0 - new_ratio_uncond)

        if (
            accumulated_err_cond + err_cond < magcache_thresh
            and accumulated_steps_cond < magcache_K
            and accumulated_err_uncond + err_uncond < magcache_thresh
            and accumulated_steps_uncond < magcache_K
        ):
            skip_blocks = True
            accumulated_ratio_cond = new_ratio_cond
            accumulated_ratio_uncond = new_ratio_uncond
            accumulated_err_cond += err_cond
            accumulated_err_uncond += err_uncond
            accumulated_steps_cond += 1
            accumulated_steps_uncond += 1
        else:
            accumulated_ratio_cond = 1.0
            accumulated_ratio_uncond = 1.0
            accumulated_err_cond = 0.0
            accumulated_err_uncond = 0.0
            accumulated_steps_cond = 0
            accumulated_steps_uncond = 0

    new_state = (
        accumulated_ratio_cond,
        accumulated_ratio_uncond,
        accumulated_err_cond,
        accumulated_err_uncond,
        accumulated_steps_cond,
        accumulated_steps_uncond,
    )
    return skip_blocks, new_state