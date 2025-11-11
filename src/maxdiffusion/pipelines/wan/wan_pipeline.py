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
from typing import List, Union, Optional, Type
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
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None, subfolder: str = ""
):

  def create_model(rngs: nnx.Rngs, wan_config: dict):
    wan_transformer = WanModel(**wan_config, rngs=rngs)
    return wan_transformer

  # 1. Load config.
  if restored_checkpoint:
    wan_config = restored_checkpoint["wan_config"]
  else:
    wan_config = WanModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)
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
  wan_config["scan_layers"] = config.scan_layers

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
  _SUBCLASS_MAP: dict[str, Type['WanPipeline']] = {}
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
            weight_calibration_method=config.quantization_calibration_method,
            act_calibration_method=config.quantization_calibration_method,
            bwd_calibration_method=config.quantization_calibration_method,
            op_names=("dot_general", "einsum"),
        ),
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=jnp.float8_e4m3fn,  # conv_general_dilated requires the same dtypes
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e4m3fn,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=config.quantization_calibration_method,
            act_calibration_method=config.quantization_calibration_method,
            bwd_calibration_method=config.quantization_calibration_method,
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

    batch_size = jnp.ceil(config.per_device_batch_size * jax.local_device_count()).astype(jnp.int32)
    latents, prompt_embeds, timesteps = get_dummy_wan_inputs(config, pipeline, batch_size)
    model_inputs = (latents, timesteps, prompt_embeds)
    with mesh:
      quantized_model = qwix.quantize_model(model, q_rules, *model_inputs)
    max_logging.log("Qwix Quantization complete.")
    return quantized_model

  @classmethod
  def load_transformer(
      cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None, subfolder="transformer"):
    with mesh:
      wan_transformer = create_sharded_logical_transformer(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint, subfolder=subfolder
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
  def _create_common_components(cls, config, vae_only=False):
      devices_array = max_utils.create_device_mesh(config)
      mesh = Mesh(devices_array, config.mesh_axes)
      rng = jax.random.key(config.seed)
      rngs = nnx.Rngs(rng)

      with mesh:
          wan_vae, vae_cache = cls.load_vae(devices_array=devices_array, mesh=mesh, rngs=rngs, config=config)

      components = {
          "vae": wan_vae, "vae_cache": vae_cache,
          "devices_array": devices_array, "rngs": rngs, "mesh": mesh,
          "tokenizer": None, "text_encoder": None, "scheduler": None, "scheduler_state": None
      }

      if not vae_only:
          components["tokenizer"] = cls.load_tokenizer(config=config)
          components["text_encoder"] = cls.load_text_encoder(config=config)
          components["scheduler"], components["scheduler_state"] = cls.load_scheduler(config=config)
      return components

  @classmethod
  def _get_subclass(cls, model_key: str) -> Type['WanPipeline']:
    subclass = cls._SUBCLASS_MAP.get(model_key)
    if subclass is None:
        raise ValueError(
            f"Unknown model_key for WanPipeline: '{model_key}'. "
            f"Supported keys are: {list(cls._SUBCLASS_MAP.keys())}"
        )
    return subclass

  @classmethod
  def from_checkpoint(cls, model_key: str, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    subclass = cls._get_subclass(model_key)
    return subclass.from_checkpoint(config, restored_checkpoint=restored_checkpoint, vae_only=vae_only, load_transformer=load_transformer)

  @classmethod
  def from_pretrained(cls, model_key: str, config: HyperParameters, vae_only=False, load_transformer=True):
    subclass = cls._get_subclass(model_key)
    return subclass.from_pretrained(config, vae_only=vae_only, load_transformer=load_transformer)

  @abstractmethod
  def _get_num_channel_latents(self) -> int:
    """Returns the number of input channels for the transformer."""
    pass

  def _prepare_call_inputs(
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

class WanPipeline2_1(WanPipeline):
  """Pipeline for WAN 2.1 with a single transformer."""
  def __init__(self, config: HyperParameters, transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only:
      if load_transformer:
        transformer = super().load_transformer(
            devices_array=common_components["devices_array"],
            mesh=common_components["mesh"],
            rngs=common_components["rngs"],
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer"
        )

        pipeline = cls(
          tokenizer=common_components["tokenizer"],
          text_encoder=common_components["text_encoder"],
          transformer=transformer,
          vae=common_components["vae"],
          vae_cache=common_components["vae_cache"],
          scheduler=common_components["scheduler"],
          scheduler_state=common_components["scheduler_state"],
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          config=config,
        )

    return pipeline, transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline , transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  def _get_num_channel_latents(self) -> int:
    return self.transformer.config.in_channels

  def __call__(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    max_sequence_length: int = 512,
    latents: Optional[jax.Array] = None,
    prompt_embeds: Optional[jax.Array] = None,
    negative_prompt_embeds: Optional[jax.Array] = None,
    vae_only: bool = False,
  ):
    latents, prompt_embeds, negative_prompt_embeds, scheduler_state, num_frames = self._prepare_call_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        num_videos_per_prompt,
        max_sequence_length,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        vae_only,
    )

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    p_run_inference = partial(
        run_inference_2_1,
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
      )
      latents = self._denormalize_latents(latents)
    return self._decode_latents_to_video(latents)

class WanPipeline2_2(WanPipeline):
  """Pipeline for WAN 2.2 with dual transformers."""
  def __init__(self, config: HyperParameters, low_noise_transformer: Optional[WanModel], high_noise_transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.low_noise_transformer = low_noise_transformer
    self.high_noise_transformer = high_noise_transformer

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    low_noise_transformer, high_noise_transformer = None, None
    if not vae_only and load_transformer:
        low_noise_transformer = super().load_transformer(
            devices_array=common_components["devices_array"],
            mesh=common_components["mesh"],
            rngs=common_components["rngs"],
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer"
        )
        high_noise_transformer = super().load_transformer(
            devices_array=common_components["devices_array"],
            mesh=common_components["mesh"],
            rngs=common_components["rngs"],
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer_2"
        )

        pipeline = cls(
          tokenizer=common_components["tokenizer"],
          text_encoder=common_components["text_encoder"],
          low_noise_transformer=low_noise_transformer,
          high_noise_transformer=high_noise_transformer,
          vae=common_components["vae"],
          vae_cache=common_components["vae_cache"],
          scheduler=common_components["scheduler"],
          scheduler_state=common_components["scheduler_state"],
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          config=config,
        )
    return pipeline, low_noise_transformer, high_noise_transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline, low_noise_transformer, high_noise_transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    low_noise_transformer = cls.quantize_transformer(config, low_noise_transformer, pipeline, pipeline.mesh)
    high_noise_transformer = cls.quantize_transformer(config, high_noise_transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, low_noise_transformer, high_noise_transformer = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  def _get_num_channel_latents(self) -> int:
    return self.low_noise_transformer.config.in_channels

  def __call__(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale_low: float = 3.0,
    guidance_scale_high: float = 4.0,
    boundary: int = 875,
    num_videos_per_prompt: Optional[int] = 1,
    max_sequence_length: int = 512,
    latents: jax.Array = None,
    prompt_embeds: jax.Array = None,
    negative_prompt_embeds: jax.Array = None,
    vae_only: bool = False,
  ):
    latents, prompt_embeds, negative_prompt_embeds, scheduler_state, num_frames = self._prepare_call_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        num_videos_per_prompt,
        max_sequence_length,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        vae_only,
    )

    low_noise_graphdef, low_noise_state, low_noise_rest = nnx.split(self.low_noise_transformer, nnx.Param, ...)
    high_noise_graphdef, high_noise_state, high_noise_rest = nnx.split(self.high_noise_transformer, nnx.Param, ...)

    p_run_inference = partial(
        run_inference_2_2,
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=boundary,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      latents = p_run_inference(
          low_noise_graphdef=low_noise_graphdef,
          low_noise_state=low_noise_state,
          low_noise_rest=low_noise_rest,
          high_noise_graphdef=high_noise_graphdef,
          high_noise_state=high_noise_state,
          high_noise_rest=high_noise_rest,
          latents=latents,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )
      latents = self._denormalize_latents(latents)
    return self._decode_latents_to_video(latents)

@partial(jax.jit, static_argnames=("do_classifier_free_guidance", "guidance_scale"))
def transformer_forward_pass(
    graphdef,
    sharded_state,
    rest_of_state,
    latents,
    timestep,
    prompt_embeds,
    do_classifier_free_guidance,
    guidance_scale,
):
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  noise_pred = wan_transformer(hidden_states=latents, timestep=timestep, encoder_hidden_states=prompt_embeds)
  if do_classifier_free_guidance:
    bsz = latents.shape[0] // 2
    noise_uncond = noise_pred[bsz:]
    noise_pred = noise_pred[:bsz]
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
    latents = latents[:bsz]

  return noise_pred, latents

def run_inference_2_1(
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
):
  do_classifier_free_guidance = guidance_scale > 1.0
  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
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
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
    )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents

def run_inference_2_2(
    low_noise_graphdef,
    low_noise_state,
    low_noise_rest,
    high_noise_graphdef,
    high_noise_state,
    high_noise_rest,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale_low: float,
    guidance_scale_high: float,
    boundary: int,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
):
  do_classifier_free_guidance = guidance_scale_low > 1.0 or guidance_scale_high > 1.0
  if do_classifier_free_guidance:
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

  def low_noise_branch(operands):
    latents, timestep, prompt_embeds = operands
    return transformer_forward_pass(
        low_noise_graphdef, low_noise_state, low_noise_rest,
        latents, timestep, prompt_embeds,
        do_classifier_free_guidance, guidance_scale_low
    )

  def high_noise_branch(operands):
    latents, timestep, prompt_embeds = operands
    return transformer_forward_pass(
        high_noise_graphdef, high_noise_state, high_noise_rest,
        latents, timestep, prompt_embeds,
        do_classifier_free_guidance, guidance_scale_high
    )

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    if do_classifier_free_guidance:
      latents = jnp.concatenate([latents] * 2)
    timestep = jnp.broadcast_to(t, latents.shape[0])

    use_high_noise = jnp.greater_equal(t, boundary)

    noise_pred, latents = jax.lax.cond(
        use_high_noise,
        high_noise_branch,
        low_noise_branch,
        (latents, timestep, prompt_embeds)
    )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents

WanPipeline._SUBCLASS_MAP["wan2.1"] = WanPipeline2_1
WanPipeline._SUBCLASS_MAP["wan2.2"] = WanPipeline2_2
