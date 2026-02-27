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

from typing import Optional, Any, Tuple, List, Union, Callable, Dict
import inspect
from functools import partial

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax
import flax.linen as nn
import flax.traverse_util
from flax import nnx
from transformers import AutoTokenizer, GemmaTokenizer, GemmaTokenizerFast, Gemma3ForConditionalGeneration
from tqdm.auto import tqdm
import qwix
from ...utils import logging
from ...schedulers import FlaxFlowMatchScheduler
from ...models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from ...models.ltx2.autoencoder_kl_ltx2_audio import FlaxAutoencoderKLLTX2Audio
from ...models.ltx2.vocoder_ltx2 import LTX2Vocoder
from ...models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from ...models.ltx2.ltx2_utils import (
    load_transformer_weights,
    load_connector_weights,
    load_vae_weights,
    load_audio_vae_weights,
    load_vocoder_weights,
)
from ...models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
from ...video_processor import VideoProcessor
from .ltx2_pipeline_utils import encode_video
from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated

@flax.struct.dataclass
class LTX2PipelineOutput:
    frames: jax.Array
    audio: Optional[jax.Array] = None

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure.
    Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891).
    """
    std_text = jnp.std(noise_pred_text, axis=list(range(1, noise_pred_text.ndim)), keepdims=True)
    std_cfg = jnp.std(noise_cfg, axis=list(range(1, noise_cfg.ndim)), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

logger = logging.get_logger(__name__)


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
        return x.astype(jnp.float32)
    else:
        return x.astype(dtype_to_cast)


def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
    vs.sharding_rules = logical_axis_rules
    return vs


def create_sharded_logical_transformer(
    devices_array: np.array,
    mesh: Mesh,
    rngs: nnx.Rngs,
    config: HyperParameters,
    restored_checkpoint=None,
    subfolder: str = "",
):
    def create_model(rngs: nnx.Rngs, ltx2_config: dict):
        transformer = LTX2VideoTransformer3DModel(**ltx2_config, rngs=rngs)
        return transformer

    # 1. Load config.
    if restored_checkpoint:
        ltx2_config = restored_checkpoint["ltx2_config"]
    else:
        ltx2_config = LTX2VideoTransformer3DModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)

    if ltx2_config.get("activation_fn") == "gelu-approximate":
        ltx2_config["activation_fn"] = "gelu"

    ltx2_config["mesh"] = mesh
    ltx2_config["dtype"] = config.activations_dtype
    ltx2_config["weights_dtype"] = config.weights_dtype
    ltx2_config["attention_kernel"] = config.attention
    ltx2_config["precision"] = get_precision(config)
    ltx2_config["remat_policy"] = config.remat_policy
    ltx2_config["names_which_can_be_saved"] = config.names_which_can_be_saved
    ltx2_config["names_which_can_be_offloaded"] = config.names_which_can_be_offloaded

    # 2. eval_shape
    p_model_factory = partial(create_model, ltx2_config=ltx2_config)
    transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
    graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)

    # 3. retrieve the state shardings
    logical_state_spec = nnx.get_partition_spec(state)
    logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
    logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
    params = state.to_pure_dict()
    state = dict(nnx.to_flat_state(state))

    # 4. Load pretrained weights
    if restored_checkpoint:
        if "params" in restored_checkpoint["ltx2_state"]:
             params = restored_checkpoint["ltx2_state"]["params"]
        else:
             params = restored_checkpoint["ltx2_state"]
    else:
         params = load_transformer_weights(
             config.pretrained_model_name_or_path,
             params, # eval_shapes
             "cpu",
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

    transformer = nnx.merge(graphdef, state, rest_of_state)
    return transformer



# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    scheduler_state,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        
    timesteps = jnp.array(timesteps, dtype=scheduler.dtype) if timesteps is not None else None
    sigmas = jnp.array(sigmas, dtype=scheduler.dtype) if sigmas is not None else None
    
    scheduler_state = scheduler.set_timesteps(
        scheduler_state,
        num_inference_steps=num_inference_steps,
        timesteps=timesteps,
        sigmas=sigmas,
        **kwargs,
    )
        
    return scheduler_state

class LTX2Pipeline:
  """
  Pipeline for LTX-2.
  """

  def __init__(
      self,
      scheduler: FlaxFlowMatchScheduler,
      vae: LTX2VideoAutoencoderKL,
      audio_vae: FlaxAutoencoderKLLTX2Audio,
      text_encoder: Gemma3ForConditionalGeneration, # Using PyTorch Gemma3 encoder directly per user request
      tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
      connectors: LTX2AudioVideoGemmaTextEncoder,
      transformer: LTX2VideoTransformer3DModel,
      vocoder: LTX2Vocoder,
  ):
      self.scheduler = scheduler
      self.vae = vae
      self.audio_vae = audio_vae
      self.vocoder = vocoder
      self.text_encoder = text_encoder
      self.tokenizer = tokenizer
      self.connectors = connectors
      self.transformer = transformer
      
      # VAE compression ratios
      self.vae_spatial_compression_ratio = getattr(self.vae, "spatial_compression_ratio", 32)
      self.vae_temporal_compression_ratio = getattr(self.vae, "temporal_compression_ratio", 8)
      
      # Audio VAE compression ratios
      self.audio_vae_mel_compression_ratio = getattr(self.audio_vae, "mel_compression_ratio", 4)
      self.audio_vae_temporal_compression_ratio = getattr(self.audio_vae, "temporal_compression_ratio", 4)

      # Transformer patch sizes
      self.transformer_spatial_patch_size = getattr(self.transformer.config, "patch_size", 1)
      self.transformer_temporal_patch_size = getattr(self.transformer.config, "patch_size_t", 1)
      
      self.audio_sampling_rate = getattr(self.audio_vae.config, "sample_rate", 16000)
      self.audio_hop_length = getattr(self.audio_vae.config, "mel_hop_length", 160)
      
      # Initialize video processor
      self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
      
      self.tokenizer_max_length = getattr(self.tokenizer, "model_max_length", 1024)

  @classmethod
  def load_tokenizer(cls, config: HyperParameters):
      max_logging.log("Loading Gemma Tokenizer...")
      tokenizer = AutoTokenizer.from_pretrained(
          config.pretrained_model_name_or_path,
          subfolder="tokenizer",
      )
      return tokenizer

  @classmethod
  def load_text_encoder(cls, config: HyperParameters):
      max_logging.log("Loading Gemma3 Text Encoder...")
      text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
          config.pretrained_model_name_or_path,
          subfolder="text_encoder",
          torch_dtype=torch.bfloat16,
      )
      text_encoder.eval()
      return text_encoder

  @classmethod
  def load_connectors(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
      max_logging.log("Loading Connectors...")
      def create_model(rngs: nnx.Rngs, config: HyperParameters):
          connectors = LTX2AudioVideoGemmaTextEncoder.from_config(
              config.pretrained_model_name_or_path,
              subfolder="connectors",
              rngs=rngs,
              mesh=mesh,
              dtype=jnp.float32,
              weights_dtype=config.weights_dtype if hasattr(config, "weights_dtype") else jnp.float32,
          )
          return connectors

      p_model_factory = partial(create_model, config=config)
      connectors = nnx.eval_shape(p_model_factory, rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(connectors, nnx.Param, ...)

      logical_state_spec = nnx.get_partition_spec(state)
      logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
      logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
      params = state.to_pure_dict()
      state = dict(nnx.to_flat_state(state))

      params = load_connector_weights(config.pretrained_model_name_or_path, params, "cpu", subfolder="connectors")
      if hasattr(config, "weights_dtype"):
          params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)

      for path, val in flax.traverse_util.flatten_dict(params).items():
          sharding = logical_state_sharding.get(path)
          if sharding is not None:
             sharding = sharding.value
             state[path].value = device_put_replicated(val, sharding)
          else:
             state[path].value = jax.device_put(val)
          
      state = nnx.from_flat_state(state)
      connectors = nnx.merge(graphdef, state, rest_of_state)
      return connectors

  @classmethod
  def load_vae(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
      max_logging.log("Loading Video VAE...")
      def create_model(rngs: nnx.Rngs, config: HyperParameters):
          vae = LTX2VideoAutoencoderKL.from_config(
              config.pretrained_model_name_or_path,
              subfolder="vae",
              rngs=rngs,
              mesh=mesh,
              dtype=jnp.float32,
              weights_dtype=config.weights_dtype if hasattr(config, "weights_dtype") else jnp.float32,
          )
          return vae
      
      p_model_factory = partial(create_model, config=config)
      vae = nnx.eval_shape(p_model_factory, rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(vae, nnx.Param, ...)

      logical_state_spec = nnx.get_partition_spec(state)
      logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
      logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
      params = state.to_pure_dict()
      state = dict(nnx.to_flat_state(state))

      params = load_vae_weights(config.pretrained_model_name_or_path, params, "cpu", subfolder="vae")
      if hasattr(config, "weights_dtype"):
          params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)

      for path, val in flax.traverse_util.flatten_dict(params).items():
          sharding = logical_state_sharding.get(path)
          if sharding is not None:
             sharding = sharding.value
             try:
                 replicate_vae = config.replicate_vae
             except ValueError:
                 replicate_vae = False
             if replicate_vae:
                 sharding = NamedSharding(mesh, P())
             state[path].value = device_put_replicated(val, sharding)
          else:
             state[path].value = jax.device_put(val)
          
      state = nnx.from_flat_state(state)
      vae = nnx.merge(graphdef, state, rest_of_state)
      return vae

  @classmethod
  def load_audio_vae(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
      max_logging.log("Loading Audio VAE...")
      def create_model(rngs: nnx.Rngs, config: HyperParameters):
          audio_vae = FlaxAutoencoderKLLTX2Audio.from_config(
              config.pretrained_model_name_or_path,
              subfolder="audio_vae",
              rngs=rngs,
              mesh=mesh,
              dtype=jnp.float32,
              weights_dtype=config.weights_dtype if hasattr(config, "weights_dtype") else jnp.float32,
          )
          return audio_vae

      p_model_factory = partial(create_model, config=config)
      audio_vae = nnx.eval_shape(p_model_factory, rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(audio_vae, nnx.Param, ...)

      logical_state_spec = nnx.get_partition_spec(state)
      logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
      logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
      params = state.to_pure_dict()
      state = dict(nnx.to_flat_state(state))

      params = load_audio_vae_weights(config.pretrained_model_name_or_path, params, "cpu", subfolder="audio_vae")
      if hasattr(config, "weights_dtype"):
          params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)

      for path, val in flax.traverse_util.flatten_dict(params).items():
          sharding = logical_state_sharding.get(path)
          if sharding is not None:
             sharding = sharding.value
             try:
                 replicate_vae = config.replicate_vae
             except ValueError:
                 replicate_vae = False
             if replicate_vae:
                 sharding = NamedSharding(mesh, P())
             state[path].value = device_put_replicated(val, sharding)
          else:
             state[path].value = jax.device_put(val)
          
      state = nnx.from_flat_state(state)
      audio_vae = nnx.merge(graphdef, state, rest_of_state)
      return audio_vae
      
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
        transformer = create_sharded_logical_transformer(
            devices_array=devices_array,
            mesh=mesh,
            rngs=rngs,
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder=subfolder,
        )
      return transformer

  @classmethod
  def load_vocoder(cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters):
      max_logging.log("Loading Vocoder...")
      def create_model(rngs: nnx.Rngs, config: HyperParameters):
          vocoder = LTX2Vocoder.from_config(
              config.pretrained_model_name_or_path,
              subfolder="vocoder",
              rngs=rngs,
              mesh=mesh,
              dtype=jnp.float32,
              weights_dtype=config.weights_dtype if hasattr(config, "weights_dtype") else jnp.float32,
          )
          return vocoder

      p_model_factory = partial(create_model, config=config)
      vocoder = nnx.eval_shape(p_model_factory, rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(vocoder, nnx.Param, ...)

      logical_state_spec = nnx.get_partition_spec(state)
      logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
      logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
      params = state.to_pure_dict()
      state = dict(nnx.to_flat_state(state))

      params = load_vocoder_weights(config.pretrained_model_name_or_path, params, "cpu", subfolder="vocoder")
      if hasattr(config, "weights_dtype"):
          params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)

      for path, val in flax.traverse_util.flatten_dict(params).items():
          sharding = logical_state_sharding.get(path)
          if sharding is not None:
             sharding = sharding.value
             state[path].value = device_put_replicated(val, sharding)
          else:
             state[path].value = jax.device_put(val)
          
      state = nnx.from_flat_state(state)
      vocoder = nnx.merge(graphdef, state, rest_of_state)
      return vocoder

  @classmethod
  def load_scheduler(cls, config: HyperParameters):
      max_logging.log("Loading Scheduler...")
      scheduler = FlaxFlowMatchScheduler.from_pretrained(
          config.pretrained_model_name_or_path,
          subfolder="scheduler",
      )
      return scheduler

  @classmethod
  def _create_common_components(cls, config: HyperParameters, vae_only=False):
      devices_array = max_utils.create_device_mesh(config)
      mesh = Mesh(devices_array, config.mesh_axes)
      rng = jax.random.key(config.seed)
      rngs = nnx.Rngs(rng)

      vae = cls.load_vae(devices_array, mesh, rngs, config)
      
      components = {
          "vae": vae,
          "audio_vae": None,
          "vocoder": None,
          "devices_array": devices_array,
          "rngs": rngs,
          "mesh": mesh,
          "tokenizer": None,
          "text_encoder": None,
          "connectors": None,
          "scheduler": None,
      }

      if vae_only:
          return components

      components["tokenizer"] = cls.load_tokenizer(config)
      components["text_encoder"] = cls.load_text_encoder(config)
      components["connectors"] = cls.load_connectors(devices_array, mesh, rngs, config)
      components["audio_vae"] = cls.load_audio_vae(devices_array, mesh, rngs, config)
      components["vocoder"] = cls.load_vocoder(devices_array, mesh, rngs, config)
      components["scheduler"] = cls.load_scheduler(config)
      return components

  @classmethod
  def _load_and_init(cls, config: HyperParameters, restored_checkpoint, vae_only=False, load_transformer=True):
      components = cls._create_common_components(config, vae_only)
      
      transformer = None
      if load_transformer:
          max_logging.log("Loading Transformer...")
          transformer = cls.load_transformer(
              devices_array=components["devices_array"],
              mesh=components["mesh"],
              rngs=components["rngs"],
              config=config,
              restored_checkpoint=restored_checkpoint,
          )

      pipeline = cls(
          scheduler=components["scheduler"],
          vae=components["vae"],
          audio_vae=components["audio_vae"],
          text_encoder=components["text_encoder"],
          tokenizer=components["tokenizer"],
          connectors=components["connectors"],
          transformer=transformer,
          vocoder=components["vocoder"]
      )
      pipeline.mesh = components["mesh"]
      return pipeline, transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
      pipeline, _ = cls._load_and_init(config, None, vae_only, load_transformer)
      return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint, vae_only=False, load_transformer=True):
      pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
      return pipeline

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

    if config.quantization == "int8":
      return qwix.QtProvider(cls.get_basic_config(jnp.int8, config))
    elif config.quantization == "fp8":
      return qwix.QtProvider(cls.get_basic_config(jnp.float8_e4m3fn, config))
    elif config.quantization == "fp8_full":
      return qwix.QtProvider(cls.get_fp8_config(config))
    return None

  @classmethod
  def quantize_transformer(
      cls, config: HyperParameters, model: Any, pipeline: "LTX2Pipeline", mesh: Mesh, model_inputs: Tuple[Any, ...]
  ):
    """Quantizes the transformer model."""
    q_rules = cls.get_qt_provider(config)
    if not q_rules:
      return model
    max_logging.log("Quantizing transformer with Qwix.")

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
      transformer = create_sharded_logical_transformer(
          devices_array=devices_array,
          mesh=mesh,
          rngs=rngs,
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder=subfolder,
      )
    return transformer



  @staticmethod
  def _pack_text_embeds(
      text_hidden_states: jax.Array,
      sequence_lengths: jax.Array,
      padding_side: str = "left",
      scale_factor: int = 8,
      eps: float = 1e-6,
  ) -> jax.Array:
      """
      Packs and normalizes text encoder hidden states using JAX natively to minimize PyTorch/HBM transfers.
      """
      batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
      original_dtype = text_hidden_states.dtype

      # Create padding mask
      token_indices = jnp.arange(seq_len)[None, :]
      if padding_side == "right":
          mask = token_indices < sequence_lengths[:, None]
      elif padding_side == "left":
          start_indices = seq_len - sequence_lengths[:, None]
          mask = token_indices >= start_indices
      else:
          raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
      mask = mask[:, :, None, None]

      masked_text_hidden_states = jnp.where(mask, text_hidden_states, 0.0)
      num_valid_positions = (sequence_lengths * hidden_dim).reshape(batch_size, 1, 1, 1)
      masked_mean = masked_text_hidden_states.sum(axis=(1, 2), keepdims=True) / (num_valid_positions + eps)

      x_min = jnp.min(jnp.where(mask, text_hidden_states, float("inf")), axis=(1, 2), keepdims=True)
      x_max = jnp.max(jnp.where(mask, text_hidden_states, float("-inf")), axis=(1, 2), keepdims=True)

      normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
      normalized_hidden_states = normalized_hidden_states * scale_factor

      normalized_hidden_states = normalized_hidden_states.reshape(batch_size, seq_len, -1)
      mask_flat = mask.squeeze(-1)
      mask_flat = jnp.broadcast_to(mask_flat, normalized_hidden_states.shape)
      
      normalized_hidden_states = jnp.where(mask_flat, normalized_hidden_states, 0.0)
      normalized_hidden_states = normalized_hidden_states.astype(original_dtype)
      return normalized_hidden_states

  def _get_gemma_prompt_embeds(
      self,
      prompt: Union[str, List[str]],
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 1024,
      scale_factor: int = 8,
      dtype: Optional[jnp.dtype] = None,
  ):
      prompt = [prompt] if isinstance(prompt, str) else prompt
      batch_size = len(prompt)

      if self.tokenizer is not None:
          self.tokenizer.padding_side = "left"
          if self.tokenizer.pad_token is None:
              self.tokenizer.pad_token = self.tokenizer.eos_token

      prompt = [p.strip() for p in prompt]
      # Return Numpy tensors to be compatible with JAX if no text encoder, else PyTorch

      if self.text_encoder is not None:
           # PyTorch Text Encoder
           text_inputs = self.tokenizer(
               prompt,
               padding="max_length",
               max_length=max_sequence_length,
               truncation=True,
               add_special_tokens=True,
               return_tensors="pt", 
           )
           text_input_ids = text_inputs.input_ids
           prompt_attention_mask = text_inputs.attention_mask

           # Move to device if needed (assuming text_encoder is on correct device or CPU)
           # For now, keep on CPU or same device as model
           text_input_ids = text_input_ids.to(self.text_encoder.device)
           prompt_attention_mask = prompt_attention_mask.to(self.text_encoder.device)
           
           with torch.no_grad():
                text_encoder_outputs = self.text_encoder(
                    input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
                )
           
           text_encoder_hidden_states = text_encoder_outputs.hidden_states
           del text_encoder_outputs # Free memory
           
           text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
           
           # Convert to JAX via float32, then cast to bfloat16 to match user expectations and minimize memory footprint
           text_encoder_hidden_states_jax = jnp.array(text_encoder_hidden_states.cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16)
           del text_encoder_hidden_states # Free PyTorch tensor memory
           
           prompt_attention_mask = jnp.array(prompt_attention_mask.cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16)
      else:
          raise ValueError("`text_encoder` is required to encode prompts.")

      sequence_lengths = prompt_attention_mask.sum(axis=-1)

      prompt_embeds = self._pack_text_embeds(
          text_encoder_hidden_states_jax,
          sequence_lengths,
          padding_side=self.tokenizer.padding_side,
          scale_factor=scale_factor,
      )
      del text_encoder_hidden_states_jax
      if dtype is not None:
          prompt_embeds = prompt_embeds.astype(dtype)

      _, seq_len, _ = prompt_embeds.shape
      prompt_embeds = jnp.repeat(prompt_embeds, num_videos_per_prompt, axis=0)
      prompt_embeds = prompt_embeds.reshape(batch_size * num_videos_per_prompt, seq_len, -1)

      prompt_attention_mask = prompt_attention_mask.reshape(batch_size, -1)
      prompt_attention_mask = jnp.repeat(prompt_attention_mask, num_videos_per_prompt, axis=0)

      return prompt_embeds, prompt_attention_mask

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      do_classifier_free_guidance: bool = True,
      num_videos_per_prompt: int = 1,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      prompt_attention_mask: Optional[jax.Array] = None,
      negative_prompt_attention_mask: Optional[jax.Array] = None,
      max_sequence_length: int = 1024,
      scale_factor: int = 8,
      dtype: Optional[jnp.dtype] = None,
  ):
      if prompt is not None and isinstance(prompt, str):
          batch_size = 1
      elif prompt is not None and isinstance(prompt, list):
          batch_size = len(prompt)
      else:
          batch_size = prompt_embeds.shape[0]

      if prompt_embeds is None:
          prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
              prompt=prompt,
              num_videos_per_prompt=num_videos_per_prompt,
              max_sequence_length=max_sequence_length,
              scale_factor=scale_factor,
              dtype=dtype,
          )

      if do_classifier_free_guidance and negative_prompt_embeds is None:
          negative_prompt = negative_prompt or ""
          negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

          if prompt is not None and type(prompt) is not type(negative_prompt):
              raise TypeError(
                  f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                  f" {type(prompt)}."
              )

          negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
              prompt=negative_prompt,
              num_videos_per_prompt=num_videos_per_prompt,
              max_sequence_length=max_sequence_length,
              scale_factor=scale_factor,
              dtype=dtype,
          )

      return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

  def check_inputs(
      self,
      prompt,
      height,
      width,
      prompt_embeds=None,
      negative_prompt_embeds=None,
      prompt_attention_mask=None,
      negative_prompt_attention_mask=None,
  ):
      if height % 32 != 0 or width % 32 != 0:
          raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

      if prompt is not None and prompt_embeds is not None:
           raise ValueError(
               f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
               " only forward one of the two."
           )
      elif prompt is None and prompt_embeds is None:
          raise ValueError(
              "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
          )
      elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
          raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

      if prompt_embeds is not None and prompt_attention_mask is None:
          raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

      if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
           raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

      if prompt_embeds is not None and negative_prompt_embeds is not None:
          if prompt_embeds.shape != negative_prompt_embeds.shape:
              raise ValueError(
                  "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                  f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                  f" {negative_prompt_embeds.shape}."
              )
          if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
              raise ValueError(
                  "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                  f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                  f" {negative_prompt_attention_mask.shape}."
              )

  @staticmethod
  def _pack_latents(latents: jax.Array, patch_size: int = 1, patch_size_t: int = 1) -> jax.Array:
      batch_size, num_channels, num_frames, height, width = latents.shape
      post_patch_num_frames = num_frames // patch_size_t
      post_patch_height = height // patch_size
      post_patch_width = width // patch_size
      latents = latents.reshape(
          batch_size,
          -1,
          post_patch_num_frames,
          patch_size_t,
          post_patch_height,
          patch_size,
          post_patch_width,
          patch_size,
      )
      latents = latents.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch_size, post_patch_num_frames * post_patch_height * post_patch_width, -1)
      return latents

  @staticmethod
  def _unpack_latents(
      latents: jax.Array, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
  ) -> jax.Array:
      batch_size = latents.shape[0]
      # latents: (Batch, SeqLen, Channels*Patches)
      latents = latents.reshape(batch_size, num_frames // patch_size_t, height // patch_size, width // patch_size, -1, patch_size_t, patch_size, patch_size)
      latents = latents.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(batch_size, -1, num_frames, height, width)
      return latents

  @staticmethod
  def _normalize_latents(
      latents: jax.Array, latents_mean: jax.Array, latents_std: jax.Array, scaling_factor: float = 1.0
  ) -> jax.Array:
      latents_mean = latents_mean.reshape(1, -1, 1, 1, 1).astype(latents.dtype)
      latents_std = latents_std.reshape(1, -1, 1, 1, 1).astype(latents.dtype)
      latents = (latents - latents_mean) * scaling_factor / latents_std
      return latents

  @staticmethod
  def _denormalize_latents(
      latents: jax.Array, latents_mean: jax.Array, latents_std: jax.Array, scaling_factor: float = 1.0
  ) -> jax.Array:
      latents_mean = latents_mean.reshape(1, -1, 1, 1, 1).astype(latents.dtype)
      latents_std = latents_std.reshape(1, -1, 1, 1, 1).astype(latents.dtype)
      latents = latents * latents_std / scaling_factor + latents_mean
      return latents

  @staticmethod
  def _normalize_audio_latents(latents: jax.Array, latents_mean: jax.Array, latents_std: jax.Array):
      latents_mean = latents_mean.astype(latents.dtype)
      latents_std = latents_std.astype(latents.dtype)
      return (latents - latents_mean) / latents_std

  @staticmethod
  def _denormalize_audio_latents(latents: jax.Array, latents_mean: jax.Array, latents_std: jax.Array):
      latents_mean = latents_mean.astype(latents.dtype)
      latents_std = latents_std.astype(latents.dtype)
      return (latents * latents_std) + latents_mean
  
  @staticmethod
  def _create_noised_state(
      latents: jax.Array, noise_scale: float, generator: Optional[nnx.Rngs] = None
  ):
      # Handle random generation if needed, usually passed in or managed externally
      # For inference with seeding, we usually pass rng key.
      # But here we stick to simple noise addition if noise is provided or external logic.
      # If generator is key, use it.
      if isinstance(generator, jax.Array): # PRNGKey
           noise = jax.random.normal(generator, latents.shape, dtype=latents.dtype)
      else:
           # Fallback or expect noise to be handled otherwise?
           # pipeline prepare_latents typically generates noise.
           noise = jax.random.normal(jax.random.key(0), latents.shape, dtype=latents.dtype) # Default fallback

      noised_latents = noise_scale * noise + (1 - noise_scale) * latents
      return noised_latents

  @staticmethod
  def _pack_audio_latents(
      latents: jax.Array, patch_size: Optional[int] = None, patch_size_t: Optional[int] = None
  ) -> jax.Array:
      if patch_size is not None and patch_size_t is not None:
          batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
          post_patch_latent_length = latent_length // patch_size_t
          post_patch_mel_bins = latent_mel_bins // patch_size
          latents = latents.reshape(
              batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
          )
          # Permute to (Batch, T', F', C, p_t, p)
          latents = latents.transpose(0, 2, 4, 1, 3, 5)
          latents = latents.reshape(batch_size, post_patch_latent_length * post_patch_mel_bins, -1)
      else:
          # (B, C, L) -> (B, L, C) or (B, C, L, F) -> ?
          # Assuming input is (B, C, L) or (B, C, L, F)
          # If 3D: (B, C, L) -> (B, L, C)
          if latents.ndim == 3:
               latents = latents.transpose(0, 2, 1)
          elif latents.ndim == 4:
               # (B, C, L, F) -> flatten F into C? No.
               # Check diffusers logic: `latents.transpose(1, 2).flatten(2, 3)`
               # (B, C, L, F) -> (B, L, C, F) -> (B, L, C*F)
               latents = latents.transpose(0, 2, 1, 3).reshape(latents.shape[0], latents.shape[2], -1)

      return latents

  @staticmethod
  def _unpack_audio_latents(
      latents: jax.Array,
      latent_length: int,
      num_mel_bins: int,
      patch_size: Optional[int] = None,
      patch_size_t: Optional[int] = None,
  ) -> jax.Array:
      if patch_size is not None and patch_size_t is not None:
          batch_size = latents.shape[0]
          # latents: (Batch, Seq, Dim)
          # Pack: (B, C, L, F) -> (B, C, L', pt, F', p) -> (B, C, L', pt, F', p) -> (B, L', F', C, pt, p) -> (B, L', F', C*pt*p)
          # Unpack: (B, L'*F', C*pt*p) -> (B, L', F', C, pt, p) -> (B, C, L', pt, F', p) -> (B, C, L'*pt, F'*p)
          latents = latents.reshape(batch_size, -1, num_mel_bins // patch_size, num_channels * patch_size_t * patch_size)
          latents = latents.reshape(batch_size, latent_length // patch_size_t, num_mel_bins // patch_size, num_channels, patch_size_t, patch_size)
          latents = latents.transpose(0, 3, 1, 4, 2, 5).reshape(batch_size, num_channels, latent_length, num_mel_bins)
          # Wait, reshape order needs to match pack? 
          # Pack: (B, C, L, F) -> (B, C, L', pt, F', p) -> (B, L', F', C, pt, p) -> (B, L'*F', C*pt*p)
          # Unpack: (B, L'*F', C*pt*p) -> (B, L', F', C, pt, p) -> (B, C, L', pt, F', p) -> (B, C, L'*pt, F'*p)
          # Correct.
          
      else:
          # (B, L, C*F) -> (B, L, C, F) -> (B, C, L, F)
          batch_size = latents.shape[0]
          latents = latents.reshape(batch_size, latent_length, -1, num_mel_bins)
          latents = latents.transpose(0, 2, 1, 3)
      return latents

  def prepare_latents(
      self,
      batch_size: int = 1,
      num_channels_latents: int = 128,
      height: int = 512,
      width: int = 768,
      num_frames: int = 121,
      noise_scale: float = 0.0,
      dtype: Optional[jnp.dtype] = None,
      generator: Optional[jax.Array] = None,
      latents: Optional[jax.Array] = None,
  ) -> jax.Array:
      if latents is not None:
           if latents.ndim == 5:
              latents_mean = jnp.array(self.vae.latents_mean)
              latents_std = jnp.array(self.vae.latents_std)
              scaling_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 1.0
              
              latents = self._normalize_latents(latents, latents_mean, latents_std, scaling_factor)
              latents = self._pack_latents(
                  latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
              )
           if latents.ndim != 3:
              raise ValueError("Unexpected latents shape")
           latents = self._create_noised_state(latents, noise_scale, generator)
           return latents.astype(dtype)
      
      height = height // self.vae_spatial_compression_ratio
      width = width // self.vae_spatial_compression_ratio
      num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

      shape = (batch_size, num_channels_latents, num_frames, height, width)
      if generator is None:
           generator = jax.random.key(0)
      
      latents = jax.random.normal(generator, shape, dtype=dtype or jnp.float32)
      latents = self._pack_latents(
          latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
      )
      return latents

  def prepare_audio_latents(
      self,
      batch_size: int = 1,
      num_channels_latents: int = 128,
      audio_latent_length: int = 8,
      noise_scale: float = 0.0,
      dtype: Optional[jnp.dtype] = None,
      generator: Optional[jax.Array] = None,
      latents: Optional[jax.Array] = None,
      num_mel_bins: Optional[int] = None,
  ) -> jax.Array:
      if latents is not None:
          # Assuming latents is JAX array or compatible
          if latents.ndim == 4:
              # (Batch, Channels, Length, Mel) -> Pack
              latents = self._pack_audio_latents(latents, getattr(self.audio_vae.config, "patch_size", None), getattr(self.audio_vae.config, "patch_size_t", None))
          if latents.ndim != 3:
               raise ValueError("Unexpected audio latents shape")
          
          latents_mean = jnp.array(self.audio_vae.latents_mean)
          latents_std = jnp.array(self.audio_vae.latents_std)

          latents = self._normalize_audio_latents(latents, latents_mean, latents_std)
          latents = self._create_noised_state(latents, noise_scale, generator)
          return latents.astype(dtype)

      latent_mel_bins = self.audio_vae.config.mel_bins // self.audio_vae_mel_compression_ratio
      shape = (batch_size, num_channels_latents, audio_latent_length, latent_mel_bins)
      
      if generator is None:
          generator = jax.random.key(1)
          
      latents = jax.random.normal(generator, shape, dtype=dtype or jnp.float32)
      latents = self._pack_audio_latents(
          latents, getattr(self.audio_vae.config, "patch_size", None), getattr(self.audio_vae.config, "patch_size_t", None)
      )
      return latents

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 512,
      width: int = 768,
      num_frames: int = 121,
      frame_rate: float = 24.0,
      num_inference_steps: int = 40,
      sigmas: Optional[List[float]] = None,
      timesteps: List[int] = None,
      guidance_scale: float = 3.0,
      guidance_rescale: float = 0.0,
      noise_scale: float = 1.0,
      num_videos_per_prompt: Optional[int] = 1,
      generator: Optional[jax.Array] = None,
      latents: Optional[jax.Array] = None,
      audio_latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      prompt_attention_mask: Optional[jax.Array] = None,
      negative_prompt_attention_mask: Optional[jax.Array] = None,
      decode_timestep: Union[float, List[float]] = 0.0,
      decode_noise_scale: Optional[Union[float, List[float]]] = None,
      max_sequence_length: int = 1024,
      dtype: Optional[jnp.dtype] = jnp.float32,
      output_type: str = "pil",
      return_dict: bool = True,
  ):
      # 1. Check inputs
      self.check_inputs(prompt, height, width, prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask)

      # 2. Encode inputs (Text)
      prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
          prompt, negative_prompt, 
          do_classifier_free_guidance=guidance_scale > 1.0,
          num_videos_per_prompt=num_videos_per_prompt,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
          prompt_attention_mask=prompt_attention_mask,
          negative_prompt_attention_mask=negative_prompt_attention_mask,
          max_sequence_length=max_sequence_length,
          dtype=dtype,
      )

      # 3. Prepare latents
      # Effective batch size
      batch_size = prompt_embeds.shape[0] // 2 if guidance_scale > 1.0 else prompt_embeds.shape[0]
      
      # Prepare generators
      if generator is None:
          generator = jax.random.key(0)
      
      key_latents, key_audio = jax.random.split(generator)

      latents = self.prepare_latents(
           batch_size=batch_size,
           height=height, width=width, num_frames=num_frames,
           noise_scale=noise_scale,
           dtype=dtype,
           generator=key_latents,
           latents=latents,
      )
      
      # 4. Prepare Audio Latents
      audio_channels = getattr(self.transformer, "audio_in_channels", 128)
      
      duration_s = num_frames / frame_rate
      audio_latents_per_second = (
          self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
      )
      audio_num_frames = round(duration_s * audio_latents_per_second)

      audio_latents = self.prepare_audio_latents(
          batch_size=batch_size,
          num_channels_latents=audio_channels,
          audio_latent_length=audio_num_frames,
          noise_scale=noise_scale,
          dtype=dtype,
          generator=key_audio,
          latents=audio_latents,
      )

      # 5. Prepare Timesteps
      sigmas = jnp.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
      
      video_sequence_length = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
      video_sequence_length *= (height // self.vae_spatial_compression_ratio) * (width // self.vae_spatial_compression_ratio)
      
      mu = calculate_shift(
          video_sequence_length,
          self.scheduler.config.get("base_image_seq_len", 1024),
          self.scheduler.config.get("max_image_seq_len", 4096),
          self.scheduler.config.get("base_shift", 0.95),
          self.scheduler.config.get("max_shift", 2.05),
      )
      
      scheduler_state = retrieve_timesteps(
          self.scheduler, 
          self.scheduler.create_state(), 
          num_inference_steps=num_inference_steps,
          sigmas=sigmas,
          shift=mu,
      )
      timesteps = scheduler_state.timesteps

      # 6. Prepare JAX State
      latents_jax = latents
      audio_latents_jax = audio_latents
      prompt_embeds_jax = prompt_embeds
      prompt_attention_mask_jax = prompt_attention_mask
      
      if guidance_scale > 1.0:
          negative_prompt_embeds_jax = negative_prompt_embeds
          negative_prompt_attention_mask_jax = negative_prompt_attention_mask
          prompt_embeds_jax = jnp.concatenate([negative_prompt_embeds_jax, prompt_embeds_jax], axis=0)
          prompt_attention_mask_jax = jnp.concatenate([negative_prompt_attention_mask_jax, prompt_attention_mask_jax], axis=0)
          latents_jax = jnp.concatenate([latents_jax] * 2, axis=0)
          audio_latents_jax = jnp.concatenate([audio_latents_jax] * 2, axis=0)
      
      # GraphDef and State
      graphdef, state = nnx.split(self.transformer)
      
      # 7. Denoising Loop
      connectors_graphdef, connectors_state = nnx.split(self.connectors)
      
      @jax.jit
      def run_connectors(graphdef, state, hidden_states, attention_mask):
           model = nnx.merge(graphdef, state)
           return model(hidden_states, attention_mask)

      video_embeds, audio_embeds = run_connectors(
           connectors_graphdef, connectors_state, prompt_embeds_jax, prompt_attention_mask_jax.astype(jnp.bool_)
      )
      
      for i, t in enumerate(timesteps):
          noise_pred, noise_pred_audio = transformer_forward_pass(
              graphdef, state,
              latents_jax,
              audio_latents_jax,
              t,
              video_embeds,
              audio_embeds,
              prompt_attention_mask_jax,
              prompt_attention_mask_jax,
              guidance_scale > 1.0,
              guidance_scale,
              num_frames,
              height,
              width,
              audio_num_frames,
              frame_rate,
          )

          if guidance_scale > 1.0:
               noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
               noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
               # Audio guidance
               noise_pred_audio_uncond, noise_pred_audio_text = jnp.split(noise_pred_audio, 2, axis=0)
               noise_pred_audio = noise_pred_audio_uncond + guidance_scale * (noise_pred_audio_text - noise_pred_audio_uncond)
               
               latents_step = latents_jax[batch_size:]
               audio_latents_step = audio_latents_jax[batch_size:]
          else:
               latents_step = latents_jax
               audio_latents_step = audio_latents_jax

          # Step
          latents_step, _ = self.scheduler.step(scheduler_state, noise_pred, t, latents_step, return_dict=False)
          audio_latents_step, _ = self.scheduler.step(scheduler_state, noise_pred_audio, t, audio_latents_step, return_dict=False)

          if guidance_scale > 1.0:
               latents_jax = jnp.concatenate([latents_step] * 2, axis=0)
               audio_latents_jax = jnp.concatenate([audio_latents_step] * 2, axis=0)
          else:
               latents_jax = latents_step
               audio_latents_jax = audio_latents_step

      # 8. Decode Latents
      if guidance_scale > 1.0:
          latents_jax = latents_jax[batch_size:]
          audio_latents_jax = audio_latents_jax[batch_size:]

      # Unpack and Denormalize Video
      latents = self._unpack_latents(
          latents_jax, num_frames, height, width, 
          self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
      )
      latents = self._denormalize_latents(
          latents, 
          jnp.array(self.vae.latents_mean), 
          jnp.array(self.vae.latents_std), 
          self.vae.config.scaling_factor
      )
      
      # Denormalize and Unpack Audio (Order important: Denorm THEN Unpack)
      audio_latents = self._denormalize_audio_latents(
          audio_latents_jax,
          jnp.array(self.audio_vae.latents_mean),
          jnp.array(self.audio_vae.latents_std)
      )
      
      num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
      latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

      audio_latents = self._unpack_audio_latents(
          audio_latents, 
          audio_num_frames, 
          num_mel_bins=latent_mel_bins
      )

      if output_type == "latent":
          return LTX2PipelineOutput(frames=latents, audio=audio_latents)

      if getattr(self.vae.config, "timestep_conditioning", False):
          noise = jax.random.normal(generator, latents.shape, dtype=latents.dtype)
          
          if not isinstance(decode_timestep, list):
              decode_timestep = [decode_timestep] * batch_size
          if decode_noise_scale is None:
              decode_noise_scale = decode_timestep
          elif not isinstance(decode_noise_scale, list):
              decode_noise_scale = [decode_noise_scale] * batch_size

          timestep = jnp.array(decode_timestep, dtype=latents.dtype)
          decode_noise_scale = jnp.array(decode_noise_scale, dtype=latents.dtype)[:, None, None, None, None]
          
          latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise
          
          latents = latents.astype(self.vae.dtype)
          video = self.vae.decode(latents, timestep=timestep, return_dict=False)[0]
      else:
          latents = latents.astype(self.vae.dtype)
          video = self.vae.decode(latents, return_dict=False)[0]
      # Post-process video (converts to numpy/PIL)
      # We need to pass numpy to postprocess_video usually, checking if it handles JAX
      video_np = np.array(video)
      video = self.video_processor.postprocess_video(torch.from_numpy(video_np), output_type=output_type)

      # Decode Audio
      audio_latents = audio_latents.astype(self.audio_vae.dtype)
      generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
      audio = self.vocoder(generated_mel_spectrograms)
      
      # Convert audio to numpy
      audio = np.array(audio)

      return LTX2PipelineOutput(frames=video, audio=audio)

@partial(jax.jit, static_argnames=("do_classifier_free_guidance", "guidance_scale", "num_frames", "height", "width", "audio_num_frames", "fps"))
def transformer_forward_pass(
    graphdef,
    state,
    latents,
    audio_latents,
    timestep,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    encoder_attention_mask,
    audio_encoder_attention_mask,
    do_classifier_free_guidance,
    guidance_scale,
    num_frames,
    height,
    width,
    audio_num_frames,
    fps,
):
    transformer = nnx.merge(graphdef, state)
    
    # Expand timestep to batch size
    timestep = jnp.expand_dims(timestep, 0).repeat(latents.shape[0])

    noise_pred, noise_pred_audio = transformer(
        hidden_states=latents,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        audio_hidden_states=audio_latents,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        audio_encoder_attention_mask=audio_encoder_attention_mask,
        fps=fps,
        audio_num_frames=audio_num_frames,
        return_dict=False,
    )
    
    return noise_pred, noise_pred_audio
