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
from flax import nnx
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import qwix

from ...utils import logging
from ...schedulers import FlaxFlowMatchScheduler
from ...models.ltx2.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from ...models.ltx2.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from ...models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from ...models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
from ...video_processor import VideoProcessor
from .ltx2_pipeline_utils import encode_video
from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated

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
        # Placeholder/Default config construction if not loading from checkpoint directly
        ltx2_config = {}

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
         # Placeholder for explicit weight loading
         pass

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


def get_dummy_ltx2_inputs(config, pipeline, batch_size):
    # 1. Latents
    latents = pipeline.prepare_latents(
        batch_size=batch_size,
        height=config.resolution,
        width=config.resolution,
        num_frames=getattr(config, "num_frames", 121),
    )
    
    # 2. Audio Latents
    audio_latents = pipeline.prepare_audio_latents(
        batch_size=batch_size,
        audio_latent_length=8,
    )
    
    # 3. Embeddings
    text_encoder_dim = getattr(pipeline.transformer, "cross_attention_dim", 4096)
    encoder_hidden_states = jax.random.normal(jax.random.key(0), (batch_size, 128, text_encoder_dim))
    
    audio_context_dim = getattr(pipeline.transformer, "audio_cross_attention_dim", 2048)
    audio_encoder_hidden_states = jax.random.normal(jax.random.key(0), (batch_size, 128, audio_context_dim))

    timesteps = jnp.array([0] * batch_size, dtype=jnp.int32)
    
    encoder_attention_mask = jnp.ones((batch_size, 128))
    audio_encoder_attention_mask = jnp.ones((batch_size, 128))
    
    return (latents, audio_latents, timesteps, encoder_hidden_states, audio_encoder_hidden_states, encoder_attention_mask, audio_encoder_attention_mask)


class LTX2Pipeline:
  """
  Pipeline for LTX-2.
  """

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

    def __init__(
        self,
        scheduler: FlaxFlowMatchScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Any, # Placeholder for Gemma3
        tokenizer: AutoTokenizer,
        connectors: LTX2AudioVideoGemmaTextEncoder,
        transformer: LTX2VideoTransformer3DModel,
    ):
        self.scheduler = scheduler
        self.vae = vae
        self.audio_vae = audio_vae
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
        
        # Initialize video processor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        
        self.tokenizer_max_length = getattr(self.tokenizer, "model_max_length", 1024)

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch.device],
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Packs and normalizes text encoder hidden states.
        """
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype

        # Create padding mask
        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        if padding_side == "right":
            mask = token_indices < sequence_lengths[:, None]
        elif padding_side == "left":
            start_indices = seq_len - sequence_lengths[:, None]
            mask = token_indices >= start_indices
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = mask[:, :, None, None]

        masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
        num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states = normalized_hidden_states * scale_factor

        normalized_hidden_states = normalized_hidden_states.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
        normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
        return normalized_hidden_states

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or torch.device("cpu")
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        if self.text_encoder is not None:
             text_encoder_outputs = self.text_encoder(
                input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
            )
             text_encoder_hidden_states = text_encoder_outputs.hidden_states
             text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        else:
            # Mock hidden states
            hidden_dim = 1024
            num_layers = 2
            text_encoder_hidden_states = torch.zeros(
                (batch_size, max_sequence_length, hidden_dim, num_layers), device=device, dtype=dtype or torch.float32
            )

        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        if dtype is not None:
            prompt_embeds = prompt_embeds.to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
                device=device,
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
                device=device,
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
             raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
             raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
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
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean
    
    @staticmethod
    def _create_noised_state(
        latents: torch.Tensor, noise_scale: float, generator: Optional[torch.Generator] = None
    ):
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised_latents = noise_scale * noise + (1 - noise_scale) * latents
        return noised_latents

    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: Optional[int] = None, patch_size_t: Optional[int] = None
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = latent_mel_bins // patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            latents = latents.transpose(1, 2).flatten(2, 3)
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
             if latents.ndim == 5:
                latents_mean = torch.as_tensor(np.array(self.vae.latents_mean), device=latents.device)
                latents_std = torch.as_tensor(np.array(self.vae.latents_std), device=latents.device)
                scaling_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 1.0
                
                latents = self._normalize_latents(latents, latents_mean, latents_std, scaling_factor)
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
             if latents.ndim != 3:
                raise ValueError("Unexpected latents shape")
             latents = self._create_noised_state(latents, noise_scale, generator)
             return latents.to(device=device, dtype=dtype)
        
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        return latents

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 4:
                latents = self._pack_audio_latents(latents)
            if latents.ndim != 3:
                 raise ValueError("Unexpected audio latents shape")
            
            latents_mean = torch.as_tensor(np.array(self.audio_vae.latents_mean), device=latents.device)
            latents_std = torch.as_tensor(np.array(self.audio_vae.latents_std), device=latents.device)

            latents = self._normalize_audio_latents(latents, latents_mean, latents_std)
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        shape = (batch_size, num_channels_latents, audio_latent_length, latent_mel_bins)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_audio_latents(latents)
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
        guidance_scale: float = 3.0,
        noise_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        dtype: Optional[torch.dtype] = torch.float32,
        output_type: str = "pil",
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
        
        latents = self.prepare_latents(
             batch_size=batch_size,
             height=height, width=width, num_frames=num_frames,
             noise_scale=noise_scale,
             dtype=dtype,
             generator=generator,
             latents=latents,
        )
        
        # 4. Prepare Audio Latents
        audio_channels = getattr(self.transformer, "audio_in_channels", 128)
        audio_latents = self.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=audio_channels,
            audio_latent_length=8, # Arbitrary small length
            dtype=dtype,
            generator=generator,
            noise_scale=noise_scale,
        )

        # 5. Prepare Timesteps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare JAX State
        latents_jax = jnp.array(latents.cpu().numpy())
        audio_latents_jax = jnp.array(audio_latents.cpu().numpy())
        prompt_embeds_jax = jnp.array(prompt_embeds.cpu().numpy())
        prompt_attention_mask_jax = jnp.array(prompt_attention_mask.cpu().numpy())
        
        if guidance_scale > 1.0:
            negative_prompt_embeds_jax = jnp.array(negative_prompt_embeds.cpu().numpy())
            negative_prompt_attention_mask_jax = jnp.array(negative_prompt_attention_mask.cpu().numpy())
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
        
        for i, t in enumerate(tqdm(timesteps)):
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
                guidance_scale
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

            # Step (Dummy Euler)
            latents_jax = latents_step - noise_pred
            
            if guidance_scale > 1.0:
                 latents_jax = jnp.concatenate([latents_jax] * 2, axis=0)

        return latents_jax

@partial(jax.jit, static_argnames=("do_classifier_free_guidance", "guidance_scale"))
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
):
    transformer = nnx.merge(graphdef, state)
    
    # 1. Compute Embeddings
    temb = transformer.time_embed(timestep)
    temb_audio = transformer.audio_time_embed(timestep)
    
    temb_ca_scale_shift = transformer.av_cross_attn_video_scale_shift(timestep)
    temb_ca_audio_scale_shift = transformer.av_cross_attn_audio_scale_shift(timestep)
    temb_ca_gate = transformer.av_cross_attn_video_a2v_gate(timestep)
    temb_ca_audio_gate = transformer.av_cross_attn_audio_v2a_gate(timestep)
    
    noise_pred, noise_pred_audio = transformer(
        hidden_states=latents,
        audio_hidden_states=audio_latents,
        encoder_hidden_states=encoder_hidden_states,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        temb=temb,
        temb_audio=temb_audio,
        temb_ca_scale_shift=temb_ca_scale_shift,
        temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
        temb_ca_gate=temb_ca_gate,
        temb_ca_audio_gate=temb_ca_audio_gate,
        video_rotary_emb=None, # To be implemented
        audio_rotary_emb=None,
        ca_video_rotary_emb=None,
        ca_audio_rotary_emb=None,
        encoder_attention_mask=encoder_attention_mask,
        audio_encoder_attention_mask=audio_encoder_attention_mask,
    )
    
    return noise_pred, noise_pred_audio
