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

import math
from typing import Dict, Any, Optional
import numpy as np
import jax
import jax.numpy as jnp

from maxdiffusion import multihost_dataloading, max_logging


# ============================================================================
# Helper Functions
# ============================================================================


def get_wan_dimension(
    config,
    pipeline,
    config_key: str,
    pipeline_path: str = None,
    default_value: Any = None
) -> Any:
    """
    Get dimension for WAN model with override priority:
    1. Config override (synthetic_override_{config_key}) - for height, width, num_frames
    2. Pipeline path (exact path specified by caller)
    3. Config default
    4. Hardcoded default

    Args:
        config: Configuration object
        pipeline: WAN Pipeline object
        config_key: Key to look up in config
        pipeline_path: Exact dotted path in pipeline (e.g., 'transformer.config.in_channels')
        default_value: Fallback value if not found elsewhere
    """
    # Check overrides for height, width, num_frames (WAN-specific)
    if config_key in ['height', 'width', 'num_frames']:
        override_key = f'synthetic_override_{config_key}'
        try:
            value = getattr(config, override_key)
            if value is not None:
                if jax.process_index() == 0:
                    max_logging.log(f"[WAN] Using override {config_key}: {value}")
                return value
        except (AttributeError, ValueError):
            pass  # Override not set, continue to pipeline/config

    # Check pipeline using exact path if provided
    if pipeline is not None and pipeline_path:
        try:
            # Navigate the dotted path (e.g., 'transformer.config.in_channels')
            value = pipeline
            for attr in pipeline_path.split('.'):
                value = getattr(value, attr)

            if value is not None:
                if jax.process_index() == 0:
                    max_logging.log(f"[WAN] Using {config_key} from pipeline.{pipeline_path}: {value}")
                return value
        except AttributeError:
            pass  # Path not available in pipeline

    # Check config - use try/except because config raises ValueError instead of AttributeError
    try:
        value = getattr(config, config_key)
        if jax.process_index() == 0:
            max_logging.log(f"[WAN] Using {config_key} from config: {value}")
        return value
    except (AttributeError, ValueError):
        pass  # Key not in config, use default

    # Use default
    if jax.process_index() == 0:
        max_logging.log(f"[WAN] Using default {config_key}: {default_value}")
    return default_value


def get_flux_dimension(
    config,
    pipeline,
    config_key: str,
    pipeline_path: str = None,
    default_value: Any = None
) -> Any:
    """
    Get dimension for FLUX model with override priority:
    1. Pipeline path (exact path specified by caller)
    2. Config default
    3. Hardcoded default

    Note: FLUX does not support override flags

    Args:
        config: Configuration object
        pipeline: FLUX Pipeline object
        config_key: Key to look up in config
        pipeline_path: Exact dotted path in pipeline (e.g., 'vae_scale_factor')
        default_value: Fallback value if not found elsewhere
    """
    # FLUX does not check overrides - load directly from pipeline/config

    # Check pipeline using exact path if provided
    if pipeline is not None and pipeline_path:
        try:
            # Navigate the dotted path (e.g., 'vae_scale_factor')
            value = pipeline
            for attr in pipeline_path.split('.'):
                value = getattr(value, attr)

            if value is not None:
                if jax.process_index() == 0:
                    max_logging.log(f"[FLUX] Using {config_key} from pipeline.{pipeline_path}: {value}")
                return value
        except AttributeError:
            pass  # Path not available in pipeline

    # Check config - use try/except because config raises ValueError instead of AttributeError
    try:
        value = getattr(config, config_key)
        if jax.process_index() == 0:
            max_logging.log(f"[FLUX] Using {config_key} from config: {value}")
        return value
    except (AttributeError, ValueError):
        pass  # Key not in config, use default

    # Use default
    if jax.process_index() == 0:
        max_logging.log(f"[FLUX] Using default {config_key}: {default_value}")
    return default_value


def log_synthetic_config(model_name: str, dimensions: Dict[str, Any], per_host_batch_size: int, is_training: bool, num_samples: Optional[int]):
    """Log synthetic data configuration."""
    if jax.process_index() == 0:
        info = [
            "=" * 60,
            f"{model_name.upper()} Synthetic Data Iterator Configuration:",
            f"  Per-host batch size: {per_host_batch_size}",
            f"  Mode: {'Training' if is_training else 'Evaluation'}",
            f"  Samples per iteration: {num_samples if num_samples else 'Infinite'}",
        ]
        for key, value in dimensions.items():
            info.append(f"  {key}: {value}")
        info.append("=" * 60)
        max_logging.log("\n".join(info))


# ============================================================================
# Synthetic Data Source and Iterator
# ============================================================================


class SyntheticDataSource:
    """Wrapper for synthetic data that provides iterator interface."""

    def __init__(self, generate_fn, num_samples, seed):
        self.generate_fn = generate_fn
        self.num_samples = num_samples
        self.seed = seed
        self.current_step = 0
        self.rng_key = jax.random.key(seed)

    def __iter__(self):
        self.current_step = 0
        self.rng_key = jax.random.key(self.seed)
        return self

    def __next__(self):
        if self.num_samples is not None and self.current_step >= self.num_samples:
            raise StopIteration

        self.rng_key, step_key = jax.random.split(self.rng_key)
        data = self.generate_fn(step_key)
        self.current_step += 1
        return data

    def as_numpy_iterator(self):
        return iter(self)


# ============================================================================
# WAN Model Synthetic Data Generator
# ============================================================================


def _generate_wan_sample(rng_key: jax.Array, dimensions: Dict[str, Any], is_training: bool) -> Dict[str, np.ndarray]:
    """Generate a single batch of synthetic data for WAN model."""
    keys = jax.random.split(rng_key, 3)

    per_host_batch_size = dimensions['per_host_batch_size']

    # Generate latents: (batch, channels, frames, height, width)
    latents_shape = (
        per_host_batch_size,
        dimensions['num_channels_latents'],
        dimensions['num_latent_frames'],
        dimensions['latent_height'],
        dimensions['latent_width']
    )
    latents = jax.random.normal(keys[0], shape=latents_shape, dtype=jnp.float32)

    # Generate encoder hidden states: (batch, seq_len, embed_dim)
    encoder_hidden_states_shape = (
        per_host_batch_size,
        dimensions['max_sequence_length'],
        dimensions['text_embed_dim']
    )
    encoder_hidden_states = jax.random.normal(keys[1], shape=encoder_hidden_states_shape, dtype=jnp.float32)

    data = {
        'latents': np.array(latents),
        'encoder_hidden_states': np.array(encoder_hidden_states),
    }

    # For evaluation, also generate timesteps
    if not is_training:
        timesteps = jax.random.randint(
            keys[2],
            shape=(per_host_batch_size,),
            minval=0,
            maxval=dimensions['num_train_timesteps'],
            dtype=jnp.int32
        )
        data['timesteps'] = np.array(timesteps)

    return data


def _make_wan_synthetic_iterator(config, mesh, global_batch_size, pipeline, is_training, num_samples):
    """Create synthetic data iterator for WAN model."""
    per_host_batch_size = global_batch_size // jax.process_count()

    # Initialize dimensions - explicitly specify pipeline paths for WAN model
    height = get_wan_dimension(
        config, pipeline, 'height',
        pipeline_path=None,  # Not in pipeline, use config/override
        default_value=480
    )
    width = get_wan_dimension(
        config, pipeline, 'width',
        pipeline_path=None,  # Not in pipeline, use config/override
        default_value=832
    )
    num_frames = get_wan_dimension(
        config, pipeline, 'num_frames',
        pipeline_path=None,  # Not in pipeline, use config/override
        default_value=81
    )

    # WAN-specific dimensions from transformer config
    max_sequence_length = get_wan_dimension(
        config, pipeline, 'max_sequence_length',
        pipeline_path='transformer.config.rope_max_seq_len',
        default_value=512
    )
    text_embed_dim = get_wan_dimension(
        config, pipeline, 'text_embed_dim',
        pipeline_path='transformer.config.text_dim',
        default_value=4096
    )
    num_channels_latents = get_wan_dimension(
        config, pipeline, 'num_channels_latents',
        pipeline_path='transformer.config.in_channels',
        default_value=16
    )

    # VAE scale factors from pipeline attributes
    vae_scale_factor_spatial = get_wan_dimension(
        config, pipeline, 'vae_scale_factor_spatial',
        pipeline_path='vae_scale_factor_spatial',
        default_value=8
    )
    vae_scale_factor_temporal = get_wan_dimension(
        config, pipeline, 'vae_scale_factor_temporal',
        pipeline_path='vae_scale_factor_temporal',
        default_value=4
    )

    # Calculate latent dimensions
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial

    # Get num_train_timesteps from scheduler
    num_train_timesteps = get_wan_dimension(
        config, pipeline, 'num_train_timesteps',
        pipeline_path='scheduler.config.num_train_timesteps',
        default_value=1000
    )
    # Fallback to scheduler.num_train_timesteps if config doesn't exist
    if pipeline is not None and hasattr(pipeline, 'scheduler') and num_train_timesteps == 1000:
        try:
            num_train_timesteps = pipeline.scheduler.num_train_timesteps
            if jax.process_index() == 0:
                max_logging.log(f"Using num_train_timesteps from pipeline.scheduler: {num_train_timesteps}")
        except AttributeError:
            pass

    dimensions = {
        'per_host_batch_size': per_host_batch_size,
        'height': height,
        'width': width,
        'num_frames': num_frames,
        'num_latent_frames': num_latent_frames,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'max_sequence_length': max_sequence_length,
        'text_embed_dim': text_embed_dim,
        'num_channels_latents': num_channels_latents,
        'vae_scale_factor_spatial': vae_scale_factor_spatial,
        'vae_scale_factor_temporal': vae_scale_factor_temporal,
        'num_train_timesteps': num_train_timesteps,
    }

    log_synthetic_config('WAN', dimensions, per_host_batch_size, is_training, num_samples)

    # Create generate function with dimensions bound
    def generate_fn(rng_key):
        return _generate_wan_sample(rng_key, dimensions, is_training)

    data_source = SyntheticDataSource(generate_fn, num_samples, config.seed)
    return multihost_dataloading.MultiHostDataLoadIterator(data_source, mesh)


# ============================================================================
# FLUX Model Synthetic Data Generator
# ============================================================================


def _generate_flux_sample(rng_key: jax.Array, dimensions: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Generate a single batch of synthetic data for FLUX model."""
    keys = jax.random.split(rng_key, 4)

    per_host_batch_size = dimensions['per_host_batch_size']
    latent_height = dimensions['latent_height']
    latent_width = dimensions['latent_width']
    latent_seq_len = dimensions['latent_seq_len']

    # Generate pixel values (packed latents) - should be float16 to match trainer
    pixel_values_shape = (per_host_batch_size, latent_seq_len, dimensions['packed_latent_dim'])
    pixel_values = jax.random.normal(keys[0], shape=pixel_values_shape, dtype=jnp.float16)

    # Generate text embedding IDs (position encodings)
    input_ids_shape = (per_host_batch_size, dimensions['max_sequence_length'], 3)
    input_ids = jax.random.normal(keys[1], shape=input_ids_shape, dtype=jnp.float32)

    # Generate text embeddings (T5)
    text_embeds_shape = (per_host_batch_size, dimensions['max_sequence_length'], dimensions['t5_embed_dim'])
    text_embeds = jax.random.normal(keys[2], shape=text_embeds_shape, dtype=jnp.float32)

    # Generate pooled prompt embeddings (CLIP)
    prompt_embeds_shape = (per_host_batch_size, dimensions['pooled_embed_dim'])
    prompt_embeds = jax.random.normal(keys[3], shape=prompt_embeds_shape, dtype=jnp.float32)

    # Generate image position IDs - matching pipeline.prepare_latent_image_ids
    # Create base img_ids for single sample (without batch dimension)
    img_ids_base = jnp.zeros((latent_height, latent_width, 3), dtype=jnp.float16)
    # Channel 0 stays 0
    # Channel 1 = height indices
    img_ids_base = img_ids_base.at[..., 1].set(jnp.arange(latent_height)[:, None])
    # Channel 2 = width indices
    img_ids_base = img_ids_base.at[..., 2].set(jnp.arange(latent_width)[None, :])

    # Reshape to (latent_seq_len, 3)
    img_ids_base = img_ids_base.reshape(latent_seq_len, 3)

    # Tile for batch dimension
    img_ids = jnp.tile(img_ids_base[None, ...], (per_host_batch_size, 1, 1))

    return {
        'pixel_values': np.array(pixel_values),
        'input_ids': np.array(input_ids),
        'text_embeds': np.array(text_embeds),
        'prompt_embeds': np.array(prompt_embeds),
        'img_ids': np.array(img_ids),
    }


def _make_flux_synthetic_iterator(config, mesh, global_batch_size, pipeline, is_training, num_samples):
    """Create synthetic data iterator for FLUX model."""
    per_host_batch_size = global_batch_size // jax.process_count()

    # Initialize dimensions - explicitly specify pipeline paths for FLUX model
    resolution = get_flux_dimension(
        config, pipeline, 'resolution',
        pipeline_path=None,  # Not in pipeline, use config
        default_value=512
    )
    max_sequence_length = get_flux_dimension(
        config, pipeline, 'max_sequence_length',
        pipeline_path=None,  # Not in pipeline, use config
        default_value=512
    )
    t5_embed_dim = get_flux_dimension(
        config, pipeline, 't5_embed_dim',
        pipeline_path='text_encoder_2.config.d_model',  # T5 model dimension
        default_value=4096
    )
    pooled_embed_dim = get_flux_dimension(
        config, pipeline, 'pooled_embed_dim',
        pipeline_path='text_encoder.config.projection_dim',  # CLIP projection dimension
        default_value=768
    )
    vae_scale_factor = get_flux_dimension(
        config, pipeline, 'vae_scale_factor',
        pipeline_path='vae_scale_factor',  # Direct pipeline attribute
        default_value=8
    )

    # Calculate packed latent dimensions
    latent_height = math.ceil(resolution // (vae_scale_factor * 2))
    latent_width = math.ceil(resolution // (vae_scale_factor * 2))
    latent_seq_len = latent_height * latent_width
    packed_latent_dim = 64  # 16 channels * 2 * 2 packing

    dimensions = {
        'per_host_batch_size': per_host_batch_size,
        'max_sequence_length': max_sequence_length,
        't5_embed_dim': t5_embed_dim,
        'pooled_embed_dim': pooled_embed_dim,
        'resolution': resolution,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'latent_seq_len': latent_seq_len,
        'packed_latent_dim': packed_latent_dim,
    }

    log_synthetic_config('FLUX', dimensions, per_host_batch_size, is_training, num_samples)

    # Create generate function with dimensions bound
    def generate_fn(rng_key):
        return _generate_flux_sample(rng_key, dimensions)

    data_source = SyntheticDataSource(generate_fn, num_samples, config.seed)
    return multihost_dataloading.MultiHostDataLoadIterator(data_source, mesh)


# ============================================================================
# Public API
# ============================================================================


def make_synthetic_iterator(config, mesh, global_batch_size, pipeline=None, is_training=True):
    """
    Create a synthetic data iterator for the specified model.

    Args:
        config: Configuration object with model_name
        mesh: JAX mesh for sharding
        global_batch_size: Total batch size across all devices
        pipeline: Optional pipeline object to extract dimensions from
        is_training: Whether this is for training or evaluation

    Returns:
        MultiHostDataLoadIterator wrapping the synthetic data source
    """
    num_samples = getattr(config, 'synthetic_num_samples', None)

    try:
        model_name = getattr(config, 'model_name', None)
        if model_name in ('wan2.1', 'wan2.2'):
            return _make_wan_synthetic_iterator(config, mesh, global_batch_size, pipeline, is_training, num_samples)
    except (AttributeError, ValueError):
        pass
    try:
        model_name = getattr(config, 'flux_name', None)
        if model_name in ('flux', 'flux-dev', 'flux-schnell'):
            return _make_flux_synthetic_iterator(config, mesh, global_batch_size, pipeline, is_training, num_samples)
    except (AttributeError, ValueError):
        pass

    raise ValueError(
            "No synthetic iterator implemented for model."
            "Supported models: wan2.1, wan2.2, flux, flux-dev, flux-schnell"
        )
