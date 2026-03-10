"""
Flax/JAX pipeline for LTX-2 Latent Upsampling.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from typing import Dict, List, Optional, Tuple, Union

from maxdiffusion import max_logging
from ...video_processor import VideoProcessor
from ..pipeline_flax_utils import FlaxDiffusionPipeline


class FlaxLTX2LatentUpsamplePipeline(FlaxDiffusionPipeline):
    def __init__(self, vae, latent_upsampler):
        super().__init__()
        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        # Fallback to defaults if config isn't fully populated in VAE yet
        self.vae_spatial_compression_ratio = getattr(self.vae.config, "spatial_compression_ratio", 32)
        self.vae_temporal_compression_ratio = getattr(self.vae.config, "temporal_compression_ratio", 8)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def _unpack_latents(
        self, latents: jax.Array, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> jax.Array:
        # Assuming packed latents are [B, S, D] -> unpacked to [B, F, H, W, C]
        batch_size = latents.shape[0]
        c = latents.shape[-1] // (patch_size_t * patch_size * patch_size)

        f_p = num_frames // patch_size_t
        h_p = height // patch_size
        w_p = width // patch_size

        latents = jnp.reshape(latents, (batch_size, f_p, h_p, w_p, patch_size_t, patch_size, patch_size, c))
        # Permute dimensions to match target video latent shape: B, F//pt, pt, H//p, p, W//p, p, c
        latents = jnp.transpose(latents, (0, 1, 4, 2, 5, 3, 6, 7))
        latents = jnp.reshape(latents, (batch_size, num_frames, height, width, c))
        return latents

    def _denormalize_latents(
        self, latents: jax.Array, latents_mean: Union[List[float], jax.Array], latents_std: Union[List[float], jax.Array], scaling_factor: float = 1.0
    ) -> jax.Array:
        # Reshape to match latent dimensions
        latents_mean = jnp.reshape(jnp.array(latents_mean), (1, 1, 1, 1, -1))
        latents_std = jnp.reshape(jnp.array(latents_std), (1, 1, 1, 1, -1))
        
        # Snap the mean and std arrays to the exact same device sharding as the latents!
        latents_mean = jax.device_put(latents_mean, latents.sharding)
        latents_std = jax.device_put(latents_std, latents.sharding)
        
        return latents * latents_std / scaling_factor + latents_mean

    def adain_filter_latent(self, latents: jax.Array, reference_latents: jax.Array, factor: float = 1.0) -> jax.Array:
        # Compute stats over the spatial and temporal axes: Frames, Height, Width
        axes = (1, 2, 3)
        r_sd = jnp.std(reference_latents, axis=axes, keepdims=True)
        r_mean = jnp.mean(reference_latents, axis=axes, keepdims=True)

        i_sd = jnp.std(latents, axis=axes, keepdims=True)
        i_mean = jnp.mean(latents, axis=axes, keepdims=True)

        result = ((latents - i_mean) / (i_sd + 1e-5)) * r_sd + r_mean
        
        # JAX equivalent to torch.lerp
        result = latents + factor * (result - latents)
        return result

    def tone_map_latents(self, latents: jax.Array, compression: float) -> jax.Array:
        scale_factor = compression * 0.75
        abs_latents = jnp.abs(latents)

        # Sigmoid-based compression to regularize high-variance latents
        sigmoid_term = jax.nn.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        return latents * scales

    def check_inputs(self, video, height, width, latents, tone_map_compression_ratio):
        if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {self.vae_spatial_compression_ratio}.")

        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` can be provided.")
        if video is None and latents is None:
            raise ValueError("One of `video` or `latents` has to be provided.")

        if not (0 <= tone_map_compression_ratio <= 1):
            raise ValueError("`tone_map_compression_ratio` must be in the range [0, 1]")

    def prepare_latents(
        self,
        params: Union[Dict, FrozenDict],
        video: Optional[np.ndarray],
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        rng: jax.Array,
        latents: Optional[jax.Array] = None,
    ) -> jax.Array:
        if latents is not None:
            if latents.ndim == 3:
                latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
                latent_height = height // self.vae_spatial_compression_ratio
                latent_width = width // self.vae_spatial_compression_ratio
                latents = self._unpack_latents(
                    latents, latent_num_frames, latent_height, latent_width, spatial_patch_size, temporal_patch_size
                )
            return latents

        # If video is provided, pass through VAE encode
        encoder_output = self.vae.encode(video)
        
        if hasattr(encoder_output, "latent_dist"):
            latents = encoder_output.latent_dist.sample(rng)
        else:
            latents = encoder_output.latents

        return latents

    def __call__(
        self,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        video: Optional[np.ndarray] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        latents: Optional[jax.Array] = None,
        latents_normalized: bool = False,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        adain_factor: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        self.check_inputs(video, height, width, latents, tone_map_compression_ratio)

        if video is not None:
            batch_size = 1
            num_frames = len(video)
            if num_frames % self.vae_temporal_compression_ratio != 1:
                num_frames = (num_frames // self.vae_temporal_compression_ratio) * self.vae_temporal_compression_ratio + 1
                video = video[:num_frames]
                max_logger.warning(
                    f"Video length expected to be of the form `k * {self.vae_temporal_compression_ratio} + 1`. Truncating to {num_frames} frames."
                )
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = jnp.array(video, dtype=jnp.float32)
        else:
            batch_size = latents.shape[0]

        # Manage pseudo-random generators
        rng = prng_seed
        rng, vae_rng = jax.random.split(rng)

        latents_supplied = latents is not None
        latents = self.prepare_latents(
            params=params,
            video=video,
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            rng=vae_rng,
            latents=latents,
        )

        if latents_supplied and latents_normalized:
            # Handle both dictionary (FrozenDict) and object-based configs
            vae_config = self.vae.config
            if isinstance(vae_config, (dict, FrozenDict)):
                latents_mean = vae_config.get("latents_mean")
                latents_std = vae_config.get("latents_std")
                scaling_factor = vae_config.get("scaling_factor", 1.0)
            else:
                latents_mean = getattr(vae_config, "latents_mean", None)
                latents_std = getattr(vae_config, "latents_std", None)
                scaling_factor = getattr(vae_config, "scaling_factor", 1.0)
                
            # Fallback in case they are attached to the VAE object directly (like PyTorch)
            if latents_mean is None:
                latents_mean = getattr(self.vae, "latents_mean")
            if latents_std is None:
                latents_std = getattr(self.vae, "latents_std")

            latents = self._denormalize_latents(
                latents, 
                latents_mean, 
                latents_std, 
                scaling_factor
            )
        # Run Latent Upsampler model
        latents_upsampled = self.latent_upsampler.apply({'params': params['latent_upsampler']}, latents)

        if adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
        else:
            latents = latents_upsampled

        if tone_map_compression_ratio > 0.0:
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)

        if output_type == "latent":
            return (latents,) if not return_dict else {"frames": latents}

        # ---------------------------------------------
        # Decode Latents to Video
        # ---------------------------------------------
        vae_config = self.vae.config
        if isinstance(vae_config, (dict, FrozenDict)):
            timestep_conditioning = vae_config.get("timestep_conditioning", False)
        else:
            timestep_conditioning = getattr(vae_config, "timestep_conditioning", False)

        if timestep_conditioning:
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, latents.shape, dtype=latents.dtype)

            decode_noise_scale = decode_noise_scale if decode_noise_scale is not None else decode_timestep
            timestep = jnp.array([decode_timestep] * batch_size, dtype=latents.dtype)
            
            # Broadcast scalar scale to 5D [B, 1, 1, 1, 1]
            scale_array = jnp.array([decode_noise_scale] * batch_size, dtype=latents.dtype).reshape(-1, 1, 1, 1, 1)

            latents = (1 - scale_array) * latents + scale_array * noise
        else:
            timestep = None

        # NNX module inherently holds its own weights, no params dict needed!
        if timestep is not None:
            video = self.vae.decode(latents, timestep=timestep)
        else:
            video = self.vae.decode(latents)
        
        if hasattr(video, "sample"):
            video = video.sample

        import torch
        import numpy as np
        import jax.numpy as jnp

        if video.dtype == jnp.bfloat16:
            video = video.astype(jnp.float32)
        
        # Convert JAX array to PyTorch tensor AND rearrange dimensions:
        # (Batch, Frames, Height, Width, Channels) -> (Batch, Channels, Frames, Height, Width)
        video = torch.from_numpy(np.array(video)).permute(0, 4, 1, 2, 3)
                
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)
        return {"frames": video}