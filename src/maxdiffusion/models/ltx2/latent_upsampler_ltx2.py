"""
Flax/JAX implementation of the LTX-2 Latent Upsampler.
"""

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

RATIONAL_RESAMPLER_SCALE_MAPPING = {
    0.75: (3, 4),
    1.5: (3, 2),
    2.0: (2, 1),
    4.0: (4, 1),
}


class ResBlock(nn.Module):
    channels: int
    mid_channels: Optional[int] = None
    dims: int = 3

    @nn.compact
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        mid_channels = self.mid_channels if self.mid_channels is not None else self.channels
        
        kernel_size = (3,) * self.dims
        padding = ((1, 1),) * self.dims

        residual = hidden_states

        hidden_states = nn.Conv(mid_channels, kernel_size=kernel_size, padding=padding, name="conv1")(hidden_states)
        hidden_states = nn.GroupNorm(num_groups=32, name="norm1")(hidden_states)
        hidden_states = nn.silu(hidden_states)

        hidden_states = nn.Conv(self.channels, kernel_size=kernel_size, padding=padding, name="conv2")(hidden_states)
        hidden_states = nn.GroupNorm(num_groups=32, name="norm2")(hidden_states)
        
        # FIX: Removed the SiLU! Latents must be allowed to stay negative.
        hidden_states = hidden_states + residual

        return hidden_states

class PixelShuffleND(nn.Module):
    dims: int
    upscale_factors: Tuple[int, ...] = (2, 2, 2)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.dims == 3:
            p1, p2, p3 = self.upscale_factors[:3]
            b, d, h, w, c_p = x.shape
            c = c_p // (p1 * p2 * p3)
            x = jnp.reshape(x, (b, d, h, w, c, p1, p2, p3))
            x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
            x = jnp.reshape(x, (b, d * p1, h * p2, w * p3, c))
            return x
        elif self.dims == 2:
            p1, p2 = self.upscale_factors[:2]
            b, h, w, c_p = x.shape
            c = c_p // (p1 * p2)
            x = jnp.reshape(x, (b, h, w, c, p1, p2))
            x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
            x = jnp.reshape(x, (b, h * p1, w * p2, c))
            return x
        elif self.dims == 1:
            p1 = self.upscale_factors[0]
            b, f, h, w, c_p = x.shape
            c = c_p // p1
            x = jnp.reshape(x, (b, f, h, w, c, p1))
            x = jnp.transpose(x, (0, 1, 5, 2, 3, 4))
            x = jnp.reshape(x, (b, f * p1, h, w, c))
            return x


class BlurDownsample(nn.Module):
    dims: int
    stride: int
    kernel_size: int = 5

    def setup(self):
        if self.dims not in (2, 3):
            raise ValueError(f"`dims` must be either 2 or 3 but is {self.dims}")
        if self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError(f"`kernel_size` must be an odd number >= 3 but is {self.kernel_size}")

        k = jnp.array([math.comb(self.kernel_size - 1, i) for i in range(self.kernel_size)], dtype=jnp.float32)
        k2d = jnp.outer(k, k)
        k2d = k2d / jnp.sum(k2d)
        self.kernel = jnp.reshape(k2d, (self.kernel_size, self.kernel_size, 1, 1))

    def __call__(self, x: jax.Array) -> jax.Array:
        pad = self.kernel_size // 2

        if self.dims == 2:
            c = x.shape[-1]
            weight = jnp.tile(self.kernel, (1, 1, 1, c))
            x = jax.lax.conv_general_dilated(
                lhs=x, 
                rhs=weight,
                window_strides=(self.stride, self.stride),
                padding=((pad, pad), (pad, pad)),
                feature_group_count=c,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
        else:
            b, f, h, w, c = x.shape
            x = jnp.reshape(x, (b * f, h, w, c))
            
            weight = jnp.tile(self.kernel, (1, 1, 1, c))
            x = jax.lax.conv_general_dilated(
                lhs=x, 
                rhs=weight,
                window_strides=(self.stride, self.stride),
                padding=((pad, pad), (pad, pad)),
                feature_group_count=c, 
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
            
            h2, w2 = x.shape[1], x.shape[2]
            x = jnp.reshape(x, (b, f, h2, w2, c))
            
        return x


class SpatialRationalResampler(nn.Module):
    mid_channels: int = 1024
    scale: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.scale not in RATIONAL_RESAMPLER_SCALE_MAPPING:
            raise ValueError(f"scale {self.scale} not supported.")
        num, den = RATIONAL_RESAMPLER_SCALE_MAPPING[self.scale]

        x = nn.Conv((num**2) * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), name="Conv_0")(x)
        x = PixelShuffleND(dims=2, upscale_factors=(num, num))(x)
        x = BlurDownsample(dims=2, stride=den)(x)
        return x


class LTX2LatentUpsamplerModel(nn.Module):
    in_channels: int = 128
    mid_channels: int = 1024
    num_blocks_per_stage: int = 4
    dims: int = 3
    spatial_upsample: bool = True
    temporal_upsample: bool = False
    rational_spatial_scale: Optional[float] = 2.0

    @nn.compact
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        b, f, h, w, c = hidden_states.shape

        if self.dims == 2:
            hidden_states = jnp.reshape(hidden_states, (b * f, h, w, c))

            hidden_states = nn.Conv(self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), name="initial_conv")(hidden_states)
            hidden_states = nn.GroupNorm(num_groups=32, name="initial_norm")(hidden_states)
            hidden_states = nn.silu(hidden_states)

            for i in range(self.num_blocks_per_stage):
                hidden_states = ResBlock(channels=self.mid_channels, dims=2, name=f"ResBlock_{i}")(hidden_states)

            if self.spatial_upsample:
                if self.rational_spatial_scale is not None:
                    hidden_states = SpatialRationalResampler(self.mid_channels, self.rational_spatial_scale, name="SpatialRationalResampler_0")(hidden_states)
                else:
                    hidden_states = nn.Conv(4 * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), name="Conv_0")(hidden_states)
                    hidden_states = PixelShuffleND(dims=2)(hidden_states)

            for i in range(self.num_blocks_per_stage):
                hidden_states = ResBlock(channels=self.mid_channels, dims=2, name=f"post_upsample_ResBlock_{i}")(hidden_states)

            hidden_states = nn.Conv(self.in_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), name="final_conv")(hidden_states)
            
            h2, w2 = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states = jnp.reshape(hidden_states, (b, f, h2, w2, self.in_channels))

        else:
            hidden_states = nn.Conv(self.mid_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name="initial_conv")(hidden_states)
            hidden_states = nn.GroupNorm(num_groups=32, name="initial_norm")(hidden_states)
            hidden_states = nn.silu(hidden_states)

            for i in range(self.num_blocks_per_stage):
                hidden_states = ResBlock(channels=self.mid_channels, dims=3, name=f"ResBlock_{i}")(hidden_states)

            if self.temporal_upsample:
                hidden_states = nn.Conv(2 * self.mid_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name="Conv_0")(hidden_states)
                hidden_states = PixelShuffleND(dims=1)(hidden_states)
                hidden_states = hidden_states[:, 1:, :, :, :] 
            else:
                hidden_states = jnp.reshape(hidden_states, (b * f, h, w, self.mid_channels))
                if self.rational_spatial_scale is not None:
                    hidden_states = SpatialRationalResampler(self.mid_channels, self.rational_spatial_scale, name="SpatialRationalResampler_0")(hidden_states)
                else:
                    hidden_states = nn.Conv(4 * self.mid_channels, kernel_size=(3, 3), padding=((1, 1), (1, 1)), name="Conv_0")(hidden_states)
                    hidden_states = PixelShuffleND(dims=2)(hidden_states)
                h2, w2 = hidden_states.shape[1], hidden_states.shape[2]
                hidden_states = jnp.reshape(hidden_states, (b, f, h2, w2, self.mid_channels))

            for i in range(self.num_blocks_per_stage):
                hidden_states = ResBlock(channels=self.mid_channels, dims=3, name=f"post_upsample_ResBlock_{i}")(hidden_states)

            hidden_states = nn.Conv(self.in_channels, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name="final_conv")(hidden_states)

        return hidden_states