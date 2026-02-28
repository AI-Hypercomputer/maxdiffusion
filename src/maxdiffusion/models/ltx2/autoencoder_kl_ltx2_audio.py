"""Audio VAE model for MaxDiffusion."""

from typing import Tuple, Optional, Set
import math

from flax import nnx
import jax
import jax.numpy as jnp
import flax

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..vae_flax import FlaxDiagonalGaussianDistribution


LATENT_DOWNSAMPLE_FACTOR = 4

# Define a custom variable type for non-trainable buffers
class Buffer(nnx.Variable):
    pass

@flax.struct.dataclass
class FlaxDecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`jnp.ndarray` of shape `(batch_size, time, freq, num_channels)`):
            The decoded output sample from the last layer of the model.
    """
    sample: jnp.ndarray


@flax.struct.dataclass
class FlaxAutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`FlaxDiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `FlaxDiagonalGaussianDistribution`.
            `FlaxDiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """
    latent_dist: FlaxDiagonalGaussianDistribution


class FlaxLTX2AudioCausalConv(nnx.Module):
    """A causal 2D convolution that pads asymmetrically along the causal axis."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        strides: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        causality_axis: str = "height",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.causality_axis = causality_axis
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.strides = strides
        
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding="VALID", # Manual padding applied
            kernel_dilation=(dilation, dilation),
            feature_group_count=groups,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        # x is (batch, height/time, width/freq, channels)
        dilation = self.dilation
        kernel_size = self.kernel_size
        
        pad_h = (kernel_size - 1) * dilation
        pad_w = (kernel_size - 1) * dilation
        
        # Flax padding format: ((top, bottom), (left, right))
        if self.causality_axis == "none":
             padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        elif self.causality_axis in ("width", "width-compatibility"):
             padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w, 0))
        elif self.causality_axis == "height":
             padding = ((pad_h, 0), (pad_w // 2, pad_w - pad_w // 2))
        else:
             raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        # Apply padding (Batch, Time, Freq, Channels)
        x = jnp.pad(x, ((0, 0), padding[0], padding[1], (0, 0)))
        return self.conv(x)


class FlaxLTX2AudioPixelNorm(nnx.Module):
    """Per-pixel (per-location) RMS normalization layer."""
    def __init__(self, dim: int = -1, epsilon: float = 1e-8, dtype: jnp.dtype = jnp.float32):
        self.dim = dim
        self.epsilon = epsilon

    def __call__(self, x):
        mean_sq = jnp.mean(jnp.square(x), axis=self.dim, keepdims=True)
        rms = jnp.sqrt(mean_sq + self.epsilon)
        return x / rms


class FlaxLTX2AudioAttnBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.norm_type = norm_type
        
        if self.norm_type == "group":
            self.norm = nnx.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-6, dtype=dtype, rngs=rngs)
        elif self.norm_type == "pixel":
            self.norm = FlaxLTX2AudioPixelNorm(dim=-1, epsilon=1e-6, dtype=dtype)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")
            
        self.q = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=dtype, rngs=rngs)
        self.k = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=dtype, rngs=rngs)
        self.v = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=dtype, rngs=rngs)
        self.proj_out = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=dtype, rngs=rngs)

    def __call__(self, x):
        residual = x
        h_ = self.norm(x)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        batch, height, width, channels = q.shape
        
        # Reshape to (batch, seq_len, channels)
        q = q.reshape(batch, height * width, channels)
        k = k.reshape(batch, height * width, channels)
        v = v.reshape(batch, height * width, channels)
        
        # Attention
        scale = 1.0 / math.sqrt(channels)
        attn = jnp.matmul(q, jnp.swapaxes(k, 1, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        h_ = jnp.matmul(attn, v)
        h_ = h_.reshape(batch, height, width, channels)
        
        h_ = self.proj_out(h_)
        return residual + h_


class FlaxLTX2AudioResnetBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str = "height",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.conv_shortcut = conv_shortcut
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        
        if self.norm_type == "group":
            self.norm1 = nnx.GroupNorm(num_groups=32, num_channels=self.in_channels, epsilon=1e-6, dtype=dtype, rngs=rngs)
        elif self.norm_type == "pixel":
            self.norm1 = FlaxLTX2AudioPixelNorm(dim=-1, epsilon=1e-6, dtype=dtype)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv1 = FlaxLTX2AudioCausalConv(
                 in_features=self.in_channels,
                 out_features=self.out_channels, 
                 kernel_size=3, 
                 causality_axis=self.causality_axis,
                 dtype=dtype,
                 rngs=rngs
             )
        else:
             self.conv1 = nnx.Conv(
                 in_features=self.in_channels,
                 out_features=self.out_channels, 
                 kernel_size=(3, 3), 
                 padding="SAME", 
                 dtype=dtype,
                 rngs=rngs
             )

        if temb_channels > 0:
            self.temb_proj = nnx.Linear(temb_channels, self.out_channels, dtype=dtype, rngs=rngs)
        else:
            self.temb_proj = None

        if self.norm_type == "group":
            self.norm2 = nnx.GroupNorm(num_groups=32, num_channels=self.out_channels, epsilon=1e-6, dtype=dtype, rngs=rngs)
        elif self.norm_type == "pixel":
            self.norm2 = FlaxLTX2AudioPixelNorm(dim=-1, epsilon=1e-6, dtype=dtype)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        self.dropout_layer = nnx.Dropout(dropout, rngs=rngs)

        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv2 = FlaxLTX2AudioCausalConv(
                 in_features=self.out_channels,
                 out_features=self.out_channels, 
                 kernel_size=3, 
                 causality_axis=self.causality_axis,
                 dtype=dtype,
                 rngs=rngs
             )
        else:
             self.conv2 = nnx.Conv(
                 in_features=self.out_channels,
                 out_features=self.out_channels, 
                 kernel_size=(3, 3), 
                 padding="SAME", 
                 dtype=dtype,
                 rngs=rngs
             )

        if self.in_channels != self.out_channels:
            kernel_sz = 3 if self.conv_shortcut else 1
            if self.causality_axis is not None and self.causality_axis != "none":
                self.conv_shortcut_layer = FlaxLTX2AudioCausalConv(
                    in_features=self.in_channels,
                    out_features=self.out_channels, 
                    kernel_size=kernel_sz, 
                    causality_axis=self.causality_axis,
                    dtype=dtype,
                    rngs=rngs
                )
            else:
                self.conv_shortcut_layer = nnx.Conv(
                    in_features=self.in_channels,
                    out_features=self.out_channels, 
                    kernel_size=(kernel_sz, kernel_sz), 
                    padding="SAME" if kernel_sz == 3 else "VALID", 
                    dtype=dtype,
                    rngs=rngs
                )
        else:
            self.conv_shortcut_layer = None

    def __call__(self, x, temb=None, train: bool = False):
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(jax.nn.silu(temb))[:, None, None, :]

        h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.dropout_layer(h, deterministic=not train)
        h = self.conv2(h)

        if self.conv_shortcut_layer is not None:
            x = self.conv_shortcut_layer(x)

        return x + h


class FlaxLTX2AudioDownsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool = True,
        causality_axis: Optional[str] = "height",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        
        if self.with_conv:
             self.conv = nnx.Conv(
                 in_features=self.in_channels,
                 out_features=self.in_channels, 
                 kernel_size=(3, 3), 
                 strides=(2, 2), 
                 padding="VALID",
                 dtype=dtype,
                 rngs=rngs
             )

    def __call__(self, x):
        if self.with_conv:
             if self.causality_axis == "none":
                 pad = ((0, 1), (0, 1))
             elif self.causality_axis == "width":
                 pad = ((0, 1), (2, 0))
             elif self.causality_axis == "height":
                 pad = ((2, 0), (0, 1))
             elif self.causality_axis == "width-compatibility":
                 pad = ((0, 1), (1, 0))
             else:
                 raise ValueError(f"Invalid `causality_axis` {self.causality_axis}")
            
             x = jnp.pad(x, ((0, 0), pad[0], pad[1], (0, 0)), mode="constant", constant_values=0)
             x = self.conv(x)
        else:
             x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class FlaxLTX2AudioUpsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool = True,
        causality_axis: Optional[str] = "height",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        
        if self.with_conv:
            if self.causality_axis is not None and self.causality_axis != "none":
                self.conv = FlaxLTX2AudioCausalConv(
                    in_features=self.in_channels,
                    out_features=self.in_channels,
                    kernel_size=3,
                    causality_axis=self.causality_axis,
                    dtype=dtype,
                    rngs=rngs
                )
            else:
                self.conv = nnx.Conv(
                    in_features=self.in_channels,
                    out_features=self.in_channels,
                    kernel_size=(3, 3),
                    padding="SAME",
                    dtype=dtype,
                    rngs=rngs
                )

    def __call__(self, x):
        batch, height, width, channels = x.shape
        x = jax.image.resize(x, shape=(batch, height * 2, width * 2, channels), method="nearest")
        
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis is None or self.causality_axis == "none":
                pass
            elif self.causality_axis == "height":
                x = x[:, 1:, :, :]
            elif self.causality_axis == "width":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width-compatibility":
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
        return x


class FlaxLTX2AudioAudioPatchifier:
    def __init__(
        self,
        patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self._patch_size = (1, patch_size, patch_size)

    def patchify(self, audio_latents: jnp.ndarray) -> jnp.ndarray:
        # Input: (batch, time, freq, channels) -> Output: (batch, time, channels * freq)
        batch, time, freq, channels = audio_latents.shape
        # PyTorch equivalent: permute(0, 2, 1, 3).reshape(...) -> NHWC -> (B, F, T, C) -> (B, T, C*F)
        # Actually PyTorch (B,C,T,F).permute(0,2,1,3) -> (B,T,C,F) -> reshape(B,T,C*F).
        # We are already in NHWC (B, T, F, C). So we permute to (B, T, C, F) to match PyTorch flattening.
        x = jnp.transpose(audio_latents, (0, 1, 3, 2))
        return x.reshape(batch, time, channels * freq)

    def unpatchify(self, patched_latents: jnp.ndarray, channels: int, freq: int) -> jnp.ndarray:
        # Input: (batch, time, channels * freq)
        batch, time, _ = patched_latents.shape
        x = patched_latents.reshape(batch, time, channels, freq)
        # Revert to NHWC (B, T, F, C)
        return jnp.transpose(x, (0, 1, 3, 2))


class FlaxLTX2AudioEncoder(nnx.Module):
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Set[int]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        double_z: bool = True,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.base_channels = base_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.resolution = resolution
        self.mid_block_add_attention = mid_block_add_attention
        self.double_z = double_z
        self.latent_channels = latent_channels
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        
        base_block_channels = self.base_channels
        curr_res = self.resolution
        
        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv_in = FlaxLTX2AudioCausalConv(in_channels, base_block_channels, kernel_size=3, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        else:
             self.conv_in = nnx.Conv(in_channels, base_block_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
             
        self.down_stages = nnx.List()
        block_in = base_block_channels
        
        for level in range(len(self.ch_mult)):
            block_out = self.base_channels * self.ch_mult[level]
            stage_blocks = nnx.List()
            stage_attns = nnx.List()
            
            for _ in range(self.num_res_blocks):
                stage_blocks.append(FlaxLTX2AudioResnetBlock(block_in, block_out, temb_channels=0, dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs))
                block_in = block_out
                
                if self.attn_resolutions and curr_res in self.attn_resolutions:
                    stage_attns.append(FlaxLTX2AudioAttnBlock(block_in, norm_type=self.norm_type, dtype=dtype, rngs=rngs))
                else:
                    stage_attns.append(None)
            
            downsample = None
            if level != len(self.ch_mult) - 1:
                downsample = FlaxLTX2AudioDownsample(block_in, with_conv=True, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
                curr_res = curr_res // 2
            
            self.down_stages.append({"blocks": stage_blocks, "attns": stage_attns, "downsample": downsample})

        self.mid_block1 = FlaxLTX2AudioResnetBlock(block_in, block_in, temb_channels=0, dropout=dropout, norm_type=norm_type, causality_axis=causality_axis, dtype=dtype, rngs=rngs)
        self.mid_attn = FlaxLTX2AudioAttnBlock(block_in, norm_type=norm_type, dtype=dtype, rngs=rngs) if mid_block_add_attention else None
        self.mid_block2 = FlaxLTX2AudioResnetBlock(block_in, block_in, temb_channels=0, dropout=dropout, norm_type=norm_type, causality_axis=causality_axis, dtype=dtype, rngs=rngs)
        
        z_channels = 2 * self.latent_channels if self.double_z else self.latent_channels
        if self.norm_type == "group":
            self.norm_out = nnx.GroupNorm(num_groups=32, num_channels=block_in, epsilon=1e-6, dtype=dtype, rngs=rngs)
        elif self.norm_type == "pixel":
            self.norm_out = FlaxLTX2AudioPixelNorm(dim=-1, epsilon=1e-6, dtype=dtype)
            
        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv_out = FlaxLTX2AudioCausalConv(block_in, z_channels, kernel_size=3, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        else:
             self.conv_out = nnx.Conv(block_in, z_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)

    def __call__(self, x, train: bool = False):
        h = self.conv_in(x)
             
        for stage in self.down_stages:
            for block, attn in zip(stage["blocks"], stage["attns"]):
                h = block(h, train=train)
                if attn is not None:
                    h = attn(h)
            if stage["downsample"] is not None:
                h = stage["downsample"](h)

        h = self.mid_block1(h, train=train)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, train=train)
        
        h = self.norm_out(h)
        h = jax.nn.silu(h)
        h = self.conv_out(h)
             
        return h


class FlaxLTX2AudioDecoder(nnx.Module):
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Set[int]] = None,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        mel_bins: Optional[int] = 64,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.resolution = resolution
        self.ch_mult = ch_mult
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.mel_bins = mel_bins
        
        base_block_channels = self.base_channels * self.ch_mult[-1]
        
        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv_in = FlaxLTX2AudioCausalConv(latent_channels, base_block_channels, kernel_size=3, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        else:
             self.conv_in = nnx.Conv(latent_channels, base_block_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
             
        self.mid_block1 = FlaxLTX2AudioResnetBlock(base_block_channels, base_block_channels, temb_channels=0, dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        self.mid_attn = FlaxLTX2AudioAttnBlock(base_block_channels, norm_type=self.norm_type, dtype=dtype, rngs=rngs) if mid_block_add_attention else None
        self.mid_block2 = FlaxLTX2AudioResnetBlock(base_block_channels, base_block_channels, temb_channels=0, dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        
        self.up_stages = nnx.List()
        curr_res = self.resolution // (2 ** (len(self.ch_mult) - 1))
        block_in = base_block_channels
        
        for level in reversed(range(len(self.ch_mult))):
            block_out = self.base_channels * self.ch_mult[level]
            stage_blocks = nnx.List()
            stage_attns = nnx.List()
            
            for _ in range(self.num_res_blocks + 1):
                stage_blocks.append(FlaxLTX2AudioResnetBlock(block_in, block_out, temb_channels=0, dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs))
                block_in = block_out
                
                if self.attn_resolutions and curr_res in self.attn_resolutions:
                    stage_attns.append(FlaxLTX2AudioAttnBlock(block_in, norm_type=self.norm_type, dtype=dtype, rngs=rngs))
                else:
                    stage_attns.append(None)
            
            upsample = None
            if level != 0:
                upsample = FlaxLTX2AudioUpsample(block_in, with_conv=True, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
                curr_res *= 2
            
            self.up_stages.append({"blocks": stage_blocks, "attns": stage_attns, "upsample": upsample})
        
        if self.norm_type == "group":
            self.norm_out = nnx.GroupNorm(num_groups=32, num_channels=block_in, epsilon=1e-6, dtype=dtype, rngs=rngs)
        elif self.norm_type == "pixel":
            self.norm_out = FlaxLTX2AudioPixelNorm(dim=-1, epsilon=1e-6, dtype=dtype)
            
        if self.causality_axis is not None and self.causality_axis != "none":
             self.conv_out = FlaxLTX2AudioCausalConv(block_in, self.output_channels, kernel_size=3, causality_axis=self.causality_axis, dtype=dtype, rngs=rngs)
        else:
             self.conv_out = nnx.Conv(block_in, self.output_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)

    def __call__(self, z, target_frames=None, target_mel_bins=None, train: bool = False):
        h = self.conv_in(z)
             
        h = self.mid_block1(h, train=train)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, train=train)
        
        for stage in self.up_stages:
            for block, attn in zip(stage["blocks"], stage["attns"]):
                h = block(h, train=train)
                if attn is not None:
                    h = attn(h)
            if stage["upsample"] is not None:
                h = stage["upsample"](h)
        
        h = self.norm_out(h)
        h = jax.nn.silu(h)
        h = self.conv_out(h)

        if target_frames is not None and target_mel_bins is not None:
            batch, current_time, current_freq, _ = h.shape
            h = h[:, :min(current_time, target_frames), :min(current_freq, target_mel_bins), :self.output_channels]
            
            pad_time = max(target_frames - h.shape[1], 0)
            pad_freq = max(target_mel_bins - h.shape[2], 0)
            
            if pad_time > 0 or pad_freq > 0:
                h = jnp.pad(h, ((0, 0), (0, pad_time), (0, pad_freq), (0, 0)))
                
        return h


class FlaxAutoencoderKLLTX2Audio(nnx.Module, ConfigMixin):
    """
    LTX2 audio VAE wrapper handling normalization, patchification, and latent sampling.
    Operates in NHWC format (Batch, Time, Freq, Channels).
    """
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 2,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int, ...]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        norm_type: str = "pixel",
        causality_axis: Optional[str] = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
        double_z: bool = True,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.base_channels = base_channels
        self.double_z = double_z
        self.latent_channels = latent_channels
        self.causality_axis = causality_axis
        self.out_ch = output_channels
        self.mel_bins = mel_bins
        
        attn_res_set = set(attn_resolutions) if attn_resolutions else None

        self.encoder = FlaxLTX2AudioEncoder(
            base_channels=base_channels,
            output_channels=output_channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res_set,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            ch_mult=ch_mult,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            double_z=double_z,
            dtype=dtype,
            rngs=rngs
        )
        
        self.decoder = FlaxLTX2AudioDecoder(
            base_channels=base_channels,
            output_channels=output_channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res_set,
            resolution=resolution,
            latent_channels=latent_channels,
            ch_mult=ch_mult,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            mel_bins=mel_bins,
            dtype=dtype,
            rngs=rngs
        )
        
        self.patchifier = FlaxLTX2AudioAudioPatchifier(
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            is_causal=is_causal
        )
        
        # Buffers for normalization statistics
        self.latents_mean = Buffer(jnp.zeros((base_channels,), dtype=dtype))
        self.latents_std = Buffer(jnp.ones((base_channels,), dtype=dtype))

    def _normalize_latents(self, h: jnp.ndarray) -> jnp.ndarray:
        if self.double_z:
            means, logvars = jnp.split(h, 2, axis=-1)
        else:
            means = h
            logvars = None

        batch, time, freq, channels = means.shape
        
        # Normalize means ONLY
        means_patched = self.patchifier.patchify(means) 
        means_normalized = (means_patched - self.latents_mean[...]) / self.latents_std[...]
        means_normalized = self.patchifier.unpatchify(means_normalized, channels, freq)

        if logvars is not None:
            # Leave logvars unmodified as per PyTorch implementation
            return jnp.concatenate([means_normalized, logvars], axis=-1)
        return means_normalized

    def _denormalize_latents(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, int, int, int]]:
        batch, time, freq, channels = z.shape
        
        # Denormalize latents (which are just means)
        patched_z = self.patchifier.patchify(z)
        denorm_patched_z = (patched_z * self.latents_std[...]) + self.latents_mean[...]
        z = self.patchifier.unpatchify(denorm_patched_z, channels, freq)

        target_frames = time * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis is not None and self.causality_axis != "none":
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = (batch, target_frames, self.mel_bins, self.out_ch)
        return z, target_shape

    def encode(self, x: jnp.ndarray, return_dict: bool = True, train: bool = False):
        h = self.encoder(x, train=train)
        h = self._normalize_latents(h)  # Apply normalization here
        posterior = FlaxDiagonalGaussianDistribution(h)
        
        if not return_dict:
            return (posterior,)
        return FlaxAutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: jnp.ndarray, return_dict: bool = True, train: bool = False):
        z, target_shape = self._denormalize_latents(z)
        target_frames, target_mel_bins = target_shape[1], target_shape[2]
        
        decoded = self.decoder(z, target_frames=target_frames, target_mel_bins=target_mel_bins, train=train)

        if not return_dict:
            return (decoded,)
        return FlaxDecoderOutput(sample=decoded)

    def __call__(self, sample, sample_posterior=False, return_dict: bool = True, train: bool = False, rng=None):
        posterior = self.encode(sample, return_dict=True, train=train).latent_dist
        
        if sample_posterior:
            if rng is None:
                raise ValueError("rng must be provided for sampling")
            z = posterior.sample(rng)
        else:
            z = posterior.mode()
            
        dec = self.decode(z, return_dict=return_dict, train=train)
        return dec
