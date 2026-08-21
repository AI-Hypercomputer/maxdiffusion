"""
Copyright 2026 Google LLC

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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


class NNXUpsample2D(nnx.Module):
  """2D Nearest-neighbor Upsample + Conv layer in NNX."""

  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    out_channels = out_channels or in_channels
    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    batch, height, width, channels = x.shape
    x = jnp.broadcast_to(x[:, :, None, :, None, :], (batch, height, 2, width, 2, channels))
    x = jnp.reshape(x, (batch, height * 2, width * 2, channels))
    return self.conv(x)


class NNXDownsample2D(nnx.Module):
  """2D Downsample layer with asymmetric padding in NNX."""

  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    out_channels = out_channels or in_channels
    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    pad_width = ((0, 0), (0, 1), (0, 1), (0, 0))
    x = jnp.pad(x, pad_width)
    return self.conv(x)


class NNXResnetBlock2D(nnx.Module):
  """2D ResNet Block with GroupNorm and SiLU activations in NNX."""

  def __init__(
      self,
      in_channels: int,
      out_channels: Optional[int] = None,
      groups: int = 32,
      use_conv_shortcut: Optional[bool] = None,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    out_channels = out_channels or in_channels
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.norm1 = nnx.GroupNorm(
        num_groups=groups,
        num_features=in_channels,
        epsilon=1e-6,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.conv1 = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.norm2 = nnx.GroupNorm(
        num_groups=groups,
        num_features=out_channels,
        epsilon=1e-6,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.conv2 = nnx.Conv(
        in_features=out_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    use_shortcut = (in_channels != out_channels) if use_conv_shortcut is None else use_conv_shortcut
    if use_shortcut:
      self.conv_shortcut = nnx.Conv(
          in_features=in_channels,
          out_features=out_channels,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          rngs=rngs,
          dtype=dtype,
          param_dtype=param_dtype,
      )
    else:
      self.conv_shortcut = None

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = self.conv_shortcut(x) if self.conv_shortcut is not None else x
    h = self.norm1(x)
    h = nnx.silu(h)
    h = self.conv1(h)
    h = self.norm2(h)
    h = nnx.silu(h)
    h = self.conv2(h)
    return h + residual


class NNXAttentionBlock(nnx.Module):
  """Self-Attention block with GroupNorm in NNX."""

  def __init__(
      self,
      channels: int,
      groups: int = 32,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.channels = channels
    self.group_norm = nnx.GroupNorm(
        num_groups=groups,
        num_features=channels,
        epsilon=1e-6,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.to_q = nnx.Linear(
        in_features=channels,
        out_features=channels,
        use_bias=True,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.to_k = nnx.Linear(
        in_features=channels,
        out_features=channels,
        use_bias=True,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.to_v = nnx.Linear(
        in_features=channels,
        out_features=channels,
        use_bias=True,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.to_out = nnx.Linear(
        in_features=channels,
        out_features=channels,
        use_bias=True,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x
    b, h, w, c = x.shape
    h_states = self.group_norm(x)
    h_flat = h_states.reshape((b, h * w, c))

    q = self.to_q(h_flat)
    k = self.to_k(h_flat)
    v = self.to_v(h_flat)

    scale = 1.0 / math.sqrt(c)
    attn_weights = jnp.einsum("bqc,bkc->bqk", q * scale, k)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    out = jnp.einsum("bqk,bkc->bqc", attn_weights, v)
    out = self.to_out(out)
    out = out.reshape((b, h, w, c))
    return out + residual


class NNXUNetMidBlock2D(nnx.Module):
  """Mid-Block module in NNX with resnets and attention."""

  def __init__(
      self,
      in_channels: int,
      groups: int = 32,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.resnets_0 = NNXResnetBlock2D(
        in_channels=in_channels,
        out_channels=in_channels,
        groups=groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.attentions_0 = NNXAttentionBlock(
        channels=in_channels,
        groups=groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.resnets_1 = NNXResnetBlock2D(
        in_channels=in_channels,
        out_channels=in_channels,
        groups=groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.resnets_0(x)
    x = self.attentions_0(x)
    x = self.resnets_1(x)
    return x


class NNXDownEncoderBlock2D(nnx.Module):
  """Down-Encoder block containing ResNet layers and an optional Downsampler in NNX."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_layers: int = 2,
      groups: int = 32,
      add_downsample: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    resnets = []
    for i in range(num_layers):
      in_ch = in_channels if i == 0 else out_channels
      resnets.append(
          NNXResnetBlock2D(
              in_channels=in_ch,
              out_channels=out_channels,
              groups=groups,
              rngs=rngs,
              dtype=dtype,
              param_dtype=param_dtype,
          )
      )
    self.resnets = nnx.List(resnets)

    if add_downsample:
      self.downsamplers_0 = NNXDownsample2D(
          in_channels=out_channels,
          out_channels=out_channels,
          rngs=rngs,
          dtype=dtype,
          param_dtype=param_dtype,
      )
    else:
      self.downsamplers_0 = None

  def __call__(self, x: jax.Array) -> jax.Array:
    for resnet in self.resnets:
      x = resnet(x)
    if self.downsamplers_0 is not None:
      x = self.downsamplers_0(x)
    return x


class NNXUpDecoderBlock2D(nnx.Module):
  """Up-Decoder block containing ResNet layers and an optional Upsampler in NNX."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_layers: int = 3,
      groups: int = 32,
      add_upsample: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    resnets = []
    for i in range(num_layers):
      in_ch = in_channels if i == 0 else out_channels
      resnets.append(
          NNXResnetBlock2D(
              in_channels=in_ch,
              out_channels=out_channels,
              groups=groups,
              rngs=rngs,
              dtype=dtype,
              param_dtype=param_dtype,
          )
      )
    self.resnets = nnx.List(resnets)

    if add_upsample:
      self.upsamplers_0 = NNXUpsample2D(
          in_channels=out_channels,
          out_channels=out_channels,
          rngs=rngs,
          dtype=dtype,
          param_dtype=param_dtype,
      )
    else:
      self.upsamplers_0 = None

  def __call__(self, x: jax.Array) -> jax.Array:
    for resnet in self.resnets:
      x = resnet(x)
    if self.upsamplers_0 is not None:
      x = self.upsamplers_0(x)
    return x


class NNXEncoder(nnx.Module):
  """Complete VAE Encoder in NNX."""

  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 32,
      block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
      layers_per_block: int = 2,
      norm_num_groups: int = 32,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.conv_in = nnx.Conv(
        in_features=in_channels,
        out_features=block_out_channels[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    down_blocks = []
    output_ch = block_out_channels[0]
    for i, ch in enumerate(block_out_channels):
      input_ch = output_ch
      output_ch = ch
      is_final = i == len(block_out_channels) - 1
      down_blocks.append(
          NNXDownEncoderBlock2D(
              in_channels=input_ch,
              out_channels=output_ch,
              num_layers=layers_per_block,
              groups=norm_num_groups,
              add_downsample=not is_final,
              rngs=rngs,
              dtype=dtype,
              param_dtype=param_dtype,
          )
      )
    self.down_blocks = nnx.List(down_blocks)

    self.mid_block = NNXUNetMidBlock2D(
        in_channels=block_out_channels[-1],
        groups=norm_num_groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.conv_norm_out = nnx.GroupNorm(
        num_groups=norm_num_groups,
        num_features=block_out_channels[-1],
        epsilon=1e-6,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.conv_out = nnx.Conv(
        in_features=block_out_channels[-1],
        out_features=2 * out_channels,  # double_z for Gaussian distribution moments
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.conv_in(x)
    for block in self.down_blocks:
      x = block(x)
    x = self.mid_block(x)
    x = self.conv_norm_out(x)
    x = nnx.silu(x)
    x = self.conv_out(x)
    return x


class NNXDecoder(nnx.Module):
  """Complete VAE Decoder in NNX."""

  def __init__(
      self,
      in_channels: int = 32,
      out_channels: int = 3,
      block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
      layers_per_block: int = 3,
      norm_num_groups: int = 32,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    reversed_channels = list(reversed(block_out_channels))
    self.conv_in = nnx.Conv(
        in_features=in_channels,
        out_features=reversed_channels[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.mid_block = NNXUNetMidBlock2D(
        in_channels=reversed_channels[0],
        groups=norm_num_groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    up_blocks = []
    output_ch = reversed_channels[0]
    for i, ch in enumerate(reversed_channels):
      input_ch = output_ch
      output_ch = ch
      is_final = i == len(reversed_channels) - 1
      up_blocks.append(
          NNXUpDecoderBlock2D(
              in_channels=input_ch,
              out_channels=output_ch,
              num_layers=layers_per_block,
              groups=norm_num_groups,
              add_upsample=not is_final,
              rngs=rngs,
              dtype=dtype,
              param_dtype=param_dtype,
          )
      )
    self.up_blocks = nnx.List(up_blocks)

    self.conv_norm_out = nnx.GroupNorm(
        num_groups=norm_num_groups,
        num_features=reversed_channels[-1],
        epsilon=1e-6,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.conv_out = nnx.Conv(
        in_features=reversed_channels[-1],
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.conv_in(x)
    x = self.mid_block(x)
    for block in self.up_blocks:
      x = block(x)
    x = self.conv_norm_out(x)
    x = nnx.silu(x)
    x = self.conv_out(x)
    return x


class NNXAutoencoderKLFlux2(nnx.Module):
  """Full FLUX.2-Klein Variational Autoencoder (VAE) in Flax NNX."""

  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 3,
      latent_channels: int = 32,
      block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
      layers_per_block: int = 2,
      norm_num_groups: int = 32,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    rngs = rngs or nnx.Rngs(0)
    self.latent_channels = latent_channels
    self.dtype = dtype

    self.encoder = NNXEncoder(
        in_channels=in_channels,
        out_channels=latent_channels,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.quant_conv = nnx.Conv(
        in_features=2 * latent_channels,
        out_features=2 * latent_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.post_quant_conv = nnx.Conv(
        in_features=latent_channels,
        out_features=latent_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.decoder = NNXDecoder(
        in_channels=latent_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block + 1,  # 3 resnet blocks in decoder
        norm_num_groups=norm_num_groups,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def encode(self, sample: jax.Array) -> jax.Array:
    """Encodes image tensor of shape (B, 3, H, W) to mode latents of shape (B, 32, H/8, W/8)."""
    # Transpose to channels last (B, H, W, 3)
    x = jnp.transpose(sample, (0, 2, 3, 1))
    h = self.encoder(x)
    moments = self.quant_conv(h)
    # Extract mean / mode (first latent_channels)
    mean, _ = jnp.split(moments, 2, axis=-1)  # (B, H/8, W/8, 32)
    # Transpose back to (B, 32, H/8, W/8)
    return jnp.transpose(mean, (0, 3, 1, 2))

  def decode(self, latents: jax.Array) -> jax.Array:
    """Decodes latent tensor of shape (B, 32, H/8, W/8) to image tensor of shape (B, 3, H, W)."""
    # Transpose to channels last (B, H/8, W/8, 32)
    z = jnp.transpose(latents, (0, 2, 3, 1))
    h = self.post_quant_conv(z)
    img = self.decoder(h)
    # Transpose back to (B, 3, H, W)
    return jnp.transpose(img, (0, 3, 1, 2))


def load_and_convert_flux2klein_nnx_vae_weights(
    safetensors_path: str,
    nnx_vae: NNXAutoencoderKLFlux2,
    dtype: Optional[jnp.dtype] = None,
    pt_state_dict: Optional[dict] = None,
):
  """Directly loads and maps PyTorch safetensors into NNXAutoencoderKLFlux2 State."""
  from safetensors.numpy import load_file

  if pt_state_dict is None:
    pt_state_dict = load_file(safetensors_path)

  target_dtype = dtype if dtype is not None else jnp.float32

  def get_pt_tensor(key, is_norm=False):
    tensor = pt_state_dict[key]
    leaf_dtype = jnp.float32 if is_norm else target_dtype
    return jnp.array(tensor, dtype=leaf_dtype)

  def get_conv_kernel(key):
    return jnp.array(pt_state_dict[key].transpose(2, 3, 1, 0), dtype=target_dtype)

  def get_linear_kernel(key):
    return jnp.array(pt_state_dict[key].T, dtype=target_dtype)

  flat_state = dict(nnx.to_flat_state(nnx.state(nnx_vae, nnx.Param)))

  def set_val(var, val):
    var[...] = val

  # =========================================================================
  # 1. ENCODER
  # =========================================================================
  set_val(flat_state[("encoder", "conv_in", "kernel")], get_conv_kernel("encoder.conv_in.weight"))
  set_val(flat_state[("encoder", "conv_in", "bias")], get_pt_tensor("encoder.conv_in.bias"))

  for b_idx in range(4):
    down_block_pt = f"encoder.down_blocks.{b_idx}"
    for r_idx in range(2):
      res_pt = f"{down_block_pt}.resnets.{r_idx}"
      res_path = ("encoder", "down_blocks", b_idx, "resnets", r_idx)

      set_val(flat_state[res_path + ("norm1", "scale")], get_pt_tensor(f"{res_pt}.norm1.weight", is_norm=True))
      set_val(flat_state[res_path + ("norm1", "bias")], get_pt_tensor(f"{res_pt}.norm1.bias", is_norm=True))
      set_val(flat_state[res_path + ("conv1", "kernel")], get_conv_kernel(f"{res_pt}.conv1.weight"))
      set_val(flat_state[res_path + ("conv1", "bias")], get_pt_tensor(f"{res_pt}.conv1.bias"))

      set_val(flat_state[res_path + ("norm2", "scale")], get_pt_tensor(f"{res_pt}.norm2.weight", is_norm=True))
      set_val(flat_state[res_path + ("norm2", "bias")], get_pt_tensor(f"{res_pt}.norm2.bias", is_norm=True))
      set_val(flat_state[res_path + ("conv2", "kernel")], get_conv_kernel(f"{res_pt}.conv2.weight"))
      set_val(flat_state[res_path + ("conv2", "bias")], get_pt_tensor(f"{res_pt}.conv2.bias"))

      shortcut_key = f"{res_pt}.conv_shortcut.weight"
      if shortcut_key in pt_state_dict:
        set_val(flat_state[res_path + ("conv_shortcut", "kernel")], get_conv_kernel(shortcut_key))
        set_val(flat_state[res_path + ("conv_shortcut", "bias")], get_pt_tensor(f"{res_pt}.conv_shortcut.bias"))

    if b_idx < 3:
      ds_pt = f"{down_block_pt}.downsamplers.0.conv"
      ds_path = ("encoder", "down_blocks", b_idx, "downsamplers_0", "conv")
      set_val(flat_state[ds_path + ("kernel",)], get_conv_kernel(f"{ds_pt}.weight"))
      set_val(flat_state[ds_path + ("bias",)], get_pt_tensor(f"{ds_pt}.bias"))

  # Encoder Mid Block
  for r_idx in [0, 1]:
    res_pt = f"encoder.mid_block.resnets.{r_idx}"
    res_path = ("encoder", "mid_block", f"resnets_{r_idx}")
    set_val(flat_state[res_path + ("norm1", "scale")], get_pt_tensor(f"{res_pt}.norm1.weight", is_norm=True))
    set_val(flat_state[res_path + ("norm1", "bias")], get_pt_tensor(f"{res_pt}.norm1.bias", is_norm=True))
    set_val(flat_state[res_path + ("conv1", "kernel")], get_conv_kernel(f"{res_pt}.conv1.weight"))
    set_val(flat_state[res_path + ("conv1", "bias")], get_pt_tensor(f"{res_pt}.conv1.bias"))
    set_val(flat_state[res_path + ("norm2", "scale")], get_pt_tensor(f"{res_pt}.norm2.weight", is_norm=True))
    set_val(flat_state[res_path + ("norm2", "bias")], get_pt_tensor(f"{res_pt}.norm2.bias", is_norm=True))
    set_val(flat_state[res_path + ("conv2", "kernel")], get_conv_kernel(f"{res_pt}.conv2.weight"))
    set_val(flat_state[res_path + ("conv2", "bias")], get_pt_tensor(f"{res_pt}.conv2.bias"))

  attn_pt = "encoder.mid_block.attentions.0"
  attn_path = ("encoder", "mid_block", "attentions_0")
  set_val(flat_state[attn_path + ("group_norm", "scale")], get_pt_tensor(f"{attn_pt}.group_norm.weight", is_norm=True))
  set_val(flat_state[attn_path + ("group_norm", "bias")], get_pt_tensor(f"{attn_pt}.group_norm.bias", is_norm=True))
  set_val(flat_state[attn_path + ("to_q", "kernel")], get_linear_kernel(f"{attn_pt}.to_q.weight"))
  set_val(flat_state[attn_path + ("to_q", "bias")], get_pt_tensor(f"{attn_pt}.to_q.bias"))
  set_val(flat_state[attn_path + ("to_k", "kernel")], get_linear_kernel(f"{attn_pt}.to_k.weight"))
  set_val(flat_state[attn_path + ("to_k", "bias")], get_pt_tensor(f"{attn_pt}.to_k.bias"))
  set_val(flat_state[attn_path + ("to_v", "kernel")], get_linear_kernel(f"{attn_pt}.to_v.weight"))
  set_val(flat_state[attn_path + ("to_v", "bias")], get_pt_tensor(f"{attn_pt}.to_v.bias"))
  set_val(flat_state[attn_path + ("to_out", "kernel")], get_linear_kernel(f"{attn_pt}.to_out.0.weight"))
  set_val(flat_state[attn_path + ("to_out", "bias")], get_pt_tensor(f"{attn_pt}.to_out.0.bias"))

  set_val(flat_state[("encoder", "conv_norm_out", "scale")], get_pt_tensor("encoder.conv_norm_out.weight", is_norm=True))
  set_val(flat_state[("encoder", "conv_norm_out", "bias")], get_pt_tensor("encoder.conv_norm_out.bias", is_norm=True))
  set_val(flat_state[("encoder", "conv_out", "kernel")], get_conv_kernel("encoder.conv_out.weight"))
  set_val(flat_state[("encoder", "conv_out", "bias")], get_pt_tensor("encoder.conv_out.bias"))

  # =========================================================================
  # 2. QUANT CONV & POST QUANT CONV
  # =========================================================================
  set_val(flat_state[("quant_conv", "kernel")], get_conv_kernel("quant_conv.weight"))
  set_val(flat_state[("quant_conv", "bias")], get_pt_tensor("quant_conv.bias"))
  set_val(flat_state[("post_quant_conv", "kernel")], get_conv_kernel("post_quant_conv.weight"))
  set_val(flat_state[("post_quant_conv", "bias")], get_pt_tensor("post_quant_conv.bias"))

  # =========================================================================
  # 3. DECODER
  # =========================================================================
  set_val(flat_state[("decoder", "conv_in", "kernel")], get_conv_kernel("decoder.conv_in.weight"))
  set_val(flat_state[("decoder", "conv_in", "bias")], get_pt_tensor("decoder.conv_in.bias"))

  for r_idx in [0, 1]:
    res_pt = f"decoder.mid_block.resnets.{r_idx}"
    res_path = ("decoder", "mid_block", f"resnets_{r_idx}")
    set_val(flat_state[res_path + ("norm1", "scale")], get_pt_tensor(f"{res_pt}.norm1.weight", is_norm=True))
    set_val(flat_state[res_path + ("norm1", "bias")], get_pt_tensor(f"{res_pt}.norm1.bias", is_norm=True))
    set_val(flat_state[res_path + ("conv1", "kernel")], get_conv_kernel(f"{res_pt}.conv1.weight"))
    set_val(flat_state[res_path + ("conv1", "bias")], get_pt_tensor(f"{res_pt}.conv1.bias"))
    set_val(flat_state[res_path + ("norm2", "scale")], get_pt_tensor(f"{res_pt}.norm2.weight", is_norm=True))
    set_val(flat_state[res_path + ("norm2", "bias")], get_pt_tensor(f"{res_pt}.norm2.bias", is_norm=True))
    set_val(flat_state[res_path + ("conv2", "kernel")], get_conv_kernel(f"{res_pt}.conv2.weight"))
    set_val(flat_state[res_path + ("conv2", "bias")], get_pt_tensor(f"{res_pt}.conv2.bias"))

  dec_attn_pt = "decoder.mid_block.attentions.0"
  dec_attn_path = ("decoder", "mid_block", "attentions_0")
  set_val(
      flat_state[dec_attn_path + ("group_norm", "scale")], get_pt_tensor(f"{dec_attn_pt}.group_norm.weight", is_norm=True)
  )
  set_val(flat_state[dec_attn_path + ("group_norm", "bias")], get_pt_tensor(f"{dec_attn_pt}.group_norm.bias", is_norm=True))
  set_val(flat_state[dec_attn_path + ("to_q", "kernel")], get_linear_kernel(f"{dec_attn_pt}.to_q.weight"))
  set_val(flat_state[dec_attn_path + ("to_q", "bias")], get_pt_tensor(f"{dec_attn_pt}.to_q.bias"))
  set_val(flat_state[dec_attn_path + ("to_k", "kernel")], get_linear_kernel(f"{dec_attn_pt}.to_k.weight"))
  set_val(flat_state[dec_attn_path + ("to_k", "bias")], get_pt_tensor(f"{dec_attn_pt}.to_k.bias"))
  set_val(flat_state[dec_attn_path + ("to_v", "kernel")], get_linear_kernel(f"{dec_attn_pt}.to_v.weight"))
  set_val(flat_state[dec_attn_path + ("to_v", "bias")], get_pt_tensor(f"{dec_attn_pt}.to_v.bias"))
  set_val(flat_state[dec_attn_path + ("to_out", "kernel")], get_linear_kernel(f"{dec_attn_pt}.to_out.0.weight"))
  set_val(flat_state[dec_attn_path + ("to_out", "bias")], get_pt_tensor(f"{dec_attn_pt}.to_out.0.bias"))

  for b_idx in range(4):
    up_block_pt = f"decoder.up_blocks.{b_idx}"
    for r_idx in range(3):
      res_pt = f"{up_block_pt}.resnets.{r_idx}"
      res_path = ("decoder", "up_blocks", b_idx, "resnets", r_idx)

      set_val(flat_state[res_path + ("norm1", "scale")], get_pt_tensor(f"{res_pt}.norm1.weight", is_norm=True))
      set_val(flat_state[res_path + ("norm1", "bias")], get_pt_tensor(f"{res_pt}.norm1.bias", is_norm=True))
      set_val(flat_state[res_path + ("conv1", "kernel")], get_conv_kernel(f"{res_pt}.conv1.weight"))
      set_val(flat_state[res_path + ("conv1", "bias")], get_pt_tensor(f"{res_pt}.conv1.bias"))

      set_val(flat_state[res_path + ("norm2", "scale")], get_pt_tensor(f"{res_pt}.norm2.weight", is_norm=True))
      set_val(flat_state[res_path + ("norm2", "bias")], get_pt_tensor(f"{res_pt}.norm2.bias", is_norm=True))
      set_val(flat_state[res_path + ("conv2", "kernel")], get_conv_kernel(f"{res_pt}.conv2.weight"))
      set_val(flat_state[res_path + ("conv2", "bias")], get_pt_tensor(f"{res_pt}.conv2.bias"))

      shortcut_key = f"{res_pt}.conv_shortcut.weight"
      if shortcut_key in pt_state_dict:
        set_val(flat_state[res_path + ("conv_shortcut", "kernel")], get_conv_kernel(shortcut_key))
        set_val(flat_state[res_path + ("conv_shortcut", "bias")], get_pt_tensor(f"{res_pt}.conv_shortcut.bias"))

    if b_idx < 3:
      ups_pt = f"{up_block_pt}.upsamplers.0.conv"
      ups_path = ("decoder", "up_blocks", b_idx, "upsamplers_0", "conv")
      set_val(flat_state[ups_path + ("kernel",)], get_conv_kernel(f"{ups_pt}.weight"))
      set_val(flat_state[ups_path + ("bias",)], get_pt_tensor(f"{ups_pt}.bias"))

  set_val(flat_state[("decoder", "conv_norm_out", "scale")], get_pt_tensor("decoder.conv_norm_out.weight", is_norm=True))
  set_val(flat_state[("decoder", "conv_norm_out", "bias")], get_pt_tensor("decoder.conv_norm_out.bias", is_norm=True))
  set_val(flat_state[("decoder", "conv_out", "kernel")], get_conv_kernel("decoder.conv_out.weight"))
  set_val(flat_state[("decoder", "conv_out", "bias")], get_pt_tensor("decoder.conv_out.bias"))

  # Update nnx_vae state
  nnx.update(nnx_vae, nnx.from_flat_state(flat_state))

  # Extract Batch Normalization running stats
  bn_mean = jnp.array(get_pt_tensor("bn.running_mean")).reshape(1, -1, 1, 1)
  bn_var = jnp.array(get_pt_tensor("bn.running_var")).reshape(1, -1, 1, 1)
  batch_norm_eps = 0.0001
  bn_std = jnp.sqrt(bn_var + batch_norm_eps)

  return bn_mean, bn_std
