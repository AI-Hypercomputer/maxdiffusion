import os
from dataclasses import dataclass

import jax
import torch  # need for torch 2 jax
from chex import Array
from flax import nnx
from huggingface_hub import hf_hub_download
from jax import numpy as jnp
from safetensors import safe_open
from einops import rearrange
from maxdiffusion.models.ae_flux_nnx import AutoEncoder, AutoEncoderParams
from maxdiffusion.models.transformers.transformer_flux_flax import FluxParams

##############################################################################################
# AUTOENCODER MODEL PORTING
##############################################################################################


def port_group_norm(group_norm, tensors, prefix):
  group_norm.scale.value = tensors[f"{prefix}.weight"]
  group_norm.bias.value = tensors[f"{prefix}.bias"]

  return group_norm


def port_conv(conv, tensors, prefix):
  conv.kernel.value = rearrange(tensors[f"{prefix}.weight"], "i o k1 k2 -> k1 k2 o i")
  conv.bias.value = tensors[f"{prefix}.bias"]

  return conv


def port_attn_block(attn_block, tensors, prefix):
  # port the norm
  attn_block.norm = port_group_norm(
      group_norm=attn_block.norm,
      tensors=tensors,
      prefix=f"{prefix}.norm",
  )

  # port the k, q, v layers
  attn_block.k = port_conv(
      conv=attn_block.k,
      tensors=tensors,
      prefix=f"{prefix}.k",
  )

  attn_block.q = port_conv(
      conv=attn_block.q,
      tensors=tensors,
      prefix=f"{prefix}.q",
  )

  attn_block.v = port_conv(
      conv=attn_block.v,
      tensors=tensors,
      prefix=f"{prefix}.v",
  )

  # port the proj_out layer
  attn_block.proj_out = port_conv(
      conv=attn_block.proj_out,
      tensors=tensors,
      prefix=f"{prefix}.proj_out",
  )

  return attn_block


def port_resent_block(resnet_block, tensors, prefix):
  # port the norm
  resnet_block.norm1 = port_group_norm(
      group_norm=resnet_block.norm1,
      tensors=tensors,
      prefix=f"{prefix}.norm1",
  )
  resnet_block.norm2 = port_group_norm(
      group_norm=resnet_block.norm2,
      tensors=tensors,
      prefix=f"{prefix}.norm2",
  )

  # port the convs
  resnet_block.conv1 = port_conv(
      conv=resnet_block.conv1,
      tensors=tensors,
      prefix=f"{prefix}.conv1",
  )
  resnet_block.conv2 = port_conv(
      conv=resnet_block.conv2,
      tensors=tensors,
      prefix=f"{prefix}.conv2",
  )

  if resnet_block.in_channels != resnet_block.out_channels:
    resnet_block.nin_shortcut = port_conv(
        conv=resnet_block.nin_shortcut,
        tensors=tensors,
        prefix=f"{prefix}.nin_shortcut",
    )

  return resnet_block


def port_downsample(downsample, tensors, prefix):
  # port the conv
  downsample.conv = port_conv(
      conv=downsample.conv,
      tensors=tensors,
      prefix=f"{prefix}.conv",
  )

  return downsample


def port_upsample(upsample, tensors, prefix):
  # port the conv
  upsample.conv = port_conv(
      conv=upsample.conv,
      tensors=tensors,
      prefix=f"{prefix}.conv",
  )

  return upsample


def port_encoder(encoder, tensors, prefix):
  # conv in
  encoder.conv_in = port_conv(
      conv=encoder.conv_in,
      tensors=tensors,
      prefix=f"{prefix}.conv_in",
  )

  # down
  for i, down_layer in enumerate(encoder.down.layers):
    # block
    for j, block_layer in enumerate(down_layer.block.layers):
      block_layer = port_resent_block(
          resnet_block=block_layer,
          tensors=tensors,
          prefix=f"{prefix}.down.{i}.block.{j}",
      )
    # attn
    for j, attn_layer in enumerate(down_layer.attn.layers):
      attn_layer = port_attn_block(
          attn_block=attn_layer,
          tensors=tensors,
          prefix=f"{prefix}.attn.{i}.block.{j}",
      )

    # downsample
    if i != encoder.num_resolutions - 1:
      downsample = down_layer.downsample
      downsample = port_downsample(
          downsample=downsample,
          tensors=tensors,
          prefix=f"{prefix}.down.{i}.downsample",
      )

  # mid
  encoder.mid.block_1 = port_resent_block(
      resnet_block=encoder.mid.block_1,
      tensors=tensors,
      prefix=f"{prefix}.mid.block_1",
  )
  encoder.mid.attn_1 = port_attn_block(
      attn_block=encoder.mid.attn_1,
      tensors=tensors,
      prefix=f"{prefix}.mid.attn_1",
  )
  encoder.mid.block_2 = port_resent_block(
      resnet_block=encoder.mid.block_2,
      tensors=tensors,
      prefix=f"{prefix}.mid.block_2",
  )

  # norm out
  encoder.norm_out = port_group_norm(
      group_norm=encoder.norm_out,
      tensors=tensors,
      prefix=f"{prefix}.norm_out",
  )

  # conv out
  encoder.conv_out = port_conv(
      conv=encoder.conv_out,
      tensors=tensors,
      prefix=f"{prefix}.conv_out",
  )

  return encoder


def port_decoder(decoder, tensors, prefix):
  # conv in
  decoder.conv_in = port_conv(
      conv=decoder.conv_in,
      tensors=tensors,
      prefix=f"{prefix}.conv_in",
  )

  # mid
  decoder.mid.block_1 = port_resent_block(
      resnet_block=decoder.mid.block_1,
      tensors=tensors,
      prefix=f"{prefix}.mid.block_1",
  )
  decoder.mid.attn_1 = port_attn_block(
      attn_block=decoder.mid.attn_1,
      tensors=tensors,
      prefix=f"{prefix}.mid.attn_1",
  )
  decoder.mid.block_2 = port_resent_block(
      resnet_block=decoder.mid.block_2,
      tensors=tensors,
      prefix=f"{prefix}.mid.block_2",
  )

  for i, up_layer in enumerate(decoder.up.layers):
    # block
    for j, block_layer in enumerate(up_layer.block.layers):
      block_layer = port_resent_block(
          resnet_block=block_layer,
          tensors=tensors,
          prefix=f"{prefix}.up.{i}.block.{j}",
      )

    # attn
    for j, attn_layer in enumerate(up_layer.attn.layers):
      attn_layer = port_attn_block(
          attn_block=attn_layer,
          tensors=tensors,
          prefix=f"{prefix}.up.{i}.attn.{j}",
      )

    # upsample
    if i != 0:
      up_layer.upsample = port_upsample(
          upsample=up_layer.upsample,
          tensors=tensors,
          prefix=f"{prefix}.up.{i}.upsample",
      )

  # norm out
  decoder.norm_out = port_group_norm(
      group_norm=decoder.norm_out,
      tensors=tensors,
      prefix=f"{prefix}.norm_out",
  )

  # conv out
  decoder.conv_out = port_conv(
      conv=decoder.conv_out,
      tensors=tensors,
      prefix=f"{prefix}.conv_out",
  )

  return decoder


def port_autoencoder(autoencoder, tensors):
  autoencoder.encoder = port_encoder(
      encoder=autoencoder.encoder,
      tensors=tensors,
      prefix="encoder",
  )
  autoencoder.decoder = port_decoder(
      decoder=autoencoder.decoder,
      tensors=tensors,
      prefix="decoder",
  )
  return autoencoder


def torch2jax(torch_tensor: torch.Tensor) -> Array:
  is_bfloat16 = torch_tensor.dtype == torch.bfloat16
  if is_bfloat16:
    # upcast the tensor to fp32
    torch_tensor = torch_tensor.to(dtype=torch.float32)

  if torch.device.type != "cpu":
    torch_tensor = torch_tensor.to("cpu")

  numpy_value = torch_tensor.numpy()
  jax_array = jnp.array(numpy_value, dtype=jnp.bfloat16)
  return jax_array


@dataclass
class ModelSpec:
  params: FluxParams
  ae_params: AutoEncoderParams
  ckpt_path: str | None
  ae_path: str | None
  repo_id: str | None
  repo_flow: str | None
  repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
  if len(missing) > 0 and len(unexpected) > 0:
    print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    print("\n" + "-" * 79 + "\n")
    print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
  elif len(missing) > 0:
    print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
  elif len(unexpected) > 0:
    print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_ae(name: str, device: str, hf_download: bool = True) -> AutoEncoder:
  device = jax.devices(device)[0]
  with jax.default_device(device):
    ckpt_path = configs[name].ae_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_ae is not None and hf_download:
      ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    print(f"Load and port autoencoder on {device}")
    ae = AutoEncoder(params=configs[name].ae_params)

    if ckpt_path is not None:
      tensors = {}
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
      ae = port_autoencoder(autoencoder=ae, tensors=tensors)

      del tensors
      jax.clear_caches()
  return ae
