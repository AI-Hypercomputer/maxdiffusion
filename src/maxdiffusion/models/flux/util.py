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

# copied from https://github.com/ml-gde/jflux/blob/main/jflux/util.py
import os
from dataclasses import dataclass

import jax
from jax.typing import DTypeLike
from chex import Array
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import hf_hub_download
from jax import numpy as jnp
from safetensors import safe_open

from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax)
from maxdiffusion import max_logging


@dataclass
class FluxParams:
  in_channels: int
  vec_in_dim: int
  context_in_dim: int
  hidden_size: int
  mlp_ratio: float
  num_heads: int
  depth: int
  depth_single_blocks: int
  axes_dim: list[int]
  theta: int
  qkv_bias: bool
  guidance_embed: bool
  rngs: Array
  param_dtype: DTypeLike


@dataclass
class ModelSpec:
  params: FluxParams
  ckpt_path: str | None
  repo_id: str | None
  repo_flow: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
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
            rngs=jax.random.PRNGKey(42),
            param_dtype=jnp.bfloat16,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
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
            rngs=jax.random.PRNGKey(47),
            param_dtype=jnp.bfloat16,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
  if len(missing) > 0 and len(unexpected) > 0:
    max_logging.log(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    max_logging.log("\n" + "-" * 79 + "\n")
    max_logging.log(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
  elif len(missing) > 0:
    max_logging.log(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
  elif len(unexpected) > 0:
    max_logging.log(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def validate_flax_state_dict(expected_pytree: dict, new_pytree: dict):
  """
  expected_pytree: dict - a pytree that comes from initializing the model.
  new_pytree: dict - a pytree that has been created from pytorch weights.
  """
  expected_pytree = flatten_dict(expected_pytree)
  if len(expected_pytree.keys()) != len(new_pytree.keys()):
    set1 = set(expected_pytree.keys())
    set2 = set(new_pytree.keys())
    missing_keys = set1 ^ set2
    max_logging.log(f"missing keys : {missing_keys}")
  for key in expected_pytree.keys():
    if key in new_pytree.keys():
      try:
        expected_pytree_shape = expected_pytree[key].shape
      except Exception:
        expected_pytree_shape = expected_pytree[key].value.shape
      if expected_pytree_shape != new_pytree[key].shape:
        max_logging.log(
            f"shape mismatch, expected shape of {expected_pytree[key].shape}, but got shape of {new_pytree[key].shape}"
        )
    else:
      max_logging.log(f"key: {key} not found...")


def load_flow_model(name: str, eval_shapes: dict, device: str, hf_download: bool = True):  # -> Flux:
  device = jax.devices(device)[0]
  with jax.default_device(device):
    ckpt_path = configs[name].ckpt_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_flow is not None and hf_download:
      ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    max_logging.log(f"Load and port flux on {device}")

    if ckpt_path is not None:
      tensors = {}
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        if "double_blocks" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("img_mlp_", "img_mlp.layers_")
          renamed_pt_key = renamed_pt_key.replace("txt_mlp_", "txt_mlp.layers_")
          renamed_pt_key = renamed_pt_key.replace("img_mod", "img_norm1")
          renamed_pt_key = renamed_pt_key.replace("txt_mod", "txt_norm1")
          renamed_pt_key = renamed_pt_key.replace("img_attn.qkv", "attn.i_qkv")
          renamed_pt_key = renamed_pt_key.replace("img_attn.proj", "attn.i_proj")
          renamed_pt_key = renamed_pt_key.replace("img_attn.norm", "attn")
          renamed_pt_key = renamed_pt_key.replace("txt_attn.qkv", "attn.e_qkv")
          renamed_pt_key = renamed_pt_key.replace("txt_attn.proj", "attn.e_proj")
          renamed_pt_key = renamed_pt_key.replace("txt_attn.norm.key_norm", "attn.encoder_key_norm")
          renamed_pt_key = renamed_pt_key.replace("txt_attn.norm.query_norm", "attn.encoder_query_norm")
        elif "time_guidance_embed" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("time_guidance_embed.timestep_embedder", "time_text_embed.FlaxTimestepEmbedding_0")
          renamed_pt_key = renamed_pt_key.replace("time_guidance_embed.guidance_embedder", "time_text_embed.FlaxTimestepEmbedding_1")
        elif "guidance_in" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("guidance_in", "time_text_embed.FlaxTimestepEmbedding_1")
          renamed_pt_key = renamed_pt_key.replace("in_layer", "linear_1")
          renamed_pt_key = renamed_pt_key.replace("out_layer", "linear_2")
        elif "single_blocks" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("modulation", "norm")
          renamed_pt_key = renamed_pt_key.replace("norm.key_norm", "attn.key_norm")
          renamed_pt_key = renamed_pt_key.replace("norm.query_norm", "attn.query_norm")
        elif "vector_in" in renamed_pt_key or "time_in" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("vector_in", "time_text_embed.PixArtAlphaTextProjection_0")
          renamed_pt_key = renamed_pt_key.replace("time_in", "time_text_embed.FlaxTimestepEmbedding_0")
          renamed_pt_key = renamed_pt_key.replace("in_layer", "linear_1")
          renamed_pt_key = renamed_pt_key.replace("out_layer", "linear_2")
        elif "final_layer" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("final_layer.linear", "proj_out")
          renamed_pt_key = renamed_pt_key.replace("final_layer.adaLN_modulation_1", "norm_out.Dense_0")
        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
  return flax_state_dict


# -----------------------------------------------------------------------------
# Latent Packing & Unpacking Helpers
# -----------------------------------------------------------------------------


def pack_latents(latents):
  """
  Groups spatial 2x2 latent neighborhoods into a single channel dimension.
  Transforms unpacked shape (batch_size, channels, height, width)
  to packed sequence shape (batch_size, (height//2)*(width//2), channels*4).
  """
  import numpy as np
  import jax.numpy as jnp

  batch_size, channels, height, width = latents.shape
  latents = np.reshape(latents, (batch_size, channels, height // 2, 2, width // 2, 2))
  latents = np.transpose(latents, (0, 2, 4, 1, 3, 5))
  latents = np.reshape(latents, (batch_size, (height // 2) * (width // 2), channels * 4))
  return jnp.array(latents)


def unpack_latents(latents, batch_size, num_channels_latents, height, width):
  """
  Unpacks packed sequence of shape (batch_size, (height//16)*(width//16), channels*4)
  back to the unpacked spatial grid shape (batch_size, channels, height//8, width//8).
  """
  import numpy as np

  h_latent = height // 8
  w_latent = width // 8

  # 1. Reshape to split spatial grid and packed channel blocks
  latents = np.reshape(latents, (batch_size, h_latent // 2, w_latent // 2, num_channels_latents, 2, 2))
  # 2. Permute dimensions back to unpacked order
  latents = np.transpose(latents, (0, 3, 1, 4, 2, 5))
  # 3. Flatten back to 4D unpacked latent shape
  latents = np.reshape(latents, (batch_size, num_channels_latents, h_latent, w_latent))
  return latents


def unpack_latents_with_ids(x, x_ids, height, width):
  """[B, H*W, C] -> [B, C, H, W] using coordinate IDs."""
  import jax.numpy as jnp

  batch_size, seq_len, ch = x.shape
  x_list = []
  for b in range(batch_size):
    data = x[b]
    pos = x_ids[b]
    h_ids = pos[:, 1].astype(jnp.int32)
    w_ids = pos[:, 2].astype(jnp.int32)
    flat_ids = h_ids * width + w_ids
    out = jnp.zeros((height * width, ch), dtype=x.dtype)
    out = out.at[flat_ids].set(data)
    out = jnp.transpose(jnp.reshape(out, (height, width, ch)), (2, 0, 1))
    x_list.append(out)
  return jnp.stack(x_list, axis=0)


def unpatchify_latents(latents):
  """Reverses the 2x2 spatial patch grouping: [B, C, H, W] -> [B, C/4, H*2, W*2]"""
  import jax.numpy as jnp

  batch_size, num_channels_latents, height, width = latents.shape
  x = jnp.reshape(latents, (batch_size, num_channels_latents // 4, 2, 2, height, width))
  x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
  x = jnp.reshape(x, (batch_size, num_channels_latents // 4, height * 2, width * 2))
  return x


# -----------------------------------------------------------------------------
# 4D RoPE Position Grid Helpers
# -----------------------------------------------------------------------------


def prepare_latent_image_ids(batch_size, height, width):
  """
  Generates positional identifiers (Height and Width coordinates) for images to build RoPE grids.
  Shape: (batch_size, height * width, 4)
  """
  import jax.numpy as jnp

  grid = jnp.zeros((height, width, 4), dtype=jnp.int32)
  grid = grid.at[..., 1].set(jnp.arange(height)[:, None])
  grid = grid.at[..., 2].set(jnp.arange(width)[None, :])
  latent_image_ids = grid.reshape(-1, 4)
  return jnp.tile(latent_image_ids[None, ...], (batch_size, 1, 1))


def prepare_text_ids(batch_size, seq_len):
  """
  Generates positional identifiers for text prompt tokens in Flux.2-Klein (all zeros).
  Shape: (batch_size, seq_len, 4)
  """
  import jax.numpy as jnp
  text_ids = jnp.zeros((seq_len, 4))
  return jnp.tile(text_ids[None, ...], (batch_size, 1, 1))


# -----------------------------------------------------------------------------
# Parameter In-place Casting Helper
# -----------------------------------------------------------------------------


def cast_dict_to_bfloat16_inplace(d, device=None):
  """Casts a nested dictionary of JAX/numpy arrays to bfloat16 in-place, freeing memory immediately."""
  import gc
  import jax
  import jax.numpy as jnp

  for k, v in list(d.items()):
    if isinstance(v, dict):
      cast_dict_to_bfloat16_inplace(v, device=device)
    elif hasattr(v, "astype"):
      if device is not None:
        with jax.default_device(device):
          d[k] = jnp.array(v, dtype=jnp.bfloat16)
      else:
        d[k] = jnp.array(v, dtype=jnp.bfloat16)
      if hasattr(d[k], "block_until_ready"):
        d[k].block_until_ready()
      del v
      gc.collect()


# -----------------------------------------------------------------------------
# Safetensors Weight Loader & Key Converter Functions
# -----------------------------------------------------------------------------


def load_and_convert_flux_klein_weights(safetensors_path, params, num_double_layers, num_single_layers):
  """
  Loads PyTorch weights from safetensors and converts them to JAX parameter dictionary.
  Supports dynamic layer counts (double and single stream blocks) and sharded safetensors directories.
  """
  from safetensors.torch import load_file
  import torch
  import numpy as np
  import jax.numpy as jnp
  import glob
  import os
  import gc
  from flax.traverse_util import flatten_dict, unflatten_dict
  import flax

  pt_state_dict = {}
  if os.path.isdir(safetensors_path):
    shards = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
    print(f"Loading sharded PyTorch weights from directory: {safetensors_path} (Found {len(shards)} shards)...")
    for shard in sorted(shards):
      print(f"Loading shard: {shard}...")
      pt_state_dict.update(load_file(shard, device="cpu"))
  else:
    print(f"Loading PyTorch weights from: {safetensors_path}")
    pt_state_dict = load_file(safetensors_path, device="cpu")

  print("Mapping PyTorch weights to JAX parameters...")

  first_leaf = jax.tree_util.tree_leaves(params)[0]
  target_dtype = first_leaf.dtype

  def cvt(tensor, transpose=False):
    if transpose:
      tensor = tensor.T
    return jnp.array(tensor.to(torch.float32).cpu().numpy(), dtype=target_dtype)

  # Flatten JAX params for safe & direct tuple-key assignment
  params_flat = dict(flatten_dict(flax.core.unfreeze(params)))

  # Global layers
  if ("context_embedder", "kernel") in params_flat:
    params_flat[("context_embedder", "kernel")] = cvt(pt_state_dict.pop("context_embedder.weight"), transpose=True)
    if "context_embedder.bias" in pt_state_dict:
      params_flat[("context_embedder", "bias")] = cvt(pt_state_dict.pop("context_embedder.bias"))

  if ("x_embedder", "kernel") in params_flat:
    params_flat[("x_embedder", "kernel")] = cvt(pt_state_dict.pop("x_embedder.weight"), transpose=True)
    if "x_embedder.bias" in pt_state_dict:
      params_flat[("x_embedder", "bias")] = cvt(pt_state_dict.pop("x_embedder.bias"))

  params_flat[("double_stream_modulation_img", "kernel")] = cvt(
      pt_state_dict.pop("double_stream_modulation_img.linear.weight"), transpose=True
  )
  params_flat[("double_stream_modulation_txt", "kernel")] = cvt(
      pt_state_dict.pop("double_stream_modulation_txt.linear.weight"), transpose=True
  )
  params_flat[("single_stream_modulation", "kernel")] = cvt(
      pt_state_dict.pop("single_stream_modulation.linear.weight"), transpose=True
  )
  params_flat[("proj_out", "kernel")] = cvt(pt_state_dict.pop("proj_out.weight"), transpose=True)
  if "proj_out.bias" in pt_state_dict:
    params_flat[("proj_out", "bias")] = cvt(pt_state_dict.pop("proj_out.bias"))

  # norm_out
  params_flat[("norm_out", "linear", "kernel")] = cvt(pt_state_dict.pop("norm_out.linear.weight"), transpose=True)

  # time_text_embed (Timestep Embedding)
  if "time_guidance_embed.timestep_embedder.linear_1.weight" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_0", "linear_1", "kernel")] = cvt(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_1.weight"), transpose=True
    )
  if "time_guidance_embed.timestep_embedder.linear_1.bias" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_0", "linear_1", "bias")] = cvt(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_1.bias")
    )
  if "time_guidance_embed.timestep_embedder.linear_2.weight" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_0", "linear_2", "kernel")] = cvt(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_2.weight"), transpose=True
    )
  if "time_guidance_embed.timestep_embedder.linear_2.bias" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_0", "linear_2", "bias")] = cvt(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_2.bias")
    )

  if "time_guidance_embed.guidance_embedder.linear_1.weight" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_1", "linear_1", "kernel")] = cvt(
        pt_state_dict.pop("time_guidance_embed.guidance_embedder.linear_1.weight"), transpose=True
    )
  if "time_guidance_embed.guidance_embedder.linear_1.bias" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_1", "linear_1", "bias")] = cvt(
        pt_state_dict.pop("time_guidance_embed.guidance_embedder.linear_1.bias")
    )
  if "time_guidance_embed.guidance_embedder.linear_2.weight" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_1", "linear_2", "kernel")] = cvt(
        pt_state_dict.pop("time_guidance_embed.guidance_embedder.linear_2.weight"), transpose=True
    )
  if "time_guidance_embed.guidance_embedder.linear_2.bias" in pt_state_dict:
    params_flat[("time_text_embed", "FlaxTimestepEmbedding_1", "linear_2", "bias")] = cvt(
        pt_state_dict.pop("time_guidance_embed.guidance_embedder.linear_2.bias")
    )

  # Double Blocks
  print(f"Mapping {num_double_layers} double-stream attention blocks...")
  for block_idx in range(num_double_layers):
    str_idx = str(block_idx)
    prefix = f"transformer_blocks.{block_idx}."

    # Concatenate QKV projections
    to_q = pt_state_dict.pop(prefix + "attn.to_q.weight").to(torch.float32).T.cpu().numpy()
    to_k = pt_state_dict.pop(prefix + "attn.to_k.weight").to(torch.float32).T.cpu().numpy()
    to_v = pt_state_dict.pop(prefix + "attn.to_v.weight").to(torch.float32).T.cpu().numpy()
    params_flat[("double_blocks", str_idx, "attn", "i_qkv", "kernel")] = jnp.array(
        np.concatenate([to_q, to_k, to_v], axis=1), dtype=target_dtype
    )

    add_q = pt_state_dict.pop(prefix + "attn.add_q_proj.weight").to(torch.float32).T.cpu().numpy()
    add_k = pt_state_dict.pop(prefix + "attn.add_k_proj.weight").to(torch.float32).T.cpu().numpy()
    add_v = pt_state_dict.pop(prefix + "attn.add_v_proj.weight").to(torch.float32).T.cpu().numpy()
    params_flat[("double_blocks", str_idx, "attn", "e_qkv", "kernel")] = jnp.array(
        np.concatenate([add_q, add_k, add_v], axis=1), dtype=target_dtype
    )

    # Projections out
    params_flat[("double_blocks", str_idx, "attn", "i_proj", "kernel")] = cvt(pt_state_dict.pop(prefix + "attn.to_out.0.weight"), transpose=True)
    params_flat[("double_blocks", str_idx, "attn", "e_proj", "kernel")] = cvt(pt_state_dict.pop(prefix + "attn.to_add_out.weight"), transpose=True)

    # Norm scales
    params_flat[("double_blocks", str_idx, "attn", "query_norm", "scale")] = cvt(pt_state_dict.pop(prefix + "attn.norm_q.weight"))
    params_flat[("double_blocks", str_idx, "attn", "key_norm", "scale")] = cvt(pt_state_dict.pop(prefix + "attn.norm_k.weight"))
    params_flat[("double_blocks", str_idx, "attn", "encoder_query_norm", "scale")] = cvt(pt_state_dict.pop(prefix + "attn.norm_added_q.weight"))
    params_flat[("double_blocks", str_idx, "attn", "encoder_key_norm", "scale")] = cvt(pt_state_dict.pop(prefix + "attn.norm_added_k.weight"))


    # SwiGLU MLPs
    params_flat[("double_blocks", str_idx, "img_mlp", "layers_0", "kernel")] = cvt(pt_state_dict.pop(prefix + "ff.linear_in.weight"), transpose=True)
    params_flat[("double_blocks", str_idx, "img_mlp", "layers_1", "kernel")] = cvt(pt_state_dict.pop(prefix + "ff.linear_out.weight"), transpose=True)
    params_flat[("double_blocks", str_idx, "txt_mlp", "layers_0", "kernel")] = cvt(pt_state_dict.pop(prefix + "ff_context.linear_in.weight"), transpose=True)
    params_flat[("double_blocks", str_idx, "txt_mlp", "layers_1", "kernel")] = cvt(pt_state_dict.pop(prefix + "ff_context.linear_out.weight"), transpose=True)

  # Single Blocks
  print(f"Mapping {num_single_layers} single-stream attention blocks...")
  for block_idx in range(num_single_layers):
    str_idx = str(block_idx)
    s_prefix = f"single_transformer_blocks.{block_idx}."

    # Joint projections
    params_flat[("single_blocks", str_idx, "linear1", "kernel")] = cvt(pt_state_dict.pop(s_prefix + "attn.to_qkv_mlp_proj.weight"), transpose=True)
    params_flat[("single_blocks", str_idx, "linear2", "kernel")] = cvt(pt_state_dict.pop(s_prefix + "attn.to_out.weight"), transpose=True)

    # Norm scales & modulations
    params_flat[("single_blocks", str_idx, "attn", "query_norm", "scale")] = cvt(pt_state_dict.pop(s_prefix + "attn.norm_q.weight"))
    params_flat[("single_blocks", str_idx, "attn", "key_norm", "scale")] = cvt(pt_state_dict.pop(s_prefix + "attn.norm_k.weight"))

  del pt_state_dict
  gc.collect()
  print("Weight conversion complete!")
  return unflatten_dict(params_flat)


def load_and_convert_vae_weights(safetensors_path, jax_params):
  """Loads PyTorch VAE weights from safetensors, maps them to JAX, and extracts BN stats."""
  from safetensors.torch import load_file
  import torch
  import flax
  import jax.numpy as jnp

  print(f"Loading PyTorch VAE weights from: {safetensors_path}")
  pt_state_dict = load_file(safetensors_path)

  # Helper to safely convert PyTorch bfloat16 tensors to numpy float32
  def get_w(key):
    return pt_state_dict[key].to(torch.float32).cpu().numpy()

  # Unfreeze JAX params so we can load the weights
  jax_params = flax.core.unfreeze(jax_params)

  # Map weights
  print("Mapping VAE decoder weights to JAX parameters...")

  # post_quant_conv
  jax_params["post_quant_conv"]["kernel"] = jnp.array(get_w("post_quant_conv.weight").transpose(2, 3, 1, 0))
  jax_params["post_quant_conv"]["bias"] = jnp.array(get_w("post_quant_conv.bias"))

  # decoder.conv_in
  jax_params["decoder"]["conv_in"]["kernel"] = jnp.array(get_w("decoder.conv_in.weight").transpose(2, 3, 1, 0))
  jax_params["decoder"]["conv_in"]["bias"] = jnp.array(get_w("decoder.conv_in.bias"))

  # decoder.mid_block
  # resnets
  for idx in [0, 1]:
    res_jax = jax_params["decoder"]["mid_block"][f"resnets_{idx}"]
    res_pt_prefix = f"decoder.mid_block.resnets.{idx}"

    res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.weight"))
    res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.bias"))
    res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.weight").transpose(2, 3, 1, 0))
    res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.bias"))

    res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.weight"))
    res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.bias"))
    res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.weight").transpose(2, 3, 1, 0))
    res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.bias"))

  # attentions
  attn_pt_prefix = "decoder.mid_block.attentions.0"
  attn_jax = jax_params["decoder"]["mid_block"]["attentions_0"]

  attn_jax["group_norm"]["scale"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.weight"))
  attn_jax["group_norm"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.bias"))

  attn_jax["query"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.weight").T)
  attn_jax["query"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.bias"))
  attn_jax["key"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.weight").T)
  attn_jax["key"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.bias"))
  attn_jax["value"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.weight").T)
  attn_jax["value"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.bias"))

  attn_jax["proj_attn"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.weight").T)
  attn_jax["proj_attn"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.bias"))

  # decoder.up_blocks
  for b_idx in range(4):
    up_block_jax = jax_params["decoder"][f"up_blocks_{b_idx}"]
    up_block_pt = f"decoder.up_blocks.{b_idx}"

    for r_idx in range(3):
      res_jax = up_block_jax[f"resnets_{r_idx}"]
      res_pt = f"{up_block_pt}.resnets.{r_idx}"

      res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt}.norm1.weight"))
      res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt}.norm1.bias"))
      res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv1.weight").transpose(2, 3, 1, 0))
      res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt}.conv1.bias"))

      res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt}.norm2.weight"))
      res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt}.norm2.bias"))
      res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv2.weight").transpose(2, 3, 1, 0))
      res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt}.conv2.bias"))

      shortcut_key = f"{res_pt}.conv_shortcut.weight"
      if shortcut_key in pt_state_dict:
        res_jax["conv_shortcut"]["kernel"] = jnp.array(get_w(shortcut_key).transpose(2, 3, 1, 0))
        res_jax["conv_shortcut"]["bias"] = jnp.array(get_w(f"{res_pt}.conv_shortcut.bias"))

    if b_idx < 3:
      upsampler_jax = up_block_jax["upsamplers_0"]
      upsampler_pt = f"{up_block_pt}.upsamplers.0"

      upsampler_jax["conv"]["kernel"] = jnp.array(get_w(f"{upsampler_pt}.conv.weight").transpose(2, 3, 1, 0))
      upsampler_jax["conv"]["bias"] = jnp.array(get_w(f"{upsampler_pt}.conv.bias"))

  # decoder.conv_norm_out & conv_out
  jax_params["decoder"]["conv_norm_out"]["scale"] = jnp.array(get_w("decoder.conv_norm_out.weight"))
  jax_params["decoder"]["conv_norm_out"]["bias"] = jnp.array(get_w("decoder.conv_norm_out.bias"))
  jax_params["decoder"]["conv_out"]["kernel"] = jnp.array(get_w("decoder.conv_out.weight").transpose(2, 3, 1, 0))
  jax_params["decoder"]["conv_out"]["bias"] = jnp.array(get_w("decoder.conv_out.bias"))

  # Freeze parameters
  jax_params = flax.core.freeze(jax_params)

  # Extract Batch Normalization running stats
  print("Extracting VAE Batch Normalization running stats...")
  bn_mean = jnp.array(get_w("bn.running_mean")).reshape(1, -1, 1, 1)
  bn_var = jnp.array(get_w("bn.running_var")).reshape(1, -1, 1, 1)
  batch_norm_eps = 0.0001
  bn_std = jnp.sqrt(bn_var + batch_norm_eps)

  print("VAE weights and BN stats loaded successfully!")
  return jax_params, bn_mean, bn_std
