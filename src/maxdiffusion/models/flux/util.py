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
  expected_flat = (
      flatten_dict(expected_pytree) if not isinstance(next(iter(expected_pytree.keys()), None), tuple) else expected_pytree
  )
  new_flat = flatten_dict(new_pytree) if not isinstance(next(iter(new_pytree.keys()), None), tuple) else new_pytree

  if len(expected_flat.keys()) != len(new_flat.keys()):
    set1 = set(expected_flat.keys())
    set2 = set(new_flat.keys())
    missing_keys = set1 ^ set2
    max_logging.log(
        f"Missing or extra parameter keys count mismatch ({len(expected_flat)} expected vs {len(new_flat)} converted): {missing_keys}"
    )

  for key in expected_flat.keys():
    if key in new_flat.keys():
      try:
        expected_pytree_shape = expected_flat[key].shape
      except Exception:
        expected_pytree_shape = getattr(expected_flat[key], "value", expected_flat[key]).shape
      if expected_pytree_shape != new_flat[key].shape:
        max_logging.log(
            f"shape mismatch for key '{key}': expected shape of {expected_pytree_shape}, but got shape of {new_flat[key].shape}"
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

      double_blocks_tensors = {}
      single_blocks_tensors = {}

      for pt_key, tensor in tensors.items():
        if pt_key.startswith("double_blocks."):
          parts = pt_key.split(".")
          layer_idx = int(parts[1])
          pt_key_without_idx = "double_blocks." + ".".join(parts[2:])
          renamed_pt_key = rename_key(pt_key_without_idx)
          renamed_pt_key = renamed_pt_key.replace("double_blocks", "scanned_double_blocks.FluxTransformerBlock_0")
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

          pt_tuple_key = tuple(renamed_pt_key.split("."))
          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes, scan_layers=True)
          if flax_key not in double_blocks_tensors:
            double_blocks_tensors[flax_key] = {}
          double_blocks_tensors[flax_key][layer_idx] = flax_tensor
          continue

        elif pt_key.startswith("single_blocks."):
          parts = pt_key.split(".")
          layer_idx = int(parts[1])
          pt_key_without_idx = "single_blocks." + ".".join(parts[2:])
          renamed_pt_key = rename_key(pt_key_without_idx)
          renamed_pt_key = renamed_pt_key.replace("single_blocks", "scanned_single_blocks.FluxSingleTransformerBlock_0")
          renamed_pt_key = renamed_pt_key.replace("modulation", "norm")
          renamed_pt_key = renamed_pt_key.replace("norm.key_norm", "attn.key_norm")
          renamed_pt_key = renamed_pt_key.replace("norm.query_norm", "attn.query_norm")

          if "linear1" in renamed_pt_key:
            if tensor.ndim == 2:
              qkv_tensor = tensor[:9216, :]
              mlp_tensor = tensor[9216:, :]
            else:
              qkv_tensor = tensor[:9216]
              mlp_tensor = tensor[9216:]
            qkv_pt_key = renamed_pt_key.replace("linear1", "lin_qkv")
            mlp_pt_key = renamed_pt_key.replace("linear1", "mlp_and_out.lin_mlp")

            flax_key_qkv, flax_tensor_qkv = rename_key_and_reshape_tensor(
                tuple(qkv_pt_key.split(".")), qkv_tensor, eval_shapes, scan_layers=True
            )
            flax_key_mlp, flax_tensor_mlp = rename_key_and_reshape_tensor(
                tuple(mlp_pt_key.split(".")), mlp_tensor, eval_shapes, scan_layers=True
            )

            if flax_key_qkv not in single_blocks_tensors:
              single_blocks_tensors[flax_key_qkv] = {}
            single_blocks_tensors[flax_key_qkv][layer_idx] = flax_tensor_qkv

            if flax_key_mlp not in single_blocks_tensors:
              single_blocks_tensors[flax_key_mlp] = {}
            single_blocks_tensors[flax_key_mlp][layer_idx] = flax_tensor_mlp
            continue

          elif "linear2" in renamed_pt_key:
            renamed_pt_key = renamed_pt_key.replace("linear2", "mlp_and_out.linear2")

          pt_tuple_key = tuple(renamed_pt_key.split("."))
          flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes, scan_layers=True)
          if flax_key not in single_blocks_tensors:
            single_blocks_tensors[flax_key] = {}
          single_blocks_tensors[flax_key][layer_idx] = flax_tensor
          continue

        renamed_pt_key = rename_key(pt_key)
        if "guidance_in" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("guidance_in", "time_text_embed.FlaxTimestepEmbedding_1")
          renamed_pt_key = renamed_pt_key.replace("in_layer", "linear_1")
          renamed_pt_key = renamed_pt_key.replace("out_layer", "linear_2")
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

      # Stack double blocks
      for flax_key, layers in double_blocks_tensors.items():
        sorted_indices = sorted(layers.keys())
        stacked_tensor = jnp.stack([layers[i] for i in sorted_indices], axis=0)
        flax_state_dict[flax_key] = jax.device_put(stacked_tensor, device=cpu)

      # Stack single blocks
      for flax_key, layers in single_blocks_tensors.items():
        sorted_indices = sorted(layers.keys())
        stacked_tensor = jnp.stack([layers[i] for i in sorted_indices], axis=0)
        flax_state_dict[flax_key] = jax.device_put(stacked_tensor, device=cpu)

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
  import jax.numpy as jnp

  h_latent = height // 8
  w_latent = width // 8

  # 1. Reshape to split spatial grid and packed channel blocks
  latents = jnp.reshape(latents, (batch_size, h_latent // 2, w_latent // 2, num_channels_latents, 2, 2))
  # 2. Permute dimensions back to unpacked order
  latents = jnp.transpose(latents, (0, 3, 1, 4, 2, 5))
  # 3. Flatten back to 4D unpacked latent shape
  latents = jnp.reshape(latents, (batch_size, num_channels_latents, h_latent, w_latent))
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
  Generates sequence index coordinate identifiers for text prompt tokens to build RoPE grids.
  Shape: (batch_size, seq_len, 4)
  """
  import jax.numpy as jnp

  # Text ids: [batch, seq_len, 4]. Fill text token index.
  text_ids = jnp.zeros((seq_len, 4))
  # The first element is the frame index (0). The 4th is text sequence index.
  text_ids = text_ids.at[..., 3].set(jnp.arange(seq_len))
  return jnp.tile(text_ids[None, ...], (batch_size, 1, 1))


# -----------------------------------------------------------------------------
# Parameter In-place Casting Helper
# -----------------------------------------------------------------------------


def cast_dict_to_bfloat16_inplace(d, device=None, exclude_keywords=None, parent_key=""):
  """Casts a nested dictionary of JAX/numpy arrays to bfloat16 in-place, freeing memory immediately.

  Optionally keeps parameters matching any keyword in `exclude_keywords` in float32 for numerical stability.
  """
  import gc
  import jax.numpy as jnp

  for k, v in list(d.items()):
    current_key = f"{parent_key}.{k}" if parent_key else str(k)
    if isinstance(v, dict):
      cast_dict_to_bfloat16_inplace(v, device=device, exclude_keywords=exclude_keywords, parent_key=current_key)
    elif hasattr(v, "astype"):
      is_excluded = exclude_keywords and any(kw.lower() in current_key.lower() for kw in exclude_keywords)
      target_dtype = jnp.float32 if is_excluded else jnp.bfloat16

      if v.dtype != target_dtype:
        d[k] = v.astype(target_dtype)
        if hasattr(d[k], "block_until_ready"):
          d[k].block_until_ready()
        del v
        gc.collect()


# -----------------------------------------------------------------------------
# Safetensors Weight Loader & Key Converter Functions
# -----------------------------------------------------------------------------


def load_and_convert_flux_klein_weights(safetensors_path, params, num_double_layers, num_single_layers, dtype=None, pt_state_dict=None):
  """
  Loads weights from safetensors via zero-copy safetensors.numpy and converts them to JAX parameter dictionary.
  Supports dynamic layer counts (double and single stream blocks) and sharded safetensors directories.
  """
  from safetensors.numpy import load_file
  import numpy as np
  import jax.numpy as jnp
  import glob
  import os
  import gc

  if pt_state_dict is None:
    pt_state_dict = {}
    if os.path.isdir(safetensors_path):
      shards = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
      max_logging.log(f"Loading sharded weights from directory: {safetensors_path} (Found {len(shards)} shards)...")
      for shard in sorted(shards):
        max_logging.log(f"Loading shard: {shard}...")
        pt_state_dict.update(load_file(shard))
    else:
      max_logging.log(f"Loading weights from: {safetensors_path}")
      pt_state_dict = load_file(safetensors_path)

  max_logging.log("Mapping weights to JAX parameters...")

  expected_pytree = jax.tree_util.tree_map(lambda leaf: leaf, params)

  first_leaf = jax.tree_util.tree_leaves(params)[0]
  target_dtype = dtype if dtype is not None else first_leaf.dtype

  def convert_and_transpose_tensor(tensor, transpose=False, is_norm=False):
    if transpose and len(tensor.shape) == 2:
      tensor = tensor.T
    leaf_dtype = jnp.float32 if is_norm else target_dtype
    return jnp.array(tensor, dtype=leaf_dtype)

  # Global layers
  params["context_embedder"]["kernel"] = convert_and_transpose_tensor(
      pt_state_dict.pop("context_embedder.weight"), transpose=True
  )
  params["x_embedder"]["kernel"] = convert_and_transpose_tensor(pt_state_dict.pop("x_embedder.weight"), transpose=True)
  params["double_stream_modulation_img"]["kernel"] = convert_and_transpose_tensor(
      pt_state_dict.pop("double_stream_modulation_img.linear.weight"), transpose=True
  )
  params["double_stream_modulation_txt"]["kernel"] = convert_and_transpose_tensor(
      pt_state_dict.pop("double_stream_modulation_txt.linear.weight"), transpose=True
  )
  params["single_stream_modulation"]["kernel"] = convert_and_transpose_tensor(
      pt_state_dict.pop("single_stream_modulation.linear.weight"), transpose=True
  )
  params["proj_out"]["kernel"] = convert_and_transpose_tensor(pt_state_dict.pop("proj_out.weight"), transpose=True)

  # norm_out
  params["norm_out"]["linear"]["kernel"] = convert_and_transpose_tensor(
      pt_state_dict.pop("norm_out.linear.weight"), transpose=True
  )

  # time_text_embed (Timestep Embedding)
  if "time_guidance_embed.timestep_embedder.linear_1.weight" in pt_state_dict:
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_1"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_1.weight"), transpose=True
    )
  if "time_guidance_embed.timestep_embedder.linear_1.bias" in pt_state_dict:
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_1"]["bias"] = convert_and_transpose_tensor(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_1.bias")
    )
  if "time_guidance_embed.timestep_embedder.linear_2.weight" in pt_state_dict:
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_2"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_2.weight"), transpose=True
    )
  if "time_guidance_embed.timestep_embedder.linear_2.bias" in pt_state_dict:
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_2"]["bias"] = convert_and_transpose_tensor(
        pt_state_dict.pop("time_guidance_embed.timestep_embedder.linear_2.bias")
    )

  # Double Blocks
  max_logging.log(f"Mapping {num_double_layers} double-stream attention blocks...")
  for block_idx in range(num_double_layers):
    jax_db = params[f"double_blocks_{block_idx}"]
    prefix = f"transformer_blocks.{block_idx}."

    # Concatenate QKV projections
    to_q = pt_state_dict.pop(prefix + "attn.to_q.weight").T
    to_k = pt_state_dict.pop(prefix + "attn.to_k.weight").T
    to_v = pt_state_dict.pop(prefix + "attn.to_v.weight").T
    jax_db["attn"]["i_qkv"]["kernel"] = jnp.array(np.concatenate([to_q, to_k, to_v], axis=1), dtype=target_dtype)

    add_q = pt_state_dict.pop(prefix + "attn.add_q_proj.weight").T
    add_k = pt_state_dict.pop(prefix + "attn.add_k_proj.weight").T
    add_v = pt_state_dict.pop(prefix + "attn.add_v_proj.weight").T
    jax_db["attn"]["e_qkv"]["kernel"] = jnp.array(np.concatenate([add_q, add_k, add_v], axis=1), dtype=target_dtype)

    # Projections out
    jax_db["attn"]["i_proj"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "attn.to_out.0.weight"), transpose=True
    )
    jax_db["attn"]["e_proj"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "attn.to_add_out.weight"), transpose=True
    )

    # Norm scales
    jax_db["attn"]["query_norm"]["scale"] = convert_and_transpose_tensor(pt_state_dict.pop(prefix + "attn.norm_q.weight"))
    jax_db["attn"]["key_norm"]["scale"] = convert_and_transpose_tensor(pt_state_dict.pop(prefix + "attn.norm_k.weight"))
    jax_db["attn"]["encoder_query_norm"]["scale"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "attn.norm_added_q.weight")
    )
    jax_db["attn"]["encoder_key_norm"]["scale"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "attn.norm_added_k.weight")
    )

    # SwiGLU MLPs
    jax_db["ff"]["linear_in"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "ff.linear_in.weight"), transpose=True
    )
    jax_db["ff"]["linear_out"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "ff.linear_out.weight"), transpose=True
    )
    jax_db["ff_context"]["linear_in"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "ff_context.linear_in.weight"), transpose=True
    )
    jax_db["ff_context"]["linear_out"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(prefix + "ff_context.linear_out.weight"), transpose=True
    )

  # Single Blocks
  max_logging.log(f"Mapping {num_single_layers} single-stream attention blocks...")
  for block_idx in range(num_single_layers):
    jax_sb = params[f"single_blocks_{block_idx}"]
    s_prefix = f"single_transformer_blocks.{block_idx}."

    # Joint projections
    jax_sb["linear1"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(s_prefix + "attn.to_qkv_mlp_proj.weight"), transpose=True
    )
    jax_sb["linear2"]["kernel"] = convert_and_transpose_tensor(
        pt_state_dict.pop(s_prefix + "attn.to_out.weight"), transpose=True
    )

    # Norm scales
    jax_sb["attn"]["query_norm"]["scale"] = convert_and_transpose_tensor(pt_state_dict.pop(s_prefix + "attn.norm_q.weight"))
    jax_sb["attn"]["key_norm"]["scale"] = convert_and_transpose_tensor(pt_state_dict.pop(s_prefix + "attn.norm_k.weight"))

  params = jax.tree_util.tree_map(
      lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, params
  )
  del pt_state_dict
  gc.collect()
  max_logging.log("Validating converted Flax PyTree state dict structure...")
  validate_flax_state_dict(expected_pytree, params)
  max_logging.log("Weight conversion complete & verified!")
  return params


def load_and_convert_vae_weights(safetensors_path, jax_params, dtype=None, pt_state_dict=None):
  """Loads VAE weights from safetensors via zero-copy safetensors.numpy, maps them to JAX, and extracts BN stats."""
  from safetensors.numpy import load_file
  import flax
  import jax.numpy as jnp

  if pt_state_dict is None:
    max_logging.log(f"Loading VAE weights from: {safetensors_path}")
    pt_state_dict = load_file(safetensors_path)

  # Unfreeze JAX params so we can load the weights
  jax_params = flax.core.unfreeze(jax_params)

  first_leaf = jax.tree_util.tree_leaves(jax_params)[0]
  target_dtype = dtype if dtype is not None else first_leaf.dtype

  def get_pytorch_weight_tensor(key, dtype_val=target_dtype):
    tensor = pt_state_dict[key]
    is_norm = any(kw in key.lower() for kw in ("norm", "layernorm", "rmsnorm", "groupnorm"))
    leaf_dtype = jnp.float32 if is_norm else dtype_val
    return jnp.array(tensor, dtype=leaf_dtype)

  # Map weights
  max_logging.log("Mapping VAE decoder weights to JAX parameters...")

  # post_quant_conv
  jax_params["post_quant_conv"]["kernel"] = jnp.array(
      get_pytorch_weight_tensor("post_quant_conv.weight").transpose(2, 3, 1, 0)
  )
  jax_params["post_quant_conv"]["bias"] = jnp.array(get_pytorch_weight_tensor("post_quant_conv.bias"))

  # decoder.conv_in
  jax_params["decoder"]["conv_in"]["kernel"] = jnp.array(
      get_pytorch_weight_tensor("decoder.conv_in.weight").transpose(2, 3, 1, 0)
  )
  jax_params["decoder"]["conv_in"]["bias"] = jnp.array(get_pytorch_weight_tensor("decoder.conv_in.bias"))

  # decoder.mid_block
  # resnets
  for idx in [0, 1]:
    res_jax = jax_params["decoder"]["mid_block"][f"resnets_{idx}"]
    res_pt_prefix = f"decoder.mid_block.resnets.{idx}"

    res_jax["norm1"]["scale"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.norm1.weight"))
    res_jax["norm1"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.norm1.bias"))
    res_jax["conv1"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.conv1.weight").transpose(2, 3, 1, 0))
    res_jax["conv1"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.conv1.bias"))

    res_jax["norm2"]["scale"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.norm2.weight"))
    res_jax["norm2"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.norm2.bias"))
    res_jax["conv2"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.conv2.weight").transpose(2, 3, 1, 0))
    res_jax["conv2"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt_prefix}.conv2.bias"))

  # attentions
  attn_pt_prefix = "decoder.mid_block.attentions.0"
  attn_jax = jax_params["decoder"]["mid_block"]["attentions_0"]

  attn_jax["group_norm"]["scale"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.group_norm.weight"))
  attn_jax["group_norm"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.group_norm.bias"))

  attn_jax["query"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_q.weight").T)
  attn_jax["query"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_q.bias"))
  attn_jax["key"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_k.weight").T)
  attn_jax["key"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_k.bias"))
  attn_jax["value"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_v.weight").T)
  attn_jax["value"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_v.bias"))

  attn_jax["proj_attn"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_out.0.weight").T)
  attn_jax["proj_attn"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{attn_pt_prefix}.to_out.0.bias"))

  # decoder.up_blocks
  for b_idx in range(4):
    up_block_jax = jax_params["decoder"][f"up_blocks_{b_idx}"]
    up_block_pt = f"decoder.up_blocks.{b_idx}"

    for r_idx in range(3):
      res_jax = up_block_jax[f"resnets_{r_idx}"]
      res_pt = f"{up_block_pt}.resnets.{r_idx}"

      res_jax["norm1"]["scale"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.norm1.weight"))
      res_jax["norm1"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.norm1.bias"))
      res_jax["conv1"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.conv1.weight").transpose(2, 3, 1, 0))
      res_jax["conv1"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.conv1.bias"))

      res_jax["norm2"]["scale"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.norm2.weight"))
      res_jax["norm2"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.norm2.bias"))
      res_jax["conv2"]["kernel"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.conv2.weight").transpose(2, 3, 1, 0))
      res_jax["conv2"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.conv2.bias"))

      shortcut_key = f"{res_pt}.conv_shortcut.weight"
      if shortcut_key in pt_state_dict:
        res_jax["conv_shortcut"]["kernel"] = jnp.array(get_pytorch_weight_tensor(shortcut_key).transpose(2, 3, 1, 0))
        res_jax["conv_shortcut"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{res_pt}.conv_shortcut.bias"))

    if b_idx < 3:
      upsampler_jax = up_block_jax["upsamplers_0"]
      upsampler_pt = f"{up_block_pt}.upsamplers.0"

      upsampler_jax["conv"]["kernel"] = jnp.array(
          get_pytorch_weight_tensor(f"{upsampler_pt}.conv.weight").transpose(2, 3, 1, 0)
      )
      upsampler_jax["conv"]["bias"] = jnp.array(get_pytorch_weight_tensor(f"{upsampler_pt}.conv.bias"))

  # decoder.conv_norm_out & conv_out
  jax_params["decoder"]["conv_norm_out"]["scale"] = jnp.array(get_pytorch_weight_tensor("decoder.conv_norm_out.weight"))
  jax_params["decoder"]["conv_norm_out"]["bias"] = jnp.array(get_pytorch_weight_tensor("decoder.conv_norm_out.bias"))
  jax_params["decoder"]["conv_out"]["kernel"] = jnp.array(
      get_pytorch_weight_tensor("decoder.conv_out.weight").transpose(2, 3, 1, 0)
  )
  jax_params["decoder"]["conv_out"]["bias"] = jnp.array(get_pytorch_weight_tensor("decoder.conv_out.bias"))

  jax_params = jax.tree_util.tree_map(
      lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, jax_params
  )
  # Freeze parameters
  jax_params = flax.core.freeze(jax_params)

  # Extract Batch Normalization running stats
  max_logging.log("Extracting VAE Batch Normalization running stats...")
  bn_mean = jnp.array(get_pytorch_weight_tensor("bn.running_mean")).reshape(1, -1, 1, 1)
  bn_var = jnp.array(get_pytorch_weight_tensor("bn.running_var")).reshape(1, -1, 1, 1)
  batch_norm_eps = 0.0001
  bn_std = jnp.sqrt(bn_var + batch_norm_eps)

  max_logging.log("VAE weights and BN stats loaded successfully!")
  return jax_params, bn_mean, bn_std
