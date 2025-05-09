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
