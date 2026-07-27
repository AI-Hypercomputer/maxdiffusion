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

"""Checkpoint conversion helpers for Diffusers Z-Image checkpoints."""

import json
import os

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors import safe_open


_PREFIX_RENAMES = (
    ("all_x_embedder.2-1.", "x_embedder."),
    ("all_final_layer.2-1.", "final_layer."),
    ("t_embedder.mlp.0.", "t_embedder.mlp_in."),
    ("t_embedder.mlp.2.", "t_embedder.mlp_out."),
    ("mlp.0.", "mlp_in."),
    ("mlp.2.", "mlp_out."),
    ("cap_embedder.0.", "cap_embedder_norm."),
    ("cap_embedder.1.", "cap_embedder."),
    ("adaln_modulation.0.", "adaln_modulation."),
    ("adaln_modulation.1.", "adaln_modulation."),
    ("adaLN_modulation.0.", "adaln_modulation."),
    ("adaLN_modulation.1.", "adaln_modulation."),
)


def z_image_pytorch_key_to_nnx_key(key: str, scan_layers: bool = False) -> tuple[tuple[str | int, ...], bool, int | None]:
  """Return the target NNX state path, whether a linear weight needs transpose, and layer index if scanned."""
  for source, destination in _PREFIX_RENAMES:
    if key.startswith(source):
      key = destination + key[len(source) :]
      break
  key = key.replace(".adaln_modulation.1.", ".adaln_modulation.")
  key = key.replace(".adaln_modulation.0.", ".adaln_modulation.")
  key = key.replace(".adaLN_modulation.1.", ".adaln_modulation.")
  key = key.replace(".adaLN_modulation.0.", ".adaln_modulation.")
  key = key.replace(".attention.to_out.0.", ".attention.to_out.")
  key = key.replace("attention.to_out.0.", "attention.to_out.")
  key = key.replace(".weight", ".kernel")
  key = key.replace(".norm_final.kernel", ".norm_final.scale")
  key = key.replace(".attention_norm1.kernel", ".attention_norm1.scale")
  key = key.replace(".attention_norm2.kernel", ".attention_norm2.scale")
  key = key.replace(".ffn_norm1.kernel", ".ffn_norm1.scale")
  key = key.replace(".ffn_norm2.kernel", ".ffn_norm2.scale")
  key = key.replace(".norm_q.kernel", ".norm_q.scale")
  key = key.replace(".norm_k.kernel", ".norm_k.scale")
  key = key.replace("cap_embedder_norm.kernel", "cap_embedder_norm.scale")
  for norm_name in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2", "norm_q", "norm_k"):
    if key == f"{norm_name}.kernel":
      key = f"{norm_name}.scale"
  path = tuple(int(part) if part.isdigit() else part for part in key.split("."))

  block_index = None
  if scan_layers:
    if len(path) >= 2 and path[0] in ("layers", "noise_refiner", "context_refiner") and isinstance(path[1], int):
      block_index = path[1]
      path = (path[0],) + path[2:]

  # Every converted .kernel is a torch Linear weight. Norm scales and pad
  # tokens deliberately bypass this path.
  return path, path[-1] == "kernel", block_index


def load_z_image_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str = "cpu",
    hf_download: bool = True,
    subfolder: str = "transformer",
    target_shardings: dict | None = None,
    scan_layers: bool = False,
) -> dict:
  """Load and convert a Diffusers Z-Image transformer into an NNX parameter tree.

  Shards are streamed one at a time.  This avoids retaining the 12B PyTorch
  state dict alongside the converted JAX tree on the host.
  """
  index_name = "diffusion_pytorch_model.safetensors.index.json"
  if os.path.isdir(pretrained_model_name_or_path):
    index_path = os.path.join(pretrained_model_name_or_path, subfolder, index_name)
  elif hf_download:
    index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=index_name)
  else:
    raise ValueError("A local model path is required when hf_download is False.")
  with open(index_path, encoding="utf-8") as handle:
    weight_map = json.load(handle)["weight_map"]

  expected = flatten_dict(eval_shapes)
  converted = {}
  default_device = jax.devices()[0]
  for filename in sorted(set(weight_map.values())):
    path = (
        os.path.join(pretrained_model_name_or_path, subfolder, filename)
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
    )
    with safe_open(path, framework="pt", device="cpu") as tensors:
      for source_key in tensors.keys():
        target_key, transpose, block_index = z_image_pytorch_key_to_nnx_key(source_key, scan_layers=scan_layers)
        if target_key not in expected:
          raise KeyError(f"Z-Image checkpoint key `{source_key}` maps to unknown NNX key `{target_key}`.")
        value = tensors.get_tensor(source_key).float().numpy()
        if transpose:
          value = value.T
        value = jnp.asarray(value, dtype=expected[target_key].dtype)

        exp_spec = expected[target_key]
        if block_index is not None:
          if value.shape != exp_spec.shape[1:]:
            raise ValueError(f"Shape mismatch for slice `{source_key}`: {value.shape} != {exp_spec.shape[1:]}.")
          if target_key not in converted:
            converted[target_key] = jnp.zeros(exp_spec.shape, dtype=exp_spec.dtype)
          converted[target_key] = converted[target_key].at[block_index].set(value)
        else:
          if value.shape != exp_spec.shape:
            raise ValueError(f"Shape mismatch for `{source_key}`: {value.shape} != {exp_spec.shape}.")
          converted[target_key] = value

  # Place converted parameters onto target shardings or devices
  for target_key, value in converted.items():
    target = target_shardings.get(target_key, default_device) if target_shardings else default_device
    converted[target_key] = jax.device_put(value, target)

  missing = set(expected) - set(converted)
  if missing:
    raise ValueError(f"Z-Image checkpoint is missing {len(missing)} parameters, e.g. {sorted(missing)[:5]}.")
  return unflatten_dict(converted)
