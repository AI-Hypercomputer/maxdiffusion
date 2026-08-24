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

"""Weight conversion and streaming checkpoint loader for PRXPixel Transformer."""

import glob
import os
from typing import Dict, List, Optional, Tuple, Any

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from maxdiffusion import max_logging


def prx_pixel_pytorch_key_to_nnx_key(path: Tuple[Any, ...]) -> Tuple[str, bool]:
  """Maps Flax NNX parameter tuple path to PyTorch checkpoint key and transpose flag.

  Linear weights (`kernel`) are transposed from PyTorch [out_features, in_features] to JAX [in_features, out_features].
  Biases and RMSNorm scales are not transposed.
  """
  p0 = path[0]

  # 1. Image input bottleneck
  if p0 == "img_in":
    idx = path[1]
    param_type = path[2]
    suffix = "weight" if param_type == "kernel" else "bias"
    return f"img_in.{idx}.{suffix}", param_type == "kernel"

  # 2. Text input projection
  if p0 == "txt_in":
    param_type = path[1]
    suffix = "weight" if param_type == "kernel" else "bias"
    return f"txt_in.{suffix}", param_type == "kernel"

  # 3. Timestep embedding
  if p0 == "time_in":
    layer = path[1]
    param_type = path[2]
    suffix = "weight" if param_type == "kernel" else "bias"
    return f"time_in.{layer}.{suffix}", param_type == "kernel"

  # 4. Resolution embedder
  if p0 == "resolution_embedder":
    # path: ('resolution_embedder', 'mlp', 'in_layer'/'out_layer', 'kernel'/'bias')
    layer = path[2]
    param_type = path[3]
    suffix = "weight" if param_type == "kernel" else "bias"
    return f"resolution_embedder.mlp.{layer}.{suffix}", param_type == "kernel"

  # 5. Final layer
  if p0 == "final_layer":
    if path[1] == "linear":
      param_type = path[2]
      suffix = "weight" if param_type == "kernel" else "bias"
      return f"final_layer.linear.{suffix}", param_type == "kernel"
    elif path[1] == "adaLN_modulation":
      # path: ('final_layer', 'adaLN_modulation', 0, 'kernel'/'bias')
      param_type = path[3]
      suffix = "weight" if param_type == "kernel" else "bias"
      return f"final_layer.adaLN_modulation.1.{suffix}", param_type == "kernel"

  # 6. Transformer Blocks
  if p0 == "blocks":
    block_idx = path[1]
    sub = path[2]

    if sub == "attention":
      proj = path[3]
      if proj == "to_out":
        param_type = path[5]
        return f"blocks.{block_idx}.attention.to_out.0.weight", True
      elif proj in ("img_qkv_proj", "txt_kv_proj"):
        return f"blocks.{block_idx}.attention.{proj}.weight", True
      elif proj in ("norm_q", "norm_k", "norm_added_k"):
        return f"blocks.{block_idx}.attention.{proj}.weight", False

    elif sub in ("gate_proj", "up_proj", "down_proj"):
      return f"blocks.{block_idx}.{sub}.weight", True

    elif sub == "modulation":
      param_type = path[4]
      suffix = "weight" if param_type == "kernel" else "bias"
      return f"blocks.{block_idx}.modulation.lin.{suffix}", param_type == "kernel"

  raise KeyError(f"Unknown PRXPixel NNX parameter path: {path}")


def load_prx_pixel_weights(
    safetensors_path: str,
    eval_shapes: dict,
    target_shardings: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
  """Stream-loads PyTorch PRXPixel safetensors checkpoint into Flax NNX parameter tree."""
  if os.path.isdir(safetensors_path):
    shards = sorted(glob.glob(os.path.join(safetensors_path, "*.safetensors")))
    if not shards:
      # Check subfolder
      shards = sorted(glob.glob(os.path.join(safetensors_path, "transformer/*.safetensors")))
  else:
    shards = [safetensors_path]

  if not shards:
    raise ValueError(f"No safetensors found in {safetensors_path}")

  expected = flatten_dict(eval_shapes)
  sources = {}
  for path in expected:
    source_key, transpose = prx_pixel_pytorch_key_to_nnx_key(path)
    sources[source_key] = (path, transpose)

  converted = {}
  dev_target = jax.local_devices(backend=device)[0] if device is not None else jax.local_devices()[0]

  for shard in shards:
    max_logging.log(f"Loading PRXPixel shard: {os.path.basename(shard)}...")
    with safe_open(shard, framework="pt", device="cpu") as tensors:
      for source_key in tensors.keys():
        if source_key not in sources:
          continue
        target_key, transpose = sources[source_key]
        value = tensors.get_tensor(source_key).float().numpy()
        if transpose:
          value = value.T
        value = jnp.asarray(value, dtype=expected[target_key].dtype)
        if value.shape != expected[target_key].shape:
          raise ValueError(
              f"Shape mismatch for `{source_key}`: {value.shape} != {expected[target_key].shape}"
          )
        target = target_shardings.get(target_key) if target_shardings is not None else dev_target
        converted[target_key] = jax.device_put(value, target)

  missing = set(expected) - set(converted)
  if missing:
    raise ValueError(f"PRXPixel checkpoint missing {len(missing)} parameters! Examples: {sorted(missing)[:5]}")

  max_logging.log(f"✅ Successfully converted all {len(converted)} PRXPixel transformer parameters!")
  return unflatten_dict(converted)
