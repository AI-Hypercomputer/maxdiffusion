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

"""Checkpoint conversion helpers for PyTorch Qwen3 checkpoints.

Kept apart from `qwen3_flax`, which is the model definition alone.
"""

from typing import Optional, Tuple

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import max_logging
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config


def qwen3_flax_key_to_pytorch_key(path: Tuple[str, ...]) -> Tuple[str, bool]:
  """Return the PyTorch key feeding a Flax param path, and whether it transposes.

  Handles both module layouts: NNX nests its blocks in an `nnx.List`, giving
  ('layers', 0, ...), while Linen names them flatly, giving ('layers_0', ...).
  Every Flax `.kernel` is a torch `nn.Linear` weight and needs a transpose;
  norm scales and the embedding table do not.
  """
  if path[0] == "embed_tokens":
    return "model.embed_tokens.weight", False
  if path[0] == "norm":
    return "model.norm.weight", False
  if path[0] == "layers":  # NNX: ('layers', index, ...)
    layer_index, inner = path[1], path[2:]
  elif path[0].startswith("layers_"):  # Linen: ('layers_{index}', ...)
    layer_index, inner = path[0].split("_")[1], path[1:]
  else:
    raise KeyError(f"Unknown Qwen3 Flax parameter path `{path}`.")
  return f"model.layers.{layer_index}." + ".".join(inner[:-1]) + ".weight", path[-1] == "kernel"


def load_qwen3_weights(
    safetensors_path: str,
    eval_shapes: dict,
    target_shardings: Optional[dict] = None,
    device: str = "cpu",
) -> dict:
  """Stream a PyTorch Qwen3 checkpoint into a Flax parameter tree.

  Streaming: tensors are read and converted one at a time through the torch
  backend, so the full PyTorch state dict is never held on the host alongside
  the converted tree, and each parameter is placed on device as soon as it is
  read. The torch backend also means bfloat16 checkpoints load, which
  `load_and_convert_qwen3_weights` below cannot do (numpy has no bfloat16).
  Each parameter takes the dtype and sharding of its entry in `eval_shapes`.
  """
  import glob
  import os
  from safetensors import safe_open

  if os.path.isdir(safetensors_path):
    shards = sorted(glob.glob(os.path.join(safetensors_path, "*.safetensors")))
  else:
    shards = [safetensors_path]
  if not shards:
    raise ValueError(f"No safetensors found in {safetensors_path}")

  expected = flatten_dict(eval_shapes)
  sources = {}
  for path in expected:
    source_key, transpose = qwen3_flax_key_to_pytorch_key(path)
    sources[source_key] = (path, transpose)

  converted = {}
  cpu = jax.local_devices(backend=device)[0]
  for shard in shards:
    max_logging.log(f"Loading Qwen3 shard: {os.path.basename(shard)}...")
    with safe_open(shard, framework="pt", device="cpu") as tensors:
      for source_key in tensors.keys():
        # Checkpoints carry tensors this text-encoder stack does not use
        # (lm_head, and rotary buffers), which are skipped here.
        if source_key not in sources:
          continue
        target_key, transpose = sources[source_key]
        value = tensors.get_tensor(source_key).float().numpy()
        if transpose:
          value = value.T
        value = jnp.asarray(value, dtype=expected[target_key].dtype)
        if value.shape != expected[target_key].shape:
          raise ValueError(f"Shape mismatch for `{source_key}`: {value.shape} != {expected[target_key].shape}.")
        target = target_shardings.get(target_key) if target_shardings is not None else cpu
        converted[target_key] = jax.device_put(value, target)

  missing = set(expected) - set(converted)
  if missing:
    raise ValueError(f"Qwen3 checkpoint is missing {len(missing)} parameters, e.g. {sorted(missing)[:5]}.")
  return unflatten_dict(converted)


def load_and_convert_qwen3_weights(safetensors_path: str, jax_params: dict, config: FlaxQwen3Config) -> dict:
  """
  Loads weights from safetensors via zero-copy safetensors.numpy and converts them to JAX parameter dictionary.
  """
  import glob
  import os
  from safetensors.numpy import load_file

  torch_weights: dict = {}
  if os.path.isdir(safetensors_path):
    # Find all safetensors shards
    shards = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
    max_logging.log(f"Loading sharded Qwen3 weights from directory: {safetensors_path} (Found {len(shards)} shards)...")
    for shard in sorted(shards):
      max_logging.log(f"Loading shard: {shard}...")
      torch_weights.update(load_file(shard))
  else:
    # Single file path
    max_logging.log(f"Loading Qwen3 weights from file: {safetensors_path}...")
    torch_weights = load_file(safetensors_path)
  max_logging.log("Safetensors weights loaded successfully. Starting JAX parameter mapping...")

  # Helper to transpose and cast weight
  def get_w(name: str, transpose: bool = True) -> np.ndarray:
    nonlocal torch_weights
    if name not in torch_weights:
      raise KeyError(f"Weight '{name}' not found in safetensors!")
    t = torch_weights[name]
    if len(t.shape) == 2 and transpose:
      t = t.T
    return t

  # Create mutable copy of JAX params to populate
  import flax

  flat_params = flax.traverse_util.flatten_dict(jax_params)
  converted_flat = {}

  for k, v in flat_params.items():
    # Reconstruct path string for debugging/matching
    path_str = ".".join(k)

    # 1. Token Embeddings
    if k[0] == "embed_tokens" and k[1] == "embedding":
      converted_flat[k] = get_w("model.embed_tokens.weight", transpose=False)

    # 2. Decoder Layer Normalizations (RMSNorm)
    elif "input_layernorm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.input_layernorm.weight")

    elif "post_attention_layernorm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

    # 3. Attention Projections & QK-Norm
    elif "self_attn" in path_str and k[-1] == "kernel":
      layer_idx = k[0].split("_")[1]
      proj_name = k[2]  # q_proj, k_proj, v_proj, o_proj
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.{proj_name}.weight")

    elif "self_attn" in path_str and "q_norm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.q_norm.weight")

    elif "self_attn" in path_str and "k_norm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.k_norm.weight")

    # 4. MLP Block
    elif "mlp" in path_str and k[-1] == "kernel":
      layer_idx = k[0].split("_")[1]
      proj_name = k[2]  # gate_proj, up_proj, down_proj
      converted_flat[k] = get_w(f"model.layers.{layer_idx}.mlp.{proj_name}.weight")

    # 5. Final RMSNorm
    elif k[0] == "norm" and k[1] == "weight":
      converted_flat[k] = get_w("model.norm.weight")

    else:
      max_logging.log(f"WARNING: JAX parameter '{path_str}' did not match any PyTorch weights!")
      converted_flat[k] = np.zeros(v.shape, dtype=np.float32) if hasattr(v, "shape") and not isinstance(v, np.ndarray) else v

  # Clean up PyTorch memory immediately
  del torch_weights
  import gc

  gc.collect()

  res = flax.traverse_util.unflatten_dict(converted_flat)
  return jax.tree_util.tree_map(
      lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, res
  )
