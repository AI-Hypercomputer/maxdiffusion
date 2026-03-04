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

import re
import torch
import jax
from jax import dlpack
import jax.numpy as jnp
from flax import nnx
from .. import max_logging
import numpy as np

# -----------------------------------------------------------------------------
# JIT Helpers (The Fix for Sharding & Device-Side Computation)
# -----------------------------------------------------------------------------


@jax.jit
def _compute_and_add_single_jit(kernel, bias, down, up, scale, w_diff, b_diff):
  """
  Applies LoRA + Weight Diff + Bias Diff on device.
  """
  # 1. Apply LoRA (if valid)
  if down is not None and up is not None:
    # down: (Rank, In), up: (Out, Rank) -> Result: (In, Out)
    # Note: We reshape to kernel shape to handle 1x1 convs
    delta = (down.T @ up.T).reshape(kernel.shape)
    kernel = kernel + (delta * scale).astype(kernel.dtype)

  # 2. Apply Full Weight Diff (if valid)
  if w_diff is not None:
    kernel = kernel + w_diff.astype(kernel.dtype)

  # 3. Apply Bias Diff (if valid and bias exists)
  if bias is not None and b_diff is not None:
    bias = bias + b_diff.astype(bias.dtype)

  return kernel, bias


@jax.jit
def _compute_and_add_scanned_jit(kernel, downs, ups, alphas, global_scale, w_diffs=None, b_diffs=None, bias=None):
  """
  Applies scanned LoRA + Diffs.
  """
  # 1. Apply LoRA
  if downs is not None and ups is not None:
    rank = downs.shape[1]
    scales = global_scale * alphas / rank
    # Batch Matmul: (L, In, Out)
    delta = jnp.matmul(jnp.swapaxes(downs, 1, 2), jnp.swapaxes(ups, 1, 2))
    delta = (delta * scales).astype(kernel.dtype)
    kernel = kernel + delta.reshape(kernel.shape)

  # 2. Apply Scanned Weight Diffs (L, ...)
  if w_diffs is not None:
    kernel = kernel + w_diffs.astype(kernel.dtype)

  # 3. Apply Scanned Bias Diffs (L, ...)
  # Note: Scanned bias is usually shape (L, Out)
  if bias is not None and b_diffs is not None:
    bias = bias + b_diffs.astype(bias.dtype)

  return kernel, bias


# -----------------------------------------------------------------------------


def _to_jax_array(v, dtype):
  jax_dtype = jnp.dtype(dtype)
  if isinstance(v, torch.Tensor):
    return dlpack.from_dlpack(v).astype(jax_dtype)
  return jnp.array(v, dtype=jax_dtype)


def parse_lora_dict(state_dict, dtype):
  """
  Helper to parse state_dict into structured params including diffs.
  Supports keys in original WAN LoRA format, like:
  '...lora_A.weight', '...lora_up.weight', '...alpha',
  as well as '.diff' and '.diff_b' for weight and bias fine-tuning.
  Supports ComfyUI and AI toolkit lora formats.
  """
  lora_params = {}
  for k, v in state_dict.items():
    # Alpha
    if k.endswith(".alpha"):
      key_base = k[: -len(".alpha")]
      if key_base not in lora_params:
        lora_params[key_base] = {}
      lora_params[key_base]["alpha"] = _to_jax_array(v, dtype=dtype)
      continue

    # Bias Diff (e.g., "layer.diff_b")
    if k.endswith(".diff_b"):
      key_base = k[: -len(".diff_b")]
      if key_base not in lora_params:
        lora_params[key_base] = {}
      lora_params[key_base]["diff_b"] = _to_jax_array(v, dtype=dtype)
      continue

    # Weight Diff (e.g., "layer.diff")
    if k.endswith(".diff"):
      key_base = k[: -len(".diff")]
      if key_base not in lora_params:
        lora_params[key_base] = {}
      lora_params[key_base]["diff"] = _to_jax_array(v, dtype=dtype)
      continue

    # Standard LoRA
    m = re.match(r"^(.*?)_lora\.(down|up)\.weight$", k)
    if not m:
      m = re.match(r"^(.*?)\.lora\.(down|up)\.weight$", k)
    if not m:
      m = re.match(r"^(.*?)\.(lora_down|lora_up)\.weight$", k)
    if not m:
      m = re.match(r"^(.*?)\.(lora_A|lora_B)\.weight$", k)

    if m:
      key_base, weight_type = m.group(1), m.group(2).replace("lora_", "")
      if weight_type == "A":
        weight_type = "down"
      elif weight_type == "B":
        weight_type = "up"
      if key_base not in lora_params:
        lora_params[key_base] = {}
      lora_params[key_base][weight_type] = _to_jax_array(v, dtype=dtype)
    else:
      # Fallback for exact matches of diffs if regex failed above
      max_logging.log(f"Key {k} did not match any LoRA pattern.")
      pass

  return lora_params


def _merge_lora_layer(module, weights, scale):
  """Merges LoRA weights into a single non-scanned layer."""
  is_conv_kxk_locon = False
  if isinstance(module, nnx.Conv) and module.kernel_size != (1, 1) and "down" in weights and "up" in weights:
    is_conv_kxk_locon = True

  updated = False
  # Handle Embeddings
  if isinstance(module, nnx.Embed):
    if "diff" in weights and hasattr(module, "embedding"):
      module.embedding.value += np.array(weights["diff"]).reshape(module.embedding.shape).astype(module.embedding.dtype)
      updated = True
  # Handle Norms
  elif isinstance(module, (nnx.LayerNorm, nnx.RMSNorm)):
    scale_diff = weights.get("diff", None)
    bias_diff = weights.get("diff_b", None)
    if scale_diff is not None and hasattr(module, "scale") and module.scale is not None:
      module.scale.value += np.array(scale_diff).reshape(module.scale.shape).astype(module.scale.dtype)
      updated = True
    if bias_diff is not None and isinstance(module, nnx.LayerNorm) and hasattr(module, "bias") and module.bias is not None:
      module.bias.value += np.array(bias_diff).reshape(module.bias.shape).astype(module.bias.dtype)
      updated = True
  elif isinstance(module, nnx.Param):
    if "diff" in weights:
      module.value += np.array(weights["diff"]).reshape(module.shape).astype(module.dtype)
      updated = True
  elif isinstance(module, (nnx.Linear, nnx.Conv)):
    # Prepare LoRA terms
    down_w, up_w, current_scale = None, None, None
    if "down" in weights and "up" in weights and not is_conv_kxk_locon:
      down_w, up_w = weights["down"], weights["up"]
      down_w, up_w = np.array(down_w), np.array(up_w)  # CPU convert

      # Squeeze dimensions if needed (Conv 1x1 or Linear)
      if isinstance(module, nnx.Conv) and module.kernel_size == (1, 1):
        down_w, up_w = np.squeeze(down_w), np.squeeze(up_w)

      rank = down_w.shape[0] if down_w.ndim > 0 else 0
      alpha = float(weights.get("alpha", rank))
      current_scale = scale * alpha / rank

    # Prepare Diff terms
    w_diff = weights.get("diff", None)
    b_diff = weights.get("diff_b", None)

    if w_diff is not None:
      w_diff = np.array(w_diff)
      # Transpose weights from PyTorch OIHW/OIDHW to Flax HWIO/DHWIO if needed.
      if isinstance(module, nnx.Conv):
        if w_diff.ndim == 5:
          w_diff = w_diff.transpose((2, 3, 4, 1, 0))
        elif w_diff.ndim == 4:
          w_diff = w_diff.transpose((2, 3, 1, 0))
      elif isinstance(module, nnx.Linear) and w_diff.ndim == 2:
        w_diff = w_diff.transpose((1, 0))
    if b_diff is not None:
      b_diff = np.array(b_diff)

    # If LoCON, compute delta and add to w_diff
    if is_conv_kxk_locon:
      dw, uw = np.array(weights["down"]), np.array(weights["up"])
      rank, in_c, *k_dims = dw.shape
      out_c = uw.shape[0]
      alpha = float(weights.get("alpha", rank))

      delta_pt = (uw.reshape(out_c, rank) @ dw.reshape(rank, -1)).reshape(out_c, in_c, *k_dims)

      # Transpose to flax
      if delta_pt.ndim == 5:
        delta_fx = delta_pt.transpose((2, 3, 4, 1, 0))
      else:
        delta_fx = delta_pt.transpose((2, 3, 1, 0))

      lora_delta = delta_fx * (scale * alpha / rank)
      if w_diff is None:
        w_diff = lora_delta.astype(np.float32)
      else:
        w_diff += lora_delta.astype(w_diff.dtype)

    # Check for Bias existence
    bias_val = module.bias.value if module.bias is not None else None

    # --- EXECUTE JIT UPDATE ---
    if down_w is not None or w_diff is not None or b_diff is not None:
      new_kernel, new_bias = _compute_and_add_single_jit(
          module.kernel.value, bias_val, down_w, up_w, current_scale, w_diff, b_diff
      )

      module.kernel.value = new_kernel
      if new_bias is not None:
        module.bias.value = new_bias

      updated = True
    else:
      max_logging.log("Matched key but found no actionable weights.")
  return updated


def merge_lora(model: nnx.Module, state_dict: dict, rank: int, scale: float, translate_fn=None, dtype: str = "float32"):
  """
  Merges weights for non-scanned layers (Embeddings, singular Dense, etc).
  Now supports diff and diff_b.
  """
  lora_params = parse_lora_dict(state_dict, dtype=dtype)
  max_logging.log(f"Parsed {len(lora_params)} unique module keys.")
  matched_keys = set()

  assigned_count = 0
  for path, module in nnx.iter_graph(model):
    if not isinstance(module, (nnx.Linear, nnx.Conv, nnx.LayerNorm, nnx.RMSNorm, nnx.Embed, nnx.Param)):
      continue

    nnx_path_str = ".".join(map(str, path))
    lora_key = translate_fn(nnx_path_str) if translate_fn else None

    if lora_key and lora_key in lora_params:
      matched_keys.add(lora_key)
      weights = lora_params[lora_key]
      if _merge_lora_layer(module, weights, scale):
        assigned_count += 1

  max_logging.log(f"Merged weights into {assigned_count} layers.")
  unmatched_keys = set(lora_params.keys()) - matched_keys
  if unmatched_keys:
    max_logging.log(
        f"{len(unmatched_keys)} key(s) in LoRA dictionary were not applied to any layer in the model: {unmatched_keys}"
    )


def merge_lora_for_scanned(
    model: nnx.Module, state_dict: dict, rank: int, scale: float, translate_fn=None, dtype: str = "float32"
):
  """
  Device-Side Optimized Merge for Scanned Layers.
  Now supports diff and diff_b.
  """
  lora_params = parse_lora_dict(state_dict, dtype=dtype)
  max_logging.log(f"Parsed {len(lora_params)} keys for scanned merge.")
  matched_keys = set()

  assigned_count = 0
  for path, module in nnx.iter_graph(model):
    if not isinstance(module, (nnx.Linear, nnx.Conv, nnx.LayerNorm, nnx.RMSNorm, nnx.Embed, nnx.Param)):
      continue

    nnx_path_str = ".".join(map(str, path))
    lora_key_template = translate_fn(nnx_path_str) if translate_fn else None

    if not lora_key_template:
      continue

    # Determine if layer is scanned based on parameter dimensions
    is_scanned = False
    if isinstance(module, nnx.Embed) and hasattr(module, "embedding"):
      is_scanned = module.embedding.ndim > 2
    elif isinstance(module, (nnx.LayerNorm, nnx.RMSNorm)) and hasattr(module, "scale") and module.scale is not None:
      is_scanned = module.scale.ndim > 1
    elif isinstance(module, nnx.Linear):
      is_scanned = module.kernel.ndim == 3
    elif isinstance(module, nnx.Conv):
      is_scanned = module.kernel.ndim == 5
    elif isinstance(module, nnx.Param):
      # Use template format to disambiguate: if template has {}, then it is scanned.
      is_scanned = "{}" in lora_key_template

    if not is_scanned:
      lora_key = lora_key_template
      if lora_key in lora_params:
        matched_keys.add(lora_key)
        weights = lora_params[lora_key]
        if _merge_lora_layer(module, weights, scale):
          assigned_count += 1
      continue

    # If we reach here, layer is SCANNED
    if isinstance(module, nnx.Embed):
      num_layers = module.embedding.shape[0]
      embed_diffs_to_add = np.zeros_like(module.embedding.value)
      updated = False
      for i in range(num_layers):
        lora_key = lora_key_template.format(i)
        if lora_key in lora_params:
          matched_keys.add(lora_key)
          if "diff" in lora_params[lora_key]:
            embed_diffs_to_add[i] = np.array(lora_params[lora_key]["diff"]).reshape(module.embedding.shape[1:])
            updated = True
      if updated:
        module.embedding.value += embed_diffs_to_add.astype(module.embedding.dtype)
        assigned_count += 1
    elif isinstance(module, (nnx.LayerNorm, nnx.RMSNorm)):
      num_layers = module.scale.shape[0]
      scale_diffs_to_add = np.zeros_like(module.scale.value)
      bias_diffs_to_add = (
          np.zeros_like(module.bias.value)
          if isinstance(module, nnx.LayerNorm) and hasattr(module, "bias") and module.bias is not None
          else None
      )
      updated_scale, updated_bias = False, False
      for i in range(num_layers):
        lora_key = lora_key_template.format(i)
        if lora_key in lora_params:
          matched_keys.add(lora_key)
          weights = lora_params[lora_key]
          if "diff" in weights:
            scale_diffs_to_add[i] = np.array(weights["diff"]).reshape(module.scale.shape[1:])
            updated_scale = True
          if "diff_b" in weights and bias_diffs_to_add is not None:
            bias_diffs_to_add[i] = np.array(weights["diff_b"]).reshape(module.bias.shape[1:])
            updated_bias = True
      if updated_scale:
        module.scale.value += scale_diffs_to_add.astype(module.scale.dtype)
      if updated_bias and bias_diffs_to_add is not None:
        module.bias.value += bias_diffs_to_add.astype(module.bias.dtype)
      if updated_scale or updated_bias:
        assigned_count += 1
    elif isinstance(module, nnx.Param):
      num_layers = module.shape[0]
      param_diffs_to_add = np.zeros_like(module.value)
      updated = False
      for i in range(num_layers):
        lora_key = lora_key_template.format(i)
        if lora_key in lora_params:
          matched_keys.add(lora_key)
          if "diff" in lora_params[lora_key]:
            param_diffs_to_add[i] = np.array(lora_params[lora_key]["diff"]).reshape(module.shape[1:])
            updated = True
      if updated:
        module.value += param_diffs_to_add.astype(module.dtype)
        assigned_count += 1
    elif isinstance(module, (nnx.Linear, nnx.Conv)):
      is_linear = isinstance(module, nnx.Linear)
      is_conv = isinstance(module, nnx.Conv)
      is_conv_kxk = isinstance(module, nnx.Conv) and module.kernel_size != (1, 1)
      if is_linear:
        num_layers, in_feat, out_feat = module.kernel.shape
      else:  # Conv
        num_layers = module.kernel.shape[0]
        in_feat, out_feat = module.kernel.shape[3], module.kernel.shape[4]

      # 1. Scan for Rank (Fallback use rank in config file)
      found_rank = rank
      for i in range(num_layers):
        k = lora_key_template.format(i)
        if k in lora_params and "down" in lora_params[k]:
          found_rank = lora_params[k]["down"].shape[0]
          break

      # 2. Pre-allocate Buffers (CPU)
      # LoRA Buffers
      stack_down = np.zeros((num_layers, found_rank, in_feat), dtype=np.float32)
      stack_up = np.zeros((num_layers, out_feat, found_rank), dtype=np.float32)
      stack_alpha = np.zeros((num_layers, 1, 1), dtype=np.float32)

      # Diff Buffers
      # Initialize as None, allocate only if found to save memory
      stack_w_diff = None
      stack_b_diff = None

      has_lora = False
      has_diff = False

      for i in range(num_layers):
        lora_key = lora_key_template.format(i)
        if lora_key in lora_params:
          matched_keys.add(lora_key)
          w = lora_params[lora_key]

          # --- Fill LoRA ---
          if "down" in w:
            d, u = np.array(w["down"]), np.array(w["up"])
            alpha = float(w.get("alpha", d.shape[0]))
            rank_ = d.shape[0]

            if is_conv_kxk:
              # For LoCON kxk, compute delta and merge into stack_w_diff
              rank_, in_c, *k_dims = d.shape
              out_c = u.shape[0]
              delta_pt = (u.reshape(out_c, rank_) @ d.reshape(rank_, -1)).reshape(out_c, in_c, *k_dims)
              if delta_pt.ndim == 5:
                delta_fx = delta_pt.transpose((2, 3, 4, 1, 0))
              else:
                delta_fx = delta_pt.transpose((2, 3, 1, 0))

              lora_delta = delta_fx * (scale * alpha / rank_)
              if stack_w_diff is None:
                stack_w_diff = np.zeros(module.kernel.shape, dtype=np.float32)
              stack_w_diff[i] += lora_delta.reshape(stack_w_diff[i].shape).astype(stack_w_diff.dtype)
              has_diff = True  # Mark as having diff because we merged LoRA into w_diff
            else:
              # For Linear or 1x1 Conv, prepare for JIT
              if d.ndim > 2:
                d = np.squeeze(d)
              if u.ndim > 2:
                u = np.squeeze(u)
              stack_down[i] = d
              stack_up[i] = u
              stack_alpha[i] = alpha
              has_lora = True

          # --- Fill Weight Diff ---
          if "diff" in w:
            if stack_w_diff is None:
              stack_w_diff = np.zeros(module.kernel.shape, dtype=np.float32)
            wd = np.array(w["diff"])
            # Transpose weights from PyTorch OIHW/OIDHW to Flax HWIO/DHWIO if needed.
            if is_conv:
              if wd.ndim == 5:
                wd = wd.transpose((2, 3, 4, 1, 0))
              elif wd.ndim == 4:
                wd = wd.transpose((2, 3, 1, 0))
            elif is_linear and wd.ndim == 2:
              wd = wd.transpose((1, 0))

            stack_w_diff[i] += wd.reshape(stack_w_diff[i].shape)
            has_diff = True

          # --- Fill Bias Diff ---
          if "diff_b" in w:
            if stack_b_diff is None:
              # Bias shape: Linear (L, Out), Conv (L, Out) usually
              stack_b_diff = np.zeros((num_layers, out_feat), dtype=np.float32)
            bd = np.array(w["diff_b"])
            stack_b_diff[i] = bd.flatten()
            has_diff = True

      if has_lora or has_diff:
        bias_val = module.bias.value if module.bias is not None else None

        # Call JIT
        new_k, new_b = _compute_and_add_scanned_jit(
            module.kernel.value,
            stack_down if has_lora else None,
            stack_up if has_lora else None,
            stack_alpha if has_lora else None,
            scale,
            stack_w_diff,
            stack_b_diff,
            bias_val,
        )

        module.kernel.value = new_k
        if new_b is not None:
          module.bias.value = new_b

        assigned_count += 1
    else:
      # Should not happen based on is_scanned logic
      max_logging.log(f"Module {nnx_path_str} has scanned weights but is not Linear, Conv, Embed, or Norm type.")
      continue

  max_logging.log(f"Merged weights into {assigned_count} scanned layers.")
  unmatched_keys = set(lora_params.keys()) - matched_keys
  if unmatched_keys:
    max_logging.log(
        f"{len(unmatched_keys)} key(s) in LoRA dictionary were not applied to any layer in the model: {unmatched_keys}"
    )
