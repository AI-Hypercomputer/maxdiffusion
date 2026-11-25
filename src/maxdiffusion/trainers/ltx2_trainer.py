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

"""LTX-2 Joint Audio-Video DiT Trainer with 3D/1D Bucketing, Masked MSE, NNX LoRA,
Per-Bucket JIT Executable Cache, and Unbiased Cross-Bucket Token-Weighted GA.
"""

import datetime
import json
import os
import threading
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import flax
import flax.linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from safetensors.numpy import save_file as save_numpy_safetensors

from maxdiffusion import max_logging, max_utils, train_utils
from maxdiffusion.checkpointing.ltx2_checkpointer import LTX2Checkpointer
from maxdiffusion.data_preprocessing.ltx2_bucketing import (
    LTX2BucketSpec,
    LTX2BucketedBatchIterator,
    LTX2SyntheticBucketedIterator,
    get_ltx2_batch_shardings,
    iter_preencoded_npz_samples,
)
from maxdiffusion.loaders import lora_conversion_utils
from maxdiffusion.models.ltx2.logical_sharding_ltx2 import get_sharding_specs
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import (
    LTX2Pipeline,
    calculate_shift,
    cast_with_exclusion,
    svg_attention_enabled,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import _metrics_queue, _tensorboard_writer_worker


def _normalize_opt_path(path: Tuple[Any, ...]) -> Tuple[Any, ...]:
  """Normalizes a PyTree path tuple for optimizer state matching (`base_wan_trainer.py:L53-L73`)."""
  normalized = []
  for p in path:
    if hasattr(p, "key"):
      val = p.key
    elif hasattr(p, "idx"):
      val = p.idx
    elif hasattr(p, "name"):
      val = p.name
    else:
      val = p
    if isinstance(val, str) and val.isdigit():
      val = int(val)
    normalized.append(val)
  if normalized and normalized[-1] == "value":
    normalized = normalized[:-1]
  return tuple(normalized)


def restore_optimizer_state_by_path(target_opt_shardings: Any, restored_opt_state: Any) -> Any:
  """Restores optimizer state by matching normalized PyTree paths (`base_wan_trainer.py:L76-L98`)."""
  raw_restored_map, _ = jax.tree_util.tree_flatten_with_path(restored_opt_state)
  normalized_restored_map = {_normalize_opt_path(raw_path): val for raw_path, val in raw_restored_map}

  def _restore_leaf(path, target_sharding):
    norm_path = _normalize_opt_path(path)
    if norm_path not in normalized_restored_map:
      raise KeyError(f"Optimizer checkpoint missing path: {norm_path}.")
    return max_utils.device_put_replicated(normalized_restored_map[norm_path], target_sharding)

  return jax.tree_util.tree_map_with_path(_restore_leaf, target_opt_shardings)


DEFAULT_LTX2_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out",
    "ff.net_0",
    "ff.net_2",
    "audio_ff.net_0",
    "audio_ff.net_2",
)


class LTX2TrainState(train_state.TrainState):
  """Single shared TrainState PyTree across all static sequence buckets."""

  graphdef: nnx.GraphDef = flax.struct.field(pytree_node=False)
  rest_of_state: nnx.State


def _to_array(x: Any) -> jax.Array:
  return x if isinstance(x, jax.Array) else jnp.asarray(x)


def _tree_l2_norm(tree: Any) -> jax.Array:
  sq_sum = jax.tree_util.tree_reduce(
      lambda acc, leaf: acc + jnp.sum(leaf.astype(jnp.float32) ** 2),
      tree,
      initializer=jnp.asarray(0.0, dtype=jnp.float32),
  )
  return jnp.sqrt(sq_sum)


# ==============================================================================
# 1. NNX LoRA Adapter Module, Wrapping, State Merge & Diffusers Export
# ==============================================================================


class LTX2LoRALinear(nnx.Module):
  """NNX Linear LoRA wrapper supporting both scanned (`3D`) and unscanned (`2D`) kernels.

  Preserves `base_out.dtype` unconditionally so `float32` `nnx.LoRAParam` master weights
  never upcast `bfloat16` activations (`attention_flax.py:L463`) and `lora_b=0` yields
  bit-exact zero-init parity with `self.base`.
  """

  def __init__(
      self,
      base: nnx.Linear,
      rank: int,
      alpha: float,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    if rank <= 0:
      raise ValueError(f"lora_rank must be positive, got {rank}.")
    self.base = base
    self.rank = int(rank)
    self.alpha = float(alpha)
    self.scale = float(alpha) / float(rank)

    k_meta = base.kernel.get_metadata()
    out_sh = k_meta.get("out_sharding", None)
    is_scanned = base.kernel.ndim == 3

    if is_scanned:
      num_layers, in_features, out_features = base.kernel.shape
      in_ax = out_sh[1] if (out_sh is not None and len(out_sh) >= 3) else None
      out_ax = out_sh[2] if (out_sh is not None and len(out_sh) >= 3) else None
      a_sharding = ("layers", in_ax, None)
      b_sharding = ("layers", None, out_ax)
      a_val = nnx.initializers.lecun_normal()(rngs.params(), (num_layers, in_features, self.rank), param_dtype)
      b_val = jnp.zeros((num_layers, self.rank, out_features), dtype=param_dtype)
    elif base.kernel.ndim == 2:
      in_features, out_features = base.kernel.shape
      in_ax = out_sh[0] if (out_sh is not None and len(out_sh) >= 2) else None
      out_ax = out_sh[1] if (out_sh is not None and len(out_sh) >= 2) else None
      a_sharding = (in_ax, None)
      b_sharding = (None, out_ax)
      a_val = nnx.initializers.lecun_normal()(rngs.params(), (in_features, self.rank), param_dtype)
      b_val = jnp.zeros((self.rank, out_features), dtype=param_dtype)
    else:
      raise ValueError(f"Expected 2D or 3D scanned nnx.Linear kernel, got shape {base.kernel.shape}.")

    self.lora_a = nnx.LoRAParam(a_val, out_sharding=a_sharding)
    self.lora_b = nnx.LoRAParam(b_val, out_sharding=b_sharding)

  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    base_out = self.base(x, *args, **kwargs)
    prec = getattr(self.base, "precision", None)
    hidden = jnp.matmul(x.astype(jnp.float32), self.lora_a[...].astype(jnp.float32), precision=prec)
    delta = jnp.matmul(hidden, self.lora_b[...].astype(jnp.float32), precision=prec)
    return base_out + (delta * self.scale).astype(base_out.dtype)


def _replace_dotted_linear(
    root: Any,
    dotted_path: str,
    rank: int,
    alpha: float,
    rngs: nnx.Rngs,
    param_dtype: jnp.dtype,
) -> bool:
  parts = dotted_path.split(".")
  parent = root
  for attr in parts[:-1]:
    if not hasattr(parent, attr):
      return False
    parent = getattr(parent, attr)
  leaf_attr = parts[-1]
  if not hasattr(parent, leaf_attr):
    return False
  target = getattr(parent, leaf_attr)
  if isinstance(target, LTX2LoRALinear):
    return True
  if not isinstance(target, nnx.Linear):
    raise TypeError(f"Target {dotted_path!r} is {type(target).__name__}, expected nnx.Linear.")
  setattr(parent, leaf_attr, LTX2LoRALinear(target, rank=rank, alpha=alpha, rngs=rngs, param_dtype=param_dtype))
  return True


def apply_lora_to_ltx2_transformer(
    transformer: LTX2VideoTransformer3DModel,
    rank: int,
    alpha: float,
    target_modules: Sequence[str] = DEFAULT_LTX2_LORA_TARGET_MODULES,
    *,
    rngs: Optional[nnx.Rngs] = None,
    param_dtype: jnp.dtype = jnp.float32,
) -> int:
  """Wraps target `nnx.Linear` layers in `transformer` with `LTX2LoRALinear` in-place.

  Returns the number of wrapped `LTX2LoRALinear` module instances.
  """
  if rngs is None:
    rngs = nnx.Rngs(0)

  wrapped_count = 0
  for target in target_modules:
    # 1. Top-level module path (e.g. "proj_in", "caption_projection.linear_1")
    if _replace_dotted_linear(transformer, target, rank, alpha, rngs, param_dtype):
      wrapped_count += 1
      continue

    # 2. Block-relative suffix path (e.g. "attn1.to_q", "ff.net_0")
    suffix = target[len("transformer_blocks.") :] if target.startswith("transformer_blocks.") else target
    if transformer.scan_layers:
      if _replace_dotted_linear(transformer.transformer_blocks, suffix, rank, alpha, rngs, param_dtype):
        wrapped_count += 1
    else:
      for block in transformer.transformer_blocks:
        if _replace_dotted_linear(block, suffix, rank, alpha, rngs, param_dtype):
          wrapped_count += 1

  if wrapped_count == 0:
    raise ValueError(f"No nnx.Linear modules matched lora_target_modules={list(target_modules)}.")
  return wrapped_count


def extract_unmerged_base_params(state: LTX2TrainState) -> nnx.State:
  """Extracts unmerged base `nnx.Param` state mapped back to unwrapped `LTX2VideoTransformer3DModel` paths."""
  flat_rest = dict(nnx.to_flat_state(state.rest_of_state))
  out_entries = []
  for path, var in flat_rest.items():
    if not isinstance(var, nnx.Param):
      continue
    if len(path) >= 2 and path[-2] == "base":
      clean_path = path[:-2] + (path[-1],)
    else:
      clean_path = path
    out_entries.append((clean_path, var))
  return nnx.from_flat_state(out_entries)


def materialize_merged_ltx2_params(
    state: LTX2TrainState,
    lora_rank: int,
    lora_alpha: float,
) -> nnx.State:
  """Materializes `W_base + (alpha / rank) * (lora_a @ lora_b)` at standard `LTX2VideoTransformer3DModel` paths."""
  scale = float(lora_alpha) / float(lora_rank)
  flat_lora = dict(nnx.to_flat_state(state.params))
  flat_rest = dict(nnx.to_flat_state(state.rest_of_state))

  out_entries = []
  for path, var in flat_rest.items():
    if not isinstance(var, nnx.Param):
      continue
    if len(path) >= 2 and path[-2] == "base":
      mod_prefix = path[:-2]
      param_name = path[-1]
      clean_path = mod_prefix + (param_name,)
      if param_name == "kernel" and (mod_prefix + ("lora_a",)) in flat_lora:
        a_val = flat_lora[mod_prefix + ("lora_a",)].get_value().astype(jnp.float32)
        b_val = flat_lora[mod_prefix + ("lora_b",)].get_value().astype(jnp.float32)
        base_k = var.get_value()
        merged_k = base_k + (jnp.matmul(a_val, b_val) * scale).astype(base_k.dtype)
        out_entries.append((clean_path, var.replace(merged_k)))
      else:
        out_entries.append((clean_path, var))
    else:
      out_entries.append((path, var))
  return nnx.from_flat_state(out_entries)


def export_lora_to_diffusers_dict(
    lora_params: nnx.State,
    lora_alpha: float,
    scan_layers: bool,
) -> Dict[str, np.ndarray]:
  """Exports `nnx.LoRAParam` state into Diffusers `diffusion_model.*` LoRA keys (`LTX2NNXLoraLoader`-compatible).

  Emits `<diffusers_prefix>.lora_A.weight` (`(rank, in_features)`),
  `<diffusers_prefix>.lora_B.weight` (`(out_features, rank)`), and `<diffusers_prefix>.alpha`.
  """
  flat_lora = dict(nnx.to_flat_state(lora_params))
  module_prefixes = sorted({path[:-1] for path in flat_lora.keys() if path[-1] in ("lora_a", "lora_b")})

  state_dict: Dict[str, np.ndarray] = {}
  alpha_arr = np.asarray(float(lora_alpha), dtype=np.float32)

  for mod_path in module_prefixes:
    nnx_path_str = ".".join(str(p) for p in mod_path)
    diffusers_key = lora_conversion_utils.translate_ltx2_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=scan_layers)
    if diffusers_key is None:
      raise ValueError(
          f"NNX LoRA module path {nnx_path_str!r} has no Diffusers mapping in " "translate_ltx2_nnx_path_to_diffusers_lora."
      )
    if not diffusers_key.startswith("diffusion_model."):
      diffusers_key = f"diffusion_model.{diffusers_key}"

    a_np = np.asarray(jax.device_get(flat_lora[mod_path + ("lora_a",)].get_value()), dtype=np.float32)
    b_np = np.asarray(jax.device_get(flat_lora[mod_path + ("lora_b",)].get_value()), dtype=np.float32)

    if a_np.ndim == 3:
      if "{}" not in diffusers_key:
        raise ValueError(f"Expected '{{}}' placeholder in scanned Diffusers key {diffusers_key!r}.")
      num_layers = a_np.shape[0]
      for layer_idx in range(num_layers):
        layer_prefix = diffusers_key.format(layer_idx)
        state_dict[f"{layer_prefix}.lora_A.weight"] = np.ascontiguousarray(a_np[layer_idx].T)
        state_dict[f"{layer_prefix}.lora_B.weight"] = np.ascontiguousarray(b_np[layer_idx].T)
        state_dict[f"{layer_prefix}.alpha"] = alpha_arr.copy()
    elif a_np.ndim == 2:
      state_dict[f"{diffusers_key}.lora_A.weight"] = np.ascontiguousarray(a_np.T)
      state_dict[f"{diffusers_key}.lora_B.weight"] = np.ascontiguousarray(b_np.T)
      state_dict[f"{diffusers_key}.alpha"] = alpha_arr.copy()
    else:
      raise ValueError(f"Unexpected LoRA parameter rank {a_np.ndim} at {nnx_path_str!r}.")

  return state_dict


def save_lora_safetensors(
    output_path: str,
    lora_params: nnx.State,
    lora_alpha: float,
    scan_layers: bool,
) -> Dict[str, np.ndarray]:
  """Exports `lora_params` in Diffusers `diffusion_model.*` format to `output_path` (`.safetensors`)."""
  exported = export_lora_to_diffusers_dict(lora_params, lora_alpha=lora_alpha, scan_layers=scan_layers)
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  save_numpy_safetensors(exported, output_path)
  return exported


# ==============================================================================
# 2. Rectified-Flow Timestep Sampling, Masked MSE & Micro-Step Loss
# ==============================================================================


def sample_ltx2_sigmas(
    rng: jax.Array,
    video_attention_mask: jax.Array,
    *,
    timestep_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    use_dynamic_shifting: bool = True,
    static_shift: float = 1.0,
    base_seq_len: int = 1024,
    max_seq_len: int = 4096,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
) -> jax.Array:
  """Samples padding-invariant continuous noise scales `sigma_01` in `(0, 1)` of shape `[B]`.

  Dynamic shifting uses `jnp.sum(video_attention_mask, axis=1)` (valid video token count,
  never padded `B_vid`), matching `LTX2Pipeline` (`ltx2_pipeline.py:L1957-L1968`) and
  `FlaxFlowMatchScheduler.set_timesteps_ltx2` (`scheduling_flow_match_flax.py:L205`).
  """
  batch_size = video_attention_mask.shape[0]
  if timestep_sampling == "logit_normal":
    z = logit_mean + logit_std * jax.random.normal(rng, (batch_size,), dtype=jnp.float32)
    u = jax.nn.sigmoid(z)
  elif timestep_sampling == "uniform":
    u = jax.random.uniform(rng, (batch_size,), dtype=jnp.float32, minval=1e-6, maxval=1.0 - 1e-6)
  else:
    raise ValueError(f"Unsupported timestep_sampling={timestep_sampling!r}. Expected 'logit_normal' or 'uniform'.")

  u_clipped = jnp.clip(u, 1e-7, 1.0 - 1e-7)
  if use_dynamic_shifting:
    valid_vid_tokens = jnp.sum(video_attention_mask, axis=1, dtype=jnp.float32)
    shift_seq_len = jnp.minimum(6144.0, jnp.maximum(valid_vid_tokens, 1.0))
    mu = calculate_shift(
        shift_seq_len,
        base_seq_len=base_seq_len,
        max_seq_len=max_seq_len,
        base_shift=base_shift,
        max_shift=max_shift,
    )
    exp_mu = jnp.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / u_clipped - 1.0))
  else:
    s = jnp.float32(static_shift)
    sigmas = (s * u_clipped) / (1.0 + (s - 1.0) * u_clipped)

  return jnp.float32(sigma_min) + sigmas * jnp.float32(sigma_max - sigma_min)


def compute_ltx2_microbatch_errors(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    rest_of_state: nnx.State,
    batch: Mapping[str, jax.Array],
    rng: jax.Array,
    *,
    scheduler: FlaxFlowMatchScheduler,
    timestep_scale_multiplier: float = 1000.0,
    timestep_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    use_dynamic_shifting: bool = True,
    disable_training_weights: bool = True,
    use_cross_timestep: bool = False,
    sigmas_override: Optional[jax.Array] = None,
    video_noise_override: Optional[jax.Array] = None,
    audio_noise_override: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
  """Runs one dual-stream LTX-2 rectified-flow forward pass and returns `(E_vid, E_aud, aux)`."""
  model = nnx.merge(graphdef, params, rest_of_state)
  model_dtype = model.dtype

  clean_vid = batch["hidden_states"]
  clean_aud = batch["audio_hidden_states"]
  batch_size, b_vid, c_vid = clean_vid.shape
  _, b_aud, c_aud = clean_aud.shape

  vmask = batch.get("video_attention_mask", None)
  if vmask is None:
    vmask = jnp.ones((batch_size, b_vid), dtype=jnp.bool_)
  amask = batch.get("audio_attention_mask", None)
  if amask is None:
    amask = jnp.ones((batch_size, b_aud), dtype=jnp.bool_)

  sig_rng, vnoise_rng, anoise_rng = jax.random.split(rng, 3)

  if sigmas_override is not None:
    sigmas_01 = sigmas_override.astype(jnp.float32)
  else:
    sched_cfg = getattr(scheduler, "config", {})
    cfg_get = sched_cfg.get if isinstance(sched_cfg, Mapping) else lambda k, d: getattr(sched_cfg, k, d)
    sigmas_01 = sample_ltx2_sigmas(
        sig_rng,
        vmask,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        use_dynamic_shifting=use_dynamic_shifting,
        static_shift=float(cfg_get("shift", 1.0)),
        base_seq_len=int(cfg_get("base_image_seq_len", 1024)),
        max_seq_len=int(cfg_get("max_image_seq_len", 4096)),
        base_shift=float(cfg_get("base_shift", 0.95)),
        max_shift=float(cfg_get("max_shift", 2.05)),
        sigma_min=float(cfg_get("sigma_min", 0.0)),
        sigma_max=float(cfg_get("sigma_max", 1.0)),
    )

  clean_vid_f32 = clean_vid.astype(jnp.float32)
  clean_aud_f32 = clean_aud.astype(jnp.float32)
  eps_vid_f32 = (
      video_noise_override.astype(jnp.float32)
      if video_noise_override is not None
      else jax.random.normal(vnoise_rng, clean_vid.shape, dtype=jnp.float32)
  )
  eps_aud_f32 = (
      audio_noise_override.astype(jnp.float32)
      if audio_noise_override is not None
      else jax.random.normal(anoise_rng, clean_aud.shape, dtype=jnp.float32)
  )

  sig_b = sigmas_01[:, None, None]
  noisy_vid_f32 = jnp.where(vmask[:, :, None], (1.0 - sig_b) * clean_vid_f32 + sig_b * eps_vid_f32, 0.0)
  noisy_aud_f32 = jnp.where(amask[:, :, None], (1.0 - sig_b) * clean_aud_f32 + sig_b * eps_aud_f32, 0.0)

  v_target_vid = eps_vid_f32 - clean_vid_f32
  v_target_aud = eps_aud_f32 - clean_aud_f32

  # Serve-identical [0, 1000] timestep scaling across timestep, audio_timestep, sigma, and audio_sigma
  # (ltx2_pipeline.py:L1977, L2170, L2181-L2183, L2632-L2635, L2670, L2681-L2683).
  timesteps = sigmas_01 * jnp.float32(timestep_scale_multiplier)

  enc_vid = batch["encoder_hidden_states"].astype(model_dtype)
  enc_aud = batch["audio_encoder_hidden_states"].astype(model_dtype)

  out = model(
      hidden_states=noisy_vid_f32.astype(model_dtype),
      audio_hidden_states=noisy_aud_f32.astype(model_dtype),
      encoder_hidden_states=enc_vid,
      audio_encoder_hidden_states=enc_aud,
      timestep=timesteps,
      audio_timestep=timesteps,
      sigma=timesteps,
      audio_sigma=timesteps,
      encoder_attention_mask=batch.get("encoder_attention_mask", None),
      audio_encoder_attention_mask=batch.get("audio_encoder_attention_mask", None),
      video_coords=batch.get("video_coords", None),
      audio_coords=batch.get("audio_coords", None),
      use_cross_timestep=use_cross_timestep,
      modality_mask=batch.get("modality_mask", None),
      return_dict=True,
      video_attention_mask=batch.get("video_attention_mask", None),
      audio_attention_mask=batch.get("audio_attention_mask", None),
  )

  pred_vid_f32 = out["sample"].astype(jnp.float32)
  pred_aud_f32 = out["audio_sample"].astype(jnp.float32)

  if disable_training_weights:
    w_t = jnp.ones((batch_size, 1, 1), dtype=jnp.float32)
  else:
    num_train_ts = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    w_t = scheduler._calculate_training_weights(timesteps, num_train_ts).reshape(batch_size, 1, 1).astype(jnp.float32)

  vid_sq = jnp.where(vmask[:, :, None], w_t * (pred_vid_f32 - v_target_vid) ** 2, 0.0)
  aud_sq = jnp.where(amask[:, :, None], w_t * (pred_aud_f32 - v_target_aud) ** 2, 0.0)

  e_vid = jnp.sum(vid_sq, dtype=jnp.float32)
  e_aud = jnp.sum(aud_sq, dtype=jnp.float32)

  valid_vid_elems = jnp.sum(vmask, dtype=jnp.float32) * jnp.float32(c_vid)
  valid_aud_elems = jnp.sum(amask, dtype=jnp.float32) * jnp.float32(c_aud)
  aux = {
      "e_vid": e_vid,
      "e_aud": e_aud,
      "valid_vid_elems": valid_vid_elems,
      "valid_aud_elems": valid_aud_elems,
      "sample": out["sample"],
      "audio_sample": out["audio_sample"],
  }
  return e_vid, e_aud, aux


# ==============================================================================
# 3. LTX2Trainer: Per-Bucket JIT Cache & Cross-Bucket Token-Weighted GA
# ==============================================================================


class LTX2Trainer:
  """Joint Audio-Video LTX-2 Trainer with static 3D/1D bucketing and cross-bucket GA."""

  _profiler: Optional[max_utils.Profiler] = None

  def __init__(self, config: Any):
    if getattr(config, "train_text_encoder", False):
      raise ValueError("LTX2Trainer does not support training the Gemma text encoder.")
    if svg_attention_enabled(config):
      raise ValueError("SVG sparse attention is incompatible with bucketed padding-mask training.")

    self.config = config
    self.checkpointer = LTX2Checkpointer(config=config)
    self._jit_micro_grad_cache: Dict[Tuple[int, int, int], Any] = {}
    self._jit_add_grads: Optional[Any] = None
    self._jit_apply_grads: Optional[Any] = None
    self._jit_window_denoms: Optional[Any] = None
    self.state_shardings: Optional[LTX2TrainState] = None
    self._base_transformer_config_dict: Optional[Dict[str, Any]] = None

  def create_scheduler(self) -> Tuple[FlaxFlowMatchScheduler, Any]:
    pretrained = getattr(self.config, "pretrained_model_name_or_path", "")
    init_from_cfg = bool(getattr(self.config, "init_transformer_from_config_only", False))
    if pretrained and not init_from_cfg:
      try:
        scheduler, scheduler_state = FlaxFlowMatchScheduler.from_pretrained(pretrained, subfolder="scheduler")
        return scheduler, scheduler_state
      except Exception as e:
        max_logging.log(f"Falling back to default LTX-2 FlaxFlowMatchScheduler config ({e}).")

    scheduler = FlaxFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.0,
        use_dynamic_shifting=bool(getattr(self.config, "use_dynamic_shifting", True)),
        time_shift_type="exponential",
        dtype=jnp.float32,
    )
    scheduler.register_to_config(
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        base_shift=0.95,
        max_shift=2.05,
    )
    return scheduler, scheduler.create_state()

  def _resolve_buckets(self, mesh: Mesh) -> Tuple[LTX2BucketSpec, ...]:
    raw_buckets = getattr(self.config, "ltx2_buckets", None)
    if not raw_buckets:
      raw_buckets = ((1024, 256, 1024),)
    cp = mesh.shape.get("context", 1)
    specs: List[LTX2BucketSpec] = []
    for entry in raw_buckets:
      if isinstance(entry, LTX2BucketSpec):
        b = entry
      elif isinstance(entry, (list, tuple)) and len(entry) == 3:
        b = LTX2BucketSpec(int(entry[0]), int(entry[1]), int(entry[2]))
      else:
        raise ValueError(f"Each entry in ltx2_buckets must be [b_vid, b_aud, b_txt], got {entry!r}.")
      b.validate_for_mesh(context_parallelism=cp)
      specs.append(b)
    return tuple(specs)

  def _build_transformer_from_config_only(
      self,
      mesh: Mesh,
      rngs: nnx.Rngs,
      restored_checkpoint: Optional[Mapping[str, Any]] = None,
  ) -> LTX2VideoTransformer3DModel:
    """Builds and shards `LTX2VideoTransformer3DModel` directly from config overrides (offline/CI-safe)."""
    cfg = self.config
    sharding_cfg = getattr(cfg, "sharding", {}) or {}
    dit_specs = get_sharding_specs(sharding_cfg.get("transformer", "default"), "ltx2_dit")

    if restored_checkpoint and "ltx2_config" in restored_checkpoint:
      ltx2_cfg = dict(restored_checkpoint["ltx2_config"])
    else:
      in_ch = int(getattr(cfg, "transformer_in_channels", 128))
      aud_in_ch = int(getattr(cfg, "transformer_audio_in_channels", 128))
      heads = int(getattr(cfg, "transformer_num_attention_heads", 32))
      head_dim = int(getattr(cfg, "transformer_attention_head_dim", 128))
      aud_heads = int(getattr(cfg, "transformer_audio_num_attention_heads", 32))
      aud_head_dim = int(getattr(cfg, "transformer_audio_attention_head_dim", 64))
      num_layers = int(getattr(cfg, "transformer_num_layers", 48))
      cap_ch = int(getattr(cfg, "transformer_caption_channels", 3840))
      ltx2_cfg = {
          "in_channels": in_ch if in_ch > 0 else 128,
          "out_channels": in_ch if in_ch > 0 else 128,
          "patch_size": int(getattr(cfg, "transformer_patch_size", 1)),
          "patch_size_t": int(getattr(cfg, "transformer_patch_size_t", 1)),
          "num_attention_heads": heads if heads > 0 else 32,
          "attention_head_dim": head_dim if head_dim > 0 else 128,
          "cross_attention_dim": (heads if heads > 0 else 32) * (head_dim if head_dim > 0 else 128),
          "audio_in_channels": aud_in_ch if aud_in_ch > 0 else 128,
          "audio_out_channels": aud_in_ch if aud_in_ch > 0 else 128,
          "audio_num_attention_heads": aud_heads if aud_heads > 0 else 32,
          "audio_attention_head_dim": aud_head_dim if aud_head_dim > 0 else 64,
          "audio_cross_attention_dim": (aud_heads if aud_heads > 0 else 32) * (aud_head_dim if aud_head_dim > 0 else 64),
          "num_layers": num_layers if num_layers > 0 else 48,
          "caption_channels": cap_ch if cap_ch > 0 else 3840,
          "cross_attn_mod": bool(getattr(cfg, "transformer_cross_attn_mod", False)),
          "gated_attn": bool(getattr(cfg, "transformer_gated_attn", False)),
      }

    ltx2_cfg.update({
        "scan_layers": bool(getattr(cfg, "scan_layers", True)),
        "mesh": mesh,
        "dtype": cfg.activations_dtype,
        "weights_dtype": cfg.weights_dtype,
        "attention_kernel": cfg.attention,
        "a2v_attention_kernel": getattr(cfg, "a2v_attention_kernel", "dot_product"),
        "v2a_attention_kernel": getattr(cfg, "v2a_attention_kernel", "dot_product"),
        "precision": max_utils.get_precision(cfg),
        "flash_block_sizes": max_utils.get_flash_block_sizes(cfg),
        "flash_min_seq_length": int(getattr(cfg, "flash_min_seq_length", 4096)),
        "ulysses_shards": int(getattr(cfg, "ulysses_shards", -1)),
        "ulysses_attention_chunks": int(getattr(cfg, "ulysses_attention_chunks", 1)),
        "use_base2_exp": bool(getattr(cfg, "use_base2_exp", False)),
        "use_experimental_scheduler": bool(getattr(cfg, "use_experimental_scheduler", False)),
        "enable_jax_named_scopes": bool(getattr(cfg, "enable_jax_named_scopes", False)),
        "remat_policy": getattr(cfg, "remat_policy", "NONE"),
        "names_which_can_be_saved": list(getattr(cfg, "names_which_can_be_saved", [])),
        "names_which_can_be_offloaded": list(getattr(cfg, "names_which_can_be_offloaded", [])),
        "sharding_specs": dit_specs,
        "attention_config": {"use_svg_attention": False},
    })

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      transformer = LTX2VideoTransformer3DModel(**ltx2_cfg, rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)
      logical_spec = nnx.get_partition_spec(state)
      shardings = nn.logical_to_mesh_sharding(logical_spec, mesh, cfg.logical_axis_rules)

      if restored_checkpoint and "ltx2_state" in restored_checkpoint:
        raw_params = (
            restored_checkpoint["ltx2_state"]["params"]
            if "params" in restored_checkpoint["ltx2_state"]
            else restored_checkpoint["ltx2_state"]
        )
        raw_params = jax.tree_util.tree_map_with_path(
            lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=cfg.weights_dtype),
            raw_params,
        )
        flat_sh = dict(nnx.to_flat_state(shardings))
        flat_st = dict(nnx.to_flat_state(state))
        for path, val in flax.traverse_util.flatten_dict(raw_params).items():
          key_path = path[:-1] if path and path[-1] == "value" else path
          sh = flat_sh[key_path].get_value()
          flat_st[key_path].set_value(max_utils.device_put_replicated(val, sh))
        state = nnx.from_flat_state(flat_st)
      else:
        state = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), state, shardings)

      transformer = nnx.merge(graphdef, state, rest_of_state)
    return transformer

  def load_or_init_pipeline(self) -> Tuple[LTX2Pipeline, Dict[str, Any]]:
    """Loads or initializes `(pipeline, restore_args)`, skipping VAE/Gemma/Vocoder when unneeded."""
    cfg = self.config
    restored_checkpoint, step = self.checkpointer.load_ltx2_configs_from_orbax(step=None)
    restore_args: Dict[str, Any] = {}
    if restored_checkpoint is not None:
      ltx2_st = restored_checkpoint["ltx2_state"]
      if "opt_state" in ltx2_st:
        restore_args["opt_state"] = ltx2_st["opt_state"]
      if "lora_params" in ltx2_st:
        restore_args["lora_params"] = ltx2_st["lora_params"]
      if step is not None:
        restore_args["step"] = step

    load_encoders = bool(getattr(cfg, "load_encoders_for_training", False))
    init_from_cfg = bool(getattr(cfg, "init_transformer_from_config_only", False))

    if load_encoders and not init_from_cfg:
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        if restored_checkpoint is not None:
          pipeline = LTX2Pipeline.from_checkpoint(cfg, restored_checkpoint, vae_only=False, load_transformer=True)
        else:
          pipeline = LTX2Pipeline.from_pretrained(cfg, vae_only=False, load_transformer=True)
      scheduler, scheduler_state = self.create_scheduler()
      pipeline.scheduler = scheduler
      pipeline.scheduler_state = scheduler_state
      return pipeline, restore_args

    devices_array = max_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    rngs = nnx.Rngs(jax.random.key(int(cfg.seed)))
    scheduler, scheduler_state = self.create_scheduler()

    if init_from_cfg:
      transformer = self._build_transformer_from_config_only(mesh, rngs, restored_checkpoint=restored_checkpoint)
    else:
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        transformer = LTX2Pipeline.load_transformer(
            devices_array=devices_array,
            mesh=mesh,
            rngs=rngs,
            config=cfg,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer",
        )

    pipeline = LTX2Pipeline.__new__(LTX2Pipeline)
    pipeline.scheduler = scheduler
    pipeline.vae = None
    pipeline.audio_vae = None
    pipeline.vocoder = None
    pipeline.text_encoder = None
    pipeline.tokenizer = None
    pipeline.connectors = None
    pipeline.transformer = transformer
    pipeline.latent_upsampler = None
    pipeline.latent_upsampler_params = None
    pipeline.mesh = mesh
    pipeline.config = cfg
    pipeline.vae_spatial_compression_ratio = 32
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.transformer_spatial_patch_size = getattr(transformer, "patch_size", 1)
    pipeline.transformer_temporal_patch_size = getattr(transformer, "patch_size_t", 1)
    pipeline.audio_sampling_rate = getattr(transformer, "audio_sampling_rate", 16000)
    pipeline.audio_hop_length = getattr(transformer, "audio_hop_length", 160)
    pipeline.tokenizer_max_length = int(getattr(cfg, "max_sequence_length", 1024))
    pipeline.scheduler_state = scheduler_state
    return pipeline, restore_args

  def load_dataset(self, mesh: Mesh, transformer: LTX2VideoTransformer3DModel) -> Iterator[Dict[str, Any]]:
    """Builds sharded multi-bucket batch iterator (`synthetic` or `preencoded_npz`)."""
    cfg = self.config
    buckets = self._resolve_buckets(mesh)
    global_batch_size = int(cfg.global_batch_size_to_train_on)
    text_padding_mode = str(getattr(cfg, "text_padding_mode", "post_connector"))
    num_regs = int(getattr(cfg, "connector_num_learnable_registers", 128))

    if cfg.dataset_type == "synthetic":
      num_unique = int(getattr(cfg, "synthetic_num_unique_batches_per_bucket", 0))
      return iter(
          LTX2SyntheticBucketedIterator(
              buckets=buckets,
              global_batch_size=global_batch_size,
              in_channels=transformer.in_channels,
              audio_in_channels=transformer.audio_in_channels,
              caption_channels=transformer.caption_channels,
              audio_caption_channels=transformer.audio_caption_channels,
              patch_size=transformer.patch_size,
              patch_size_t=transformer.patch_size_t,
              vae_scale_factors=tuple(transformer.vae_scale_factors),
              audio_scale_factor=transformer.audio_scale_factor,
              text_padding_mode=text_padding_mode,
              connector_num_learnable_registers=num_regs,
              video_only_prob=float(getattr(cfg, "synthetic_video_only_prob", 0.0)),
              num_unique_batches_per_bucket=num_unique if num_unique > 0 else None,
              num_steps=None,
              seed=int(cfg.seed),
              dtype=np.float32,
              mesh=mesh,
          )
      )

    if cfg.dataset_type == "preencoded_npz":
      train_data_dir = str(getattr(cfg, "train_data_dir", ""))
      if not train_data_dir:
        raise ValueError("train_data_dir must be set when dataset_type='preencoded_npz'.")
      collate_kwargs = {
          "patch_size": transformer.patch_size,
          "patch_size_t": transformer.patch_size_t,
          "vae_scale_factors": tuple(transformer.vae_scale_factors),
          "audio_in_channels": transformer.audio_in_channels,
          "audio_scale_factor": transformer.audio_scale_factor,
          "text_padding_mode": text_padding_mode,
          "connector_num_learnable_registers": num_regs,
          "dtype": np.float32,
      }
      return iter(
          LTX2BucketedBatchIterator(
              iter_preencoded_npz_samples(train_data_dir, repeat=True),
              buckets=buckets,
              batch_size=global_batch_size,
              mesh=mesh,
              overflow_policy=str(getattr(cfg, "bucket_overflow_policy", "error")),
              drop_remainder=True,
              input_is_process_local=False,
              collate_kwargs=collate_kwargs,
          )
      )

    raise ValueError(
        f"Unsupported dataset_type={cfg.dataset_type!r} for LTX2Trainer. Expected 'synthetic' or 'preencoded_npz'."
    )

  def init_train_state(
      self,
      pipeline: LTX2Pipeline,
      optimizer: Any,
      restore_args: Optional[Dict[str, Any]] = None,
  ) -> Tuple[LTX2TrainState, LTX2TrainState, int]:
    """Creates sharded `LTX2TrainState` (full-param or NNX `LoRAParam` filter) and computes `state_shardings` once."""
    cfg = self.config
    mesh = pipeline.mesh
    restore_args = dict(restore_args or {})

    # Record clean unwrapped transformer config JSON before optional LoRA wrapping.
    self._base_transformer_config_dict = json.loads(pipeline.transformer.to_json_string())

    enable_lora = bool(getattr(cfg, "enable_lora_training", False))
    if enable_lora:
      lora_rank = int(getattr(cfg, "lora_rank", 32))
      lora_alpha = float(getattr(cfg, "lora_alpha", lora_rank))
      target_modules = tuple(getattr(cfg, "lora_target_modules", DEFAULT_LTX2_LORA_TARGET_MODULES))
      lora_rng = nnx.Rngs(jax.random.fold_in(jax.random.key(int(cfg.seed)), 9001))
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        wrapped = apply_lora_to_ltx2_transformer(
            pipeline.transformer,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=target_modules,
            rngs=lora_rng,
            param_dtype=jnp.float32,
        )
      max_logging.log(f"Wrapped {wrapped} LTX-2 Linear modules with LTX2LoRALinear (rank={lora_rank}, alpha={lora_alpha}).")
      trainable_filter = nnx.LoRAParam
    else:
      trainable_filter = nnx.Param

    graphdef, params, rest_of_state = nnx.split(pipeline.transformer, trainable_filter, ...)
    start_step = 0

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      state = LTX2TrainState.create(
          apply_fn=graphdef.apply,
          params=params,
          tx=optimizer,
          graphdef=graphdef,
          rest_of_state=rest_of_state,
      )
      state = jax.tree.map(_to_array, state)
      state_logical_spec = nnx.get_partition_spec(state)
      state_shardings = nn.logical_to_mesh_sharding(state_logical_spec, mesh, cfg.logical_axis_rules)
      state = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), state, state_shardings)

      if enable_lora and restore_args.get("lora_params") is not None:
        restored_lora = restore_args["lora_params"]
        flat_sh = dict(nnx.to_flat_state(state_shardings.params))
        flat_st = dict(nnx.to_flat_state(state.params))
        for path, val in flax.traverse_util.flatten_dict(restored_lora).items():
          key_path = path[:-1] if path and path[-1] == "value" else path
          sh = flat_sh[key_path].get_value()
          flat_st[key_path].set_value(max_utils.device_put_replicated(val, sh))
        state = state.replace(params=nnx.from_flat_state(flat_st))

      if restore_args.get("opt_state") is not None:
        resharded_opt_state = restore_optimizer_state_by_path(
            target_opt_shardings=state_shardings.opt_state,
            restored_opt_state=restore_args["opt_state"],
        )
        state = state.replace(opt_state=resharded_opt_state)
        if restore_args.get("step") is not None:
          resumed_step = int(restore_args["step"]) + 1
          state = state.replace(step=jnp.asarray(resumed_step, dtype=state.step.dtype))
          start_step = resumed_step

    self.state_shardings = state_shardings
    self._compile_shared_grad_ops(mesh)
    return state, state_shardings, start_step

  def _compile_shared_grad_ops(self, mesh: Mesh) -> None:
    """Compiles bucket-independent gradient accumulation & single-TrainState optimizer update ops."""
    assert self.state_shardings is not None
    param_shardings = self.state_shardings.params
    scalar_sharding = NamedSharding(mesh, P())

    self._jit_add_grads = jax.jit(
        lambda g_acc, g_new: jax.tree_util.tree_map(jnp.add, g_acc, g_new),
        in_shardings=(param_shardings, param_shardings),
        out_shardings=param_shardings,
        donate_argnums=(0,),
    )

    def _apply_grads_fn(state: LTX2TrainState, grads: nnx.State):
      grad_norm = _tree_l2_norm(grads)
      max_abs_grad = jax.tree_util.tree_reduce(
          lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr.astype(jnp.float32)))),
          grads,
          initializer=jnp.asarray(0.0, dtype=jnp.float32),
      )
      new_state = state.apply_gradients(grads=grads)
      return new_state, grad_norm, max_abs_grad

    self._jit_apply_grads = jax.jit(
        _apply_grads_fn,
        in_shardings=(self.state_shardings, param_shardings),
        out_shardings=(self.state_shardings, scalar_sharding, scalar_sharding),
        donate_argnums=(0, 1),
    )

    def _compute_window_denoms(
        vmasks: Tuple[jax.Array, ...],
        amasks: Tuple[jax.Array, ...],
        c_vid: int,
        c_aud: int,
        audio_loss_weight: float,
    ) -> Tuple[jax.Array, jax.Array]:
      d_vid = sum((jnp.sum(m, dtype=jnp.float32) for m in vmasks), jnp.float32(0.0)) * jnp.float32(c_vid)
      d_aud = sum((jnp.sum(m, dtype=jnp.float32) for m in amasks), jnp.float32(0.0)) * jnp.float32(c_aud)
      inv_vid = jnp.where(d_vid > 0.0, 1.0 / jnp.maximum(d_vid, 1.0), 0.0)
      inv_aud = jnp.where(d_aud > 0.0, jnp.float32(audio_loss_weight) / jnp.maximum(d_aud, 1.0), 0.0)
      return inv_vid, inv_aud

    self._jit_window_denoms = jax.jit(
        _compute_window_denoms,
        static_argnums=(2, 3, 4),
        out_shardings=(scalar_sharding, scalar_sharding),
    )

  def get_or_compile_bucket_grad_step(
      self,
      bucket: LTX2BucketSpec,
      mesh: Mesh,
      scheduler: FlaxFlowMatchScheduler,
      graphdef: nnx.GraphDef,
      timestep_scale_multiplier: float = 1000.0,
  ) -> Callable[..., Any]:
    """Returns cached per-bucket `@jax.jit` micro-step VJP executable sharing `self.state_shardings`."""
    bucket_key = bucket.as_tuple()
    if bucket_key in self._jit_micro_grad_cache:
      return self._jit_micro_grad_cache[bucket_key]

    if self.state_shardings is None:
      raise RuntimeError("init_train_state must be called before get_or_compile_bucket_grad_step.")

    cfg = self.config
    batch_shardings = get_ltx2_batch_shardings(mesh, bucket)
    scalar_sharding = NamedSharding(mesh, P())
    timestep_sampling = str(getattr(cfg, "timestep_sampling", "logit_normal"))
    logit_mean = float(getattr(cfg, "logit_normal_mean", 0.0))
    logit_std = float(getattr(cfg, "logit_normal_std", 1.0))
    use_dynamic_shifting = bool(getattr(cfg, "use_dynamic_shifting", True))
    disable_training_weights = bool(getattr(cfg, "disable_training_weights", True))
    use_cross_timestep = bool(getattr(cfg, "use_cross_timestep", False))

    def _micro_grad_fn(
        params: nnx.State,
        rest_of_state: nnx.State,
        batch_tensors: Dict[str, jax.Array],
        micro_rng: jax.Array,
        inv_vid_denom: jax.Array,
        inv_aud_denom: jax.Array,
    ):
      def _loss_fn(p_trainable: nnx.State, p_rest: nnx.State):
        e_vid, e_aud, _ = compute_ltx2_microbatch_errors(
            graphdef,
            p_trainable,
            p_rest,
            batch_tensors,
            micro_rng,
            scheduler=scheduler,
            timestep_scale_multiplier=timestep_scale_multiplier,
            timestep_sampling=timestep_sampling,
            logit_mean=logit_mean,
            logit_std=logit_std,
            use_dynamic_shifting=use_dynamic_shifting,
            disable_training_weights=disable_training_weights,
            use_cross_timestep=use_cross_timestep,
        )
        vid_contrib = e_vid * inv_vid_denom
        aud_contrib = e_aud * inv_aud_denom
        return vid_contrib + aud_contrib, (vid_contrib, aud_contrib)

      (micro_loss, (vid_contrib, aud_contrib)), grads = nnx.value_and_grad(_loss_fn, argnums=0, has_aux=True)(
          params, rest_of_state
      )
      return (micro_loss, vid_contrib, aud_contrib), grads

    compiled_fn = jax.jit(
        _micro_grad_fn,
        in_shardings=(
            self.state_shardings.params,
            self.state_shardings.rest_of_state,
            batch_shardings,
            scalar_sharding,
            scalar_sharding,
            scalar_sharding,
        ),
        out_shardings=((scalar_sharding, scalar_sharding, scalar_sharding), self.state_shardings.params),
    )
    self._jit_micro_grad_cache[bucket_key] = compiled_fn
    return compiled_fn

  def run_optimizer_step(
      self,
      state: LTX2TrainState,
      micro_batches: Sequence[Mapping[str, Any]],
      rng: jax.Array,
      *,
      mesh: Mesh,
      scheduler: FlaxFlowMatchScheduler,
      timestep_scale_multiplier: float = 1000.0,
  ) -> Tuple[LTX2TrainState, Dict[str, Any], jax.Array]:
    """Executes one unbiased token-weighted cross-bucket GA window of `K = len(micro_batches)` micro-steps."""
    if not micro_batches:
      raise ValueError("micro_batches must contain at least 1 batch.")

    k_steps = len(micro_batches)
    c_vid = int(micro_batches[0]["hidden_states"].shape[-1])
    c_aud = int(micro_batches[0]["audio_hidden_states"].shape[-1])
    audio_loss_weight = float(getattr(self.config, "audio_loss_weight", 1.0))

    vmasks = tuple(b["video_attention_mask"] for b in micro_batches)
    amasks = tuple(b["audio_attention_mask"] for b in micro_batches)
    inv_vid_denom, inv_aud_denom = self._jit_window_denoms(vmasks, amasks, c_vid, c_aud, audio_loss_weight)

    scalar_sharding = NamedSharding(mesh, P())
    rng = jax.device_put(rng, scalar_sharding)
    rngs = [jax.device_put(r, scalar_sharding) for r in jax.random.split(rng, k_steps + 1)]
    next_rng = rngs[0]

    grads_acc = None
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_vid_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_aud_loss = jnp.asarray(0.0, dtype=jnp.float32)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
      for idx, batch in enumerate(micro_batches):
        bucket = LTX2BucketSpec(*batch["bucket_key"])
        tensor_batch = {k: v for k, v in batch.items() if k != "bucket_key"}
        micro_grad_fn = self.get_or_compile_bucket_grad_step(
            bucket,
            mesh,
            scheduler,
            state.graphdef,
            timestep_scale_multiplier=timestep_scale_multiplier,
        )
        (m_loss, m_vid_loss, m_aud_loss), m_grads = micro_grad_fn(
            state.params,
            state.rest_of_state,
            tensor_batch,
            rngs[idx + 1],
            inv_vid_denom,
            inv_aud_denom,
        )
        total_loss = total_loss + m_loss
        total_vid_loss = total_vid_loss + m_vid_loss
        total_aud_loss = total_aud_loss + m_aud_loss
        grads_acc = m_grads if grads_acc is None else self._jit_add_grads(grads_acc, m_grads)

      new_state, grad_norm, max_abs_grad = self._jit_apply_grads(state, grads_acc)

    metrics = {
        "scalar": {
            "learning/loss": total_loss,
            "learning/video_loss": total_vid_loss,
            "learning/audio_loss": total_aud_loss,
            "learning/grad_norm": grad_norm,
            "learning/max_grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
        },
        "scalars": {},
    }
    return new_state, metrics, next_rng

  def save_training_checkpoint(
      self,
      step: int,
      pipeline: LTX2Pipeline,
      state: LTX2TrainState,
      *,
      merge_lora_on_save: Optional[bool] = None,
  ) -> None:
    """Saves full-param or LoRA checkpoint via `LTX2Checkpointer`."""
    cfg = self.config
    enable_lora = bool(getattr(cfg, "enable_lora_training", False))
    save_opt = bool(getattr(cfg, "save_optimizer", False))
    if merge_lora_on_save is None:
      merge_lora_on_save = bool(getattr(cfg, "merge_lora_on_save", False))

    if enable_lora:
      lora_rank = int(getattr(cfg, "lora_rank", 32))
      lora_alpha = float(getattr(cfg, "lora_alpha", lora_rank))
      if merge_lora_on_save:
        merged_params = materialize_merged_ltx2_params(state, lora_rank=lora_rank, lora_alpha=lora_alpha)
        payload: Dict[str, Any] = {"params": merged_params}
      else:
        payload = {
            "params": extract_unmerged_base_params(state),
            "lora_params": state.params,
        }
    else:
      payload = {"params": state.params}

    if save_opt:
      payload["opt_state"] = state.opt_state

    self.checkpointer.save_checkpoint(
        step,
        pipeline,
        payload,
        transformer_config_dict=self._base_transformer_config_dict,
    )

    lora_export_dir = str(getattr(cfg, "lora_safetensors_export_dir", ""))
    if enable_lora and lora_export_dir:
      gathered_lora_params = (
          multihost_utils.process_allgather(state.params, tiled=True) if jax.process_count() > 1 else state.params
      )
      if jax.process_index() == 0:
        out_file = os.path.join(lora_export_dir, f"step_{step:06d}", "ltx2_lora_adapter.safetensors")
        save_lora_safetensors(
            out_file,
            gathered_lora_params,
            lora_alpha=float(getattr(cfg, "lora_alpha", getattr(cfg, "lora_rank", 32))),
            scan_layers=bool(getattr(cfg, "scan_layers", True)),
        )

  def start_training(self) -> LTX2Pipeline:
    """End-to-end training loop supporting cross-bucket GA, metrics logging, and Orbax checkpointing."""
    cfg = self.config
    pipeline, restore_args = self.load_or_init_pipeline()
    mesh = pipeline.mesh

    optimizer, learning_rate_scheduler = self.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
    state, _, start_step = self.init_train_state(pipeline, optimizer, restore_args=restore_args)
    train_iter = self.load_dataset(mesh, pipeline.transformer)

    grad_accum_steps = max(1, int(getattr(cfg, "gradient_accumulation_steps", 1)))
    timestep_scale_multiplier = float(getattr(pipeline.transformer, "timestep_scale_multiplier", 1000.0))

    is_main_process = jax.process_index() == 0
    writer = max_utils.initialize_summary_writer(cfg)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, cfg), daemon=True)
    writer_thread.start()

    num_trainable_params = max_utils.calculate_num_params_from_pytree(state.params)
    if is_main_process:
      max_logging.log("***** Running LTX-2 Bucketed Joint Audio-Video Training *****")
      max_logging.log(f"  Global micro-batch size = {cfg.global_batch_size_to_train_on}")
      max_logging.log(f"  Gradient accumulation steps = {grad_accum_steps}")
      max_logging.log(f"  Trainable parameters = {num_trainable_params}")
      max_logging.log(f"  Max train steps = {cfg.max_train_steps}")

    local_metrics_file = (
        open(cfg.metrics_file, "a", encoding="utf8") if (is_main_process and getattr(cfg, "metrics_file", "")) else None
    )
    running_gcs_metrics = [] if (is_main_process and getattr(cfg, "gcs_metrics", False)) else None
    rng = jax.random.key(int(cfg.seed))

    last_step = start_step - 1
    try:
      for step in range(start_step, int(cfg.max_train_steps)):
        last_step = step
        t0 = datetime.datetime.now()
        window_batches = [next(train_iter) for _ in range(grad_accum_steps)]

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          state, train_metric, rng = self.run_optimizer_step(
              state,
              window_batches,
              rng,
              mesh=mesh,
              scheduler=pipeline.scheduler,
              timestep_scale_multiplier=timestep_scale_multiplier,
          )
          train_metric["scalar"]["learning/loss"].block_until_ready()

        dt = datetime.datetime.now() - t0
        train_utils.record_scalar_metrics(
            train_metric,
            dt,
            per_device_tflops=0.0,
            lr=float(learning_rate_scheduler(step)),
            total_weights=num_trainable_params,
        )
        if is_main_process and getattr(cfg, "write_metrics", False):
          train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, cfg)

        ckpt_every = int(getattr(cfg, "checkpoint_every", -1))
        if step != 0 and ckpt_every > 0 and step % ckpt_every == 0:
          self.save_training_checkpoint(step, pipeline, state)

      if bool(getattr(cfg, "save_final_checkpoint", False)) and last_step >= start_step:
        self.save_training_checkpoint(last_step, pipeline, state)
        if self.checkpointer.checkpoint_manager is not None:
          self.checkpointer.checkpoint_manager.wait_until_finished()
    finally:
      _metrics_queue.put(None)
      writer_thread.join()
      if writer:
        writer.flush()
      if local_metrics_file is not None and not local_metrics_file.closed:
        local_metrics_file.close()

    pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
    return pipeline
