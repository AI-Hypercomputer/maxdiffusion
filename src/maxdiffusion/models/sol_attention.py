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

Model-agnostic integration layer for Sol-Attn (arXiv:2607.24027).

Sol-Attn is a training-free drop-in replacement for full self-attention: it
only needs per-head Q/K/V over the full token sequence. Integrating it into a
model takes three steps (see `FlaxWanAttention` for an example):

  1. In the attention module's constructor, resolve the `sol_*` settings with
     `init_sol_config(attention_config, num_layers, is_self_attention)` and
     store them on the module.
  2. In `__call__`, wrap the attention op with `apply_sol_or_dense(...)`, which
     evaluates the dense warm-up schedule (steps / layers) and passes a
     `sparse_config_override` to the attention op on sparse steps.
  3. The attention dispatcher (`attention_flax._apply_attention`) routes that
     override to `sol_sharded_attention`, which gathers full sequences per head
     (Ulysses all-to-all if the sequence is sharded) and runs the TPU kernel.

Nothing here depends on video-specific token layouts, so the same path works
for image DiTs. MMDiT-style joint attention can force text tokens into the
exact branch via the kernel's `always_exact` argument.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Optional

import flax.linen as nn
import jax
from jax.sharding import PartitionSpec as P

from maxdiffusion.common_types import BATCH, HEAD, LENGTH
from maxdiffusion.kernels.sol_attention import routing as sol_routing
from maxdiffusion.kernels.sol_attention import sol_attention_kernel
from maxdiffusion.models import sparse_schedule

# Attention-level Sol-Attn keys and their defaults. `sol_num_layers` is model
# specific and supplied by the caller.
SOL_ATTENTION_DEFAULTS = {
    "use_sol_attention": False,
    # Standardized threshold cutoff; expected exact density ~= 1 - Phi(beta).
    # beta=1.0 -> ~16% density (~84% sparsity), close to the paper's Wan setup.
    "sol_beta": 1.0,
    # Query block (routing granularity and kernel query tile) and key/value
    # block (exact-branch granularity). Multiples of 128.
    "sol_block_q": 1024,
    "sol_block_kv": 256,
    # Dense warm-up schedule. Explicit [start, end) intervals win over fractions.
    "sol_active_start_step": -1,
    "sol_active_end_step": -1,
    "sol_active_start_layer": -1,
    "sol_active_end_layer": -1,
    "sol_dense_layer_fraction": 0.0,
    "sol_dense_timestep_fraction": 0.0,
    "sol_num_train_timesteps": 1000,
    # Optional Mosaic VMEM limit for the kernel (None = compiler default).
    "sol_vmem_limit_bytes": None,
    # Print the per-call exact-branch density (debug; adds a host callback).
    "sol_log_density": False,
}


def attention_config_from_pyconfig(config: Any, num_layers: int) -> dict[str, Any]:
  """Extracts the transformer `attention_config` entries from a pipeline config.

  Pipelines merge the result into the `attention_config` they hand to the
  transformer. When Sol-Attn is disabled nothing is emitted, so dense runs keep
  their exact transformer config (and AOT cache identity).
  """
  if not bool(getattr(config, "use_sol_attention", False)):
    return {}
  resolved = {"use_sol_attention": True, "sol_num_layers": int(getattr(config, "sol_num_layers", num_layers) or num_layers)}
  for name in SOL_ATTENTION_DEFAULTS:
    if name != "use_sol_attention" and getattr(config, name, None) is not None:
      resolved[name] = getattr(config, name)
  return resolved


def init_sol_config(
    attention_config: Optional[Mapping[str, Any]],
    default_num_layers: int,
    is_self_attention: bool = True,
) -> dict[str, Any]:
  """Resolves attention-level Sol-Attn settings from `attention_config`.

  Returns a dict keyed by `SOL_ATTENTION_DEFAULTS` plus `sol_num_layers`. Sol
  only runs on self-attention with `use_sol_attention` set; otherwise the
  canonical defaults are returned so unused `sol_*` overrides do not change the
  module's static graph (and thus the AOT cache key).
  """
  attention_config = attention_config or {}
  resolved = {**SOL_ATTENTION_DEFAULTS, "sol_num_layers": default_num_layers}
  if not (is_self_attention and bool(attention_config.get("use_sol_attention", False))):
    return resolved
  if bool(attention_config.get("use_svg_attention", False)):
    raise ValueError("use_sol_attention and use_svg_attention are mutually exclusive.")
  for name in resolved:
    if name in attention_config and attention_config[name] is not None:
      resolved[name] = attention_config[name]
  resolved["use_sol_attention"] = True
  # Fail at construction time instead of at trace time.
  sol_routing.SolBlockSizes(block_q=int(resolved["sol_block_q"]), block_kv=int(resolved["sol_block_kv"]))
  if float(resolved["sol_beta"]) < 0.0:
    raise ValueError(f"sol_beta must be >= 0, got {resolved['sol_beta']}.")
  return resolved


def is_sol_active(
    sol: Any,
    layer_index: int | jax.Array | None,
    step_index: int | jax.Array | None,
    timestep: int | float | jax.Array | None,
) -> bool | jax.Array:
  """Dense warm-up schedule for an object carrying `init_sol_config` attributes."""
  return sparse_schedule.is_sparse_active(
      step_index=step_index,
      layer_index=layer_index,
      timestep=timestep,
      start_step=sol.sol_active_start_step,
      end_step=sol.sol_active_end_step,
      start_layer=sol.sol_active_start_layer,
      end_layer=sol.sol_active_end_layer,
      dense_layer_fraction=sol.sol_dense_layer_fraction,
      dense_timestep_fraction=sol.sol_dense_timestep_fraction,
      num_train_timesteps=sol.sol_num_train_timesteps,
      num_layers=sol.sol_num_layers,
      name="Sol-Attn",
  )


def sparse_config(sol: Any, layer_index=None, step_index=None) -> dict[str, Any]:
  """The `sparse_config_override` consumed by `sol_sharded_attention`."""
  return {
      "use_sol_attention": True,
      "beta": float(sol.sol_beta),
      "block_q": int(sol.sol_block_q),
      "block_kv": int(sol.sol_block_kv),
      "vmem_limit_bytes": sol.sol_vmem_limit_bytes or None,
      "log_density": bool(sol.sol_log_density),
      "layer_index": layer_index,
      "step_index": step_index,
  }


def apply_sol_or_dense(
    sol: Any,
    run_attention: Callable[..., jax.Array],
    layer_index: int | jax.Array | None,
    step_index: int | jax.Array | None,
    timestep: int | float | jax.Array | None,
    named_scope: Callable[[str], Any] = jax.named_scope,
) -> jax.Array:
  """Runs Sol-Attn on active steps/layers and dense attention otherwise.

  Args:
    sol: Object exposing the attributes resolved by `init_sol_config`
      (typically the attention module itself).
    run_attention: Calls the attention op. Invoked with no arguments for the
      dense path and with a `sparse_config_override` keyword for the sparse path.
    layer_index: Transformer layer index (static int or traced array).
    step_index: Denoising step index (static int or traced array).
    timestep: Diffusion timestep, used by fraction-based schedules.
    named_scope: Context manager factory used to label the dispatch.
  """
  is_active = is_sol_active(sol, layer_index, step_index, timestep)

  def run_dense(_):
    return run_attention()

  def run_sparse(_):
    return run_attention(sparse_config_override=sparse_config(sol, layer_index, step_index))

  with named_scope("apply_attention"):
    if isinstance(is_active, bool):
      return run_sparse(None) if is_active else run_dense(None)
    return jax.lax.cond(is_active, run_sparse, run_dense, operand=None)


def _to_bhld(x: jax.Array, heads: int) -> jax.Array:
  if x.ndim == 4:
    return x
  b, l, hd = x.shape
  return x.reshape(b, l, heads, hd // heads).transpose(0, 2, 1, 3)


def _sequence_axis(spec_entry) -> Optional[str]:
  if spec_entry is None:
    return None
  if isinstance(spec_entry, str):
    return spec_entry
  if isinstance(spec_entry, (tuple, list)) and len(spec_entry) == 1:
    return spec_entry[0]
  raise ValueError(f"Sol-Attn supports sequence sharding over a single mesh axis, got {spec_entry!r}.")


def sol_sharded_attention(query: jax.Array, key: jax.Array, value: jax.Array, context: Mapping[str, Any]) -> jax.Array:
  """Sol-Attn entry point for `attention_flax._apply_attention`.

  Accepts [B, L, H*D] or [B, H, L, D] self-attention inputs and returns
  [B, L, H*D]. If the sequence is sharded over a mesh axis, heads are
  exchanged with an all-to-all so every device sees full sequences for a
  subset of heads (Ulysses), which is what block routing needs.
  """
  cfg = context["spatiotemporal_config"]
  heads = context["heads"]
  mesh = context["mesh"]
  if context.get("attention_mask") is not None:
    raise ValueError("Sol-Attn does not support attention masks.")
  if context.get("is_causal", False):
    raise ValueError("Sol-Attn supports bidirectional attention only.")

  q, k, v = (_to_bhld(x, heads) for x in (query, key, value))
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError(f"Sol-Attn requires matched self-attention q/k/v, got {q.shape}, {k.shape}, {v.shape}.")

  kernel = functools.partial(
      sol_attention_kernel.sol_attention,
      scale=context["scale"],
      beta=cfg["beta"],
      block_sizes=sol_routing.SolBlockSizes(block_q=cfg["block_q"], block_kv=cfg["block_kv"]),
      vmem_limit_bytes=cfg.get("vmem_limit_bytes"),
      return_density=True,
  )

  if mesh is None:
    out, density = kernel(q, k, v)
  else:
    qspec = nn.logical_to_mesh_axes(context["axis_names_q"])
    kvspec = nn.logical_to_mesh_axes(context["axis_names_kv"])
    if qspec != kvspec:
      raise ValueError(f"Sol-Attn requires identical q/kv sharding, got {qspec} vs {kvspec}.")
    if qspec[3] is not None:
      raise ValueError("Sol-Attn does not support sharding the head dimension.")
    seq_axis = _sequence_axis(qspec[2])
    mesh_axes = tuple(mesh.axis_names)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(qspec, kvspec, kvspec),
        out_specs=(qspec, P()),
        check_vma=False,
    )
    def local(q, k, v):
      if seq_axis is not None:
        a2a = functools.partial(jax.lax.all_to_all, axis_name=seq_axis, tiled=True)
        q, k, v = (a2a(x, split_axis=1, concat_axis=2) for x in (q, k, v))
      out, density = kernel(q, k, v)
      if seq_axis is not None:
        out = a2a(out, split_axis=2, concat_axis=1)
      return out, jax.lax.pmean(density, mesh_axes)

    out, density = local(q, k, v)

  if cfg.get("log_density", False):
    jax.debug.print(
        "sol_attention density={d:.4f} layer={l} step={s}",
        d=density,
        l=cfg.get("layer_index") if cfg.get("layer_index") is not None else -1,
        s=cfg.get("step_index") if cfg.get("step_index") is not None else -1,
    )

  b, h, l, d = out.shape
  out = out.transpose(0, 2, 1, 3).reshape(b, l, h * d)
  return nn.with_logical_constraint(out, (BATCH, LENGTH, HEAD))
