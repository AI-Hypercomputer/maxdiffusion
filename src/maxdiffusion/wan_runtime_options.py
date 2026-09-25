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

"""Wan inference switches that change the compiled graph.

These were previously read ad hoc from `WAN_*` environment variables at trace
time. They now come from the YAML config (`wan_rope_norm_mode`, ...), loaded
once by `configure_from_config` at the start of a run, so a run is fully
described by its config and every switch lands in the AOT cache fingerprint.

Resolution order for `get(name)`:
  1. the value set by `configure_from_config` (the YAML / command line);
  2. otherwise the legacy `WAN_*` environment variable, so code paths that
     never load a config (unit tests, notebooks) keep working;
  3. otherwise the built-in default, which matches the shipped YAML.
"""

import os
from typing import Any

from maxdiffusion import max_logging

# config key -> (legacy env var, default, kind)
_OPTIONS: dict[str, tuple[str, Any, str]] = {
    "wan_rope_norm_mode": ("WAN_ROPE_NORM_MODE", "exact", "str"),
    "wan_fuse_qk_prescale": ("WAN_FUSE_QK_PRESCALE", True, "bool"),
    "wan_splash_transpose_out": ("WAN_SPLASH_TRANSPOSE_OUT", False, "bool"),
    "wan_cfg_before_unpatchify": ("WAN_CFG_BEFORE_UNPATCHIFY", True, "bool"),
    "wan_cross_attn_prescale_kv": ("WAN_CROSS_ATTN_PRESCALE_KV", False, "bool"),
    "wan_rope_accum": ("WAN_ROPE_ACCUM", "auto", "str"),
    "wan_cross_attn_kernel": ("WAN_CROSS_ATTN_KERNEL", "xla", "str"),
    "wan_patch_embed_mode": ("WAN_PATCH_EMBED_MODE", "conv", "str"),
    "wan_ulysses_out_a2a": ("WAN_ULYSSES_OUT_A2A", "flat", "str"),
    "wan_seq_pad": ("WAN_SEQ_PAD", "off", "str"),
    "wan_cross_attn_cpu_interpret": ("WAN_CROSS_ATTN_CPU_INTERPRET", False, "bool"),
}

_configured: dict[str, Any] = {}


def _coerce(name: str, value: Any) -> Any:
  kind = _OPTIONS[name][2]
  if kind == "bool":
    if isinstance(value, bool):
      return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
      return True
    if v in ("0", "false", "no", "off"):
      return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")
  val = str(value)
  if name == "wan_rope_norm_mode" and val not in ("exact", "fused"):
    raise ValueError(f"wan_rope_norm_mode must be 'exact' or 'fused', got {value!r}.")
  return val


def names() -> tuple[str, ...]:
  return tuple(_OPTIONS)


def env_var(name: str) -> str:
  return _OPTIONS[name][0]


def configure_from_config(config) -> None:
  """Loads every switch from `config`, which becomes authoritative for this process."""
  keys = config.get_keys() if hasattr(config, "get_keys") else vars(config)
  for name, (env, default, _) in _OPTIONS.items():
    value = _coerce(name, keys[name]) if name in keys else default
    legacy = os.environ.get(env)
    if legacy is not None and _coerce(name, legacy) != value:
      max_logging.log(
          f"[wan_runtime_options] Ignoring legacy env {env}={legacy!r}: config sets {name}={value!r}. "
          f"Set {name} in the YAML or on the command line instead."
      )
    _configured[name] = value


def reset() -> None:
  """Forgets configured values (tests only)."""
  _configured.clear()


def get(name: str) -> Any:
  if name in _configured:
    return _configured[name]
  env, default, _ = _OPTIONS[name]
  legacy = os.environ.get(env)
  return _coerce(name, legacy) if legacy is not None else default


def snapshot() -> dict[str, str]:
  """Effective value of every switch, as strings, for fingerprints and logs."""
  return {name: str(get(name)) for name in _OPTIONS}
