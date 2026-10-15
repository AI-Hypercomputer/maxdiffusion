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
time. They now come from the YAML config (`wan_rope_norm_mode`, ...). The
pipelines resolve them once at model-build time and store them on the modules
(`attention_config` / `WanModel.wan_cfg_before_unpatchify`), so the values are
part of the GraphDef and of the AOT cache fingerprint. There is no process-wide
store: nothing reads these switches at trace time, and a module built without
them uses the built-in default (`default(name)`), never a value left behind by
an earlier run in the same process.

Resolution order for `resolve_from_config(config, name)`, evaluated at build
time only:
  1. the value in `config` (the YAML / command line), if the key is present
     and not None;
  2. otherwise the legacy `WAN_*` environment variable;
  3. otherwise the built-in default, which matches the shipped YAML.

The shipped YAMLs set every key, so on a normal run the env var only triggers a
one-time "ignoring" log when it disagrees. The env var takes effect only when
`resolve_from_config` is called with a config that lacks the key (or sets it to
None). Modules built without going through `resolve_from_config` (e.g. a bare
`FlaxWanAttention(...)` or `WanModel(...)` with no such `attention_config`
entry) use `default(name)` and ignore the environment.
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
  if name == "wan_rope_accum" and val not in ("auto", "dtype", "f32"):
    raise ValueError(f"wan_rope_accum must be 'auto', 'dtype', or 'f32', got {value!r}.")
  if name == "wan_cross_attn_kernel" and val not in ("xla", "pallas"):
    raise ValueError(f"wan_cross_attn_kernel must be 'xla' or 'pallas', got {value!r}.")
  if name == "wan_patch_embed_mode" and val not in ("conv", "tokens"):
    raise ValueError(f"wan_patch_embed_mode must be 'conv' or 'tokens', got {value!r}.")
  if name == "wan_ulysses_out_a2a" and val not in ("flat", "chunked", "shard_major"):
    raise ValueError(f"wan_ulysses_out_a2a must be 'flat', 'chunked', or 'shard_major', got {value!r}.")
  if name == "wan_seq_pad" and val not in ("off", "lane"):
    raise ValueError(f"wan_seq_pad must be 'off' or 'lane', got {value!r}.")
  return val


coerce = _coerce


def names() -> tuple[str, ...]:
  return tuple(_OPTIONS)


def env_var(name: str) -> str:
  return _OPTIONS[name][0]


def _extract_keys(config: Any) -> dict[str, Any]:
  if config is None:
    return {}
  if hasattr(config, "get_keys"):
    return config.get_keys()
  if isinstance(config, dict):
    return config
  return vars(config)


def default(name: str) -> Any:
  """Built-in default of `name` (matches the shipped YAML)."""
  return _OPTIONS[name][1]


_WARNED_ENV: set[str] = set()


def resolve_from_config(config: Any, name: str) -> Any:
  """Coerced value of `name`: `config` first, then the legacy env var, then the default."""
  env, dflt, _ = _OPTIONS[name]
  keys = _extract_keys(config)
  legacy = os.environ.get(env)
  if name in keys and keys[name] is not None:
    value = _coerce(name, keys[name])
    if legacy is not None and env not in _WARNED_ENV:
      try:
        legacy_coerced = _coerce(name, legacy)
      except ValueError:
        legacy_coerced = legacy
      if legacy_coerced != value:
        _WARNED_ENV.add(env)
        max_logging.log(
            f"[wan_runtime_options] Ignoring legacy env {env}={legacy!r}: config sets {name}={value!r}. "
            f"Set {name} in the YAML or on the command line instead."
        )
    return value
  return _coerce(name, legacy) if legacy is not None else dflt


ATTENTION_OPTIONS = (
    "wan_rope_norm_mode",
    "wan_fuse_qk_prescale",
    "wan_splash_transpose_out",
    "wan_cross_attn_prescale_kv",
    "wan_rope_accum",
    "wan_cross_attn_kernel",
    "wan_patch_embed_mode",
    "wan_ulysses_out_a2a",
    "wan_seq_pad",
    "wan_cross_attn_cpu_interpret",
)


def attention_config_entries(config: Any) -> dict[str, Any]:
  """The `attention_config` entries every Wan pipeline passes to its attention modules."""
  return {name: resolve_from_config(config, name) for name in ATTENTION_OPTIONS}


def snapshot_from_config(config: Any = None) -> dict[str, str]:
  """Coerced string snapshot of every switch, preferring explicit `config` fields."""
  return {name: str(resolve_from_config(config, name)) for name in _OPTIONS}
