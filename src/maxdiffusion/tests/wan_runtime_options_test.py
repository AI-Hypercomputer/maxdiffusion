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

import glob
import os
import types
import unittest
from unittest import mock

import yaml

from maxdiffusion import wan_runtime_options as opts

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


class WanRuntimeOptionsTest(unittest.TestCase):

  def test_every_wan_yaml_declares_every_switch_with_the_module_default(self):
    """The YAML is the source of truth, so it must list every switch -- and agree with the code default."""
    paths = sorted(glob.glob(os.path.join(_CONFIG_DIR, "base_wan*.yml")))
    self.assertTrue(paths)
    for path in paths:
      with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
      for name in opts.names():
        with self.subTest(config=os.path.basename(path), option=name):
          self.assertIn(name, cfg)
          self.assertEqual(opts.coerce(name, cfg[name]), opts.default(name))

  def test_config_is_authoritative_over_legacy_env(self):
    with mock.patch.dict(os.environ, {"WAN_SPLASH_TRANSPOSE_OUT": "1", "WAN_ROPE_NORM_MODE": "fused"}):
      cfg = types.SimpleNamespace(wan_splash_transpose_out=False, wan_rope_norm_mode="exact")
      self.assertFalse(opts.resolve_from_config(cfg, "wan_splash_transpose_out"))
      self.assertEqual(opts.resolve_from_config(cfg, "wan_rope_norm_mode"), "exact")

  def test_legacy_env_used_only_without_config_key(self):
    with mock.patch.dict(os.environ, {"WAN_CFG_BEFORE_UNPATCHIFY": "0"}):
      self.assertFalse(opts.resolve_from_config(types.SimpleNamespace(), "wan_cfg_before_unpatchify"))
      self.assertFalse(opts.resolve_from_config(None, "wan_cfg_before_unpatchify"))
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("WAN_CFG_BEFORE_UNPATCHIFY", None)
      self.assertEqual(
          opts.resolve_from_config(None, "wan_cfg_before_unpatchify"), opts.default("wan_cfg_before_unpatchify")
      )

  def test_command_line_strings_are_coerced(self):
    cfg = types.SimpleNamespace(wan_fuse_qk_prescale="false", wan_splash_transpose_out="true")
    self.assertFalse(opts.resolve_from_config(cfg, "wan_fuse_qk_prescale"))
    self.assertTrue(opts.resolve_from_config(cfg, "wan_splash_transpose_out"))
    with self.assertRaises(ValueError):
      opts.resolve_from_config(types.SimpleNamespace(wan_fuse_qk_prescale="maybe"), "wan_fuse_qk_prescale")

  def test_invalid_norm_mode_raises(self):
    with self.assertRaises(ValueError):
      opts.resolve_from_config(types.SimpleNamespace(wan_rope_norm_mode="invalid"), "wan_rope_norm_mode")

  def test_rope_accum_validation_and_snapshot_from_config(self):
    for mode in ("auto", "dtype", "f32"):
      self.assertEqual(opts.resolve_from_config(types.SimpleNamespace(wan_rope_accum=mode), "wan_rope_accum"), mode)
    with self.assertRaises(ValueError):
      opts.resolve_from_config(types.SimpleNamespace(wan_rope_accum="fp32"), "wan_rope_accum")

    snap_default = opts.snapshot_from_config(types.SimpleNamespace())
    snap_custom = opts.snapshot_from_config(types.SimpleNamespace(wan_splash_transpose_out=True))
    self.assertNotEqual(snap_default, snap_custom)

  def test_no_process_global_store(self):
    """Resolving one config must not change what another (config-less) caller sees."""
    self.assertFalse(hasattr(opts, "get"))
    self.assertFalse(hasattr(opts, "configure_from_config"))
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("WAN_SPLASH_TRANSPOSE_OUT", None)
      opts.attention_config_entries(types.SimpleNamespace(wan_splash_transpose_out=True))
      self.assertEqual(opts.resolve_from_config(None, "wan_splash_transpose_out"), opts.default("wan_splash_transpose_out"))

  def test_pyconfig_initialize_coerces_wan_switches(self):
    from maxdiffusion import pyconfig

    pyconfig.initialize([
        None,
        os.path.join(_CONFIG_DIR, "base_wan_14b.yml"),
        "run_name=test_wan_opts",
        "wan_splash_transpose_out=true",
        "wan_rope_accum=f32",
    ])
    self.assertIs(pyconfig.config.wan_splash_transpose_out, True)
    self.assertEqual(pyconfig.config.wan_rope_accum, "f32")
    entries = opts.attention_config_entries(pyconfig.config)
    self.assertTrue(entries["wan_splash_transpose_out"])
    self.assertEqual(entries["wan_rope_accum"], "f32")


class WanModuleOptionsTest(unittest.TestCase):
  """Wan modules carry every switch on the GraphDef; None is resolved at build time."""

  def _attn(self, **attention_config):
    from flax import nnx
    from maxdiffusion.models.attention_flax import FlaxWanAttention

    return FlaxWanAttention(
        rngs=nnx.Rngs(0),
        query_dim=64,
        heads=2,
        dim_head=32,
        attention_kernel="dot_product",
        attention_config=attention_config or None,
    )

  def test_unset_options_resolve_to_defaults_not_env(self):
    env = {
        opts.env_var(name): "1" if opts.default(name) is False else "0"
        for name in opts.names()
        if isinstance(opts.default(name), bool)
    }
    env.update({"WAN_ROPE_NORM_MODE": "fused", "WAN_ROPE_ACCUM": "f32"})
    with mock.patch.dict(os.environ, env):
      attn = self._attn()
    # wan_patch_embed_mode / wan_seq_pad ride in attention_config too, but are
    # consumed by WanModel (covered in wan_patch_embed_test / wan_seq_pad_test).
    for name in sorted(set(opts.ATTENTION_OPTIONS) - {"wan_patch_embed_mode", "wan_seq_pad"}):
      with self.subTest(option=name):
        self.assertEqual(getattr(attn, name), opts.default(name))

  def test_explicit_options_are_coerced_onto_the_module(self):
    attn = self._attn(wan_splash_transpose_out="true", wan_rope_norm_mode="fused", wan_cross_attn_prescale_kv=True)
    self.assertIs(attn.wan_splash_transpose_out, True)
    self.assertEqual(attn.wan_rope_norm_mode, "fused")
    self.assertTrue(attn.cross_attn_prescale_kv)
    self.assertIs(attn.attention_op.transpose_out, True)

  def test_wan_model_resolves_cfg_before_unpatchify(self):
    from flax import nnx
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    kwargs = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 32,
        "freq_dim": 32,
        "ffn_dim": 64,
        "num_layers": 1,
        "scan_layers": False,
    }
    with mock.patch.dict(os.environ, {"WAN_CFG_BEFORE_UNPATCHIFY": "0"}):
      model = nnx.eval_shape(lambda: WanModel(rngs=nnx.Rngs(0), **kwargs))
    self.assertIs(model.wan_cfg_before_unpatchify, opts.default("wan_cfg_before_unpatchify"))
    model = nnx.eval_shape(lambda: WanModel(rngs=nnx.Rngs(0), **kwargs, wan_cfg_before_unpatchify="false"))
    self.assertIs(model.wan_cfg_before_unpatchify, False)

  def test_enum_options_validation(self):
    valid_cases = {
        "wan_cross_attn_kernel": ("xla", "pallas"),
        "wan_patch_embed_mode": ("conv", "tokens"),
        "wan_ulysses_out_a2a": ("flat", "chunked", "shard_major"),
        "wan_seq_pad": ("off", "lane"),
    }
    for name, valid_values in valid_cases.items():
      for val in valid_values:
        with self.subTest(option=name, valid=val):
          self.assertEqual(opts.resolve_from_config(types.SimpleNamespace(**{name: val}), name), val)
      with self.subTest(option=name, invalid="bogus"):
        with self.assertRaisesRegex(ValueError, name):
          opts.resolve_from_config(types.SimpleNamespace(**{name: "bogus"}), name)


if __name__ == "__main__":
  unittest.main()
