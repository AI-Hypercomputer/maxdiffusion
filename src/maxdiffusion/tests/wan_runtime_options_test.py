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

  def setUp(self):
    super().setUp()
    opts.reset()
    self.addCleanup(opts.reset)

  def test_every_wan_yaml_declares_every_switch_with_the_module_default(self):
    """The YAML is the source of truth, so it must list every switch -- and agree with the code default."""
    paths = sorted(glob.glob(os.path.join(_CONFIG_DIR, "base_wan*.yml")))
    self.assertTrue(paths)
    unconfigured = {name: opts.get(name) for name in opts.names()}
    for path in paths:
      with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
      for name in opts.names():
        with self.subTest(config=os.path.basename(path), option=name):
          self.assertIn(name, cfg)
          with mock.patch.dict(os.environ, {opts.env_var(name): str(cfg[name])}):
            self.assertEqual(opts.get(name), unconfigured[name])

  def test_config_is_authoritative_over_legacy_env(self):
    with mock.patch.dict(os.environ, {"WAN_SPLASH_TRANSPOSE_OUT": "1", "WAN_ROPE_NORM_MODE": "fused"}):
      opts.configure_from_config(types.SimpleNamespace(wan_splash_transpose_out=False, wan_rope_norm_mode="exact"))
      self.assertFalse(opts.get("wan_splash_transpose_out"))
      self.assertEqual(opts.get("wan_rope_norm_mode"), "exact")

  def test_missing_config_keys_fall_back_to_defaults_not_env(self):
    with mock.patch.dict(os.environ, {"WAN_CROSS_ATTN_PRESCALE_KV": "1"}):
      opts.configure_from_config(types.SimpleNamespace())
      self.assertFalse(opts.get("wan_cross_attn_prescale_kv"))

  def test_legacy_env_used_only_without_config(self):
    with mock.patch.dict(os.environ, {"WAN_CFG_BEFORE_UNPATCHIFY": "0"}):
      self.assertFalse(opts.get("wan_cfg_before_unpatchify"))

  def test_command_line_strings_are_coerced(self):
    opts.configure_from_config(types.SimpleNamespace(wan_fuse_qk_prescale="false", wan_splash_transpose_out="true"))
    self.assertFalse(opts.get("wan_fuse_qk_prescale"))
    self.assertTrue(opts.get("wan_splash_transpose_out"))
    with self.assertRaises(ValueError):
      opts.configure_from_config(types.SimpleNamespace(wan_fuse_qk_prescale="maybe"))

  def test_snapshot_changes_when_a_switch_changes(self):
    base = opts.snapshot()
    opts.configure_from_config(types.SimpleNamespace(wan_splash_transpose_out=True))
    self.assertNotEqual(base, opts.snapshot())

  def test_invalid_norm_mode_raises(self):
    with self.assertRaises(ValueError):
      opts.configure_from_config(types.SimpleNamespace(wan_rope_norm_mode="invalid"))


if __name__ == "__main__":
  unittest.main()
