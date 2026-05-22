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

import unittest
from unittest.mock import patch
from maxdiffusion.models.ltx2.logical_sharding_ltx2 import (
    get_sharding_specs,
    LTX2DiTShardingSpecs,
    TextConnectorShardingSpecs,
    VAEShardingSpecs,
)


class TestLogicalShardingLTX2(unittest.TestCase):

  def test_get_sharding_specs_ironwood(self):
    specs = get_sharding_specs("ironwood", "ltx2_dit")
    self.assertIsInstance(specs, LTX2DiTShardingSpecs)
    # Check specific ironwood overrides
    self.assertEqual(specs.qkv_kernel, (None, "heads"))
    self.assertEqual(specs.out_kernel, ("heads", None))
    self.assertEqual(specs.embed_kernel, (None, None))
    self.assertEqual(specs.norm_scale, (None,))
    # Check specific defaults that should be retained
    self.assertEqual(specs.qkv_bias, ("heads",))
    self.assertEqual(specs.net_0_bias, ("mlp",))

  def test_get_sharding_specs_trillium(self):
    specs = get_sharding_specs("trillium", "ltx2_dit")
    self.assertIsInstance(specs, LTX2DiTShardingSpecs)
    # Check specific trillium defaults (FSDP)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))
    self.assertEqual(specs.out_kernel, ("heads", "embed"))
    self.assertEqual(specs.embed_kernel, (None, "embed"))
    self.assertEqual(specs.norm_scale, ("norm",))

  @patch("maxdiffusion.models.ltx2.logical_sharding_ltx2.get_tpu_type")
  def test_get_sharding_specs_default_cpu(self, mock_get_tpu_type):
    from maxdiffusion.tpu_utils import TpuType

    mock_get_tpu_type.return_value = TpuType.UNKNOWN
    specs = get_sharding_specs("default", "ltx2_dit")
    self.assertIsInstance(specs, LTX2DiTShardingSpecs)
    # Unknown should fallback to trillium
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  @patch("maxdiffusion.models.ltx2.logical_sharding_ltx2.get_tpu_type")
  def test_get_sharding_specs_default_ironwood(self, mock_get_tpu_type):
    from maxdiffusion.tpu_utils import TpuType

    mock_get_tpu_type.return_value = TpuType.TPU_7X
    specs = get_sharding_specs("default", "ltx2_dit")
    self.assertIsInstance(specs, LTX2DiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, (None, "heads"))

  def test_get_sharding_specs_invalid_strategy(self):
    # Should fallback to trillium with a warning
    specs = get_sharding_specs("some_made_up_strategy", "ltx2_dit")
    self.assertIsInstance(specs, LTX2DiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  def test_get_sharding_specs_invalid_component(self):
    with self.assertRaises(ValueError):
      get_sharding_specs("trillium", "invalid_component")

  def test_text_connector_specs_ironwood(self):
    specs = get_sharding_specs("ironwood", "text_connector")
    self.assertIsInstance(specs, TextConnectorShardingSpecs)
    # Projection specs should be fully replicated on Ironwood
    self.assertEqual(specs.proj_kernel, (None, None))
    self.assertEqual(specs.proj_bias, (None,))
    # Attention specs should match DiT pattern
    self.assertEqual(specs.qkv_kernel, (None, "heads"))

  def test_text_connector_specs_trillium(self):
    specs = get_sharding_specs("trillium", "text_connector")
    self.assertIsInstance(specs, TextConnectorShardingSpecs)
    # Projection specs should be replicated (matching original behavior)
    self.assertEqual(specs.proj_kernel, (None, None))
    self.assertEqual(specs.proj_bias, (None,))
    # Attention specs should use FSDP defaults
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  def test_vae_specs_ironwood(self):
    specs = get_sharding_specs("ironwood", "vae")
    self.assertIsInstance(specs, VAEShardingSpecs)
    # ResNet specs should be fully replicated on Ironwood
    self.assertEqual(specs.scale_shift_table, (None, None))
    self.assertEqual(specs.per_channel_scale, (None,))
    # Embedding specs
    self.assertEqual(specs.emb_linear_1_kernel, (None, "mlp"))

  def test_vae_specs_trillium(self):
    specs = get_sharding_specs("trillium", "vae")
    self.assertIsInstance(specs, VAEShardingSpecs)
    # ResNet specs should be replicated (matching original behavior)
    self.assertEqual(specs.scale_shift_table, (None, None))
    self.assertEqual(specs.per_channel_scale, (None,))
    # Embedding specs should use FSDP defaults
    self.assertEqual(specs.emb_linear_1_kernel, ("embed", "mlp"))


if __name__ == "__main__":
  unittest.main()
