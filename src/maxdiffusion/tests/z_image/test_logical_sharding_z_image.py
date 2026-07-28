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
from maxdiffusion.models.z_image.logical_sharding_z_image import (
    get_sharding_specs,
    ZImageDiTShardingSpecs,
)


class TestLogicalShardingZImage(unittest.TestCase):

  def test_get_sharding_specs_ironwood(self):
    specs = get_sharding_specs("ironwood", "z_image_dit")
    self.assertIsInstance(specs, ZImageDiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))
    self.assertEqual(specs.out_kernel, ("heads", "embed"))
    self.assertEqual(specs.embed_kernel, ("embed", "heads"))
    self.assertEqual(specs.norm_scale, ("norm",))

  def test_get_sharding_specs_trillium(self):
    specs = get_sharding_specs("trillium", "z_image_dit")
    self.assertIsInstance(specs, ZImageDiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))
    self.assertEqual(specs.out_kernel, ("heads", "embed"))
    self.assertEqual(specs.embed_kernel, ("embed", "heads"))
    self.assertEqual(specs.norm_scale, ("norm",))

  @patch("maxdiffusion.models.z_image.logical_sharding_z_image.get_tpu_type")
  def test_get_sharding_specs_default_cpu(self, mock_get_tpu_type):
    from maxdiffusion.tpu_utils import TpuType

    mock_get_tpu_type.return_value = TpuType.UNKNOWN
    specs = get_sharding_specs("default", "z_image_dit")
    self.assertIsInstance(specs, ZImageDiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  @patch("maxdiffusion.models.z_image.logical_sharding_z_image.get_tpu_type")
  def test_get_sharding_specs_default_ironwood(self, mock_get_tpu_type):
    from maxdiffusion.tpu_utils import TpuType

    mock_get_tpu_type.return_value = TpuType.TPU_7X
    specs = get_sharding_specs("default", "z_image_dit")
    self.assertIsInstance(specs, ZImageDiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  def test_get_sharding_specs_invalid_strategy(self):
    specs = get_sharding_specs("some_made_up_strategy", "z_image_dit")
    self.assertIsInstance(specs, ZImageDiTShardingSpecs)
    self.assertEqual(specs.qkv_kernel, ("embed", "heads"))

  def test_get_sharding_specs_invalid_component(self):
    with self.assertRaises(ValueError):
      get_sharding_specs("trillium", "invalid_component")


if __name__ == "__main__":
  unittest.main()
