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

The kernel-family registry is the single source of truth: every consumer must
use it, and every kernel it names must actually be registered.
"""

import unittest

from maxdiffusion import attention_kernel_registry as registry
from maxdiffusion.utils import tile_size_grid_search as tss


class AttentionKernelRegistryTest(unittest.TestCase):

  def test_every_family_kernel_is_registered(self):
    from maxdiffusion.models import attention_flax  # pylint: disable=g-import-not-at-top

    families = (
        registry.PURE_RING_ATTENTION_KERNELS
        | registry.ULYSSES_RING_ATTENTION_KERNELS
        | registry.ULYSSES_CUSTOM_ATTENTION_KERNELS
    )
    self.assertEqual(sorted(families - set(attention_flax.KERNEL_REGISTRY)), [])

  def test_iperm_kernels_are_ulysses_ring_kernels(self):
    self.assertTrue(registry.INTERNAL_PERM_KERNELS <= registry.ULYSSES_RING_ATTENTION_KERNELS)
    self.assertIn("ulysses_ring_custom_iperm_fixed_m_hybrid", registry.INTERNAL_PERM_KERNELS)

  def test_consumers_share_the_registry_objects(self):
    # Identity, not equality: a consumer holding its own copy can drift again.
    self.assertIs(tss.INTERNAL_PERM_KERNELS, registry.INTERNAL_PERM_KERNELS)
    self.assertIs(tss.ULYSSES_RING_ATTENTION_KERNELS, registry.ULYSSES_RING_ATTENTION_KERNELS)
    self.assertIs(tss.PURE_RING_ATTENTION_KERNELS, registry.PURE_RING_ATTENTION_KERNELS)

  def test_cross_attention_remap_covers_all_sequence_sharded_custom_kernels(self):
    remap = registry.CROSS_ATTENTION_REMAPPED_TO_FLASH_KERNELS
    self.assertTrue(registry.ULYSSES_RING_ATTENTION_KERNELS <= remap)
    self.assertTrue(registry.PURE_RING_ATTENTION_KERNELS <= remap)
    self.assertTrue(registry.ULYSSES_CUSTOM_ATTENTION_KERNELS <= remap)
    self.assertNotIn("flash", remap)

  def test_auto_block_q_sub(self):
    # Largest 128-multiple divisor <= floor128(outer / 4).
    self.assertEqual(registry.auto_block_q_sub(48640), 12160)  # the production OOM config
    self.assertEqual(registry.auto_block_q_sub(37632), 6272)  # the 10s dp2cp4 winner
    self.assertEqual(registry.auto_block_q_sub(28160), 7040)
    self.assertEqual(registry.auto_block_q_sub(512), 128)
    self.assertEqual(registry.auto_block_q_sub(128), 128)
    # 45952 = 128 * 359 (prime): no divisor in range -> the 128 floor.
    self.assertEqual(registry.auto_block_q_sub(45952), 128)
    for outer in (4992, 9984, 18816, 23040, 46080):
      q_sub = registry.auto_block_q_sub(outer)
      self.assertEqual(outer % q_sub, 0)
      self.assertEqual(q_sub % 128, 0)
      self.assertLessEqual(q_sub, max(128, outer // 4))


if __name__ == "__main__":
  unittest.main()
