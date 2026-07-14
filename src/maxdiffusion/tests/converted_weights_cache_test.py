# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the memoized torch->flax converted-weights cache."""

import os
import tempfile
import unittest

import ml_dtypes
import numpy as np

from maxdiffusion.models.wan.wan_utils import save_converted_weights, try_load_converted_weights


def _flat_tree():
  return {
      ("blocks", "attn1", "kernel"): np.arange(24, dtype=np.float32).reshape(2, 3, 4),
      ("proj_out", 0, "bias"): np.ones(5, dtype=np.float16),
      # bf16 exercises the uint bit-view path (npy mmap cannot resolve
      # ml_dtypes descriptors directly).
      ("blocks", "ffn", "kernel"): np.arange(12, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(3, 4),
  }


def _eval_shapes(flat):
  shapes = {}
  for key, value in flat.items():
    node = shapes
    for part in key[:-1]:
      node = node.setdefault(part, {})
    node[key[-1]] = value  # only keys/structure are validated
  return shapes


class ConvertedWeightsCacheTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.cache_dir = os.path.join(self._tmp.name, "cache")
    self.flat = _flat_tree()
    self.eval_shapes = _eval_shapes(self.flat)

  def tearDown(self):
    self._tmp.cleanup()

  def test_round_trip(self):
    save_converted_weights(self.cache_dir, self.flat)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, None)
    self.assertIsNotNone(loaded)
    np.testing.assert_array_equal(loaded["blocks"]["attn1"]["kernel"], self.flat[("blocks", "attn1", "kernel")])
    np.testing.assert_array_equal(loaded["proj_out"][0]["bias"], self.flat[("proj_out", 0, "bias")])
    bf16 = loaded["blocks"]["ffn"]["kernel"]
    self.assertEqual(bf16.dtype, np.dtype(ml_dtypes.bfloat16))
    np.testing.assert_array_equal(bf16.view(np.uint16), self.flat[("blocks", "ffn", "kernel")].view(np.uint16))

  def test_missing_cache_returns_none(self):
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None))

  def test_dtype_policy_change_invalidates(self):
    save_converted_weights(self.cache_dir, self.flat)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, lambda key: np.dtype(np.float64))
    self.assertIsNone(loaded)

  def test_key_set_change_invalidates(self):
    save_converted_weights(self.cache_dir, self.flat)
    bigger = dict(self.flat)
    bigger[("new_param", "kernel")] = np.zeros(2, dtype=np.float32)
    loaded = try_load_converted_weights(self.cache_dir, _eval_shapes(bigger), None)
    self.assertIsNone(loaded)

  def test_dtypes_preserved(self):
    save_converted_weights(self.cache_dir, self.flat)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, None)
    for key, value in self.flat.items():
      node = loaded
      for part in key:
        node = node[part]
      self.assertEqual(node.dtype, value.dtype)


if __name__ == "__main__":
  unittest.main()
