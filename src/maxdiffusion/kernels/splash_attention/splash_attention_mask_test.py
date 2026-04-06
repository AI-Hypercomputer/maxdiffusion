# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from . import splash_attention_mask as mask_lib
from . import splash_attention_mask_info as mask_info_lib
from . import splash_attention_test_utils as test_utils


jax.config.parse_flags_with_absl()

# pylint: disable=line-too-long


def _make_lazy_causal_mask(*args, **kwargs):
  mask = mask_lib.CausalMask(*args, **kwargs)
  return mask[:, :]


def _make_causal_mask(*args, **kwargs):
  return mask_lib.make_causal_mask(*args, **kwargs)


def _make_lazy_local_attention_mask(*args, **kwargs):
  mask = mask_lib.LocalMask(*args, **kwargs)
  return mask[:, :]


def _make_local_attention_mask(*args, **kwargs):
  return mask_lib.make_local_attention_mask(*args, **kwargs)


def _make_lazy_chunked_causal_mask(shape, chunk_size):
  mask = mask_lib.ChunkedCausalMask(shape=shape, chunk_size=chunk_size)
  return mask[:, :]


def _make_chunked_causal_mask(shape, chunk_size):
  return mask_lib.make_chunk_attention_mask(shape=shape, chunk_size=chunk_size)


class SplashAttentionMaskTest(test_utils.SplashAttentionTestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.parameters([_make_lazy_causal_mask, _make_causal_mask])
  def test_causal_mask(self, make_causal_mask):
    expected = np.array([[1]], dtype=np.bool_)
    actual = make_causal_mask((1, 1))

    with self.subTest("unit"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_causal_mask((4, 4))

    with self.subTest("square"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_causal_mask((4, 6))

    with self.subTest("wide_rectangle"):
      self._assert_array_equal(actual, expected)

    actual = make_causal_mask((6, 4))
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    with self.subTest("tall_rectangle"):
      self._assert_array_equal(actual, expected)

    actual = make_causal_mask((4, 4), -1)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.bool_,
    )

    with self.subTest("negative_offset"):
      self._assert_array_equal(actual, expected)

    actual = make_causal_mask((4, 4), 1)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    with self.subTest("positive_offset"):
      self._assert_array_equal(actual, expected)

  @parameterized.parameters([_make_lazy_local_attention_mask, _make_local_attention_mask])
  def test_local_attention_mask(self, make_local_attention_mask):
    expected = np.array([[1]], dtype=np.bool_)
    actual = make_local_attention_mask((1, 1), (0, None), offset=0)
    self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, None), offset=0)
    with self.subTest("left_1"):
      self._assert_array_equal(actual, expected)
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (None, 2), offset=0)
    with self.subTest("right_2"):
      self._assert_array_equal(actual, expected)
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self._assert_array_equal(actual, expected)
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self._assert_array_equal(actual, expected)
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self._assert_array_equal(actual, expected)

  @parameterized.parameters([_make_lazy_local_attention_mask, _make_local_attention_mask])
  def test_local_attention_mask_wide_rectangle(self, make_local_attention_mask):
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, None), offset=0)
    with self.subTest("left_1"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (None, 2), offset=0)
    with self.subTest("right_2"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self._assert_array_equal(actual, expected)

  @parameterized.parameters([_make_lazy_local_attention_mask, _make_local_attention_mask])
  def test_local_attention_mask_tall_rectangle(self, make_local_attention_mask):
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, None), offset=0)
    with self.subTest("left_1"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (None, 2), offset=0)
    with self.subTest("right_2"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self._assert_array_equal(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self._assert_array_equal(actual, expected)

  @parameterized.product(
      block_size=[(256, 256), (256, 128), (128, 256)],
      shape=[(1024, 1024), (1024, 2048), (2048, 1024)],
  )
  def test_lazy_causal_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int]):
    dense_mask = mask_lib.make_causal_mask(shape=shape)
    self._compare_masks(
        dense_mask,
        mask_lib.CausalMask(shape),
        block_size,
    )

  @parameterized.parameters(
      [
          ((256, 256), (1024, 1024), (128, None), 0),
          ((256, 128), (1024, 1024), (128, None), 16),
          ((128, 256), (1024, 1024), (128, None), 16),
          ((256, 256), (1024, 1024), (128, 256), 0),
          ((256, 128), (1024, 1024), (128, 256), 0),
          ((128, 256), (1024, 1024), (128, 256), 16),
          ((256, 256), (1024, 1024), (None, 256), 0),
          ((256, 128), (1024, 1024), (None, 256), 32),
          ((128, 256), (1024, 1024), (None, 256), 32),
          #
          ((256, 256), (1024, 2048), (128, None), 0),
          ((256, 128), (1024, 2048), (128, None), 16),
          ((128, 256), (1024, 2048), (128, None), 16),
          ((256, 256), (1024, 2048), (128, 256), 0),
          ((256, 128), (1024, 2048), (128, 256), 0),
          ((128, 256), (1024, 2048), (128, 256), 16),
          ((256, 256), (1024, 2048), (None, 256), 0),
          ((256, 128), (1024, 2048), (None, 256), 32),
          ((128, 256), (1024, 2048), (None, 256), 32),
          #
          ((256, 256), (2048, 1024), (128, None), 0),
          ((256, 128), (2048, 1024), (128, None), 16),
          ((128, 256), (2048, 1024), (128, None), 16),
          ((256, 256), (2048, 1024), (128, 256), 0),
          ((256, 128), (2048, 1024), (128, 256), 0),
          ((128, 256), (2048, 1024), (128, 256), 16),
          ((256, 256), (2048, 1024), (None, 256), 0),
          ((256, 128), (2048, 1024), (None, 256), 32),
          ((128, 256), (2048, 1024), (None, 256), 32),
      ]
  )
  def test_lazy_local_mask_chunking(
      self,
      block_size: tuple[int, int],
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
  ):
    dense_mask = mask_lib.make_local_attention_mask(shape, window_size, offset=offset)
    self._compare_masks(
        dense_mask,
        mask_lib.LocalMask(shape, window_size, offset),
        block_size,
    )

  @parameterized.parameters([_make_lazy_chunked_causal_mask, _make_chunked_causal_mask])
  def test_chunked_causal_mask(self, make_chunked_mask):
    """Tests the chunked causal mask logic for various shapes and chunk sizes."""
    with self.subTest("unit"):
      expected = np.array([[1]], dtype=np.bool_)
      actual = make_chunked_mask(shape=(1, 1), chunk_size=1)
      self._assert_array_equal(actual, expected)
      actual = make_chunked_mask(shape=(1, 1), chunk_size=2)
      self._assert_array_equal(actual, expected)

    with self.subTest("square_exact_chunks"):
      # Chunk 0: [0, 1], Chunk 1: [2, 3]
      expected = np.array(
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
          ],
          dtype=np.bool_,
      )
      actual = make_chunked_mask(shape=(4, 4), chunk_size=2)
      self._assert_array_equal(actual, expected)

    with self.subTest("square_uneven_chunks"):
      expected = np.array(
          [
              [1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 1],
          ],
          dtype=np.bool_,
      )
      actual = make_chunked_mask(shape=(5, 5), chunk_size=3)
      self._assert_array_equal(actual, expected)

    with self.subTest("wide_rectangle"):
      expected = np.array(
          [
              [1, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
          ],
          dtype=np.bool_,
      )
      actual = make_chunked_mask(shape=(4, 6), chunk_size=3)
      self._assert_array_equal(actual, expected)

    with self.subTest("tall_rectangle"):
      expected = np.array(
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 1],
          ],
          dtype=np.bool_,
      )
      actual = make_chunked_mask(shape=(6, 4), chunk_size=3)
      self._assert_array_equal(actual, expected)

    with self.subTest("chunk_size_1"):
      # Should only allow self-attention q==k and chunk_size == 1
      expected = np.array(
          [
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
          ],
          dtype=np.bool_,
      )
      actual = make_chunked_mask(shape=(4, 4), chunk_size=1)
      self._assert_array_equal(actual, expected)

    with self.subTest("chunk_size_greater_equal_seqlen"):
      # Should behave like a normal causal mask
      expected = np.array(
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 1],
          ],
          dtype=np.bool_,
      )
      # Test chunk_size == seqlen
      actual_eq = make_chunked_mask(shape=(4, 4), chunk_size=4)
      self._assert_array_equal(actual_eq, expected)
      # Test chunk_size > seqlen
      actual_gt = make_chunked_mask(shape=(4, 4), chunk_size=5)
      self._assert_array_equal(actual_gt, expected)

  @parameterized.product(
      block_size=[(128, 128), (256, 128), (128, 256)],
      shape=[(512, 512), (512, 1024), (1024, 512)],
      chunk_size=[64, 128, 256, 512, 1024],
  )
  def test_lazy_chunked_causal_mask_chunking(
      self,
      block_size: tuple[int, int],
      shape: tuple[int, int],
      chunk_size: int,
  ):
    """Compares lazy chunked mask evaluation against the dense version block-by-block."""
    q_len, kv_len = shape
    # Adjust block size if it exceeds shape dimensions
    adjusted_block_size = (
        min(block_size[0], q_len),
        min(block_size[1], kv_len),
    )

    if q_len % adjusted_block_size[0] != 0 or kv_len % adjusted_block_size[1] != 0:
      self.skipTest(f"Shape {shape} not divisible by block_size {adjusted_block_size}")

    dense_mask = _make_chunked_causal_mask(shape=shape, chunk_size=chunk_size)
    lazy_mask = mask_lib.ChunkedCausalMask(shape=shape, chunk_size=chunk_size)
    self._compare_masks(
        dense_mask,
        lazy_mask,
        adjusted_block_size,
    )

  def test_chunked_causal_mask_invalid_chunk_size(self):
    """Tests that invalid chunk_size raises ValueError."""
    with self.assertRaises(ValueError):
      mask_lib.ChunkedCausalMask(shape=(10, 10), chunk_size=0)
    with self.assertRaises(ValueError):
      mask_lib.ChunkedCausalMask(shape=(10, 10), chunk_size=-1)
    with self.assertRaises(ValueError):
      mask_lib.make_chunk_attention_mask(shape=(10, 10), chunk_size=0)

  def test_chunked_causal_mask_minimal_equality_hash(self):
    """Tests for __eq__ and __hash__ of ChunkedCausalMask."""
    shape1, chunk_size1 = (128, 256), 16
    shape2, chunk_size2 = (128, 128), 32  # Different shape/chunk_size

    # Create three masks: two identical, one with different shape/chunk_size.
    mask1 = mask_lib.ChunkedCausalMask(shape=shape1, chunk_size=chunk_size1)
    mask2 = mask_lib.ChunkedCausalMask(shape=shape1, chunk_size=chunk_size1)
    mask_diff_shape = mask_lib.ChunkedCausalMask(shape=shape2, chunk_size=chunk_size1)
    mask_diff_chunk = mask_lib.ChunkedCausalMask(shape=shape1, chunk_size=chunk_size2)
    other_obj = object()

    # Test __eq__
    self.assertEqual(mask1, mask2)
    self.assertNotEqual(mask1, mask_diff_shape)
    self.assertNotEqual(mask1, mask_diff_chunk)
    self.assertNotEqual(mask1, other_obj)

    # Test __hash__ of identical masks
    self.assertEqual(hash(mask1), hash(mask2))

    mask_set = {mask1, mask2, mask_diff_chunk}
    self.assertLen(mask_set, 2)  # mask1 and mask2 are duplicates
    self.assertIn(mask1, mask_set)
    self.assertIn(mask_diff_chunk, mask_set)
    self.assertNotIn(mask_diff_shape, mask_set)

  def test_using_logical_operators_raises_exception(self):
    if sys.version_info == (3, 14, 0, "candidate", 1):
      # Fails due to Python bug on 3.14.0rc1
      # https://github.com/python/cpython/issues/137288
      self.skipTest("Expected failure.")
    mask_1 = mask_lib.NumpyMask(mask_lib.make_random_mask((256, 256), 0.5, seed=1))
    mask_2 = mask_lib.NumpyMask(mask_lib.make_random_mask((256, 256), 0.5, seed=2))

    with self.subTest("logical_or"):
      with self.assertRaises(NotImplementedError):
        res = mask_1 or mask_2
        del res

    with self.subTest("logical_and"):
      with self.assertRaises(NotImplementedError):
        res = mask_1 and mask_2
        del res

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_mask_or(self, shape: tuple[int, int]):
    mask_1 = mask_lib.make_random_mask(shape, 0.5, seed=1)
    mask_2 = mask_lib.make_random_mask(shape, 0.5, seed=2)

    lazy_or = mask_lib.NumpyMask(mask_1) | mask_lib.NumpyMask(mask_2)
    dense = np.logical_or(mask_1, mask_2)

    self._compare_masks(dense, lazy_or, (256, 256))

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_mask_and(self, shape: tuple[int, int]):
    mask_1 = mask_lib.make_random_mask(shape, 0.5, seed=1)
    mask_2 = mask_lib.make_random_mask(shape, 0.5, seed=2)

    lazy_and = mask_lib.NumpyMask(mask_1) & mask_lib.NumpyMask(mask_2)
    dense = np.logical_and(mask_1, mask_2)

    self._compare_masks(dense, lazy_and, (256, 256))

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_full_mask(self, shape: tuple[int, int]):
    lazy_full = mask_lib.FullMask(shape)
    dense = np.ones(shape, dtype=np.bool_)

    self._compare_masks(dense, lazy_full, (256, 256))

  def _compare_masks(
      self,
      dense_mask: np.ndarray,
      lazy_mask: mask_lib.Mask,
      block_size: tuple[int, int],
  ):
    self.assertEqual(dense_mask.shape, lazy_mask.shape)

    *prefix, width, height = dense_mask.shape

    assert width % block_size[0] == 0
    assert height % block_size[1] == 0

    full_lazy_mask = lazy_mask[(*[slice(p) for p in prefix], slice(None), slice(None))]
    self._assert_array_equal(dense_mask, full_lazy_mask)
    for i, j in np.ndindex(width // block_size[0], height // block_size[1]):
      indexer = (
          *[slice(p) for p in prefix],
          slice(i * block_size[0], (i + 1) * block_size[0]),
          slice(j * block_size[1], (j + 1) * block_size[1]),
      )
      dense_chunk = dense_mask[indexer]
      lazy_chunk = lazy_mask[indexer]
      self._assert_array_equal(dense_chunk, lazy_chunk)


class SplashAttentionMaskInfoTest(test_utils.SplashAttentionTestCase):
  """Check the construction of MaskInfo from Mask."""

  def _assert_mask_info_match(self, actual: mask_info_lib.MaskInfo, expected: mask_info_lib.MaskInfo):
    def _check_presence(actual, expected):
      return self.assertEqual(actual is not None, expected is not None)

    # TODO: refactor so that all of MaskInfo is possibly None
    _check_presence(actual.mask_next, expected.mask_next)
    _check_presence(actual.partial_mask_blocks, expected.partial_mask_blocks)
    _check_presence(actual.q_sequence, expected.q_sequence)
    _check_presence(actual.block_mask, expected.block_mask)
    _check_presence(actual.active_rows, expected.active_rows)
    _check_presence(actual.active_cols, expected.active_cols)

    self._assert_array_equal(
        actual.num_active_blocks,
        expected.num_active_blocks,
        err_msg="num_active_blocks",
        verbose=True,
    )
    self._assert_array_equal(
        actual.block_mask,
        expected.block_mask,
        err_msg="block_mask",
        verbose=True,
    )
    self._assert_array_equal(
        actual.active_rows,
        expected.active_rows,
        err_msg="active_rows",
        verbose=True,
    )
    self._assert_array_equal(
        actual.active_cols,
        expected.active_cols,
        err_msg="active_cols",
        verbose=True,
    )
    self._assert_array_equal(
        actual.mask_next,
        expected.mask_next,
        err_msg="mask_next",
        verbose=True,
    )
    self._assert_array_equal(
        actual.partial_mask_blocks,
        expected.partial_mask_blocks,
        err_msg="partial_mask_blocks",
        verbose=True,
    )
    self._assert_array_equal(
        actual.q_sequence,
        expected.q_sequence,
        err_msg="q_sequence",
        verbose=True,
    )

  def _process_mask(self, *args, **kwargs):
    mask_info, mask_function = mask_info_lib.process_mask(*args, **kwargs)
    mask_info_dkv, dkv_mask_function = mask_info_lib.process_mask_dkv(*args, **kwargs)
    self.assertEqual(mask_function, dkv_mask_function)
    return mask_info, mask_info_dkv, mask_function

  @parameterized.parameters((True,), (False,))
  def test_full_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      full_mask = mask_lib.FullMask(sequence_lengths)
    else:
      full_mask = mask_lib.NumpyMask(np.ones(sequence_lengths, dtype=np.bool_))

    mask_info, mask_info_dkv, mask_function = self._process_mask(full_mask, block_shape)
    self.assertIsNone(mask_function)

    expected_mask_info = mask_info_lib.MaskInfo(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info)

  def test_no_partial_mask_blocks(self):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    mask = np.ones(sequence_lengths).astype(np.bool_)
    mask[:32, 32:] = False
    mask = mask_lib.NumpyMask(mask)

    mask_info, mask_info_dkv, mask_function = self._process_mask(mask, block_shape)
    self.assertIsNone(mask_function)

    expected_mask_info = mask_info_lib.MaskInfo(
        mask_next=None,
        active_rows=np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int8),
        active_cols=np.array([0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
        block_mask=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        num_active_blocks=np.array([12], dtype=np.int32),
        partial_mask_blocks=None,
        q_sequence=None,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        mask_next=None,
        active_rows=np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3], dtype=np.int8),
        active_cols=np.array([0, 1, 2, 3, 0, 1, 2, 3, 2, 3, 2, 3], dtype=np.int8),
        block_mask=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        num_active_blocks=np.array([12], dtype=np.int32),
        partial_mask_blocks=None,
        q_sequence=None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.product(is_lazy_mask=[True, False], return_dynamic_grid=[True, False])
  def test_rectangular_wide_causal_mask(self, is_lazy_mask: bool, return_dynamic_grid: bool):
    sequence_lengths = (64, 128)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(mask_lib.make_causal_mask(sequence_lengths))

    args = (causal_mask, block_shape)
    mask_info, mask_function = mask_info_lib.process_mask(*args)
    mask_info_dkv, _ = mask_info_lib.process_mask_dkv(*args, return_dynamic_grid=return_dynamic_grid)
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_causal_mask_next = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_active_rows = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int8)
    expected_active_cols = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=np.int8)
    expected_causal_block_mask = np.array([1, 2, 1, 2, 2, 1, 2, 2, 2, 1], dtype=np.int8)
    expected_num_active_blocks = np.array([10], dtype=np.int32)

    if not is_lazy_mask:
      expected_mask_info = mask_info_lib.MaskInfo(
          expected_causal_mask_next,
          expected_active_rows,
          expected_active_cols,
          expected_causal_block_mask,
          expected_num_active_blocks,
          np.tri(*block_shape, dtype=np.int8)[None, ...],
          None,
      )
    else:
      expected_mask_info = mask_info_lib.MaskInfo(
          None,
          expected_active_rows,
          expected_active_cols,
          expected_causal_block_mask,
          expected_num_active_blocks,
          None,
          np.arange(sequence_lengths[0], dtype=np.int32),
      )

    if return_dynamic_grid:
      expected_causal_mask_next_dkv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
      # The grid is extended to visit empty rows to initialize dk/dv.
      expected_active_rows_dkv = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7], dtype=np.int8)
      expected_active_cols_dkv = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3, 0, 0, 0, 0], dtype=np.int8)
      expected_causal_block_mask_dkv = np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 0, 0, 0, 0], dtype=np.int8)
      expected_num_active_blocks_dkv = np.array([14], dtype=np.int32)
    else:
      expected_causal_mask_next_dkv = np.zeros((32,), dtype=np.int8)
      expected_active_rows_dkv = None
      expected_active_cols_dkv = None
      expected_causal_block_mask_dkv = np.array(
          [
              [1, 2, 2, 2],
              [0, 1, 2, 2],
              [0, 0, 1, 2],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
          ],
          dtype=np.int8,
      ).flatten()
      expected_num_active_blocks_dkv = None

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_causal_mask_next_dkv if not is_lazy_mask else None,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_causal_block_mask_dkv,
        expected_num_active_blocks_dkv,
        np.tri(*block_shape, dtype=np.int8).T[None, ...] if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )
    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_rectangular_tall_causal_mask(self, is_lazy_mask: bool):
    sequence_lengths = (128, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(mask_lib.make_causal_mask(sequence_lengths))

    mask_info, mask_info_dkv, mask_function = self._process_mask(causal_mask, block_shape)
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_causal_mask_next = np.array([0] * 26, dtype=np.int8)
    expected_active_rows = np.array(
        [
            0,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
        ],
        dtype=np.int8,
    )
    expected_active_cols = np.array(
        [
            0,
            0,
            1,
            0,
            1,
            2,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
        ],
        dtype=np.int8,
    )
    expected_causal_block_mask = np.array([1, 2, 1, 2, 2, 1, 2, 2, 2, 1] + [2] * 16, dtype=np.int8)
    expected_num_active_blocks = np.array([26], dtype=np.int32)

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_causal_mask_next if not is_lazy_mask else None,
        expected_active_rows,
        expected_active_cols,
        expected_causal_block_mask,
        expected_num_active_blocks,
        np.tri(*block_shape, dtype=np.int8)[None, ...] if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )

    expected_causal_mask_next_dkv = np.array([0] * 26, dtype=np.int8)
    expected_active_rows_dkv = np.array([0] * 8 + [1] * 7 + [2] * 6 + [3] * 5, dtype=np.int8)
    expected_active_cols_dkv = np.concatenate(
        [np.arange(8), np.arange(1, 8), np.arange(2, 8), np.arange(3, 8)],
        dtype=np.int8,
    )
    expected_causal_block_mask_dkv = np.array(
        [1, 2, 2, 2, 2, 2, 2, 2] + [1, 2, 2, 2, 2, 2, 2] + [1, 2, 2, 2, 2, 2] + [1, 2, 2, 2, 2],
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_causal_mask_next_dkv if not is_lazy_mask else None,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_causal_block_mask_dkv,
        expected_num_active_blocks,
        np.tri(*block_shape, dtype=np.int8).T[None, ...] if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_local_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8
    if is_lazy_mask:
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, window_size),
          offset=0,
      )
    else:
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(sequence_lengths, window_size=(window_size, window_size), offset=0)
      )

    mask_info, mask_info_dkv, mask_function = self._process_mask(local_mask, block_shape)
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)

    expected_partial_mask_blocks = np.stack(
        [
            np.triu(np.tri(*block_shape, window_size, dtype=np.int8), -window_size),
            np.tri(*block_shape, -window_size, dtype=np.int8),
            np.triu(np.ones(block_shape, dtype=np.int8), window_size),
        ],
    )
    expected_local_mask_next = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int8)
    expected_active_rows = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=np.int8)
    expected_active_cols = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int8)
    expected_local_block_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    expected_num_active_blocks = np.array([10], dtype=np.int32)

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_local_mask_next if not is_lazy_mask else None,
        expected_active_rows,
        expected_active_cols,
        expected_local_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )

    expected_local_mask_next_dkv = np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0], dtype=np.int8)
    expected_active_rows_dkv = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
        ],
        dtype=np.int8,
    )
    expected_active_cols_dkv = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int8)
    expected_local_block_mask_dkv = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_local_mask_next_dkv if not is_lazy_mask else None,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_local_block_mask_dkv,
        expected_num_active_blocks,
        expected_partial_mask_blocks.mT if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_local_mask_narrow(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8
    if is_lazy_mask:
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, 0),
          offset=0,
      )
    else:
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(sequence_lengths, window_size=(window_size, 0), offset=0)
      )

    mask_info, mask_info_dkv, mask_function = self._process_mask(local_mask, block_shape)

    if is_lazy_mask:
      self.assertIsNotNone(mask_function)

    expected_partial_mask_blocks = np.stack(
        [
            np.triu(np.tri(*block_shape, 0, dtype=np.int8), -window_size),
            np.triu(np.ones(block_shape, dtype=np.int8), window_size),
        ],
    )

    expected_local_mask_next = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    expected_active_rows = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int8)
    expected_active_cols = np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int8)
    expected_local_block_mask = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    expected_num_active_blocks = np.array([7], dtype=np.int32)

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_local_mask_next if not is_lazy_mask else None,
        expected_active_rows,
        expected_active_cols,
        expected_local_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )
    expected_active_rows_dkv = np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int8)
    expected_active_cols_dkv = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int8)

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_local_mask_next if not is_lazy_mask else None,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_local_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks.mT if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32) if is_lazy_mask else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_two_qseq_shards_causal_local_stacked(self):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    causal_mask = mask_lib.make_causal_mask(sequence_lengths)
    local_mask = mask_lib.make_local_attention_mask(sequence_lengths, window_size=(window_size, window_size), offset=0)
    mask = np.concatenate((causal_mask, local_mask), axis=0)
    mask = mask_lib.NumpyMask(mask)

    mask_info, mask_info_dkv, mask_function = self._process_mask(mask, block_shape, q_seq_shards=2)
    self.assertIsNone(mask_function)

    expected_mask_next = np.concatenate(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # causal mask
            np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]),  # local mask
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_rows = np.concatenate(
        [
            np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_cols = np.concatenate(
        [
            np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
            np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_block_mask = np.concatenate(
        [
            np.array([1, 2, 1, 2, 2, 1, 2, 2, 2, 1]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_num_active_blocks = np.array([10, 10], dtype=np.int32)

    expected_partial_mask_blocks = np.stack(
        [
            np.tri(*block_shape, dtype=np.int8),
            np.triu(
                np.tri(*block_shape, window_size, dtype=np.int8),
                -window_size,
            ),
            np.tri(*block_shape, -window_size, dtype=np.int8),
            np.triu(np.ones(block_shape, dtype=np.int8), window_size),
        ]
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_mask_next,
        expected_active_rows,
        expected_active_cols,
        expected_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks,
        None,
    )

    expected_mask_next_dkv = np.concatenate(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # causal mask
            np.array([1, 3, 2, 1, 3, 2, 1, 3, 2, 1]),  # local mask
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_rows_dkv = np.concatenate(
        [
            np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_cols_dkv = np.concatenate(
        [
            np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]),  # causal mask
            np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]),
        ],  # local mask
        axis=0,
        dtype=np.int8,
    )

    expected_block_mask_dkv = np.concatenate(
        [
            np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 1]),  # causal mask
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ],  # local mask
        axis=0,
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_mask_next_dkv,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_block_mask_dkv,
        expected_num_active_blocks,
        expected_partial_mask_blocks.mT,
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.named_parameters(
      {
          "testcase_name": "q_seq_shards_2",
          "q_seq_shards": 2,
          "kv_seq_shards": 1,
      },
      {
          "testcase_name": "kv_seq_shards_2",
          "q_seq_shards": 1,
          "kv_seq_shards": 2,
      },
  )
  def test_two_shards_local_wide_local_narrow_stacked(self, q_seq_shards, kv_seq_shards):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    local_mask_wide = mask_lib.make_local_attention_mask(sequence_lengths, window_size=(window_size, window_size), offset=0)
    local_mask_narrow = mask_lib.make_local_attention_mask(sequence_lengths, window_size=(window_size, 0), offset=0)

    concat_axis = 0 if q_seq_shards > 1 else 1
    mask = np.concatenate((local_mask_wide, local_mask_narrow), axis=concat_axis)

    mask = mask_lib.NumpyMask(mask)

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask,
        block_shape,
        q_seq_shards=q_seq_shards,
        kv_seq_shards=kv_seq_shards,
    )
    self.assertIsNone(mask_function)

    expected_block_mask = np.concatenate(
        [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # local wide block mask
            np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1]),  # local narrow block mask
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_rows = np.concatenate(
        [
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 1, 2, 2, 3, 3, -1, -1, -1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_cols = np.concatenate(
        [
            np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]),
            np.array([0, 0, 1, 1, 2, 2, 3, -1, -1, -1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_num_active_blocks = np.array([10, 7], dtype=np.int32)

    block_wide_1 = np.triu(np.tri(*block_shape, window_size, dtype=np.int8), -window_size)
    block_wide_2 = np.tri(*block_shape, -window_size, dtype=np.int8)
    block_wide_3 = np.triu(np.ones(block_shape, dtype=np.int8), window_size)
    block_narrow = np.triu(np.tri(*block_shape, 0, dtype=np.int8), -window_size)

    if q_seq_shards == 2:
      expected_partial_mask_blocks = np.stack([block_wide_1, block_wide_2, block_wide_3, block_narrow]).astype(np.int8)

      expected_mask_next = np.array(
          [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] + [3, 2, 3, 2, 3, 2, 3, -1, -1, -1],  # local wide mask  # local narrow mask
          dtype=np.int8,
      )

      expected_local_mask_next_dkv = np.array(
          [0, 2, 1, 0, 2, 1, 0, 2, 1, 0] + [3, 2, 3, 2, 3, 2, 3, -1, -1, -1],
          dtype=np.int8,
      )

    else:
      assert kv_seq_shards == 2
      # The global mask is different so the partial mask blocks are processed
      # in a different order.
      expected_partial_mask_blocks = np.stack(
          [block_wide_1, block_wide_2, block_narrow, block_wide_3],
      ).astype(np.int8)

      expected_mask_next = np.array(
          [0, 1, 3, 0, 1, 3, 0, 1, 3, 0] + [2, 3, 2, 3, 2, 3, 2, -1, -1, -1],  # local narrow mask  # local wide mask
          dtype=np.int8,
      )

      expected_local_mask_next_dkv = np.array(
          [0, 3, 1, 0, 3, 1, 0, 3, 1, 0] + [2, 3, 2, 3, 2, 3, 2, -1, -1, -1],
          dtype=np.int8,
      )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_mask_next,
        expected_active_rows,
        expected_active_cols,
        expected_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks,
        None,
    )

    expected_active_rows_dkv = np.concatenate(
        [
            np.array(
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                ]
            ),
            np.array([0, 0, 1, 1, 2, 2, 3, -1, -1, -1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_active_cols_dkv = np.concatenate(
        [
            np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]),
            np.array([0, 1, 1, 2, 2, 3, 3, -1, -1, -1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_block_mask_dkv = np.concatenate(
        [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1]),
        ],
        axis=0,
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_local_mask_next_dkv,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_block_mask_dkv,
        expected_num_active_blocks,
        expected_partial_mask_blocks.mT,
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters(False, True)
  def test_causal_two_q_shards_two_kv_shards(self, return_dynamic_grid):
    q_seq_shards = kv_seq_shards = 2
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    mask = mask_lib.make_causal_mask(sequence_lengths, 0)
    mask = mask_lib.NumpyMask(mask)

    args = (mask, block_shape)
    kwargs = {
        "q_seq_shards": q_seq_shards,
        "kv_seq_shards": kv_seq_shards,
    }
    mask_info, _ = mask_info_lib.process_mask(*args, **kwargs)
    mask_info_dkv, _ = mask_info_lib.process_mask_dkv(
        *args,
        **kwargs,
        return_dynamic_grid=return_dynamic_grid,
    )

    partial_mask_blocks = np.tri(*(block_shape), dtype=np.int8)[None]
    expected_mask_info = mask_info_lib.MaskInfo(
        mask_next=np.array(
            [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1],
            dtype=np.int8,
        ),
        active_rows=np.array(
            [0, 1, 1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 1, 1, -1],
            dtype=np.int8,
        ),
        active_cols=np.array(
            [0, 0, 1, -1, -1, -1, -1, -1, 0, 1, 0, 1, 0, 0, 1, -1],
            dtype=np.int8,
        ),
        block_mask=np.array(
            [1, 2, 1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 1, 2, 1, -1],
            dtype=np.int8,
        ),
        num_active_blocks=np.array([3, 0, 4, 3], dtype=np.int32),
        partial_mask_blocks=partial_mask_blocks,
        q_sequence=None,
    )
    if return_dynamic_grid:
      expected_mask_info_dkv = mask_info_lib.MaskInfo(
          mask_next=np.array(
              [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1],
              dtype=np.int8,
          ),
          active_rows=np.array([0, 0, 1, -1, 0, 1, -1, -1, 0, 0, 1, 1, 0, 0, 1, -1], dtype=np.int8),
          active_cols=np.array([0, 1, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, 1, 1, -1], dtype=np.int8),
          block_mask=np.array([1, 2, 1, -1, 0, 0, -1, -1, 2, 2, 2, 2, 1, 2, 1, -1], dtype=np.int8),
          num_active_blocks=np.array([3, 2, 4, 3], dtype=np.int32),
          partial_mask_blocks=partial_mask_blocks.mT,
          q_sequence=None,
      )
    else:
      expected_mask_info_dkv = mask_info_lib.MaskInfo(
          mask_next=np.array(
              [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
              dtype=np.int8,
          ),
          active_rows=None,
          active_cols=None,
          block_mask=np.array([1, 2, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 0, 1], dtype=np.int8),
          num_active_blocks=None,
          partial_mask_blocks=partial_mask_blocks.mT,
          q_sequence=None,
      )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_huge_mask(self):
    # Don't go too high with the mask size to avoid timeouts. Prefer covering
    # multiple cases rather one very large one. This configuration replicates
    # a realistic training shape. In particular, a large number of head shards
    # and interleaving contribute to increasing processing time.
    sequence_length = (32 * 1024, 32 * 1024)
    block_shape = (512, 1024)

    num_shards = 16
    causal_mask = mask_lib.CausalMask(sequence_length, 0, shard_count=num_shards)

    mask_info, mask_function = mask_info_lib.process_mask(causal_mask, block_shape, q_seq_shards=16)

    self.assertIsNotNone(mask_function)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNone(mask_info.mask_next)
    self.assertIsNone(mask_info.partial_mask_blocks)
    self.assertIsNotNone(mask_info.q_sequence)

  def test_huge_mask2(self):
    sequence_lengths = (32 * 1024, 32 * 1024)
    block_shape = (1024, 1024)
    window_size = 8

    local_mask = mask_lib.LocalMask(
        sequence_lengths,
        window_size=(window_size, window_size),
        offset=0,
    )

    mask_info, mask_function = mask_info_lib.process_mask(local_mask, block_shape)

    self.assertIsNotNone(mask_function)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNone(mask_info.mask_next)
    self.assertIsNone(mask_info.partial_mask_blocks)
    self.assertIsNotNone(mask_info.q_sequence)

  def test_process_invalid_mask(self):
    """Masks with of an all-0 row causes undefined softmax, reject them."""
    sequence_length = 32

    invalid_mask = np.ones((sequence_length, sequence_length), dtype=np.bool_)
    invalid_mask[14, :] = False
    invalid_mask = mask_lib.NumpyMask(invalid_mask)

    with self.assertRaises(ValueError) as ctx:
      mask_info_lib._check_mask(invalid_mask)

    self.assertIn("softmax", str(ctx.exception))

  def test_dynamic_mask(self):
    q_seq_len, kv_seq_len = 8, 8
    block_shape = (2, 4)

    mask = _make_causal_mask((q_seq_len, kv_seq_len))

    process_dynamic_mask_fn = jax.jit(
        mask_info_lib.process_dynamic_mask,
        static_argnames=["block_shape", "is_dkv"],
    )

    args = (mask, block_shape)
    mask_info = process_dynamic_mask_fn(*args)
    mask_info_dkv = process_dynamic_mask_fn(*args, is_dkv=True)

    expected_mask_next = np.array([0, 2, 0, 5, 0, 7, 0, 0], dtype=np.int8)
    expected_block_mask = np.array([1, 1, 2, 1, 2, 1, 0, 0], dtype=np.int8)
    expected_active_rows = np.array([0, 1, 2, 2, 3, 3, -1, -1], dtype=np.int32)
    expected_active_cols = np.array([0, 0, 0, 1, 0, 1, -1, -1], dtype=np.int32)
    expected_num_active_blocks = np.array([6], dtype=np.int32)
    expected_partial_mask_blocks = np.array(
        [
            [[1, 0, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 0, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 0], [1, 1, 1, 1]],
        ],
        dtype=np.int8,
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_mask_next,
        expected_active_rows,
        expected_active_cols,
        expected_block_mask,
        expected_num_active_blocks,
        expected_partial_mask_blocks,
        None,
    )

    expected_mask_next_dkv = np.array([0, 2, 0, 0, 5, 7, 0, 0], dtype=np.int8)
    expected_active_rows_dkv = np.array([0, 0, 0, 0, 1, 1, -1, -1], dtype=np.int32)
    expected_active_cols_dkv = np.array([0, 1, 2, 3, 2, 3, -1, -1], dtype=np.int32)
    expected_block_mask_dkv = np.array([1, 1, 2, 2, 1, 1, 0, 0], dtype=np.int8)
    expected_num_active_blocks_dkv = np.array([6], dtype=np.int32)

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_mask_next_dkv,
        expected_active_rows_dkv,
        expected_active_cols_dkv,
        expected_block_mask_dkv,
        expected_num_active_blocks_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )
    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_find_bounds(self):
    test_cases = [
        ("standard", [0, 0, 1, 1, 2], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], 5),
        ("homogeneous", [5, 5, 5, 5], [1, 0, 0, 0], [0, 0, 0, 1], 5),
        ("alternating", [0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1], 4),
        ("wrap_around", [1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], 4),
        ("padding", [0, 0, -1], [1, 0, 0], [0, 1, 0], 2),
    ]

    for name, arr, exp_start, exp_end, n in test_cases:
      with self.subTest(name):
        start, end = mask_info_lib.find_bounds(np.array(arr))
        np.testing.assert_array_equal(start[:n], np.array(exp_start)[:n])
        np.testing.assert_array_equal(end[:n], np.array(exp_end)[:n])


if __name__ == "__main__":
  absltest.main()
