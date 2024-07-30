# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transpose."""

import math
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import transpose
import jax.numpy as jnp


class AqtTransposeTest(parameterized.TestCase):

  @parameterized.parameters(
      ((10, 1, 10), (0, 2, 1)),
      ((10, 1, 5), (1, 0, 2)),
      ((5, 1, 10), (0, 1, 2)),
      ((10, 20, 30), (1, 2, 0)),
      ((10, 1, 1), (0, 2, 1)),
      ((10, 1, 1), (2, 1, 0)),
      ((10, 1, 1), (1, 2, 0)),
  )
  def test_transpose(self, tensor_shape, transpose_axes):
    t = jnp.arange(math.prod(tensor_shape)).reshape(tensor_shape)
    t_t = transpose.transpose(t, transpose_axes)
    self.assertTrue((t_t == jnp.transpose(t, transpose_axes)).all())

  @parameterized.parameters(
      # 'bmnts,bsnh->bmtnh'
      (
          (2, 1, 4, 7, 1),
          (2, 6, 4, 7, 3),
          (2, 3, 4, 5),
          (((4,), (1,)), ((0, 2), (0, 2))),
          (2, 4, 1, 7, 1),
      ),
      # 'bmkgts,bskh->bmtkgh'
      (
          (2, 1, 4, 1, 1, 1),
          (2, 6, 4, 7, 8, 3),
          (2, 3, 4, 5),
          (((5,), (1,)), ((0, 2), (0, 2))),
          (2, 4, 1, 1, 1, 1),
      ),
  )
  def test_lhs_scale_transpose_to_output(
      self,
      lhs_scale_shape,
      lhs_shape,
      rhs_shape,
      dimension_numbers,
      expected_qlhs_scale_t_shape
  ):
    lhs_scale = jnp.zeros(lhs_scale_shape)
    lhs = jnp.zeros(lhs_shape)
    rhs = jnp.zeros(rhs_shape)
    qlhs_scale_t = transpose.lhs_scale_transpose_to_output(
        lhs_scale, dimension_numbers, lhs.shape, rhs.shape
    )

    self.assertIsNotNone(qlhs_scale_t)
    self.assertEqual(qlhs_scale_t.shape, expected_qlhs_scale_t_shape)

    # Test recover.
    qlhs_scale_recovered = transpose.lhs_recover_scale_from_scale_t(
        qlhs_scale_t, dimension_numbers, lhs.shape, rhs_shape
    )
    self.assertEqual(lhs_scale.shape, qlhs_scale_recovered.shape)
    assert (lhs_scale == qlhs_scale_recovered).all()

  @parameterized.parameters(
      # 'bmnts,bsnh->bmtnh'
      (
          (2, 3, 4, 5, 6),
          (2, 1, 4, 1),
          (2, 6, 4, 7),
          (((4,), (1,)), ((0, 2), (0, 2))),
          (2, 4, 1, 1, 1),
      ),
      # 'bmkgts,bskh->bmtkgh'
      (
          (2, 3, 4, 5, 6, 7),
          (2, 1, 4, 1),
          (2, 7, 4, 8),
          (((5,), (1,)), ((0, 2), (0, 2))),
          (2, 4, 1, 1, 1, 1),
      ),
  )
  def test_rhs_scale_transpose_to_output(
      self,
      lhs_shape,
      rhs_scale_shape,
      rhs_shape,
      dimension_numbers,
      expected_qrhs_scale_t_shape
  ):
    lhs = jnp.zeros(lhs_shape)
    rhs = jnp.zeros(rhs_shape)
    rhs_scale = jnp.zeros(rhs_scale_shape)
    qrhs_scale_t = transpose.rhs_scale_transpose_to_output(
        rhs_scale, dimension_numbers, lhs.shape, rhs.shape
    )

    self.assertIsNotNone(qrhs_scale_t)
    self.assertEqual(qrhs_scale_t.shape, expected_qrhs_scale_t_shape)

    # Test recover.
    qrhs_scale_recovered = transpose.rhs_recover_scale_from_scale_t(
        qrhs_scale_t, dimension_numbers, lhs.shape, rhs_shape
    )
    self.assertEqual(rhs_scale.shape, qrhs_scale_recovered.shape)
    assert (rhs_scale == qrhs_scale_recovered).all()

  @parameterized.parameters(
      # 'bmnts,bsnh->bmtnh'
      (
          (2, 1, 4, 1, 3),
          (2, 3, 4, 5),
          (((4,), (1,)), ((0, 2), (0, 2))),
          (2, 3, 4, 1),
      ),
      # 'bmkgts,bskh->bmtkgh'
      (
          (2, 1, 4, 1, 1, 3),
          (2, 3, 4, 5),
          (((5,), (1,)), ((0, 2), (0, 2))),
          (2, 3, 4, 1),
      ),
  )
  def test_lhs_scale_transpose_for_rhs_input(
      self, lhs_scale_shape, rhs_shape, dimension_numbers, expected_shape
  ):
    lhs_scale = jnp.zeros(lhs_scale_shape)
    rhs = jnp.zeros(rhs_shape)
    result = transpose.lhs_scale_transpose_for_rhs_input(
        lhs_scale, dimension_numbers, rhs.shape
    )

    self.assertEqual(result.shape, expected_shape)

  @parameterized.parameters(
      # 'bmnts,bsnh->bmtnh'
      (
          (2, 3, 4, 5, 6),
          (2, 6, 4, 1),
          (((4,), (1,)), ((0, 2), (0, 2))),
          (2, 1, 4, 1, 6),
      ),
      # 'bmkgts,bskh->bmtkgh'
      (
          (2, 3, 4, 5, 6, 7),
          (2, 7, 4, 1),
          (((5,), (1,)), ((0, 2), (0, 2))),
          (2, 1, 4, 1, 1, 7),
      ),
  )
  def test_rhs_scale_transpose_for_lhs_input(
      self, lhs_shape, rhs_scale_shape, dimension_numbers, expected_shape
  ):
    lhs = jnp.zeros(lhs_shape)
    rhs_scale = jnp.zeros(rhs_scale_shape)
    result = transpose.rhs_scale_transpose_for_lhs_input(
        rhs_scale, dimension_numbers, lhs.shape
    )

    self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
  absltest.main()
