# Copyright 2024 Google LLC
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
"""Tests for quantizer."""

# TODO(wppark): Remove this file. This is a temporary module before the
# official release of AQT quant / dequant API.

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.pallas import quantizer
import jax
import jax.numpy as jnp


QTensor = aqt_tensor.QTensor


class AqtPallasTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          (1024, 1024),
          (1,),
          (1024, 1),
      ),
      (
          (1024, 1024),
          (0,),
          (1, 1024),
      ),
      (
          (10, 512, 1024),
          (1,),
          (10, 1, 1024),
      ),
      (
          (10, 512, 1024),
          (2,),
          (10, 512, 1),
      ),
  )
  def test_quant(self, tensor_shape, calibration_axes, expected_scale_shape):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, tensor_shape, minval=-3, maxval=3)
    qx = quantizer.quant(x, 8, calibration_axes)

    self.assertEqual(qx.qvalue.shape, x.shape)
    self.assertEqual(qx.qvalue.dtype, jnp.int8)
    self.assertEqual(qx.scale[0].shape, expected_scale_shape)
    self.assertEqual(qx.scale[0].dtype, jnp.float32)

if __name__ == "__main__":
  absltest.main()
