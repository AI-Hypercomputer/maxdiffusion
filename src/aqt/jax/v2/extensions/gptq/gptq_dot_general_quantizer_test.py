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

"""This test file only tests utility functions for gptq dot_general.

For e2e test, see flax_e2e_model_test.py.
"""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2.extensions.gptq import gptq_dot_general_quantizer
import jax


class GptqDotGeneralQuantizerTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          param_shape=(4, 5, 16),
          ca=[2],
          block_size=8,
          expected_converted_kernel_shape=(2, 8, 20),
      ),
      dict(
          param_shape=(4, 5, 6, 8),
          ca=[1, 3],
          block_size=4,
          expected_converted_kernel_shape=(10, 4, 24),
      ),
      dict(
          param_shape=(6, 8, 10, 12),
          ca=[0, 1, 2],
          block_size=24,
          expected_converted_kernel_shape=(20, 24, 12),
      ),
  ])
  def test_kernel_shape_transform(
      self, param_shape, ca, block_size, expected_converted_kernel_shape
  ):
    """Tests kernel transform before and after applying GPTQ."""

    kernel = jax.random.uniform(jax.random.PRNGKey(0), shape=param_shape)
    new_kernel, kernel_feature_grouped_shape = (
        gptq_dot_general_quantizer._reshape_kernel_for_gptq(
            kernel, ca, None, False, None, block_size
        )
    )
    self.assertEqual(expected_converted_kernel_shape, new_kernel.shape)
    recovered_kernel = (
        gptq_dot_general_quantizer._recover_kernel_from_gptq_result(
            new_kernel,
            ca,
            None,
            False,
            None,
            kernel.dtype,
            kernel_feature_grouped_shape,
        )
    )

    assert (kernel == recovered_kernel).all()


if __name__ == "__main__":
  absltest.main()
