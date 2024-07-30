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
"""Test for pallas_tensor."""
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.pallas import pallas_tensor
from jax.experimental import pallas as pl
import jax.numpy as jnp


QTensor = aqt_tensor.QTensor
TransposedTensor = pallas_tensor.TransposedTensor


class PallasTensorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="2d_tensor_channelwise_1",
          qvalue_shape=(512, 512),
          scale_shape=(512, 1),
          block_shape=(128, 128),
          expected_scale_block_shape=(128, 1),
          index_and_expected_index=((2, 3), (2, 0)),
      ),
      dict(
          testcase_name="2d_tensor_channelwise_2",
          qvalue_shape=(512, 512),
          scale_shape=(1, 512),
          block_shape=(128, 128),
          expected_scale_block_shape=(1, 128),
          index_and_expected_index=((2, 3), (0, 3)),
      ),
      dict(
          testcase_name="3d_tensor_channelwise_1",
          qvalue_shape=(10, 512, 512),
          scale_shape=(10, 512, 1),
          block_shape=(1, 128, 128),
          expected_scale_block_shape=(1, 128, 1),
          index_and_expected_index=((3, 2, 3), (3, 2, 0)),
      ),
      dict(
          testcase_name="3d_tensor_channelwise_2",
          qvalue_shape=(10, 512, 512),
          scale_shape=(10, 1, 512),
          block_shape=(1, 128, 128),
          expected_scale_block_shape=(1, 1, 128),
          index_and_expected_index=((3, 2, 3), (3, 0, 3)),
      ),
  )
  def test_qtensor_blockspec_correctness(
      self,
      qvalue_shape,
      scale_shape,
      block_shape,
      expected_scale_block_shape,
      index_and_expected_index,
  ):
    qt = QTensor(
        qvalue=jnp.ones(qvalue_shape, dtype=jnp.int8),
        scale=[jnp.ones(scale_shape, dtype=jnp.float32)],
        scale_t=None,
        dequant_dtype=jnp.float32,
    )
    block_spec = pl.BlockSpec(block_shape, lambda *args: args)
    qt_block_spec = pallas_tensor.make_qtensor_blockspec(qt, block_spec)

    self.assertEqual(qt_block_spec.qvalue, block_spec)
    self.assertEqual(
        qt_block_spec.scale[0].block_shape, expected_scale_block_shape
    )

    index, expected_index = index_and_expected_index
    self.assertEqual(qt_block_spec.scale[0].index_map(*index), expected_index)

  @parameterized.named_parameters(
      dict(
          testcase_name="2d_tensor_channelwise_1",
          tensor_shape=(512, 1),
          block_shape=(128, 1),
          expect_transpose=True,
          expected_permute_axes=[1, 0],
          expected_transposed_tensor_shape=(1, 512),
          expected_block_shape=(1, 128),
      ),
      dict(
          testcase_name="2d_tensor_channelwise_2",
          tensor_shape=(1, 512),
          block_shape=(1, 128),
          expect_transpose=False,
      ),
      dict(
          testcase_name="3d_tensor_channelwise_1",
          tensor_shape=(10, 512, 1),
          block_shape=(1, 128, 1),
          expect_transpose=True,
          expected_permute_axes=[0, 2, 1],
          expected_transposed_tensor_shape=(10, 1, 512),
          expected_block_shape=(1, 1, 128),
      ),
      dict(
          testcase_name="3d_tensor_channelwise_2",
          tensor_shape=(10, 1, 512),
          block_shape=(1, 1, 128),
          expect_transpose=False,
      ),
  )
  def test_transpose_for_memory_saving(
      self,
      tensor_shape,
      block_shape,
      expect_transpose,
      expected_permute_axes=None,
      expected_transposed_tensor_shape=None,
      expected_block_shape=None,
  ):
    t = jnp.ones(tensor_shape)
    block_spec = pl.BlockSpec(block_shape, lambda *args: args)

    transposed_t, transposed_block_spec = (
        pallas_tensor.transpose_tensor_for_memory_saving(t, block_spec)
    )

    if expect_transpose:
      self.assertIsInstance(transposed_t, TransposedTensor)
      self.assertIsInstance(transposed_block_spec, TransposedTensor)
      self.assertEqual(transposed_t.permute_axes, expected_permute_axes)
      self.assertEqual(
          transposed_t.transposed_tensor.shape, expected_transposed_tensor_shape
      )
      self.assertEqual(
          transposed_block_spec.transposed_tensor.block_shape,
          expected_block_shape,
      )
    else:
      self.assertTrue((transposed_t == t).all())
      self.assertEqual(transposed_block_spec, block_spec)


if __name__ == "__main__":
  absltest.main()
