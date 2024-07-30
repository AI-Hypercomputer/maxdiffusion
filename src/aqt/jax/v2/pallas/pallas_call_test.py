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
"""Test for AQT pallas."""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.pallas import pallas_call as aqt_pl
from aqt.jax.v2.pallas import quantizer
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


QTensor = aqt_tensor.QTensor


class AqtPallasTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          (1024, 1024),
          (1,),
          (256, 256),
      ),
      (
          (1024, 1024),
          (0,),
          (256, 256),
      ),
      (
          (10, 512, 1024),
          (1,),
          (1, 256, 256),
      ),
      (
          (10, 512, 1024),
          (2,),
          (1, 256, 256),
      ),
  )
  def test_pallas_call_with_single_arg(
      self, tensor_shape, calibration_axes, block_shape
  ):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, tensor_shape, minval=-3, maxval=3)
    block_spec = pl.BlockSpec(block_shape, lambda *args: args)

    qx = quantizer.quant(x, 8, calibration_axes=calibration_axes)

    def pallas_dequant(qx, block_spec):
      grid = [
          ndim // blk_ndim for ndim, blk_ndim in zip(tensor_shape, block_shape)
      ]

      def dequant_kernel(qx: QTensor, out_ref):
        qx.qvalue = qx.qvalue[...]
        qx.scale[0] = qx.scale[0][...]
        out_ref[...] = qx.dequant()

      dequant_out = aqt_pl.pallas_call(
          dequant_kernel,
          grid=tuple(grid),
          in_specs=[block_spec],
          out_specs=block_spec,
          out_shape=jax.ShapeDtypeStruct(shape=tensor_shape, dtype=jnp.float32),
          interpret=False,
      )(qx)
      return dequant_out

    # `pallas_dequant` dequantizes the each tile of qx whereas qx.dequant()
    # dequantizes the whole tensor. They should produces the same result.
    self.assertTrue((pallas_dequant(qx, block_spec) == qx.dequant()).all())

  @parameterized.parameters(
      (
          (10, 512, 1024),
          (1,),
          (1, 256, 256),
          3.3
      ),
      (
          (10, 512, 1024),
          (2,),
          (1, 256, 256),
          4.1
      ),
  )
  def test_pallas_call_prefetch_scalar(
      self, tensor_shape, calibration_axes, block_shape, multiple
  ):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, tensor_shape, minval=-3, maxval=3)

    def index_map(i, j, k, _):
      return i, j, k

    block_spec = pl.BlockSpec(block_shape, index_map)

    qx = quantizer.quant(x, 8, calibration_axes=calibration_axes)

    # All values of multiple are the same.
    multiple = jnp.empty((128,)).at[:].set(multiple)

    def dequant_and_multiply(qx, multiple, block_spec):
      grid = [
          ndim // blk_ndim for ndim, blk_ndim in zip(tensor_shape, block_shape)
      ]

      def dequant_kernel(s, qx: QTensor, out_ref):
        qx.qvalue = qx.qvalue[...]
        qx.scale[0] = qx.scale[0][...]
        out_ref[...] = s[0] * qx.dequant()

      dequant_out = aqt_pl.pallas_call(
          dequant_kernel,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              grid=tuple(grid),
              in_specs=[block_spec],
              out_specs=block_spec,
          ),
          out_shape=jax.ShapeDtypeStruct(shape=tensor_shape, dtype=jnp.float32),
          interpret=False,
      )(multiple, qx)
      return dequant_out

    # `pallas_dequant` dequantizes the each tile of qx whereas qx.dequant()
    # dequantizes the whole tensor. They should produces the same result.
    self.assertTrue(
        (
            dequant_and_multiply(qx, multiple, block_spec)
            == (multiple[0] * qx.dequant())
        ).all()
    )


if __name__ == "__main__":
  absltest.main()
