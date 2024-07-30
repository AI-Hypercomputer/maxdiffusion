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
"""Test for dot_general for pallas."""

import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
import aqt.jax.v2.pallas as aqt_pl
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


QTensor = aqt_tensor.QTensor
DequantMode = aqt_pl.DequantMode


class DotGeneralTest(parameterized.TestCase):

  @parameterized.product(
      mkn_and_blk=[
          ((512, 512, 512), (128, 128, 128)),
          ((1024, 1024, 1024), (256, 128, 256)),
          ((2048, 2048, 2048), (512, 512, 512)),
          ((4096, 4096, 4096), (512, 512, 512)),
          ((8192, 8192, 8192), (512, 512, 512)),
      ],
      quantize_lhs=[True, False],
      quantize_rhs=[True, False],
  )
  def test_quantized_matmul_error(
      self, mkn_and_blk, quantize_lhs, quantize_rhs
  ):
    (m, k, n), mkn_blk = mkn_and_blk

    key = jax.random.PRNGKey(0)
    lhs = jax.random.uniform(key, shape=(m, k))
    rhs = jax.random.uniform(key, shape=(k, n))

    @functools.partial(jax.jit, static_argnames=["block_size"])
    def q_matmul(lhs, rhs, block_size=(512, 512, 512)):
      assert lhs.shape[1] == rhs.shape[0]
      m, k = lhs.shape
      _, n = rhs.shape

      m_blk, k_blk, n_blk = block_size

      if quantize_lhs:
        lhs = aqt_pl.quant(lhs, n_bits=8, calibration_axes=(1,))
      if quantize_rhs:
        rhs = aqt_pl.quant(rhs, n_bits=8, calibration_axes=(0,))

      def kernel(
          lhs_ref: QTensor,
          rhs_ref: QTensor,
          out_ref: jnp.ndarray,
      ):
        @pl.when(pl.program_id(axis=2) == 0)
        def _():
          out_ref[...] = jnp.zeros_like(out_ref)

        if isinstance(lhs_ref, QTensor):
          lhs = aqt_pl.load_qtensor(lhs_ref)
        else:
          lhs = lhs_ref[...]

        if isinstance(rhs_ref, QTensor):
          rhs = aqt_pl.load_qtensor(rhs_ref)
        else:
          rhs = rhs_ref[...]

        out = aqt_pl.dot_general(
            lhs,
            rhs,
            dimension_numbers=(((1,), (0,)), ((), ())),
        )
        out_ref[...] += out

      lhs_inspec = pl.BlockSpec((m_blk, k_blk), lambda i, j, k: (i, k))
      rhs_inspec = pl.BlockSpec((k_blk, n_blk), lambda i, j, k: (k, j))
      out_spec = pl.BlockSpec((m_blk, n_blk), lambda i, j, k: (i, j))

      out = aqt_pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
          grid=(
              math.ceil(m / m_blk),
              math.ceil(n // n_blk),
              math.ceil(k // k_blk),
          ),
          in_specs=[
              lhs_inspec,
              rhs_inspec,
          ],
          out_specs=out_spec,
          interpret=False,
      )(lhs, rhs)
      return out

    quantized_out = q_matmul(lhs, rhs, block_size=mkn_blk)
    reference_out = lhs @ rhs

    difference = jnp.abs(quantized_out - reference_out)
    relative_error = difference / jnp.abs(reference_out)
    max_relative_error = relative_error.max().item()

    if quantize_lhs or quantize_rhs:
      self.assertLess(max_relative_error, 1e-2)
    else:
      # If none of tensor are quantized, the result should exactly identical
      # or the error should be very small.
      self.assertLess(max_relative_error, 1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name="OUTPUT_lhs_OUTPUT_rhs",
          lhs_calibration_axes=(1,),
          rhs_calibration_axes=(0,),
          dequant_mode_lhs=DequantMode.OUTPUT,
          dequant_mode_rhs=DequantMode.OUTPUT,
      ),
      dict(
          testcase_name="THIS_INPUT_lhs_OUTPUT_rhs",
          lhs_calibration_axes=(1,),
          rhs_calibration_axes=(0,),
          dequant_mode_lhs=DequantMode.THIS_INPUT,
          dequant_mode_rhs=DequantMode.OUTPUT,
      ),
      dict(
          testcase_name="OUTPUT_lhs_THIS_INPUT_rhs",
          lhs_calibration_axes=(1,),
          rhs_calibration_axes=(0,),
          dequant_mode_lhs=DequantMode.OUTPUT,
          dequant_mode_rhs=DequantMode.THIS_INPUT,
      ),
      dict(
          testcase_name="OTHER_INPUT_lhs",
          lhs_calibration_axes=(0,),
          rhs_calibration_axes=None,
          dequant_mode_lhs=DequantMode.OTHER_INPUT,
      ),
      dict(
          testcase_name="OTHER_INPUT_rhs",
          lhs_calibration_axes=None,
          rhs_calibration_axes=(1,),
          dequant_mode_rhs=DequantMode.OTHER_INPUT,
      ),
  )
  def test_dequantization_location(
      self,
      lhs_shape=(2048, 512),
      rhs_shape=(512, 2048),
      lhs_calibration_axes=None,
      rhs_calibration_axes=None,
      dequant_mode_lhs=DequantMode.OUTPUT,
      dequant_mode_rhs=DequantMode.OUTPUT,
      dimension_numbers=(((1,), (0,)), ((), ())),
  ):
    lhs = jnp.zeros(lhs_shape)
    rhs = jnp.zeros(rhs_shape)

    if lhs_calibration_axes is not None:
      lhs_qtensor = aqt_pl.quant(
          lhs, n_bits=8, calibration_axes=lhs_calibration_axes
      )
    else:
      lhs_qtensor = lhs

    if rhs_calibration_axes is not None:
      rhs_qtensor = aqt_pl.quant(
          rhs, n_bits=8, calibration_axes=rhs_calibration_axes
      )
    else:
      rhs_qtensor = rhs

    # Calling this function implictly pass the tests.
    aqt_pl.dot_general(
        lhs_qtensor,
        rhs_qtensor,
        dimension_numbers=dimension_numbers,
        lhs_dequant_mode=dequant_mode_lhs,
        rhs_dequant_mode=dequant_mode_rhs,
    )


if __name__ == "__main__":
  absltest.main()
