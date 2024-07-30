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
"""dot_general implementation for Pallas."""

from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
import jax
import jax.numpy as jnp

QTensor = aqt_tensor.QTensor
DequantMode = aqt_dot_general.DequantMode


# TODO(wppark): Match signature of load_qtensor with pl.load to support
# partial loading of QTensor.
def load_qtensor(qtensor: QTensor) -> QTensor:
  """Materialize QTensor of MemoryRef of pallas into QTensor of jax.Array."""
  qvalue = qtensor.qvalue
  scale = qtensor.scale
  scale_t = qtensor.scale_t

  if qvalue is not None:
    qvalue = qvalue[...]
  if scale is not None:
    scale = [s[...] for s in scale]
  if scale_t is not None:
    scale_t = [st[...] for st in scale_t]

  return qtensor.replace(qvalue=qvalue, scale=scale, scale_t=scale_t)


def _dtype_to_bits(dtype) -> int | None:
  if dtype in [jnp.bfloat16, jnp.float32]:
    return None
  if dtype == jnp.int8:
    return 8
  elif dtype == jnp.int4:
    return 4
  else:
    raise ValueError(
        'dtype must be one one of {jnp.bfloat16, jnp.float32, jnp.int8,'
        ' jnp.int4}'
    )


def dot_general(
    lhs: QTensor | jax.Array,
    rhs: QTensor | jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision=None,
    preferred_element_type=None,
    lhs_dequant_mode: DequantMode = DequantMode.OUTPUT,
    rhs_dequant_mode: DequantMode = DequantMode.OUTPUT,
) -> jax.Array:
  """lax.dot_general replacement for pallas which always returns dequantized output.

  Args:
    lhs: left hand side tensor.
    rhs: right hand side tensor.
    dimension_numbers: dimension numbers for dot_general.
    precision: this is ignored, but added to match the signature of
      jax.lax.dot_general.
    preferred_element_type: preferred output element type after dequantization.
    lhs_dequant_mode: This decicdes where lhs is dequantized. Default is OUTPUT
      where dequantization is applied to the output of dot_general. OTHER_INPUT
      applies dequantization to the other input before dot_general and
      THIS_INPUT applies dequantization to the current input before dot_general.
    rhs_dequant_mode: This decicdes where rhs is dequantized. Default is OUTPUT.

  Returns:
    Dequantized output of dot_general.
  """
  # precision is not used.
  del precision

  # The code below is supposed to be executed inside pallas kernel.

  # When only one of operands is quantized, Jax impliclty cast int8 into float
  # and performs dot_general. However, pallas requires explict casting when only
  # one of operands is quantized.
  is_both_quantized = isinstance(lhs, QTensor) and isinstance(rhs, QTensor)
  if isinstance(lhs, QTensor) and not is_both_quantized:
    promoted_dtype = jnp.promote_types(lhs.dequant_dtype, rhs)
    lhs = lhs.replace(qvalue=lhs.qvalue.astype(promoted_dtype))
  if isinstance(rhs, QTensor) and not is_both_quantized:
    promoted_dtype = jnp.promote_types(rhs.dequant_dtype, lhs)
    rhs = rhs.replace(qvalue=rhs.qvalue.astype(promoted_dtype))

  if isinstance(lhs, jax.Array):
    lhs = QTensor(
        qvalue=lhs, scale=[], scale_t=None, dequant_dtype=lhs.dtype
    )
  if isinstance(rhs, jax.Array):
    rhs = QTensor(
        qvalue=rhs, scale=[], scale_t=None, dequant_dtype=rhs.dtype
    )

  if preferred_element_type is None:
    preferred_element_type = jnp.promote_types(
        lhs.dequant_dtype, rhs.dequant_dtype
    )

  lhs_bits = _dtype_to_bits(lhs.qvalue.dtype)
  rhs_bits = _dtype_to_bits(rhs.qvalue.dtype)
  cfg = aqt_dot_general.dot_general_raw_make(
      lhs_bits=lhs_bits, rhs_bits=rhs_bits
  )

  cfg.lhs.dequant_mode = lhs_dequant_mode
  cfg.rhs.dequant_mode = rhs_dequant_mode
  # Accumulator dtype does not need to be modified. Unless both inputs are
  # dequantized on output.
  if (
      cfg.lhs.dequant_mode != DequantMode.OUTPUT
      or cfg.rhs.dequant_mode != DequantMode.OUTPUT
  ):
    cfg.dg_accumulator_dtype = None

  out_qtensor = aqt_dot_general._qtensor_dot_general(  # pylint: disable=protected-access
      lhs,
      rhs,
      dimension_numbers,
      cfg,
      dequant_dtype=preferred_element_type,
  )
  return out_qtensor.dequant()
