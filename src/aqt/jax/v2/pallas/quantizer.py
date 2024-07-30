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
"""AQT Quantizr API."""

# TODO(wppark): Remove this file. This is a temporary module before the
# official release of AQT quant / dequant API.

from typing import Sequence

from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import aqt_tensor

import jax
import jax.numpy as jnp

QTensor = aqt_tensor.QTensor


def quant(
    x: jax.Array,
    n_bits: int,
    calibration_axes: Sequence[int],
) -> QTensor:
  """Apply channel-wise quantization to x.

  x is quantized channel-wisely on the calibration axes.

  Args:
    x: input tensor
    n_bits: the precision for quantization.
    calibration_axes: the calibration axes.

  Returns:
    A quantized QTensor
  """
  quantizer = aqt_quantizer.quantizer_make(n_bits)
  # jax.lax.stop_gradient is not supported in pallas, thus disable
  # scale_stop_grad in the quantizer.
  quantizer.scale_stop_grad = False
  qx, _ = quantizer.quant(x, calibration_axes=calibration_axes)
  # VPU ops only support float32. jax implicitly converts tensor into float32.
  # However, pallas requires explicit casting. Therefore, we need to enforce
  # the dequant dtype to be float32.
  qx.dequant_dtype = jnp.float32  # enforce dequant dtype to be float32
  return qx
