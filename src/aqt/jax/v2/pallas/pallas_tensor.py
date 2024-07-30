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
"""QTensor states holder for Pallas."""

from __future__ import annotations

from typing import Generic, TypeVar
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import transpose
from aqt.jax.v2 import utils

import flax.struct
import jax  # pylint: disable=unused-import
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


T = TypeVar('T')
Array = jnp.ndarray
QTensor = aqt_tensor.QTensor

transpose = transpose.transpose


def _called_within_pallas_kernel(func):
  """Decorator to indicate that the function is called within Pallas kernel."""
  return func


@utils.flax_slots_kw_only_dataclass
class TransposedTensor(Generic[T]):
  """Transposed tensor."""

  transposed_tensor: T
  permute_axes: list[utils.AxisIdx] = flax.struct.field(
      default=None, pytree_node=False
  )

  @property
  @_called_within_pallas_kernel
  def untransposed(self):
    return transpose(
        self.transposed_tensor[...], list(np.argsort(self.permute_axes))
    )


def _count_less(list_, item):
  """Count the number of elements in list_ that are less than item."""
  return len([i for i in list_ if i < item])


def transpose_tensor_for_memory_saving(
    s: jax.Array, block_spec: pl.BlockSpec
) -> (
    tuple[jax.Array, pl.BlockSpec]
    | tuple[TransposedTensor[jax.Array], TransposedTensor[pl.BlockSpec]]
):
  """Transpose given tensor s for memory saving."""

  # If the size of the last dimension is less than 128, padding will be
  # added to make up the difference, which results in wasted memory space.
  # To prevent this, if the second minor most dimension is greater than 128,
  # a transpose between the most minor and second minor most dimensions
  # will be performed.
  transpose_last_two_axes = (s.shape[-1] == 1 and s.shape[-2] >= 128)
  if not transpose_last_two_axes:
    return s, block_spec

  permute_axes = list(range(s.ndim))
  permute_axes[-1], permute_axes[-2] = permute_axes[-2], permute_axes[-1]
  transposed_tensor = transpose(s, permute_axes)

  transposed_block_shape = tuple(
      block_spec.block_shape[i] for i in permute_axes
  )

  def transposed_index_map(
      *args,
      permute_axes=tuple(permute_axes),
  ):
    index = list(block_spec.index_map(*args))
    return tuple(index[i] for i in permute_axes)

  # if there are None in block_shape, it means that those axes are
  # reduced when fetched inside pallas kernel. Therefore, we should amend
  # permute_axes accordingly.
  dims_to_be_reduced = [
      i for i, size in enumerate(transposed_block_shape) if size is None
  ]
  permute_axes = [
      i - _count_less(dims_to_be_reduced, i)
      for i in permute_axes
      if i not in dims_to_be_reduced
  ]

  transposed_tensor = TransposedTensor(
      transposed_tensor=transposed_tensor, permute_axes=permute_axes
  )
  transposed_block_spec = transposed_tensor.replace(
      transposed_tensor=pl.BlockSpec(
          block_shape=transposed_block_shape,
          index_map=transposed_index_map,
      )
  )
  return transposed_tensor, transposed_block_spec


def make_qtensor_blockspec(
    qtensor: aqt_tensor.QTensor, block_spec: pl.BlockSpec
) -> QTensor:
  """Build a block spec for QTensor.

  Args:
    qtensor: a QTensor.
    block_spec: Block spec of the unquantized tensor.

  Returns:
    A QTensor of block spec.
  """

  assert qtensor.scale_t is None

  def _make_scale_block_spec(scale: jax.Array, block_spec: pl.BlockSpec):
    scale_blk_shape = list(block_spec.block_shape)

    # Find calibration axes, and change scale_block_shape accordingly.
    calibration_axes = [axes for axes, s in enumerate(scale.shape) if s == 1]

    for i in calibration_axes:
      if scale_blk_shape[i] is not None:
        scale_blk_shape[i] = 1

    def scale_index_map(
        *args,
        calibration_axes=calibration_axes,
    ):
      index = list(block_spec.index_map(*args))
      for i in calibration_axes:
        index[i] = 0
      return tuple(index)

    return pl.BlockSpec(
        block_shape=tuple(scale_blk_shape),
        index_map=scale_index_map,
    )

  return QTensor(
      qvalue=block_spec,
      scale=[_make_scale_block_spec(s, block_spec) for s in qtensor.scale],
      scale_t=None,
      dequant_dtype=qtensor.dequant_dtype,
  )
