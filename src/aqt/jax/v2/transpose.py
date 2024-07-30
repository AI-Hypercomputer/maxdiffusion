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
"""Transpose functions."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

from typing import Sequence
from aqt.jax.v2 import utils
import jax
import jax.numpy as jnp


def transpose(t: jax.Array, axes: list[utils.AxisIdx]):
  """Transpose tensor through reshape if possible.

  This is a workaround to use AQT inside pallas kernel. Pallas support
  transposing only 2D tensor. However, tensor with a specific pattern can be
  transposed through reshaping. This pattern is observed a lot in scale factor
  of QTensor.

  If axes with dimensions larger than one are ordered in ascending order, then
  the tensor can be transposed by reshaping. For example,

  Let's say the shape of tensor `x` is [10, 1, 1]. Then, there are only one
  axis with dimension larger than one. The tensor can be transposed by
  reshaping no matter what the order of permutation, e.g., `jnp.transpose(x, [1,
  2, 0]) == jnp.reshape(x, [x.shape[i] for i in [1, 2, 0]])`.

  Another example is `x` with shape [10, 1, 30]. Then, there are two axes with
  dimension larger than one. The tensor can be transposed by reshaping if
  tensor is permuted while maintaining the order of axes with dimension larger
  than one, i.e., the first axes remains in front of third axis after
  permutation, e.g.,  `jnp.transpose(x, [0, 2, 1]) == jnp.reshape(x, [x.shape[i]
  for i in [0, 2, 1]])`.

  Args:
    t: input tensor.
    axes: transpose axes.

  Returns:
    A transposed tensor.

  Example:
    >>> x = np.random.randn([10, 1, 1]) # x is transposable by reshaping.
    >>> axes = [2, 1, 0]
    >>> y = jnp.transpose(x, axes)
    >>> (jnp.reshape(x, y.shape) == y).all()
    True
    >>> (transpose(x, axes) == y).all()

    >>> x = np.random.randn([10, 20, 30]) # x is not transposable by reshaping.
    >>> axes = [2, 1, 0]
    >>> y = jnp.transpose(x, axes)
    >>> (jnp.reshape(x, y.shape) == y).all()
    False
    >>> (transpose(x, axes) == y).all()
    True
  """

  # To check if a tensor's shape is not polymorphic, validate that the shape
  # consists of only integers.
  if all(map(lambda i: isinstance(i, int), t.shape)):
    axes_bigger_than_one = [i for i, size in enumerate(t.shape) if size > 1]
    permutation_axes_bigger_than_one = [
        axes.index(i) for i in axes_bigger_than_one
    ]
    # Check if axes bigger than one are ordered in ascending order.
    reshapable = (
        sorted(permutation_axes_bigger_than_one)
        == permutation_axes_bigger_than_one
    )
  else:
    reshapable = False

  if reshapable:
    t = jnp.reshape(t, (t.shape[i] for i in axes))
  else:
    t = jax.lax.transpose(t, axes)
  return t


def _scale_trans(x, ca, ba):
  """Transposes x to output dimension order."""
  ca = list(ca)
  ba = list(ba)
  for i in ca:
    assert x.shape[i] == 1
  ra = utils.get_remaining_axes(x.ndim, ca, ba)
  x = transpose(x, ba + ra + ca)
  # TODO(lew): x = jnp.squeeze(x, axis=range(len(ba+ra): len(x.shape))
  shape_ba = x.shape[: len(ba)]
  shape_ra = x.shape[len(ba) : len(x.shape) - len(ca)]
  # Will need to add additional axes (size 1) for the other shape_ra
  x = x.reshape(shape_ba + shape_ra)
  return x


def lhs_scale_transpose_to_output(
    lhs_scale, dimension_numbers, lhs_shape, rhs_shape
):
  """Transposes lhs_scale to output dimension order."""
  if lhs_scale is None:
    return None
  # The axis order in out is as follows: batch, lhs_ra, rhs_ra
  # - batch axes order is uniquely determined by either lhs_ba or rhs_ba
  # - contraction axes ca disappear from the output
  # - order of the remaining axes (ra) is preserved.
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qlhs_scale_t = _scale_trans(lhs_scale, lhs_ca, lhs_ba)
  # inserting dummy axes for rhs_ra
  assert len(qlhs_scale_t.shape) == len(lhs_shape) - len(lhs_ca)
  start = len(qlhs_scale_t.shape)
  end = len(rhs_shape) - len(rhs_ca) - len(rhs_ba) + start
  lhs_dummy_axes = range(start, end)
  qlhs_scale_t = jnp.expand_dims(qlhs_scale_t, axis=lhs_dummy_axes)
  return qlhs_scale_t


def rhs_scale_transpose_to_output(
    rhs_scale, dimension_numbers, lhs_shape, rhs_shape
):
  """Transposes rhs_scale to output dimension order."""
  if rhs_scale is None:
    return None
  del rhs_shape
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qrhs_scale_t = _scale_trans(rhs_scale, rhs_ca, rhs_ba)
  start = len(rhs_ba)
  end = len(lhs_shape) - len(lhs_ca) - len(lhs_ba) + start
  rhs_dummy_axes = range(start, end)
  qrhs_scale_t = jnp.expand_dims(qrhs_scale_t, axis=rhs_dummy_axes)
  return qrhs_scale_t


def _scale_trans_back(
    scale_t: jax.Array, ca: Sequence[utils.AxisIdx], ba: Sequence[utils.AxisIdx]
) -> jax.Array:
  """Transposes scale (transposed for output) back to its original dimension.

  Args:
    scale_t: scale transposed for output, without other arguments' remaining
      axis dimensions. Output of _scale_trans.
    ca: contracting axis.
    ba: batching axis.

  Returns:
    Recovered scale from the scale_t.
  """
  ca, ba = list(ca), list(ba)

  start = len(scale_t.shape)
  end = start + len(ca)
  scale = jnp.expand_dims(scale_t, axis=range(start, end))

  ra = utils.get_remaining_axes(scale.ndim, ca, ba)

  transpose_back = [-1] * len(ba + ra + ca)
  for axis_orig, axis_transposed in enumerate(ba + ra + ca):
    transpose_back[axis_transposed] = axis_orig

  assert -1 not in transpose_back

  scale = transpose(scale, transpose_back)
  return scale


def lhs_recover_scale_from_scale_t(
    lhs_scale_t: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
):
  """Recovers lhs_scale from lhs_scale_t."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  # Remove dummy axes.
  start = len(lhs_shape) - len(lhs_ca)
  rhs_ra_ndim = len(rhs_shape) - len(rhs_ca) - len(rhs_ba)
  lhs_scale = jnp.squeeze(lhs_scale_t, axis=range(start, start + rhs_ra_ndim))

  return _scale_trans_back(lhs_scale, lhs_ca, lhs_ba)


def rhs_recover_scale_from_scale_t(
    rhs_scale_t: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
):
  """Recovers rhs_scale from rhs_scale_t."""
  del rhs_shape

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  start = len(rhs_ba)
  end = len(lhs_shape) - len(lhs_ca) - len(lhs_ba) + start
  rhs_scale = jnp.squeeze(rhs_scale_t, axis=range(start, end))

  return _scale_trans_back(rhs_scale, rhs_ca, rhs_ba)


def _scale_trans_for_other_input(
    x: jax.Array,
    my_ca: Sequence[utils.AxisIdx],
    my_ba: Sequence[utils.AxisIdx],
    other_ca: Sequence[utils.AxisIdx],
    other_ba: Sequence[utils.AxisIdx],
    other_rank: int,
):
  """Transposes x to other inputs' dimension order."""
  my_ca = list(my_ca)
  my_ba = list(my_ba)
  other_ca = list(other_ca)
  other_ba = list(other_ba)

  # Match the rank.
  if len(x.shape) < other_rank:
    x = x.reshape(list(x.shape) + [1] * (other_rank - len(x.shape)))

  transpose_dim = [-1] * len(x.shape)
  my_axis_mapped = my_ca + my_ba
  other_axis_mapped = other_ca + other_ba
  my_ra = utils.get_remaining_axes(x.ndim, my_ca, my_ba)

  for axis in my_ra:
    assert x.shape[axis] == 1
  for my_axis, other_axis in zip(my_axis_mapped, other_axis_mapped):
    transpose_dim[other_axis] = my_axis

  # Fill unrelated axis with remaining axis.
  ra_idx = 0
  for transpose_dim_idx, transpose_dim_value in enumerate(transpose_dim):
    if transpose_dim_value == -1:
      transpose_dim[transpose_dim_idx] = my_ra[ra_idx]
      ra_idx += 1
  assert ra_idx == len(my_ra)

  # Transpose.
  x = transpose(x, transpose_dim)

  # Remove redundant axis.
  if len(x.shape) > other_rank:
    for idx in range(len(x.shape), other_rank):
      assert x.shape[idx] == 1
    x = x.reshape(x.shape[:other_rank])

  return x


def lhs_scale_transpose_for_rhs_input(lhs_scale, dimension_numbers, rhs_shape):
  """Transposes lhs_scale to rhs input dimension order."""
  if lhs_scale is None:
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  return _scale_trans_for_other_input(
      lhs_scale, lhs_ca, lhs_ba, rhs_ca, rhs_ba, len(rhs_shape)
  )


def rhs_scale_transpose_for_lhs_input(rhs_scale, dimension_numbers, lhs_shape):
  """Transposes lhs_scale to rhs input dimension order."""
  if rhs_scale is None:
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  return _scale_trans_for_other_input(
      rhs_scale, rhs_ca, rhs_ba, lhs_ca, lhs_ba, len(lhs_shape)
  )
