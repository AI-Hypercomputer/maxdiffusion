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
"""utils contains all kinds of small pieces of code used throughout AQT.

Code in this file can't depend on any other AQT file.
However, it is acceptable to grow pieces of funcionality in this file and later
promote them to dedicated files.
"""

import dataclasses
import difflib
import enum
import functools
import pprint
import re
from typing import Any, Sequence
import flax.struct
import jax
from jax import numpy as jnp


# Some common types.
AxisIdx = int
AxisSize = int

# None means that the template matches any axis size
ShapeTemplate = Sequence[int | None]

# TODO(lew): We should get a better type for jax.lax.dot_general and variants.
DotGeneralT = Any


def assert_shape(shape: Sequence[int], shape_template: ShapeTemplate, msg: str):
  shape = list(shape)
  shape_template = list(shape_template)
  assert len(shape) == len(shape_template), msg
  for axis_size, expected_size in zip(shape, shape_template):
    assert (
        expected_size is None or axis_size == expected_size
    ), f'{msg}: {shape} does not match {shape_template}'


def assert_eq(value: Any, expected: Any, value_name: str):
  msg = f'{value_name}={value}, {expected=}'
  assert value == expected, msg


flax_slots_dataclass = functools.partial(
    flax.struct.dataclass, frozen=False, slots=True
)

# TODO: b/332418233 - Consolidate flax_slots_kw_only_dataclass and
# flax_slots_dataclass into a single decorator once the migration is
# done. This will exist only temporarily while completing the
# migration.
flax_slots_kw_only_dataclass = functools.partial(
    flax_slots_dataclass, kw_only=True
)


def dataclass_field(default='no default value'):
  # We can't use None as the default's default,
  # because someone actually might want to have None as default of the field.
  if default == 'no default value':
    return dataclasses.field()
  else:
    return dataclasses.field(default_factory=default)


class QuantMode(enum.Enum):
  TRAIN = 1
  CALIBRATE = 2
  CONVERT = 3
  SERVE = 4


def static_field(**kwargs):
  return flax.struct.field(pytree_node=False, **kwargs)


def dynamic_field(**kwargs):
  return flax.struct.field(pytree_node=True, **kwargs)


def print_diff(str_a: str, str_b: str):
  diff_generator = difflib.context_diff(str_a.split(' '), str_b.split(' '))
  print('Diff:')
  for diff in diff_generator:
    print(diff)
  print(f'first string (actual):\n{str_a}')


def test_pprint_eq(
    input_a: Any, input_b: Any, remove_memory_addresses: bool = False
):
  str_input_a = input_a if isinstance(input_a, str) else pprint.pformat(input_a)
  str_input_b = input_b if isinstance(input_b, str) else pprint.pformat(input_b)
  if remove_memory_addresses:
    str_input_a = re.sub(r' at 0x.*>', '>', str_input_a, 0, re.MULTILINE)
    str_input_b = re.sub(r' at 0x.*>', '>', str_input_b, 0, re.MULTILINE)
  assert str_input_a == str_input_b, print_diff(str_input_a, str_input_b)


def infer_dtype_from_bits(bits: int) -> jnp.dtype | None:
  """Get the dtype for the number of bits provided.

  Args:
    bits: number of bits for the dtype.

  Returns:
    The corresponding container dtype for the number of bits provided.
  """
  if bits == 4:
    return jnp.int4
  else:
    if bits <= 8 and bits >= 2:
      return jnp.int8
    else:
      return None


def get_remaining_axes(
    rank: int,
    contraction_axes: Sequence[AxisIdx],
    batch_axes: Sequence[AxisIdx],
):
  """Returns the remaining axes."""
  ret = []
  for i in range(rank):
    if i not in list(contraction_axes) + list(batch_axes):
      ret.append(i)
  return ret


@flax_slots_dataclass
class Context:
  key: jax.Array | None = dynamic_field()
  train_step: int | None = dynamic_field()
  quant_mode: QuantMode = static_field(default=QuantMode.TRAIN)
