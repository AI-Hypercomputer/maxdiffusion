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
"""Freezer for writing & storing general Flax structure."""

import enum
from typing import Any, Callable

from flax.core import meta as nn_meta
import flax.linen as nn


class FreezerMode(enum.Enum):
  NONE = 1
  WRITE = 2
  READ = 3


_FREEZE_VAR_NAME = 'frozen'


class Freezer(nn.Module):
  """Flax module that can freeze its input.

  On default it is an identity function that saves the input in a variable.
  In 'mode=FreezerMode.READ' mode, ignores the input and returns the frozen
  value. It is usefult to implement 'constant folding' and put quantized weights
  and scales in the checkpoint for serving. Specifically:

  self.get() returns None when freeze_mode=NONE or WRITE, returns variable when
  freeze_mode=READ.
  self.set() does nothing when freeze_mode=NONE or READ, creates and stores
  input value freeze_mode=CONVERT.
  """

  collection: str
  mode: FreezerMode
  axis_metadata_wrapper: Callable[..., nn_meta.AxisMetadata] | None = None

  @nn.compact
  def _get_or_set(self, inputs: Any, is_set: bool) -> Any | None:
    def initializer():
      if self.axis_metadata_wrapper is not None:
        return self.axis_metadata_wrapper(inputs)
      return inputs

    if is_set:
      match self.mode:
        case FreezerMode.NONE:
          pass
        case FreezerMode.WRITE:
          s = self.variable(self.collection, _FREEZE_VAR_NAME, initializer)
          s.value = inputs
          return None
        case FreezerMode.READ:
          # Set in READ mode works as an initializer for checkpoint reading.
          # we don't want to change the variable during the serving
          _ = self.variable(self.collection, _FREEZE_VAR_NAME, initializer)
          return None
        case _:
          # Nothing matched.
          assert False, 'Unknown quant mode.'
      return None
    else:
      match self.mode:
        case FreezerMode.NONE:
          return None
        case FreezerMode.WRITE:
          return None
        case FreezerMode.READ:
          if not self.has_variable(self.collection, _FREEZE_VAR_NAME):
            return None

          msg = 'Initialization should not happen in Get mode, but in Set mode.'

          def initializer_poison():
            assert False, msg

          return self.variable(
              self.collection, _FREEZE_VAR_NAME, initializer_poison
          ).value
        case _:
          # Nothing matched.
          assert False, 'Unknown quant mode.'

  def get(self) -> Any | None:
    return self._get_or_set(None, is_set=False)

  def set(self, inputs: Any) -> Any | None:
    return self._get_or_set(inputs, is_set=True)
