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
"""Base abstract class for all numerics."""
import abc


class AqtNumerics(abc.ABC):
  """Numerics for int8, int4, binary, etc."""

  # TODO(lew): Currently this is a part of API, only because it is used to set
  # it in test. Remove and leave only get_dtype(

  @abc.abstractmethod
  def get_dtype(self):
    pass

  @abc.abstractmethod
  def abs_val_mapped_to(self):
    """The value returned is the end of quantization range.

    It could be biggest value that can be represented by numerical format
    exactly. E.g. in case of int8, 127 . Or it could be edge of the last bucket.
    Edge in case of int8, 127.5
    """
    pass

  @abc.abstractmethod
  def vjp_fwd(self, x, context):
    pass

  @abc.abstractmethod
  def vjp_bwd(self, res, grad):
    pass
