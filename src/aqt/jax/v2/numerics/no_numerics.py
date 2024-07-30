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
"""Identity numerics for passing through input as-is."""

from typing import Any, Optional
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics


@utils.flax_slots_kw_only_dataclass
class NoNumerics(numerics.AqtNumerics):
  """No quantization, use a native type such as bf16."""

  # TODO(lew): This is a workaround. We should separate Stochastic Rounding.
  # noise_fn has no effect in NoNumerics.
  noise_fn: Optional[stochastic_rounding.NoiseFn] = None
  dtype: Optional[Any] = None

  # TODO(lew): This is a hack. We treat check isinstance(NoNumerics) and treat
  # it in a special way right now. These functions are never called
  def get_dtype(self):
    return None

  def abs_val_mapped_to(self):
    pass

  def vjp_fwd(self, x, context):
    pass

  def vjp_bwd(self, res, grad):
    pass
