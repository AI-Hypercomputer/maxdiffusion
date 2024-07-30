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

"""Injectable AQT ops.

Pax implements vanilla op wrappers in praxis/layers/base_ops.py
This file contains AQT variants of these.
"""

import functools
from aqt.jax.v2 import tiled_dot_general
import aqt.jax.v2.aqt_dot_general as aqt
import aqt.jax.v2.config as aqt_config
import jax
import jax.numpy as jnp
from praxis import base_layer


class AqtEinsum(base_layer.BaseLayer):
  """Quantized Einsum class for model injection."""

  cfg: aqt_config.DotGeneral | None = None
  # Einsum can switch the argument order before they are passed to dot_general
  # tiling_cfg is a config for the underlying dot_general, not for the einsum.
  # TODO(lew): Port the switch logic from flax/ to pax/
  tiling_cfg: tiled_dot_general.Cfg | None = None
  track_train_step: bool = False

  def setup(self) -> None:
    if self.track_train_step:
      self.create_variable(
          'train_step',
          base_layer.WeightHParams(
              shape=[],
              init=base_layer.WeightInit.Constant(0),
              dtype=jnp.int32,
          ),
          trainable=False,
      )

  def __call__(self, eqn, lhs, rhs):
    if self.track_train_step:
      if not self.do_eval:
        train_step = self.get_var('train_step')
        self.update_var('train_step', train_step + 1)
      # train_step starts from 0 and ends at exactly the total_train_step-1
      train_step = self.get_var('train_step')
    else:
      train_step = None
    dg = self.cfg
    if dg is not None:
      key = self.next_prng_key()
      dg = aqt_config.set_context(dg, key, train_step)
    else:
      dg = jax.lax.dot_general
    if self.tiling_cfg is not None:
      dg = functools.partial(
          tiled_dot_general.tiled_dot_general,
          self.tiling_cfg,
          dot_general=dg,
      )
    # jnp.einsum is by default jitted, which makes the key storing in cfg
    # cross the jit boundary. We need to call a non-jitted jnp.einsum below
    return aqt.einsum(eqn, lhs, rhs, dg)
