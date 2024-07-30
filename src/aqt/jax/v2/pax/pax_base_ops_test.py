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

"""Test for AQT pax."""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.pax import pax_base_ops
import jax
import jax.numpy as jnp


class PaxBaseOpsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(cfg=None),
      dict(cfg=config.config_v4()),
  )
  def test_einsum_is_quantized(self, cfg):
    prng_key = jax.random.PRNGKey(seed=123)
    rngs = {'params': prng_key, 'random': prng_key}
    lhs = jax.random.normal(prng_key, [10, 10, 10])
    rhs = jax.random.normal(prng_key, [10, 10])
    eqn = '...x,xy->y'

    layer = pax_base_ops.AqtEinsum(cfg=cfg, track_train_step=True)
    variable = layer.init(rngs, eqn, lhs, rhs)

    @jax.jit
    def apply_fn(lhs, rhs):
      return layer.apply(variable, eqn, lhs, rhs, mutable=True, rngs=rngs)

    @jax.jit
    @jax.value_and_grad
    def train_step(lhs, rhs):
      return jnp.sum(apply_fn(lhs, rhs)[0])

    # Run one value and grad to test if training will work
    train_step(lhs, rhs)

    out_float = jnp.einsum(eqn, lhs, rhs)
    out_quant, new_variable = apply_fn(lhs, rhs)
    if cfg is None:
      # Ensure results are the same
      self.assertEqual(jnp.mean(jnp.square(out_float - out_quant)), 0)
    else:
      # Ensure quantized einsum does not produce the same result.
      self.assertGreater(jnp.mean(jnp.square(out_float - out_quant)), 0)

    self.assertEqual(new_variable['non_trainable']['train_step'], 1)


if __name__ == '__main__':
  absltest.main()
