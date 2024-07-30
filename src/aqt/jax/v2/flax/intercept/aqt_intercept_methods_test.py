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

"""Test for AQT flax intercept methods."""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax.intercept import aqt_intercept_methods
import flax.linen as nn
import jax
import numpy as np


class MlpBlock(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = nn.Dense(features=inputs.shape[-1] * 4)(inputs)
    x = nn.relu(x)
    x = nn.Dense(features=inputs.shape[-1])(x)
    return x


class NestedMlpBlock(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = MlpBlock()(inputs)
    x = nn.relu(x)
    x = MlpBlock()(x)
    return x


class MockDotGeneralGenerator(aqt_intercept_methods.DotGeneralGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.count = 0

  def __call__(self, *args, **kwargs):
    self.count += 1
    return super().__call__(*args, **kwargs)


class MockDotGeneralGeneratorByModule(
    aqt_intercept_methods.DotGeneralGeneratorByModule):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.count = 0

  def generate_by_module(self, module: nn.Module):
    if isinstance(module, nn.Dense):
      self.count += 1
    return None


class AqtInterceptMethodsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="dot_general_generator",
           model_cls=MlpBlock,
           dot_general_generator_cls=MockDotGeneralGenerator,
           expected_count=3),
      dict(testcase_name="dot_general_generator_by_module",
           model_cls=MlpBlock,
           dot_general_generator_cls=MockDotGeneralGeneratorByModule,
           expected_count=2),
      dict(testcase_name="dot_general_generator_nested",
           model_cls=NestedMlpBlock,
           dot_general_generator_cls=MockDotGeneralGenerator,
           expected_count=7),
      dict(testcase_name="dot_general_generator_by_module_nested",
           model_cls=NestedMlpBlock,
           dot_general_generator_cls=MockDotGeneralGeneratorByModule,
           expected_count=4))
  def test_intercept_methods_replace_dot_general_count(
      self, model_cls, dot_general_generator_cls, expected_count):
    np_seed = 0
    init_seed = 0
    eval_seed = 0
    mlp_block = model_cls()
    np.random.seed(np_seed)
    inputs = np.random.normal(size=(3, 4))
    mock_dot_general_generator = dot_general_generator_cls(
        dot_general=jax.lax.dot_general)
    with aqt_intercept_methods.intercept_methods_replace_dot_general(
        mock_dot_general_generator):
      model = mlp_block.init(jax.random.PRNGKey(init_seed), inputs)
      _ = mlp_block.apply(
          model, inputs, rngs={"params": jax.random.key(eval_seed)})

    self.assertEqual(mock_dot_general_generator.count, expected_count)

  def test_aqt_dot_general_generator(self):
    aqt_dot_general_generator = (
        aqt_intercept_methods.AqtDotGeneralGenerator(config.config_v4())
    )
    # Only support Dense.
    self.assertIsInstance(
        aqt_dot_general_generator.generate_by_module(nn.Dense(10)),
        aqt_flax.AqtDotGeneral)
    self.assertIsNone(
        aqt_dot_general_generator.generate_by_module(
            nn.Conv(10, (3, 3))))
    self.assertIsNone(
        aqt_dot_general_generator.generate_by_module(
            nn.Einsum((10, 10), "ab,bc->ac")))


if __name__ == "__main__":
  absltest.main()
