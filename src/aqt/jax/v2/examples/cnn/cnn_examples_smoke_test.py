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

"""Unit tests for AQT examples."""

from collections.abc import Callable

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2.examples.cnn import serve_calibrated_ptq
from aqt.jax.v2.examples.cnn import serve_ptq
from aqt.jax.v2.examples.cnn import serve_unquantized
import jax


def _dummy_dataset(
    ds_size: int, image_rng: jax.Array, label_rng: jax.Array
) -> dict[str, jax.Array]:
  return {
      "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
      "label": jax.random.randint(
          key=label_rng, shape=(ds_size,), minval=0, maxval=10
      ),
  }


class CnnExamplesSmokeTest(parameterized.TestCase):

  @parameterized.parameters([
      serve_calibrated_ptq.run,
      serve_ptq.run,
      serve_unquantized.run,
  ])
  def test_run_function(
      self,
      test_func: Callable[[dict[str, jax.Array], dict[str, jax.Array]], None],
  ):
    rng = jax.random.key(0)
    rng, train_image_rng = jax.random.split(rng)
    rng, train_label_rng = jax.random.split(rng)
    rng, test_image_rng = jax.random.split(rng)
    _, test_label_rng = jax.random.split(rng)
    train_ds = _dummy_dataset(
        ds_size=100,
        image_rng=train_image_rng,
        label_rng=train_label_rng,
    )
    test_ds = _dummy_dataset(
        ds_size=100,
        image_rng=test_image_rng,
        label_rng=test_label_rng,
    )
    # This is not a behavioral test. It only makes sure the AQT API usages in
    # the example code are compatible with the latest code base by ensuring
    # they run without errors.
    test_func(train_ds, test_ds)


if __name__ == "__main__":
  absltest.main()
