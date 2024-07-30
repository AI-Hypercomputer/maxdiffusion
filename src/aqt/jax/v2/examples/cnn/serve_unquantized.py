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

"""A very simple non-quantized execution sample of a CNN model."""

from collections.abc import Sequence

from absl import app
from aqt.jax.v2.examples.cnn import cnn_model
from aqt.jax.v2.examples.cnn import model_utils
import jax


def run(train_ds: dict[str, jax.Array], test_ds: dict[str, jax.Array]) -> None:
  """Extracted main function for unit testing."""
  # 1. Train
  state = model_utils.create_train_state(
      rng=jax.random.key(0),
      train_model=cnn_model.CNN(True),
      eval_model=cnn_model.CNN(False),
  )
  state = model_utils.train_and_evaluate(
      num_epochs=1,
      workdir='/tmp/aqt_mnist_example',
      train_ds=train_ds,
      test_ds=test_ds,
      state=state,
  )

  # 2. Serve and test
  loss = model_utils.serve(
      serve_model=cnn_model.CNN(False),
      model_vars=state.model_vars,
      test_ds=test_ds,
  )
  print('serve loss on sample ds: {}'.format(loss))


def main(argv: Sequence[str]) -> None:
  del argv

  train_ds, test_ds = cnn_model.get_datasets()
  run(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)
