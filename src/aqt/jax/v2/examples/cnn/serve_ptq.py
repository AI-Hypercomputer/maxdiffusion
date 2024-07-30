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

"""A sample to show how to quantize a model without calibration using AQT."""


from collections.abc import Sequence

from absl import app
from absl import flags
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.examples.cnn import aqt_utils
from aqt.jax.v2.examples.cnn import cnn_model
from aqt.jax.v2.examples.cnn import model_utils
import jax

_WEIGHT_ONLY = flags.DEFINE_bool(
    'weight_only', False, 'Perform weight-only quantization.'
)


def run(train_ds: dict[str, jax.Array], test_ds: dict[str, jax.Array]) -> None:
  """Extracted main function for unit testing."""
  # 1. Train
  state = model_utils.create_train_state(
      jax.random.key(0), cnn_model.CNN(True), cnn_model.CNN(False)
  )
  state = model_utils.train_and_evaluate(
      num_epochs=1,
      workdir='/tmp/aqt_mnist_example',
      train_ds=train_ds,
      test_ds=test_ds,
      state=state,
  )

  # 2. Quantize and test
  aqt_cfg = aqt_config.default_unquantized_config()
  aqt_config.set_bits(
      aqt_cfg,
      fwd_lhs_bit=None if _WEIGHT_ONLY.value else 8,
      fwd_rhs_bit=8,
      dlhs_lhs_bit=None,
      dlhs_rhs_bit=None,
      drhs_lhs_bit=None,
      drhs_rhs_bit=None,
  )
  loss = aqt_utils.serve_quantized(
      cnn_model.CNN, test_ds, aqt_cfg, state.model_vars, act_calibrated=False
  )
  print('serve loss on sample ds: {}'.format(loss))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_ds, test_ds = cnn_model.get_datasets()
  run(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)
