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

"""A sample to show how to calibrate and quantize a CNN model using AQT."""

from collections.abc import Sequence
import functools

from absl import app
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.examples.cnn import aqt_utils
from aqt.jax.v2.examples.cnn import cnn_model
from aqt.jax.v2.examples.cnn import model_utils
from aqt.jax.v2.flax import aqt_flax_calibration
import jax


def _set_static_range_calib_options(aqt_cfg: aqt_config.DotGeneral) -> None:
  sr_calibration_cls = functools.partial(
      aqt_flax_calibration.MeanOfAbsMaxCalibration, quant_collection='qc'
  )
  aqt_config.set_fwd_calibration(aqt_cfg, sr_calibration_cls)
  aqt_cfg.fwd.dg_quantizer.lhs.calib_shared_axes = 'per_tensor'


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

  # 2. Calibrate
  aqt_cfg = aqt_config.config_v4(fwd_bits=8)
  _set_static_range_calib_options(aqt_cfg)
  state = aqt_utils.calibrate(cnn_model.CNN, aqt_cfg, state, 10, train_ds)

  # 3. Quantize and test
  loss = aqt_utils.serve_quantized(
      cnn_model.CNN, test_ds, aqt_cfg, state.model_vars, act_calibrated=True
  )
  print('serve loss on sample ds: {}'.format(loss))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_ds, test_ds = cnn_model.get_datasets()
  run(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)
