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
"""Mnist example."""

from absl import app
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.examples import flax_e2e_model_v0
from aqt.jax.v2.extensions.gptq import gptq_dot_general_quantizer
from aqt.jax.v2.numerics import int_numerics

create_train_state = flax_e2e_model_v0.create_train_state
train_and_evaluate = flax_e2e_model_v0.train_and_evaluate
train_epoch = flax_e2e_model_v0.train_epoch
calibrate = flax_e2e_model_v0.calibrate
calibration_conversion = flax_e2e_model_v0.calibration_conversion
calibrate_epoch = flax_e2e_model_v0.calibrate_epoch
serve = flax_e2e_model_v0.serve
serving_conversion = flax_e2e_model_v0.serving_conversion


def update_cfg_with_gptq(aqt_cfg: aqt_dot_general.DotGeneral) -> None:
  """Updates aqt_cfg for GPTQ.

  Args:
    aqt_cfg: aqt_dot_general configuration.
  """

  assert isinstance(
      aqt_cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      aqt_cfg.fwd.dg_quantizer.lhs.numerics, int_numerics.IntNumerics
  )
  assert isinstance(
      aqt_cfg.fwd.dg_quantizer.rhs.numerics, int_numerics.IntNumerics
  )
  lhs_bits = aqt_cfg.fwd.dg_quantizer.lhs.numerics.bits
  rhs_bits = aqt_cfg.fwd.dg_quantizer.rhs.numerics.bits
  lhs = aqt_quantizer.quantizer_make(lhs_bits, initialize_calibration=False)
  rhs = aqt_quantizer.quantizer_make(rhs_bits, initialize_calibration=False)
  gptq_dg_quantizer = gptq_dot_general_quantizer.GptqDotGeneralQuantizer(
      lhs=lhs, rhs=rhs, sharding_axes=None, quant_collection='gptq'
  )

  aqt_cfg.fwd.dg_quantizer = gptq_dg_quantizer


def main(argv):
  del argv

  # 1. TRAIN.
  aqt_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
  state = train_and_evaluate(
      num_epochs=1, workdir='/tmp/aqt_mnist_example', aqt_cfg=aqt_cfg
  )

  # 2. Apply GPTQ Calibration. (Hinv collection).
  update_cfg_with_gptq(state.cnn_train.aqt_cfg)
  update_cfg_with_gptq(state.cnn_eval.aqt_cfg)
  state = calibrate(state, calibration_steps=10)

  # 3. CONVERT & SERVE.
  loss = serve(state, weight_only=True)
  print('serve loss on sample ds: {}'.format(loss))


if __name__ == '__main__':
  app.run(main)
