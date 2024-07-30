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

"""Test for flax e2e model."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
from aqt.jax.v2.extensions.gptq.examples import gptq_flax_e2e_model
import jax
import jax.numpy as jnp


def _dummy_dataset(ds_size, image_rng, label_rng):
  return {
      "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
      "label": jax.random.randint(
          key=label_rng, shape=(ds_size,), minval=0, maxval=10
      ),
  }


class GptqTest(parameterized.TestCase):

  def test_gptq(self):
    aqt_cfg = config.config_v4()

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng, label_rng = jax.random.split(rng, 3)
    rng, input_rng = jax.random.split(rng)
    rng, calibration_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 64
    batch_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)

    # Stage 1: regular training
    state = gptq_flax_e2e_model.create_train_state(init_rng, aqt_cfg)

    state, _, _ = gptq_flax_e2e_model.train_epoch(
        state, ds, batch_size, rng=input_rng
    )

    # Stage 2: Calibration.
    gptq_flax_e2e_model.update_cfg_with_gptq(state.cnn_train.aqt_cfg)
    gptq_flax_e2e_model.update_cfg_with_gptq(state.cnn_eval.aqt_cfg)

    calibrate_f, model_calibrate = gptq_flax_e2e_model.calibration_conversion(
        state
    )

    calibration_steps = 4
    calibrated_params = gptq_flax_e2e_model.calibrate_epoch(
        calibrate_f,
        model_calibrate,
        ds,
        batch_size,
        rng=calibration_rng,
        calibration_steps=calibration_steps,
    )
    calibration_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), calibrated_params
    )
    dtype = jnp.dtype
    expected_gptq_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "GptqHinvCollector_0": {
                    "collected_hinv": (dtype("float32"), (5, 5)),
                    "num_calibrated_batches": (dtype("int32"), ()),
                }
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "GptqHinvCollector_0": {
                    "collected_hinv": (dtype("float32"), (1568, 1568)),
                    "num_calibrated_batches": (dtype("int32"), ()),
                }
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "GptqHinvCollector_0": {
                    "collected_hinv": (dtype("float32"), (128, 128)),
                    "num_calibrated_batches": (dtype("int32"), ()),
                }
            }
        },
    }

    utils.test_pprint_eq(expected_gptq_pytree, calibration_pytree["gptq"])

    # The count number should be equal to the number of calibration.
    einsum_params = calibrated_params["gptq"]["AqtEinsum_0"]["AqtDotGeneral_0"]
    gptq_collected = einsum_params["GptqHinvCollector_0"]
    collected_count = gptq_collected["num_calibrated_batches"]
    self.assertEqual(calibration_steps, collected_count)

    # Stage 3. Convert the calibrated checkpoint.
    state = state.replace(model=copy.deepcopy(calibrated_params))
    _, model_serving = gptq_flax_e2e_model.serving_conversion(state)
    dtype = jnp.dtype
    expected_dtype = dtype("int8")
    expected_aqt_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 5, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 1568, 256)),
                        scale=[(dtype("float32"), (2, 1, 256))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                    )
                }
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 128, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                    )
                }
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )

    utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

    # Since the GPTQ changes the weights to better quantize it, the logits
    # before and after the conversion should be different.
    # As a conclusion, we do not put the logits comparison tests here.


if __name__ == "__main__":
  absltest.main()
