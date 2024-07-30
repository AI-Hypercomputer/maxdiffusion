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
import functools

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
from aqt.jax.v2.examples import flax_e2e_model
from aqt.jax.v2.flax import aqt_flax_calibration
import jax
import jax.numpy as jnp
import numpy as np


def _dummy_dataset(ds_size, image_rng, label_rng):
  return {
      "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
      "label": jax.random.randint(
          key=label_rng, shape=(ds_size,), minval=0, maxval=10
      ),
  }



# @parameterized.parameters([
#     (
#         {
#             "drhs_bits": 8,
#             "drhs_accumulator_dtype": jnp.int32,  # overwrite the default None
#         },
#         8,
#     ),
#     (
#         {
#             "fwd_bits": 4,
#             "fwd_accumulator_dtype": None,
#             "dlhs_accumulator_dtype": None,
#         },
#         4,
#     ),
# ])
def test_mnist_calibration(configs, bits):
  print('running mnist calib')
  aqt_cfg = config.config_v4(**configs)
  aqt_cfg_conv = config.conv_general_dilated_make(
      2, bits, bits, initialize_calibration=False
  )
  device_kind = jax.devices()[0].device_kind
  if device_kind == "cpu" and bits == 4:
    # Some 4-bit operations are not supported on cpu.
    # Omitting tests on cpu with 4-bits.
    return

  # RNGs
  rng = jax.random.key(0)
  rng, init_rng = jax.random.split(rng)
  rng, image_rng1, image_rng2 = jax.random.split(rng, 3)
  rng, label_rng1, label_rng2 = jax.random.split(rng, 3)
  rng, input_rng = jax.random.split(rng)
  rng, calibration_rng = jax.random.split(rng)
  del rng

  # Dataset
  ds_size = 64
  batch_size = 8
  ds = _dummy_dataset(ds_size, image_rng1, label_rng1)
  ds2 = _dummy_dataset(ds_size, image_rng2, label_rng2)

  # Stage 1: regular training
  state = flax_e2e_model.create_train_state(init_rng, aqt_cfg, aqt_cfg_conv)

  breakpoint()
  state, _, _ = flax_e2e_model.train_epoch(
      state, ds, batch_size, rng=input_rng
  )

  # Stage 2: Calibration.
  flax_e2e_model.update_cfg_with_calibration(state.cnn_train.aqt_cfg)
  flax_e2e_model.update_cfg_with_calibration(state.cnn_eval.aqt_cfg)
  flax_e2e_model.update_cfg_raw_with_calibration(state.cnn_eval.aqt_cfg_conv)
  calibrate_f, model_calibrate = flax_e2e_model.calibration_conversion(state)

  calibration_steps = 4
  calibrated_params = flax_e2e_model.calibrate_epoch(
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
  expected_calibration_pytree = {
      "AqtEinsum_0": {
          # For the Einsum case, lhs and rhs are swapped.
          "AqtDotGeneral_0": {
              "MeanOfAbsMaxCalibration_0": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1)),
              },
              "MeanOfAbsMaxCalibration_1": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (2, 1, 10)),
              },
          }
      },
      "Conv_0": {
          "AqtConvGeneralDilated_0": {
              "MeanOfAbsMaxCalibration_0": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1, 1))},
              "MeanOfAbsMaxCalibration_1": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1, 32))
                  }
              }
          },
      "Conv_1": {
          "AqtConvGeneralDilated_0": {
              "MeanOfAbsMaxCalibration_0": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1, 1))},
              "MeanOfAbsMaxCalibration_1": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1, 64))}}},
      "Dense_0": {
          "AqtDotGeneral_0": {
              "MeanOfAbsMaxCalibration_0": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1)),
              },
              "MeanOfAbsMaxCalibration_1": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (2, 1, 256)),
              },
          }
      },
      "Dense_1": {
          "AqtDotGeneral_0": {
              "MeanOfAbsMaxCalibration_0": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (1, 1, 1)),
              },
              "MeanOfAbsMaxCalibration_1": {
                  "count": (dtype("int32"), ()),
                  "sum_of_max": (dtype("float32"), (2, 1, 10)),
              },
          }
      },
  }
  # print("SS ======== calib pytree", calibration_pytree["qc"])
  utils.test_pprint_eq(
      expected_calibration_pytree, calibration_pytree["qc"])

  # The count number should be equal to the number of calibration.
  einsum_params = calibrated_params["qc"]["AqtEinsum_0"]["AqtDotGeneral_0"]
  einsum_count = einsum_params["MeanOfAbsMaxCalibration_0"]["count"]
  # self.assertEqual(calibration_steps, einsum_count)
  breakpoint()
  assert calibration_steps == einsum_count, "calib step != einsum count"

  # Stage 3: Training with the calibrated numbers.
  print("SS ======== calibrated_params Conv_0 ",
        calibrated_params["params"]["Conv_0"]["kernel"][:,:,:, 0],
        flush=True)
  state = state.replace(model=copy.deepcopy(calibrated_params))
  print("SS ======== replaced state Conv_0 ",
        state.model["params"]["Conv_0"]["kernel"][:,:,:, 0],
        flush=True)
  state, _, _ = flax_e2e_model.train_epoch(
      state, ds2, batch_size, rng=input_rng
  )
  print("SS ======== after trainig Conv_0 ",
        state.model["params"]["Conv_0"]["kernel"][:,:,:, 0],
        flush=True)

  # The calibrated parameters must not change.
  jax.tree.map(
      np.testing.assert_array_equal,
      calibrated_params["qc"],
      state.model["qc"],
  )

  # Other parameters should change due to the training.
  def assert_array_not_equal(x, y):
    mean_err = jnp.mean(jnp.abs(x - y))
    if mean_err == 0.0:
      assert False

  # breakpoint()
  jax.tree.map(
      assert_array_not_equal,
      calibrated_params["params"],
      state.model["params"],
  )

  # Stage 4. Convert the calibrated checkpoint.
  serve_fn, model_serving = flax_e2e_model.serving_conversion(
      state, weight_only=False
  )
  dtype = jnp.dtype
  expected_dtype = dtype("int4") if bits == 4 else dtype("int8")
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
              "qrhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=None,
                      scale=[(dtype("float32"), (1, 1, 1))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"),
                  )
              },
          }
      },
      "Conv_0": {
          "AqtConvGeneralDilated_0": {
              "qlhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=None,
                      scale=[(dtype("float32"), (1, 1, 1, 1))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"))},
              "qrhs": {"frozen": aqt_tensor.QTensor(
                  qvalue=(dtype("float32"), (3, 3, 1, 32)),
                  scale=[(dtype("float32"), (1, 1, 1, 32))],
                  scale_t=None,
                  dequant_dtype=dtype("float32"))
                        }
              }
          },
      "Conv_1": {
          "AqtConvGeneralDilated_0": {
              "qlhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=None,
                      scale=[(dtype("float32"), (1, 1, 1, 1))],
                      scale_t=None, dequant_dtype=dtype("float32"))
                  },
              "qrhs": {"frozen": aqt_tensor.QTensor(
                  qvalue=(dtype("float32"), (3, 3, 32, 64)),
                  scale=[(dtype("float32"), (1, 1, 1, 64))],
                  scale_t=None,
                  dequant_dtype=dtype("float32"))
                        }
              }
          },
      "Dense_0": {
          "AqtDotGeneral_0": {
              "qlhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=None,
                      scale=[(dtype("float32"), (1, 1, 1))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"),
                  )
              },
              "qrhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=(expected_dtype, (2, 1568, 256)),
                      scale=[(dtype("float32"), (2, 1, 256))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"),
                  )
              },
          }
      },
      "Dense_1": {
          "AqtDotGeneral_0": {
              "qlhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=None,
                      scale=[(dtype("float32"), (1, 1, 1))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"),
                  )
              },
              "qrhs": {
                  "frozen": aqt_tensor.QTensor(
                      qvalue=(expected_dtype, (2, 128, 10)),
                      scale=[(dtype("float32"), (2, 1, 10))],
                      scale_t=None,
                      dequant_dtype=dtype("float32"),
                  )
              },
          }
      },
  }

  serving_pytree = jax.tree_util.tree_map(
      lambda x: (x.dtype, x.shape), model_serving
  )
  # print("SS ======== serving pytree", serving_pytree["aqt"])
  utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

  # Compare logits of models before conversion and after conversion.
  def forward(model, apply_fn):
    return apply_fn(
        model,
        ds["image"],
        rngs={"params": jax.random.PRNGKey(0)},
        mutable=True,
    )

  logits_before_conversion, _ = forward(state.model, state.cnn_eval.apply)
  logits_after_conversion, _ = forward(model_serving, serve_fn)
  # print(
  #     "SS ********* logits_before_conversion",
  #     logits_before_conversion, flush=True)
  # print(
  #     "SS ********* logits_after_conversion",
  #     logits_after_conversion, flush=True)
  # breakpoint()
  assert (logits_before_conversion == logits_after_conversion).all()


if __name__ == "__main__":
  configs = {
            "fwd_bits": 4,
            "fwd_accumulator_dtype": None,
            "dlhs_accumulator_dtype": None,
        }
  bits = 4
  test_mnist_calibration(configs, bits)
