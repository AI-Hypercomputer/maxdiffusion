"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Fast structural tests for the JAX Z-Image transformer."""

import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.z_image.transformer_z_image import ZImageTransformer2DModel


import numpy as np
from maxdiffusion.models.z_image.z_image_utils import z_image_pytorch_key_to_nnx_key
from maxdiffusion.utils.testing_utils import cpu_only


def _tiny_model(scan_layers: bool = True):
  return ZImageTransformer2DModel(
      rngs=nnx.Rngs(jax.random.key(0)),
      in_channels=4,
      dim=32,
      n_layers=2,
      n_refiner_layers=2,
      n_heads=4,
      n_kv_heads=4,
      cap_feat_dim=8,
      axes_dims=(2, 2, 4),
      axes_lens=(64, 64, 64),
      attention_kernel="dot_product",
      scan_layers=scan_layers,
  )


def test_variable_prompt_and_image_lengths():
  for scan_layers in (True, False):
    model = _tiny_model(scan_layers=scan_layers)
    output = model(
        [jnp.ones((4, 1, 4, 4)), jnp.ones((4, 1, 4, 6))],
        jnp.array([0.1, 0.9]),
        [jnp.ones((5, 8)), jnp.ones((17, 8))],
    ).sample
    assert output[0].shape == (4, 1, 4, 4)
    assert output[1].shape == (4, 1, 4, 6)


@cpu_only
def test_scan_layers_true_and_false_output_parity():
  model_scanned = _tiny_model(scan_layers=True)
  model_unrolled = _tiny_model(scan_layers=False)

  # Copy weights from model_scanned to model_unrolled
  _, state_scanned, rest_scanned = nnx.split(model_scanned, nnx.Param, ...)
  flat_scanned = dict(nnx.to_flat_state(state_scanned))

  _, state_unrolled, rest_unrolled = nnx.split(model_unrolled, nnx.Param, ...)
  flat_unrolled = dict(nnx.to_flat_state(state_unrolled))

  for key_scanned, param_scanned in flat_scanned.items():
    if key_scanned[0] in ("layers", "noise_refiner", "context_refiner"):
      num_layers = param_scanned[...].shape[0]
      for i in range(num_layers):
        unrolled_key = (key_scanned[0], i) + key_scanned[1:]
        flat_unrolled[unrolled_key][...] = param_scanned[...][i]
    else:
      flat_unrolled[key_scanned][...] = param_scanned[...]

  model_unrolled = nnx.merge(nnx.graphdef(model_unrolled), nnx.from_flat_state(flat_unrolled), rest_unrolled)

  inputs_x = [jnp.ones((4, 1, 4, 4)), jnp.ones((4, 1, 4, 6))]
  inputs_t = jnp.array([0.1, 0.9])
  inputs_cap = [jnp.ones((5, 8)), jnp.ones((17, 8))]

  out_scanned = model_scanned(inputs_x, inputs_t, inputs_cap).sample
  out_unrolled = model_unrolled(inputs_x, inputs_t, inputs_cap).sample

  for s, u in zip(out_scanned, out_unrolled):
    np.testing.assert_allclose(np.asarray(s), np.asarray(u), atol=1e-5, rtol=1e-5)


def test_scan_layers_key_mapping():
  key_layer = "layers.3.attention.to_q.weight"
  path_unrolled, transpose_unrolled, idx_unrolled = z_image_pytorch_key_to_nnx_key(key_layer, scan_layers=False)
  assert path_unrolled == ("layers", 3, "attention", "to_q", "kernel")
  assert transpose_unrolled is True
  assert idx_unrolled is None

  path_scanned, transpose_scanned, idx_scanned = z_image_pytorch_key_to_nnx_key(key_layer, scan_layers=True)
  assert path_scanned == ("layers", "attention", "to_q", "kernel")
  assert transpose_scanned is True
  assert idx_scanned == 3
