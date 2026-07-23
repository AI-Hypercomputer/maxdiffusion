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


def _tiny_model():
  return ZImageTransformer2DModel(
      rngs=nnx.Rngs(jax.random.key(0)),
      in_channels=4,
      dim=32,
      n_layers=1,
      n_refiner_layers=1,
      n_heads=4,
      n_kv_heads=4,
      cap_feat_dim=8,
      axes_dims=(2, 2, 4),
      axes_lens=(64, 64, 64),
      attention_kernel="dot_product",
  )


def test_variable_prompt_and_image_lengths():
  model = _tiny_model()
  output = model(
      [jnp.ones((4, 1, 4, 4)), jnp.ones((4, 1, 4, 6))],
      jnp.array([0.1, 0.9]),
      [jnp.ones((5, 8)), jnp.ones((17, 8))],
  ).sample
  assert output[0].shape == (4, 1, 4, 4)
  assert output[1].shape == (4, 1, 4, 6)
