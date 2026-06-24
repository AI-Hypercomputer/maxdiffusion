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
