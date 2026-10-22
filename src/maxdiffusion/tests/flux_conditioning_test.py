# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the FLUX conditioning path (CPU backend).

FLUX shares CombinedTimestepGuidanceTextProjEmbeddings and
AdaLayerNormContinuous with Flux.2-Klein, which reads them with different
conventions. These cover the conventions FLUX needs.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.embeddings_flax import CombinedTimestepGuidanceTextProjEmbeddings
from maxdiffusion.models.flux.transformers.transformer_flux_flax import Flux2KleinTransformer2DModel
from maxdiffusion.models.normalization_flax import AdaLayerNormContinuous


def _layer_norm(x, eps=1e-5):
  mean = np.mean(x, axis=-1, keepdims=True)
  variance = np.var(x, axis=-1, keepdims=True)
  return (x - mean) / np.sqrt(variance + eps)


class FluxConditioningTest(unittest.TestCase):

  def test_guidance_embeddings_accept_projected_timesteps(self):
    """FluxTransformer2DModel projects timestep and guidance before this module."""
    batch, frequency_embedding_size, embedding_dim, pooled_projection_dim = 2, 256, 64, 32
    module = CombinedTimestepGuidanceTextProjEmbeddings(
        embedding_dim=embedding_dim, pooled_projection_dim=pooled_projection_dim
    )
    timestep = jnp.zeros((batch, frequency_embedding_size), jnp.float32)
    guidance = jnp.zeros((batch, frequency_embedding_size), jnp.float32)
    pooled_projection = jnp.zeros((batch, pooled_projection_dim), jnp.float32)

    variables = module.init(jax.random.PRNGKey(0), timestep, guidance, pooled_projection)
    conditioning = module.apply(variables, timestep, guidance, pooled_projection)

    self.assertEqual(conditioning.shape, (batch, embedding_dim))

  def test_norm_out_dense_is_named_linear(self):
    """load_flow_model writes final_layer.adaLN_modulation_1 to norm_out/linear."""
    module = AdaLayerNormContinuous(embedding_dim=4, elementwise_affine=False)
    x = jnp.zeros((2, 3, 4), jnp.float32)
    conditioning_embedding = jnp.zeros((2, 8), jnp.float32)

    variables = module.init(jax.random.PRNGKey(0), x, conditioning_embedding)

    self.assertIn("linear", variables["params"])

  def test_shift_scale_order_reads_shift_first(self):
    """The original FLUX checkpoint emits shift ahead of scale."""
    embedding_dim, conditioning_dim, batch, sequence = 4, 8, 2, 3
    x = jax.random.normal(jax.random.PRNGKey(1), (batch, sequence, embedding_dim))
    conditioning_embedding = jax.random.normal(jax.random.PRNGKey(2), (batch, conditioning_dim))

    def modulate(scale_shift_order):
      module = AdaLayerNormContinuous(
          embedding_dim=embedding_dim, elementwise_affine=False, scale_shift_order=scale_shift_order
      )
      variables = flax.linen.meta.unbox(flax.core.unfreeze(module.init(jax.random.PRNGKey(0), x, conditioning_embedding)))
      # A zero kernel leaves the bias as the whole modulation, so the halves are
      # known: 2.0 in the first, 0.0 in the second.
      variables["params"]["linear"]["kernel"] = jnp.zeros_like(variables["params"]["linear"]["kernel"])
      variables["params"]["linear"]["bias"] = jnp.concatenate([jnp.full((embedding_dim,), 2.0), jnp.zeros((embedding_dim,))])
      return module.apply(variables, x, conditioning_embedding)

    normalized = _layer_norm(np.asarray(x))

    # shift first: (1 + 0) * norm + 2
    np.testing.assert_allclose(np.asarray(modulate("shift_scale")), normalized + 2.0, atol=1e-4)
    # scale first: (1 + 2) * norm + 0
    np.testing.assert_allclose(np.asarray(modulate("scale_shift")), 3.0 * normalized, atol=1e-4)

  def test_flux2klein_linen_projects_timesteps_before_embedding(self):
    """The Linen Flux.2-Klein model shares the guidance embeddings with FLUX."""
    batch, image_tokens, text_tokens = 2, 4, 3
    model = Flux2KleinTransformer2DModel(
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dim=(2, 2, 2, 2),
        axes_dims_rope=(2, 2, 2, 2),
    )
    inputs = dict(
        hidden_states=jnp.ones((batch, image_tokens, 4), jnp.float32),
        encoder_hidden_states=jnp.ones((batch, text_tokens, 8), jnp.float32),
        pooled_projections=jnp.ones((batch, 8), jnp.float32),
        timestep=jnp.full((batch,), 0.5, jnp.float32),
        img_ids=jnp.zeros((image_tokens, 4), jnp.float32),
        txt_ids=jnp.zeros((text_tokens, 4), jnp.float32),
        guidance=jnp.full((batch,), 0.5, jnp.float32),
        return_dict=False,
    )

    variables = model.init(jax.random.PRNGKey(0), **inputs)
    time_text_embed = flax.linen.meta.unbox(variables["params"]["time_text_embed"])
    for embedder in ("FlaxTimestepEmbedding_0", "FlaxTimestepEmbedding_1"):
      self.assertEqual(time_text_embed[embedder]["linear_1"]["kernel"].shape, (256, 16))

    (sample,) = model.apply(variables, **inputs)
    self.assertEqual(sample.shape, (batch, image_tokens, 4))


if __name__ == "__main__":
  unittest.main()
