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

"""
Unit test suite for Flax NNX FLUX.2-klein models and components.
"""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFluxTransformer2DModel
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, NNXFlaxQwen3Model
from maxdiffusion.models.vae_flax import NNXFlaxAutoencoderKL
from maxdiffusion.models.embeddings_flax import NNXCombinedTimestepGuidanceTextProjEmbeddings


class NNXFlux2KleinTest(unittest.TestCase):

  def test_nnx_combined_timestep_embeddings(self):
    rngs = nnx.Rngs(0)
    embedder = NNXCombinedTimestepGuidanceTextProjEmbeddings(
        rngs=rngs,
        embedding_dim=768,
        pooled_projection_dim=768,
        guidance_embeds=True,
    )
    timestep = jnp.array([500.0])
    guidance = jnp.array([3.5])
    pooled_projection = jnp.ones((1, 768))

    out = embedder(timestep, guidance, pooled_projection)
    self.assertEqual(out.shape, (1, 768))

  def test_nnx_qwen3_text_encoder_forward(self):
    rngs = nnx.Rngs(0)
    config = FlaxQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
    )
    model = NNXFlaxQwen3Model(rngs=rngs, config=config)
    dummy_ids = jnp.ones((1, 16), dtype=jnp.int32)
    last_hidden_state, all_hidden_states = model(dummy_ids)

    self.assertEqual(last_hidden_state.shape, (1, 16, 256))
    self.assertEqual(len(all_hidden_states), 3)  # Embeddings + 2 layers

  def test_nnx_vae_decoder_forward(self):
    rngs = nnx.Rngs(0)
    vae = NNXFlaxAutoencoderKL(
        rngs=rngs,
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(64, 128),
        layers_per_block=1,
    )
    dummy_latents = jnp.ones((1, 16, 16, 16))
    init_rng = jax.random.PRNGKey(0)
    decoder_params = vae.decoder.init(init_rng, jnp.zeros((1, 16, 16, 16)))["params"]
    out = vae.decode(dummy_latents, decoder_params=decoder_params)
    self.assertEqual(out.sample.shape, (1, 3, 32, 32))

  def test_nnx_flux_transformer_forward(self):
    rngs = nnx.Rngs(0)
    transformer = NNXFluxTransformer2DModel(
        rngs=rngs,
        in_channels=16,
        num_layers=1,
        num_single_layers=2,
        attention_head_dim=128,
        num_attention_heads=4,
        joint_attention_dim=128,
        pooled_projection_dim=128,
        guidance_embeds=True,
        axes_dim=(16, 56, 56),
    )
    hidden_states = jnp.ones((1, 64, 16))
    encoder_hidden_states = jnp.ones((1, 16, 128))
    pooled_projections = jnp.ones((1, 128))
    timestep = jnp.array([100.0])
    guidance = jnp.array([3.5])
    img_ids = jnp.zeros((64, 3))
    txt_ids = jnp.zeros((16, 3))

    output = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
    )
    self.assertEqual(output.shape, (1, 64, 16))


if __name__ == "__main__":
  unittest.main()
