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

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from ...models.ltx2.text_encoders.text_encoders_ltx2 import LTX2VideoGemmaTextEncoder, LTX2AudioVideoGemmaTextEncoder


class LTX2TextEncodersTest(unittest.TestCase):

  def setUp(self):
    self.rng = nnx.Rngs(0)
    self.B = 2
    self.T = 16
    self.gemma_dim = 32
    self.gemma_layers = 3
    self.proj_dim = 64

    # Mock Gemma hidden states
    self.hidden_states = [jnp.array(np.random.randn(self.B, self.T, self.gemma_dim)) for _ in range(self.gemma_layers)]

    self.attention_mask = jnp.ones((self.B, self.T))

  def test_video_encoder_forward(self):
    encoder = LTX2VideoGemmaTextEncoder(
        gemma_dim=self.gemma_dim,
        gemma_layers=self.gemma_layers,
        projection_dim=self.proj_dim,
        connector_heads=4,
        connector_head_dim=16,
        connector_layers=1,
        num_thinking_tokens=8,
        attention_kernel="dot_product",
        mesh=None,
        rngs=self.rng,
    )

    output = encoder(tuple(self.hidden_states), self.attention_mask)

    # Expected shape: [B, T, proj_dim]
    self.assertEqual(output.shape, (self.B, self.T, self.proj_dim))
    print("\n[PASS] Video Encoder Forward Pass Verified.")

  def test_av_encoder_forward(self):
    encoder = LTX2AudioVideoGemmaTextEncoder(
        gemma_dim=self.gemma_dim,
        gemma_layers=self.gemma_layers,
        projection_dim=self.proj_dim,
        connector_heads=4,
        connector_head_dim=16,
        connector_layers=1,
        num_thinking_tokens=8,
        attention_kernel="dot_product",
        mesh=None,
        rngs=self.rng,
    )

    video_out, audio_out = encoder(tuple(self.hidden_states), self.attention_mask)

    # Expected shapes: Both [B, T, proj_dim]
    self.assertEqual(video_out.shape, (self.B, self.T, self.proj_dim))
    self.assertEqual(audio_out.shape, (self.B, self.T, self.proj_dim))

    # Ensure they are different (different random init for connectors)
    # Note: In reality they are initialized differently, so outputs should differ
    self.assertFalse(
        jnp.allclose(video_out, audio_out), "Video and Audio outputs should differ due to different connector weights"
    )

    print("\n[PASS] Audio-Video Encoder Forward Pass Verified.")


if __name__ == "__main__":
  unittest.main()
