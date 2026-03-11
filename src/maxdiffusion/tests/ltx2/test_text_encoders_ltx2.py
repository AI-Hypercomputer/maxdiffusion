"""
Copyright 2025 Google LLC

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
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder


class LTX2TextEncodersTest(unittest.TestCase):

  def setUp(self):
    self.rng = nnx.Rngs(0)
    self.B = 2
    self.T = 16
    self.gemma_dim = 32
    self.gemma_layers = 3

    # Mock Gemma hidden states
    self.hidden_states = [jnp.array(np.random.randn(self.B, self.T, self.gemma_dim)) for _ in range(self.gemma_layers)]

    self.attention_mask = jnp.ones((self.B, self.T))

  def test_av_encoder_forward(self):
    encoder = LTX2AudioVideoGemmaTextEncoder(
        caption_channels=self.gemma_dim,
        text_proj_in_factor=self.gemma_layers,
        video_connector_num_attention_heads=4,
        video_connector_attention_head_dim=8,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=8,
        audio_connector_num_attention_heads=4,
        audio_connector_attention_head_dim=8,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=8,
        rope_type="split",
        attention_kernel="dot_product",
        mesh=None,
        rngs=self.rng,
    )

    video_out, audio_out, new_mask = encoder(tuple(self.hidden_states), self.attention_mask)

    # Expected shapes: Both [B, T, caption_channels]
    self.assertEqual(video_out.shape, (self.B, self.T, self.gemma_dim))
    self.assertEqual(audio_out.shape, (self.B, self.T, self.gemma_dim))

    # Ensure they are different (different random init for connectors)
    self.assertFalse(
        jnp.allclose(video_out, audio_out), "Video and Audio outputs should differ due to different connector weights"
    )

    print("\n[PASS] Audio-Video Encoder Forward Pass Verified.")


if __name__ == "__main__":
  unittest.main()
