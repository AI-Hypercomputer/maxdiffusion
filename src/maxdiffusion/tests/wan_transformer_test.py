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

import jax
import jax.numpy as jnp
import unittest
from absl.testing import absltest
from flax import nnx

from ..models.wan.transformers.transformer_wan import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from ..models.embeddings_flax import NNXTimestepEmbedding, NNXPixArtAlphaTextProjection

class WanTransformerTest(unittest.TestCase):
  def setUp(self):
    WanTransformerTest.dummy_data = {}
  
  def test_rotary_pos_embed(self):
    batch_size = 1
    channels = 16
    frames = 21
    height = 90
    width = 160
    hidden_states_shape = (batch_size, frames, height, width, channels)
    dummy_hidden_states = jnp.ones(hidden_states_shape)
    wan_rot_embed = WanRotaryPosEmbed(
      attention_head_dim=128,
      patch_size=[1, 2, 2],
      max_seq_len=1024
    )
    dummy_output = wan_rot_embed(dummy_hidden_states)
    assert dummy_output.shape == (1, 1, 75600, 64)

  def test_nnx_pixart_alpha_text_projection(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    dummy_caption = jnp.ones((1, 512, 4096))
    layer = NNXPixArtAlphaTextProjection(
      rngs=rngs,
      in_features=4096,
      hidden_size=5120
    )
    dummy_output = layer(dummy_caption)
    dummy_output.shape == (1, 512, 5120)

  def test_nnx_timestep_embedding(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    dummy_sample = jnp.ones((1, 256))
    layer = NNXTimestepEmbedding(
      rngs=rngs,
      in_channels=256,
      time_embed_dim=5120
    )
    dummy_output = layer(dummy_sample)
    assert dummy_output.shape == (1, 5120)

  def test_wan_time_text_embedding(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch_size = 1
    dim=5120
    time_freq_dim=256
    time_proj_dim=30720
    text_embed_dim=4096
    layer = WanTimeTextImageEmbedding(
      rngs=rngs,
      dim=dim,
      time_freq_dim=time_freq_dim,
      time_proj_dim=time_proj_dim,
      text_embed_dim=text_embed_dim
    )
    
    dummy_timestep = jnp.ones(batch_size)

    encoder_hidden_states_shape = (batch_size, time_freq_dim * 2, text_embed_dim)
    dummy_encoder_hidden_states = jnp.ones(encoder_hidden_states_shape)
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = layer(dummy_timestep, dummy_encoder_hidden_states)
    assert temb.shape == (batch_size, dim)
    assert timestep_proj.shape == (batch_size, time_proj_dim)
    assert encoder_hidden_states.shape == (batch_size, time_freq_dim * 2, dim)

if __name__ == "__main__":
  absltest.main()