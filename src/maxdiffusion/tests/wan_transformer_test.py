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

from ..models.wan.transformers.transformer_wan import WanRotaryPosEmbed

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
    # output shape should be torch.Size([1, 1, 75600, 64])

if __name__ == "__main__":
  absltest.main()