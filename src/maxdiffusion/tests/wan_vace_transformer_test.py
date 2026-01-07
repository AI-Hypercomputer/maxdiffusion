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

import os
import jax
import jax.numpy as jnp
import unittest
from absl.testing import absltest
from flax import nnx
from jax.sharding import Mesh

from .. import pyconfig
from ..max_utils import (create_device_mesh, get_flash_block_sizes)
from ..models.wan.transformers.transformer_wan import (
    WanRotaryPosEmbed,
)
from ..models.wan.transformers.transformer_wan_vace import (
    WanVACETransformerBlock,
)
import qwix
import flax

flax.config.update("flax_always_shard_variable", False)
RealQtRule = qwix.QtRule


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanVaceTransformerTest(unittest.TestCase):
  def test_wan_vace_block_returns_the_correct_shape(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config

    devices_array = create_device_mesh(config)

    flash_block_sizes = get_flash_block_sizes(config)

    mesh = Mesh(devices_array, config.mesh_axes)

    dim = 5120
    ffn_dim = 13824
    num_heads = 40
    qk_norm = "rms_norm_across_heads"
    cross_attn_norm = True
    eps = 1e-6

    batch_size = 1
    channels = 16
    frames = 21
    height = 90
    width = 160
    hidden_dim = 75600

    # for rotary post embed.
    hidden_states_shape = (batch_size, frames, height, width, channels)
    dummy_hidden_states = jnp.ones(hidden_states_shape)

    wan_rot_embed = WanRotaryPosEmbed(attention_head_dim=128, patch_size=[1, 2, 2], max_seq_len=1024)
    dummy_rotary_emb = wan_rot_embed(dummy_hidden_states)
    assert dummy_rotary_emb.shape == (batch_size, 1, hidden_dim, 64)

    # for transformer block
    dummy_hidden_states = jnp.ones((batch_size, hidden_dim, dim))

    dummy_control_hidden_states = jnp.ones((batch_size, hidden_dim, dim))

    dummy_encoder_hidden_states = jnp.ones((batch_size, 512, dim))

    dummy_temb = jnp.ones((batch_size, 6, dim))

    wan_vace_block = WanVACETransformerBlock(
        rngs=rngs,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        qk_norm=qk_norm,
        cross_attn_norm=cross_attn_norm,
        eps=eps,
        attention="flash",
        mesh=mesh,
        flash_block_sizes=flash_block_sizes,
        apply_input_projection=True,
        apply_output_projection=True,
    )
    with mesh:
      conditioning_states, control_hidden_states = wan_vace_block(
          hidden_states=dummy_hidden_states,
          encoder_hidden_states=dummy_encoder_hidden_states,
          control_hidden_states=dummy_control_hidden_states,
          temb=dummy_temb,
          rotary_emb=dummy_rotary_emb,
      )
    assert conditioning_states.shape == dummy_hidden_states.shape
    assert control_hidden_states.shape == dummy_hidden_states.shape

if __name__ == "__main__":
  absltest.main()
