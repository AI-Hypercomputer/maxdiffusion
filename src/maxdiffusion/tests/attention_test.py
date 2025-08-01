"""
 Copyright 2024 Google LLC

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
import unittest
from absl.testing import absltest
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
from ..models.attention_flax import FlaxAttention
from .. import max_utils
from .. import pyconfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class AttentionTest(unittest.TestCase):
  """Test Attention"""

  def setUp(self):
    AttentionTest.dummy_data = {}

  def test_splash_attention(self):
    """Test numerics of splash attention are equivalent to dot_product"""

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base21.yml"),
            'flash_block_sizes={"block_q" : 512, "block_kv_compute": 512, "block_kv": 512,'
            '"block_q_dkv": 512, "block_kv_dkv": 512, "block_kv_dkv_compute": 512,'
            '"block_q_dq": 512, "block_kv_dq": 512}',
        ],
        unittest=True,
    )
    config = pyconfig.config

    batch = 8
    length = 4096
    heads = 10
    head_depth = 64

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key1, (batch, length, heads * head_depth))
    dot_product_attention = FlaxAttention(
        heads * head_depth,
        heads,
        head_depth,
        split_head_dim=True,
        attention_kernel="dot_product",
        mesh=None,
        dtype=jnp.bfloat16,
    )

    params = dot_product_attention.init(key2, x)["params"]
    p_apply = jax.jit(dot_product_attention.apply).lower({"params": params}, x).compile()
    dot_attention_out = p_apply({"params": params}, x)

    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    flash_block_sizes = max_utils.get_flash_block_sizes(config)
    with mesh:
      splash_attention = FlaxAttention(
          heads * head_depth,
          heads,
          head_depth,
          split_head_dim=True,
          attention_kernel="flash",
          mesh=mesh,
          dtype=jnp.bfloat16,
          flash_block_sizes=flash_block_sizes,
      )

      params = splash_attention.init(key2, x)["params"]
      p_apply = jax.jit(splash_attention.apply).lower({"params": params}, x).compile()
      splash_attention_out = p_apply({"params": params}, x)

    diff_norm = jnp.linalg.norm(dot_attention_out - splash_attention_out)

    assert diff_norm < 1.0


if __name__ == "__main__":
  absltest.main()
