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
from flax.linen import partitioning as nn_partitioning
from ..models.attention_flax import FlaxAttention
from .. import common_types
from .. import max_utils
from .. import pyconfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class AttentionTest(unittest.TestCase):
  """Test Attention"""

  @staticmethod
  def _resolve_mesh_axis(logical_axis_rules, axis_name):
    with nn_partitioning.axis_rules(logical_axis_rules):
      return nn_partitioning.logical_to_mesh_axes((axis_name,))[0]

  def setUp(self):
    AttentionTest.dummy_data = {}

  def test_splash_attention(self):
    """Test numerics of splash attention are equivalent to dot_product"""
    for attention_kernel in ["flash", "ulysses", "ulysses_fsdp"]:
      config_overrides = []
      if attention_kernel == "ulysses_fsdp":
        config_overrides.extend(["ici_fsdp_parallelism=-1", "ici_context_parallelism=1"])
      pyconfig.initialize(
          [
              None,
              os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
              f"attention={attention_kernel}",
              'flash_block_sizes={"block_q" : 512, "block_kv_compute": 512, "block_kv": 512,'
              '"block_q_dkv": 512, "block_kv_dkv": 512, "block_kv_dkv_compute": 512,'
              '"block_q_dq": 512, "block_kv_dq": 512}',
              *config_overrides,
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
            attention_kernel=attention_kernel,
            mesh=mesh,
            dtype=jnp.bfloat16,
            flash_block_sizes=flash_block_sizes,
        )

        params = splash_attention.init(key2, x)["params"]
        p_apply = jax.jit(splash_attention.apply).lower({"params": params}, x).compile()
        splash_attention_out = p_apply({"params": params}, x)

      diff_norm = jnp.linalg.norm(dot_attention_out - splash_attention_out)
      assert diff_norm < 1.0

  def test_ulysses_axis_rules(self):
    pyconfig.initialize(
        [None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"), "attention=ulysses"],
        unittest=True,
    )
    logical_axis_rules = pyconfig.config.logical_axis_rules
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.KV_LENGTH) == common_types.CONTEXT
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.SELF_ATTN_KV_LENGTH) == common_types.CONTEXT
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.CROSS_ATTN_KV_LENGTH) == common_types.CONTEXT

  def test_ulysses_fsdp_axis_rules(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
            "attention=ulysses_fsdp",
            "ici_fsdp_parallelism=-1",
            "ici_context_parallelism=1",
        ],
        unittest=True,
    )
    logical_axis_rules = pyconfig.config.logical_axis_rules
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.LENGTH) == common_types.FSDP
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.KV_LENGTH) == common_types.FSDP
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.SELF_ATTN_Q_LENGTH) == common_types.FSDP
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.SELF_ATTN_KV_LENGTH) == common_types.FSDP
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.CROSS_ATTN_Q_LENGTH) == common_types.FSDP
    assert self._resolve_mesh_axis(logical_axis_rules, common_types.CROSS_ATTN_KV_LENGTH) == common_types.FSDP


if __name__ == "__main__":
  absltest.main()
