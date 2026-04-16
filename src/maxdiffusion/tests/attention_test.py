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
from unittest import mock

from absl.testing import absltest
from flax.linen import partitioning as nn_partitioning
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
import numpy as np
from ..models import attention_flax
from ..models.attention_flax import FlaxAttention, _select_flash_block_sizes
from .. import max_utils
from .. import pyconfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class AttentionTest(unittest.TestCase):
  """Test Attention"""

  def setUp(self):
    AttentionTest.dummy_data = {}

  def _ulysses_mesh(self):
    devices = np.array(jax.devices()[:2]).reshape(1, 1, 2, 1)
    return Mesh(devices, ("data", "fsdp", "context", "tensor"))

  def _ulysses_axis_rules(self):
    return (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
    )

  def _flash_axis_rules(self):
    return (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, None),
        (attention_flax.D_KV, None),
    )

  def _ulysses_block_sizes(self, block_size=4):
    return attention_flax.BlockSizes(
        block_q=block_size,
        block_kv_compute=block_size,
        block_kv=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
        use_fused_bwd_kernel=False,
    )

  def test_splash_attention(self):
    """Test numerics of splash attention are equivalent to dot_product"""

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
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

  def test_cross_attention_overrides_configured_flash_block_sizes(self):
    query = jnp.zeros((1, 1024, 256), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 257, 256), dtype=jnp.bfloat16)
    configured_block_sizes = splash_attention_kernel.BlockSizes(
        block_q=384,
        block_kv_compute=192,
        block_kv=320,
        block_q_dkv=256,
        block_kv_dkv=288,
        block_kv_dkv_compute=160,
        block_q_dq=128,
        block_kv_dq=96,
        use_fused_bwd_kernel=False,
    )

    block_sizes = _select_flash_block_sizes(
        query=query,
        key=key,
        flash_block_sizes=configured_block_sizes,
        dtype=jnp.bfloat16,
        attention_kernel="flash",
    )

    assert block_sizes.block_q == configured_block_sizes.block_q
    assert block_sizes.block_q_dkv == configured_block_sizes.block_q
    assert block_sizes.block_q_dq == configured_block_sizes.block_q
    assert block_sizes.block_kv_compute == 257
    assert block_sizes.block_kv == 257
    assert block_sizes.block_kv_dkv == 257
    assert block_sizes.block_kv_dkv_compute == 384
    assert block_sizes.block_kv_dq == 384

  def test_default_flash_block_sizes_use_sequence_axis_for_3d_inputs(self):
    query = jnp.zeros((1, 128, 4096), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 257, 4096), dtype=jnp.bfloat16)

    block_sizes = _select_flash_block_sizes(
        query=query,
        key=key,
        flash_block_sizes=None,
        dtype=jnp.bfloat16,
        attention_kernel="flash",
    )

    assert block_sizes.block_q == 1024
    assert block_sizes.block_kv_compute == 257
    assert block_sizes.block_kv == 257
    assert block_sizes.block_q_dkv == 1024
    assert block_sizes.block_kv_dkv == 257
    assert block_sizes.block_kv_dkv_compute == 128
    assert block_sizes.block_q_dq == 1024
    assert block_sizes.block_kv_dq == 128

  def test_select_flash_block_sizes_returns_configured_for_self_attention(self):
    """Block-size selection should return the configured sizes unchanged for self-attention."""
    custom_block_sizes = self._ulysses_block_sizes(block_size=16)
    query = jnp.zeros((1, 128, 1), dtype=jnp.float32)
    key = jnp.zeros((1, 128, 1), dtype=jnp.float32)

    self_attention_block_sizes = _select_flash_block_sizes(
        query=query,
        key=key,
        flash_block_sizes=custom_block_sizes,
        dtype=jnp.float32,
        attention_kernel="flash",
    )
    self.assertIs(self_attention_block_sizes, custom_block_sizes)

  def test_select_flash_block_sizes_derives_cross_attn_defaults_for_tokamax(self):
    """Block-size selection should derive cross-attn defaults and set tokamax_flash flags."""
    custom_block_sizes = self._ulysses_block_sizes(block_size=16)
    query = jnp.zeros((1, 257, 1), dtype=jnp.float32)
    key = jnp.zeros((1, 513, 1), dtype=jnp.float32)

    cross_attention_block_sizes = _select_flash_block_sizes(
        query=query,
        key=key,
        flash_block_sizes=custom_block_sizes,
        dtype=jnp.float32,
        attention_kernel="tokamax_flash",
    )
    self.assertEqual(cross_attention_block_sizes.block_q, 16)
    self.assertEqual(cross_attention_block_sizes.block_kv, 513)
    self.assertEqual(cross_attention_block_sizes.block_kv_compute, 513)
    self.assertEqual(cross_attention_block_sizes.block_kv_dkv_compute, 257)
    self.assertIsNone(cross_attention_block_sizes.block_q_dq)
    self.assertIsNone(cross_attention_block_sizes.block_kv_dq)
    self.assertTrue(cross_attention_block_sizes.use_fused_bwd_kernel)

  def test_ulysses_attention_round_trips_query_when_heads_are_divisible(self):
    """Ulysses attention should preserve the query layout after its collectives."""
    batch = 2
    length = 5
    heads = 4
    head_depth = 4
    query = jnp.arange(batch * length * heads * head_depth, dtype=jnp.float32).reshape(batch, length, heads * head_depth)
    key = query + 1000.0
    value = query + 2000.0
    mesh = self._ulysses_mesh()

    def fake_make_splash_mha(**unused_kwargs):
      def fake_kernel(q, k, v, segment_ids):
        del k, v, segment_ids
        return q

      return fake_kernel

    with (
        mesh,
        nn_partitioning.axis_rules(self._ulysses_axis_rules()),
        mock.patch.object(
            attention_flax.splash_attention_kernel,
            "make_splash_mha",
            side_effect=fake_make_splash_mha,
        ),
    ):
      output = attention_flax._ulysses_attention(
          query,
          key,
          value,
          heads=heads,
          mesh=mesh,
          axis_names_q=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_Q_LENGTH,
              attention_flax.D_KV,
          ),
          axis_names_kv=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_KV_LENGTH,
              attention_flax.D_KV,
          ),
          flash_block_sizes=self._ulysses_block_sizes(),
          dtype=jnp.float32,
      )

    self.assertEqual(output.shape, query.shape)
    self.assertTrue(jnp.array_equal(output, query))

  def test_ulysses_attention_raises_when_heads_are_not_divisible_by_context_shards(self):
    """Ulysses attention should fail fast when heads cannot be evenly sharded."""
    batch = 2
    length = 5
    heads = 3
    head_depth = 4
    query = jnp.arange(batch * length * heads * head_depth, dtype=jnp.float32).reshape(batch, length, heads * head_depth)
    key = query + 1000.0
    value = query + 2000.0
    mesh = self._ulysses_mesh()

    with mesh, nn_partitioning.axis_rules(self._ulysses_axis_rules()):
      with self.assertRaisesRegex(
          ValueError,
          r"heads=3 and context_shards=2",
      ):
        attention_flax._ulysses_attention(
            query,
            key,
            value,
            heads=heads,
            mesh=mesh,
            axis_names_q=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_Q_LENGTH,
                attention_flax.D_KV,
            ),
            axis_names_kv=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_KV_LENGTH,
                attention_flax.D_KV,
            ),
            flash_block_sizes=self._ulysses_block_sizes(),
            dtype=jnp.float32,
        )

  def test_ulysses_attention_matches_flash_attention_with_same_local_kernel(self):
    """Flash and Ulysses should agree when the local splash kernel is shared."""
    batch = 2
    length = 6
    heads = 4
    head_depth = 3
    query = jnp.arange(batch * length * heads * head_depth, dtype=jnp.float32).reshape(batch, length, heads * head_depth)
    key = query + 100.0
    value = query + 200.0
    mesh = self._ulysses_mesh()

    def fake_make_splash_mha(**unused_kwargs):
      def fake_kernel(q, k, v, segment_ids):
        del k, segment_ids
        return q + jnp.mean(v, axis=1, keepdims=True)

      return fake_kernel

    with mock.patch.object(
        attention_flax.splash_attention_kernel,
        "make_splash_mha",
        side_effect=fake_make_splash_mha,
    ):
      with mesh, nn_partitioning.axis_rules(self._flash_axis_rules()):
        flash_output = attention_flax._tpu_flash_attention(
            query,
            key,
            value,
            heads=heads,
            mesh=mesh,
            axis_names_q=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_Q_LENGTH,
                attention_flax.D_KV,
            ),
            axis_names_kv=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_KV_LENGTH,
                attention_flax.D_KV,
            ),
            flash_block_sizes=self._ulysses_block_sizes(),
            dtype=jnp.float32,
            attention_kernel="flash",
        )

      with mesh, nn_partitioning.axis_rules(self._ulysses_axis_rules()):
        ulysses_output = attention_flax._ulysses_attention(
            query,
            key,
            value,
            heads=heads,
            mesh=mesh,
            axis_names_q=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_Q_LENGTH,
                attention_flax.D_KV,
            ),
            axis_names_kv=(
                attention_flax.BATCH,
                attention_flax.SELF_ATTN_HEAD,
                attention_flax.SELF_ATTN_KV_LENGTH,
                attention_flax.D_KV,
            ),
            flash_block_sizes=self._ulysses_block_sizes(),
            dtype=jnp.float32,
        )

    self.assertEqual(flash_output.shape, ulysses_output.shape)
    self.assertTrue(jnp.array_equal(flash_output, ulysses_output))

  def test_ulysses_attention_uses_attention_mask_for_segment_ids(self):
    """Ulysses attention should forward the attention mask into kv segment ids."""
    batch = 2
    length = 5
    heads = 4
    head_depth = 3
    query = jnp.arange(batch * length * heads * head_depth, dtype=jnp.float32).reshape(batch, length, heads * head_depth)
    key = query + 100.0
    value = query + 200.0
    attention_mask = jnp.array([[1, 0, 1, 0, 1]], dtype=jnp.int32)
    mesh = self._ulysses_mesh()

    def fake_make_splash_mha(**unused_kwargs):
      def fake_kernel(q, k, v, segment_ids):
        del k, v
        kv_mask = segment_ids.kv.astype(q.dtype)[None, :, None]
        return q + kv_mask

      return fake_kernel

    with (
        mesh,
        nn_partitioning.axis_rules(self._ulysses_axis_rules()),
        mock.patch.object(
            attention_flax.splash_attention_kernel,
            "make_splash_mha",
            side_effect=fake_make_splash_mha,
        ),
    ):
      output = attention_flax._ulysses_attention(
          query,
          key,
          value,
          heads=heads,
          mesh=mesh,
          axis_names_q=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_Q_LENGTH,
              attention_flax.D_KV,
          ),
          axis_names_kv=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_KV_LENGTH,
              attention_flax.D_KV,
          ),
          flash_block_sizes=self._ulysses_block_sizes(),
          dtype=jnp.float32,
          attention_mask=attention_mask,
      )

    expected = query + jnp.broadcast_to(attention_mask[:, :, None], query.shape)
    self.assertEqual(output.shape, query.shape)
    self.assertTrue(jnp.array_equal(output, expected))


if __name__ == "__main__":
  absltest.main()
