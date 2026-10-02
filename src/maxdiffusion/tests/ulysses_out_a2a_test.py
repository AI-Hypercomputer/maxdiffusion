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

"""Tests for the inverse Ulysses exchange (`wan_ulysses_out_a2a`).

Inside the shard_map each shard holds [B, H/n, ..., S] (its heads, full
sequence) and must end with [B, H, ..., S/n] (every head, its sequence chunk).
Seen globally that is a pure resharding, so the output must equal the input
exactly, in both modes. Needs >= 2 devices; on CPU run with
XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.models import attention_flax

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# Kernel numerics grids and multi-device tests are skipped in CI; see
# end_to_end/tpu/run_wan_stack_tests.sh.
_SKIP_IN_GITHUB_ACTIONS = unittest.skipIf(
    IN_GITHUB_ACTIONS, "TPU kernel / multi-device test, skipped in GitHub Actions; run end_to_end/tpu/run_wan_stack_tests.sh"
)


class UlyssesSeqToHeadsTest(unittest.TestCase):

  def _require_devices(self, min_devices: int = 2) -> int:
    n = len(jax.devices())
    if n < min_devices:
      self.skipTest(f"needs >= {min_devices} devices (XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU)")
    return 4 if n >= 4 else 2

  def _exchange(self, mode, x, seq_axis):
    n = self._require_devices(2)
    mesh = Mesh(np.array(jax.devices()[:n]), ("context",))
    in_spec = P(None, "context", *([None] * (x.ndim - 2)))
    out_spec = [None] * x.ndim
    out_spec[seq_axis] = "context"
    fn = jax.shard_map(
        lambda t: attention_flax._ulysses_seq_to_heads(t, axis_name="context", seq_axis=seq_axis, num_shards=n, mode=mode),
        mesh=mesh,
        in_specs=(in_spec,),
        out_specs=P(*out_spec),
        check_vma=False,
    )
    return jax.jit(fn)(x)

  def test_both_modes_are_a_pure_reshard(self):
    n = self._require_devices(2)
    heads, d, seq = 2 * n, 128, 24 * n
    shapes = {3: (1, heads, d, seq), 2: (1, heads, seq, d)}  # [B,H,D,S] (kernel O^T) and [B,H,S,D]
    for seq_axis, shape in shapes.items():
      x = jax.random.normal(jax.random.PRNGKey(seq_axis), shape, jnp.float32).astype(jnp.bfloat16)
      for mode in attention_flax.ULYSSES_OUT_A2A_MODES:
        with self.subTest(seq_axis=seq_axis, mode=mode):
          got = self._exchange(mode, x, seq_axis)
          self.assertEqual(got.shape, x.shape)
          np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(x, np.float32))

  def test_rejects_unknown_mode(self):
    n = self._require_devices(2)
    x = jnp.zeros((1, 2 * n, 8 * n, 128), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "wan_ulysses_out_a2a"):
      self._exchange("bogus", x, 2)

  @_SKIP_IN_GITHUB_ACTIONS
  def test_pallas_shard_major_q_and_out_ring_kernel_bit_exact(self):
    u, h, s_loc, d = 2, 2, 256, 128
    s_total = u * s_loc
    bs = custom_splash._BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    q_3d = jax.random.normal(jax.random.PRNGKey(10), (h, s_total, d), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(11), (h, s_total, d), jnp.bfloat16) * 0.08
    v = jax.random.normal(jax.random.PRNGKey(12), (h, s_total, d), jnp.bfloat16)
    q_4d = jnp.swapaxes(q_3d.reshape(h, u, s_loc, d), 0, 1)

    grid_h = s_total // bs.block_q
    mk = jnp.stack([jnp.full((h, grid_h), -10.0, jnp.float32), jnp.ones((h, grid_h), jnp.float32)], axis=0)

    o_ref, m_ref, l_ref = custom_splash._splash_attention_forward_ring(
        q_3d,
        k,
        v,
        bs,
        q_seq_len=s_total,
        kv_seq_len=s_total,
        use_fixed_m=True,
        mk=mk,
        uniform_fixed_m=True,
        interpret=True,
        out_num_shards=1,
    )
    o_sm, m_sm, l_sm = custom_splash._splash_attention_forward_ring(
        q_4d,
        k,
        v,
        bs,
        q_seq_len=s_total,
        kv_seq_len=s_total,
        use_fixed_m=True,
        mk=mk,
        uniform_fixed_m=True,
        interpret=True,
        out_num_shards=u,
    )
    self.assertEqual(o_sm.shape, (u, h, s_loc, d))
    self.assertEqual(m_sm.shape, (u, h, s_loc))
    self.assertEqual(l_sm.shape, (u, h, s_loc))
    np.testing.assert_array_equal(np.asarray(jnp.swapaxes(o_sm, 0, 1).reshape(h, s_total, d)), np.asarray(o_ref))
    np.testing.assert_array_equal(np.asarray(jnp.swapaxes(m_sm, 0, 1).reshape(h, s_total)), np.asarray(m_ref))
    np.testing.assert_array_equal(np.asarray(jnp.swapaxes(l_sm, 0, 1).reshape(h, s_total)), np.asarray(l_ref))

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_ring_custom_chunked_matches_flat_bit_exact(self):
    from flax.linen import partitioning as nn_partitioning

    self._require_devices(4)
    mesh_4 = Mesh(np.array(jax.devices()[:4]).reshape(1, 1, 4, 1), ("data", "fsdp", "context", "tensor"))
    rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
    )
    b, heads, s_per_shard, d = 1, 4, 256, 128
    s_total = 4 * s_per_shard
    q = jax.random.normal(jax.random.PRNGKey(20), (b, s_total, heads * d), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(21), (b, s_total, heads * d), jnp.bfloat16) * 0.08
    v = jax.random.normal(jax.random.PRNGKey(22), (b, s_total, heads * d), jnp.bfloat16)
    bs = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128, "block_kv_compute_in": 128}
    axis_names_q = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_Q_LENGTH,
        attention_flax.D_KV,
    )
    axis_names_kv = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_KV_LENGTH,
        attention_flax.D_KV,
    )
    for use_fixed_m, per_q_block, bidirectional, mode in (
        (True, False, False, "chunked"),
        (True, True, False, "shard_major"),
        (False, False, False, "chunked"),
        (False, False, True, "shard_major"),
    ):
      with self.subTest(use_fixed_m=use_fixed_m, per_q_block=per_q_block, bidirectional=bidirectional, mode=mode):
        with mesh_4, nn_partitioning.axis_rules(rules):
          out_flat = attention_flax._ulysses_ring_custom_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh_4,
              axis_names_q=axis_names_q,
              axis_names_kv=axis_names_kv,
              flash_block_sizes=bs,
              ulysses_shards=2,
              use_fixed_m=use_fixed_m,
              per_q_block=per_q_block,
              bidirectional=bidirectional,
              wan_ulysses_out_a2a="flat",
          )
          out_chunked = attention_flax._ulysses_ring_custom_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh_4,
              axis_names_q=axis_names_q,
              axis_names_kv=axis_names_kv,
              flash_block_sizes=bs,
              ulysses_shards=2,
              use_fixed_m=use_fixed_m,
              per_q_block=per_q_block,
              bidirectional=bidirectional,
              wan_ulysses_out_a2a=mode,
          )
        self.assertEqual(out_chunked.shape, out_flat.shape)
        np.testing.assert_array_equal(np.asarray(out_chunked, jnp.float32), np.asarray(out_flat, jnp.float32))

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_custom_chunked_matches_flat_bit_exact(self):
    from flax.linen import partitioning as nn_partitioning

    n = self._require_devices(2)
    mesh_u = Mesh(np.array(jax.devices()[:n]).reshape(1, 1, n, 1), ("data", "fsdp", "context", "tensor"))
    rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
        (attention_flax.LENGTH, "context"),
        (attention_flax.HEAD, None),
    )
    b, heads, s_per_shard, d = 1, 2 * n, 256, 128
    s_total = n * s_per_shard
    q = jax.random.normal(jax.random.PRNGKey(30), (b, s_total, heads * d), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(31), (b, s_total, heads * d), jnp.bfloat16) * 0.08
    v = jax.random.normal(jax.random.PRNGKey(32), (b, s_total, heads * d), jnp.bfloat16)
    bs = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128, "block_kv_compute_in": 128}
    axis_names_q = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_Q_LENGTH,
        attention_flax.D_KV,
    )
    axis_names_kv = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_KV_LENGTH,
        attention_flax.D_KV,
    )
    for use_fixed_m, per_q_block, use_k_centering, mode in (
        (True, True, True, "chunked"),
        (True, False, False, "shard_major"),
        (False, False, False, "chunked"),
    ):
      with self.subTest(use_fixed_m=use_fixed_m, per_q_block=per_q_block, use_k_centering=use_k_centering, mode=mode):
        with mesh_u, nn_partitioning.axis_rules(rules):
          out_flat = attention_flax._ulysses_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh_u,
              axis_names_q=axis_names_q,
              axis_names_kv=axis_names_kv,
              flash_block_sizes=bs,
              use_custom_kernel=True,
              use_fixed_m=use_fixed_m,
              per_q_block=per_q_block,
              use_k_centering=use_k_centering,
              wan_ulysses_out_a2a="flat",
          )
          out_chunked = attention_flax._ulysses_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh_u,
              axis_names_q=axis_names_q,
              axis_names_kv=axis_names_kv,
              flash_block_sizes=bs,
              use_custom_kernel=True,
              use_fixed_m=use_fixed_m,
              per_q_block=per_q_block,
              use_k_centering=use_k_centering,
              wan_ulysses_out_a2a=mode,
          )
        self.assertEqual(out_chunked.shape, out_flat.shape)
        np.testing.assert_array_equal(np.asarray(out_chunked, jnp.float32), np.asarray(out_flat, jnp.float32))


if __name__ == "__main__":
  unittest.main()
