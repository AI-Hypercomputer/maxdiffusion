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

"""Tests for token sequence padding and lane alignment (`wan_seq_pad`)."""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.attention_flax import TokenPadding
from maxdiffusion.models.wan.transformers import transformer_wan
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# Kernel numerics grids and multi-device tests are skipped in CI; see
# end_to_end/tpu/run_wan_stack_tests.sh.
_SKIP_IN_GITHUB_ACTIONS = unittest.skipIf(
    IN_GITHUB_ACTIONS, "TPU kernel / multi-device test, skipped in GitHub Actions; run end_to_end/tpu/run_wan_stack_tests.sh"
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanSeqPadTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml")], unittest=True)
    self.config = pyconfig.config
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)

  def test_token_padding_inversion(self):
    pad = TokenPadding(real_len=37800, padded_len=37888, num_segments=2)
    x = jax.random.normal(jax.random.key(0), (1, 75600, 64), dtype=jnp.float32)
    padded = WanModel._pad_tokens(x, pad)
    self.assertEqual(padded.shape, (1, 75776, 64))
    unpadded = WanModel._unpad_tokens(padded, pad)
    self.assertEqual(unpadded.shape, (1, 75600, 64))
    np.testing.assert_array_equal(np.asarray(unpadded), np.asarray(x))

  def test_rotary_emb_padding(self):
    pad = TokenPadding(real_len=75600, padded_len=75776, num_segments=1)
    r = jax.random.normal(jax.random.key(1), (1, 1, 75600, 64), dtype=jnp.float32)
    padded_r = WanModel._pad_rotary_emb(r, pad)
    self.assertEqual(padded_r.shape, (1, 1, 75776, 64))
    np.testing.assert_array_equal(np.asarray(padded_r[:, :, :75600, :]), np.asarray(r))
    np.testing.assert_array_equal(np.asarray(padded_r[:, :, 75600:, :]), 0.0)

  def test_get_token_padding_v6e_and_tpu7x(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      # v6e: ulysses=4, ring=1
      model_v6e = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m",
          attention_config={"ulysses_shards": 4, "wan_seq_pad": "lane"},
      )
      pad_v6e = model_v6e._get_token_padding(75600)
      self.assertIsNotNone(pad_v6e)
      self.assertEqual(pad_v6e.real_len, 75600)
      self.assertEqual(pad_v6e.padded_len, 75776)
      self.assertEqual(pad_v6e.num_segments, 1)

      # tpu7x: ulysses=2, ring=2
      model_7x = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_ring_custom_fixed_m",
          attention_config={"ulysses_shards": 2, "wan_seq_pad": "lane"},
      )
      pad_7x = model_7x._get_token_padding(75600)
      self.assertIsNotNone(pad_7x)
      self.assertEqual(pad_7x.real_len, 37800)
      self.assertEqual(pad_7x.padded_len, 37888)
      self.assertEqual(pad_7x.num_segments, 2)
      self.assertEqual(pad_7x.total_len, 75776)

  def test_per_token_t_disables_token_padding(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"ulysses_shards": 4, "wan_seq_pad": "lane"},
      )
    # With per_token_t=True (TI2V), token padding must gracefully return None without raising NotImplementedError
    pad = model._get_token_padding(75600, per_token_t=True)
    self.assertIsNone(pad)

  def test_unsupported_attention_kernel_disables_token_padding(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="dot_product",
          attention_config={"wan_seq_pad": "lane"},
      )
    pad = model._get_token_padding(75600)
    self.assertIsNone(pad)

  def test_rejects_unknown_mode(self):
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=2,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          attention="dot_product",
      )
      # An invalid mode is rejected when the model is built ...
      with self.assertRaises(ValueError):
        WanModel(
            rngs=nnx.Rngs(0),
            num_attention_heads=2,
            attention_head_dim=64,
            in_channels=4,
            out_channels=4,
            text_dim=32,
            freq_dim=32,
            ffn_dim=128,
            num_layers=1,
            attention="dot_product",
            attention_config={"wan_seq_pad": "invalid"},
        )
    # ... and by the padding planner itself.
    model.seq_pad_mode = "invalid"
    with self.assertRaises(ValueError):
      model._get_token_padding(75600)

  def test_svg_attention_disables_token_padding(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"ulysses_shards": 4, "wan_seq_pad": "lane", "use_svg_attention": True},
      )
    pad = model._get_token_padding(75600)
    self.assertIsNone(pad)

  def test_short_sequence_disables_token_padding(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          flash_min_seq_length=8192,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"ulysses_shards": 4, "wan_seq_pad": "lane"},
      )
    # 6000 > default 4096, so this verifies the constructor's flash_min_seq_length=8192 is respected.
    pad = model._get_token_padding(6000)
    self.assertIsNone(pad)

  def test_memory_efficient_attention_disables_token_padding(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"ulysses_shards": 4, "wan_seq_pad": "lane", "use_memory_efficient_attention": True},
      )
    pad = model._get_token_padding(75600)
    self.assertIsNone(pad)

  def test_non_ring_ulysses_uses_context_shards_when_unset_and_rejects_mismatch(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      # Default ulysses_shards=-1 on non-ring ulysses_custom* uses full context=4
      model_default_u = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"wan_seq_pad": "lane"},
      )
      pad = model_default_u._get_token_padding(75600)
      self.assertIsNotNone(pad)
      self.assertEqual(pad.padded_len, 75776)

      # Explicit mismatched ulysses_shards=2 on non-ring ulysses_custom* returns None
      model_mismatch_u = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"ulysses_shards": 2, "wan_seq_pad": "lane"},
      )
      with mock.patch.object(transformer_wan, "_warn_once") as warn:
        self.assertIsNone(model_mismatch_u._get_token_padding(75600))
      warn.assert_called_once()
      self.assertEqual(warn.call_args.args[0], "wan_seq_pad_ulysses_shards_mismatch")
      self.assertIn("ulysses_shards=2", warn.call_args.args[1])

      # Ring attention with unset (-1) or non-dividing ulysses_shards (3 on context=4) returns None
      model_ring_invalid = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_ring_custom_fixed_m",
          attention_config={"ulysses_shards": 3, "wan_seq_pad": "lane"},
      )
      with mock.patch.object(transformer_wan, "_warn_once") as warn:
        self.assertIsNone(model_ring_invalid._get_token_padding(75600))
      warn.assert_called_once()
      self.assertEqual(warn.call_args.args[0], "wan_seq_pad_ring_ulysses_shards")
      self.assertIn("ulysses_shards=3", warn.call_args.args[1])

  def test_indivisible_seq_len_warns_and_aligned_seq_len_is_silent(self):
    mock_mesh = mock.MagicMock()
    mock_mesh.shape = {"context": 4}
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=4,
          attention_head_dim=64,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=128,
          num_layers=1,
          mesh=mock_mesh,
          attention="ulysses_custom_fixed_m_per_q_block",
          attention_config={"wan_seq_pad": "lane"},
      )
    with mock.patch.object(transformer_wan, "_warn_once") as warn:
      self.assertIsNone(model._get_token_padding(75601))  # 75601 % 4 context shards != 0
    warn.assert_called_once()
    self.assertEqual(warn.call_args.args[0], "wan_seq_pad_indivisible_seq")
    self.assertIn("seq_len=75601", warn.call_args.args[1])

    # 65536 tokens over 4 shards is 16384 per shard, already a multiple of 128: a legitimate no-op.
    with mock.patch.object(transformer_wan, "_warn_once") as warn:
      self.assertIsNone(model._get_token_padding(65536))
    warn.assert_not_called()

  def test_unsupported_kernel_with_active_token_padding_raises(self):
    from maxdiffusion.models import attention_flax

    pad = TokenPadding(real_len=128, padded_len=256, num_segments=1)
    q = jnp.zeros((1, 256, 2 * 128), dtype=jnp.bfloat16)
    with attention_flax.self_attention_token_padding(pad):
      with self.assertRaisesRegex(NotImplementedError, "self_attention_token_padding"):
        attention_flax._apply_attention_dot(
            q,
            q,
            q,
            jnp.bfloat16,
            heads=2,
            dim_head=128,
            scale=1.0,
            split_head_dim=True,
            float32_qk_product=False,
            use_memory_efficient_attention=False,
        )

  _BS = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128, "block_kv_compute_in": 128}

  @staticmethod
  def _masking_rules_and_names():
    from maxdiffusion.models import attention_flax

    rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
        (attention_flax.LENGTH, "context"),
        (attention_flax.HEAD, None),
    )
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
    return rules, axis_names_q, axis_names_kv

  @staticmethod
  def _with_sentinel_pad(x, num_segments, real_len, padded_len, value=500.0):
    b, _, f = x.shape
    x_seg = x.reshape(b, num_segments, real_len, f)
    sent = jnp.full((b, num_segments, padded_len - real_len, f), value, dtype=x.dtype)
    return jnp.concatenate([x_seg, sent], axis=2).reshape(b, num_segments * padded_len, f)

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_custom_masks_tail_pad_tokens(self):
    """Pure Ulysses (U=2) with sentinel-valued tail pad tokens matches the unpadded run.

    Covers the production pad path: sublane-unaligned real_len (300 % 8 != 0,
    padded K/V) and aligned real_len (304, `kv_pad_size=1`, i.e. K/V passed
    unpadded and the kernel's ragged tail read), each with the flat and the
    shard-major output exchange, with sentinels in Q, K and V pad rows.
    """
    from maxdiffusion.models import attention_flax

    if len(jax.devices()) < 2:
      self.skipTest("needs >= 2 devices for Ulysses attention")
    rules, axis_names_q, axis_names_kv = self._masking_rules_and_names()
    b, heads, d, padded_len = 1, 4, 128, 512
    mesh_2 = Mesh(np.array(jax.devices()[:2]).reshape(1, 1, 2, 1), ("data", "fsdp", "context", "tensor"))

    for real_len in (300, 304):
      q_real = jax.random.normal(jax.random.PRNGKey(100), (b, real_len, heads * d), jnp.bfloat16)
      k_real = jax.random.normal(jax.random.PRNGKey(101), (b, real_len, heads * d), jnp.bfloat16) * 0.08
      v_real = jax.random.normal(jax.random.PRNGKey(102), (b, real_len, heads * d), jnp.bfloat16)
      pad_u = TokenPadding(real_len=real_len, padded_len=padded_len, num_segments=1)
      q_pad, k_pad, v_pad = (self._with_sentinel_pad(x, 1, real_len, padded_len) for x in (q_real, k_real, v_real))

      for out_a2a in ("flat", "shard_major"):
        for use_fixed_m, per_q_block in ((True, True), (True, False), (False, False)):
          with self.subTest(real_len=real_len, out_a2a=out_a2a, use_fixed_m=use_fixed_m, per_q_block=per_q_block):

            def run(q, k, v, out_a2a=out_a2a, use_fixed_m=use_fixed_m, per_q_block=per_q_block):
              return attention_flax._ulysses_attention(
                  q,
                  k,
                  v,
                  heads=heads,
                  mesh=mesh_2,
                  axis_names_q=axis_names_q,
                  axis_names_kv=axis_names_kv,
                  flash_block_sizes=self._BS,
                  use_custom_kernel=True,
                  use_fixed_m=use_fixed_m,
                  per_q_block=per_q_block,
                  use_k_centering=True,
                  wan_ulysses_out_a2a=out_a2a,
              )

            with mesh_2, nn_partitioning.axis_rules(rules):
              out_unpadded = run(q_real, k_real, v_real)
              with attention_flax.self_attention_token_padding(pad_u):
                out_padded = run(q_pad, k_pad, v_pad)
            np.testing.assert_allclose(
                np.asarray(WanModel._unpad_tokens(out_padded, pad_u), np.float32),
                np.asarray(out_unpadded, np.float32),
                atol=2e-2,
                rtol=2e-2,
            )

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_fixed_m_metadata_ignores_q_pad_rows(self):
    """Sentinel Q pad rows must not push pure Ulysses off the uniform fixed-m path."""
    from maxdiffusion.models import attention_flax

    b, heads, d, real_len, padded_len, bq = 1, 2, 128, 300, 512, 128
    q = jax.random.normal(jax.random.PRNGKey(1), (b, heads, real_len, d), jnp.float32) * 0.1
    k = jax.random.normal(jax.random.PRNGKey(2), (b, heads, real_len, d), jnp.float32) * 0.1
    v = jax.random.normal(jax.random.PRNGKey(3), (b, heads, real_len, d), jnp.float32)
    q_pad = jnp.concatenate([q, jnp.full((b, heads, padded_len - real_len, d), 500.0)], axis=2).astype(jnp.bfloat16)
    k, v = k.astype(jnp.bfloat16), v.astype(jnp.bfloat16)
    for per_q_block in (True, False):
      with self.subTest(per_q_block=per_q_block):
        _, all_fixed_unmasked = attention_flax._compute_fixed_m_metadata(
            q_pad, k, block_q=bq, per_q_block=per_q_block, value=v
        )
        mk, all_fixed = attention_flax._compute_fixed_m_metadata(
            q_pad, k, block_q=bq, per_q_block=per_q_block, value=v, q_valid_len=real_len
        )
        self.assertFalse(bool(all_fixed_unmasked))  # the sentinels alone would force the online path
        self.assertTrue(bool(all_fixed))
        self.assertTrue(bool(jnp.all(mk[:, 1] > 0.5)))
    # 5-D shard-major layout: global position = shard * shard_len + i.
    q5 = q_pad.reshape(b, heads, 2, padded_len // 2, d).transpose(0, 2, 1, 3, 4)
    _, all_fixed5 = attention_flax._compute_fixed_m_metadata(q5, k, block_q=bq, value=v, q_valid_len=real_len)
    self.assertTrue(bool(all_fixed5))

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_ring_custom_masks_tail_pad_tokens(self):
    """2D Ulysses x Ring (U=2, R=2, two pad segments) with sentinel pad tokens matches the unpadded run.

    Runs with K-centering on and off (off is the ring default and the tpu7x
    production setting).
    """
    from maxdiffusion.models import attention_flax

    if len(jax.devices()) < 4:
      self.skipTest("needs >= 4 devices for U=2 x R=2 (XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU)")
    rules, axis_names_q, axis_names_kv = self._masking_rules_and_names()
    b, heads, d, real_len, padded_len = 1, 4, 128, 300, 512
    mesh_4 = Mesh(np.array(jax.devices()[:4]).reshape(1, 1, 4, 1), ("data", "fsdp", "context", "tensor"))
    total_real = 2 * real_len
    q_real = jax.random.normal(jax.random.PRNGKey(200), (b, total_real, heads * d), jnp.bfloat16)
    k_real = jax.random.normal(jax.random.PRNGKey(201), (b, total_real, heads * d), jnp.bfloat16) * 0.08
    v_real = jax.random.normal(jax.random.PRNGKey(202), (b, total_real, heads * d), jnp.bfloat16)
    pad_r = TokenPadding(real_len=real_len, padded_len=padded_len, num_segments=2)
    q_pad, k_pad, v_pad = (self._with_sentinel_pad(x, 2, real_len, padded_len) for x in (q_real, k_real, v_real))

    for use_k_centering in (True, False):
      for use_fixed_m, per_q_block in ((True, True), (True, False), (False, False)):
        with self.subTest(use_k_centering=use_k_centering, use_fixed_m=use_fixed_m, per_q_block=per_q_block):

          def run(q, k, v, use_k_centering=use_k_centering, use_fixed_m=use_fixed_m, per_q_block=per_q_block):
            return attention_flax._ulysses_ring_custom_attention(
                q,
                k,
                v,
                heads=heads,
                mesh=mesh_4,
                axis_names_q=axis_names_q,
                axis_names_kv=axis_names_kv,
                flash_block_sizes=self._BS,
                ulysses_shards=2,
                use_fixed_m=use_fixed_m,
                per_q_block=per_q_block,
                use_k_centering=use_k_centering,
            )

          with mesh_4, nn_partitioning.axis_rules(rules):
            out_unpadded = run(q_real, k_real, v_real)
            with attention_flax.self_attention_token_padding(pad_r):
              out_padded = run(q_pad, k_pad, v_pad)
          np.testing.assert_allclose(
              np.asarray(WanModel._unpad_tokens(out_padded, pad_r), np.float32),
              np.asarray(out_unpadded, np.float32),
              atol=2e-2,
              rtol=2e-2,
          )

  @_SKIP_IN_GITHUB_ACTIONS
  def test_ulysses_ring_custom_aligned_real_len_shard_major_masks_tail_pad_tokens(self):
    """The tpu7x production pad path, scaled down: U=2 x R=2, aligned real_len, shard-major exchange.

    real_len=304 is a multiple of 8, so K/V go to the kernel unpadded
    (`kv_pad_size=1`) and the kernel reads the ragged tail; each 512-token
    segment splits into 256-row ring shards, a multiple of block_q, so
    `wan_ulysses_out_a2a="chunked"` takes the shard-major layout. K-centering
    is off, as on tpu7x. Sentinels fill the Q, K and V pad rows.
    """
    from maxdiffusion.models import attention_flax

    if len(jax.devices()) < 4:
      self.skipTest("needs >= 4 devices for U=2 x R=2 (XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU)")
    rules, axis_names_q, axis_names_kv = self._masking_rules_and_names()
    b, heads, d, real_len, padded_len = 1, 4, 128, 304, 512
    mesh_4 = Mesh(np.array(jax.devices()[:4]).reshape(1, 1, 4, 1), ("data", "fsdp", "context", "tensor"))
    total_real = 2 * real_len
    q_real = jax.random.normal(jax.random.PRNGKey(300), (b, total_real, heads * d), jnp.bfloat16)
    k_real = jax.random.normal(jax.random.PRNGKey(301), (b, total_real, heads * d), jnp.bfloat16) * 0.08
    v_real = jax.random.normal(jax.random.PRNGKey(302), (b, total_real, heads * d), jnp.bfloat16)
    pad_r = TokenPadding(real_len=real_len, padded_len=padded_len, num_segments=2)
    q_pad, k_pad, v_pad = (self._with_sentinel_pad(x, 2, real_len, padded_len) for x in (q_real, k_real, v_real))

    for use_fixed_m, per_q_block in ((True, False), (True, True)):
      with self.subTest(use_fixed_m=use_fixed_m, per_q_block=per_q_block):

        def run(q, k, v, out_a2a, use_fixed_m=use_fixed_m, per_q_block=per_q_block):
          return attention_flax._ulysses_ring_custom_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh_4,
              axis_names_q=axis_names_q,
              axis_names_kv=axis_names_kv,
              flash_block_sizes=self._BS,
              ulysses_shards=2,
              use_fixed_m=use_fixed_m,
              per_q_block=per_q_block,
              use_k_centering=False,
              wan_ulysses_out_a2a=out_a2a,
          )

        ring_mod = attention_flax.tokamax_ring_attention_kernel
        with mesh_4, nn_partitioning.axis_rules(rules):
          out_unpadded = run(q_real, k_real, v_real, "flat")
          with attention_flax.self_attention_token_padding(pad_r):
            with mock.patch.object(ring_mod, "make_custom_ring_attention", wraps=ring_mod.make_custom_ring_attention) as spy:
              out_padded = run(q_pad, k_pad, v_pad, "chunked")
        # The shard-major layout was taken (out_num_shards == U), not the flat fallback.
        self.assertEqual(spy.call_args.kwargs["out_num_shards"], 2)
        np.testing.assert_allclose(
            np.asarray(WanModel._unpad_tokens(out_padded, pad_r), np.float32),
            np.asarray(out_unpadded, np.float32),
            atol=2e-2,
            rtol=2e-2,
        )

  @_SKIP_IN_GITHUB_ACTIONS
  def test_wan_model_end_to_end_lane_padding_matches_off(self):
    """End-to-end WanModel forward with wan_seq_pad='lane' matches 'off'."""
    if len(jax.devices()) < 2:
      self.skipTest("needs >= 2 devices")
    mesh_2 = Mesh(np.array(jax.devices()[:2]).reshape(1, 1, 2, 1), ("data", "fsdp", "context", "tensor"))
    bs = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128, "block_kv_compute_in": 128}
    with mesh_2, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      # num_frames=5, height=16, width=20 with patch_size=(1, 2, 2) -> 5 * 8 * 10 = 400 tokens
      # Across 2 context shards: 200 tokens/shard -> padded to 256 tokens/shard (512 total).
      common_kwargs = {
          "patch_size": (1, 2, 2),
          "num_attention_heads": 4,
          "attention_head_dim": 128,
          "in_channels": 4,
          "out_channels": 4,
          "text_dim": 64,
          "freq_dim": 32,
          "ffn_dim": 128,
          "num_layers": 1,
          "scan_layers": False,
          "mesh": mesh_2,
          "dtype": jnp.bfloat16,
          "weights_dtype": jnp.float32,
          "flash_min_seq_length": 64,
          "flash_block_sizes": bs,
          "attention": "ulysses_custom_fixed_m_per_q_block",
      }
      model_off = WanModel(
          rngs=nnx.Rngs(42),
          attention_config={"ulysses_shards": 2, "use_base2_exp": True, "wan_seq_pad": "off"},
          **common_kwargs,
      )
      model_lane = WanModel(
          rngs=nnx.Rngs(42),
          attention_config={"ulysses_shards": 2, "use_base2_exp": True, "wan_seq_pad": "lane"},
          **common_kwargs,
      )
      pad = model_lane._get_token_padding(400)
      self.assertIsNotNone(pad)
      self.assertEqual(pad.real_len, 400)
      self.assertEqual(pad.padded_len, 512)

      latents = jax.random.normal(jax.random.PRNGKey(300), (1, 4, 5, 16, 20), jnp.bfloat16)
      timestep = jnp.array([500], dtype=jnp.int32)
      enc = jax.random.normal(jax.random.PRNGKey(301), (1, 16, 64), jnp.bfloat16)
      out_off = model_off(latents, timestep, enc, deterministic=True)
      out_lane = model_lane(latents, timestep, enc, deterministic=True)
    self.assertEqual(out_lane.shape, out_off.shape)
    np.testing.assert_allclose(np.asarray(out_lane, np.float32), np.asarray(out_off, np.float32), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
  unittest.main()
