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

from maxdiffusion import pyconfig, wan_runtime_options
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.attention_flax import TokenPadding
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanSeqPadTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    wan_runtime_options.reset()
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml")], unittest=True)
    self.config = pyconfig.config
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)

  def tearDown(self):
    wan_runtime_options.reset()
    super().tearDown()

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
          attention_config={"ulysses_shards": 4},
      )
      with mock.patch.dict(os.environ, {"WAN_SEQ_PAD": "lane"}):
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
          attention_config={"ulysses_shards": 2},
      )
      with mock.patch.dict(os.environ, {"WAN_SEQ_PAD": "lane"}):
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
    with mock.patch.dict(os.environ, {"WAN_SEQ_PAD": "invalid"}):
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
      self.assertIsNone(model_mismatch_u._get_token_padding(75600))

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
      self.assertIsNone(model_ring_invalid._get_token_padding(75600))


if __name__ == "__main__":
  unittest.main()
