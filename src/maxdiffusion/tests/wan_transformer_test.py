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
import pytest
import unittest
from unittest.mock import Mock, patch
from absl.testing import absltest
from flax import nnx
from jax.sharding import Mesh

from .. import pyconfig
from ..max_utils import (create_device_mesh, get_flash_block_sizes)
from ..models.wan.transformers.transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
    WanModel,
)
from ..models.embeddings_flax import NNXTimestepEmbedding, NNXPixArtAlphaTextProjection
from ..models.normalization_flax import FP32LayerNorm
from ..models.attention_flax import FlaxWanAttention
from maxdiffusion.pyconfig import HyperParameters
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    wan_rot_embed = WanRotaryPosEmbed(attention_head_dim=128, patch_size=[1, 2, 2], max_seq_len=1024)
    dummy_output = wan_rot_embed(dummy_hidden_states)
    assert dummy_output.shape == (1, 1, 75600, 64)

  def test_nnx_pixart_alpha_text_projection(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    dummy_caption = jnp.ones((1, 512, 4096))
    layer = NNXPixArtAlphaTextProjection(rngs=rngs, in_features=4096, hidden_size=5120)
    dummy_output = layer(dummy_caption)
    dummy_output.shape == (1, 512, 5120)

  def test_nnx_timestep_embedding(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    dummy_sample = jnp.ones((1, 256))
    layer = NNXTimestepEmbedding(rngs=rngs, in_channels=256, time_embed_dim=5120)
    dummy_output = layer(dummy_sample)
    assert dummy_output.shape == (1, 5120)

  def test_fp32_layer_norm(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch_size = 1
    dummy_hidden_states = jnp.ones((batch_size, 75600, 5120))
    # expected same output shape with same dtype
    layer = FP32LayerNorm(rngs=rngs, dim=5120, eps=1e-6, elementwise_affine=False)
    dummy_output = layer(dummy_hidden_states)
    assert dummy_output.shape == dummy_hidden_states.shape

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_wan_time_text_embedding(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch_size = 1
    dim = 5120
    time_freq_dim = 256
    time_proj_dim = 30720
    text_embed_dim = 4096
    layer = WanTimeTextImageEmbedding(
        rngs=rngs, dim=dim, time_freq_dim=time_freq_dim, time_proj_dim=time_proj_dim, text_embed_dim=text_embed_dim
    )

    dummy_timestep = jnp.ones(batch_size)

    encoder_hidden_states_shape = (batch_size, time_freq_dim * 2, text_embed_dim)
    dummy_encoder_hidden_states = jnp.ones(encoder_hidden_states_shape)
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = layer(
        dummy_timestep, dummy_encoder_hidden_states
    )
    assert temb.shape == (batch_size, dim)
    assert timestep_proj.shape == (batch_size, time_proj_dim)
    assert encoder_hidden_states.shape == (batch_size, time_freq_dim * 2, dim)

  def test_wan_block(self):
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

    dummy_encoder_hidden_states = jnp.ones((batch_size, 512, dim))

    dummy_temb = jnp.ones((batch_size, 6, dim))

    wan_block = WanTransformerBlock(
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
    )
    with mesh:
      dummy_output = wan_block(dummy_hidden_states, dummy_encoder_hidden_states, dummy_temb, dummy_rotary_emb)
    assert dummy_output.shape == dummy_hidden_states.shape

  def test_wan_attention(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config

    batch_size = 1
    channels = 16
    frames = 21
    height = 90
    width = 160
    hidden_states_shape = (batch_size, frames, height, width, channels)
    dummy_hidden_states = jnp.ones(hidden_states_shape)
    wan_rot_embed = WanRotaryPosEmbed(attention_head_dim=128, patch_size=[1, 2, 2], max_seq_len=1024)
    dummy_rotary_emb = wan_rot_embed(dummy_hidden_states)

    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    devices_array = create_device_mesh(config)

    flash_block_sizes = get_flash_block_sizes(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    batch_size = 1
    query_dim = 5120
    attention = FlaxWanAttention(
        rngs=rngs,
        query_dim=query_dim,
        heads=40,
        dim_head=128,
        attention_kernel="flash",
        mesh=mesh,
        flash_block_sizes=flash_block_sizes,
    )

    dummy_hidden_states_shape = (batch_size, 75600, query_dim)

    dummy_hidden_states = jnp.ones(dummy_hidden_states_shape)
    dummy_encoder_hidden_states = jnp.ones(dummy_hidden_states_shape)
    with mesh:
      dummy_output = attention(
          hidden_states=dummy_hidden_states, encoder_hidden_states=dummy_encoder_hidden_states, rotary_emb=dummy_rotary_emb
      )
    assert dummy_output.shape == dummy_hidden_states_shape

    # dot product
    try:
      attention = FlaxWanAttention(
          rngs=rngs,
          query_dim=query_dim,
          heads=40,
          dim_head=128,
          attention_kernel="dot_product",
          split_head_dim=True,
          mesh=mesh,
          flash_block_sizes=flash_block_sizes,
      )
    except NotImplementedError:
      pass

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_wan_model(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config

    batch_size = 1
    channels = 16
    frames = 1
    height = 90
    width = 160
    hidden_states_shape = (batch_size, channels, frames, height, width)
    dummy_hidden_states = jnp.ones(hidden_states_shape)

    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    devices_array = create_device_mesh(config)

    flash_block_sizes = get_flash_block_sizes(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    batch_size = 1
    num_layers = 1
    wan_model = WanModel(rngs=rngs, attention="flash", mesh=mesh, flash_block_sizes=flash_block_sizes, num_layers=num_layers)

    dummy_timestep = jnp.ones((batch_size))
    dummy_encoder_hidden_states = jnp.ones((batch_size, 512, 4096))
    with mesh:
      dummy_output = wan_model(
          hidden_states=dummy_hidden_states, timestep=dummy_timestep, encoder_hidden_states=dummy_encoder_hidden_states
      )
    assert dummy_output.shape == hidden_states_shape

  def test_get_qt_provider(self):
    """
    Tests the provider logic for all config branches.
    """
    # Case 1: Quantization disabled
    config_disabled = Mock(spec=HyperParameters)
    config_disabled.use_qwix_quantization = False
    self.assertIsNone(WanPipeline.get_qt_provider(config_disabled))

    # Case 2: Quantization enabled, type 'int8'
    config_int8 = Mock(spec=HyperParameters)
    config_int8.use_qwix_quantization = True
    config_int8.quantization = "int8"
    provider_int8 = WanPipeline.get_qt_provider(config_int8)
    self.assertIsNotNone(provider_int8)
    self.assertEqual(provider_int8.rules[0].kwargs['weight_qtype'], jnp.int8)

    # Case 3: Quantization enabled, type 'fp8'
    config_fp8 = Mock(spec=HyperParameters)
    config_fp8.use_qwix_quantization = True
    config_fp8.quantization = "fp8"
    provider_fp8 = WanPipeline.get_qt_provider(config_fp8)
    self.assertIsNotNone(provider_fp8)
    self.assertEqual(provider_fp8.rules[0].kwargs['weight_qtype'], jnp.float8_e4m3fn)

    # Case 4: Quantization enabled, type 'fp8_full'
    config_fp8_full = Mock(spec=HyperParameters)
    config_fp8_full.use_qwix_quantization = True
    config_fp8_full.quantization = "fp8_full"
    config_fp8_full.quantization_calibration_method = "absmax"
    provider_fp8_full = WanPipeline.get_qt_provider(config_fp8_full)
    self.assertIsNotNone(provider_fp8_full)
    self.assertEqual(provider_fp8_full.rules[0].kwargs['bwd_qtype'], jnp.float8_e5m2)

    # Case 5: Invalid quantization type
    config_invalid = Mock(spec=HyperParameters)
    config_invalid.use_qwix_quantization = True
    config_invalid.quantization = "invalid_type"
    self.assertIsNone(WanPipeline.get_qt_provider(config_invalid))

  # To test quantize_transformer, we patch its external dependencies
  @patch('maxdiffusion.pipelines.wan.wan_pipeline.qwix.quantize_model')
  @patch('maxdiffusion.pipelines.wan.wan_pipeline.get_dummy_wan_inputs')
  def test_quantize_transformer_enabled(self, mock_get_dummy_inputs, mock_quantize_model):
    """
    Tests that quantize_transformer calls qwix when quantization is enabled.
    """
    # Setup Mocks
    mock_config = Mock(spec=HyperParameters)
    mock_config.use_qwix_quantization = True
    mock_config.quantization = "fp8_full"
    mock_config.per_device_batch_size = 1

    mock_model = Mock(spec=WanModel)
    mock_pipeline = Mock()
    mock_mesh = Mock()

    # Mock the return values of dependencies
    mock_get_dummy_inputs.return_value = (Mock(), Mock(), Mock())
    mock_quantized_model_obj = Mock(spec=WanModel)
    mock_quantize_model.return_value = mock_quantized_model_obj

    # Call the method under test
    result = WanPipeline.quantize_transformer(mock_config, mock_model, mock_pipeline, mock_mesh)

    # Assertions
    mock_get_dummy_inputs.assert_called_once()
    mock_quantize_model.assert_called_once()
    # Check that the model returned is the new quantized model
    self.assertIs(result, mock_quantized_model_obj)

  @patch('maxdiffusion.pipelines.wan.wan_pipeline.qwix.quantize_model')
  def test_quantize_transformer_disabled(self, mock_quantize_model):
    """
    Tests that quantize_transformer is skipped when quantization is disabled.
    """
    # Setup Mocks
    mock_config = Mock(spec=HyperParameters)
    mock_config.use_qwix_quantization = False # Main condition for this test

    mock_model = Mock(spec=WanModel)

    # Call the method under test
    result = WanPipeline.quantize_transformer(mock_config, mock_model, Mock(), Mock())

    # Assertions
    mock_quantize_model.assert_not_called()
    # Check that the model returned is the original model instance
    self.assertIs(result, mock_model)


if __name__ == "__main__":
  absltest.main()
