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
from flax.linen import partitioning as nn_partitioning
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.ltx2.transformer_ltx2 import (
    LTX2VideoTransformerBlock,
    LTX2VideoTransformer3DModel,
    LTX2AdaLayerNormSingle,
    LTX2RotaryPosEmbed,
)
import flax
from unittest.mock import Mock, patch, MagicMock
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
from maxdiffusion.pyconfig import HyperParameters
import qwix

flax.config.update("flax_always_shard_variable", False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class LTX2TransformerTest(unittest.TestCase):

  def setUp(self):
    LTX2TransformerTest.dummy_data = {}
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "ltx2_video.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    self.config = config
    devices_array = create_device_mesh(config)
    self.mesh = Mesh(devices_array, config.mesh_axes)

    self.batch_size = 1
    self.num_frames = 4
    self.height = 32
    self.width = 32
    self.patch_size = 1
    self.patch_size_t = 1

    self.in_channels = 8
    self.out_channels = 8
    self.audio_in_channels = 4

    self.seq_len = (
        (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
    )

    self.dim = 1024
    self.num_heads = 8
    self.head_dim = 128
    self.cross_dim = 1024  # context dim

    self.audio_dim = 1024
    self.audio_num_heads = 8
    self.audio_head_dim = 128
    self.audio_cross_dim = 1024

  def test_ltx2_rope(self):
    """Tests LTX2RotaryPosEmbed output shapes and basic functionality."""
    dim = self.dim
    patch_size = self.patch_size
    patch_size_t = self.patch_size_t
    base_num_frames = 8
    base_height = 32
    base_width = 32

    # Video RoPE
    rope = LTX2RotaryPosEmbed(
        dim=dim,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        base_num_frames=base_num_frames,
        base_height=base_height,
        base_width=base_width,
        modality="video",
    )
    ids = jnp.ones((1, 3, 10))  # (B, Axes, S) for 3D coords
    cos, sin = rope(ids)

    # Check output shape
    self.assertEqual(cos.shape, (1, 10, dim))
    self.assertEqual(sin.shape, (1, 10, dim))

  def test_ltx2_rope_split(self):
    """Tests LTX2RotaryPosEmbed with rope_type='split'."""
    dim = self.dim
    patch_size = self.patch_size
    patch_size_t = self.patch_size_t
    base_num_frames = 8
    base_height = 32
    base_width = 32

    # Video RoPE Split
    rope = LTX2RotaryPosEmbed(
        dim=dim,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        base_num_frames=base_num_frames,
        base_height=base_height,
        base_width=base_width,
        modality="video",
        rope_type="split",
    )
    ids = jnp.ones((1, 3, 10))  # (B, Axes, S)
    cos, sin = rope(ids)

    # Check output shape
    # Split RoPE returns [B, H, S, D//2]
    # dim=1024, heads=32 => head_dim=32 => D//2 = 16
    self.assertEqual(cos.shape, (1, 32, 10, 16))
    self.assertEqual(sin.shape, (1, 32, 10, 16))

  def test_ltx2_ada_layer_norm_single(self):
    """Tests LTX2AdaLayerNormSingle initialization and execution."""
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    embedding_dim = self.dim

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      layer = LTX2AdaLayerNormSingle(
          rngs=rngs, embedding_dim=embedding_dim, num_mod_params=6, use_additional_conditions=False  # Default
      )

      timestep = jnp.array([1.0])
      batch_size = self.batch_size

      # Forward
      output, embedded_timestep = layer(timestep)

      # Expected output shape: (B, num_mod_params * embedding_dim)
      # embedded_timestep shape: (B, embedding_dim)
      self.assertEqual(output.shape, (batch_size, 6 * embedding_dim))
      self.assertEqual(embedded_timestep.shape, (batch_size, embedding_dim))

  def test_ltx2_transformer_block(self):
    """Tests LTX2VideoTransformerBlock with video and audio inputs."""
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    dim = self.dim
    audio_dim = self.audio_dim
    cross_attention_dim = self.cross_dim
    audio_cross_attention_dim = self.audio_cross_dim

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = LTX2VideoTransformerBlock(
          rngs=rngs,
          dim=dim,
          num_attention_heads=self.num_heads,
          attention_head_dim=self.head_dim,
          cross_attention_dim=cross_attention_dim,
          audio_dim=audio_dim,
          audio_num_attention_heads=self.audio_num_heads,
          audio_attention_head_dim=self.audio_head_dim,
          audio_cross_attention_dim=audio_cross_attention_dim,
          mesh=self.mesh,
      )

      batch_size = self.batch_size
      seq_len = self.seq_len
      audio_seq_len = 128  # Matching parity test

      hidden_states = jnp.zeros((batch_size, seq_len, dim))
      audio_hidden_states = jnp.zeros((batch_size, audio_seq_len, audio_dim))
      encoder_hidden_states = jnp.zeros((batch_size, 128, cross_attention_dim))
      audio_encoder_hidden_states = jnp.zeros((batch_size, 128, audio_cross_attention_dim))

      # Mock modulation parameters
      # sizes based on `transformer_ltx2.py` logic
      temb_dim = 6 * dim  # 6 params * dim
      temb = jnp.zeros((batch_size, temb_dim))
      temb_audio = jnp.zeros((batch_size, 6 * audio_dim))

      temb_ca_scale_shift = jnp.zeros((batch_size, 4 * dim))
      temb_ca_audio_scale_shift = jnp.zeros((batch_size, 4 * audio_dim))
      temb_ca_gate = jnp.zeros((batch_size, 1 * dim))
      temb_ca_audio_gate = jnp.zeros((batch_size, 1 * audio_dim))

      output_hidden, output_audio = block(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          temb=temb,
          temb_audio=temb_audio,
          temb_ca_scale_shift=temb_ca_scale_shift,
          temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
          temb_ca_gate=temb_ca_gate,
          temb_ca_audio_gate=temb_ca_audio_gate,
      )

      self.assertEqual(output_hidden.shape, hidden_states.shape)
      self.assertEqual(output_audio.shape, audio_hidden_states.shape)

  def test_ltx2_transformer_model(self):
    """Tests LTX2VideoTransformer3DModel full forward pass."""
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    in_channels = self.in_channels
    out_channels = self.out_channels
    audio_in_channels = self.audio_in_channels

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = LTX2VideoTransformer3DModel(
          rngs=rngs,
          in_channels=in_channels,
          out_channels=out_channels,
          patch_size=self.patch_size,
          patch_size_t=self.patch_size_t,
          num_attention_heads=self.num_heads,
          attention_head_dim=self.head_dim,
          cross_attention_dim=self.cross_dim,
          caption_channels=32,
          audio_in_channels=audio_in_channels,
          audio_out_channels=audio_in_channels,
          audio_num_attention_heads=self.audio_num_heads,
          audio_attention_head_dim=self.audio_head_dim,
          audio_cross_attention_dim=self.audio_cross_dim,
          num_layers=1,
          mesh=self.mesh,
          attention_kernel="dot_product",
      )

      batch_size = self.batch_size
      seq_len = self.seq_len
      audio_seq_len = 128

      hidden_states = jnp.zeros((batch_size, seq_len, in_channels))
      audio_hidden_states = jnp.zeros((batch_size, audio_seq_len, audio_in_channels))

      timestep = jnp.array([1.0])
      encoder_hidden_states = jnp.zeros((batch_size, 128, 32))  # (B, L, D) match caption_channels
      audio_encoder_hidden_states = jnp.zeros((batch_size, 128, 32))

      encoder_attention_mask = jnp.ones((batch_size, 128))
      audio_encoder_attention_mask = jnp.ones((batch_size, 128))

      output = model(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          timestep=timestep,
          num_frames=self.num_frames,
          height=self.height,
          width=self.width,
          audio_num_frames=audio_seq_len,
          encoder_attention_mask=encoder_attention_mask,
          audio_encoder_attention_mask=audio_encoder_attention_mask,
          return_dict=True,
      )

      self.assertEqual(output["sample"].shape, (batch_size, seq_len, out_channels))
      self.assertEqual(output["audio_sample"].shape, (batch_size, audio_seq_len, audio_in_channels))

  def test_get_qt_provider(self):
    config = Mock(spec=HyperParameters)

    # Test disabled
    config.use_qwix_quantization = False
    self.assertIsNone(LTX2Pipeline.get_qt_provider(config))

    # Test int8
    config.use_qwix_quantization = True
    config.quantization = "int8"
    config.qwix_module_path = ".*"
    provider = LTX2Pipeline.get_qt_provider(config)
    self.assertIsNotNone(provider)

    # Test fp8
    config.quantization = "fp8"
    # Mocking calibration method attributes which might be accessed
    config.weight_quantization_calibration_method = "max"
    config.act_quantization_calibration_method = "max"
    config.bwd_quantization_calibration_method = "max"
    provider = LTX2Pipeline.get_qt_provider(config)
    self.assertIsNotNone(provider)

    # Test fp8_full
    config.quantization = "fp8_full"
    provider = LTX2Pipeline.get_qt_provider(config)
    self.assertIsNotNone(provider)

  def get_dummy_inputs(self, config):
    batch_size = config.global_batch_size_to_train_on
    num_tokens = 256
    in_channels = 128
    caption_channels = 4096

    hidden_states = jnp.ones((batch_size, num_tokens, in_channels), dtype=jnp.float32)
    indices_grid = jnp.ones((batch_size, 3, num_tokens), dtype=jnp.float32)
    encoder_hidden_states = jnp.ones((batch_size, 128, caption_channels), dtype=jnp.float32)
    timestep = jnp.ones((batch_size, 256), dtype=jnp.float32)
    class_labels = None
    cross_attention_kwargs = None
    segment_ids = jnp.ones((batch_size, 256), dtype=jnp.int32)
    encoder_attention_segment_ids = jnp.ones((batch_size, 128), dtype=jnp.int32)

    return (
        hidden_states,
        indices_grid,
        encoder_hidden_states,
        timestep,
        class_labels,
        cross_attention_kwargs,
        segment_ids,
        encoder_attention_segment_ids,
    )

  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.qwix.quantize_model")
  def test_quantize_transformer(self, mock_quantize_model):
    config = Mock(spec=HyperParameters)
    config.use_qwix_quantization = True
    config.quantization = "int8"
    config.qwix_module_path = ".*"
    config.global_batch_size_to_train_on = 1

    model = Mock()
    pipeline = Mock()
    mesh = MagicMock()
    mesh.__enter__.return_value = None
    mesh.__exit__.return_value = None

    mock_quantized_model = Mock()
    mock_quantize_model.return_value = mock_quantized_model

    dummy_inputs = self.get_dummy_inputs(config)
    result = LTX2Pipeline.quantize_transformer(config, model, pipeline, mesh, dummy_inputs)

    self.assertEqual(result, mock_quantized_model)
    mock_quantize_model.assert_called_once()

    # Check arguments passed to quantize_model
    args, _ = mock_quantize_model.call_args
    self.assertEqual(args[0], model)
    # args[1] is rules
    # args[2:] are dummy inputs
    self.assertTrue(len(args) > 2)
