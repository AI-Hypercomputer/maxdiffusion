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

import unittest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
import numpy as np

from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline, calculate_shift, rescale_noise_cfg


class LTX2PipelineTest(unittest.TestCase):
  """Tests for LTX2Pipeline core logic (non-execution)."""

  def setUp(self):
    self.config = MagicMock()
    self.config.pretrained_model_name_or_path = "test_model"

  def test_calculate_shift(self):
    """Test shift calculation math."""
    # Test base condition
    shift = calculate_shift(256, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    self.assertAlmostEqual(shift, 0.5)

    # Test max condition
    shift = calculate_shift(4096, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    self.assertAlmostEqual(shift, 1.15)

    # Test midpoint
    mid_seq_len = (256 + 4096) / 2
    mid_shift = (0.5 + 1.15) / 2
    shift = calculate_shift(mid_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    self.assertAlmostEqual(shift, mid_shift)

  def test_rescale_noise_cfg(self):
    """Test rescaling noise cfg based on guidance rescale factor."""
    noise_cfg = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    noise_pred_text = jnp.array([[[1.0, 1.0], [1.0, 1.0]]])

    # with guidance_rescale = 0.0, output should be identical to noise_cfg
    rescaled_0 = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0)
    np.testing.assert_allclose(rescaled_0, noise_cfg, rtol=1e-5)

  def test_pipeline_init(self):
    """Test LTX2Pipeline initialization and property extraction."""
    mock_vae = MagicMock()
    mock_vae.spatial_compression_ratio = 8
    mock_vae.temporal_compression_ratio = 4

    mock_audio_vae = MagicMock()
    mock_audio_vae.mel_compression_ratio = 4
    mock_audio_vae.temporal_compression_ratio = 4
    mock_audio_vae.config.sample_rate = 24000
    mock_audio_vae.config.mel_hop_length = 256

    mock_transformer = MagicMock()
    mock_transformer.config.patch_size = 2
    mock_transformer.config.patch_size_t = 2

    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 512

    pipeline = LTX2Pipeline(
        scheduler=MagicMock(),
        vae=mock_vae,
        audio_vae=mock_audio_vae,
        text_encoder=MagicMock(),
        tokenizer=mock_tokenizer,
        connectors=MagicMock(),
        transformer=mock_transformer,
        vocoder=MagicMock(),
    )

    self.assertEqual(pipeline.vae_spatial_compression_ratio, 8)
    self.assertEqual(pipeline.vae_temporal_compression_ratio, 4)
    self.assertEqual(pipeline.audio_vae_mel_compression_ratio, 4)
    self.assertEqual(pipeline.audio_vae_temporal_compression_ratio, 4)
    self.assertEqual(pipeline.transformer_spatial_patch_size, 2)
    self.assertEqual(pipeline.transformer_temporal_patch_size, 2)
    self.assertEqual(pipeline.audio_sampling_rate, 24000)
    self.assertEqual(pipeline.audio_hop_length, 256)
    self.assertEqual(pipeline.tokenizer_max_length, 512)

  def test_check_inputs(self):
    """Test that check_inputs validates divisibility requirements."""
    pipeline = LTX2Pipeline(
        scheduler=MagicMock(),
        vae=MagicMock(),
        audio_vae=MagicMock(),
        text_encoder=MagicMock(),
        tokenizer=MagicMock(),
        connectors=MagicMock(),
        transformer=MagicMock(),
        vocoder=MagicMock(),
    )

    # Valid check shouldn't raise
    pipeline.check_inputs(prompt="test", height=64, width=64)

    # Invalid height should raise
    with self.assertRaises(ValueError):
      pipeline.check_inputs(prompt="test", height=63, width=64)

    # Invalid width should raise
    with self.assertRaises(ValueError):
        pipeline.check_inputs(prompt="test", height=64, width=63)

  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline._get_gemma_prompt_embeds")
  def test_encode_prompt(self, list_embed_mock):
    """Test conditional encoding of positive and negative prompts."""
    pipeline = LTX2Pipeline(
        scheduler=MagicMock(),
        vae=MagicMock(),
        audio_vae=MagicMock(),
        text_encoder=MagicMock(),
        tokenizer=MagicMock(),
        connectors=MagicMock(),
        transformer=MagicMock(),
        vocoder=MagicMock(),
    )

    prompt_embeds = jnp.zeros((1, 10, 10))
    prompt_attention_mask = jnp.ones((1, 10))
    neg_prompt_embeds = jnp.zeros((1, 10, 10))
    neg_prompt_attention_mask = jnp.ones((1, 10))

    # Mock return values for positive then negative prompt encoding
    list_embed_mock.side_effect = [
        (prompt_embeds, prompt_attention_mask),
        (neg_prompt_embeds, neg_prompt_attention_mask),
    ]

    p_e, p_a, n_e, n_a = pipeline.encode_prompt(
        prompt=["A cute cat"], negative_prompt=["ugly"], do_classifier_free_guidance=True
    )

    # Check mock calls
    self.assertEqual(list_embed_mock.call_count, 2)
    
    # Check returns
    np.testing.assert_array_equal(p_e, prompt_embeds)
    np.testing.assert_array_equal(p_a, prompt_attention_mask)
    np.testing.assert_array_equal(n_e, neg_prompt_embeds)
    np.testing.assert_array_equal(n_a, neg_prompt_attention_mask)

  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline._get_gemma_prompt_embeds")
  def test_encode_prompt_no_cfg(self, list_embed_mock):
    """Test encoding string prompt without classifier free guidance."""
    pipeline = LTX2Pipeline(
        scheduler=MagicMock(),
        vae=MagicMock(),
        audio_vae=MagicMock(),
        text_encoder=MagicMock(),
        tokenizer=MagicMock(),
        connectors=MagicMock(),
        transformer=MagicMock(),
        vocoder=MagicMock(),
    )

    prompt_embeds = jnp.zeros((1, 10, 10))
    prompt_attention_mask = jnp.ones((1, 10))

    list_embed_mock.return_value = (prompt_embeds, prompt_attention_mask)

    p_e, p_a, n_e, n_a = pipeline.encode_prompt(
        prompt="A cute cat", do_classifier_free_guidance=False
    )

    # We only expect one call
    self.assertEqual(list_embed_mock.call_count, 1)

    np.testing.assert_array_equal(p_e, prompt_embeds)
    np.testing.assert_array_equal(p_a, prompt_attention_mask)

    # Should be None since CFG is False
    self.assertIsNone(n_e)
    self.assertIsNone(n_a)

  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.load_transformer")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline._create_common_components")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.quantize_transformer")
  def test_load_and_init(self, mock_quantize, mock_create_common, mock_load_transformer):
    """Test that pipeline loading correctly wires all the dependencies down to __init__."""
    mock_config = MagicMock()
    mock_mesh = MagicMock()

    mock_common = {
        "vae": MagicMock(),
        "audio_vae": MagicMock(),
        "vocoder": MagicMock(),
        "devices_array": MagicMock(),
        "rngs": MagicMock(),
        "mesh": mock_mesh,
        "tokenizer": MagicMock(),
        "text_encoder": MagicMock(),
        "connectors": MagicMock(),
        "scheduler": MagicMock(),
    }
    mock_create_common.return_value = mock_common
    mock_transformer = MagicMock()
    mock_load_transformer.return_value = mock_transformer

    # Make quantize transformer pass-through the mock
    mock_quantize.return_value = mock_transformer

    pipeline, transformer = LTX2Pipeline._load_and_init(mock_config, None, vae_only=False, load_transformer=True)

    # Assert load_transformer was called with the components
    mock_load_transformer.assert_called_once_with(
        devices_array=mock_common["devices_array"],
        mesh=mock_common["mesh"],
        rngs=mock_common["rngs"],
        config=mock_config,
        restored_checkpoint=None,
    )

    mock_quantize.assert_called_once_with(mock_config, mock_transformer, pipeline, mock_mesh)

    self.assertEqual(pipeline.transformer, mock_transformer)
    self.assertEqual(pipeline.mesh, mock_mesh)
    self.assertEqual(pipeline.config, mock_config)


if __name__ == "__main__":
  unittest.main()
