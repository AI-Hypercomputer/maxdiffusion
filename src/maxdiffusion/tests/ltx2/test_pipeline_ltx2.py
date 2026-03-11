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


  def test_pack_unpack_latents(self):
    """Test video latents packing and unpacking math."""
    latents = jnp.arange(1 * 8 * 4 * 16 * 16).reshape(1, 8, 4, 16, 16).astype(jnp.float32)
    packed = LTX2Pipeline._pack_latents(latents, patch_size=2, patch_size_t=2)
    # 4//2 = 2 frames, 16//2 = 8 height, 16//2 = 8 width -> 2 * 8 * 8 = 128 seq_len
    # Channels 8, * patch_t 2 * patch_h 2 * patch_w 2 = 8 * 8 = 64
    self.assertEqual(packed.shape, (1, 128, 64))
    
    unpacked = LTX2Pipeline._unpack_latents(packed, num_frames=4, height=16, width=16, patch_size=2, patch_size_t=2)
    self.assertEqual(unpacked.shape, latents.shape)
    np.testing.assert_array_equal(unpacked, latents)

  def test_normalize_denormalize_latents(self):
    """Test normalization and denormalization of video latents."""
    latents = jnp.ones((1, 8, 4, 16, 16))
    mean = jnp.ones((8,)) * 0.5
    std = jnp.ones((8,)) * 0.2
    
    normalized = LTX2Pipeline._normalize_latents(latents, mean, std, scaling_factor=1.0)
    # (1 - 0.5)/0.2 = 2.5
    np.testing.assert_allclose(normalized, 2.5 * jnp.ones((1, 8, 4, 16, 16)), rtol=1e-5)
    
    denormalized = LTX2Pipeline._denormalize_latents(normalized, mean, std, scaling_factor=1.0)
    np.testing.assert_allclose(denormalized, latents, rtol=1e-5)

  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.jax.jit", lambda f, *args, **kwargs: f)
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.nnx.split")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.nnx.merge")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.transformer_forward_pass")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.retrieve_timesteps")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.encode_prompt")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.prepare_latents")
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.prepare_audio_latents")
  def test_call_method(
      self,
      mock_prepare_audio,
      mock_prepare_video,
      mock_encode,
      mock_retrieve,
      mock_forward,
      mock_merge,
      mock_split,
  ):
    """Test the core denoising loop execution structure."""
    
    # Mock return values for methods called directly on pipeline (before transformer)
    mock_encode.return_value = (
        jnp.zeros((1, 10, 32)), jnp.ones((1, 10)), jnp.zeros((1, 10, 32)), jnp.ones((1, 10))
    )
    # latent_num_frames = (9-1)//4 + 1 = 3
    # latent_height = 64//8 = 8, latent_width = 8
    # with patch_size=1, video latents packed: (1, 3*8*8, 8) = (1, 192, 8)
    mock_prepare_video.return_value = jnp.zeros((1, 192, 8))
    # audio latents packed: (1, 9, 8*16) = (1, 9, 128)
    mock_prepare_audio.return_value = jnp.zeros((1, 9, 128))

    scheduler_state_mock = MagicMock()
    scheduler_state_mock.timesteps = jnp.array([1.0, 0.5]) # 2 steps
    mock_retrieve.return_value = scheduler_state_mock

    # mock noise output (batch size * 2 for guidance)
    mock_forward.return_value = (jnp.zeros((2, 192, 8)), jnp.zeros((2, 9, 128)))
    
    mock_connectors_model = MagicMock()
    mock_connectors_model.return_value = (jnp.zeros((2, 10, 32)), jnp.zeros((2, 10, 32)), jnp.ones((2, 10)))
    mock_merge.return_value = mock_connectors_model

    mock_split.return_value = (MagicMock(), MagicMock())
    
    mock_vae = MagicMock()
    mock_vae.config.scaling_factor = 1.0
    mock_vae.latents_mean.value = jnp.zeros((8,))
    mock_vae.latents_std.value = jnp.ones((8,))
    mock_vae.decode.return_value = (jnp.zeros((1, 4, 32, 32, 3)),)
    
    mock_audio_vae = MagicMock()
    mock_audio_vae.config.latent_channels = 8
    mock_audio_vae.config.patch_size = None
    mock_audio_vae.config.patch_size_t = None
    mock_audio_vae.config.mel_bins = 64
    mock_audio_vae.latents_mean.value = jnp.zeros((8,))
    mock_audio_vae.latents_std.value = jnp.ones((8,))
    mock_audio_vae.decode.return_value = (jnp.zeros((1, 4, 32, 8)),)
    
    mock_scheduler = MagicMock()
    mock_scheduler.config = {}
    mock_scheduler.step.side_effect = lambda state, pd, t, latents, return_dict: (latents, None)
    
    mock_vocoder = MagicMock()
    mock_vocoder.return_value = jnp.zeros((1, 2, 1000, 10))

    mock_transformer = MagicMock()
    mock_transformer.config.patch_size = 1
    mock_transformer.config.patch_size_t = 1

    pipeline = LTX2Pipeline(
        scheduler=mock_scheduler,
        vae=mock_vae,
        audio_vae=mock_audio_vae,
        text_encoder=MagicMock(),
        tokenizer=MagicMock(),
        connectors=MagicMock(),
        transformer=mock_transformer,
        vocoder=mock_vocoder,
    )
    
    # Needs to match spatial and temporal compression
    pipeline.vae_spatial_compression_ratio = 8
    pipeline.vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.audio_sampling_rate = 16000
    pipeline.audio_hop_length = 160
    pipeline.transformer_spatial_patch_size = 1
    pipeline.transformer_temporal_patch_size = 1
    
    # Call the pipeline
    output = pipeline(
        prompt="Test Prompt",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=3.0,
    )

    # 1. Check prepare latents
    mock_prepare_video.assert_called_once()
    mock_prepare_audio.assert_called_once()
    
    # 2. Check timesteps were retrieved
    mock_retrieve.assert_called_once()
    
    # 3. Check loop execution
    self.assertEqual(mock_forward.call_count, 2)
    self.assertEqual(mock_scheduler.step.call_count, 4) # 2 steps * (video + audio)
    
    # 4. Check Decoding
    mock_vae.decode.assert_called_once()
    mock_audio_vae.decode.assert_called_once()
    mock_vocoder.assert_called_once()
    
    # 5. Output structure
    self.assertIsNotNone(output.frames)
    self.assertIsNotNone(output.audio)

if __name__ == "__main__":
  unittest.main()
