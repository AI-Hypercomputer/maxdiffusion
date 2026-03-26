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

import unittest
from unittest.mock import MagicMock
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion.models.ltx2.ltx2_utils import adain_filter_latent, tone_map_latents
from maxdiffusion.models.ltx2.latent_upsampler_ltx2 import LTX2LatentUpsamplerModel
from maxdiffusion.pipelines.ltx2.pipeline_ltx2_latent_upsample import FlaxLTX2LatentUpsamplePipeline


class LTX2LatentUpsamplerTest(unittest.TestCase):
  """Tests for LTX2 Latent Upsampler components and pipeline."""

  def test_adain_filter_latent(self):
    """Test ADAIN filtering matches global statistics."""
    # Create latents and reference latents with different statistics
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Target (High-res) latents: mean ~ 0, std ~ 1
    latents = jax.random.normal(key1, (1, 4, 16, 16, 8))

    # Reference (Low-res) latents: mean ~ 5, std ~ 2
    reference_latents = jax.random.normal(key2, (1, 4, 16, 16, 8)) * 2.0 + 5.0

    # Apply AdaIN with factor=1.0 (full replacement of style)
    filtered = adain_filter_latent(latents, reference_latents, factor=1.0)

    # Validate shapes
    self.assertEqual(filtered.shape, latents.shape)

    # Validate statistics: output should now roughly match reference stats
    axes = (1, 2, 3)
    ref_mean = jnp.mean(reference_latents, axis=axes, keepdims=True)
    ref_std = jnp.std(reference_latents, axis=axes, keepdims=True)

    out_mean = jnp.mean(filtered, axis=axes, keepdims=True)
    out_std = jnp.std(filtered, axis=axes, keepdims=True)

    np.testing.assert_allclose(out_mean, ref_mean, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_std, ref_std, rtol=1e-4, atol=1e-4)

    # Test factor = 0.0 (no change)
    unfiltered = adain_filter_latent(latents, reference_latents, factor=0.0)
    np.testing.assert_allclose(unfiltered, latents, rtol=1e-5)

  def test_tone_map_latents(self):
    """Test tone mapping compression scale logic."""
    latents = jnp.ones((1, 4, 16, 16, 8)) * 2.0

    # Compress with 0 ratio should do nothing when properly scaled,
    # but based on the code: scale_factor = compression * 0.75
    # If compression = 0.0, scale_factor = 0, scales = 1.0
    mapped_0 = tone_map_latents(latents, compression=0.0)
    np.testing.assert_allclose(mapped_0, latents, rtol=1e-5)

    # Compress with > 0 ratio should reduce the magnitude
    mapped_compressed = tone_map_latents(latents, compression=1.0)
    self.assertTrue(jnp.all(jnp.abs(mapped_compressed) < jnp.abs(latents)))
    self.assertEqual(mapped_compressed.shape, latents.shape)

  def test_upsampler_model_forward(self):
    """Test the neural network component of the upsampler for shape validity."""
    b, f, h, w, c = 2, 3, 16, 16, 8
    key = jax.random.PRNGKey(0)

    # Instantiate the module with small channels/blocks to keep test fast.
    # mid_channels MUST be a multiple of 32 because GroupNorm uses num_groups=32 natively.
    model = LTX2LatentUpsamplerModel(
        in_channels=c,
        mid_channels=32,
        num_blocks_per_stage=1,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        rational_spatial_scale=2.0,  # Maps to 2x upscaling
        rngs=nnx.Rngs(0),
    )

    dummy_input = jax.random.normal(key, (b, f, h, w, c))

    # Forward pass is now just a direct function call in nnx!
    output = model(dummy_input)

    # Assert temporal unchanged, spatial doubled, channels restored to `in_channels`
    self.assertEqual(output.shape, (b, f, h * 2, w * 2, c))

  def test_pipeline_latent_upsample_logic(self):
    """Test FlaxLTX2LatentUpsamplePipeline call pipeline properties."""
    mock_vae = MagicMock()
    mock_vae.config = {"spatial_compression_ratio": 32, "temporal_compression_ratio": 8}
    mock_vae.latents_mean = [0.0] * 8
    mock_vae.latents_std = [1.0] * 8
    mock_vae.dtype = jnp.float32

    dummy_video = jnp.zeros((1, 1, 32, 32, 3))
    mock_vae.decode.return_value = (dummy_video,)

    # FIX: Use a minimal real nnx.Module instead of MagicMock
    # so nnx.split and nnx.update can traverse it without crashing.
    class DummyUpsampler(nnx.Module):

      def __call__(self, x):
        # Return a tensor of ones to prove it was executed
        return jnp.ones((1, 4, 16, 16, 8))

    mock_upsampler = DummyUpsampler()

    pipeline = FlaxLTX2LatentUpsamplePipeline(
        vae=mock_vae,
        latent_upsampler=mock_upsampler,
    )

    pipeline.video_processor.postprocess_video = MagicMock(return_value=np.zeros((1, 3, 1, 32, 32)))

    # Empty params skips `nnx.update` completely
    params = {}
    prng_seed = jax.random.PRNGKey(0)
    latents = jnp.zeros((1, 4, 16, 16, 8))

    out_latents = pipeline(
        params=params,
        prng_seed=prng_seed,
        latents=latents,
        latents_normalized=False,
        adain_factor=1.0,
        tone_map_compression_ratio=0.5,
        output_type="latent",
        return_dict=True,
    )

    self.assertIn("frames", out_latents)
    self.assertEqual(out_latents["frames"].shape, (1, 4, 16, 16, 8))

    out_decoded = pipeline(
        params=params, prng_seed=prng_seed, latents=latents, latents_normalized=False, output_type="pil", return_dict=True
    )

    mock_vae.decode.assert_called_once()
    self.assertIn("frames", out_decoded)


if __name__ == "__main__":
  unittest.main()
