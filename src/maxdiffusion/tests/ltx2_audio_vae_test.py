"""Tests for the LTX-2 Audio VAE."""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from maxdiffusion.models.ltx2.audio_vae import FlaxAutoencoderKLLTX2Audio


class FlaxLTX2AudioVAETest(absltest.TestCase):

    def test_forward_pass_basic(self):
        """Verifies basic shape arithmetic and default mode."""
        rngs = nnx.Rngs(0)
        model = FlaxAutoencoderKLLTX2Audio(
            in_channels=1,
            output_channels=1,
            latent_channels=8,
            base_channels=64,
            ch_mult=(1, 2, 4), # 2 downsamples (factor of 4)
            resolution=64,
            num_res_blocks=1,
            mel_bins=32,
            causality_axis='none',
            rngs=rngs
        )
        
        input_shape = (1, 64, 32, 1) # (batch, time, freq, channels)
        dummy_input = jnp.ones(input_shape)
        
        # 1. Test encode
        posterior = model.encode(dummy_input).latent_dist
        mean = posterior.mean
        # With factor 4: Time: 64 -> 32 -> 16. Freq: 32 -> 16 -> 8.
        self.assertEqual(mean.shape, (1, 16, 8, 8))
        self.assertEqual(posterior.logvar.shape, (1, 16, 8, 8))
        
        # 2. Test decode
        z = posterior.mode()
        output = model.decode(z).sample
        self.assertEqual(output.shape, (1, 64, 32, 1))
        
        # 3. Test full call
        output_full = model(dummy_input, sample_posterior=False).sample
        self.assertEqual(output_full.shape, (1, 64, 32, 1))

    def test_causality_axes(self):
        """Verifies that custom asymmetric padding works for all causality axes."""
        axes = ['none', 'height', 'width', 'width-compatibility']
        input_shape = (1, 64, 32, 1)
        dummy_input = jnp.ones(input_shape)
        
        for axis in axes:
            with self.subTest(causality_axis=axis):
                rngs = nnx.Rngs(0)
                model = FlaxAutoencoderKLLTX2Audio(
                    in_channels=1, output_channels=1, latent_channels=8, base_channels=64,
                    ch_mult=(1, 2, 4), resolution=64, num_res_blocks=1, mel_bins=32,
                    causality_axis=axis, rngs=rngs
                )
                
                output = model(dummy_input, sample_posterior=False).sample
                
                # Causal convolutions shift the timeline, dropping 3 future context frames 
                # over the course of the 2-stage (factor 4) downsample/upsample cycle.
                expected_time = 64 if axis == 'none' else 61
                self.assertEqual(output.shape, (1, expected_time, 32, 1), f"Shape mismatch on causality_axis='{axis}'")

    def test_posterior_sampling(self):
        """Verifies the reparameterization trick and RNG threading."""
        rngs = nnx.Rngs(0)
        model = FlaxAutoencoderKLLTX2Audio(
            in_channels=1, output_channels=1, latent_channels=8, base_channels=64,
            ch_mult=(1, 2, 4), resolution=64, num_res_blocks=1, mel_bins=32,
            causality_axis='none', rngs=rngs
        )
        
        dummy_input = jnp.ones((1, 64, 32, 1))
        sample_rng = jax.random.PRNGKey(42)
        
        # Should execute successfully and return correct shape
        output = model(dummy_input, sample_posterior=True, rng=sample_rng).sample
        self.assertEqual(output.shape, (1, 64, 32, 1))

    def test_variable_length_audio(self):
        """Verifies the decoder's _adjust_output_shape logic with non-power-of-2 times."""
        rngs = nnx.Rngs(0)
        model = FlaxAutoencoderKLLTX2Audio(
            in_channels=1, output_channels=1, latent_channels=8, base_channels=64,
            ch_mult=(1, 2, 4), resolution=64, num_res_blocks=1, mel_bins=32,
            causality_axis='height', rngs=rngs
        )
        
        input_shape = (1, 68, 32, 1) 
        dummy_input = jnp.ones(input_shape)
        
        output = model(dummy_input, sample_posterior=False).sample
        
        # Time 68 -> Downsampled to 17. 
        # Target upsampled = 17 * 4 - 3 (causal context drop) = 65.
        self.assertEqual(output.shape, (1, 65, 32, 1))


if __name__ == '__main__':
    absltest.main()