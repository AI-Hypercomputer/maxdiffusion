import pytest
import numpy as np

from maxdiffusion.models.ltx2.text_encoders.hf_gemma3_encoder import HFGemma3TextEncoder


class TestHFGemma3TextEncoder:
  """Test suite for the Hugging Face CPU-based Gemma 3 Text Encoder."""

  @pytest.fixture(scope="class")
  def encoder(self):
    """Initialize the encoder. We use a small max_length to save memory and time."""
    print("Initializing HFGemma3TextEncoder on CPU...")
    # Note: Depending on your system memory, loading 12B on CPU might take ~25GB RAM.
    # Ensure the test node has enough CPU RAM.
    encoder = HFGemma3TextEncoder("google/gemma-3-12b-it", max_length=16)
    return encoder

  def test_encode_output_shape(self, encoder):
    """Verify that the encode method returns the correctly flattened numpy array."""
    prompt = "A test prompt for HF Gemma 3"

    # Run encode
    print("Running encode forward pass on CPU...")
    output_array = encoder.encode(prompt)

    # Verify it's a numpy array
    assert isinstance(output_array, np.ndarray), "Output must be a numpy array for JAX integration."

    # Verify shape
    # Expected: (batch_size, sequence_length, 49 * 3840) -> (1, 16, 188160)
    expected_shape = (1, 16, 49 * 3840)
    assert output_array.shape == expected_shape, f"Expected shape {expected_shape}, got {output_array.shape}"

    print(f"✅ Output successfully shaped for GemmaFeaturesExtractorProjLinear: {output_array.shape}")
