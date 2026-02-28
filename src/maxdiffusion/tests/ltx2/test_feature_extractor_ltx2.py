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
import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from ...models.ltx2.text_encoders.feature_extractor_ltx2 import LTX2GemmaFeatureExtractor, _norm_and_concat_padded_batch


# ==========================================
# PyTorch Reference Logic
# ==========================================
def pt_norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
  b, t, d, l = encoded_text.shape
  device = encoded_text.device

  token_indices = torch.arange(t, device=device)[None, :]
  if padding_side == "right":
    mask = token_indices < sequence_lengths[:, None]
  elif padding_side == "left":
    start_indices = t - sequence_lengths[:, None]
    mask = token_indices >= start_indices
  else:
    raise ValueError

  mask = mask[:, :, None, None]  # [B, T, 1, 1]

  eps = 1e-6
  masked = encoded_text.masked_fill(~mask, 0.0)
  denom = (sequence_lengths * d).view(b, 1, 1, 1)
  mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

  x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
  x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
  range_ = x_max - x_min

  normed = 8 * (encoded_text - mean) / (range_ + eps)
  normed = normed.reshape(b, t, -1)

  # Apply mask
  mask_flattened = mask.view(b, t, 1).expand(-1, -1, d * l)
  normed = normed.masked_fill(~mask_flattened, 0.0)

  return normed


class LTX2FeatureExtractorTest(unittest.TestCase):

  def setUp(self):
    self.rng = nnx.Rngs(0)
    self.B = 2
    self.T = 10
    self.D = 8
    self.L = 3
    self.target_dim = 16

  def test_norm_parity(self):
    # Create random input with some padding
    np_input = np.random.randn(self.B, self.T, self.D, self.L).astype(np.float32)

    # Lengths: e.g. [5, 8] out of 10
    lengths = np.array([5, 8], dtype=np.int32)

    # PyTorch Reference
    pt_input = torch.from_numpy(np_input)
    pt_lengths = torch.from_numpy(lengths)
    pt_out = pt_norm_and_concat_padded_batch(pt_input, pt_lengths)

    # JAX Implementation
    jax_input = jnp.array(np_input)
    jax_lengths = jnp.array(lengths)
    jax_out = _norm_and_concat_padded_batch(jax_input, jax_lengths)

    diff = np.abs(pt_out.numpy() - np.array(jax_out)).max()
    print(f"\n[Norm Parity] Max Diff: {diff:.6f}")

    np.testing.assert_allclose(pt_out.numpy(), np.array(jax_out), atol=1e-5)
    print("[PASS] Normalization Logic Parity Verified.")

  def test_module_forward(self):
    # Test full module
    model = LTX2GemmaFeatureExtractor(input_dim=self.D * self.L, output_dim=self.target_dim, rngs=self.rng)

    # Create input tuple (simulate Gemma output)
    hidden_states = [jnp.array(np.random.randn(self.B, self.T, self.D)) for _ in range(self.L)]

    # Attention Mask [B, T]
    mask = np.zeros((self.B, self.T), dtype=np.int32)
    mask[0, :5] = 1
    mask[1, :8] = 1
    jax_mask = jnp.array(mask)

    output = model(tuple(hidden_states), jax_mask)

    expected_shape = (self.B, self.T, self.target_dim)
    self.assertEqual(output.shape, expected_shape)

    # Check padding regions are zero
    # Batch 0, indices 5: should be 0
    padding_val = output[0, 5:, :]
    self.assertTrue(jnp.all(padding_val == 0.0), "Padding region should be zero")

    print("\n[PASS] Feature Extractor Module Forward Pass Verified.")


if __name__ == "__main__":
  unittest.main()
