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
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxdiffusion.models.ltx2.text_encoders.embeddings_connector_ltx2 import Embeddings1DConnector


class Embeddings1DConnectorTest(unittest.TestCase):

  def setUp(self):
    self.rng = nnx.Rngs(0)
    self.B = 2
    self.T = 16  # Must be divisible by num_learnable_registers if we want tiling to work simply
    self.D = 64  # inner_dim

    # Test config
    self.num_learnable_registers = 8
    self.heads = 4
    self.head_dim = 16

    # input dim = heads * head_dim = 64

  def test_thinking_tokens_replacement(self):
    connector = Embeddings1DConnector(
        input_dim=self.D,
        heads=self.heads,
        head_dim=self.head_dim,
        layers=1,
        num_learnable_registers=self.num_learnable_registers,
        mesh=None,
        rngs=self.rng,
    )

    # Create input [B, T, D]
    hidden_states = jnp.zeros((self.B, self.T, self.D))

    # Create mask [B, T]
    # Batch 0: First 4 valid, rest padding
    # Batch 1: First 8 valid, rest padding
    mask = np.zeros((self.B, self.T), dtype=np.int32)
    mask[0, :4] = 1
    mask[1, :8] = 1

    # Explicitly run replacement method
    output, new_mask = connector._replace_padded_with_learnable_registers(hidden_states, jnp.array(mask))

    # 1. Check Mask is all-ones after registers replace padding (matching diffusers)
    self.assertTrue(jnp.all(new_mask == 1), "New mask should be all-ones after register replacement")

    # 2. Check Valid Tokens are shifted to the left
    # Batch 0: 4 valid tokens shifted left, rest replaced by registers
    valid_b0 = output[0, :4, :]
    self.assertTrue(jnp.all(valid_b0 == 0.0), "Valid tokens should remain unchanged (zeros)")

    # 3. Check register tokens fill the padding area
    register_b0 = output[0, 4:, :]
    self.assertFalse(jnp.all(register_b0 == 0.0), "Padding should be replaced by register values")

    print("\n[PASS] Thinking Tokens Replacement Logic Verified.")

  def test_forward_shape_and_run(self):
    connector = Embeddings1DConnector(
        input_dim=self.D,
        heads=self.heads,
        head_dim=self.head_dim,
        layers=2,
        num_learnable_registers=self.num_learnable_registers,
        attention_kernel="dot_product",  # Use dot_product for testing on CPU
        mesh=None,
        rngs=self.rng,
    )

    hidden_states = jnp.array(np.random.randn(self.B, self.T, self.D))
    mask = jnp.ones((self.B, self.T))  # All valid

    output, new_mask = connector(hidden_states, mask)

    self.assertEqual(output.shape, (self.B, self.T, self.D))
    self.assertFalse(jnp.isnan(output).any(), "Output should not contain NaNs")
    print("\n[PASS] Embeddings1DConnector Forward Pass Verified.")


if __name__ == "__main__":
  unittest.main()
