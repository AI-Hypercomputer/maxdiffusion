"""Copyright 2025 Google LLC

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
from flax import nnx
import jax
import jax.numpy as jnp
# Adjust this import to match your file structure
from ..models.ltx2.attention_ltx2 import LTX2Attention, LTX2RotaryPosEmbed


class LTX2AttentionTest(unittest.TestCase):

  def test_rope_video_shapes_3d(self):
    """Test 3D RoPE generation for Video (Time, Height, Width)."""
    dim = 64
    # LTX-2 splits dim across axes. 60 is divisible by 3 (20 per axis).
    dim = 60
    rope = LTX2RotaryPosEmbed(dim=dim, theta=10000.0)

    batch_size = 2
    seq_len = 16
    
    # Create dummy position IDs for [Time, Height, Width]
    # Shape: [B, S, 3]
    ids = jnp.zeros((batch_size, seq_len, 3), dtype=jnp.float32)

    cos, sin = rope(ids)

    # Expected output: [B, S, 1, D] (The 1 is for broadcasting across heads)
    # This confirms the RoPE module outputs the correct broadcasting shape.
    self.assertEqual(cos.shape, (batch_size, seq_len, 1, dim))
    self.assertEqual(sin.shape, (batch_size, seq_len, 1, dim))

  def test_rope_audio_shapes_1d(self):
    """Test 1D RoPE generation for Audio."""
    dim = 64
    rope = LTX2RotaryPosEmbed(dim=dim, theta=10000.0)

    batch_size = 2
    seq_len = 20
    
    # Create dummy position IDs for [Time]
    # Shape: [B, S, 1]
    ids = jnp.zeros((batch_size, seq_len, 1), dtype=jnp.float32)

    cos, sin = rope(ids)

    # Expected output: [B, S, 1, D]
    self.assertEqual(cos.shape, (batch_size, seq_len, 1, dim))
    self.assertEqual(sin.shape, (batch_size, seq_len, 1, dim))

  def test_self_attention_forward(self):
    """Test basic Self-Attention forward pass (Video <-> Video)."""
    dim = 64
    heads = 4
    dim_head = 16 # inner_dim = 64
    
    model = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rngs=nnx.Rngs(0),
    )
    
    # Standard input [B, S, D]
    x = jnp.ones((1, 16, dim))

    # Forward
    out = model(hidden_states=x)
    
    self.assertEqual(out.shape, (1, 16, dim))

  def test_cross_attention_forward(self):
    """Test Cross-Attention forward pass (Video Query <-> Audio Context)."""
    query_dim = 64
    context_dim = 128 # Audio latents often have different dim
    heads = 4
    dim_head = 16
    
    model = LTX2Attention(
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        context_dim=context_dim, # Triggers cross-attention init
        rngs=nnx.Rngs(0),
    )
    
    x = jnp.ones((1, 16, query_dim))       # Video
    context = jnp.ones((1, 20, context_dim)) # Audio

    out = model(hidden_states=x, encoder_hidden_states=context)
    
    self.assertEqual(out.shape, (1, 16, query_dim))

  def test_attention_with_rope_integration(self):
    """Test that passing RoPE embeddings works without shape errors."""
    dim = 64
    heads = 4
    dim_head = 16
    
    model = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rngs=nnx.Rngs(0),
    )
    
    x = jnp.ones((2, 8, dim))
    
    # Create manual RoPE embeddings matching the output of LTX2RotaryPosEmbed
    # Shape: [B, S, 1, inner_dim]
    cos = jnp.ones((2, 8, 1, 64))
    sin = jnp.ones((2, 8, 1, 64))
    rope_emb = (cos, sin)

    out = model(hidden_states=x, rotary_emb=rope_emb)
    self.assertEqual(out.shape, (2, 8, dim))
    
  def test_cross_modal_temporal_rope(self):
    """
    Test the specific LTX-2 requirement: 
    Video (Spatial) attends to Audio (Temporal) using Temporal-Only RoPE.
    """
    query_dim = 64
    heads = 4
    dim_head = 16
    
    model = LTX2Attention(
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        context_dim=query_dim, 
        rngs=nnx.Rngs(0),
    )
    
    x = jnp.ones((1, 16, query_dim))      # Video
    context = jnp.ones((1, 20, query_dim)) # Audio
    
    # 1. Generate 3D IDs for Video (Time, Height, Width)
    video_ids_3d = jnp.zeros((1, 16, 3))
    
    # 2. Extract ONLY Time axis for Cross-Attention RoPE
    # The pipeline must do this slicing: ids[:, :, 0:1]
    video_ids_temporal = video_ids_3d[..., 0:1] # Shape [1, 16, 1]
    
    # 3. Generate 1D IDs for Audio
    audio_ids = jnp.zeros((1, 20, 1))
    
    rope_gen = LTX2RotaryPosEmbed(dim=64)
    
    # Generate RoPE using only the temporal IDs
    q_rope = rope_gen(video_ids_temporal) 
    k_rope = rope_gen(audio_ids)
    
    # Verify dimensions match for broadcasting
    self.assertEqual(q_rope[0].shape, (1, 16, 1, 64))
    self.assertEqual(k_rope[0].shape, (1, 20, 1, 64))
    
    # Forward Pass
    out = model(
        hidden_states=x, 
        encoder_hidden_states=context, 
        rotary_emb=q_rope, 
        k_rotary_emb=k_rope
    )
    
    self.assertEqual(out.shape, (1, 16, query_dim))

if __name__ == "__main__":
  unittest.main()