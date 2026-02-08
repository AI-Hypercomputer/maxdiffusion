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
from ..models.ltx2.attention_ltx2 import LTX2Attention, LTX2AudioVideoRotaryPosEmbed


class LTX2AttentionTest(unittest.TestCase):

  def test_rope_video_shapes(self):
    dim = 64
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=dim,
        patch_size=1,
        patch_size_t=1,
        base_num_frames=8,
        base_height=32,
        base_width=32,
        scale_factors=(1, 1, 1),  # Simplified scales
        modality="video",
        rope_type="interleaved",
    )

    batch_size = 2
    num_frames = 8
    height = 32
    width = 32

    # We need to simulate prepare_video_coords call (usually done in model forward)
    # But here we can call it manually.
    coords = rope.prepare_video_coords(batch_size, num_frames, height, width)
    # coords shape: [B, 3, patches, 2]
    # patches = 8*32*32 = 8192

    self.assertEqual(coords.shape[0], batch_size)
    # prepare_video_coords returns [B, 3, P, 2]
    self.assertEqual(coords.shape[1], 3)

    cos, sin = rope(coords)

    # Check shapes
    # interleaved: [B, patches, dim] (or padded)
    # freqs from rope:
    # output: [B, Heads, T, D/2] ?? No wait, let's check LTX2AudioVideoRotaryPosEmbed.__call__ return
    # It returns cos_freqs, sin_freqs
    # If interleaved: [B, P, dim] (padded if needed)
    # If split: [B, H, P, dim/2] roughly

    # Let's check what __call__ actually returns for interleaved
    # `freqs_out` is [B, P, steps, ndim] -> flatten -> [B, P, steps*ndim] = [B, P, dim//2]
    # repeat -> [B, P, dim]

    # Oh wait, prepare_video_coords in my optimized code returns [B, 3, P, 2].
    # dimensions of grid: [B, 3, P].
    # Oh wait, line 268: `grid = coords / max_positions_b`
    # `coords` is passed to `__call__`.
    # If `coords` is [B, 3, P, 2], `coords.ndim == 4`.
    # Line 253: `coords = (coords[..., 0] + coords[..., 1]) / 2.0` -> [B, 3, P].

    # So `grid` is [B, 3, P].

    self.assertEqual(cos.shape[-1], dim)
    self.assertEqual(sin.shape[-1], dim)
    # self.assertEqual(cos.shape[1], num_frames * height * width) # P

  def test_rope_audio_shapes(self):
    dim = 64
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=dim,
        modality="audio",
        rope_type="interleaved",
        scale_factors=(1,),
    )

    batch_size = 2
    num_frames = 100

    coords = rope.prepare_audio_coords(batch_size, num_frames)
    # Coords: [B, 1, patches, 2]

    cos, sin = rope(coords)
    self.assertEqual(cos.shape[-1], dim)

  def test_attention_forward(self):
    dim = 128
    heads = 4
    dim_head = 32
    model = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rngs=nnx.Rngs(0),
    )

    # Modify attention_op to use dot_product for CPU testing
    model.attention_op.attention_kernel = "dot_product"

    x = jnp.ones((1, 16, dim))

    # Simple forward
    out = model(x)

    self.assertEqual(out.shape, (1, 16, dim))

  def test_attention_with_rope(self):
    dim = 128
    heads = 4
    dim_head = 32

    # Rope logic in LTX2Attention:
    # It accepts `query_rotary_emb` which is usually (cos, sin).
    # If interleaved, apply_interleaved_rotary_emb(query, rope)
    # query: [B, S, H*D].
    # apply_interleaved_rotary_emb expects [..., D].

    # LTX2AudioVideoRotaryPosEmbed.__call__ returns cos, sin.
    # shapes?
    # If interleaved: [B, P, dim].
    # If query is [B, P, H*D], and rope is [B, P, dim]?
    # H*D usually equals dim if rope_dim covers all heads.
    # LTX2Attention sets inner_dim = dim_head * heads.
    # So query is [B, S, inner_dim].
    # So rope should match inner_dim?
    # Yes, LTX2AudioVideoRotaryPosEmbed should be initialized with dim = dim_head * heads.

    rope_dim = dim_head * heads

    model = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rngs=nnx.Rngs(0),
    )
    model.attention_op.attention_kernel = "dot_product"

    x = jnp.ones((1, 8, dim))

    # Fake rope
    cos = jnp.ones((1, 8, rope_dim))
    sin = jnp.zeros((1, 8, rope_dim))
    rope_emb = (cos, sin)

    out = model(x, image_rotary_emb=rope_emb)
    self.assertEqual(out.shape, (1, 8, dim))


if __name__ == "__main__":
  unittest.main()
