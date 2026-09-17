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

Tests for Sparse VideoGen (SVG) attention in LTX2.
"""

import unittest
from unittest.mock import patch, MagicMock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning

from maxdiffusion.models.ltx2.attention_ltx2 import LTX2Attention
from maxdiffusion.models.ltx2.transformer_ltx2 import (
    LTX2VideoTransformerBlock,
    LTX2VideoTransformer3DModel,
    LTX2StaticContext,
    LTX2BlockContext,
)
from maxdiffusion.models.wan.transformers.svg_attention import is_svg_active


class LTX2SVGAttentionTest(unittest.TestCase):

  def setUp(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    self.mesh = Mesh(devices, ("data", "fsdp"))
    self.rngs = nnx.Rngs(0)
    self.logical_axis_rules = (
        ("activation_batch", ("data", "fsdp")),
        ("activation_length", None),
        ("activation_embed", None),
    )

  def test_is_svg_active_ltx2_boundaries(self):
    """Verifies that SVG active predicate correctly checks step, layer, and density."""
    # Active range: step in [10, 30), layer in [1, 28)
    # Outside active steps: step 5, 30, 35 -> False
    self.assertFalse(
        is_svg_active(
            layer_index=5,
            step_index=5,
            timestep=500.0,
            start_step=10,
            end_step=30,
            start_layer=1,
            end_layer=28,
        )
    )
    self.assertFalse(
        is_svg_active(
            layer_index=5,
            step_index=30,
            timestep=200.0,
            start_step=10,
            end_step=30,
            start_layer=1,
            end_layer=28,
        )
    )

    # Outside active layers: layer 0, layer 28 -> False
    self.assertFalse(
        is_svg_active(
            layer_index=0,
            step_index=15,
            timestep=500.0,
            start_step=10,
            end_step=30,
            start_layer=1,
            end_layer=28,
        )
    )
    self.assertFalse(
        is_svg_active(
            layer_index=28,
            step_index=15,
            timestep=500.0,
            start_step=10,
            end_step=30,
            start_layer=1,
            end_layer=28,
        )
    )

    # Inside active steps & layers: step 15, layer 5 -> True
    self.assertTrue(
        is_svg_active(
            layer_index=5,
            step_index=15,
            timestep=500.0,
            start_step=10,
            end_step=30,
            start_layer=1,
            end_layer=28,
        )
    )

    # Dynamic execution under JIT with JAX Array inputs
    @jax.jit
    def check_dynamic(step, layer):
      return is_svg_active(
          layer_index=layer,
          step_index=step,
          timestep=500.0,
          start_step=10,
          end_step=30,
          start_layer=1,
          end_layer=28,
      )

    self.assertTrue(bool(check_dynamic(jnp.int32(15), jnp.int32(5))))
    self.assertFalse(bool(check_dynamic(jnp.int32(5), jnp.int32(5))))
    self.assertFalse(bool(check_dynamic(jnp.int32(15), jnp.int32(0))))

  def test_ltx2_attention_forward_dense_and_sparse_dispatch(self):
    """Tests LTX2Attention routing under inactive and active SVG steps."""
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W
    dim = 128
    num_heads = 4
    head_dim = dim // num_heads

    attention_config = {
        "use_svg_attention": True,
        "svg_spatial_density": 0.25,
        "svg_active_start_step": 2,
        "svg_active_end_step": 10,
        "svg_active_start_layer": 0,
        "svg_active_end_layer": 10,
        "svg_sample_max_row": 100,
        "svg_profile_query_count": 16,
    }

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      attn = LTX2Attention(
          rngs=self.rngs,
          query_dim=dim,
          context_dim=None,
          heads=num_heads,
          dim_head=head_dim,
          attention_kernel="dot_product",
          mesh=self.mesh,
          attention_config=attention_config,
      )

      hidden_states = jax.random.normal(jax.random.key(1), (B, seq_len, dim), dtype=jnp.float32)

      # 1. Inactive step (step 0 < active_start_step 2) -> runs dense path successfully
      out_dense = attn(
          hidden_states=hidden_states,
          spatiotemporal_shape=(F, H, W),
          svg_layer_index=0,
          svg_timestep=jnp.array([100.0]),
          svg_step_index=0,
      )
      self.assertEqual(out_dense.shape, (B, seq_len, dim))
      self.assertTrue(jnp.all(jnp.isfinite(out_dense)))

      # 2. Active step (step 3 in [2, 10)) -> triggers SVG branch which requires custom Ulysses backend
      with self.assertRaisesRegex(ValueError, "Head-local SVG requires a custom Ulysses attention backend"):
        attn(
            hidden_states=hidden_states,
            spatiotemporal_shape=(F, H, W),
            svg_layer_index=0,
            svg_timestep=jnp.array([100.0]),
            svg_step_index=3,
        )

      # 3. Verify mock dispatch receives SVG sparse_config_override and spatiotemporal_shape
      mock_apply = MagicMock(return_value=jnp.zeros((B, seq_len, dim), dtype=jnp.float32))
      with patch.object(attn.attention_op, "apply_attention", mock_apply):
        _ = attn(
            hidden_states=hidden_states,
            spatiotemporal_shape=(F, H, W),
            svg_layer_index=0,
            svg_timestep=jnp.array([100.0]),
            svg_step_index=3,
        )
        mock_apply.assert_called_once()
        _, kwargs = mock_apply.call_args
        self.assertEqual(kwargs.get("spatiotemporal_shape"), (F, H, W))
        sp_cfg = kwargs.get("sparse_config_override")
        self.assertIsNotNone(sp_cfg)
        self.assertTrue(sp_cfg.get("use_svg_attention"))

  def test_ltx2_transformer_block_forward_svg_dispatch(self):
    """Tests LTX2VideoTransformerBlock forward pass routing with SVG."""
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W
    audio_seq_len = 16
    dim = 64
    audio_dim = 64
    num_heads = 2
    head_dim = dim // num_heads

    attention_config = {
        "use_svg_attention": True,
        "svg_spatial_density": 0.25,
        "svg_active_start_step": 2,
        "svg_active_end_step": 10,
        "svg_active_start_layer": 0,
        "svg_active_end_layer": 10,
        "svg_sample_max_row": 100,
        "svg_profile_query_count": 16,
    }

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      block = LTX2VideoTransformerBlock(
          rngs=self.rngs,
          dim=dim,
          num_attention_heads=num_heads,
          attention_head_dim=head_dim,
          cross_attention_dim=dim,
          audio_dim=audio_dim,
          audio_num_attention_heads=num_heads,
          audio_attention_head_dim=head_dim,
          audio_cross_attention_dim=audio_dim,
          attention_kernel="dot_product",
          a2v_attention_kernel="dot_product",
          v2a_attention_kernel="dot_product",
          mesh=self.mesh,
          attention_config=attention_config,
      )

      hidden_states = jax.random.normal(jax.random.key(1), (B, seq_len, dim), dtype=jnp.float32)
      audio_hidden_states = jax.random.normal(jax.random.key(2), (B, audio_seq_len, audio_dim), dtype=jnp.float32)
      encoder_hidden_states = jax.random.normal(jax.random.key(3), (B, 16, dim), dtype=jnp.float32)
      audio_encoder_hidden_states = jax.random.normal(jax.random.key(4), (B, 16, audio_dim), dtype=jnp.float32)

      # Inactive step 0 -> dense execution succeeds
      static_ctx_inactive = LTX2StaticContext(
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          temb=jnp.zeros((B, 6 * dim)),
          temb_audio=jnp.zeros((B, 6 * audio_dim)),
          temb_ca_scale_shift=jnp.zeros((B, 4 * dim)),
          temb_ca_audio_scale_shift=jnp.zeros((B, 4 * audio_dim)),
          temb_ca_gate=jnp.zeros((B, 1 * dim)),
          temb_ca_audio_gate=jnp.zeros((B, 1 * audio_dim)),
          spatiotemporal_shape=(F, H, W),
          svg_timestep=jnp.array([100.0]),
          svg_step_index=0,
      )
      block_ctx_inactive = LTX2BlockContext(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          static=static_ctx_inactive,
          layer_index=0,
      )
      out_h, out_a = block(block_ctx_inactive)
      self.assertEqual(out_h.shape, (B, seq_len, dim))
      self.assertEqual(out_a.shape, (B, audio_seq_len, audio_dim))

      # Active step 3 -> triggers SVG
      static_ctx_active = LTX2StaticContext(
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          temb=jnp.zeros((B, 6 * dim)),
          temb_audio=jnp.zeros((B, 6 * audio_dim)),
          temb_ca_scale_shift=jnp.zeros((B, 4 * dim)),
          temb_ca_audio_scale_shift=jnp.zeros((B, 4 * audio_dim)),
          temb_ca_gate=jnp.zeros((B, 1 * dim)),
          temb_ca_audio_gate=jnp.zeros((B, 1 * audio_dim)),
          spatiotemporal_shape=(F, H, W),
          svg_timestep=jnp.array([100.0]),
          svg_step_index=3,
      )
      block_ctx_active = LTX2BlockContext(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          static=static_ctx_active,
          layer_index=0,
      )
      with self.assertRaisesRegex(ValueError, "Head-local SVG requires a custom Ulysses attention backend"):
        block(block_ctx_active)

  def test_ltx2_model_full_forward_with_svg(self):
    """Tests LTX2VideoTransformer3DModel full forward pass with SVG configuration."""
    B = 1
    F, H, W = 2, 8, 8
    seq_len = F * H * W
    audio_seq_len = 16
    in_channels = 8
    out_channels = 8
    audio_in_channels = 4
    num_heads = 2
    head_dim = 32

    # Step range [5, 15) so step 0 is inactive and completes full dense pass, while step 6 triggers SVG
    attention_config = {
        "use_svg_attention": True,
        "svg_spatial_density": 0.25,
        "svg_active_start_step": 5,
        "svg_active_end_step": 15,
        "svg_active_start_layer": 0,
        "svg_active_end_layer": 10,
        "svg_sample_max_row": 100,
        "svg_profile_query_count": 16,
    }

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      # Non-scanned blocks: step 0 statically resolves inactive and executes dense
      model_unscanned = LTX2VideoTransformer3DModel(
          rngs=nnx.Rngs(0),
          in_channels=in_channels,
          out_channels=out_channels,
          patch_size=1,
          patch_size_t=1,
          num_attention_heads=num_heads,
          attention_head_dim=head_dim,
          cross_attention_dim=num_heads * head_dim,
          caption_channels=16,
          audio_in_channels=audio_in_channels,
          audio_out_channels=audio_in_channels,
          audio_num_attention_heads=num_heads,
          audio_attention_head_dim=head_dim,
          audio_cross_attention_dim=num_heads * head_dim,
          num_layers=2,
          mesh=self.mesh,
          attention_kernel="dot_product",
          a2v_attention_kernel="dot_product",
          v2a_attention_kernel="dot_product",
          scan_layers=False,
          attention_config=attention_config,
      )

      hidden_states = jax.random.normal(jax.random.key(10), (B, seq_len, in_channels), dtype=jnp.float32)
      audio_hidden_states = jax.random.normal(jax.random.key(11), (B, audio_seq_len, audio_in_channels), dtype=jnp.float32)
      timestep = jnp.array([1.0])
      encoder_hidden_states = jax.random.normal(jax.random.key(12), (B, 16, 16), dtype=jnp.float32)
      audio_encoder_hidden_states = jax.random.normal(jax.random.key(13), (B, 16, 16), dtype=jnp.float32)

      # 1. Inactive step (svg_step_index=0) -> executes full dense pass successfully
      output = model_unscanned(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          timestep=timestep,
          num_frames=F,
          height=H,
          width=W,
          audio_num_frames=audio_seq_len,
          svg_step_index=0,
          return_dict=True,
      )

      self.assertEqual(output["sample"].shape, (B, seq_len, out_channels))
      self.assertEqual(output["audio_sample"].shape, (B, audio_seq_len, audio_in_channels))
      self.assertTrue(jnp.all(jnp.isfinite(output["sample"])))
      self.assertTrue(jnp.all(jnp.isfinite(output["audio_sample"])))

      # 2. Active step (svg_step_index=6) -> triggers SVG routing and checks backend
      with self.assertRaisesRegex(ValueError, "Head-local SVG requires a custom Ulysses attention backend"):
        model_unscanned(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            num_frames=F,
            height=H,
            width=W,
            audio_num_frames=audio_seq_len,
            svg_step_index=6,
            return_dict=True,
        )

      # 3. Scanned blocks: fail-closed validation when backend is not a custom Ulysses kernel
      model_scanned = LTX2VideoTransformer3DModel(
          rngs=nnx.Rngs(0),
          in_channels=in_channels,
          out_channels=out_channels,
          patch_size=1,
          patch_size_t=1,
          num_attention_heads=num_heads,
          attention_head_dim=head_dim,
          cross_attention_dim=num_heads * head_dim,
          caption_channels=16,
          audio_in_channels=audio_in_channels,
          audio_out_channels=audio_in_channels,
          audio_num_attention_heads=num_heads,
          audio_attention_head_dim=head_dim,
          audio_cross_attention_dim=num_heads * head_dim,
          num_layers=2,
          mesh=self.mesh,
          attention_kernel="dot_product",
          a2v_attention_kernel="dot_product",
          v2a_attention_kernel="dot_product",
          scan_layers=True,
          attention_config=attention_config,
      )
      with self.assertRaisesRegex(ValueError, "Head-local SVG requires a custom Ulysses attention backend"):
        model_scanned(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            num_frames=F,
            height=H,
            width=W,
            audio_num_frames=audio_seq_len,
            svg_step_index=6,
            return_dict=True,
        )


if __name__ == "__main__":
  unittest.main()
