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

Configuration propagation tests for Sparse VideoGen attention in LTX2.
"""

from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import MagicMock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion.pipelines.ltx2 import ltx2_pipeline
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import create_sharded_logical_transformer, LTX2Pipeline


class LTX2SVGConfigPropagationTest(unittest.TestCase):

  def _get_test_ltx2_config(self):
    return {
        "num_layers": 2,
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "cross_attention_dim": 64,
        "in_channels": 16,
        "out_channels": 16,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 32,
        "audio_cross_attention_dim": 64,
        "caption_channels": 32,
        "patch_size": 1,
        "patch_size_t": 1,
        "pos_embed_max_pos": 16,
        "base_height": 32,
        "base_width": 32,
        "audio_pos_embed_max_pos": 16,
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_scale_factor": 1.0,
    }

  def test_svg_config_propagation_through_transformer_construction(self):
    test_ltx2_config = self._get_test_ltx2_config()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    rngs = nnx.Rngs(0)

    cfg = SimpleNamespace(
        use_svg_attention=True,
        svg_spatial_density=0.25,
        svg_active_start_step=8,
        svg_active_end_step=30,
        svg_active_start_layer=1,
        svg_active_end_layer=28,
        precision="DEFAULT",
        flash_block_sizes={},
        activations_dtype="bfloat16",
        weights_dtype="bfloat16",
        attention="dot_product",
        a2v_attention_kernel="dot_product",
        v2a_attention_kernel="dot_product",
        remat_policy="none",
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=[],
        flash_min_seq_length=0,
        dropout=0.0,
        scan_layers=False,
        enable_jax_named_scopes=False,
        use_base2_exp=False,
        use_experimental_scheduler=False,
        logical_axis_rules=(),
    )

    m = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=rngs,
        config=cfg,
        restored_checkpoint={"ltx2_config": dict(test_ltx2_config), "ltx2_state": {}},
        subfolder="",
    )

    # Video self-attention (attn1) must have SVG enabled
    first_block = m.transformer_blocks[0]
    self.assertTrue(first_block.attn1.use_svg_attention)
    self.assertEqual(first_block.attn1.svg_spatial_density, 0.25)
    self.assertEqual(first_block.attn1.svg_active_start_step, 8)
    self.assertEqual(first_block.attn1.svg_active_end_step, 30)
    self.assertEqual(first_block.attn1.svg_active_start_layer, 1)
    self.assertEqual(first_block.attn1.svg_active_end_layer, 28)

    # Audio self-attention (audio_attn1) and cross attentions must remain dense
    self.assertFalse(first_block.audio_attn1.use_svg_attention)
    self.assertFalse(first_block.attn2.use_svg_attention)
    self.assertFalse(first_block.audio_attn2.use_svg_attention)
    self.assertFalse(first_block.audio_to_video_attn.use_svg_attention)
    self.assertFalse(first_block.video_to_audio_attn.use_svg_attention)

  def test_svg_block_sizes_are_independent_of_the_dense_block_sizes(self):
    test_ltx2_config = self._get_test_ltx2_config()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    sparse_blocks = {"block_q": 3328, "block_kv": 2816, "block_kv_compute": 256, "block_kv_compute_in": 256}
    common = {
        "use_svg_attention": True,
        "svg_spatial_density": 0.25,
        "precision": "DEFAULT",
        "activations_dtype": "bfloat16",
        "weights_dtype": "bfloat16",
        "attention": "dot_product",
        "a2v_attention_kernel": "dot_product",
        "v2a_attention_kernel": "dot_product",
        "remat_policy": "none",
        "names_which_can_be_saved": [],
        "names_which_can_be_offloaded": [],
        "flash_min_seq_length": 0,
        "dropout": 0.0,
        "scan_layers": False,
        "enable_jax_named_scopes": False,
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "logical_axis_rules": (),
    }

    def build(**extra):
      return create_sharded_logical_transformer(
          devices_array=devices,
          mesh=mesh,
          rngs=nnx.Rngs(0),
          config=SimpleNamespace(**common, **extra),
          restored_checkpoint={"ltx2_config": dict(test_ltx2_config), "ltx2_state": {}},
          subfolder="",
      )

    tuned = build(flash_block_sizes={}, svg_flash_block_sizes=sparse_blocks)
    self.assertEqual(tuned.transformer_blocks[0].attn1.svg_flash_block_sizes, sparse_blocks)

    default = build(flash_block_sizes={}, svg_flash_block_sizes={})
    self.assertIsNone(default.transformer_blocks[0].attn1.svg_flash_block_sizes)

    absent = build(flash_block_sizes={})
    self.assertIsNone(absent.transformer_blocks[0].attn1.svg_flash_block_sizes)

  def test_svg_disabled_when_use_svg_attention_is_false(self):
    test_ltx2_config = self._get_test_ltx2_config()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "fsdp"))

    cfg_dense = SimpleNamespace(
        use_svg_attention=False,
        precision="DEFAULT",
        flash_block_sizes={},
        activations_dtype="bfloat16",
        weights_dtype="bfloat16",
        attention="dot_product",
        a2v_attention_kernel="dot_product",
        v2a_attention_kernel="dot_product",
        remat_policy="none",
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=[],
        flash_min_seq_length=0,
        dropout=0.0,
        scan_layers=False,
        enable_jax_named_scopes=False,
        use_base2_exp=False,
        use_experimental_scheduler=False,
        logical_axis_rules=(),
    )
    m_dense = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=nnx.Rngs(0),
        config=cfg_dense,
        restored_checkpoint={"ltx2_config": dict(test_ltx2_config), "ltx2_state": {}},
        subfolder="",
    )
    self.assertFalse(m_dense.transformer_blocks[0].attn1.use_svg_attention)

  def test_svg_disabled_when_spatial_density_is_one(self):
    test_ltx2_config = self._get_test_ltx2_config()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "fsdp"))

    cfg_full_density = SimpleNamespace(
        use_svg_attention=True,
        svg_spatial_density=1.0,
        precision="DEFAULT",
        flash_block_sizes={},
        activations_dtype="bfloat16",
        weights_dtype="bfloat16",
        attention="dot_product",
        a2v_attention_kernel="dot_product",
        v2a_attention_kernel="dot_product",
        remat_policy="none",
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=[],
        flash_min_seq_length=0,
        dropout=0.0,
        scan_layers=False,
        enable_jax_named_scopes=False,
        use_base2_exp=False,
        use_experimental_scheduler=False,
        logical_axis_rules=(),
    )
    m = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=nnx.Rngs(0),
        config=cfg_full_density,
        restored_checkpoint={"ltx2_config": dict(test_ltx2_config), "ltx2_state": {}},
        subfolder="",
    )
    # A density of 1.0 is dense attention, so SVG must fall back to the dense path.
    self.assertFalse(m.transformer_blocks[0].attn1.use_svg_attention)

  def test_transformer_forward_pass_forwards_svg_step_index(self):
    captured = {}

    def fake_transformer(**kwargs):
      captured.update(kwargs)
      return kwargs["hidden_states"], kwargs["audio_hidden_states"]

    latents = jnp.zeros((1, 4, 8), dtype=jnp.float32)
    audio_latents = jnp.zeros((1, 2, 8), dtype=jnp.float32)
    with mock.patch.object(ltx2_pipeline.nnx, "merge", return_value=fake_transformer):
      # Call the unjitted function so a cached trace cannot bypass the fake transformer.
      ltx2_pipeline.transformer_forward_pass.fn(
          None,
          {},
          latents,
          audio_latents,
          jnp.asarray(0.5, dtype=jnp.float32),
          jnp.zeros((1, 3, 8), dtype=jnp.float32),
          jnp.zeros((1, 3, 8), dtype=jnp.float32),
          None,
          None,
          latent_num_frames=1,
          latent_height=2,
          latent_width=2,
          audio_num_frames=2,
          fps=24,
          global_batch_size=1,
          svg_step_index=7,
      )

    self.assertIn("svg_step_index", captured)
    self.assertEqual(captured["svg_step_index"], 7)

  def test_cfg_and_magcache_incompatibility_validation(self):
    pipeline = LTX2Pipeline.__new__(LTX2Pipeline)
    pipeline.check_inputs = MagicMock()
    pipeline.config = SimpleNamespace(use_svg_attention=True, use_cfg_cache=True, use_magcache=False)

    with self.assertRaisesRegex(ValueError, "SVG sparse attention cannot be combined with CFG cache or MagCache"):
      pipeline(prompt="test prompt")

    pipeline.config = SimpleNamespace(use_svg_attention=True, use_cfg_cache=False, use_magcache=True)
    with self.assertRaisesRegex(ValueError, "SVG sparse attention cannot be combined with CFG cache or MagCache"):
      pipeline(prompt="test prompt")

  def test_cfg_and_magcache_allowed_when_svg_density_is_one(self):
    class ReachedEncodeError(Exception):
      pass

    pipeline = LTX2Pipeline.__new__(LTX2Pipeline)
    pipeline.check_inputs = MagicMock()
    pipeline.encode_prompt = MagicMock(side_effect=ReachedEncodeError)
    for cache_flags in ({"use_cfg_cache": True, "use_magcache": False}, {"use_cfg_cache": False, "use_magcache": True}):
      pipeline.config = SimpleNamespace(use_svg_attention=True, svg_spatial_density=1.0, **cache_flags)
      # SVG is dense at density 1.0, so the call must get past the SVG check.
      with self.assertRaises(ReachedEncodeError):
        pipeline(prompt="test prompt")

  def test_svg_attention_enabled(self):
    tests = [
        ("flag off", SimpleNamespace(use_svg_attention=False, svg_spatial_density=0.25), False),
        ("flag missing", SimpleNamespace(), False),
        ("sparse density", SimpleNamespace(use_svg_attention=True, svg_spatial_density=0.25), True),
        ("density one", SimpleNamespace(use_svg_attention=True, svg_spatial_density=1.0), False),
        ("density above one", SimpleNamespace(use_svg_attention=True, svg_spatial_density=1.5), False),
        ("density missing uses default", SimpleNamespace(use_svg_attention=True), True),
        ("density none uses default", SimpleNamespace(use_svg_attention=True, svg_spatial_density=None), True),
    ]
    for name, config, want in tests:
      with self.subTest(name):
        self.assertEqual(ltx2_pipeline.svg_attention_enabled(config), want)


if __name__ == "__main__":
  unittest.main()
