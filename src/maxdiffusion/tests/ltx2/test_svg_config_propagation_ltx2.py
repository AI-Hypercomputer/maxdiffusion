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
from unittest.mock import MagicMock

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh

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

  def test_cfg_and_magcache_incompatibility_validation(self):
    pipeline = LTX2Pipeline.__new__(LTX2Pipeline)
    pipeline.check_inputs = MagicMock()
    pipeline.config = SimpleNamespace(use_svg_attention=True, use_cfg_cache=True, use_magcache=False)

    with self.assertRaisesRegex(ValueError, "SVG sparse attention cannot be combined with CFG cache or MagCache"):
      pipeline(prompt="test prompt")

    pipeline.config = SimpleNamespace(use_svg_attention=True, use_cfg_cache=False, use_magcache=True)
    with self.assertRaisesRegex(ValueError, "SVG sparse attention cannot be combined with CFG cache or MagCache"):
      pipeline(prompt="test prompt")


if __name__ == "__main__":
  unittest.main()
