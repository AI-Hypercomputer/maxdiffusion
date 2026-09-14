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

Configuration propagation tests for Sparse VideoGen attention.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh

from maxdiffusion.pipelines.wan.wan_pipeline import create_sharded_logical_transformer


class SVGConfigPropagationTest(unittest.TestCase):

  def test_svg_config_propagation_through_transformer_construction(self):
    test_wan_config = {
        "num_layers": 2,
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "in_channels": 16,
        "patch_size": (1, 2, 2),
        "text_dim": 64,
        "freq_dim": 64,
        "ffn_dim": 64,
        "eps": 1e-6,
    }

    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    rngs = nnx.Rngs(0)

    cfg = SimpleNamespace(
        use_svg_attention=True,
        svg_high_noise_density=0.50,
        svg_low_noise_density=0.20,
        svg_spatial_density=0.25,
        svg_active_start_step=8,
        svg_active_end_step=30,
        svg_active_start_layer=1,
        svg_active_end_layer=40,
        precision="DEFAULT",
        flash_block_sizes={},
        activations_dtype="bfloat16",
        weights_dtype="bfloat16",
        attention="dot_product",
        remat_policy="none",
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=[],
        flash_min_seq_length=0,
        dropout=0.0,
        mask_padding_tokens=False,
        scan_layers=False,
        enable_jax_named_scopes=False,
        use_base2_exp=False,
        use_experimental_scheduler=False,
        logical_axis_rules=(),
        model_type="T2V",
        model_name="wan2.2",
    )

    m_high = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=rngs,
        config=cfg,
        restored_checkpoint={"wan_config": dict(test_wan_config), "wan_state": {}},
        subfolder="transformer",
    )
    self.assertTrue(m_high.blocks[0].attn1.use_svg_attention)
    self.assertEqual(m_high.blocks[0].attn1.svg_spatial_density, 0.50)
    self.assertEqual(m_high.blocks[0].attn1.svg_active_start_step, 8)
    self.assertEqual(m_high.blocks[0].attn1.svg_active_end_step, 30)
    self.assertEqual(m_high.blocks[0].attn1.svg_active_start_layer, 1)
    self.assertEqual(m_high.blocks[0].attn1.svg_active_end_layer, 40)

    m_low = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=rngs,
        config=cfg,
        restored_checkpoint={"wan_config": dict(test_wan_config), "wan_state": {}},
        subfolder="transformer_2",
    )
    self.assertTrue(m_low.blocks[0].attn1.use_svg_attention)
    self.assertEqual(m_low.blocks[0].attn1.svg_spatial_density, 0.20)

    cfg_dense = SimpleNamespace(
        use_svg_attention=False,
        precision="DEFAULT",
        flash_block_sizes={},
        activations_dtype="bfloat16",
        weights_dtype="bfloat16",
        attention="dot_product",
        remat_policy="none",
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=[],
        flash_min_seq_length=0,
        dropout=0.0,
        mask_padding_tokens=False,
        scan_layers=False,
        enable_jax_named_scopes=False,
        use_base2_exp=False,
        use_experimental_scheduler=False,
        logical_axis_rules=(),
        model_type="T2V",
        model_name="wan2.2",
    )
    m_dense = create_sharded_logical_transformer(
        devices_array=devices,
        mesh=mesh,
        rngs=rngs,
        config=cfg_dense,
        restored_checkpoint={"wan_config": dict(test_wan_config), "wan_state": {}},
        subfolder="transformer_2",
    )
    self.assertFalse(m_dense.blocks[0].attn1.use_svg_attention)

  def test_svg_block_sizes_are_independent_of_the_dense_block_sizes(self):
    """The sparse kernel must be tunable separately from the dense one.

    The two kernels want opposite tilings: the dense ring kernel is tuned for
    large kv compute blocks, the sparse kernel for small ones. Before this key
    existed the sparse path silently reused the dense sizes, so a tuned sparse
    configuration was unreachable from config.
    """
    test_wan_config = {
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "in_channels": 16,
        "patch_size": (1, 2, 2),
        "text_dim": 64,
        "freq_dim": 64,
        "ffn_dim": 64,
        "eps": 1e-6,
    }
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
        "remat_policy": "none",
        "names_which_can_be_saved": [],
        "names_which_can_be_offloaded": [],
        "flash_min_seq_length": 0,
        "dropout": 0.0,
        "mask_padding_tokens": False,
        "scan_layers": False,
        "enable_jax_named_scopes": False,
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "logical_axis_rules": (),
        "model_type": "T2V",
        "model_name": "wan2.2",
    }

    def build(**extra):
      return create_sharded_logical_transformer(
          devices_array=devices,
          mesh=mesh,
          rngs=nnx.Rngs(0),
          config=SimpleNamespace(**common, **extra),
          restored_checkpoint={"wan_config": dict(test_wan_config), "wan_state": {}},
          subfolder="transformer",
      )

    tuned = build(flash_block_sizes={}, svg_flash_block_sizes=sparse_blocks)
    self.assertEqual(tuned.blocks[0].attn1.svg_flash_block_sizes, sparse_blocks)

    # An empty override must mean "reuse the dense sizes", not "use {}", which
    # would silently fall back to the kernel's own unrelated defaults.
    default = build(flash_block_sizes={}, svg_flash_block_sizes={})
    self.assertIsNone(default.blocks[0].attn1.svg_flash_block_sizes)

    absent = build(flash_block_sizes={})
    self.assertIsNone(absent.blocks[0].attn1.svg_flash_block_sizes)


if __name__ == "__main__":
  unittest.main()
