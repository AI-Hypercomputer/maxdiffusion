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

Configuration propagation tests for Sol-Attn in the Wan transformer.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh

from maxdiffusion.pipelines.wan.wan_pipeline import create_sharded_logical_transformer

_WAN_CONFIG = {
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

_COMMON = {
    "precision": "DEFAULT",
    "flash_block_sizes": {},
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
    "model_name": "wan2.1",
}


def _build(**extra):
  devices = np.array(jax.devices()[:1]).reshape((1, 1))
  return create_sharded_logical_transformer(
      devices_array=devices,
      mesh=Mesh(devices, ("data", "fsdp")),
      rngs=nnx.Rngs(0),
      config=SimpleNamespace(**_COMMON, **extra),
      restored_checkpoint={"wan_config": dict(_WAN_CONFIG), "wan_state": {}},
      subfolder="transformer",
  )


class SolConfigPropagationTest(unittest.TestCase):

  def test_sol_settings_reach_self_attention_only(self):
    model = _build(
        use_sol_attention=True,
        sol_beta=1.25,
        sol_block_q=1024,
        sol_block_kv=256,
        sol_active_start_step=10,
        sol_active_end_step=50,
        sol_active_start_layer=1,
        sol_active_end_layer=40,
    )
    attn1 = model.blocks[0].attn1
    self.assertTrue(attn1.use_sol_attention)
    self.assertFalse(attn1.use_svg_attention)
    self.assertEqual(attn1.sol_beta, 1.25)
    self.assertEqual((attn1.sol_block_q, attn1.sol_block_kv), (1024, 256))
    self.assertEqual((attn1.sol_active_start_step, attn1.sol_active_end_step), (10, 50))
    self.assertEqual((attn1.sol_active_start_layer, attn1.sol_active_end_layer), (1, 40))
    self.assertEqual(attn1.sol_num_layers, 2)
    # Cross-attention stays dense, as in the paper.
    self.assertFalse(model.blocks[0].attn2.use_sol_attention)

  def test_dense_config_is_untouched(self):
    model = _build(use_sol_attention=False, sol_beta=3.0)
    self.assertFalse(model.blocks[0].attn1.use_sol_attention)
    self.assertNotIn("sol_beta", model.config.attention_config)

  def test_sol_and_svg_are_mutually_exclusive(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      _build(use_sol_attention=True, use_svg_attention=True, svg_spatial_density=0.25)


if __name__ == "__main__":
  unittest.main()
