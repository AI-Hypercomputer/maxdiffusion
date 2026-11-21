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

"""Z-Image text encoder config handling, without downloading weights."""

import types
import unittest

import jax.numpy as jnp
from flax import nnx
from transformers import AutoConfig

from maxdiffusion.checkpointing.z_image_checkpointer import ZImageCheckpointer
from maxdiffusion.utils.testing_utils import cpu_only

_TINY_QWEN3 = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
}


@cpu_only
class ZImageTextEncoderConfigTest(unittest.TestCase):

  def setUp(self):
    config = types.SimpleNamespace(weights_dtype=jnp.float32, logical_axis_rules=(), seed=0)
    self.checkpointer = ZImageCheckpointer(config, mesh=None)

  def _rope_theta(self, text_encoder_config):
    graphdef, state, rest, _ = self.checkpointer.build_text_encoder(text_encoder_config)
    return nnx.merge(graphdef, state, rest).config.rope_theta

  def test_rope_theta_top_level(self):
    """transformers 4.x writes rope_theta at the top level of the config dict."""
    self.assertEqual(self._rope_theta({**_TINY_QWEN3, "rope_theta": 12345.0}), 12345.0)

  def test_rope_theta_in_rope_parameters(self):
    """transformers 5.x keeps rope_theta only under rope_parameters."""
    text_encoder_config = {**_TINY_QWEN3, "rope_parameters": {"rope_type": "default", "rope_theta": 12345.0}}
    self.assertEqual(self._rope_theta(text_encoder_config), 12345.0)

  def test_missing_rope_theta_raises(self):
    with self.assertRaises(KeyError):
      self._rope_theta({**_TINY_QWEN3, "rope_parameters": {"rope_type": "default"}})

  def test_rope_theta_from_installed_transformers(self):
    """The loader passes `AutoConfig.from_pretrained(...).to_dict()`, whichever layout it uses."""
    text_encoder_config = AutoConfig.for_model("qwen3", **_TINY_QWEN3, rope_theta=12345.0).to_dict()
    self.assertEqual(self._rope_theta(text_encoder_config), 12345.0)


if __name__ == "__main__":
  unittest.main()
