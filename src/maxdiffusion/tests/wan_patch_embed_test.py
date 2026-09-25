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

"""Tests for the token-sharded Wan patch embedding (`wan_patch_embed_mode`)."""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import pyconfig, wan_runtime_options
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.wan.transformers import transformer_wan
from maxdiffusion.models.wan.transformers.transformer_wan import (
    WanModel,
    can_embed_patches_as_tokens,
    embed_patches_as_tokens,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _conv(patch_size, in_channels=4, out_channels=256, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, **kw):
  return nnx.Conv(
      in_channels,
      out_channels,
      kernel_size=patch_size,
      strides=patch_size,
      dtype=dtype,
      param_dtype=dtype,
      precision=precision,
      rngs=nnx.Rngs(0),
      bias_init=nnx.initializers.normal(0.5),
      **kw,
  )


class PatchEmbedTokensTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml")], unittest=True)
    self.config = pyconfig.config
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)

  def _both(self, conv, x, patch_size):
    def conv_path(x):
      return jax.lax.collapse(conv(x), 1, -1)

    def token_path(x):
      return embed_patches_as_tokens(conv, x, patch_size)

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      return jax.jit(conv_path)(x), jax.jit(token_path)(x)

  def test_matches_conv(self):
    for patch_size in ((1, 2, 2), (2, 2, 2)):
      with self.subTest(patch_size=patch_size):
        conv = _conv(patch_size)
        x = jax.random.normal(jax.random.key(1), (2, 4, 8, 16, 4), jnp.float32)
        want, got = self._both(conv, x, patch_size)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)

  def test_matches_conv_in_bf16(self):
    patch_size = (1, 2, 2)
    conv = _conv(patch_size, in_channels=16, dtype=jnp.bfloat16, precision=None)
    x = jax.random.normal(jax.random.key(2), (2, 3, 8, 16, 16), jnp.float32).astype(jnp.bfloat16)
    want, got = self._both(conv, x, patch_size)
    self.assertEqual(got.dtype, want.dtype)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=2e-2)

  def test_guard(self):
    conv = _conv((1, 2, 2))
    self.assertTrue(can_embed_patches_as_tokens(conv, (1, 21, 90, 160, 4), (1, 2, 2)))
    # A spatial size not divisible by the patch makes 'SAME' pad.
    self.assertFalse(can_embed_patches_as_tokens(conv, (1, 21, 91, 160, 4), (1, 2, 2)))
    self.assertFalse(can_embed_patches_as_tokens(conv, (1, 21, 90, 160, 4), (1, 1, 2)))
    overlapping = nnx.Conv(4, 8, kernel_size=(1, 2, 2), strides=(1, 1, 1), rngs=nnx.Rngs(0))
    self.assertFalse(can_embed_patches_as_tokens(overlapping, (1, 21, 90, 160, 4), (1, 2, 2)))
    dilated = _conv((1, 2, 2), kernel_dilation=(1, 2, 2))
    self.assertFalse(can_embed_patches_as_tokens(dilated, (1, 21, 90, 160, 4), (1, 2, 2)))


class WanModelPatchEmbedModeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    wan_runtime_options.reset()
    self.addCleanup(wan_runtime_options.reset)
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml")], unittest=True)
    self.config = pyconfig.config
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      self.model = WanModel(
          rngs=nnx.Rngs(0),
          num_attention_heads=2,
          attention_head_dim=128,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=256,
          num_layers=1,
          mesh=self.mesh,
          attention="dot_product",
      )
    self.inputs = {
        "hidden_states": jax.random.normal(jax.random.key(3), (1, 4, 2, 8, 16), jnp.float32),
        "timestep": jnp.ones((1,)),
        "encoder_hidden_states": jax.random.normal(jax.random.key(4), (1, 16, 32), jnp.float32),
    }

  def _run(self, mode):
    with mock.patch.dict(os.environ, {"WAN_PATCH_EMBED_MODE": mode}):
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        return self.model(**self.inputs)

  def test_tokens_mode_uses_the_token_path_and_matches_conv(self):
    want = self._run("conv")
    with mock.patch.object(transformer_wan, "embed_patches_as_tokens", wraps=transformer_wan.embed_patches_as_tokens) as spy:
      got = self._run("tokens")
    spy.assert_called_once()
    self.assertEqual(got.shape, want.shape)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-2, atol=2e-2)

  def test_conv_mode_does_not_use_the_token_path(self):
    with mock.patch.object(transformer_wan, "embed_patches_as_tokens") as spy:
      self._run("conv")
    spy.assert_not_called()

  def test_rejects_unknown_mode(self):
    with self.assertRaises(ValueError):
      self._run("bogus")


if __name__ == "__main__":
  unittest.main()
