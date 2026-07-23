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

"""Parity coverage for every Z-Image transformer layer against Diffusers PyTorch."""

import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from diffusers.models.transformers.transformer_z_image import (
    FeedForward as HFFeedForward,
    FinalLayer as HFFinalLayer,
    RopeEmbedder as HFRopeEmbedder,
    TimestepEmbedder as HFTimestepEmbedder,
    ZImageTransformer2DModel as HFZImageTransformer2DModel,
    ZImageTransformerBlock as HFZImageTransformerBlock,
)
from maxdiffusion.models.z_image.transformer_z_image import (
    ZImageFeedForward,
    ZImageFinalLayer,
    ZImageRopeEmbedder,
    ZImageTimestepEmbedder,
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from maxdiffusion.models.z_image.z_image_utils import z_image_pytorch_key_to_nnx_key


def to_numpy(value):
  if isinstance(value, torch.Tensor):
    if value.dtype == torch.bfloat16:
      value = value.float()
    return value.detach().cpu().numpy()
  return np.asarray(value)


def assert_close(test_case, actual, expected, atol=2e-5, rtol=2e-5):
  test_case.assertEqual(to_numpy(actual).shape, to_numpy(expected).shape)
  np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), atol=atol, rtol=rtol)


def copy_parameters(local_module, torch_module):
  _, state, rest = nnx.split(local_module, nnx.Param, ...)
  flat_state = dict(nnx.to_flat_state(state))
  mapped = set()
  for source_key, tensor in torch_module.state_dict().items():
    target_key, transpose = z_image_pytorch_key_to_nnx_key(source_key)
    if target_key not in flat_state:
      continue
    value = to_numpy(tensor)
    if transpose:
      value = value.T
    flat_state[target_key][...] = jnp.asarray(value)
    mapped.add(target_key)
  missing = set(flat_state) - mapped
  if missing:
    raise AssertionError(f"Unmapped NNX parameters: {sorted(missing)}")
  return nnx.merge(nnx.graphdef(local_module), nnx.from_flat_state(flat_state), rest)


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="PyTorch parity tests are not run in GitHub Actions")
class ZImageModuleParityTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    self.rngs = nnx.Rngs(jax.random.key(0))

  def test_timestep_embedder_parity(self):
    hf = HFTimestepEmbedder(8, mid_size=12, frequency_embedding_size=10).eval()
    local = copy_parameters(ZImageTimestepEmbedder(self.rngs, 8, 12, 10), hf)
    timestep = torch.tensor([0.1, 0.7])
    assert_close(self, local(jnp.asarray(to_numpy(timestep))), hf(timestep))

  def test_feed_forward_parity(self):
    hf = HFFeedForward(8, 20).eval()
    local = copy_parameters(ZImageFeedForward(self.rngs, 8, 20), hf)
    inputs = torch.randn(2, 7, 8)
    assert_close(self, local(jnp.asarray(to_numpy(inputs))), hf(inputs))

  def test_final_layer_parity(self):
    hf = HFFinalLayer(8, 6).eval()
    local = copy_parameters(ZImageFinalLayer(self.rngs, 8, 6), hf)
    inputs, conditioning = torch.randn(2, 7, 8), torch.randn(2, 8)
    assert_close(self, local(jnp.asarray(to_numpy(inputs)), jnp.asarray(to_numpy(conditioning))), hf(inputs, conditioning))

  def test_rope_embedder_parity(self):
    hf = HFRopeEmbedder(theta=256.0, axes_dims=[2, 2, 4], axes_lens=[64, 64, 64])
    local = ZImageRopeEmbedder(theta=256.0, axes_dims=[2, 2, 4], axes_lens=[64, 64, 64])
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    assert_close(self, local(jnp.asarray(ids.numpy())[None])[0], hf(ids), atol=1e-6, rtol=1e-6)

  def test_transformer_block_parity(self):
    hf = HFZImageTransformerBlock(0, 32, 4, 4, 1e-5, True, modulation=True).eval()
    local = copy_parameters(ZImageTransformerBlock(self.rngs, 0, 32, 4, 1e-5, True, attention_kernel="dot_product"), hf)
    inputs = torch.randn(2, 32, 32)
    ids = torch.randint(0, 16, (2, 32, 3), dtype=torch.int32)
    rope = HFRopeEmbedder(axes_dims=[2, 2, 4], axes_lens=[32, 32, 32])
    freqs = torch.stack([rope(ids[index]) for index in range(ids.shape[0])])
    conditioning = torch.randn(2, 32)
    assert_close(
        self,
        local(jnp.asarray(to_numpy(inputs)), jnp.asarray(to_numpy(freqs)), None, jnp.asarray(to_numpy(conditioning))),
        hf(inputs, None, freqs, conditioning),
        atol=3e-5,
        rtol=3e-5,
    )

  def test_full_transformer_parity(self):
    config = {
        "all_patch_size": (2,),
        "all_f_patch_size": (1,),
        "in_channels": 4,
        "dim": 32,
        "n_layers": 1,
        "n_refiner_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
        "cap_feat_dim": 8,
        "axes_dims": [2, 2, 4],
        "axes_lens": [64, 64, 64],
    }
    hf = HFZImageTransformer2DModel(**config).eval()
    local = copy_parameters(ZImageTransformer2DModel(rngs=self.rngs, attention_kernel="dot_product", **config), hf)
    images = [torch.randn(4, 1, 4, 4)]
    captions = [torch.randn(32, 8)]
    timestep = torch.tensor([0.3])
    expected = hf(images, timestep, captions, return_dict=False)[0][0]
    actual = local(
        [jnp.asarray(to_numpy(images[0]))],
        jnp.asarray(to_numpy(timestep)),
        [jnp.asarray(to_numpy(captions[0]))],
        return_dict=False,
    )[0][0]
    assert_close(self, actual, expected, atol=5e-5, rtol=5e-5)
