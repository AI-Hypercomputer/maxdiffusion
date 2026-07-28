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
from maxdiffusion.utils.testing_utils import cpu_only
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
  scan_layers = getattr(local_module, "scan_layers", False)
  _, state, rest = nnx.split(local_module, nnx.Param, ...)
  flat_state = dict(nnx.to_flat_state(state))
  mapped = set()
  for source_key, tensor in torch_module.state_dict().items():
    target_key, transpose, block_index = z_image_pytorch_key_to_nnx_key(source_key, scan_layers=scan_layers)
    if target_key not in flat_state:
      continue
    value = to_numpy(tensor)
    if transpose:
      value = value.T
    if scan_layers and block_index is not None:
      flat_state[target_key][...] = flat_state[target_key][...].at[block_index].set(jnp.asarray(value))
    else:
      flat_state[target_key][...] = jnp.asarray(value)
    mapped.add(target_key)
  missing = set(flat_state) - mapped
  if missing:
    raise AssertionError(f"Unmapped NNX parameters: {sorted(missing)}")
  return nnx.merge(nnx.graphdef(local_module), nnx.from_flat_state(flat_state), rest)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="PyTorch parity tests are not run in GitHub Actions",
)
@cpu_only
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
    assert_close(
        self,
        local(jnp.asarray(to_numpy(inputs)), jnp.asarray(to_numpy(conditioning))),
        hf(inputs, conditioning),
    )

  def test_rope_embedder_parity(self):
    hf = HFRopeEmbedder(theta=256.0, axes_dims=[2, 2, 4], axes_lens=[64, 64, 64])
    local = ZImageRopeEmbedder(theta=256.0, axes_dims=[2, 2, 4], axes_lens=[64, 64, 64])
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    local_cos, local_sin = local(jnp.asarray(ids.numpy())[None])
    hf_out = hf(ids)
    assert_close(self, local_cos[0], hf_out.real, atol=1e-6, rtol=1e-6)
    assert_close(self, local_sin[0], hf_out.imag, atol=1e-6, rtol=1e-6)

  def test_transformer_block_parity(self):
    hf = HFZImageTransformerBlock(0, 32, 4, 4, 1e-5, True, modulation=True).eval()
    local = copy_parameters(
        ZImageTransformerBlock(self.rngs, 0, 32, 4, 1e-5, True, attention_kernel="dot_product"),
        hf,
    )
    inputs = torch.randn(2, 32, 32)
    ids = torch.randint(0, 16, (2, 32, 3), dtype=torch.int32)
    rope = HFRopeEmbedder(axes_dims=[2, 2, 4], axes_lens=[32, 32, 32])
    freqs = torch.stack([rope(ids[index]) for index in range(ids.shape[0])])
    local_cos = jnp.asarray(to_numpy(freqs.real))
    local_sin = jnp.asarray(to_numpy(freqs.imag))
    conditioning = torch.randn(2, 32)
    assert_close(
        self,
        local(
            jnp.asarray(to_numpy(inputs)),
            (local_cos, local_sin),
            None,
            jnp.asarray(to_numpy(conditioning)),
        ),
        hf(inputs, None, freqs, conditioning),
        atol=3e-5,
        rtol=3e-5,
    )

  FULL_CONFIG = {
      "all_patch_size": (2,),
      "all_f_patch_size": (1,),
      "in_channels": 4,
      "dim": 32,
      "n_layers": 2,
      "n_refiner_layers": 2,
      "n_heads": 4,
      "n_kv_heads": 4,
      "cap_feat_dim": 8,
      "axes_dims": [2, 2, 4],
      # Long enough for the caption block plus the image ids that follow it.
      "axes_lens": [256, 64, 64],
  }

  def _full_pair(self, scan_layers=False):
    hf = HFZImageTransformer2DModel(**self.FULL_CONFIG).eval()
    local = copy_parameters(
        ZImageTransformer2DModel(
            rngs=self.rngs,
            attention_kernel="dot_product",
            scan_layers=scan_layers,
            **self.FULL_CONFIG,
        ),
        hf,
    )
    return hf, local

  def _assert_full_parity(self, hf, local, images, captions, timestep):
    expected = hf(images, timestep, captions, return_dict=False)[0]
    actual = local(
        [jnp.asarray(to_numpy(image)) for image in images],
        jnp.asarray(to_numpy(timestep)),
        [jnp.asarray(to_numpy(caption)) for caption in captions],
        return_dict=False,
    )[0]
    self.assertEqual(len(actual), len(expected))
    for actual_item, expected_item in zip(actual, expected):
      assert_close(self, actual_item, expected_item, atol=5e-5, rtol=5e-5)

  def test_full_transformer_parity(self):
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        hf, local = self._full_pair(scan_layers)
        self._assert_full_parity(hf, local, [torch.randn(4, 1, 4, 4)], [torch.randn(32, 8)], torch.tensor([0.3]))

  def test_full_transformer_parity_unaligned_caption_lengths(self):
    # A caption whose length is not a multiple of SEQ_MULTI_OF is padded
    # inside the denoiser, and upstream keeps running position ids across
    # that padding while offsetting the image ids by the *padded* length.
    # Real prompts are almost never aligned, so this is the common case.
    for scan_layers in (False, True):
      for caption_length in (5, 37, 40, 100):
        with self.subTest(scan_layers=scan_layers, caption_length=caption_length):
          hf, local = self._full_pair(scan_layers)
          self._assert_full_parity(
              hf,
              local,
              [torch.randn(4, 1, 4, 4)],
              [torch.randn(caption_length, 8)],
              torch.tensor([0.3]),
          )

  def test_full_transformer_parity_ragged_batch(self):
    # Mixed caption *and* image lengths, which is what exercises the
    # batch-padding attention mask.
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        hf, local = self._full_pair(scan_layers)
        self._assert_full_parity(
            hf,
            local,
            [torch.randn(4, 1, 4, 4), torch.randn(4, 1, 4, 8)],
            [torch.randn(37, 8), torch.randn(70, 8)],
            torch.tensor([0.3, 0.7]),
        )
