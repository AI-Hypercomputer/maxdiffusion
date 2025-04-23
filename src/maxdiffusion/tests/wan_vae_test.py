"""
 Copyright 2025 Google LLC

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

import os
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import unittest
import pytest
from absl.testing import absltest
from ..models.wan.autoencoder_kl_wan import WanCausalConv3d, WanUpsample, AutoencoderKLWan, WanRMS_norm

class WanVaeTest(unittest.TestCase):
  def setUp(self):
    WanVaeTest.dummy_data = {}
  
  # def test_clear_cache(self):
  #   key = jax.random.key(0)
  #   rngs = nnx.Rngs(key)
  #   wan_vae = AutoencoderKLWan(rngs=rngs)
  #   wan_vae.clear_cache()

  def test_wanrms_norm(self):
    """Test against the Pytorch implementation"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchWanRMS_norm(nn.Module):
      r"""
      A custom RMS normalization layer.

      Args:
          dim (int): The number of dimensions to normalize over.
          channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
              Default is True.
          images (bool, optional): Whether the input represents image data. Default is True.
          bias (bool, optional): Whether to include a learnable bias term. Default is False.
      """

      def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

      def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias
    
    # --- Test Case 1: images == True ---
    model = TorchWanRMS_norm(2)
    input_shape = (1, 2, 2, 2, 3)
    input = torch.ones(input_shape)
    torch_output = model(input)
    torch_output_np = torch_output.detach().numpy()
    
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    wanrms_norm = WanRMS_norm(dim=2, rngs=rngs)
    input_shape = (1, 2, 2, 2, 3)
    dummy_input = jnp.ones(input_shape)
    output = wanrms_norm(dummy_input)
    output_np = np.array(output)
    assert np.allclose(output_np, torch_output_np) == True

    # --- Test Case 2: images == False ---
    model = TorchWanRMS_norm(2, images=False)
    input_shape = (1, 2, 2, 2, 3)
    input = torch.ones(input_shape)
    torch_output = model(input)
    torch_output_np = torch_output.detach().numpy()
    
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    wanrms_norm = WanRMS_norm(dim=2, rngs=rngs, images=False)
    input_shape = (1, 2, 2, 2, 3)
    dummy_input = jnp.ones(input_shape)
    output = wanrms_norm(dummy_input)
    output_np = np.array(output)
    assert np.allclose(output_np, torch_output_np) == True

  def test_wan_upsample(self):
    batch_size=1
    in_depth, in_height, in_width = 10, 32, 32
    in_channels = 3

    dummy_input = jnp.ones((batch_size, in_depth, in_height, in_width, in_channels))

    upsample = WanUpsample(scale_factor=(2.0, 2.0))

    # --- Test Case 1: depth > 1 ---
    output = upsample(dummy_input)
    assert output.shape == (1, 10, 64, 64, 3)

    in_depth = 1
    dummy_input = jnp.ones((batch_size, in_depth, in_height, in_width, in_channels))
    # --- Test Case 1: depth == 1 ---
    output = upsample(dummy_input)
    assert output.shape == (1, 1, 64, 64, 3)

  def test_3d_conv(self):
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    batch_size=1
    in_depth, in_height, in_width = 10, 32, 32
    in_channels = 3
    out_channels = 16
    kernel_d, kernel_h, kernel_w = 3, 3, 3 # Kernel size (Depth, Height, Width)
    padding_d, padding_h, padding_w = 1, 1, 1 # Base padding (Depth, Height, Width)

    # Create dummy input data
    dummy_input = jnp.ones((batch_size, in_depth, in_height, in_width, in_channels))

    # Create dummy cache data (from a previous step)
    cache_depth = 2 * padding_d
    dummy_cache = jnp.zeros((batch_size, cache_depth, in_height, in_width, in_channels))

    # Instantiate the module
    causal_conv_layer = WanCausalConv3d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(kernel_d, kernel_h, kernel_w),
      padding=(padding_d, padding_h, padding_w),
      rngs=rngs # Pass rngs for initialization
    )

    # --- Test Case 1: No Cache ---
    output_no_cache = causal_conv_layer(dummy_input)
    assert output_no_cache.shape == (1, 10, 32, 32, 16)

    # --- Test Case 2: With Cache ---
    output_with_cache = causal_conv_layer(dummy_input, cache_x=dummy_cache)
    assert output_with_cache.shape == (1, 10, 32, 32, 16)

    # --- Test Case 3: With Cache larger than padding ---
    larger_cache_depth = 4 # Larger than needed padding (2*padding_d = 2)
    dummy_larger_cache = jnp.zeros((batch_size, larger_cache_depth, in_height, in_width, in_channels))
    output_with_larger_cache = causal_conv_layer(dummy_input, cache_x=dummy_larger_cache)
    assert output_with_larger_cache.shape == (1, 10, 32, 32, 16)

if __name__ == "__main__":
  absltest.main()