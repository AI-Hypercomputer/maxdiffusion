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

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import math
import pandas as pd

# Force float32 for precision checks
jax.config.update("jax_default_matmul_precision", "float32")

# ==========================================
# 1. PyTorch Reference Implementation
# ==========================================


class PytorchResBlock(nn.Module):

  def __init__(
      self,
      channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      dilations: tuple = (1, 3, 5),
      leaky_relu_negative_slope: float = 0.1,
  ):
    super().__init__()
    self.dilations = dilations
    self.negative_slope = leaky_relu_negative_slope

    self.convs1 = nn.ModuleList(
        [nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=d, padding="same") for d in dilations]
    )
    self.convs2 = nn.ModuleList(
        [nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=1, padding="same") for _ in dilations]
    )

  def forward(self, x):
    for conv1, conv2 in zip(self.convs1, self.convs2):
      xt = F.leaky_relu(x, negative_slope=self.negative_slope)
      xt = conv1(xt)
      xt = F.leaky_relu(xt, negative_slope=self.negative_slope)
      xt = conv2(xt)
      x = x + xt
    return x


class PytorchLTX2Vocoder(nn.Module):

  def __init__(
      self,
      in_channels: int = 128,
      hidden_channels: int = 1024,
      out_channels: int = 2,
      upsample_kernel_sizes: list = [16, 15, 8, 4, 4],
      upsample_factors: list = [6, 5, 2, 2, 2],
      resnet_kernel_sizes: list = [3, 7, 11],
      resnet_dilations: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      leaky_relu_negative_slope: float = 0.1,
  ):
    super().__init__()
    self.num_upsample_layers = len(upsample_kernel_sizes)
    self.resnets_per_upsample = len(resnet_kernel_sizes)
    self.negative_slope = leaky_relu_negative_slope

    self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3)

    self.upsamplers = nn.ModuleList()
    self.resnets = nn.ModuleList()
    input_channels = hidden_channels

    for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
      output_channels = input_channels // 2
      # Calculate padding to match "SAME" logic roughly or use explicit formula
      padding = (kernel_size - stride) // 2
      self.upsamplers.append(
          nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
      )

      for k_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
        self.resnets.append(
            PytorchResBlock(
                output_channels, k_size, dilations=dilations, leaky_relu_negative_slope=leaky_relu_negative_slope
            )
        )
      input_channels = output_channels

    self.conv_out = nn.Conv1d(input_channels, out_channels, kernel_size=7, stride=1, padding=3)

  def forward(self, x):
    stats = {}
    # x: [B, C, L]
    x = self.conv_in(x)
    stats["conv_in"] = x

    for i in range(self.num_upsample_layers):
      x = F.leaky_relu(x, negative_slope=self.negative_slope)
      x = self.upsamplers[i](x)
      stats[f"upsampler_{i}"] = x

      # Parallel ResNets aggregation
      start = i * self.resnets_per_upsample
      end = (i + 1) * self.resnets_per_upsample

      xs = []
      for j in range(start, end):
        xs.append(self.resnets[j](x))
      x = torch.stack(xs, dim=0).mean(dim=0)
      stats[f"resnet_agg_{i}"] = x

    x = F.leaky_relu(x, negative_slope=0.01)
    x = self.conv_out(x)
    stats["conv_out"] = x
    x = torch.tanh(x)
    stats["final"] = x
    return x, stats


# ==========================================
# 2. JAX Implementation Import
# ==========================================
from ...models.ltx2.vocoder_ltx2 import LTX2Vocoder, ResBlock


class LTX2VocoderTest(unittest.TestCase):

  def setUp(self):
    self.in_channels = 128
    self.hidden_channels = 32  # Reduced for faster unit testing
    self.out_channels = 2
    self.rng = nnx.Rngs(0)

    # Reduced config for speed, but structurally identical
    self.upsample_kernels = [4, 4]
    self.upsample_factors = [2, 2]
    self.resnet_kernels = [3, 5]
    self.resnet_dilations = [[1, 3], [1, 3]]

    self.B, self.T = 1, 16  # Small sequence length

  def _sync_conv(self, jax_layer, pt_layer):
    """Copies weights from PyTorch Conv1d to Flax Conv."""
    # PyTorch: (Out, In, Kernel)
    # Flax: (Kernel, In, Out)
    w_pt = pt_layer.weight.detach().numpy()
    w_jax = w_pt.transpose(2, 1, 0)
    jax_layer.kernel[...] = jnp.array(w_jax)

    if pt_layer.bias is not None:
      b_pt = pt_layer.bias.detach().numpy()
      jax_layer.bias[...] = jnp.array(b_pt)

  def _sync_conv_transpose(self, jax_layer, pt_layer):
    """Copies weights from PyTorch ConvTranspose1d to Flax ConvTranspose."""
    # PyTorch ConvTranspose: (In, Out, Kernel)
    # Flax ConvTranspose: (Kernel, In, Out) - NOTE THE IN/OUT SWAP in PyTorch storage
    w_pt = pt_layer.weight.detach().numpy()
    # Transpose PyTorch (In, Out, K) -> (K, In, Out) implies (K, In, Out)
    # However, ConvTranspose semantics usually swap In/Out in the weight tensor definition.
    # PyTorch ConvTranspose1d weight shape is [in_channels, out_channels/groups, kernel_size] -> (In, Out, K)
    # JAX runtime error indicates kernel shape is (K, In, Out).
    # We need to map (In, Out, K) -> (K, In, Out) using transpose (2,0,1)
    w_jax = w_pt.transpose(2, 0, 1)
    # Flip kernel for conv vs xcorr difference between PT and JAX
    w_jax = np.flip(w_jax, axis=0)
    jax_layer.kernel[...] = jnp.array(w_jax)

    if pt_layer.bias is not None:
      b_pt = pt_layer.bias.detach().numpy()
      jax_layer.bias[...] = jnp.array(b_pt)

  def test_shape_correctness(self):
    """Verifies output shapes matches upsampling factors."""
    model = LTX2Vocoder(
        in_channels=self.in_channels,
        hidden_channels=self.hidden_channels,
        out_channels=self.out_channels,
        upsample_kernel_sizes=self.upsample_kernels,
        upsample_factors=self.upsample_factors,
        resnet_kernel_sizes=self.resnet_kernels,
        resnet_dilations=self.resnet_dilations,
        rngs=self.rng,
    )

    # Input to JAX model needs to be 4D: (B, C', T, F) or (B, C', F, T)
    # We simulate a flattened (128 feature) input by setting C'=1, F=128
    x = jnp.zeros((self.B, 1, self.T, self.in_channels))

    # Expected Output Length: T * prod(upsample_factors)
    expected_len = self.T * math.prod(self.upsample_factors)

    out = model(x, time_last=False)

    # Output should be (Batch, OutChannels, AudioLength) to match PyTorch output format
    self.assertEqual(out.shape, (self.B, self.out_channels, expected_len))
    print("\n[PASS] Shape Test Passed.")

  def test_resblock_parity(self):
    """Verifies a single ResBlock matches PyTorch."""
    ch = 16
    k = 3
    dilations = [1, 3]

    # Init Models
    pt_res = PytorchResBlock(ch, k, dilations=dilations)
    pt_res.eval()

    jax_res = ResBlock(channels=ch, kernel_size=k, dilations=dilations, rngs=self.rng)

    # Sync Weights
    for i in range(len(dilations)):
      self._sync_conv(jax_res.convs1[i], pt_res.convs1[i])
      self._sync_conv(jax_res.convs2[i], pt_res.convs2[i])

    # Run
    np_in = np.random.randn(1, ch, 32).astype(np.float32)
    pt_in = torch.from_numpy(np_in)
    jax_in = jnp.array(np_in.transpose(0, 2, 1))  # (B, C, L) -> (B, L, C)

    with torch.no_grad():
      pt_out = pt_res(pt_in).numpy()

    jax_out = jax_res(jax_in)
    # Transpose JAX back to (B, C, L) for comparison
    jax_out_t = jax_out.transpose(0, 2, 1)

    np.testing.assert_allclose(pt_out, np.array(jax_out_t), atol=1e-5)
    print("[PASS] ResBlock Parity Verified.")

  def test_full_vocoder_parity(self):
    """Verifies full Vocoder parity."""
    pt_model = PytorchLTX2Vocoder(
        in_channels=self.in_channels,
        hidden_channels=self.hidden_channels,
        out_channels=self.out_channels,
        upsample_kernel_sizes=self.upsample_kernels,
        upsample_factors=self.upsample_factors,
        resnet_kernel_sizes=self.resnet_kernels,
        resnet_dilations=self.resnet_dilations,
    ).eval()

    jax_model = LTX2Vocoder(
        in_channels=self.in_channels,
        hidden_channels=self.hidden_channels,
        out_channels=self.out_channels,
        upsample_kernel_sizes=self.upsample_kernels,
        upsample_factors=self.upsample_factors,
        resnet_kernel_sizes=self.resnet_kernels,
        resnet_dilations=self.resnet_dilations,
        rngs=self.rng,
    )

    # --- Weight Syncing ---
    # 1. Conv In
    self._sync_conv(jax_model.conv_in, pt_model.conv_in)

    # 2. Upsamplers
    for jax_up, pt_up in zip(jax_model.upsamplers, pt_model.upsamplers):
      self._sync_conv_transpose(jax_up, pt_up)

    # 3. Resnets
    # Flatten the nested lists for easier iteration
    # PyTorch Resnets are a flat ModuleList
    for i, pt_res in enumerate(pt_model.resnets):
      jax_res = jax_model.resnets[i]
      for d_idx in range(len(pt_res.dilations)):
        self._sync_conv(jax_res.convs1[d_idx], pt_res.convs1[d_idx])
        self._sync_conv(jax_res.convs2[d_idx], pt_res.convs2[d_idx])

    # 4. Conv Out
    self._sync_conv(jax_model.conv_out, pt_model.conv_out)

    # --- Run ---
    # Input (B, C, L)
    np_in = np.random.randn(self.B, self.in_channels, self.T).astype(np.float32)
    pt_in = torch.from_numpy(np_in)

    # Transpose (B,C,T)->(B,T,C) then reshape to 4D for JAX model: (B, 1, T, C) and use time_last=False
    jax_in_4d = jnp.array(np_in.transpose(0, 2, 1)).reshape(self.B, 1, self.T, self.in_channels)

    with torch.no_grad():
      pt_out_tensor, pt_stats = pt_model(pt_in)
      pt_out = pt_out_tensor.numpy()

    # Replicate JAX __call__ to get intermediates
    jax_stats = {}
    time_last = False
    hidden_states = jax_in_4d
    if not time_last:
      hidden_states = jnp.transpose(hidden_states, (0, 1, 3, 2))
    batch, channels, mel_bins, time = hidden_states.shape
    hidden_states = hidden_states.reshape(batch, channels * mel_bins, time)
    hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

    hidden_states = jax_model.conv_in(hidden_states)
    jax_stats["conv_in"] = hidden_states

    for i in range(jax_model.num_upsample_layers):
      hidden_states = jax.nn.leaky_relu(hidden_states, negative_slope=jax_model.negative_slope)
      hidden_states = jax_model.upsamplers[i](hidden_states)
      jax_stats[f"upsampler_{i}"] = hidden_states

      start = i * jax_model.resnets_per_upsample
      end = (i + 1) * jax_model.resnets_per_upsample
      res_sum = 0.0
      for j in range(start, end):
        res_sum = res_sum + jax_model.resnets[j](hidden_states)
      hidden_states = res_sum / jax_model.resnets_per_upsample
      jax_stats[f"resnet_agg_{i}"] = hidden_states

    hidden_states = jax.nn.leaky_relu(hidden_states, negative_slope=0.01)
    hidden_states = jax_model.conv_out(hidden_states)
    jax_stats["conv_out"] = hidden_states
    hidden_states = jnp.tanh(hidden_states)
    jax_stats["final_jax"] = hidden_states  # This is (B, T, C)
    jax_out = jnp.transpose(hidden_states, (0, 2, 1))  # Final output (B,C,T)

    # --- Build Stats Table ---
    stats_list = []

    def add_stat(name, pt_t, jax_t):
      pt_val = pt_t.detach().numpy()
      # jax_t is (B,L,C), transpose to (B,C,L) for comparison
      jax_val = np.array(jax_t).transpose(0, 2, 1)
      stats_list.append({
          "Layer": name,
          "PT Max": f"{pt_val.max():.4f}",
          "JAX Max": f"{jax_val.max():.4f}",
          "PT Mean": f"{pt_val.mean():.4f}",
          "JAX Mean": f"{jax_val.mean():.4f}",
          "PT Min": f"{pt_val.min():.4f}",
          "JAX Min": f"{jax_val.min():.4f}",
          "Diff (L1)": f"{np.abs(pt_val - jax_val).mean():.6f}",
      })

    add_stat("Conv In", pt_stats["conv_in"], jax_stats["conv_in"])
    for i in range(jax_model.num_upsample_layers):
      add_stat(f"Upsampler {i}", pt_stats[f"upsampler_{i}"], jax_stats[f"upsampler_{i}"])
      add_stat(f"ResNet Agg {i}", pt_stats[f"resnet_agg_{i}"], jax_stats[f"resnet_agg_{i}"])
    add_stat("Conv Out", pt_stats["conv_out"], jax_stats["conv_out"])
    add_stat("Final Output", pt_stats["final"], jax_stats["final_jax"])

    df = pd.DataFrame(stats_list)
    print("\n[DIAGNOSTIC] Layer-wise Stats:")
    print(df.to_string(index=False))

    # Compare
    diff = np.abs(pt_out - np.array(jax_out)).max()
    print(f"\n[Vocoder Parity] Max Diff: {diff:.6f}")

    np.testing.assert_allclose(pt_out, np.array(jax_out), atol=1e-4)
    print("[PASS] Full Vocoder Parity Verified.")


if __name__ == "__main__":
  unittest.main()
