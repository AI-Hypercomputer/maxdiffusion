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

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion.models import lora_nnx

IN_FEATURES, OUT_FEATURES, RANK, NUM_LAYERS = 64, 4096, 8, 4


class _Model(nnx.Module):

  def __init__(self, dtype, num_layers=None):
    self.proj = nnx.Linear(IN_FEATURES, OUT_FEATURES, param_dtype=dtype, rngs=nnx.Rngs(0))
    if num_layers is not None:
      # Stacked weights, as in a scanned transformer.
      self.proj.kernel.value = jnp.zeros((num_layers, IN_FEATURES, OUT_FEATURES), dtype)
      self.proj.bias.value = jnp.zeros((num_layers, OUT_FEATURES), dtype)


def _lora(rng, base_std, delta_ratio):
  """Returns (lora_A, lora_B, delta) with delta = (B @ A).T scaled to delta_ratio * base_std."""
  down = rng.normal(size=(RANK, IN_FEATURES)).astype(np.float32)
  up = rng.normal(size=(OUT_FEATURES, RANK)).astype(np.float32)
  delta = (up @ down).T
  factor = np.sqrt(delta_ratio * base_std / delta.std())
  return down * factor, up * factor, delta * factor**2


def _retained(merged, base, delta):
  """Fraction of delta present in merged - base (1.0 means the full delta was applied)."""
  diff = np.asarray(merged, np.float32) - np.asarray(base, np.float32)
  return float((diff * delta).sum() / (delta * delta).sum())


class LoraNNXMergeTest(unittest.TestCase):
  """Tests merging LoRA weights into NNX layers."""

  # LoRA updates are typically far smaller than the weights they modify. At this ratio
  # round-to-nearest into bfloat16 keeps only about half of the update.
  BASE_STD, DELTA_RATIO = 0.02, 1e-3

  def _base(self, rng, shape):
    return rng.normal(0, self.BASE_STD, shape).astype(np.float32)

  def test_merge_keeps_small_delta_in_bfloat16(self):
    rng = np.random.default_rng(0)
    model = _Model(jnp.bfloat16)
    model.proj.kernel.value = jnp.asarray(self._base(rng, (IN_FEATURES, OUT_FEATURES)), jnp.bfloat16)
    model.proj.bias.value = jnp.asarray(self._base(rng, (OUT_FEATURES,)), jnp.bfloat16)
    kernel, bias = np.asarray(model.proj.kernel.value), np.asarray(model.proj.bias.value)
    down, up, delta = _lora(rng, self.BASE_STD, self.DELTA_RATIO)
    bias_delta = rng.normal(0, self.BASE_STD * self.DELTA_RATIO, OUT_FEATURES).astype(np.float32)
    state_dict = {"proj.lora_A.weight": down, "proj.lora_B.weight": up, "proj.diff_b": bias_delta}

    lora_nnx.merge_lora(model, state_dict, RANK, 1.0, lambda path: "proj" if path == "proj" else None, dtype="bfloat16")

    self.assertEqual(model.proj.kernel.value.dtype, jnp.bfloat16)
    self.assertAlmostEqual(_retained(model.proj.kernel.value, kernel, delta), 1.0, delta=0.05)
    self.assertAlmostEqual(_retained(model.proj.bias.value, bias, bias_delta), 1.0, delta=0.1)

  def test_scanned_merge_keeps_small_delta_in_bfloat16(self):
    rng = np.random.default_rng(1)
    model = _Model(jnp.bfloat16, num_layers=NUM_LAYERS)
    model.proj.kernel.value = jnp.asarray(self._base(rng, (NUM_LAYERS, IN_FEATURES, OUT_FEATURES)), jnp.bfloat16)
    model.proj.bias.value = jnp.asarray(self._base(rng, (NUM_LAYERS, OUT_FEATURES)), jnp.bfloat16)
    kernel, bias = np.asarray(model.proj.kernel.value), np.asarray(model.proj.bias.value)
    state_dict, deltas, bias_deltas = {}, [], []
    for i in range(NUM_LAYERS):
      down, up, delta = _lora(rng, self.BASE_STD, self.DELTA_RATIO)
      bias_delta = rng.normal(0, self.BASE_STD * self.DELTA_RATIO, OUT_FEATURES).astype(np.float32)
      state_dict.update({f"proj.{i}.lora_A.weight": down, f"proj.{i}.lora_B.weight": up, f"proj.{i}.diff_b": bias_delta})
      deltas.append(delta)
      bias_deltas.append(bias_delta)

    lora_nnx.merge_lora_for_scanned(
        model, state_dict, RANK, 1.0, lambda path: "proj.{}" if path == "proj" else None, dtype="bfloat16"
    )

    self.assertEqual(model.proj.kernel.value.dtype, jnp.bfloat16)
    self.assertAlmostEqual(_retained(model.proj.kernel.value, kernel, np.stack(deltas)), 1.0, delta=0.05)
    self.assertAlmostEqual(_retained(model.proj.bias.value, bias, np.stack(bias_deltas)), 1.0, delta=0.1)

  def test_merge_is_deterministic(self):
    rng = np.random.default_rng(2)
    base = jnp.asarray(self._base(rng, (IN_FEATURES, OUT_FEATURES)), jnp.bfloat16)
    down, up, _ = _lora(rng, self.BASE_STD, self.DELTA_RATIO)
    state_dict = {"proj.lora_A.weight": down, "proj.lora_B.weight": up}
    merged = []
    for _ in range(2):
      model = _Model(jnp.bfloat16)
      model.proj.kernel.value = base
      lora_nnx.merge_lora(model, state_dict, RANK, 1.0, lambda path: "proj" if path == "proj" else None)
      merged.append(np.asarray(model.proj.kernel.value, np.float32))
    np.testing.assert_array_equal(merged[0], merged[1])

  def test_merge_float32_is_exact_sum(self):
    rng = np.random.default_rng(3)
    model = _Model(jnp.float32)
    kernel = self._base(rng, (IN_FEATURES, OUT_FEATURES))
    model.proj.kernel.value = jnp.asarray(kernel)
    down, up, delta = _lora(rng, self.BASE_STD, self.DELTA_RATIO)
    state_dict = {"proj.lora_A.weight": down, "proj.lora_B.weight": up}

    # TPUs run float32 matmuls in reduced precision by default; check the sum at full precision.
    with jax.default_matmul_precision("highest"):
      lora_nnx.merge_lora(model, state_dict, RANK, 1.0, lambda path: "proj" if path == "proj" else None)

    self.assertEqual(model.proj.kernel.value.dtype, jnp.float32)
    np.testing.assert_allclose(np.asarray(model.proj.kernel.value), kernel + delta, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
  unittest.main()
