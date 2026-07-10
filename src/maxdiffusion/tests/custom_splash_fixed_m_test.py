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

"""Unit tests for the fixed-m path of the custom splash attention kernel.

The fixed-m optimization replaces the online-softmax running max with a
precomputed per-query Cauchy-Schwarz bound for eligible heads, falling back to
online softmax for "sink" heads whose bound exceeds the no-flush gate. These
tests check that, mirroring the production calling convention, the kernel:

  * matches an f32 softmax reference for both online and fixed-m modes,
  * produces fixed-m output that agrees with online output to bf16 precision,
  * flags an out-of-gate head ineligible and falls back without NaNs.
"""

import math
import unittest

import jax
import jax.numpy as jnp

from maxdiffusion.kernels import custom_splash_attention as custom_splash

_LOG2E = math.log2(math.e)


class CustomSplashFixedMTest(unittest.TestCase):
  """Numerical equivalence tests for the fixed-m kernel path."""

  num_heads = 5
  seq_len = 4096
  head_dim = 128

  def setUp(self):
    super().setUp()
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.block_sizes = custom_splash._BlockSizes(block_q=2048, block_kv=1024, block_kv_compute=512)

  def _random_qkv(self, q_gain: float = 1.0, k_gain: float = 1.0) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns bf16 (q, k, v), optionally amplifying head 0 of q and k."""
    shape = (self.num_heads, self.seq_len, self.head_dim)
    q = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), shape, jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), shape, jnp.bfloat16)
    q = q.at[0].multiply(q_gain)
    k = k.at[0].multiply(k_gain)
    return q, k, v

  def _reference(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Per-head f32 softmax attention reference."""
    qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
    logits = jnp.einsum("hsd,htd->hst", qf, kf) * self.scale
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hst,htd->hsd", probs, vf)

  def _run_kernel(self, q: jax.Array, k: jax.Array, v: jax.Array, use_fixed_m: bool) -> tuple[jax.Array, jax.Array | None]:
    """Runs the custom kernel using the production scaling convention.

    Args:
      q: Query tensor of shape (heads, seq, dim).
      k: Key tensor of shape (heads, seq, dim).
      v: Value tensor of shape (heads, seq, dim).
      use_fixed_m: Whether to enable the fixed-m bound path.

    Returns:
      A tuple of the f32 attention output (heads, seq, dim) and the per-head
      mk array (or None for the online path).
    """
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in = k * self.scale
    mk = None
    if use_fixed_m:
      # k-smoothing makes every logit row mean-zero so row-max >= 0.
      k_in = k_in - jnp.mean(k_in, axis=1, keepdims=True)
      qn = jnp.sqrt((q_in.astype(jnp.float32) ** 2).sum(-1)).max(axis=1)
      mk_h = jnp.sqrt((k_in.astype(jnp.float32) ** 2).sum(-1)).max(axis=1)
      eligible = (qn * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
      mk = jnp.stack([mk_h, eligible])
    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        bkv_compute_in=256,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=use_fixed_m,
    )
    out = kernel(q_in, k_in, v, mk) if use_fixed_m else kernel(q_in, k_in, v)
    out = jnp.swapaxes(out, 1, 2)  # (heads, dim, seq) -> (heads, seq, dim)
    return out.astype(jnp.float32), mk

  def test_online_matches_reference(self):
    """Online softmax path agrees with the f32 reference at bf16 precision."""
    q, k, v = self._random_qkv()
    online, _ = self._run_kernel(q, k, v, use_fixed_m=False)
    self.assertLess(float(jnp.max(jnp.abs(online - self._reference(q, k, v)))), 2e-2)

  def test_fixed_m_matches_online_when_all_eligible(self):
    """With uniform data all heads are eligible and match online output."""
    q, k, v = self._random_qkv()
    online, _ = self._run_kernel(q, k, v, use_fixed_m=False)
    fixed, mk = self._run_kernel(q, k, v, use_fixed_m=True)
    self.assertTrue(bool(jnp.all(mk[1] > 0.5)))  # every head eligible
    self.assertTrue(bool(jnp.all(jnp.isfinite(fixed))))
    self.assertLess(float(jnp.max(jnp.abs(fixed - online))), 5e-3)

  def test_fixed_m_matches_reference(self):
    """Fixed-m output agrees with the f32 softmax reference."""
    q, k, v = self._random_qkv()
    fixed, _ = self._run_kernel(q, k, v, use_fixed_m=True)
    self.assertLess(float(jnp.max(jnp.abs(fixed - self._reference(q, k, v)))), 2e-2)

  def test_sink_head_falls_back_to_online(self):
    """An out-of-gate head is flagged ineligible and stays finite (no flush)."""
    q, k, v = self._random_qkv(q_gain=6.0, k_gain=6.0)
    fixed, mk = self._run_kernel(q, k, v, use_fixed_m=True)
    self.assertEqual(float(mk[1][0]), 0.0)  # head 0 is a sink -> ineligible
    self.assertTrue(bool(jnp.all(mk[1][1:] > 0.5)))  # the rest stay eligible
    self.assertTrue(bool(jnp.all(jnp.isfinite(fixed))))


if __name__ == "__main__":
  unittest.main()
