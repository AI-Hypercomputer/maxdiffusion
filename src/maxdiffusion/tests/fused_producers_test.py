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

"""Numerical-equivalence tests for the fused attention producers.

These run on CPU; they pin numerical contracts, not kernel performance.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion.kernels.fused_producers import fused_rmsnorm_rope


def _reference_rmsnorm(x, scale, eps=1e-6):
  """Mirrors flax.nnx.RMSNorm's association: x * (rsqrt(var + eps) * scale).

  Flax's `_normalize` builds `mul = rsqrt(var + eps)`, folds the scale into it
  with `mul *= scale`, and only then applies `y *= mul`. Reassociating this as
  `(x * rsqrt) * scale` rounds differently, so the order is load-bearing.
  """
  var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
  return x.astype(jnp.float32) * (jax.lax.rsqrt(var + eps) * scale.astype(jnp.float32))


class FusedRmsNormAssociationTest(unittest.TestCase):
  """The fused producer must be a pure fusion, never a numerical change."""

  def test_matches_flax_rmsnorm_bit_for_bit(self):
    dim = 128
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (2, 64, dim), jnp.float32)
    scale = jax.random.normal(k2, (dim,), jnp.float32)

    layer = nnx.RMSNorm(
        dim,
        epsilon=1e-6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    layer.scale.value = scale

    np.testing.assert_array_equal(
        np.asarray(_reference_rmsnorm(x, scale)),
        np.asarray(layer(x)),
        err_msg="Reference helper must reproduce nnx.RMSNorm exactly.",
    )

  def test_left_to_right_association_is_not_equivalent(self):
    """Guards the reason this test exists: the two orders genuinely differ."""
    dim = 128
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (2, 64, dim), jnp.float32)
    scale = jax.random.normal(k2, (dim,), jnp.float32)

    rsqrt = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
    folded = x * (rsqrt * scale)  # Flax order
    left_to_right = (x * rsqrt) * scale

    self.assertFalse(
        bool(jnp.all(folded == left_to_right)),
        "If these ever become identical the association guard above is vacuous.",
    )

  def test_fused_producer_q_matches_flax_rmsnorm(self):
    """End-to-end: the q path of the fused producer must match nnx.RMSNorm."""
    b, seq, q_heads, dim_head = 1, 8, 2, 8
    d_model = q_heads * dim_head
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)

    raw_q = jax.random.normal(k1, (b, seq, d_model), jnp.float32)
    raw_k = jax.random.normal(k2, (b, seq, d_model), jnp.float32)
    q_scale = jax.random.normal(k3, (d_model,), jnp.float32)
    k_scale = jnp.ones((d_model,), jnp.float32)

    # Identity rotation isolates the RMSNorm from the RoPE.
    freqs_cis = jnp.ones((1, 1, seq, dim_head // 2), jnp.complex64)

    q_out, _ = fused_rmsnorm_rope(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        q_heads=q_heads,
        kv_heads=q_heads,
        dim_head=dim_head,
    )

    expected = _reference_rmsnorm(raw_q, q_scale).reshape(b, seq, q_heads, dim_head).transpose(0, 2, 1, 3)
    np.testing.assert_array_equal(
        np.asarray(q_out.astype(jnp.float32)),
        np.asarray(expected.astype(raw_q.dtype).astype(jnp.float32)),
        err_msg="Fused RMSNorm+RoPE must be bit-identical to nnx.RMSNorm under an identity rotation.",
    )

  def test_separate_q_and_k_epsilons_are_applied(self):
    """Q eps=1e-5 and K eps=1e-6 are applied separately (checked against analytic values, fp32, rtol 1e-6)."""
    b, seq, q_heads, dim_head = 1, 4, 2, 8
    d_model = q_heads * dim_head
    raw_q = jnp.full((b, seq, d_model), 1e-3, dtype=jnp.float32)
    raw_k = jnp.full((b, seq, d_model), 1e-3, dtype=jnp.float32)
    q_scale = jnp.ones((d_model,), dtype=jnp.float32)
    k_scale = jnp.ones((d_model,), dtype=jnp.float32)
    freqs_cis = jnp.ones((1, 1, seq, dim_head // 2), dtype=jnp.complex64)

    q_out, k_out = fused_rmsnorm_rope(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        q_heads=q_heads,
        kv_heads=q_heads,
        dim_head=dim_head,
        eps=1e-5,
        k_eps=1e-6,
    )
    # 1e-3 / sqrt(1e-6 + 1e-5) = 0.30151134 for Q; 1e-3 / sqrt(1e-6 + 1e-6) = 0.70710678 for K
    np.testing.assert_allclose(np.asarray(q_out), 0.30151134, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(k_out), 0.70710678, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
