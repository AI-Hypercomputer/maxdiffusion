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

from unittest import mock

from maxdiffusion.kernels.fused_producers import fused_rmsnorm_rope
from maxdiffusion.models import attention_flax
from maxdiffusion.models.attention_flax import FlaxWanAttention, _unflatten_heads


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

  def _bf16_fused_vs_flax(self, scale_dtype):
    """Returns (fused_q, fused_k, flax_q, flax_k) in bf16 under an identity rotation."""
    b, seq, q_heads, dim_head = 2, 16, 4, 32
    d_model = q_heads * dim_head
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    raw_q = jax.random.normal(k1, (b, seq, d_model), jnp.float32).astype(jnp.bfloat16)
    raw_k = jax.random.normal(k2, (b, seq, d_model), jnp.float32).astype(jnp.bfloat16)
    q_scale = jax.random.normal(k3, (d_model,), jnp.float32).astype(scale_dtype)
    k_scale = jax.random.normal(k4, (d_model,), jnp.float32).astype(scale_dtype)

    def flax_norm(scale, eps):
      layer = nnx.RMSNorm(d_model, epsilon=eps, dtype=jnp.bfloat16, param_dtype=scale_dtype, rngs=nnx.Rngs(0))
      layer.scale.value = scale
      return layer

    freqs_cis = jnp.ones((1, 1, seq, dim_head // 2), jnp.complex64)  # identity rotation
    q_out, k_out = fused_rmsnorm_rope(
        raw_q, raw_k, q_scale, k_scale, freqs_cis, q_heads=q_heads, dim_head=dim_head, eps=1e-6, k_eps=1e-5
    )
    self.assertEqual(q_out.dtype, jnp.bfloat16)
    self.assertEqual(k_out.dtype, jnp.bfloat16)

    def to_heads(x):
      return x.reshape(b, seq, q_heads, dim_head).transpose(0, 2, 1, 3)

    expected_q = to_heads(flax_norm(q_scale, 1e-6)(raw_q))
    expected_k = to_heads(flax_norm(k_scale, 1e-5)(raw_k))
    self.assertEqual(expected_q.dtype, jnp.bfloat16)

    def f32(x):
      return np.asarray(x.astype(jnp.float32))

    return f32(q_out), f32(k_out), f32(expected_q), f32(expected_k)

  def test_bfloat16_activations_and_scale_match_flax_rmsnorm_bit_for_bit(self):
    """Production dtype: bf16 activations with a bf16 scale, identity rotation.

    `nnx.RMSNorm(dtype=bf16)` computes the statistics in f32 and casts the
    normalised result to bf16; the fused producer must round identically.
    """
    q_out, k_out, expected_q, expected_k = self._bf16_fused_vs_flax(jnp.bfloat16)
    np.testing.assert_array_equal(q_out, expected_q, err_msg="bf16 fused Q must be bit-identical to nnx.RMSNorm.")
    np.testing.assert_array_equal(k_out, expected_k, err_msg="bf16 fused K must be bit-identical to nnx.RMSNorm.")

  def test_bfloat16_activations_with_f32_scale_differ_from_flax_by_at_most_one_ulp(self):
    """Documents the one case that is NOT bit-identical.

    With `dtype=bf16` and an f32 scale, `nnx.RMSNorm.promote_dtype` rounds the
    scale to bf16 *before* folding it into `rsqrt(var + eps)`, whereas the
    fused producer keeps the scale in f32 and only rounds the final product.
    The outputs therefore differ by at most one bf16 ulp.
    """
    q_out, k_out, expected_q, expected_k = self._bf16_fused_vs_flax(jnp.float32)
    self.assertFalse(np.array_equal(q_out, expected_q), "If this becomes bit-identical, fold the two bf16 tests.")
    for got, want in ((q_out, expected_q), (k_out, expected_k)):
      # One bf16 ulp is 2^-7 relative at the bottom of a binade.
      np.testing.assert_allclose(got, want, rtol=2**-7, atol=0.0)

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

  def test_nontrivial_rope_and_bfloat16_match_unfused_reference(self):
    """Non-trivial complex RoPE rotation + pair interleave in bfloat16 matches unfused RMSNorm + RoPE."""
    b, seq, q_heads, dim_head = 2, 16, 4, 16
    d_model = q_heads * dim_head
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    raw_q = jax.random.normal(k1, (b, seq, d_model), jnp.bfloat16)
    raw_k = jax.random.normal(k2, (b, seq, d_model), jnp.bfloat16)
    q_scale = jax.random.normal(k3, (d_model,), jnp.bfloat16)
    k_scale = jax.random.normal(k4, (d_model,), jnp.bfloat16)
    angles = jax.random.uniform(k5, (1, 1, seq, dim_head // 2), minval=-3.14, maxval=3.14)
    freqs_cis = jnp.exp(1j * angles).astype(jnp.complex64)

    q_out, k_out = fused_rmsnorm_rope(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        q_heads=q_heads,
        kv_heads=q_heads,
        dim_head=dim_head,
    )
    self.assertEqual(q_out.dtype, jnp.bfloat16)
    self.assertEqual(k_out.dtype, jnp.bfloat16)

    # The producer is jit-wrapped, so XLA keeps the bf16 RoPE chain in f32 inside
    # a fusion and rounds once; op-by-op eager bf16 rounds after every multiply
    # and differs by 1 ulp. Compile the reference the same way so the comparison
    # pins the fusion's numerics, not eager-vs-compiled bf16 rounding.
    @jax.jit
    def _unfused_ref(x, scale):
      normed = _reference_rmsnorm(x, scale).astype(x.dtype)
      h = normed.reshape(b, seq, q_heads, dim_head).transpose(0, 2, 1, 3)
      cos = jnp.real(freqs_cis).astype(x.dtype)
      sin = jnp.imag(freqs_cis).astype(x.dtype)
      pairs = h.reshape(b, q_heads, seq, -1, 2)
      x0, x1 = pairs[..., 0], pairs[..., 1]
      out0 = x0 * cos - x1 * sin
      out1 = x0 * sin + x1 * cos
      return jnp.stack([out0, out1], axis=-1).reshape(b, q_heads, seq, dim_head)

    np.testing.assert_array_equal(
        np.asarray(q_out.astype(jnp.float32)),
        np.asarray(_unfused_ref(raw_q, q_scale).astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(k_out.astype(jnp.float32)),
        np.asarray(_unfused_ref(raw_k, k_scale).astype(jnp.float32)),
    )

  def test_gqa_nontrivial_rope_shapes_and_values(self):
    """GQA (q_heads != kv_heads) with non-trivial RoPE produces exact reference values."""
    b, sq, sk, q_heads, kv_heads, dim_head = 2, 12, 12, 4, 2, 16
    dq = q_heads * dim_head
    dk = kv_heads * dim_head
    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    raw_q = jax.random.normal(k1, (b, sq, dq), jnp.bfloat16)
    raw_k = jax.random.normal(k2, (b, sk, dk), jnp.bfloat16)
    q_scale = jax.random.normal(k3, (dq,), jnp.bfloat16)
    k_scale = jax.random.normal(k4, (dk,), jnp.bfloat16)
    angles = jax.random.uniform(k5, (1, 1, sq, dim_head // 2), minval=-2.0, maxval=2.0)
    freqs_cis = jnp.exp(1j * angles).astype(jnp.complex64)

    q_out, k_out = fused_rmsnorm_rope(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim_head=dim_head,
    )
    self.assertEqual(q_out.shape, (b, q_heads, sq, dim_head))
    self.assertEqual(k_out.shape, (b, kv_heads, sk, dim_head))

    # Verify K against manual complex multiplication in float32. Compiled for the
    # same reason as in test_nontrivial_rope_and_bfloat16_match_unfused_reference.
    @jax.jit
    def _k_ref(raw_k, k_scale, freqs_cis):
      k_normed = _reference_rmsnorm(raw_k, k_scale).astype(jnp.bfloat16)
      k_h = k_normed.reshape(b, sk, kv_heads, dim_head).transpose(0, 2, 1, 3)
      cos_k = jnp.real(freqs_cis).astype(jnp.bfloat16)
      sin_k = jnp.imag(freqs_cis).astype(jnp.bfloat16)
      k_pairs = k_h.reshape(b, kv_heads, sk, -1, 2)
      return jnp.stack(
          [
              k_pairs[..., 0] * cos_k - k_pairs[..., 1] * sin_k,
              k_pairs[..., 0] * sin_k + k_pairs[..., 1] * cos_k,
          ],
          axis=-1,
      ).reshape(b, kv_heads, sk, dim_head)

    np.testing.assert_array_equal(
        np.asarray(k_out.astype(jnp.float32)),
        np.asarray(_k_ref(raw_k, k_scale, freqs_cis).astype(jnp.float32)),
    )

  def test_rope_length_mismatch_raises(self):
    """freqs_cis.shape[2] must match both Sq and Sk rather than silently slicing."""
    b, sq, q_heads, dim_head = 1, 8, 2, 16
    d = q_heads * dim_head
    raw_q = jnp.ones((b, sq, d), jnp.bfloat16)
    raw_k = jnp.ones((b, sq, d), jnp.bfloat16)
    scale = jnp.ones((d,), jnp.bfloat16)
    longer_freqs = jnp.ones((1, 1, sq + 4, dim_head // 2), jnp.complex64)
    with self.assertRaisesRegex(ValueError, "freqs_cis has"):
      fused_rmsnorm_rope(raw_q, raw_k, scale, scale, longer_freqs, q_heads=q_heads, dim_head=dim_head)

    raw_k_short = jnp.ones((b, sq - 2, d), jnp.bfloat16)
    matching_q_freqs = jnp.ones((1, 1, sq, dim_head // 2), jnp.complex64)
    with self.assertRaisesRegex(ValueError, "freqs_cis has"):
      fused_rmsnorm_rope(raw_q, raw_k_short, scale, scale, matching_q_freqs, q_heads=q_heads, dim_head=dim_head)

  def test_flax_wan_attention_module_wiring_and_unfused_parity(self):
    """Exercises FlaxWanAttention.__call__ fused self-attention branch against the unfused path."""
    b, seq, heads, dim_head = 1, 32, 2, 64
    query_dim = heads * dim_head
    eps = 1e-5  # distinct from flax default 1e-6 so separate q_eps/k_eps wiring is tested
    attn = FlaxWanAttention(
        rngs=nnx.Rngs(0),
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        eps=eps,
        qk_norm="rms_norm_across_heads",
        attention_kernel="dot_product",
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
        is_self_attention=True,
    )
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(123), 5)
    attn.norm_q.scale.value = jax.random.normal(k1, (query_dim,), jnp.bfloat16)
    attn.norm_k.scale.value = jax.random.normal(k2, (query_dim,), jnp.bfloat16)
    hidden = jax.random.normal(k3, (b, seq, query_dim), jnp.bfloat16)
    angles = jax.random.uniform(k4, (1, 1, seq, dim_head // 2), minval=-3.14, maxval=3.14)
    rotary_emb = jnp.exp(1j * angles).astype(jnp.complex64)

    with mock.patch.object(attention_flax, "fused_rmsnorm_rope", wraps=fused_rmsnorm_rope) as spy:
      out_fused = attn(hidden_states=hidden, rotary_emb=rotary_emb, deterministic=True)
      self.assertEqual(spy.call_count, 1)
      self.assertEqual(spy.call_args.kwargs["eps"], eps)
      self.assertEqual(spy.call_args.kwargs["k_eps"], 1e-6)

      # Cross-attention (distinct encoder_hidden_states object) must not enter the fused branch.
      enc = jax.random.normal(k5, (b, seq, query_dim), jnp.bfloat16)
      _ = attn(hidden_states=hidden, encoder_hidden_states=enc, rotary_emb=None, deterministic=True)
      self.assertEqual(spy.call_count, 1)

    # Unfused reference path: norm_q/norm_k -> _unflatten_heads -> jit(_apply_rope) -> apply_attention -> proj_attn.
    q_ref = _unflatten_heads(attn.norm_q(attn.query(hidden)), heads)
    k_ref = _unflatten_heads(attn.norm_k(attn.key(hidden)), heads)
    v_ref = _unflatten_heads(attn.value(hidden), heads)
    q_ref, k_ref = jax.jit(attn._apply_rope)(q_ref, k_ref, rotary_emb)
    out_ref = attn.proj_attn(attn.attention_op.apply_attention(q_ref, k_ref, v_ref))
    np.testing.assert_array_equal(
        np.asarray(out_fused.astype(jnp.float32)),
        np.asarray(out_ref.astype(jnp.float32)),
    )


if __name__ == "__main__":
  unittest.main()
