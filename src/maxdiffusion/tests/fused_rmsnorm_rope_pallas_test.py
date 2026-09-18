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

"""Correctness tests for the fused RMSNorm + RoPE + head-transpose Pallas kernel.

The kernel is a *pure fusion* of `fused_producers.fused_rmsnorm_rope`, so every
test here pins numerics rather than performance. Tests that only exercise the
algebra run anywhere via Pallas interpret mode; tests that must exercise the
Mosaic lowering (`pltpu.roll`, the dynamic lane-tile slice) are skipped off TPU.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from flax import nnx

from maxdiffusion.kernels.fused_producers import fused_rmsnorm_rope
from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import (
    fused_rmsnorm_rope_pallas,
    rope_pair_swap_reference,
)

DIM_HEAD = 128


def _on_tpu() -> bool:
  return jax.devices()[0].platform == "tpu"


def _make_inputs(seed, batch, seq, q_heads, kv_heads, dim_head, dtype):
  d_q = q_heads * dim_head
  d_k = kv_heads * dim_head
  k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 5)
  raw_q = jax.random.normal(k0, (batch, seq, d_q), jnp.float32).astype(dtype)
  raw_k = jax.random.normal(k1, (batch, seq, d_k), jnp.float32).astype(dtype)
  q_scale = jax.random.normal(k2, (d_q,), jnp.float32)
  k_scale = jax.random.normal(k3, (d_k,), jnp.float32)
  # A genuine unit-modulus rotation, matching how WanRotaryPosEmbed builds
  # freqs_cis; a degenerate table would hide sign/ordering bugs.
  angle = jax.random.uniform(k4, (1, 1, seq, dim_head // 2), jnp.float32, minval=-np.pi, maxval=np.pi)
  freqs_cis = jnp.cos(angle) + 1j * jnp.sin(angle)
  return raw_q, raw_k, q_scale, k_scale, freqs_cis


class PairSwapIdentityTest(unittest.TestCase):
  """The lane-rotation trick must reproduce the strided pair swap exactly."""

  def test_pair_swap_matches_strided_gather(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5, DIM_HEAD), jnp.float32)
    got = rope_pair_swap_reference(x)

    pairs = x.reshape(3, 5, DIM_HEAD // 2, 2)
    want = jnp.stack([pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_pair_swap_wraps_correctly_at_both_ends(self):
    """Circular rotation is only valid because dim_head is even; pin that."""
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 8)
    np.testing.assert_array_equal(
        np.asarray(rope_pair_swap_reference(x)),
        np.array([[1, 0, 3, 2, 5, 4, 7, 6]], dtype=np.float32),
    )


class FusedRmsNormRopePallasParityTest(parameterized.TestCase):
  """Parity against the unfused reference producer, for both norm modes.

  The admissible deviation differs per mode, and both bounds are deliberately
  tight enough to catch a real algebra bug:

  `norm_mode="exact"` keeps the FP32 feature-axis reduction in XLA, so the
  normalisation is bit-identical by construction (verified directly: under an
  identity rotation this path matches the reference exactly, in both bf16 and
  fp32). What remains is the RoPE `a*cos + b*sin`, which the kernel lowering
  may contract into a fused multiply-add where the reference does not. In
  bfloat16 -- the dtype Wan actually runs in -- the wider intermediate rounds
  back to the same bf16 value, so bf16 must be *bit-identical*. In float32
  there is no such absorption, so a single rounding of drift is allowed.

  `norm_mode="fused"` additionally moves the reduction into Mosaic, whose
  summation tree may differ from XLA's, which can flip the final rounding.

  Drift is measured relative to the norm of the *rotated pair*, not of the
  individual component. RoPE rotates each `(x[2i], x[2i+1])` 2-vector, so that
  pair norm is the rotation invariant, and the rounding error of either output
  component is bounded by eps times it. An individual component, by contrast,
  is free to be arbitrarily close to zero, which would make a
  component-relative bound meaningless.
  """

  # Roundings of slack allowed on top of the exactness each mode guarantees.
  _ULP_BUDGET = {
      ("exact", jnp.bfloat16): 0,  # bit-identical: FMA drift is absorbed by bf16 rounding
      ("exact", jnp.float32): 1,  # multiply-add contraction in the RoPE add
      ("fused", jnp.bfloat16): 1,  # + Mosaic's reduction tree flipping the final rounding
      ("fused", jnp.float32): 2,
  }

  @staticmethod
  def _pair_norm(a):
    """Norm of each RoPE 2-vector, broadcast back over both of its lanes."""
    pairs = a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
    norm = np.sqrt(np.sum(np.square(pairs.astype(np.float64)), axis=-1, keepdims=True))
    return np.repeat(norm, 2, axis=-1).reshape(a.shape).astype(np.float32)

  def _assert_parity(self, name, ref, got, dtype, norm_mode):
    self.assertEqual(ref.shape, got.shape, f"{name} shape mismatch")
    self.assertEqual(ref.dtype, got.dtype, f"{name} dtype mismatch")
    ref_np = np.asarray(ref, np.float32)
    got_np = np.asarray(got, np.float32)

    budget = self._ULP_BUDGET[(norm_mode, jnp.dtype(dtype).type)]
    if budget == 0:
      np.testing.assert_array_equal(
          ref_np,
          got_np,
          err_msg=f"{name}: norm_mode={norm_mode!r} in {jnp.dtype(dtype).name} must be bit-identical.",
      )
      return

    tol = budget * float(jnp.finfo(dtype).eps) * self._pair_norm(ref_np)
    drift = np.abs(ref_np - got_np)
    n_bad = int(np.count_nonzero(drift > tol))
    self.assertEqual(
        n_bad,
        0,
        f"{name}: norm_mode={norm_mode!r} in {jnp.dtype(dtype).name} exceeded its "
        f"{budget}-ULP budget on {n_bad}/{ref_np.size} elements "
        f"(worst excess ratio {float(np.max(drift / np.maximum(tol, np.finfo(np.float32).tiny))):.2f}x).",
    )

  @parameterized.named_parameters(
      {"testcase_name": f"_{tag}_{mode}", "q_heads": q, "kv_heads": kv, "dtype": dt, "norm_mode": mode}
      for tag, q, kv, dt in (
          ("mha_bf16", 4, 4, jnp.bfloat16),
          ("gqa_bf16", 8, 2, jnp.bfloat16),
          ("mha_f32", 2, 2, jnp.float32),
      )
      for mode in ("exact", "fused")
  )
  def test_matches_reference(self, q_heads, kv_heads, dtype, norm_mode):
    seq = 96
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(0, 1, seq, q_heads, kv_heads, DIM_HEAD, dtype)

    ref_q, ref_k = fused_rmsnorm_rope(
        raw_q, raw_k, q_scale, k_scale, freqs, q_heads=q_heads, kv_heads=kv_heads, dim_head=DIM_HEAD
    )
    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=32,
        interpret=not _on_tpu(),
    )

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      self._assert_parity(name, ref, got, dtype, norm_mode)

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_handles_sequence_not_divisible_by_block(self, norm_mode):
    """Wan's 18,900-token shard is not a multiple of any power-of-two tile."""
    seq, q_heads = 50, 3
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(1, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)

    ref_q, ref_k = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=q_heads, dim_head=DIM_HEAD)
    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
        interpret=not _on_tpu(),
    )

    self._assert_parity("q", ref_q, got_q, jnp.bfloat16, norm_mode)
    self._assert_parity("k", ref_k, got_k, jnp.bfloat16, norm_mode)

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_result_is_independent_of_block_size(self, norm_mode):
    seq, q_heads = 64, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(2, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)

    outs = []
    for block_s in (16, 32, 64):
      q, _ = fused_rmsnorm_rope_pallas(
          raw_q,
          raw_k,
          q_scale,
          k_scale,
          freqs,
          q_heads=q_heads,
          dim_head=DIM_HEAD,
          norm_mode=norm_mode,
          block_s=block_s,
          interpret=not _on_tpu(),
      )
      outs.append(np.asarray(q, np.float32))
    for other in outs[1:]:
      np.testing.assert_array_equal(outs[0], other, err_msg="Tiling must not change the result.")

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_rmsnorm_component_matches_flax(self, norm_mode):
    """Under an identity rotation the kernel must reproduce nnx.RMSNorm exactly."""
    seq, q_heads = 32, 2
    d_model = q_heads * DIM_HEAD
    k0, k1 = jax.random.split(jax.random.PRNGKey(3))
    raw_q = jax.random.normal(k0, (1, seq, d_model), jnp.float32)
    q_scale = jax.random.normal(k1, (d_model,), jnp.float32)
    freqs = jnp.ones((1, 1, seq, DIM_HEAD // 2), jnp.complex64)  # cos=1, sin=0

    layer = nnx.RMSNorm(d_model, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    layer.scale.value = q_scale
    want = layer(raw_q).reshape(1, seq, q_heads, DIM_HEAD).transpose(0, 2, 1, 3)

    got, _ = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_q,
        q_scale,
        q_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
        interpret=not _on_tpu(),
    )
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(("_bf16", jnp.bfloat16), ("_f32", jnp.float32))
  def test_exact_mode_normalisation_alone_is_bit_identical(self, dtype):
    """With the rotation switched off, `exact` must match bit-for-bit in *both* dtypes.

    This localises the float32 slack granted in `_ULP_BUDGET`. Under an
    identity rotation the kernel degenerates to `x * (rsqrt * scale)` with no
    `a*cos + b*sin` anywhere, so no multiply-add contraction is possible. If
    this test ever fails, the normalisation itself has drifted and the float32
    budget is masking a real bug rather than a benign contraction.
    """
    seq, q_heads = 32, 2
    raw_q, raw_k, q_scale, k_scale, _ = _make_inputs(8, 1, seq, q_heads, q_heads, DIM_HEAD, dtype)
    identity = jnp.ones((1, 1, seq, DIM_HEAD // 2), jnp.complex64)  # cos = 1, sin = 0

    ref_q, ref_k = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, identity, q_heads=q_heads, dim_head=DIM_HEAD)
    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        identity,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode="exact",
        block_s=16,
        interpret=not _on_tpu(),
    )

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      np.testing.assert_array_equal(
          np.asarray(ref, np.float32),
          np.asarray(got, np.float32),
          err_msg=f"{name}: exact-mode RMSNorm must be bit-identical in {jnp.dtype(dtype).name}.",
      )

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_rope_preserves_per_head_norm(self, norm_mode):
    """RoPE is a rotation, so it must not change the per-position head norm."""
    seq, q_heads = 32, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(4, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.float32)

    got_q, _ = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
        interpret=not _on_tpu(),
    )
    pre = jnp.linalg.norm(
        (raw_q * (jax.lax.rsqrt(jnp.mean(raw_q**2, -1, keepdims=True) + 1e-6) * q_scale))
        .reshape(1, seq, q_heads, DIM_HEAD)
        .transpose(0, 2, 1, 3),
        axis=-1,
    )
    post = jnp.linalg.norm(got_q.astype(jnp.float32), axis=-1)
    np.testing.assert_allclose(np.asarray(pre), np.asarray(post), rtol=2e-5, atol=2e-5)


class FusedRmsNormRopePallasGuardTest(unittest.TestCase):
  """Misconfiguration must fail loudly rather than silently produce garbage."""

  def _inputs(self, dim_head, q_heads=2, seq=16):
    return _make_inputs(5, 1, seq, q_heads, q_heads, dim_head, jnp.bfloat16)

  def test_rejects_unaligned_dim_head(self):
    raw_q, raw_k, q_scale, k_scale, freqs = self._inputs(64)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      fused_rmsnorm_rope_pallas(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=2, dim_head=64, interpret=not _on_tpu())

  def test_rejects_feature_dim_mismatch(self):
    raw_q, raw_k, q_scale, k_scale, freqs = self._inputs(DIM_HEAD)
    with self.assertRaisesRegex(ValueError, "raw_q feature dim"):
      fused_rmsnorm_rope_pallas(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=3, dim_head=DIM_HEAD, interpret=not _on_tpu())

  def test_rejects_freqs_cis_mismatch(self):
    raw_q, raw_k, q_scale, k_scale, _ = self._inputs(DIM_HEAD)
    bad_freqs = jnp.ones((1, 1, 16, 8), jnp.complex64)
    with self.assertRaisesRegex(ValueError, "freqs_cis last dim"):
      fused_rmsnorm_rope_pallas(
          raw_q, raw_k, q_scale, k_scale, bad_freqs, q_heads=2, dim_head=DIM_HEAD, interpret=not _on_tpu()
      )


class FusedRmsNormRopePallasProductionShapeTest(unittest.TestCase):
  """Exercises the real Wan 2.2 per-shard shape on TPU.

  18,900 tokens x 40 heads x 128 is the v6e-8 / tpu7x-8 per-context-shard
  self-attention projection. It is deliberately not a multiple of 8, so it also
  covers the ragged trailing sequence tile at full width.
  """

  SEQ = 18900
  HEADS = 40

  def _run(self, norm_mode):
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(6, 1, self.SEQ, self.HEADS, self.HEADS, DIM_HEAD, jnp.bfloat16)
    ref = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=self.HEADS, dim_head=DIM_HEAD)
    got = fused_rmsnorm_rope_pallas(
        raw_q, raw_k, q_scale, k_scale, freqs, q_heads=self.HEADS, dim_head=DIM_HEAD, norm_mode=norm_mode
    )
    return ref, got

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  def test_exact_mode_is_bit_identical(self):
    (ref_q, ref_k), (got_q, got_k) = self._run("exact")

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      ref_np = np.asarray(ref, np.float32)
      got_np = np.asarray(got, np.float32)
      mismatches = int(np.count_nonzero(ref_np != got_np))
      self.assertEqual(
          mismatches,
          0,
          f"{name}: {mismatches}/{ref_np.size} elements differ "
          f"(max |diff| = {float(np.max(np.abs(ref_np - got_np))):.3e}).",
      )

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  def test_fused_mode_drift_is_at_most_one_ulp(self):
    """Quantifies, rather than forbids, the Mosaic reduction-tree difference.

    This is the measurement that justifies `exact` being the default: the
    fused reduction is not hash-identical, and Wan's 40 autoregressive denoise
    steps amplify even a single flipped bf16 rounding.
    """
    (ref_q, ref_k), (got_q, got_k) = self._run("fused")

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      ref_np = np.asarray(ref, np.float32)
      got_np = np.asarray(got, np.float32)
      drift = np.abs(ref_np - got_np)
      tol = float(jnp.finfo(jnp.bfloat16).eps) * FusedRmsNormRopePallasParityTest._pair_norm(ref_np)
      mismatches = int(np.count_nonzero(ref_np != got_np))
      over_ulp = int(np.count_nonzero(drift > tol))
      print(
          f"[fused/{name}] {mismatches}/{ref_np.size} elements differ "
          f"({100.0 * mismatches / ref_np.size:.6f}%), max |diff| = {float(drift.max()):.3e}"
      )
      self.assertEqual(over_ulp, 0, f"{name}: {over_ulp} elements drifted by more than 1 bf16 ULP.")


if __name__ == "__main__":
  unittest.main()
