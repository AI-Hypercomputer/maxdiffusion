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
    self.block_sizes = custom_splash._BlockSizes(block_q=2048, block_kv=1024, block_kv_compute=512, block_kv_compute_in=256)

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
      recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
      bound = qn * mk_h
      eligible = (bound <= safe_bound).astype(jnp.float32)
      m_base = jnp.ceil(bound) - recenter
      mk = jnp.stack([m_base, eligible])
    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
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

  def _run_kernel_per_q_block(
      self, q: jax.Array, k: jax.Array, v: jax.Array, uniform_fixed_m: bool = False
  ) -> tuple[jax.Array, jax.Array]:
    """Runs the custom kernel with 3D per-Q-block mk inputs."""
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in = k * self.scale
    k_in = k_in - jnp.mean(k_in, axis=1, keepdims=True)

    bq = self.block_sizes.block_q
    num_q_blocks = self.seq_len // bq
    qf = q_in.astype(jnp.float32)
    kf = k_in.astype(jnp.float32)
    qf_blocks = qf.reshape(self.num_heads, num_q_blocks, bq, self.head_dim)
    qn_max = jnp.sqrt((qf_blocks * qf_blocks).sum(-1)).max(axis=-1)  # (heads, num_q_blocks)
    mk_h = jnp.sqrt((kf * kf).sum(-1)).max(axis=1)  # (heads,)
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
    bound = qn_max * mk_h[:, None]
    fixed_ok = (bound <= safe_bound).astype(jnp.float32)
    m_base = jnp.ceil(bound) - recenter
    mk = jnp.stack([m_base, fixed_ok], axis=0)  # (2, heads, num_q_blocks)

    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=uniform_fixed_m,
    )
    out = kernel(q_in, k_in, v, mk)
    out = jnp.swapaxes(out, 1, 2)
    return out.astype(jnp.float32), mk

  def test_sink_head_falls_back_to_online(self):
    """An out-of-gate head is flagged ineligible and stays finite (no flush)."""
    q, k, v = self._random_qkv(q_gain=6.0, k_gain=6.0)
    fixed, mk = self._run_kernel(q, k, v, use_fixed_m=True)
    self.assertEqual(float(mk[1][0]), 0.0)  # head 0 is a sink -> ineligible
    self.assertTrue(bool(jnp.all(mk[1][1:] > 0.5)))  # the rest stay eligible
    self.assertTrue(bool(jnp.all(jnp.isfinite(fixed))))

  def test_per_q_block_sink_fallback(self):
    """Per-Q-block eligibility keeps normal Q-blocks fixed while sinking outlier blocks."""
    q, k, v = self._random_qkv(k_gain=2.0)
    # Amplify only Q-block 1 of Head 0 (bq = 2048, so indices 2048:4096)
    q = q.at[0, 2048:].multiply(10.0)

    fixed, mk = self._run_kernel_per_q_block(q, k, v)
    # Head 0, Block 0 should be eligible (1.0)
    self.assertEqual(float(mk[1, 0, 0]), 1.0)
    # Head 0, Block 1 should be ineligible (0.0) due to amplified Q outlier
    self.assertEqual(float(mk[1, 0, 1]), 0.0)
    # All other heads should be eligible across both blocks
    self.assertTrue(bool(jnp.all(mk[1, 1:, :] > 0.5)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(fixed))))
    # Check numerical agreement against online kernel running on the same centered inputs
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in_centered = (k * self.scale) - jnp.mean(k * self.scale, axis=1, keepdims=True)
    kernel_online = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=False,
    )
    online_centered = jnp.swapaxes(kernel_online(q_in, k_in_centered, v), 1, 2).astype(jnp.float32)
    self.assertLess(float(jnp.max(jnp.abs(fixed - online_centered))), 1e-2)

  def test_batched_fixed_m_isolation(self):
    """Batch-isolated gating ensures outliers in one sample do not contaminate other samples."""
    q0, k0, v0 = self._random_qkv(k_gain=2.0)
    # Sample 0 has an outlier in Q-block 1 of Head 0
    q0 = q0.at[0, 2048:].multiply(10.0)

    # Sample 1 is completely clean
    q1, k1, v1 = self._random_qkv(k_gain=1.0)

    q = jnp.stack([q0, q1], axis=0)  # (2, heads, seq, dim)
    k = jnp.stack([k0, k1], axis=0)
    v = jnp.stack([v0, v1], axis=0)

    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in = k * self.scale
    k_in = k_in - jnp.mean(k_in, axis=2, keepdims=True)

    bq = self.block_sizes.block_q
    num_q_blocks = self.seq_len // bq
    qf = q_in.astype(jnp.float32)
    kf = k_in.astype(jnp.float32)
    qf_blocks = qf.reshape(2, self.num_heads, num_q_blocks, bq, self.head_dim)
    qn_max_sq = (qf_blocks * qf_blocks).sum(-1).max(axis=-1)  # (2, heads, num_q_blocks)
    mk_h_sq = (kf * kf).sum(-1).max(axis=-1)  # (2, heads)
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
    bound_sq = qn_max_sq * mk_h_sq[:, :, None]
    fixed_ok = (bound_sq <= (safe_bound**2)).astype(jnp.float32)
    m_base = jnp.ceil(jnp.sqrt(bound_sq)) - recenter
    mk_arr = jnp.stack([m_base, fixed_ok], axis=1)  # (2, 2, heads, num_q_blocks)

    # Verify Sample 0 has Head 0 Block 1 disqualified (0.0)
    self.assertEqual(float(mk_arr[0, 1, 0, 1]), 0.0)
    # Verify Sample 1 has ALL heads and ALL blocks eligible (1.0) - zero contamination!
    self.assertTrue(bool(jnp.all(mk_arr[1, 1] > 0.5)))

    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=False,
    )
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, 0, 0, 0))
    out = vmapped_kernel(q_in, k_in, v, mk_arr)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_uniform_fixed_matches_hybrid(self):
    """Uniform-fixed kernel matches hybrid kernel and f32 reference when all eligible."""
    q, k, v = self._random_qkv()
    hybrid_out, mk = self._run_kernel_per_q_block(q, k, v, uniform_fixed_m=False)
    uniform_out, _ = self._run_kernel_per_q_block(q, k, v, uniform_fixed_m=True)
    ref = self._reference(q, k, v)

    self.assertTrue(bool(jnp.all(mk[1] > 0.5)))
    self.assertLess(float(jnp.max(jnp.abs(uniform_out - hybrid_out))), 5e-3)
    self.assertLess(float(jnp.max(jnp.abs(uniform_out - ref))), 2e-2)

  def test_missing_mk_raises_value_error(self):
    """When use_fixed_m=True, omitting mk raises an immediate ValueError."""
    q, k, v = self._random_qkv()
    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
    )
    with self.assertRaises(ValueError):
      kernel(q, k, v, mk=None)

  def test_phase_transition_boundary_continuity(self):
    """Verifies seamless output continuity between fixed-m and online mode across the dynamic safe bound threshold."""
    q_base, k_base, v = self._random_qkv()
    q_normed = q_base / jnp.sqrt((q_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))
    k_normed = k_base / jnp.sqrt((k_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))

    _, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
    test_bounds = [
        safe_bound - 2.0,
        safe_bound - 0.5,
        safe_bound - 0.01,
        safe_bound,
        safe_bound + 0.01,
        safe_bound + 0.5,
        safe_bound + 2.0,
    ]
    for target_bound in test_bounds:
      factor = math.sqrt(target_bound / _LOG2E / self.scale)
      q = (q_normed * factor).astype(jnp.bfloat16)
      k = (k_normed * factor).astype(jnp.bfloat16)

      q_in = (q * _LOG2E).astype(jnp.bfloat16)
      k_in = k * self.scale
      k_in = k_in - jnp.mean(k_in, axis=1, keepdims=True)

      out_gated, _ = self._run_kernel_per_q_block(q, k, v)
      kernel_online = custom_splash.make_splash_mha(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.seq_len,
          orig_kv_seq_len=self.seq_len,
          use_base2_exp=True,
          use_fixed_m=False,
      )
      out_online = jnp.swapaxes(kernel_online(q_in, k_in, v), 1, 2).astype(jnp.float32)

      self.assertTrue(bool(jnp.all(jnp.isfinite(out_gated))))
      diff = float(jnp.max(jnp.abs(out_gated - out_online)))
      self.assertLess(diff, 2e-2, f"Discontinuity at bound={target_bound}, diff={diff}")

  def test_cpu_proof_invariant_bounds(self):
    """Verifies that mathematical underflow and overflow invariants hold across sequence lengths."""
    # Test sequence lengths across short, medium, and production Wan2.2 dimensions
    test_lengths = [1, 2, 512, 1024, 16384, 75600, 151200]

    for n in test_lengths:
      # 1. Pure Ulysses (Centered): M >= 0
      recenter, safe_bound = custom_splash.get_fixed_m_constants(n, is_ring=False)
      # Minimal shifted exponent at the boundary U = safe_bound
      exponent_centered = 0.0 - (math.ceil(safe_bound) - recenter)
      self.assertGreaterEqual(
          exponent_centered,
          -125.0,
          f"Underflow violation on Ulysses: {exponent_centered=} for N={n}",
      )
      # Non-overflow check with explicit 8-bit FP32 output headroom: log2(N) + C(N) + FP32_OUTPUT_HEADROOM_BITS <= 127
      max_accum_bits = math.ceil(math.log2(n))
      self.assertLessEqual(
          max_accum_bits + recenter + custom_splash.FP32_OUTPUT_HEADROOM_BITS,
          127.0,
          f"Overflow violation on Ulysses: max bits={max_accum_bits + recenter + custom_splash.FP32_OUTPUT_HEADROOM_BITS} for N={n}",
      )

      # 2. Ring Attention (Uncentered across R hops): M >= -U
      for ring_size in [2, 4, 8]:
        n_total = n * ring_size
        ring_recenter, ring_safe_bound = custom_splash.get_fixed_m_constants(n_total, is_ring=True)
        # Minimal shifted exponent at worst-case extremum M = -U
        exponent_ring = -ring_safe_bound - (math.ceil(ring_safe_bound) - ring_recenter)
        self.assertGreaterEqual(
            exponent_ring,
            -125.0,
            f"Underflow violation on Ring (R={ring_size}): {exponent_ring=} for N={n}",
        )
        # Ring direct accumulation non-overflow check: log2(N_total) + C_ring + FP32_OUTPUT_HEADROOM_BITS <= 127
        ring_max_bits = math.ceil(math.log2(n_total))
        self.assertLessEqual(
            ring_max_bits + ring_recenter + custom_splash.FP32_OUTPUT_HEADROOM_BITS,
            127.0,
            f"Overflow violation on Ring (R={ring_size}): max bits={ring_max_bits + ring_recenter + custom_splash.FP32_OUTPUT_HEADROOM_BITS} for N={n}",
        )

  def test_non_divisible_sequence_context_padding_fixed_m(self):
    """Verifies that non-divisible sequences (e.g. S=1001 padded for 8 shards) are correctly masked without zero-padding pollution."""
    seq_len = 1001
    context_shards = 8
    rem = seq_len % context_shards
    padded_seq_len = seq_len + (context_shards - rem)  # 1008
    heads = 4
    dim = 64
    bq = 512

    q_raw = jax.random.normal(jax.random.PRNGKey(101), (heads, seq_len, dim), jnp.bfloat16)
    k_raw = jax.random.normal(jax.random.PRNGKey(102), (heads, seq_len, dim), jnp.bfloat16)
    v_raw = jax.random.normal(jax.random.PRNGKey(103), (heads, seq_len, dim), jnp.bfloat16)

    # Reference dense attention on true unpadded inputs
    ref_out = self._reference(q_raw, k_raw, v_raw)

    # Pad inputs as _reshape_data_for_flash would for context sharding
    q_pad = jnp.pad(q_raw, ((0, 0), (0, padded_seq_len - seq_len), (0, 0)))
    k_pad = jnp.pad(k_raw, ((0, 0), (0, padded_seq_len - seq_len), (0, 0)))
    v_pad = jnp.pad(v_raw, ((0, 0), (0, padded_seq_len - seq_len), (0, 0)))

    # Compute unpadded K centering and metadata
    k_mean = jnp.mean(k_raw.astype(jnp.float32) * self.scale, axis=1)  # (heads, dim)
    recenter, safe_bound = custom_splash.get_fixed_m_constants(seq_len, is_ring=False)

    q_in = (q_pad * _LOG2E).astype(jnp.bfloat16)
    k_in = (k_pad * self.scale).astype(jnp.bfloat16)

    num_q_blocks = math.ceil(padded_seq_len / bq)
    # Pad to systolic block_q boundary
    pad_bq = num_q_blocks * bq
    q_in_padded = jnp.pad(q_in, ((0, 0), (0, pad_bq - padded_seq_len), (0, 0)))
    k_in_padded = jnp.pad(k_in, ((0, 0), (0, pad_bq - padded_seq_len), (0, 0)))
    v_in_padded = jnp.pad(v_pad, ((0, 0), (0, pad_bq - padded_seq_len), (0, 0)))

    # Metadata computed on real keys
    k_centered = (k_raw.astype(jnp.float32) * self.scale) - k_mean[:, None, :]
    mk_h = jnp.sqrt((k_centered**2).sum(-1)).max(axis=-1)
    qf_blocks = q_in_padded.astype(jnp.float32).reshape(heads, num_q_blocks, bq, dim)
    qn_max = jnp.sqrt((qf_blocks * qf_blocks).sum(-1)).max(axis=-1)
    bound = qn_max * mk_h[:, None]
    fixed_ok = (bound <= safe_bound).astype(jnp.float32)
    m_base = jnp.ceil(bound) - recenter
    mk = jnp.stack([m_base, fixed_ok], axis=0)

    block_sizes = custom_splash._BlockSizes(block_q=bq, block_kv=bq, block_kv_compute=bq, block_kv_compute_in=bq)
    kernel = custom_splash.make_splash_mha(
        block_sizes=block_sizes,
        orig_q_seq_len=padded_seq_len,
        orig_kv_seq_len=seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=True,
    )
    out = jnp.swapaxes(kernel(q_in_padded, k_in_padded, v_in_padded, mk, k_mean), 1, 2).astype(jnp.float32)
    out_sliced = out[:, :seq_len, :]

    diff = float(jnp.max(jnp.abs(out_sliced - ref_out)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_sliced))))
    self.assertLess(diff, 2e-2, f"Non-divisible sequence output diverged from reference: {diff=}")

  def test_pathological_keys_extreme_negative_logits(self):
    """Verifies stability when logits are heavily negative and close to underflow."""
    q, k, v = self._random_qkv()
    # Shift keys far into negative space so dot products are mostly negative
    k_pathological = k - 30.0
    out_fixed, _ = self._run_kernel_per_q_block(q, k_pathological, v)
    out_online, _ = self._run_kernel(q, k_pathological, v, use_fixed_m=False)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_fixed))))
    diff = float(jnp.max(jnp.abs(out_fixed - out_online)))
    self.assertLess(diff, 2e-2)

  def test_extreme_dynamic_range_inputs(self):
    """Verifies that norm computation and gating remain robust with wide dynamic ranges."""
    shape = (self.num_heads, self.seq_len, self.head_dim)
    scales = jnp.array([1e-3, 0.1, 0.5, 1.0, 1.5])[:, None, None]
    q = (jax.random.normal(jax.random.PRNGKey(42), shape, jnp.bfloat16) * scales).astype(jnp.bfloat16)
    k = (jax.random.normal(jax.random.PRNGKey(43), shape, jnp.bfloat16) * scales).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(44), shape, jnp.bfloat16)

    out, mk = self._run_kernel_per_q_block(q, k, v)
    ref = self._reference(q, k, v)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    self.assertLess(float(jnp.max(jnp.abs(out - ref))), 3e-2)

  def test_virtual_k_centering_matches_explicit(self):
    """Virtual K-centering with raw keys matches explicit K-centering numerically."""
    q, k, v = self._random_qkv()
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in_raw = (k * self.scale).astype(jnp.bfloat16)
    k_mean = jnp.mean(k_in_raw.astype(jnp.float32), axis=1)

    k_in_centered = k_in_raw.astype(jnp.float32) - k_mean[:, None, :]
    mk_h_sq = (k_in_centered**2).sum(axis=-1).max(axis=1)
    mk_h = jnp.sqrt(mk_h_sq)

    bq = self.block_sizes.block_q
    num_q_blocks = self.seq_len // bq
    qf = q_in.astype(jnp.float32)
    qf_blocks = qf.reshape(self.num_heads, num_q_blocks, bq, self.head_dim)
    qn_max = jnp.sqrt((qf_blocks * qf_blocks).sum(-1)).max(axis=-1)
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
    bound = qn_max * mk_h[:, None]
    fixed_ok = (bound <= safe_bound).astype(jnp.float32)
    m_base = jnp.ceil(bound) - recenter
    mk = jnp.stack([m_base, fixed_ok], axis=0)

    # Virtual K-centering with raw uncentered keys
    kernel_virtual = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=True,
    )
    out_virtual = jnp.swapaxes(kernel_virtual(q_in, k_in_raw, v, mk, k_mean), 1, 2).astype(jnp.float32)

    # Explicit centering with centered keys
    k_centered_bf16 = k_in_centered.astype(jnp.bfloat16)
    kernel_explicit = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=True,
    )
    out_explicit = jnp.swapaxes(kernel_explicit(q_in, k_centered_bf16, v, mk), 1, 2).astype(jnp.float32)

    ref = self._reference(q, k, v)
    diff_virtual_explicit = float(jnp.max(jnp.abs(out_virtual - out_explicit)))
    diff_virtual_ref = float(jnp.max(jnp.abs(out_virtual - ref)))

    self.assertLess(diff_virtual_explicit, 2e-3)
    self.assertLess(diff_virtual_ref, 2e-2)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_virtual))))

  def test_virtual_k_centering_per_q_block_hybrid_fallback(self):
    """Exercises Virtual K-Centering + Per-Q-Block Hybrid dispatch with mixed fixed/online tiles."""
    q, k, v = self._random_qkv()
    bq = self.block_sizes.block_q
    num_q_blocks = self.seq_len // bq
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in_raw = (k * self.scale).astype(jnp.bfloat16)
    k_mean = jnp.mean(k_in_raw.astype(jnp.float32), axis=1)

    k_in_centered = k_in_raw.astype(jnp.float32) - k_mean[:, None, :]
    mk_h_sq = (k_in_centered**2).sum(axis=-1).max(axis=1)
    mk_h = jnp.sqrt(mk_h_sq)

    # Test hybrid dispatch where Head 0 Block 0 is Fixed-M and Block 1 is Online Fallback
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len, is_ring=False)
    qf_blocks = q_in.astype(jnp.float32).reshape(self.num_heads, num_q_blocks, bq, self.head_dim)
    qn_max = jnp.sqrt((qf_blocks * qf_blocks).sum(-1)).max(axis=-1)
    bound = qn_max * mk_h[:, None]
    m_base = jnp.ceil(bound) - recenter
    fixed_ok = jnp.ones((self.num_heads, num_q_blocks), dtype=jnp.float32).at[0, 1].set(0.0)
    mk = jnp.stack([m_base, fixed_ok], axis=0)

    # Verify Block 0 is fixed (1.0), Block 1 is online fallback (0.0) on Head 0
    self.assertEqual(float(mk[1, 0, 0]), 1.0)
    self.assertEqual(float(mk[1, 0, 1]), 0.0)

    # Hybrid kernel with raw uncentered keys + k_mean
    kernel_hybrid = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        uniform_fixed_m=False,
    )
    out_hybrid = jnp.swapaxes(kernel_hybrid(q_in, k_in_raw, v, mk, k_mean), 1, 2).astype(jnp.float32)

    # Dense f32 reference
    ref = self._reference(q, k, v)
    diff = float(jnp.max(jnp.abs(out_hybrid - ref)))

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_hybrid))))
    self.assertLess(diff, 2e-2, f"Hybrid virtual K output diverged from reference: diff={diff}")

  def test_gqa_fixed_m_metadata_broadcast(self):
    """Verifies that _compute_fixed_m_metadata correctly handles GQA (num_q_heads != num_kv_heads)."""
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    batch = 2
    num_q_heads = 8
    num_kv_heads = 2
    seq_len = 2048
    dim = 64
    bq = 512
    q = jax.random.normal(jax.random.PRNGKey(10), (batch, num_q_heads, seq_len, dim), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(11), (batch, num_kv_heads, seq_len, dim), jnp.bfloat16)
    mk_arr, all_fixed = _compute_fixed_m_metadata(q, k, block_q=bq)
    expected_blocks = seq_len // bq
    self.assertEqual(mk_arr.shape, (batch, 2, num_q_heads, expected_blocks))
    self.assertTrue(bool(jnp.all(jnp.isfinite(mk_arr))))


if __name__ == "__main__":
  unittest.main()
