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
precomputed Cauchy-Schwarz bound for eligible (head, Q-block) tiles, falling
back to online softmax for "sink" tiles whose bound exceeds the no-flush gate. These
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
      recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
      bound = qn * mk_h
      eligible = (bound <= safe_bound).astype(jnp.float32)
      m_base = jnp.ceil(bound) - recenter
      num_q_blocks = self.seq_len // self.block_sizes.block_q
      mk = jnp.stack([
          jnp.broadcast_to(m_base[:, None], (self.num_heads, num_q_blocks)),
          jnp.broadcast_to(eligible[:, None], (self.num_heads, num_q_blocks)),
      ])
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
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
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
    self.assertTrue(bool(jnp.all(mk[1][0] == 0.0)))  # head 0 is a sink -> ineligible
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
    """Per-sample mk rows from `_compute_fixed_m_metadata` stay independent under vmap.

    An outlier in sample 0 disqualifies only sample 0's tile in `mk`; sample 1
    stays fully eligible. Note that production's `all_fixed` is reduced over
    the whole batch, so the outlier still routes the batch to the hybrid
    kernel, where each sample's tiles follow their own `mk`.
    """
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

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

    # Keys are already centered above, so no k_mean is passed.
    mk_arr, all_fixed = _compute_fixed_m_metadata(q_in, k_in, block_q=self.block_sizes.block_q)
    self.assertEqual(mk_arr.shape, (2, 2, self.num_heads, self.seq_len // self.block_sizes.block_q))

    # Sample 0: only head 0, block 1 is disqualified.
    self.assertEqual(float(mk_arr[0, 1, 0, 1]), 0.0)
    self.assertEqual(float(mk_arr[0, 1, 0, 0]), 1.0)
    self.assertTrue(bool(jnp.all(mk_arr[0, 1, 1:] > 0.5)))
    # Sample 1: every head and block stays eligible.
    self.assertTrue(bool(jnp.all(mk_arr[1, 1] > 0.5)))
    # The batch-wide predicate is False, so production would run the hybrid kernel.
    self.assertFalse(bool(all_fixed))

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

  def test_legacy_2d_mk_raises_value_error(self):
    """Passing a legacy 2D mk array (2, heads) raises ValueError rather than misinterpreting max||k|| as m_B."""
    q, k, v = self._random_qkv()
    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
    )
    legacy_mk = jnp.zeros((2, self.num_heads), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      kernel(q, k, v, mk=legacy_mk)

  def test_phase_transition_boundary_continuity(self):
    """Output stays continuous as the Cauchy-Schwarz bound sweeps across the fixed-m gate.

    Cases well below the gate must run fixed-m and cases well above must fall
    back; the cases within bf16 rounding of the gate may land on either side.
    """
    q_base, k_base, v = self._random_qkv()
    q_normed = q_base / jnp.sqrt((q_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))
    k_normed = k_base / jnp.sqrt((k_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))

    _, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
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

      out_gated, mk = self._run_kernel_per_q_block(q, k, v)
      if target_bound <= safe_bound - 2.0:
        self.assertTrue(bool(jnp.all(mk[1] > 0.5)), f"expected fixed-m at bound={target_bound}")
      if target_bound >= safe_bound + 2.0:
        self.assertTrue(bool(jnp.all(mk[1] == 0.0)), f"expected fallback at bound={target_bound}")
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
    headroom_bits = math.ceil(math.log2(custom_splash.DEFAULT_MAX_V_BOUND))  # |V| <= 256 -> 8 bits

    for n in test_lengths:
      # 1. Pure Ulysses (Two-sided bound): M in [-U, U]
      recenter, safe_bound = custom_splash.get_fixed_m_constants(n)
      # Minimal shifted exponent at worst-case extremum M = -U
      exponent_centered = -safe_bound - (math.ceil(safe_bound) - recenter)
      self.assertGreaterEqual(
          exponent_centered,
          -125.0,
          f"Underflow violation on Ulysses: {exponent_centered=} for N={n}",
      )
      # Non-overflow check with the default |V| headroom: ceil(log2(N)) + C(N) + headroom_bits <= 127
      max_accum_bits = math.ceil(math.log2(n))
      self.assertLessEqual(
          max_accum_bits + recenter + headroom_bits,
          127.0,
          f"Overflow violation on Ulysses: max bits={max_accum_bits + recenter + headroom_bits} for N={n}",
      )

      # 2. Ring Attention (Uncentered across R hops): M >= -U
      for ring_size in [2, 4, 8]:
        n_total = n * ring_size
        ring_recenter, ring_safe_bound = custom_splash.get_fixed_m_constants(n_total)
        # Minimal shifted exponent at worst-case extremum M = -U
        exponent_ring = -ring_safe_bound - (math.ceil(ring_safe_bound) - ring_recenter)
        self.assertGreaterEqual(
            exponent_ring,
            -125.0,
            f"Underflow violation on Ring (R={ring_size}): {exponent_ring=} for N={n}",
        )
        # Ring direct accumulation non-overflow check: ceil(log2(N_total)) + C_ring + headroom_bits <= 127
        ring_max_bits = math.ceil(math.log2(n_total))
        self.assertLessEqual(
            ring_max_bits + ring_recenter + headroom_bits,
            127.0,
            f"Overflow violation on Ring (R={ring_size}): max bits={ring_max_bits + ring_recenter + headroom_bits} for N={n}",
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
    recenter, safe_bound = custom_splash.get_fixed_m_constants(seq_len)

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
    self.assertLess(
        diff,
        2e-2,
        f"Non-divisible sequence output diverged from reference: {diff=}",
    )

  def test_anti_aligned_uncentered_keys_heavily_negative_logits(self):
    """Fixed-m stays exact when every logit is heavily negative and keys are NOT centered.

    This is the two-sided worst case the floor(W/2) gate exists for: q and k
    are anti-aligned, so every base-2 logit sits near -U (row max ~ -109) while
    U stays just under the gate. With m = ceil(U) - C(N) the shifted exponents
    land near -117, close to but above the -126 flush floor. No k_mean is
    passed, so nothing re-centers the logits.
    """
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    shape = (self.num_heads, self.seq_len, self.head_dim)
    # Kernel-domain inputs (already base-2 scaled); all values are exact in bf16.
    q_sign = jnp.where(jax.random.bernoulli(jax.random.PRNGKey(7), 0.5, shape[:2]), 0.5, -0.5)
    q_in = jnp.zeros(shape, jnp.float32).at[:, :, 0].set(10.5).at[:, :, 1].set(q_sign).astype(jnp.bfloat16)
    c = jax.random.uniform(jax.random.PRNGKey(8), shape[:2], jnp.float32, -2.0, 2.0)
    k_in = jnp.zeros(shape, jnp.float32).at[:, :, 0].set(-10.5).at[:, :, 1].set(c).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(9), shape, jnp.bfloat16)

    bq = self.block_sizes.block_q
    mk_arr, all_fixed = _compute_fixed_m_metadata(q_in[None], k_in[None], block_q=bq)
    _, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
    qf, kf = q_in.astype(jnp.float32), k_in.astype(jnp.float32)
    logits = jnp.einsum("hsd,htd->hst", qf, kf)
    u = float(jnp.sqrt((qf**2).sum(-1).max() * (kf**2).sum(-1).max()))
    self.assertGreater(u, safe_bound - 5.0)  # near the gate ...
    self.assertTrue(bool(all_fixed))  # ... but admitted
    self.assertLess(float(logits.max()), -100.0)  # every logit heavily negative

    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=self.seq_len,
        orig_kv_seq_len=self.seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
    )
    out_fixed = jnp.swapaxes(kernel(q_in, k_in, v, mk_arr[0]), 1, 2).astype(jnp.float32)
    ref = jax.nn.softmax(logits * math.log(2.0), axis=-1) @ v.astype(jnp.float32)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_fixed))))
    self.assertLess(float(jnp.max(jnp.abs(out_fixed - ref))), 2e-2)

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
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
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
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.seq_len)
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


class FixedMDtypeSafetyTest(unittest.TestCase):
  """P2 regression: dtypes that cannot represent 2**C(N) must not use fixed-m.

  Fixed-m parks the un-normalized softmax weights at up to 2**C(N), a range
  derived against FP32's exponent. The kernel narrows them to the activation
  dtype for the S@V matmul, so a dtype with a smaller exponent range overflows
  to inf even when the FP32 bound analysis passes.

  Backend-agnostic on purpose: this gate is pure Python/jnp, so it should be
  enforced in CI even where no TPU is attached.
  """

  def test_float16_is_rejected(self):
    recenter, _ = custom_splash.get_fixed_m_constants(4096)
    # C(4096) with |V| <= 256 is 107; float16 tops out at 2**16.
    self.assertGreater(recenter, 16.0)
    self.assertFalse(custom_splash.fixed_m_dtype_is_safe(jnp.float16, recenter))

  def test_bfloat16_and_float32_are_accepted(self):
    recenter, _ = custom_splash.get_fixed_m_constants(4096)
    self.assertTrue(custom_splash.fixed_m_dtype_is_safe(jnp.bfloat16, recenter))
    self.assertTrue(custom_splash.fixed_m_dtype_is_safe(jnp.float32, recenter))

  def test_gate_tracks_recenter_not_a_hardcoded_allowlist(self):
    """A small enough C(N) is representable even in float16."""
    self.assertTrue(custom_splash.fixed_m_dtype_is_safe(jnp.float16, 4.0))
    self.assertFalse(custom_splash.fixed_m_dtype_is_safe(jnp.float16, 200.0))


class FixedMMetadataSafetyTest(unittest.TestCase):
  """Backend-independent regression tests for fixed-m metadata gating."""

  def test_adversarial_v_magnitude_safely_disqualifies_fixed_m(self):
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    batch = 1
    num_heads = 4
    seq_len = 4096
    dim = 64
    bq = 512

    q = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    k = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    v_overflow = jnp.full((batch, num_heads, seq_len, dim), 512.0, dtype=jnp.bfloat16)

    # With adversarial V=512 (> 256 default bound), fixed_ok must be 0.0, safely falling back to online
    mk_arr, all_fixed = _compute_fixed_m_metadata(q, k, block_q=bq, value=v_overflow)
    self.assertFalse(bool(all_fixed))
    self.assertTrue(bool(jnp.all(mk_arr[:, 1] == 0.0)))

    # With normal V <= 256, fixed_ok should remain 1.0 (all eligible)
    v_normal = jnp.full((batch, num_heads, seq_len, dim), 1.0, dtype=jnp.bfloat16)
    mk_arr_normal, all_fixed_normal = _compute_fixed_m_metadata(q, k, block_q=bq, value=v_normal)
    self.assertTrue(bool(all_fixed_normal))
    self.assertTrue(bool(jnp.all(mk_arr_normal[:, 1] == 1.0)))

  def test_float16_query_disqualifies_fixed_m_metadata(self):
    """fp16, N=4096, Q=K=0, |V|=1 must report all_fixed=False and fixed_ok=0."""
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    batch, num_heads, seq_len, dim, bq = 1, 2, 4096, 128, 512
    q = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.float16)
    k = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.float16)
    v = jnp.full((batch, num_heads, seq_len, dim), 1.0, dtype=jnp.float16)

    mk_arr, all_fixed = _compute_fixed_m_metadata(q, k, block_q=bq, value=v)
    self.assertFalse(bool(all_fixed), "fp16 must not be eligible for fixed-m")
    self.assertTrue(bool(jnp.all(mk_arr[:, 1] == 0.0)))

  def test_bfloat16_same_case_remains_eligible(self):
    """Control: the identical case in bf16 must still take the fast path."""
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    batch, num_heads, seq_len, dim, bq = 1, 2, 4096, 128, 512
    q = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    k = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    v = jnp.full((batch, num_heads, seq_len, dim), 1.0, dtype=jnp.bfloat16)

    _, all_fixed = _compute_fixed_m_metadata(q, k, block_q=bq, value=v)
    self.assertTrue(bool(all_fixed))

  def test_adversarial_centered_keys_softmax_mass_loss(self):
    """Adversarial regression: centered keys with large query norm must NOT be eligible for fixed-m.

    If admitted under a loose one-sided bound (e.g. safe_bound ~ 232), the negative-logit terms
    (1/17 ~ 5.9% of the softmax mass) flush to zero in FP32 (S - m_base < -126), so the output
    collapses to 1.0 instead of 15/17 ~ 0.882 (a silent ~0.118 error). The tightened two-sided bound (safe_bound = safe_window // 2) rejects
    this input, ensuring safe fallback to online softmax.
    """
    from maxdiffusion.models.attention_flax import _compute_fixed_m_metadata

    seq_len = 4096
    dim = 128
    num_heads = 1
    batch = 1
    bq = 512

    # Query: [231, 2, 0, ...]
    q = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    q = q.at[:, :, :, 0].set(231.0).at[:, :, :, 1].set(2.0)

    # Half keys: [0, 1, 0, ...], half keys: [0, -1, 0, ...] (centered, mean = 0)
    k = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    k = k.at[:, :, : seq_len // 2, 1].set(1.0).at[:, :, seq_len // 2 :, 1].set(-1.0)

    # Values: half +1, half -1
    v = jnp.zeros((batch, num_heads, seq_len, dim), dtype=jnp.bfloat16)
    v = v.at[:, :, : seq_len // 2, :].set(1.0).at[:, :, seq_len // 2 :, :].set(-1.0)

    k_mean = jnp.mean(k.astype(jnp.float32), axis=2)
    mk_arr, all_fixed = _compute_fixed_m_metadata(q, k, block_q=bq, k_mean=k_mean, value=v)

    # Must be flagged ineligible for fixed-m under the tightened two-sided bound
    self.assertFalse(
        bool(all_fixed),
        "Adversarial centered input must not be eligible for fixed-m",
    )
    self.assertTrue(
        bool(jnp.all(mk_arr[:, 1] == 0.0)),
        "fixed_ok predicate must be 0.0 across all blocks",
    )

    # Verify that the kernel safely falls back to online softmax: the output is
    # 15/17 (~0.8824) rather than the flushed 1.0.
    block_sizes = custom_splash._BlockSizes(block_q=bq, block_kv=1024, block_kv_compute=512, block_kv_compute_in=256)
    out = custom_splash._splash_attention_forward(
        q[0],
        k[0],
        v[0],
        block_sizes=block_sizes,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        use_base2_exp=True,
        use_fixed_m=True,
        mk=mk_arr[0],
        k_mean=k_mean[0],
    )
    # Transpose from (heads, dim, seq_len) -> (heads, seq_len, dim)
    out = jnp.swapaxes(out, 1, 2)
    expected = 15.0 / 17.0
    self.assertLess(
        float(jnp.max(jnp.abs(out.astype(jnp.float32) - expected))),
        5e-3,
        f"Fallback online softmax output must be ~15/17 (~0.8824), got {float(out[0, 0, 0]):.6f}",
    )


class FixedMAttentionFlaxIntegrationTest(unittest.TestCase):
  """End-to-end fixed-m through the attention_flax entry points.

  The tests above drive the kernel directly. These go through the production
  wrappers (`_ulysses_attention` and the R=1 branch of
  `_ulysses_ring_custom_attention`), which build the metadata, pick the
  uniform/hybrid kernel with `lax.cond` and call `make_splash_mha`. They run on
  a 1-device mesh, so on CPU the Pallas kernel runs in interpret mode.
  """

  batch = 1
  seq_len = 256
  num_heads = 2
  head_dim = 128

  def setUp(self):
    super().setUp()
    from flax.linen import partitioning as nn_partitioning
    import numpy as np
    from jax.sharding import Mesh
    from maxdiffusion.models import attention_flax

    self.af = attention_flax
    self.nn_partitioning = nn_partitioning
    self.mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    self.rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
    )
    self.axis_names_q = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_Q_LENGTH,
        attention_flax.D_KV,
    )
    self.axis_names_kv = (
        attention_flax.BATCH,
        attention_flax.SELF_ATTN_HEAD,
        attention_flax.SELF_ATTN_KV_LENGTH,
        attention_flax.D_KV,
    )
    self.block_sizes = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128, "block_kv_compute_in": 128}

    b, s, h, d = self.batch, self.seq_len, self.num_heads, self.head_dim
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
    self.q = jax.random.normal(kq, (b, s, h * d), jnp.float32).astype(jnp.bfloat16)
    # The wrappers do not apply 1/sqrt(d); fold it into K like the Wan caller does.
    self.k = (jax.random.normal(kk, (b, s, h * d), jnp.float32) * d**-0.5).astype(jnp.bfloat16)
    self.v = jax.random.normal(kv, (b, s, h * d), jnp.float32).astype(jnp.bfloat16)

  def _reference(self) -> jax.Array:
    """Plain fp32 softmax attention on the same bf16 inputs, shape (b, s, h*d)."""
    b, s, h, d = self.batch, self.seq_len, self.num_heads, self.head_dim
    q, k, v = (x.astype(jnp.float32).reshape(b, s, h, d) for x in (self.q, self.k, self.v))
    probs = jax.nn.softmax(jnp.einsum("bshd,bthd->bhst", q, k), axis=-1)
    return jnp.einsum("bhst,bthd->bshd", probs, v).reshape(b, s, h * d)

  def _assert_inputs_take_fixed_m(self):
    """Guards against the test silently exercising only the online fallback."""
    b, s, h, d = self.batch, self.seq_len, self.num_heads, self.head_dim
    q = (self.q.reshape(b, s, h, d).transpose(0, 2, 1, 3) * _LOG2E).astype(jnp.bfloat16)
    k = self.k.reshape(b, s, h, d).transpose(0, 2, 1, 3)
    k_mean = jnp.mean(k.astype(jnp.float32), axis=2)
    _, all_fixed = self.af._compute_fixed_m_metadata(q, k, block_q=128, k_mean=k_mean, value=self.v)
    self.assertTrue(bool(all_fixed))

  def _assert_close_to_reference(self, out: jax.Array):
    self.assertEqual(out.shape, self.q.shape)
    out = out.astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    # Measured max|err| is ~5e-3 on CPU interpret mode, mostly the bf16 rounding
    # of an O(1) output (half an ulp at [1, 2) is ~3.9e-3); 1e-2 leaves headroom.
    self.assertLess(float(jnp.max(jnp.abs(out - self._reference()))), 1e-2)

  def test_ulysses_attention_fixed_m_matches_reference(self):
    self._assert_inputs_take_fixed_m()
    for per_q_block in (True, False):
      with self.subTest(per_q_block=per_q_block):
        with self.mesh, self.nn_partitioning.axis_rules(self.rules):
          out = self.af._ulysses_attention(
              self.q,
              self.k,
              self.v,
              heads=self.num_heads,
              mesh=self.mesh,
              axis_names_q=self.axis_names_q,
              axis_names_kv=self.axis_names_kv,
              flash_block_sizes=self.block_sizes,
              dtype=jnp.bfloat16,
              use_custom_kernel=True,
              use_fixed_m=True,
              per_q_block=per_q_block,
          )
        self._assert_close_to_reference(out)

  def test_ulysses_ring_custom_r1_fixed_m_matches_reference(self):
    self._assert_inputs_take_fixed_m()
    with self.mesh, self.nn_partitioning.axis_rules(self.rules):
      out = self.af._ulysses_ring_custom_attention(
          self.q,
          self.k,
          self.v,
          heads=self.num_heads,
          mesh=self.mesh,
          axis_names_q=self.axis_names_q,
          axis_names_kv=self.axis_names_kv,
          flash_block_sizes=self.block_sizes,
          dtype=jnp.bfloat16,
          ulysses_shards=1,
          use_fixed_m=True,
      )
    self._assert_close_to_reference(out)


if __name__ == "__main__":
  unittest.main()
