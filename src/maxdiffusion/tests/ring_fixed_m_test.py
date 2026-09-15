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

"""Unit tests for the fixed-m path of the custom RING attention.

The ring path gates fixed-m PER (head, K-shard) against the halved
un-smoothed bound, rotates each K shard's max row norm alongside K/V, and
merges the per-hop partials in LSE space (invariant to fixed-m's bound
overshoot). These tests check, against an f32 dense-softmax reference:

  * the untouched online ring path (regression guard),
  * fixed-m with every (head, shard) eligible,
  * a sink head ineligible on every shard (all-online fallback),
  * a head eligible on one shard but not the other -- the mixed
    fixed/online partial case that requires the LSE merge.
"""

import functools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import ring_attention_kernel

_LOG2E = math.log2(math.e)
_RING_AXIS = "ring"
_RING_SIZE = 2


class RingFixedMTest(unittest.TestCase):
  """Numerical tests for the fixed-m custom ring attention."""

  num_heads = 4
  shard_len = 2048  # per-device sequence; total = shard_len * ring_size
  head_dim = 128

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if len(jax.devices()) < _RING_SIZE:
      self.skipTest(f"Requires {_RING_SIZE} devices.")
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.block_sizes = custom_splash._BlockSizes(block_q=1024, block_kv=1024, block_kv_compute=512, block_kv_compute_in=256)
    devices = np.asarray(jax.devices()[:_RING_SIZE])
    self.mesh = jax.sharding.Mesh(devices, (_RING_AXIS,))

  def _random_qkv(self, q_gain=None, k_gain=None):
    """bf16 (q, k, v), [heads, total_seq, dim]; optional (head, row-slice) gains."""
    total = self.shard_len * _RING_SIZE
    shape = (self.num_heads, total, self.head_dim)
    q = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), shape, jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), shape, jnp.bfloat16)
    if q_gain is not None:
      head, rows, gain = q_gain
      q = q.at[head, rows].multiply(gain)
    if k_gain is not None:
      head, rows, gain = k_gain
      k = k.at[head, rows].multiply(gain)
    return q, k, v

  def _scaled_inputs(self, q, k):
    """The EXACT bf16 tensors the kernel sees (attention_flax's contract):
    k pre-scaled by the softmax scale, q pre-scaled by LOG2E (base-2
    kernel). The reference must consume these same tensors -- comparing
    against raw f32 inputs instead double-rounds k, and on an amplified
    head (logits ~2^9) the bf16 rounding alone shifts softmax weights by
    factors of ~2^2, drowning the kernel error being tested."""
    q_in = (q * _LOG2E).astype(q.dtype)
    k_in = (k.astype(jnp.float32) * self.scale).astype(k.dtype)
    return q_in, k_in

  def _reference(self, q_in, k_in, v):
    """Dense f32 log2-domain softmax on the kernel's own bf16 inputs."""
    qf, kf, vf = (x.astype(jnp.float32) for x in (q_in, k_in, v))
    logits = jnp.einsum("hqd,hkd->hqk", qf, kf)  # LOG2E & scale pre-folded
    return jax.nn.softmax(logits * math.log(2.0), axis=-1) @ vf

  def _run_ring(self, q_in, k_in, v, use_fixed_m, norms_squared: bool = True, v_ok_override=None):
    """Runs the custom ring under shard_map with per-rank fixed_m_norms
    from the LOCAL q / initial K shard.

    `norms_squared` selects which representation to hand the kernel. Both are
    valid so long as they are *declared*; the kernel never infers them.
    """
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body(ql, kl, vl):
      fixed_m_norms = None
      v_ok = None
      if use_fixed_m:
        qf = ql.astype(jnp.float32)
        kf = kl.astype(jnp.float32)
        # Squared norms are the kernel's declared default contract. sqrt is
        # monotonic, so max-then-square and square-then-max agree exactly.
        qn_max_sq = (qf * qf).sum(-1).max(axis=1)  # (heads,)
        mk_h_sq = (kf * kf).sum(-1).max(axis=1)  # (heads,) local shard
        if norms_squared:
          fixed_m_norms = (qn_max_sq, mk_h_sq)
        else:
          fixed_m_norms = (jnp.sqrt(qn_max_sq), jnp.sqrt(mk_h_sq))
        if v_ok_override is None:
          # The V/dtype safety verdict the production caller computes. It is
          # global, so it is reduced across the ring before use.
          v_max_sq = (vl.astype(jnp.float32) ** 2).max()
          dtype_safe = custom_splash.fixed_m_dtype_is_safe(ql.dtype, custom_splash._FIXED_M_RECENTER)
          v_ok_local = (v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2)) & dtype_safe
          v_ok = jax.lax.pmin(v_ok_local, axis_name=_RING_AXIS)
        else:
          v_ok = v_ok_override
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=_RING_SIZE,
          use_fixed_m=use_fixed_m,
          fixed_m_norms=fixed_m_norms,
          fixed_m_norms_squared=norms_squared,
          v_ok=v_ok,
          # These norms are per-head, not per-Q-block, which is what the
          # production ring caller supplies. Declaring it keeps the (heads,)
          # array from broadcasting against mk[:, None] into (heads, heads).
          per_q_block=False,
      )
      return ring(ql, kl, vl)

    return _body(q_in, k_in, v)

  def _gate_per_shard(self, q_in, k_in):
    """(heads, ring_size) eligibility against the halved un-smoothed bound."""
    qf = q_in.astype(jnp.float32)
    kf = k_in.astype(jnp.float32)
    qn = jnp.sqrt((qf * qf).sum(-1))  # (heads, total)
    kn = jnp.sqrt((kf * kf).sum(-1))
    gates = []
    for r in range(_RING_SIZE):
      rows = slice(r * self.shard_len, (r + 1) * self.shard_len)
      # Stationary q max is per-RANK, but for the gate check we use the global
      # q max: it upper-bounds every rank's local max, so "eligible globally"
      # implies eligible on every rank.
      bound = qn.max(axis=1) * kn[:, rows].max(axis=1)
      gates.append(bound <= custom_splash._FIXED_M_RING_SAFE_BOUND)
    return jnp.stack(gates, axis=1)

  def _run_and_compare(self, q, k, v, use_fixed_m):
    q_in, k_in = self._scaled_inputs(q, k)
    out = self._run_ring(q_in, k_in, v, use_fixed_m=use_fixed_m).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    return float(jnp.max(jnp.abs(out - self._reference(q_in, k_in, v))))

  def _gate(self, q, k):
    return self._gate_per_shard(*self._scaled_inputs(q, k))

  def test_online_ring_matches_reference(self):
    q, k, v = self._random_qkv()
    self.assertLess(self._run_and_compare(q, k, v, use_fixed_m=False), 2e-2)

  def test_fixed_m_all_eligible_matches_reference(self):
    q, k, v = self._random_qkv()
    self.assertTrue(bool(jnp.all(self._gate(q, k))))
    self.assertLess(self._run_and_compare(q, k, v, use_fixed_m=True), 2e-2)

  def test_sink_head_falls_back_everywhere(self):
    total = self.shard_len * _RING_SIZE
    q, k, v = self._random_qkv(q_gain=(0, slice(0, total), 40.0))
    gate = self._gate(q, k)
    self.assertFalse(bool(jnp.any(gate[0])))  # head 0 online on every shard
    self.assertTrue(bool(jnp.all(gate[1:])))
    self.assertLess(self._run_and_compare(q, k, v, use_fixed_m=True), 2e-2)

  def test_fixed_m_accumulate_ragged_tail(self):
    # All-eligible (accumulate merge) with tiles that leave a ragged last KV
    # block (2048 %% 768 = 512) and a ragged inner chunk (512 %% 384 = 128),
    # covering the pinned fixed-m path's exact-slice tail handling.
    self.block_sizes = custom_splash._BlockSizes(block_q=1024, block_kv=768, block_kv_compute=384, block_kv_compute_in=384)
    q, k, v = self._random_qkv()
    self.assertTrue(bool(jnp.all(self._gate(q, k))))
    self.assertLess(self._run_and_compare(q, k, v, use_fixed_m=True), 2e-2)

  def test_mixed_fixed_online_across_shards(self):
    # Amplify head 0's keys on shard 1 only: head 0 is fixed on shard 0 but
    # online on shard 1 -- the mixed-partial merge the LSE space exists for.
    q, k, v = self._random_qkv(k_gain=(0, slice(self.shard_len, self.shard_len * _RING_SIZE), 40.0))
    gate = self._gate(q, k)
    self.assertTrue(bool(gate[0, 0]))
    self.assertFalse(bool(gate[0, 1]))
    self.assertLess(self._run_and_compare(q, k, v, use_fixed_m=True), 2e-2)

  def test_declared_unsquared_norms_match_squared(self):
    """The two declared norm representations must agree exactly.

    Regression test for the removed magnitude heuristic. Previously the kernel
    guessed whether norms were squared by testing `qn.max() * mk.max() < 1000`,
    which silently mis-classifies whenever the product straddles that constant
    -- reading unsquared norms as squared under-estimates the bound by up to
    the square root of its own magnitude, admits fixed-m where it must fall
    back, and overflows to inf. With the representation declared rather than
    inferred, both spellings of the same inputs must produce the same output.
    """
    q, k, v = self._random_qkv(q_gain=(0, slice(0, self.shard_len * _RING_SIZE), 40.0))
    q_in, k_in = self._scaled_inputs(q, k)
    out_sq = self._run_ring(q_in, k_in, v, use_fixed_m=True, norms_squared=True).astype(jnp.float32)
    out_unsq = self._run_ring(q_in, k_in, v, use_fixed_m=True, norms_squared=False).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_sq))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_unsq))))
    self.assertLess(float(jnp.max(jnp.abs(out_sq - out_unsq))), 2e-2)

  def test_v_ok_false_forces_finite_output(self):
    """An explicit unsafe verdict must force the online fallback.

    This is the shape of the FP16 / Q=K=0 / V=1 overflow: when the safety
    predicate says no, fixed-m must not run, whatever the Q/K norms imply.
    """
    q, k, v = self._random_qkv()
    q_in, k_in = self._scaled_inputs(q, k)
    out = self._run_ring(q_in, k_in, v, use_fixed_m=True, v_ok_override=False).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    self.assertLess(float(jnp.max(jnp.abs(out - self._reference(q_in, k_in, v)))), 2e-2)


class RingFixedMContractTest(unittest.TestCase):
  """Backend-independent checks on the fixed-m ring API contract.

  These assert on errors raised during tracing, so they need neither a TPU nor
  a real Pallas lowering and run everywhere CI does.
  """

  def _make(self, **kwargs):
    kwargs.setdefault("per_q_block", False)
    return ring_attention_kernel.make_custom_ring_attention(
        block_sizes=custom_splash._BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128),
        orig_q_seq_len=128,
        orig_kv_seq_len=128,
        use_base2_exp=True,
        ring_axis=_RING_AXIS,
        ring_size=1,
        **kwargs,
    )

  def _trace(self, ring, num_heads: int = 1):
    """Traces the ring callable under a 1-device mesh; never reaches the device."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), (_RING_AXIS,))
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)
    shape = (num_heads, 128, 128)

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)
    def _body(q, k, v):
      return ring(q, k, v)

    zeros = jnp.zeros(shape, jnp.bfloat16)
    return jax.eval_shape(_body, zeros, zeros, zeros)

  def test_fixed_m_requires_explicit_v_ok(self):
    """Omitting the safety predicate must fail loudly, not default to 'safe'.

    The kernel cannot re-derive the V-magnitude / dtype verdict from a single
    hop's Q/K, so treating omission as permission let fixed-m run on inputs it
    cannot represent (float16 with Q=K=0 and V=1 returned inf instead of 1.0).
    """
    norms = (jnp.ones((1,), jnp.float32), jnp.ones((1,), jnp.float32))
    ring = self._make(use_fixed_m=True, fixed_m_norms=norms)
    with self.assertRaises(ValueError) as ctx:
      self._trace(ring)
    self.assertIn("v_ok", str(ctx.exception))

  def test_fixed_m_requires_norms(self):
    ring = self._make(use_fixed_m=True, v_ok=True)
    with self.assertRaises(ValueError) as ctx:
      self._trace(ring)
    self.assertIn("fixed_m_norms", str(ctx.exception))

  def test_explicit_v_ok_false_is_accepted(self):
    """v_ok=False is a valid answer and must not trip the 'omitted' check."""
    norms = (jnp.ones((1,), jnp.float32), jnp.ones((1,), jnp.float32))
    ring = self._make(use_fixed_m=True, fixed_m_norms=norms, v_ok=False)
    self._trace(ring)  # must not raise

  def test_per_head_norms_with_per_q_block_are_rejected(self):
    """A (heads,) query norm under per_q_block=True must not broadcast.

    This is the defect behind the sink-head CI failure. Both eligibility gates
    compute `qn * mk[:, None]`, so a (heads,) array does not raise under
    per_q_block=True -- it broadcasts to (heads, heads), pairing head j's query
    norm with head h's key norm. A sink head inherits a small bound from an
    unrelated head, is wrongly marked fixed-eligible, and the kernel then
    evaluates exp2(large_logit - small_m), which overflows to inf.
    """
    norms = (jnp.ones((4,), jnp.float32), jnp.ones((4,), jnp.float32))
    ring = self._make(use_fixed_m=True, fixed_m_norms=norms, v_ok=True, per_q_block=True)
    with self.assertRaises(ValueError) as ctx:
      self._trace(ring, num_heads=4)
    self.assertIn("per_q_block", str(ctx.exception))

  def test_per_q_block_norms_with_correct_shape_are_accepted(self):
    """The properly shaped (heads, num_q_blocks) array must pass."""
    # orig_q_seq_len=128 and block_q=128 give exactly one Q block.
    norms = (jnp.ones((4, 1), jnp.float32), jnp.ones((4,), jnp.float32))
    ring = self._make(use_fixed_m=True, fixed_m_norms=norms, v_ok=True, per_q_block=True)
    self._trace(ring, num_heads=4)  # must not raise

  def test_fp16_is_rejected_by_dtype_safety(self):
    """The dtype half of the safety predicate must reject narrow exponents.

    float16 has a 5-bit exponent (maxexp 16); fixed-m parks weights at
    2**recenter with recenter ~= 88, which is far beyond float16's ceiling.
    """
    recenter, _ = custom_splash.get_fixed_m_constants(4096, is_ring=False)
    self.assertFalse(custom_splash.fixed_m_dtype_is_safe(jnp.float16, recenter))
    self.assertTrue(custom_splash.fixed_m_dtype_is_safe(jnp.bfloat16, recenter))
    self.assertTrue(custom_splash.fixed_m_dtype_is_safe(jnp.float32, recenter))


class RingRawKeyBoundUnsoundTest(unittest.TestCase):
  """Why ring fixed-m requires a CENTERED key bound.

  The ring kernel derives a global key mean (`lax.pmean` over the ring axis)
  and exponentiates centered logits `q . (k_j - k_mean)`. A Cauchy-Schwarz
  bound built from the raw, uncentered `k` therefore bounds the wrong
  quantity: it can clear the eligibility threshold while the centered logit
  it is supposed to cap overflows fp32.

  These assertions are pure arithmetic -- no TPU, no kernel -- so they pin the
  failure mode itself rather than one kernel's symptom of it.
  """

  total_kv = 4096
  head_dim = 128

  def _adversarial_keys(self):
    """One key at +100, the rest at -100, on a single active dimension.

    Every key has the same norm (100), so the raw bound is small, but the
    population mean sits at ~-99.95 and the lone positive key is ~200 away
    from it.
    """
    k = jnp.full((self.total_kv,), -100.0, dtype=jnp.float32)
    k = k.at[0].set(100.0)
    keys = jnp.zeros((self.total_kv, self.head_dim), dtype=jnp.float32).at[:, 0].set(k)
    query = jnp.zeros((self.head_dim,), dtype=jnp.float32).at[0].set(1.0)
    return query, keys

  def test_raw_bound_admits_a_tile_the_centered_bound_rejects(self):
    query, keys = self._adversarial_keys()
    _, safe_bound = custom_splash.get_fixed_m_constants(self.total_kv, is_ring=True)

    q_norm = float(jnp.linalg.norm(query))
    raw_bound = q_norm * float(jnp.linalg.norm(keys, axis=-1).max())

    k_mean = keys.mean(axis=0)
    centered_bound = q_norm * float(jnp.linalg.norm(keys - k_mean, axis=-1).max())

    # The raw bound clears the gate ...
    self.assertLessEqual(raw_bound, safe_bound)
    # ... but the quantity the kernel actually exponentiates does not.
    self.assertGreater(centered_bound, safe_bound)
    # The gap is the whole bug: ~100 vs ~200 against a 116 ceiling.
    self.assertGreater(centered_bound, 1.9 * raw_bound)

  def test_centered_logit_overflows_fp32_under_the_raw_bound(self):
    """With `m` taken from the raw bound, the shifted exponent leaves fp32."""
    query, keys = self._adversarial_keys()
    recenter, _ = custom_splash.get_fixed_m_constants(self.total_kv, is_ring=True)

    k_mean = keys.mean(axis=0)
    max_centered_logit = float(((keys - k_mean) @ query).max())
    fixed_m = float(jnp.linalg.norm(query)) * float(jnp.linalg.norm(keys, axis=-1).max())

    # fixed-m parks the max weight at 2**recenter, so the realised exponent is
    # (z - m) + recenter. fp32 tops out at 2**128.
    shifted_exponent = max_centered_logit - fixed_m + recenter
    self.assertGreater(shifted_exponent, 128.0)

  def test_centering_restores_a_sound_bound(self):
    """Bounding the centered keys is what makes the gate honest again."""
    query, keys = self._adversarial_keys()
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.total_kv, is_ring=True)

    k_mean = keys.mean(axis=0)
    centered = keys - k_mean
    centered_bound = float(jnp.linalg.norm(query)) * float(jnp.linalg.norm(centered, axis=-1).max())

    # Correctly rejected, so this tile takes the online-softmax path.
    self.assertGreater(centered_bound, safe_bound)
    # And had it been admitted, the bound would genuinely cap the logit.
    max_centered_logit = float((centered @ query).max())
    self.assertLessEqual(max_centered_logit, centered_bound + 1e-3)
    self.assertLessEqual(max_centered_logit - centered_bound + recenter, 128.0)


if __name__ == "__main__":
  unittest.main()
