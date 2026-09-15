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

"""Unit tests for the fixed-m path of the custom RING attention with Global Virtual K-Centering.

The ring path computes a global Key mean across the ring axis (k_mean = pmean(mean(k), ring_axis)),
which mathematically guarantees max_j (q^T (k_j - k_mean)) >= 0 across the entire distributed sequence.
Key norms are gathered across ranks once before the scan (mk_global = mk_all.max(axis=0)) to evaluate
identical fixed m bounds across all ring hops, enabling direct FP32 accumulation and full centered
safe bounds (W(N) = 127 - ceil(log2 N) + 125).
"""

import functools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import ring_attention_kernel
from maxdiffusion.models import attention_flax
from flax.linen import partitioning as nn_partitioning

_LOG2E = math.log2(math.e)
_RING_AXIS = "ring"
_RING_SIZE = 2


class RingFixedMTest(unittest.TestCase):
  """Numerical tests for the fixed-m custom ring attention across topologies."""

  num_heads = 4
  shard_len = 2048  # per-device sequence; total = shard_len * ring_size
  head_dim = 128

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.block_sizes = custom_splash._BlockSizes(block_q=1024, block_kv=1024, block_kv_compute=512, block_kv_compute_in=256)

  def _mesh_for_size(self, ring_size: int):
    if len(jax.devices()) < ring_size:
      self.skipTest(f"Requires {ring_size} devices, but only {len(jax.devices())} available.")
    devices = np.asarray(jax.devices()[:ring_size])
    return jax.sharding.Mesh(devices, (_RING_AXIS,))

  def _random_qkv(self, ring_size: int = 2, q_gain=None, k_gain=None):
    """bf16 (q, k, v), [heads, total_seq, dim]; optional (head, row-slice) gains."""
    total = self.shard_len * ring_size
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
    """The EXACT bf16 tensors the kernel sees (attention_flax's contract)."""
    q_in = (q * _LOG2E).astype(q.dtype)
    k_in = (k.astype(jnp.float32) * self.scale).astype(k.dtype)
    return q_in, k_in

  def _reference(self, q_in, k_in, v):
    """Dense f32 log2-domain softmax on the kernel's own bf16 inputs."""
    qf, kf, vf = (x.astype(jnp.float32) for x in (q_in, k_in, v))
    logits = jnp.einsum("hqd,hkd->hqk", qf, kf)  # LOG2E & scale pre-folded
    return jax.nn.softmax(logits * math.log(2.0), axis=-1) @ vf

  def _run_ring(self, q_in, k_in, v, ring_size: int = 2, use_fixed_m: bool = True, v_ok=None):
    """Runs the custom ring under shard_map with per-rank fixed_m_norms."""
    mesh = self._mesh_for_size(ring_size)
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body(ql, kl, vl):
      fixed_m_norms = None
      k_mean = None
      if use_fixed_m:
        qf = ql.astype(jnp.float32)
        kf = kl.astype(jnp.float32)
        k_mean_local = jnp.mean(kf, axis=1)  # (heads, dim)
        k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
        bq = self.block_sizes.block_q
        num_q_blocks = qf.shape[1] // bq
        qf_blocks = qf.reshape(qf.shape[0], num_q_blocks, bq, qf.shape[-1])
        qn_blocks_sq = (qf_blocks * qf_blocks).sum(-1).max(axis=-1)  # (heads, num_q_blocks)
        kf_centered = kf - k_mean[:, None, :]
        mk_h_sq = (kf_centered * kf_centered).sum(-1).max(axis=1)  # (heads,) local shard
        fixed_m_norms = (qn_blocks_sq, mk_h_sq)
      v_ok_effective = v_ok
      if v_ok is None and use_fixed_m:
        v_max_sq = (vl.astype(jnp.float32) ** 2).max()
        v_ok_local = v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2)
        v_ok_effective = jax.lax.pmin(v_ok_local, axis_name=_RING_AXIS)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=ring_size,
          use_fixed_m=use_fixed_m,
          fixed_m_norms=fixed_m_norms,
          k_mean=k_mean,
          v_ok=v_ok_effective,
      )
      return ring(ql, kl, vl)

    return _body(q_in, k_in, v)

  def _global_v_ok(self, v, ring_size: int = 2):
    """The V-safety verdict attention_flax computes, reduced over the whole ring."""
    v_max_sq = (v.astype(jnp.float32) ** 2).max()
    return bool(v_max_sq <= custom_splash.DEFAULT_MAX_V_BOUND**2)

  def _gate_per_shard(self, q_in, k_in, ring_size: int = 2):
    """(heads, ring_size) eligibility against the dynamic centered bound with Global Virtual K-Centering."""
    qf = q_in.astype(jnp.float32)
    kf = k_in.astype(jnp.float32)
    k_mean_global = jnp.mean(kf, axis=1, keepdims=True)
    kf_centered = kf - k_mean_global
    qn = jnp.sqrt((qf * qf).sum(-1))  # (heads, total)
    kn = jnp.sqrt((kf_centered * kf_centered).sum(-1))
    _, safe_bound = custom_splash.get_fixed_m_constants(self.shard_len * ring_size, is_ring=False)
    gates = []
    for r in range(ring_size):
      rows = slice(r * self.shard_len, (r + 1) * self.shard_len)
      bound = qn.max(axis=1) * kn[:, rows].max(axis=1)
      gates.append(bound <= safe_bound)
    return jnp.stack(gates, axis=1)

  def _run_and_compare(self, q, k, v, ring_size: int = 2, use_fixed_m: bool = True):
    q_in, k_in = self._scaled_inputs(q, k)
    out = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=use_fixed_m).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    return float(jnp.max(jnp.abs(out - self._reference(q_in, k_in, v))))

  def _gate(self, q, k, ring_size: int = 2):
    return self._gate_per_shard(*self._scaled_inputs(q, k), ring_size=ring_size)

  def test_online_ring_matches_reference(self):
    q, k, v = self._random_qkv(ring_size=2)
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=False), 2e-2)

  def test_fixed_m_all_eligible_matches_reference(self):
    q, k, v = self._random_qkv(ring_size=2)
    self.assertTrue(bool(jnp.all(self._gate(q, k, ring_size=2))))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=True), 2e-2)

  def test_sink_head_falls_back_everywhere(self):
    total = self.shard_len * 2
    q, k, v = self._random_qkv(ring_size=2, q_gain=(0, slice(0, total), 40.0))
    gate = self._gate(q, k, ring_size=2)
    self.assertFalse(bool(jnp.any(gate[0])))  # head 0 online on every shard
    self.assertTrue(bool(jnp.all(gate[1:])))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=True), 2e-2)

  def test_fixed_m_accumulate_ragged_tail(self):
    self.block_sizes = custom_splash._BlockSizes(block_q=1024, block_kv=768, block_kv_compute=384, block_kv_compute_in=384)
    q, k, v = self._random_qkv(ring_size=2)
    self.assertTrue(bool(jnp.all(self._gate(q, k, ring_size=2))))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=True), 2e-2)

  def test_mixed_fixed_online_across_shards(self):
    q, k, v = self._random_qkv(ring_size=2, k_gain=(0, slice(self.shard_len, self.shard_len * 2), 40.0))
    gate = self._gate(q, k, ring_size=2)
    self.assertTrue(bool(gate[0, 0]))
    self.assertFalse(bool(gate[0, 1]))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=True), 2e-2)

  def test_per_q_block_sink_tile_ring(self):
    q, k, v = self._random_qkv(ring_size=2, q_gain=(0, slice(0, self.block_sizes.block_q), 40.0))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=2, use_fixed_m=True), 2e-2)

  def test_fixed_m_multi_hop_ring_size_4(self):
    """Verifies wrap-around collective correctness and LSE accumulation for R=4."""
    q, k, v = self._random_qkv(ring_size=4)
    self.assertTrue(bool(jnp.all(self._gate(q, k, ring_size=4))))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=4, use_fixed_m=True), 2e-2)

  def test_mixed_fixed_online_ring_size_4(self):
    """Verifies 4-hop ring with mixed fixed/online shards on separate ranks."""
    # Shard 2 is amplified: head 0 will be fixed on shards 0, 1, 3 and online on shard 2
    q, k, v = self._random_qkv(ring_size=4, k_gain=(0, slice(self.shard_len * 2, self.shard_len * 3), 40.0))
    gate = self._gate(q, k, ring_size=4)
    self.assertTrue(bool(gate[0, 0]))
    self.assertTrue(bool(gate[0, 1]))
    self.assertFalse(bool(gate[0, 2]))
    self.assertTrue(bool(gate[0, 3]))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=4, use_fixed_m=True), 2e-2)

  def test_fixed_m_multi_hop_ring_size_8(self):
    """Verifies 8-device full torus ring rotation and fixed-m numerical equivalence."""
    q, k, v = self._random_qkv(ring_size=8)
    self.assertTrue(bool(jnp.all(self._gate(q, k, ring_size=8))))
    self.assertLess(self._run_and_compare(q, k, v, ring_size=8, use_fixed_m=True), 2e-2)

  def test_batched_cfg_isolation_ring(self):
    """Verifies that CFG batch items (batch=2) are strictly isolated with zero cross-contamination."""
    # Batch 0: normal bounded activations (all fixed-m)
    # Batch 1: massive sink outlier token (forces online fallback)
    ring_size = 2
    q0, k0, v0 = self._random_qkv(ring_size=ring_size)
    q1, k1, v1 = self._random_qkv(ring_size=ring_size, q_gain=(0, slice(0, self.block_sizes.block_q), 50.0))
    q_batch = jnp.stack([q0, q1], axis=0)  # (2, heads, total_seq, dim)
    k_batch = jnp.stack([k0, k1], axis=0)
    v_batch = jnp.stack([v0, v1], axis=0)
    q_in, k_in = self._scaled_inputs(q_batch, k_batch)

    # Run batch through shard_map with vmap over batch
    mesh = self._mesh_for_size(ring_size)
    spec = jax.sharding.PartitionSpec(None, None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body_batched(qb, kb, vb):
      batch_size, num_h, q_seq, _ = qb.shape
      bq = self.block_sizes.block_q
      num_q_blocks = q_seq // bq
      qfb = qb.astype(jnp.float32)
      kfb = kb.astype(jnp.float32)
      k_mean_local = jnp.mean(kfb, axis=2)  # (batch, heads, dim)
      k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
      norm_sq = (qfb * qfb).sum(axis=-1)
      qn_dev = norm_sq.reshape(batch_size, num_h, num_q_blocks, bq).max(axis=-1)  # (batch, heads, num_q_blocks)
      kfb_centered = kfb - k_mean[:, :, None, :]
      mk_dev = (kfb_centered * kfb_centered).sum(axis=-1).max(axis=-1)  # (batch, heads)
      # The V-magnitude verdict is global, so it is reduced across the ring
      # before use; the kernel cannot re-derive it from Q/K norms.
      v_max_sq = (vb.astype(jnp.float32) ** 2).max()
      v_ok = jax.lax.pmin(v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2), axis_name=_RING_AXIS)
      ring_kernel = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=ring_size,
          use_fixed_m=True,
          v_ok=v_ok,
      )
      return jax.vmap(ring_kernel, in_axes=(0, 0, 0, (0, 0), 0))(qb, kb, vb, (qn_dev, mk_dev), k_mean)

    out_batched = _body_batched(q_in, k_in, v_batch).astype(jnp.float32)

    # Verify batch item 0 (clean prompt) matches exact unbatched fixed-m reference
    ref0 = self._reference(q_in[0], k_in[0], v0)
    diff0 = float(jnp.max(jnp.abs(out_batched[0] - ref0)))
    self.assertLess(diff0, 2e-2)

    # Verify batch item 1 (sink outlier prompt) matches exact unbatched reference
    ref1 = self._reference(q_in[1], k_in[1], v1)
    diff1 = float(jnp.max(jnp.abs(out_batched[1] - ref1)))
    self.assertLess(diff1, 2e-2)

  def test_ring_phase_transition_boundary_continuity(self):
    """Verifies seamless continuity between fixed-m and online mode across the dynamic centered Ring safe bound threshold."""
    ring_size = 2
    q_base, k_base, v = self._random_qkv(ring_size=ring_size)
    q_normed = q_base / jnp.sqrt((q_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))
    k_normed = k_base / jnp.sqrt((k_base.astype(jnp.float32) ** 2).sum(-1, keepdims=True))

    _, safe_bound = custom_splash.get_fixed_m_constants(self.shard_len * ring_size, is_ring=False)
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

      q_in, k_in = self._scaled_inputs(q, k)
      out_fixed = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=True).astype(jnp.float32)
      out_online = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=False).astype(jnp.float32)

      self.assertTrue(bool(jnp.all(jnp.isfinite(out_fixed))))
      diff = float(jnp.max(jnp.abs(out_fixed - out_online)))
      self.assertLess(diff, 2e-2, f"Ring discontinuity at bound={target_bound}, diff={diff}")

  def test_per_head_single_rank_outlier_sync_ring(self):
    """Verifies that in per_q_block=False mode, an outlier on a single ring rank synchronizes via pmin across all ranks."""
    ring_size = 2
    # Create Q with outlier only on Rank 1 (rows self.shard_len to 2 * self.shard_len)
    q, k, v = self._random_qkv(ring_size=ring_size, q_gain=(0, slice(self.shard_len, self.shard_len * 2), 40.0))
    q_in, k_in = self._scaled_inputs(q, k)
    mesh = self._mesh_for_size(ring_size)
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body_per_head(ql, kl, vl):
      qf = ql.astype(jnp.float32)
      kf = kl.astype(jnp.float32)
      k_mean_local = jnp.mean(kf, axis=1)
      k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
      qn_max_sq = (qf * qf).sum(-1).max(axis=1)  # (heads,) per-head 1D squared norm
      kf_centered = kf - k_mean[:, None, :]
      mk_h_sq = (kf_centered * kf_centered).sum(-1).max(axis=1)
      v_max_sq = (vl.astype(jnp.float32) ** 2).max()
      v_ok = jax.lax.pmin(v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2), axis_name=_RING_AXIS)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=ring_size,
          use_fixed_m=True,
          per_q_block=False,
          fixed_m_norms=(qn_max_sq, mk_h_sq),
          k_mean=k_mean,
          v_ok=v_ok,
      )
      return ring(ql, kl, vl)

    out = _body_per_head(q_in, k_in, v).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    ref = self._reference(q_in, k_in, v)
    diff = float(jnp.max(jnp.abs(out - ref)))
    self.assertLess(diff, 2e-2)

  def test_adversarial_unsmoothed_negative_keys_ring(self):
    """Verifies that un-smoothed keys with large negative bias merge safely without underflow NaNs."""
    ring_size = 2
    q, k, v = self._random_qkv(ring_size=ring_size)
    k_negative = k - 25.0  # Force strong negative bias across both ring shards
    q_in, k_in = self._scaled_inputs(q, k_negative)

    out_fixed = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=True).astype(jnp.float32)
    out_online = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=False).astype(jnp.float32)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_fixed))))
    diff = float(jnp.max(jnp.abs(out_fixed - out_online)))
    self.assertLess(diff, 2e-2)

  def test_adversarial_hybrid_ring_negative_shard(self):
    """Verifies hybrid LSE fallback when one shard is heavily negative and another positive (global mean 0)."""
    ring_size = 2
    q, k, v = self._random_qkv(ring_size=ring_size)
    # Shard 0: heavily negative, Shard 1: heavily positive
    k = k.at[:, : self.shard_len, :].add(-20.0)
    k = k.at[:, self.shard_len :, :].add(20.0)
    # Force rank 0 to exceed the global bound so execution enters the hybrid fallback branch
    q = q.at[0, : self.block_sizes.block_q, :].multiply(40.0)
    q_in, k_in = self._scaled_inputs(q, k)

    out_hybrid = self._run_ring(q_in, k_in, v, ring_size=ring_size, use_fixed_m=True).astype(jnp.float32)
    ref = self._reference(q_in, k_in, v)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_hybrid))))
    diff = float(jnp.max(jnp.abs(out_hybrid - ref)))
    self.assertLess(diff, 2e-2, f"Adversarial hybrid ring output diverged from reference: diff={diff}")

  def test_adversarial_centered_keys_softmax_mass_loss_two_rank_ring(self):
    """Verifies that centered keys with large query norm safely fall back to online ring softmax.

    If evaluated under an unsafe fixed-m bound, the negative logit term flushes to zero,
    yielding 1.0 instead of ~15/17 (~0.8824). The cross-ring safety check flags ineligibility
    and triggers online ring softmax accumulation across ranks.
    """
    ring_size = 2
    total_len = self.shard_len * ring_size
    num_heads = 1

    # Query: [231, 2, 0, ...]
    q = jnp.zeros((num_heads, total_len, self.head_dim), dtype=jnp.bfloat16)
    q = q.at[:, :, 0].set(231.0).at[:, :, 1].set(2.0)

    # Shard 0: [0, 1, 0, ...], Shard 1: [0, -1, 0, ...] (centered, mean 0)
    k = jnp.zeros((num_heads, total_len, self.head_dim), dtype=jnp.bfloat16)
    k = k.at[:, : self.shard_len, 1].set(1.0).at[:, self.shard_len :, 1].set(-1.0)

    # Values: Shard 0: +1.0, Shard 1: -1.0
    v = jnp.zeros((num_heads, total_len, self.head_dim), dtype=jnp.bfloat16)
    v = v.at[:, : self.shard_len, :].set(1.0).at[:, self.shard_len :, :].set(-1.0)

    out_fixed = self._run_ring(q, k, v, ring_size=ring_size, use_fixed_m=True).astype(jnp.float32)
    expected = 15.0 / 17.0
    self.assertLess(
        float(jnp.max(jnp.abs(out_fixed - expected))),
        5e-3,
        f"Two-rank ring fallback must preserve probability mass near 15/17 (~0.8824), got {float(out_fixed[0, 0, 0]):.6f}",
    )

  # --- P1 regression: the ring LSE fallback must honour the global V check ---
  #
  # The fallback re-derives per-hop eligibility from Q/K norms alone. Those are
  # the only quantities recoverable from a single hop, so a failed V check used
  # to be silently discarded and individual hops re-enabled fixed-m -- parking
  # weights at 2**C with an out-of-contract |V| and overflowing to inf. These
  # tests execute the fallback and assert on the *output*, not on metadata.

  def test_oversized_v_ring_fallback_is_finite(self):
    """|V| beyond the contract must fall back to online arithmetic, not overflow."""
    ring_size = 2
    q, k, v = self._random_qkv(ring_size=ring_size)
    v_big = (v.astype(jnp.float32) * 1024.0).astype(v.dtype)
    self.assertFalse(self._global_v_ok(v_big), "test setup: V should violate the bound")
    q_in, k_in = self._scaled_inputs(q, k)

    out = self._run_ring(q_in, k_in, v_big, ring_size=ring_size, use_fixed_m=True, v_ok=False).astype(jnp.float32)
    ref = self._reference(q_in, k_in, v_big)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out))), "oversized |V| overflowed the ring fixed-m fallback")
    # Tolerance scales with |V|: the 2e-2 used elsewhere is for unit-scale V.
    self.assertLess(float(jnp.max(jnp.abs(out - ref))), 2e-2 * 1024.0)

  def test_oversized_v_on_single_rank_disqualifies_whole_ring(self):
    """Only rank 1 has oversized V; the globally reduced verdict must protect every hop."""
    ring_size = 2
    q, k, v = self._random_qkv(ring_size=ring_size)
    # Rank 1 owns the second shard_len slice of the sequence.
    v_mixed = v.at[:, self.shard_len :, :].set((v[:, self.shard_len :, :].astype(jnp.float32) * 1024.0).astype(v.dtype))
    self.assertFalse(self._global_v_ok(v_mixed), "test setup: global V should violate the bound")
    q_in, k_in = self._scaled_inputs(q, k)

    # v_ok is a *global* reduction, so a single offending rank disqualifies all.
    # We do NOT pass v_ok=False: the automatic cross-rank pmin reduction must compute it.
    out = self._run_ring(q_in, k_in, v_mixed, ring_size=ring_size, use_fixed_m=True).astype(jnp.float32)
    ref = self._reference(q_in, k_in, v_mixed)

    self.assertTrue(bool(jnp.all(jnp.isfinite(out))), "single-rank oversized |V| overflowed the ring fallback")
    self.assertLess(float(jnp.max(jnp.abs(out - ref))), 2e-2 * 1024.0)

  def test_v_gate_is_what_prevents_the_overflow(self):
    """Witness: without the gate the same inputs are unsafe.

    Guards against the gate being quietly dropped again. If a future change makes
    the fallback safe by construction this test should be deleted, not muted --
    but it must never be allowed to pass by accident.
    """
    ring_size = 2
    q, k, v = self._random_qkv(ring_size=ring_size)
    v_big = (v.astype(jnp.float32) * 1024.0).astype(v.dtype)
    q_in, k_in = self._scaled_inputs(q, k)

    gated = self._run_ring(q_in, k_in, v_big, ring_size=ring_size, use_fixed_m=True, v_ok=False).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(gated))))

    ungated = self._run_ring(q_in, k_in, v_big, ring_size=ring_size, use_fixed_m=True, v_ok=True).astype(jnp.float32)
    if bool(jnp.all(jnp.isfinite(ungated))):
      self.skipTest("ungated path happens to stay finite for these inputs; gate still required in general")

  def test_gqa_ring_fixed_m_shard_map(self):
    """Verifies that GQA (4 Q heads, 2 KV heads) works seamlessly across ring ranks."""
    ring_size = 2
    num_q_heads = 4
    num_kv_heads = 2
    total_seq = self.shard_len * ring_size
    q = jax.random.normal(jax.random.PRNGKey(101), (num_q_heads, total_seq, self.head_dim), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(102), (num_kv_heads, total_seq, self.head_dim), jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(103), (num_kv_heads, total_seq, self.head_dim), jnp.bfloat16)
    q_in, k_in = self._scaled_inputs(q, k)

    mesh = self._mesh_for_size(ring_size)
    spec_q = jax.sharding.PartitionSpec(None, _RING_AXIS, None)
    spec_kv = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec_q, spec_kv, spec_kv),
        out_specs=spec_q,
        check_vma=False,
    )
    def _body_gqa(ql, kl, vl):
      qf = ql.astype(jnp.float32)
      kf = kl.astype(jnp.float32)
      k_mean_local = jnp.mean(kf, axis=1)  # (kv_heads, dim)
      k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
      bq = self.block_sizes.block_q
      num_q_blocks = qf.shape[1] // bq
      qf_blocks = qf.reshape(num_q_heads, num_q_blocks, bq, self.head_dim)
      qn_blocks_sq = (qf_blocks * qf_blocks).sum(-1).max(axis=-1)  # (q_heads, num_q_blocks)
      kf_centered = kf - k_mean[:, None, :]
      mk_h_sq = (kf_centered * kf_centered).sum(-1).max(axis=1)  # (kv_heads,)
      v_max_sq = (vl.astype(jnp.float32) ** 2).max()
      v_ok = jax.lax.pmin(v_max_sq <= (custom_splash.DEFAULT_MAX_V_BOUND**2), axis_name=_RING_AXIS)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=ring_size,
          use_fixed_m=True,
          fixed_m_norms=(qn_blocks_sq, mk_h_sq),
          k_mean=k_mean,
          v_ok=v_ok,
      )
      return ring(ql, kl, vl)

    out = _body_gqa(q_in, k_in, v).astype(jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    # GQA Reference
    q_rep = q_in
    k_rep = jnp.repeat(k_in, num_q_heads // num_kv_heads, axis=0)
    v_rep = jnp.repeat(v, num_q_heads // num_kv_heads, axis=0)
    ref = self._reference(q_rep, k_rep, v_rep)
    diff = float(jnp.max(jnp.abs(out - ref)))
    self.assertLess(diff, 2e-2, f"GQA ring output diverged from reference: diff={diff}")

  def test_fixed_m_mismatched_ring_size_raises(self):
    """Verifies that ring_size != axis_size raises NotImplementedError when use_fixed_m=True."""
    q, k, v = self._random_qkv(ring_size=2)
    q_in, k_in = self._scaled_inputs(q, k)
    mesh = self._mesh_for_size(2)
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body(ql, kl, vl):
      qf = ql.astype(jnp.float32)
      kf = kl.astype(jnp.float32)
      k_mean_local = jnp.mean(kf, axis=1)
      k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
      qn_max_sq = (qf * qf).sum(-1).max(axis=1)
      kf_centered = kf - k_mean[:, None, :]
      mk_h_sq = (kf_centered * kf_centered).sum(-1).max(axis=1)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=1,  # Mismatched: ring_size=1 != axis_size=2
          use_fixed_m=True,
          fixed_m_norms=(qn_max_sq, mk_h_sq),
          k_mean=k_mean,
      )
      return ring(ql, kl, vl)

    with self.assertRaises(NotImplementedError):
      _body(q_in, k_in, v)

  def test_fixed_m_non_canonical_perm_raises(self):
    """Verifies that non-canonical perm raises NotImplementedError when use_fixed_m=True."""
    q, k, v = self._random_qkv(ring_size=2)
    q_in, k_in = self._scaled_inputs(q, k)
    mesh = self._mesh_for_size(2)
    spec = jax.sharding.PartitionSpec(None, _RING_AXIS, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    def _body(ql, kl, vl):
      qf = ql.astype(jnp.float32)
      kf = kl.astype(jnp.float32)
      k_mean_local = jnp.mean(kf, axis=1)
      k_mean = jax.lax.pmean(k_mean_local, axis_name=_RING_AXIS)
      qn_max_sq = (qf * qf).sum(-1).max(axis=1)
      kf_centered = kf - k_mean[:, None, :]
      mk_h_sq = (kf_centered * kf_centered).sum(-1).max(axis=1)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=2,
          perm=[(0, 0), (1, 1)],  # Non-canonical identity permutation
          use_fixed_m=True,
          fixed_m_norms=(qn_max_sq, mk_h_sq),
          k_mean=k_mean,
      )
      return ring(ql, kl, vl)

    with self.assertRaises(NotImplementedError):
      _body(q_in, k_in, v)

  def test_gqa_with_chunked_ulysses_raises(self):
    """Verifies that GQA (Hq != Hkv) with ulysses_attention_chunks > 1 raises NotImplementedError."""
    q = jnp.zeros((1, 8, 128, 64), dtype=jnp.float32)
    k = jnp.zeros((1, 2, 128, 64), dtype=jnp.float32)
    v = jnp.zeros((1, 2, 128, 64), dtype=jnp.float32)

    with self.assertRaises(NotImplementedError):
      attention_flax._run_chunked_ulysses_attention(
          q,
          k,
          v,
          num_heads=8,
          ulysses_shards=2,
          ulysses_attention_chunks=2,
          attention_fn=lambda q, k, v: q,
      )

  def test_2d_gqa_ulysses_ring_attention(self):
    """Verifies that 2D Ulysses+Ring attention correctly executes GQA (Hq=8, Hkv=2) with chunks=1."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices for 2D Ulysses+Ring test.")

    devices = np.array(jax.devices()[:4]).reshape(1, 1, 4, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "context", "tensor"))
    axis_rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.LENGTH, "context"),
        (attention_flax.HEAD, None),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
    )

    batch = 1
    length = 2048
    q_heads = 8
    kv_heads = 2
    head_dim = 128

    q = jax.random.normal(jax.random.PRNGKey(10), (batch, length, q_heads * head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(11), (batch, length, kv_heads * head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(12), (batch, length, kv_heads * head_dim), dtype=jnp.bfloat16)

    flash_block_sizes = {
        "block_q": 1024,
        "block_kv": 1024,
        "block_kv_compute": 512,
        "block_kv_compute_in": 256,
        "heads_per_tile": 1,
        "vmem_limit_bytes": 67108864,
    }

    with mesh, nn_partitioning.axis_rules(axis_rules):
      out = attention_flax._ulysses_ring_custom_attention(
          q,
          k * (1.0 / math.sqrt(head_dim)),
          v,
          heads=q_heads,
          mesh=mesh,
          axis_names_q=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_Q_LENGTH,
              attention_flax.D_KV,
          ),
          axis_names_kv=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_KV_LENGTH,
              attention_flax.D_KV,
          ),
          flash_block_sizes=flash_block_sizes,
          dtype=jnp.bfloat16,
          ulysses_shards=2,
          use_base2_exp=True,
          use_fixed_m=True,
          per_q_block=True,
          ulysses_attention_chunks=1,
          kv_heads=kv_heads,
      )
    self.assertEqual(out.shape, (batch, length, q_heads * head_dim))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    # Reference calculation: repeat KV heads to match Q heads (Hq=8, Hkv=2 => repeat factor 4)
    q_unflat = q.reshape((batch, length, q_heads, head_dim)).swapaxes(1, 2)  # [B, Hq, L, D]
    k_unflat = k.reshape((batch, length, kv_heads, head_dim)).swapaxes(1, 2)  # [B, Hkv, L, D]
    v_unflat = v.reshape((batch, length, kv_heads, head_dim)).swapaxes(1, 2)  # [B, Hkv, L, D]

    k_repeated = jnp.repeat(k_unflat, q_heads // kv_heads, axis=1)  # [B, Hq, L, D]
    v_repeated = jnp.repeat(v_unflat, q_heads // kv_heads, axis=1)  # [B, Hq, L, D]

    # Reference scaled dot product attention in FP32
    scores = jnp.einsum(
        "bhqd,bhkd->bhqk",
        q_unflat.astype(jnp.float32) * (1.0 / math.sqrt(head_dim)),
        k_repeated.astype(jnp.float32),
    )
    attn_weights = jax.nn.softmax(scores, axis=-1)
    ref_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_repeated.astype(jnp.float32))
    ref_out = ref_out.swapaxes(1, 2).reshape((batch, length, q_heads * head_dim))

    np.testing.assert_allclose(
        np.array(out, dtype=np.float32),
        np.array(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_oversized_v_single_rank_production_dispatch(self):
    """Verifies that attention_flax._ulysses_ring_custom_attention reduces v_ok across ranks.

    Only rank 1 has oversized V. If v_ok were local, rank 0 would retain v_ok=True and
    overflow to inf on the second hop. The cross-ring reduction ensures both ranks fall back.
    """
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices for 2D Ulysses+Ring test.")

    devices = np.array(jax.devices()[:4]).reshape(1, 1, 4, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "context", "tensor"))
    axis_rules = (
        (attention_flax.BATCH, "data"),
        (attention_flax.LENGTH, "context"),
        (attention_flax.HEAD, None),
        (attention_flax.SELF_ATTN_HEAD, None),
        (attention_flax.SELF_ATTN_Q_LENGTH, "context"),
        (attention_flax.SELF_ATTN_KV_LENGTH, "context"),
        (attention_flax.D_KV, None),
    )

    batch = 1
    length = 8192
    q_heads = 4
    kv_heads = 4
    head_dim = 128

    # Q=K=0 and constant V=1 on rank 0, V=1024 on rank 1.
    # With 4 context devices and ulysses=2, ring_size=2. Each ring rank gets 4096 tokens (4 blocks of 1024).
    # If v_ok is not reduced across ranks, rank 0 retains v_ok=True and attempts fixed-m accumulation
    # on rank 1's 4 blocks of V=1024, overflowing FP32 (4096 * 1024 * 2^107 = 2^129 > 2^128).
    q = jnp.zeros((batch, length, q_heads * head_dim), dtype=jnp.bfloat16)
    k = jnp.zeros((batch, length, kv_heads * head_dim), dtype=jnp.bfloat16)
    half = length // 2
    v_mixed = jnp.ones((batch, length, kv_heads * head_dim), dtype=jnp.bfloat16)
    v_mixed = v_mixed.at[:, half:, :].set(1024.0)

    flash_block_sizes = {
        "block_q": 1024,
        "block_kv": 1024,
        "block_kv_compute": 512,
        "block_kv_compute_in": 256,
        "heads_per_tile": 1,
        "vmem_limit_bytes": 67108864,
    }

    with mesh, nn_partitioning.axis_rules(axis_rules):
      out = attention_flax._ulysses_ring_custom_attention(
          q,
          k,
          v_mixed,
          heads=q_heads,
          mesh=mesh,
          axis_names_q=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_Q_LENGTH,
              attention_flax.D_KV,
          ),
          axis_names_kv=(
              attention_flax.BATCH,
              attention_flax.SELF_ATTN_HEAD,
              attention_flax.SELF_ATTN_KV_LENGTH,
              attention_flax.D_KV,
          ),
          flash_block_sizes=flash_block_sizes,
          dtype=jnp.bfloat16,
          ulysses_shards=2,
          use_base2_exp=True,
          use_fixed_m=True,
          per_q_block=True,
          ulysses_attention_chunks=1,
          kv_heads=kv_heads,
      )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))), "Production dispatch overflowed on mixed-rank oversized V")

    # With Q=K=0, attention weights are uniform 1/N. Expected output is exactly (1.0 + 1024.0)/2 = 512.5 everywhere.
    expected_val = (1.0 + 1024.0) / 2.0
    out_f32 = np.array(out, dtype=np.float32)
    np.testing.assert_allclose(
        out_f32,
        np.full_like(out_f32, expected_val),
        rtol=2e-2,
        atol=2.0,
    )

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


class FixedMMetadataSafetyTest(unittest.TestCase):
  """Backend-independent safety tests for fixed-m metadata computation."""

  def test_adversarial_v_magnitude_safely_disqualifies_fixed_m_metadata(self):
    """Verifies that |V| > v_max_bound (e.g. V=512) disqualifies fixed-m gating to prevent FP32 overflow."""
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
    """The reviewer's case: fp16, N=4096, Q=K=0, |V|=1 previously reported all_fixed=True."""
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
