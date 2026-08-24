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

  def _run_ring(self, q_in, k_in, v, ring_size: int = 2, use_fixed_m: bool = True):
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
      )
      return ring(ql, kl, vl)

    return _body(q_in, k_in, v)

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
      ring_kernel = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=ring_size,
          use_fixed_m=True,
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


if __name__ == "__main__":
  unittest.main()
