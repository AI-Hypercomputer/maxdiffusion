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

  def _run_ring(self, q_in, k_in, v, use_fixed_m):
    """Runs the custom ring under shard_map with per-rank fixed_m_norms
    from the LOCAL q / initial K shard."""
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
      if use_fixed_m:
        qf = ql.astype(jnp.float32)
        kf = kl.astype(jnp.float32)
        qn_max = jnp.sqrt((qf * qf).sum(-1)).max(axis=1)  # (heads,)
        mk_h = jnp.sqrt((kf * kf).sum(-1)).max(axis=1)  # (heads,) local shard
        fixed_m_norms = (qn_max, mk_h)
      ring = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=self.block_sizes,
          orig_q_seq_len=self.shard_len,
          orig_kv_seq_len=self.shard_len,
          use_base2_exp=True,
          ring_axis=_RING_AXIS,
          ring_size=_RING_SIZE,
          use_fixed_m=use_fixed_m,
          fixed_m_norms=fixed_m_norms,
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
      gates.append(bound <= custom_splash.FIXED_M_RING_SAFE_BOUND)
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


if __name__ == "__main__":
  unittest.main()
