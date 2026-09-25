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

"""Tests for the fused short-KV cross-attention Pallas kernel.

The kernel is not bit-identical to the XLA dot-product path it replaces (it
normalises the softmax after PV), so parity is asserted against an FP32 golden
reference, together with the stronger property that the kernel is at least as
close to that reference as the XLA path is. Kernel and `FlaxWanAttention`
integration tests run off-TPU through Pallas interpret mode (with
`wan_cross_attn_cpu_interpret=True` for the module integration test, since
production falls back to XLA off-TPU).
"""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import pyconfig, wan_runtime_options
from maxdiffusion.kernels import cross_attention_pallas
from maxdiffusion.kernels.cross_attention_pallas import cross_attention_pallas as xattn
from maxdiffusion.kernels.cross_attention_pallas import cross_attention_reference
from maxdiffusion.max_utils import create_device_mesh, get_flash_block_sizes
from maxdiffusion.models.attention_flax import FlaxWanAttention, _apply_attention_dot

DIM_HEAD = 128
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _on_tpu() -> bool:
  return jax.devices()[0].platform == "tpu"


def _inputs(batch, seq_q, seq_kv, heads, *, qk_gain=2.0, seed=0):
  """bf16 q/k/v; `qk_gain` sets the logit spread (post-scale std ~ qk_gain**2)."""
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
  feature = heads * DIM_HEAD
  q = (qk_gain * jax.random.normal(kq, (batch, seq_q, feature), jnp.float32)).astype(jnp.bfloat16)
  k = (qk_gain * jax.random.normal(kk, (batch, seq_kv, feature), jnp.float32)).astype(jnp.bfloat16)
  v = jax.random.normal(kv, (batch, seq_kv, feature), jnp.float32).astype(jnp.bfloat16)
  return q, k, v


def _rel_err(got, ref):
  got = np.asarray(got, np.float32)
  ref = np.asarray(ref, np.float32)
  return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def _xla_path(q, k, v, heads, k_prescaled=False):
  """The XLA dot-product fallback the kernel replaces, as Wan configures it."""
  return _apply_attention_dot(
      q, k, v, jnp.bfloat16, heads, DIM_HEAD, DIM_HEAD**-0.5, True, False, False, None, k_prescaled=k_prescaled
  )


class CrossAttentionPallasParityTest(parameterized.TestCase):

  def _check(self, got, golden, rel_tol=5e-3):
    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(got_np)), "kernel output contains non-finite values")
    np.testing.assert_allclose(got_np, np.asarray(golden, np.float32), atol=3e-2, rtol=3e-2)
    self.assertLess(_rel_err(got, golden), rel_tol)

  @parameterized.product(
      heads=(4, 5),
      sum_mode=cross_attention_pallas.SUM_MODES,
      max_mode=cross_attention_pallas.MAX_MODES,
  )
  def test_matches_golden(self, heads, sum_mode, max_mode):
    q, k, v = _inputs(1, 256, 128, heads)
    head_block = 1 if heads == 5 else 2
    got = xattn(
        q,
        k,
        v,
        heads=heads,
        block_q=128,
        head_block=head_block,
        sum_mode=sum_mode,
        max_mode=max_mode,
        interpret=not _on_tpu(),
    )
    self.assertEqual(got.shape, q.shape)
    self.assertEqual(got.dtype, q.dtype)
    self._check(got, cross_attention_reference(q, k, v, heads=heads))

  @parameterized.parameters(*cross_attention_pallas.MAX_MODES)
  def test_ragged_last_query_block(self, max_mode):
    # 300 rows run as three 112-row tiles, so the last reads 36 rows of stale
    # VMEM, which must neither leak into valid rows nor reach the output.
    q, k, v = _inputs(1, 300, 64, 2, seed=1)
    got = xattn(q, k, v, heads=2, block_q=128, head_block=2, max_mode=max_mode, interpret=not _on_tpu())
    self._check(got, cross_attention_reference(q, k, v, heads=2))

  def test_result_is_independent_of_tiling(self):
    # Rows and heads never interact, so the tiling must not change a single bit.
    q, k, v = _inputs(1, 256, 128, 4, seed=2)
    outs = [
        np.asarray(xattn(q, k, v, heads=4, block_q=bq, head_block=hb, interpret=not _on_tpu()), np.float32)
        for bq, hb in ((256, 4), (128, 2), (64, 1), (96, 4))
    ]
    for other in outs[1:]:
      np.testing.assert_array_equal(outs[0], other)

  def test_at_least_as_accurate_as_the_xla_path(self):
    heads = 4
    q, k, v = _inputs(1, 512, 256, heads, seed=3)
    golden = cross_attention_reference(q, k, v, heads=heads)
    kernel_err = _rel_err(xattn(q, k, v, heads=heads, block_q=128, head_block=2, interpret=not _on_tpu()), golden)
    xla_err = _rel_err(_xla_path(q, k, v, heads), golden)
    self.assertLessEqual(kernel_err, 1.1 * xla_err, f"kernel rel err {kernel_err:.3e} vs XLA {xla_err:.3e}")

  def test_k_prescaled(self):
    heads = 2
    q, k, v = _inputs(1, 128, 64, heads, seed=4)
    k_scaled = k * jnp.asarray(DIM_HEAD**-0.5, k.dtype)
    got = xattn(q, k_scaled, v, heads=heads, block_q=64, head_block=2, k_prescaled=True, interpret=not _on_tpu())
    self._check(got, cross_attention_reference(q, k_scaled, v, heads=heads, k_prescaled=True))

  def test_multi_batch(self):
    q, k, v = _inputs(2, 128, 64, 2, seed=5)
    got = xattn(q, k, v, heads=2, block_q=64, head_block=1, interpret=not _on_tpu())
    self._check(got, cross_attention_reference(q, k, v, heads=2))

  def test_large_logits_stay_finite(self):
    # Post-scale logits reach several hundred: exp overflows without the shift.
    q, k, v = _inputs(1, 128, 64, 2, qk_gain=30.0, seed=6)
    got = xattn(q, k, v, heads=2, block_q=64, head_block=2, interpret=not _on_tpu())
    self._check(got, cross_attention_reference(q, k, v, heads=2), rel_tol=2e-2)


class CrossAttentionPallasGuardTest(unittest.TestCase):

  def test_rejects_unaligned_dim_head(self):
    q = jnp.zeros((1, 64, 128), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      xattn(q, q, q, heads=2, dim_head=64, interpret=True)

  def test_rejects_head_block_not_dividing_heads(self):
    q, k, v = _inputs(1, 64, 64, 4)
    with self.assertRaisesRegex(ValueError, "must divide heads"):
      xattn(q, k, v, heads=4, head_block=3, interpret=True)

  def test_rejects_mismatched_kv(self):
    q, k, _ = _inputs(1, 64, 64, 2)
    with self.assertRaisesRegex(ValueError, "k and v"):
      xattn(q, k, k[:, :32], heads=2, head_block=2, interpret=True)

  def test_rejects_long_kv(self):
    q, k, v = _inputs(1, 64, cross_attention_pallas.MAX_KV_LEN + 8, 1)
    with self.assertRaisesRegex(ValueError, "MAX_KV_LEN"):
      xattn(q, k, v, heads=1, head_block=1, interpret=True)

  def test_rejects_unaligned_block_q(self):
    q, k, v = _inputs(1, 64, 64, 1)
    with self.assertRaisesRegex(ValueError, "multiple of 8"):
      xattn(q, k, v, heads=1, head_block=1, block_q=20, interpret=True)

  def test_rejects_unaligned_seq_q_when_block_q_covers_seq_q(self):
    q, k, v = _inputs(1, 30, 64, 1)
    with self.assertRaisesRegex(ValueError, "multiple of 8"):
      xattn(q, k, v, heads=1, head_block=1, block_q=64, interpret=True)

  def test_rejects_unaligned_kv_len(self):
    q, k, v = _inputs(1, 64, 60, 1)
    with self.assertRaisesRegex(ValueError, "multiple of 8"):
      xattn(q, k, v, heads=1, head_block=1, block_q=64, interpret=True)

  def test_rejects_unknown_modes(self):
    q, k, v = _inputs(1, 64, 64, 1)
    with self.assertRaisesRegex(ValueError, "max_mode"):
      xattn(q, k, v, heads=1, head_block=1, max_mode="bogus", interpret=True)
    with self.assertRaisesRegex(ValueError, "sum_mode"):
      xattn(q, k, v, heads=1, head_block=1, sum_mode="bogus", interpret=True)


class WanCrossAttentionDispatchTest(unittest.TestCase):
  """`wan_cross_attn_kernel` routing inside `FlaxWanAttention`."""

  HEADS = 8
  SEQ_Q = 2048
  SEQ_KV = 512

  def setUp(self):
    super().setUp()
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml")], unittest=True)
    self.config = pyconfig.config
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)
    wan_runtime_options.reset()
    self.addCleanup(wan_runtime_options.reset)

  def _run(self, mode, seq_q=None, seq_kv=None):
    seq_q = self.SEQ_Q if seq_q is None else seq_q
    seq_kv = self.SEQ_KV if seq_kv is None else seq_kv
    query_dim = self.HEADS * DIM_HEAD
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    with mock.patch.dict(os.environ, {"WAN_CROSS_ATTN_KERNEL": mode}):
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        attn = FlaxWanAttention(
            rngs=nnx.Rngs(k1),
            query_dim=query_dim,
            heads=self.HEADS,
            dim_head=DIM_HEAD,
            attention_kernel="dot_product",
            split_head_dim=True,
            mesh=self.mesh,
            flash_block_sizes=get_flash_block_sizes(self.config),
            is_self_attention=False,
            dtype=jnp.bfloat16,
        )
        hidden = jax.random.normal(k2, (1, seq_q, query_dim), jnp.bfloat16)
        context = jax.random.normal(k3, (1, seq_kv, query_dim), jnp.bfloat16)
        return attn(hidden_states=hidden, encoder_hidden_states=context)

  def test_rejects_unknown_mode(self):
    with self.assertRaisesRegex(ValueError, "wan_cross_attn_kernel"):
      self._run("bogus")

  def test_xla_mode_never_calls_the_kernel(self):
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      self._run("xla")
    spy.assert_not_called()

  @unittest.skipIf(_on_tpu(), "Off-TPU fallback only.")
  def test_falls_back_to_xla_off_tpu(self):
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      got = self._run("pallas")
    spy.assert_not_called()
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(self._run("xla"), np.float32))

  @unittest.skipUnless(_on_tpu(), "The Pallas path is TPU-only.")
  def test_pallas_mode_uses_the_kernel_and_matches_xla(self):
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      got = self._run("pallas")
    spy.assert_called()
    ref = self._run("xla")
    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(got_np)))
    np.testing.assert_allclose(got_np, np.asarray(ref, np.float32), atol=0.08, rtol=1e-2)
    self.assertLess(_rel_err(got, ref), 1e-2)

  def test_pallas_mode_cpu_interpret_integration(self):
    """Exercises FlaxWanAttention routing with Pallas cross-attention in CI under interpret mode."""
    with mock.patch.dict(os.environ, {"WAN_CROSS_ATTN_CPU_INTERPRET": "1"}):
      with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
        got = self._run("pallas")
      spy.assert_called()
      ref = self._run("xla")
      got_np = np.asarray(got, np.float32)
      self.assertTrue(np.all(np.isfinite(got_np)))
      np.testing.assert_allclose(got_np, np.asarray(ref, np.float32), atol=0.08, rtol=1e-2)

  def test_falls_back_to_xla_on_unaligned_kv_or_short_unaligned_q(self):
    with mock.patch.dict(os.environ, {"WAN_CROSS_ATTN_CPU_INTERPRET": "1"}):
      with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
        self._run("pallas", seq_q=2048, seq_kv=769)
        self._run("pallas", seq_q=300, seq_kv=512)
      spy.assert_not_called()


if __name__ == "__main__":
  unittest.main()
