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
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import pyconfig
from maxdiffusion.kernels import cross_attention_pallas
from maxdiffusion.kernels.cross_attention_pallas import cross_attention_pallas as xattn
from maxdiffusion.kernels.cross_attention_pallas import cross_attention_reference
from maxdiffusion.max_utils import create_device_mesh, get_flash_block_sizes
from maxdiffusion.models.attention_flax import FlaxWanAttention, _apply_attention_dot

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# Kernel numerics grids and multi-device tests are skipped in CI; see
# end_to_end/tpu/run_wan_stack_tests.sh.
_SKIP_IN_GITHUB_ACTIONS = unittest.skipIf(
    IN_GITHUB_ACTIONS, "TPU kernel / multi-device test, skipped in GitHub Actions; run end_to_end/tpu/run_wan_stack_tests.sh"
)

DIM_HEAD = 128
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _on_tpu() -> bool:
  return jax.devices()[0].platform == "tpu"


def _on_unvalidated_tpu() -> bool:
  """On a TPU whose VMEM budget is not validated, FlaxWanAttention falls back to XLA by design."""
  return _on_tpu() and not cross_attention_pallas.vmem_budget_is_validated()


_UNVALIDATED_TPU_SKIP = (
    "Pallas cross-attention falls back to XLA on TPU kinds outside _VMEM_LIMIT_VALIDATED_KINDS "
    "(covered by test_falls_back_to_xla_on_unvalidated_tpu_kind)."
)


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


@_SKIP_IN_GITHUB_ACTIONS
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


class CrossAttentionVmemBudgetTest(unittest.TestCase):
  """VMEM budget resolution and the working-set estimate that gates the kernel."""

  WAN_SHAPE = {"seq_q": 18900, "seq_kv": 512, "dim_head": 128}

  @staticmethod
  def _mesh(*kinds, platform="tpu"):
    devices = np.array([types.SimpleNamespace(platform=platform, device_kind=k) for k in kinds], dtype=object)
    return types.SimpleNamespace(devices=devices)

  def test_explicit_value_wins(self):
    self.assertEqual(cross_attention_pallas._resolve_vmem_limit_bytes(123, self._mesh("TPU v4")), 123)

  def test_validated_kinds_get_the_default(self):
    for kinds in (("TPU v6 lite",) * 2, ("TPU7x",) * 2):
      with self.subTest(kinds=kinds):
        self.assertEqual(
            cross_attention_pallas._resolve_vmem_limit_bytes(None, self._mesh(*kinds)),
            cross_attention_pallas.DEFAULT_VMEM_LIMIT_BYTES,
        )

  def test_unvalidated_or_mixed_kinds_get_mosaic_default(self):
    for kinds in (("TPU v4",), ("TPU v5 lite",), ("TPU v5p",), ("TPU v6 lite", "TPU v4")):
      with self.subTest(kinds=kinds):
        self.assertIsNone(cross_attention_pallas._resolve_vmem_limit_bytes(None, self._mesh(*kinds)))

  def test_off_tpu_returns_default(self):
    self.assertEqual(
        cross_attention_pallas._resolve_vmem_limit_bytes(None, self._mesh("cpu", platform="cpu")),
        cross_attention_pallas.DEFAULT_VMEM_LIMIT_BYTES,
    )

  def test_estimate_at_wan_shape_needs_the_raised_budget(self):
    need = cross_attention_pallas.estimate_vmem_bytes(**self.WAN_SHAPE)
    # Above Mosaic's default scoped limit (<= 32 MiB on current TPUs), within 64 MiB.
    self.assertGreater(need, 32 * 2**20)
    self.assertLess(need, cross_attention_pallas.DEFAULT_VMEM_LIMIT_BYTES)

  def test_estimate_grows_with_tiles_and_kv(self):
    base = cross_attention_pallas.estimate_vmem_bytes(**self.WAN_SHAPE)
    self.assertLess(cross_attention_pallas.estimate_vmem_bytes(**self.WAN_SHAPE, block_q=1024), base)
    self.assertLess(cross_attention_pallas.estimate_vmem_bytes(**self.WAN_SHAPE, head_block=5), base)
    self.assertLess(cross_attention_pallas.estimate_vmem_bytes(**self.WAN_SHAPE, sum_mode="vpu"), base)
    self.assertGreater(cross_attention_pallas.estimate_vmem_bytes(**{**self.WAN_SHAPE, "seq_kv": 4096}), base)

  def test_unusable_reason(self):
    reason = cross_attention_pallas.vmem_unusable_reason
    self.assertIsNone(reason(**self.WAN_SHAPE, mesh=self._mesh("TPU v6 lite")))
    self.assertIsNone(reason(**self.WAN_SHAPE, mesh=self._mesh("TPU7x")))
    self.assertRegex(reason(**self.WAN_SHAPE, mesh=self._mesh("TPU v5 lite")), "not validated")
    # MAX_KV_LEN KV at the default tiles does not fit 64 MiB: fall back, don't fail to compile.
    long_kv = {**self.WAN_SHAPE, "seq_kv": cross_attention_pallas.MAX_KV_LEN}
    self.assertRegex(reason(**long_kv, mesh=self._mesh("TPU v6 lite")), "exceeds")
    # An explicit budget is trusted on any TPU.
    self.assertIsNone(reason(**self.WAN_SHAPE, mesh=self._mesh("TPU v5 lite"), vmem_limit_bytes=96 * 2**20))

  def test_kernel_rejects_a_working_set_over_an_explicit_budget(self):
    q, k, v = _inputs(1, 64, 64, 1)
    with self.assertRaisesRegex(ValueError, "VMEM working set"):
      xattn(q, k, v, heads=1, head_block=1, block_q=64, vmem_limit_bytes=1024, interpret=False)


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

  def _run(self, mode, seq_q=None, seq_kv=None, cpu_interpret=False):
    seq_q = self.SEQ_Q if seq_q is None else seq_q
    seq_kv = self.SEQ_KV if seq_kv is None else seq_kv
    query_dim = self.HEADS * DIM_HEAD
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
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
          attention_config={"wan_cross_attn_kernel": mode, "wan_cross_attn_cpu_interpret": cpu_interpret},
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
  @unittest.skipIf(_on_unvalidated_tpu(), _UNVALIDATED_TPU_SKIP)
  @_SKIP_IN_GITHUB_ACTIONS
  def test_pallas_mode_uses_the_kernel_and_matches_xla(self):
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      got = self._run("pallas")
    spy.assert_called()
    ref = self._run("xla")
    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(got_np)))
    np.testing.assert_allclose(got_np, np.asarray(ref, np.float32), atol=0.08, rtol=1e-2)
    self.assertLess(_rel_err(got, ref), 1e-2)

  @unittest.skipIf(_on_unvalidated_tpu(), _UNVALIDATED_TPU_SKIP)
  @_SKIP_IN_GITHUB_ACTIONS
  def test_pallas_mode_cpu_interpret_integration(self):
    """Exercises FlaxWanAttention routing with Pallas cross-attention in CI under interpret mode."""
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      got = self._run("pallas", cpu_interpret=True)
    spy.assert_called()
    ref = self._run("xla")
    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(got_np)))
    np.testing.assert_allclose(got_np, np.asarray(ref, np.float32), atol=0.08, rtol=1e-2)

  @unittest.skipUnless(_on_tpu(), "VMEM gating only applies on TPU.")
  def test_falls_back_to_xla_on_unvalidated_tpu_kind(self):
    with mock.patch.object(cross_attention_pallas, "_VMEM_LIMIT_VALIDATED_KINDS", ("no-such-tpu",)):
      with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
        got = self._run("pallas")
    spy.assert_not_called()
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(self._run("xla"), np.float32))

  @_SKIP_IN_GITHUB_ACTIONS
  def test_falls_back_to_xla_on_unaligned_kv_or_short_unaligned_q(self):
    with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
      self._run("pallas", seq_q=2048, seq_kv=769, cpu_interpret=True)
      self._run("pallas", seq_q=300, seq_kv=512, cpu_interpret=True)
    spy.assert_not_called()

  @unittest.skipIf(_on_unvalidated_tpu(), _UNVALIDATED_TPU_SKIP)
  @_SKIP_IN_GITHUB_ACTIONS
  def test_i2v_cross_attention_uses_pallas_for_text_and_xla_for_image(self):
    """I2V splits encoder_hidden_states into padded image KV (masked XLA) and text KV (Pallas)."""
    query_dim = self.HEADS * DIM_HEAD
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(17), 3)
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
          added_kv_proj_dim=query_dim,
          image_seq_len=257,
          dtype=jnp.bfloat16,
      )
      padded_img_len = ((257 + attn.alignment - 1) // attn.alignment) * attn.alignment
      total_kv_len = padded_img_len + self.SEQ_KV
      hidden = jax.random.normal(k2, (1, self.SEQ_Q, query_dim), jnp.bfloat16)
      context = jax.random.normal(k3, (1, total_kv_len, query_dim), jnp.bfloat16)
      img_mask = jnp.concatenate(
          [jnp.ones((1, 257), dtype=jnp.int32), jnp.zeros((1, padded_img_len - 257), dtype=jnp.int32)], axis=1
      )
      txt_mask = jnp.ones((1, self.SEQ_KV), dtype=jnp.int32)
      mask = jnp.concatenate([img_mask, txt_mask], axis=1)

      # The routing switches are build-time attributes (from attention_config).
      attn.wan_cross_attn_kernel = "xla"
      ref = attn(hidden_states=hidden, encoder_hidden_states=context, encoder_attention_mask=mask)

      attn.wan_cross_attn_kernel = "pallas"
      attn.wan_cross_attn_cpu_interpret = True
      with mock.patch.object(cross_attention_pallas, "cross_attention_pallas", wraps=xattn) as spy:
        got = attn(hidden_states=hidden, encoder_hidden_states=context, encoder_attention_mask=mask)
      self.assertEqual(spy.call_count, 1)
      self.assertEqual(spy.call_args.args[1].shape[1], self.SEQ_KV)

    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(got_np)))
    np.testing.assert_allclose(got_np, np.asarray(ref, np.float32), atol=0.08, rtol=1e-2)


if __name__ == "__main__":
  unittest.main()
