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

"""Unit tests for custom splash attention and custom ring attention backward pass."""

import functools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import ring_attention_kernel

_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)


def _reference_attention(q, k, v, actual_q_len, actual_kv_len, use_base2_exp):
  """Reference attention in fp32 returning shape (Hq, Dv, actual_q_len)."""
  q_f = q[:, :actual_q_len, :].astype(jnp.float32)
  k_f = k[:, :actual_kv_len, :].astype(jnp.float32)
  v_f = v[:, :actual_kv_len, :].astype(jnp.float32)
  hq = q_f.shape[0]
  hkv = k_f.shape[0]
  q_per_kv = hq // hkv
  k_f = jnp.repeat(k_f, q_per_kv, axis=0)
  v_f = jnp.repeat(v_f, q_per_kv, axis=0)
  logits = jnp.einsum("hsd,htd->hst", q_f, k_f)
  if use_base2_exp:
    logits = logits * _LN2
  probs = jax.nn.softmax(logits, axis=-1)
  out = jnp.einsum("hst,htd->hds", probs, v_f)
  return out.astype(q.dtype)


class CustomSplashBackwardTest(unittest.TestCase):
  """Tests custom splash attention backward pass and ring attention backward pass."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Custom Pallas splash kernel requires TPU.")
    self.block_sizes = custom_splash._BlockSizes(
        block_q=1024,
        block_kv=1024,
        block_kv_compute=512,
        block_kv_compute_in=256,
    )

  def test_single_device_bwd_exact_multiples(self):
    hq, hkv, sq, skv, d = 4, 4, 2048, 2048, 128
    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = (jax.random.normal(k1, (hq, sq, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (hkv, skv, d), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (hkv, skv, d), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (hq, d, sq), jnp.bfloat16)

    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=sq,
        orig_kv_seq_len=skv,
        use_base2_exp=True,
    )

    out_kern, vjp_kern = jax.vjp(kernel, q, k, v)
    dq_kern, dk_kern, dv_kern = vjp_kern(do)

    out_ref, vjp_ref = jax.vjp(
        lambda q_, k_, v_: _reference_attention(q_, k_, v_, sq, skv, True),
        q,
        k,
        v,
    )
    dq_ref, dk_ref, dv_ref = vjp_ref(do)

    self.assertLess(float(jnp.max(jnp.abs(out_kern.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
    self.assertLess(float(jnp.max(jnp.abs(dq_kern.astype(jnp.float32) - dq_ref.astype(jnp.float32)))), 1e-3)
    self.assertLess(float(jnp.max(jnp.abs(dk_kern.astype(jnp.float32) - dk_ref.astype(jnp.float32)))), 1e-3)
    self.assertLess(float(jnp.max(jnp.abs(dv_kern.astype(jnp.float32) - dv_ref.astype(jnp.float32)))), 2e-3)

  def test_single_device_bwd_ragged_gqa_and_vmapped(self):
    batch, hq, hkv, sq, skv, d = 2, 4, 2, 2048, 2048, 128
    actual_q_len, actual_kv_len = 1500, 1600
    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(1), 4)
    q = (jax.random.normal(k1, (batch, hq, sq, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (batch, hkv, skv, d), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (batch, hkv, skv, d), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (batch, hq, d, actual_q_len), jnp.bfloat16)

    kernel = custom_splash.make_splash_mha(
        block_sizes=self.block_sizes,
        orig_q_seq_len=actual_q_len,
        orig_kv_seq_len=actual_kv_len,
        use_base2_exp=True,
    )
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, 0, 0))

    out_kern, vjp_kern = jax.vjp(vmapped_kernel, q, k, v)
    dq_kern, dk_kern, dv_kern = vjp_kern(do)

    vmapped_ref = jax.vmap(
        lambda q_, k_, v_: _reference_attention(q_, k_, v_, actual_q_len, actual_kv_len, True),
        in_axes=(0, 0, 0),
    )
    out_ref, vjp_ref = jax.vjp(vmapped_ref, q, k, v)
    dq_ref, dk_ref, dv_ref = vjp_ref(do)

    self.assertLess(float(jnp.max(jnp.abs(out_kern.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
    self.assertLess(
        float(jnp.max(jnp.abs(dq_kern[:, :, :actual_q_len].astype(jnp.float32) - dq_ref[:, :, :actual_q_len].astype(jnp.float32)))),
        1e-3,
    )
    self.assertEqual(float(jnp.max(jnp.abs(dq_kern[:, :, actual_q_len:].astype(jnp.float32)))), 0.0)
    self.assertLess(
        float(jnp.max(jnp.abs(dk_kern[:, :, :actual_kv_len].astype(jnp.float32) - dk_ref[:, :, :actual_kv_len].astype(jnp.float32)))),
        1e-3,
    )
    self.assertEqual(float(jnp.max(jnp.abs(dk_kern[:, :, actual_kv_len:].astype(jnp.float32)))), 0.0)
    self.assertLess(
        float(jnp.max(jnp.abs(dv_kern[:, :, :actual_kv_len].astype(jnp.float32) - dv_ref[:, :, :actual_kv_len].astype(jnp.float32)))),
        2e-3,
    )
    self.assertEqual(float(jnp.max(jnp.abs(dv_kern[:, :, actual_kv_len:].astype(jnp.float32)))), 0.0)

  def test_single_device_bwd_fused_aliasing_and_unfused(self):
    hq, hkv, sq, skv, d = 4, 2, 2048, 4096, 128
    actual_q_len, actual_kv_len = 1500, 3500  # grid_width = ceil(3500/512) = 7 > 3
    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    q = (jax.random.normal(k1, (hq, sq, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (hkv, skv, d), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (hkv, skv, d), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (hq, d, actual_q_len), jnp.bfloat16)

    out_ref, vjp_ref = jax.vjp(
        lambda q_, k_, v_: _reference_attention(q_, k_, v_, actual_q_len, actual_kv_len, True),
        q,
        k,
        v,
    )
    dq_ref, dk_ref, dv_ref = vjp_ref(do)

    for use_fused, dq_red in [(True, 3), (True, None), (False, None)]:
      bs = custom_splash._BlockSizes(
          block_q=512,
          block_kv=512,
          block_kv_compute=256,
          block_kv_compute_in=256,
          use_fused_bwd_kernel=use_fused,
          dq_reduction_steps=dq_red,
      )
      kernel = custom_splash.make_splash_mha(
          block_sizes=bs,
          orig_q_seq_len=actual_q_len,
          orig_kv_seq_len=actual_kv_len,
          use_base2_exp=True,
      )
      out_k, vjp_k = jax.vjp(kernel, q, k, v)
      dq_k, dk_k, dv_k = vjp_k(do)
      self.assertLess(float(jnp.max(jnp.abs(out_k.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(
          float(jnp.max(jnp.abs(dq_k[:, :actual_q_len].astype(jnp.float32) - dq_ref[:, :actual_q_len].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dq_k[:, actual_q_len:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dk_k[:, :actual_kv_len].astype(jnp.float32) - dk_ref[:, :actual_kv_len].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dk_k[:, actual_kv_len:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dv_k[:, :actual_kv_len].astype(jnp.float32) - dv_ref[:, :actual_kv_len].astype(jnp.float32)))),
          2e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dv_k[:, actual_kv_len:].astype(jnp.float32)))), 0.0)

  def test_ring_attention_bwd_ragged_gqa_vmapped(self):
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("Requires at least 4 TPU devices for ring attention test.")
    ring_size = 4
    mesh = jax.sharding.Mesh(np.array(devices[:ring_size]), ("context",))
    batch, hq, hkv, sq_local, skv_local, d = 2, 4, 2, 1024, 1024, 128
    actual_q_local, actual_kv_local = 800, 900
    bsizes = custom_splash._BlockSizes(
        block_q=512,
        block_kv=512,
        block_kv_compute=256,
        block_kv_compute_in=256,
    )

    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(2), 4)
    q_global = (jax.random.normal(k1, (ring_size, batch, hq, sq_local, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k_global = jax.random.normal(k2, (ring_size, batch, hkv, skv_local, d), jnp.bfloat16) * scale
    v_global = jax.random.normal(k3, (ring_size, batch, hkv, skv_local, d), jnp.bfloat16) * scale
    do_global = jax.random.normal(k4, (ring_size, batch, hq, actual_q_local, d), jnp.bfloat16)

    def _ref_single_batch(qg, kg, vg):
      q_list = [qg[r, :, :actual_q_local, :].astype(jnp.float32) for r in range(ring_size)]
      k_list = [kg[r, :, :actual_kv_local, :].astype(jnp.float32) for r in range(ring_size)]
      v_list = [vg[r, :, :actual_kv_local, :].astype(jnp.float32) for r in range(ring_size)]
      q_cat = jnp.concatenate(q_list, axis=1)
      k_cat = jnp.concatenate(k_list, axis=1)
      v_cat = jnp.concatenate(v_list, axis=1)
      k_cat = jnp.repeat(k_cat, hq // hkv, axis=0)
      v_cat = jnp.repeat(v_cat, hq // hkv, axis=0)
      logits = jnp.einsum("hsd,htd->hst", q_cat, k_cat) * _LN2
      probs = jax.nn.softmax(logits, axis=-1)
      out_cat = jnp.einsum("hst,htd->hsd", probs, v_cat)
      return jnp.stack(
          [out_cat[:, r * actual_q_local : (r + 1) * actual_q_local, :] for r in range(ring_size)],
          axis=0,
      ).astype(qg.dtype)

    ref_fn = jax.vmap(_ref_single_batch, in_axes=(1, 1, 1), out_axes=1)
    out_ref, vjp_ref = jax.vjp(ref_fn, q_global, k_global, v_global)
    dq_ref, dk_ref, dv_ref = vjp_ref(do_global)

    for use_fused in [True, False]:
      bsizes = custom_splash._BlockSizes(
          block_q=512,
          block_kv=512,
          block_kv_compute=256,
          block_kv_compute_in=256,
          use_fused_bwd_kernel=use_fused,
      )
      ring_kernel = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=actual_q_local,
          orig_kv_seq_len=actual_kv_local,
          use_base2_exp=True,
          ring_axis="context",
      )
      vmapped_ring = jax.vmap(ring_kernel, in_axes=(0, 0, 0))

      p = jax.sharding.PartitionSpec("context", None, None, None, None)

      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(p, p, p, p),
          out_specs=(p, p, p, p),
          check_vma=False,
      )
      def run_ring(q_sh, k_sh, v_sh, do_sh):
        q_s, k_s, v_s, do_s = q_sh[0], k_sh[0], v_sh[0], do_sh[0]
        out_s, vjp_s = jax.vjp(vmapped_ring, q_s, k_s, v_s)
        dq_s, dk_s, dv_s = vjp_s(do_s)
        return out_s[None], dq_s[None], dk_s[None], dv_s[None]

      out_ring, dq_ring, dk_ring, dv_ring = run_ring(q_global, k_global, v_global, do_global)

      self.assertLess(float(jnp.max(jnp.abs(out_ring.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(
          float(jnp.max(jnp.abs(dq_ring[:, :, :, :actual_q_local].astype(jnp.float32) - dq_ref[:, :, :, :actual_q_local].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dq_ring[:, :, :, actual_q_local:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dk_ring[:, :, :, :actual_kv_local].astype(jnp.float32) - dk_ref[:, :, :, :actual_kv_local].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dk_ring[:, :, :, actual_kv_local:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dv_ring[:, :, :, :actual_kv_local].astype(jnp.float32) - dv_ref[:, :, :, :actual_kv_local].astype(jnp.float32)))),
          2e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dv_ring[:, :, :, actual_kv_local:].astype(jnp.float32)))), 0.0)

  def test_single_device_bwd_unpadded_non_divisible_seqlens(self):
    """Tests directly passing unpadded Q/K/V whose sequence lengths are not divisible by block size."""
    hq, hkv, sq, skv, d = 4, 2, 1350, 1777, 128
    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(101), 4)
    q = (jax.random.normal(k1, (hq, sq, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (hkv, skv, d), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (hkv, skv, d), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (hq, d, sq), jnp.bfloat16)

    out_ref, vjp_ref = jax.vjp(
        lambda q_, k_, v_: _reference_attention(q_, k_, v_, sq, skv, True),
        q,
        k,
        v,
    )
    dq_ref, dk_ref, dv_ref = vjp_ref(do)

    for use_fused, dq_red in [(True, 3), (True, None), (False, None)]:
      bs = custom_splash._BlockSizes(
          block_q=512,
          block_kv=512,
          block_kv_compute=256,
          block_kv_compute_in=128,
          use_fused_bwd_kernel=use_fused,
          dq_reduction_steps=dq_red,
      )
      kernel = custom_splash.make_splash_mha(
          block_sizes=bs,
          orig_q_seq_len=sq,
          orig_kv_seq_len=skv,
          use_base2_exp=True,
      )
      out_k, vjp_k = jax.vjp(kernel, q, k, v)
      dq_k, dk_k, dv_k = vjp_k(do)

      self.assertEqual(out_k.shape, (hq, d, sq))
      self.assertEqual(dq_k.shape, (hq, sq, d))
      self.assertEqual(dk_k.shape, (hkv, skv, d))
      self.assertEqual(dv_k.shape, (hkv, skv, d))

      self.assertLess(float(jnp.max(jnp.abs(out_k.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dq_k.astype(jnp.float32) - dq_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dk_k.astype(jnp.float32) - dk_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dv_k.astype(jnp.float32) - dv_ref.astype(jnp.float32)))), 2e-3)

  def test_single_device_bwd_padded_with_garbage_in_kv_tail(self):
    """Verifies that non-divisible orig_kv_seq_len ignores extreme garbage values in padded KV tail."""
    hq, hkv, padded_sq, padded_skv, d = 4, 2, 2048, 2048, 128
    actual_q_len, actual_kv_len = 1350, 1777
    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(102), 4)
    q = (jax.random.normal(k1, (hq, padded_sq, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (hkv, padded_skv, d), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (hkv, padded_skv, d), jnp.bfloat16) * scale
    # Inject huge garbage numbers into the KV padding tail [actual_kv_len:]
    k = k.at[:, actual_kv_len:, :].set(1000.0)
    v = v.at[:, actual_kv_len:, :].set(1000.0)
    do = jax.random.normal(k4, (hq, d, actual_q_len), jnp.bfloat16)

    out_ref, vjp_ref = jax.vjp(
        lambda q_, k_, v_: _reference_attention(q_, k_, v_, actual_q_len, actual_kv_len, True),
        q,
        k,
        v,
    )
    dq_ref, dk_ref, dv_ref = vjp_ref(do)

    for use_fused in [True, False]:
      bs = custom_splash._BlockSizes(
          block_q=512,
          block_kv=512,
          block_kv_compute=256,
          block_kv_compute_in=128,
          use_fused_bwd_kernel=use_fused,
      )
      kernel = custom_splash.make_splash_mha(
          block_sizes=bs,
          orig_q_seq_len=actual_q_len,
          orig_kv_seq_len=actual_kv_len,
          use_base2_exp=True,
      )
      out_k, vjp_k = jax.vjp(kernel, q, k, v)
      dq_k, dk_k, dv_k = vjp_k(do)

      self.assertLess(float(jnp.max(jnp.abs(out_k.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(
          float(jnp.max(jnp.abs(dq_k[:, :actual_q_len].astype(jnp.float32) - dq_ref[:, :actual_q_len].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dq_k[:, actual_q_len:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dk_k[:, :actual_kv_len].astype(jnp.float32) - dk_ref[:, :actual_kv_len].astype(jnp.float32)))),
          1e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dk_k[:, actual_kv_len:].astype(jnp.float32)))), 0.0)
      self.assertLess(
          float(jnp.max(jnp.abs(dv_k[:, :actual_kv_len].astype(jnp.float32) - dv_ref[:, :actual_kv_len].astype(jnp.float32)))),
          2e-3,
      )
      self.assertEqual(float(jnp.max(jnp.abs(dv_k[:, actual_kv_len:].astype(jnp.float32)))), 0.0)

  def test_ring_attention_bwd_unpadded_non_divisible_seqlens(self):
    """Tests 4-device ring attention with unpadded local sequence lengths not divisible by block sizes."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("Requires at least 4 TPU devices for ring attention test.")
    ring_size = 4
    mesh = jax.sharding.Mesh(np.array(devices[:ring_size]), ("context",))
    batch, hq, hkv, sq_local, skv_local, d = 2, 4, 2, 650, 733, 128

    scale = 1.0 / math.sqrt(d)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(103), 4)
    q_global = (jax.random.normal(k1, (ring_size, batch, hq, sq_local, d), jnp.bfloat16) * scale * _LOG2E).astype(jnp.bfloat16)
    k_global = jax.random.normal(k2, (ring_size, batch, hkv, skv_local, d), jnp.bfloat16) * scale
    v_global = jax.random.normal(k3, (ring_size, batch, hkv, skv_local, d), jnp.bfloat16) * scale
    do_global = jax.random.normal(k4, (ring_size, batch, hq, sq_local, d), jnp.bfloat16)

    def _ref_single_batch(qg, kg, vg):
      q_cat = jnp.concatenate([qg[r].astype(jnp.float32) for r in range(ring_size)], axis=1)
      k_cat = jnp.concatenate([kg[r].astype(jnp.float32) for r in range(ring_size)], axis=1)
      v_cat = jnp.concatenate([vg[r].astype(jnp.float32) for r in range(ring_size)], axis=1)
      k_cat = jnp.repeat(k_cat, hq // hkv, axis=0)
      v_cat = jnp.repeat(v_cat, hq // hkv, axis=0)
      logits = jnp.einsum("hsd,htd->hst", q_cat, k_cat) * _LN2
      probs = jax.nn.softmax(logits, axis=-1)
      out_cat = jnp.einsum("hst,htd->hsd", probs, v_cat)
      return jnp.stack(
          [out_cat[:, r * sq_local : (r + 1) * sq_local, :] for r in range(ring_size)],
          axis=0,
      ).astype(qg.dtype)

    ref_fn = jax.vmap(_ref_single_batch, in_axes=(1, 1, 1), out_axes=1)
    out_ref, vjp_ref = jax.vjp(ref_fn, q_global, k_global, v_global)
    dq_ref, dk_ref, dv_ref = vjp_ref(do_global)

    for use_fused in [True, False]:
      bsizes = custom_splash._BlockSizes(
          block_q=512,
          block_kv=512,
          block_kv_compute=256,
          block_kv_compute_in=128,
          use_fused_bwd_kernel=use_fused,
      )
      ring_kernel = ring_attention_kernel.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=sq_local,
          orig_kv_seq_len=skv_local,
          use_base2_exp=True,
          ring_axis="context",
      )
      vmapped_ring = jax.vmap(ring_kernel, in_axes=(0, 0, 0))

      p = jax.sharding.PartitionSpec("context", None, None, None, None)

      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(p, p, p, p),
          out_specs=(p, p, p, p),
          check_vma=False,
      )
      def run_ring(q_sh, k_sh, v_sh, do_sh):
        q_s, k_s, v_s, do_s = q_sh[0], k_sh[0], v_sh[0], do_sh[0]
        out_s, vjp_s = jax.vjp(vmapped_ring, q_s, k_s, v_s)
        dq_s, dk_s, dv_s = vjp_s(do_s)
        return out_s[None], dq_s[None], dk_s[None], dv_s[None]

      out_ring, dq_ring, dk_ring, dv_ring = run_ring(q_global, k_global, v_global, do_global)

      self.assertEqual(out_ring.shape, (ring_size, batch, hq, sq_local, d))
      self.assertEqual(dq_ring.shape, (ring_size, batch, hq, sq_local, d))
      self.assertEqual(dk_ring.shape, (ring_size, batch, hkv, skv_local, d))
      self.assertEqual(dv_ring.shape, (ring_size, batch, hkv, skv_local, d))

      self.assertLess(float(jnp.max(jnp.abs(out_ring.astype(jnp.float32) - out_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dq_ring.astype(jnp.float32) - dq_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dk_ring.astype(jnp.float32) - dk_ref.astype(jnp.float32)))), 1e-3)
      self.assertLess(float(jnp.max(jnp.abs(dv_ring.astype(jnp.float32) - dv_ref.astype(jnp.float32)))), 2e-3)


if __name__ == "__main__":
  unittest.main()
