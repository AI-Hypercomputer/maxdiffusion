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

"""C3 gate: may the custom splash kernel be handed PHYSICALLY UNPADDED q/k/v?

Context
-------
Production calls `_pad_data_for_flash` on q, k and v, materialising a full
185 MiB copy of each purely to round the sequence dimension up to a multiple of
the Pallas block size. Static analysis says most of that is unnecessary:

  * K/V: the kernel never *masks* the KV tail, it *slices* it
    (`slice_k_len = kv_seq_len % bkv_compute`, using the UNPADDED length -- see
    `last_compute_body_fixed` / `last_compute_body_online` in
    custom_splash_attention.py and the load-bearing comment above them). No
    compute path reads a padded K or V row, so their contents are irrelevant.

  * Q: rows are independent (the running max is over KV, never across the `bq`
    row axis), and the kernel's OUTPUT is already ragged today -- `out_shape`'s
    last dim is `actual_q_seq_len` against a `bq`-wide BlockSpec -- so Pallas is
    already clipping a non-divisible last block in this very kernel.

The one thing static analysis CANNOT settle is whether Pallas clips the ragged
last-block *input* DMA, or issues an unclipped read past the end of the array.
`compiler_params` sets `disable_bounds_checks=True`. If the DMA is unclipped,
passing an unpadded array is a silent out-of-bounds HBM read.

**That is the only question these tests exist to answer.** Everything else about
C3 is already proven on paper.

Why the existing coverage does not answer it
--------------------------------------------
`custom_splash_fixed_m_test.test_non_divisible_sequence_context_padding_fixed_m`
looks like it covers this, but it pads its inputs to a block boundary before the
call (`q_in_padded = jnp.pad(...)`) and only passes a ragged *logical* length.
That is exactly today's regime. No existing test ever hands the kernel a
physically non-block-aligned array.

Test design
-----------
Each case runs the kernel twice with IDENTICAL LOGICAL INPUTS:
  reference -- physically padded to a block boundary (today's behaviour)
  candidate -- physically unpadded (what C3 proposes)
and asserts the outputs are **bit-identical**. This is pure data movement: the
grid, the block sizes and every slice length are computed from the unpadded
logical length and are therefore identical between the two runs. The same
arithmetic happens in the same order on the same values, so anything other than
exact equality means memory outside the array was read.

K/V-unpadded and Q-unpadded are separate cases so the two halves of C3 can be
gated independently.

Two failure modes are guarded against explicitly:
  1. Passing by luck on a freshly-zeroed buffer -- see `_dirty_device_memory`,
     and the repeat-run case.
  2. A harness bug making both sides equally wrong -- every case also checks the
     padded reference against a dense f32 softmax reference.

> A NOTE ON THE LIMITS OF THIS TEST. `_dirty_device_memory` is a heuristic. JAX
> gives no control over HBM placement, so we cannot *guarantee* the bytes past
> the end of the array are NaN. A PASS is therefore strong but not absolute
> evidence; a FAILURE is conclusive. Treat a pass as "no evidence of an
> unclipped DMA under adversarial conditions", not as a proof of clipping.
"""

import gc
import math
import unittest

import jax
import jax.numpy as jnp

from maxdiffusion.kernels import custom_splash_attention as custom_splash

_LOG2E = math.log2(math.e)


def _dirty_device_memory(num_buffers: int = 8, mb_each: int = 64) -> None:
  """Fills and releases device buffers of NaN to poison the allocator pool.

  The failure mode this defends against: an unclipped DMA reads whatever
  happens to sit past the end of the array. On a fresh device that is often
  zero, which is exactly the padding value the kernel would have seen anyway --
  so the bug would produce a correct answer and the test would pass for the
  wrong reason.

  By allocating NaN buffers and then dropping them, subsequent allocations are
  likely (not guaranteed) to be served from memory containing NaN. NaN is a
  stronger probe than a large finite value: a large finite value could be
  squashed back to something finite by a downstream mask or a saturating
  operation, whereas NaN propagates through every arithmetic path in the
  softmax and cannot be masked away once it enters an accumulation.
  """
  elems = mb_each * 1024 * 1024 // 4
  junk = []
  for _ in range(num_buffers):
    junk.append(jax.block_until_ready(jnp.full((elems,), jnp.nan, dtype=jnp.float32)))
  del junk
  gc.collect()


class CustomSplashUnpaddedInputTest(unittest.TestCase):
  """Bit-identity of the kernel when fed physically unpadded q/k/v."""

  heads = 4
  head_dim = 64

  # Deliberately non-divisible by `bq` in BOTH dimensions.
  #   q_len  = 1008, bq = 512  -> grid_height = 2, last block covers [512, 1024)
  #                               against an array of only 1008 rows.
  #   kv_len = 1001, bkv = 512 -> grid_width  = 2, tail slice length 489.
  q_len = 1008
  kv_len = 1001
  bq = 512

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "cpu":
      self.skipTest("Pallas splash kernel requires TPU.")
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.grid_height = math.ceil(self.q_len / self.bq)
    self.q_padded_len = self.grid_height * self.bq
    self.kv_padded_len = math.ceil(self.kv_len / self.bq) * self.bq

  # ---------------------------------------------------------------- helpers

  def _inputs(self):
    """Returns logical (unpadded) bf16 q, k, v in the kernel's own convention."""
    q = jax.random.normal(jax.random.PRNGKey(11), (self.heads, self.q_len, self.head_dim), jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(12), (self.heads, self.kv_len, self.head_dim), jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(13), (self.heads, self.kv_len, self.head_dim), jnp.bfloat16)
    q_in = (q * _LOG2E).astype(jnp.bfloat16)
    k_in = (k * self.scale).astype(jnp.bfloat16)
    return q_in, k_in, v, q, k

  def _pad_seq(self, x, target):
    if x.shape[1] == target:
      return x
    return jnp.pad(x, ((0, 0), (0, target - x.shape[1]), (0, 0)))

  def _metadata(self, q_in, k_in):
    """Builds `mk` and `k_mean` from UNPADDED inputs.

    This is the C3b norm-vector trick: the per-row norms are computed on the
    unpadded query and then the *norm vector* is zero-padded to the block grid,
    instead of zero-padding the (far larger) activation. Zero rows contribute 0
    to a max over non-negative values, so the resulting `mk` is identical to the
    one derived from a zero-padded activation -- bit-identical, not merely
    close.
    """
    k_mean = jnp.mean(k_in.astype(jnp.float32), axis=1)  # (heads, dim)
    recenter, safe_bound = custom_splash.get_fixed_m_constants(self.kv_len, is_ring=False)

    k_centered = k_in.astype(jnp.float32) - k_mean[:, None, :]
    mk_h = jnp.sqrt((k_centered**2).sum(-1)).max(axis=-1)  # (heads,)

    row_norm_sq = (q_in.astype(jnp.float32) ** 2).sum(-1)  # (heads, q_len)
    pad = self.q_padded_len - row_norm_sq.shape[1]
    if pad:
      row_norm_sq = jnp.pad(row_norm_sq, ((0, 0), (0, pad)))
    qn_max = jnp.sqrt(row_norm_sq.reshape(self.heads, self.grid_height, self.bq).max(axis=-1))

    bound = qn_max * mk_h[:, None]
    mk = jnp.stack([jnp.ceil(bound) - recenter, (bound <= safe_bound).astype(jnp.float32)], axis=0)
    return mk, k_mean

  def _run(self, q_in, k_in, v_in, mk, k_mean, *, use_fixed_m, uniform_fixed_m, bkv_compute=None):
    """Invokes the kernel. Block sizes derive only from logical lengths."""
    # `uniform_fixed_m` is a sub-mode of fixed-m; the kernel rejects the
    # combination outright ("uniform_fixed_m requires use_fixed_m"). Asserting
    # here turns a misconfigured case into a loud harness failure instead of a
    # case that aborts at kernel construction and never touches the device --
    # which would look like a kernel finding but is really a test bug.
    assert not (uniform_fixed_m and not use_fixed_m), "uniform_fixed_m requires use_fixed_m"
    bkv_compute = bkv_compute or self.bq
    block_sizes = custom_splash._BlockSizes(
        block_q=self.bq,
        block_kv=self.bq,
        block_kv_compute=bkv_compute,
        block_kv_compute_in=bkv_compute,
    )
    kernel = custom_splash.make_splash_mha(
        block_sizes=block_sizes,
        orig_q_seq_len=self.q_len,
        orig_kv_seq_len=self.kv_len,
        use_base2_exp=True,
        use_fixed_m=use_fixed_m,
        uniform_fixed_m=uniform_fixed_m,
    )
    if use_fixed_m:
      out = kernel(q_in, k_in, v_in, mk, k_mean)
    else:
      out = kernel(q_in, k_in, v_in)
    return jnp.swapaxes(out, 1, 2).astype(jnp.float32)  # (heads, seq, dim)

  def _dense_reference(self, q, k, v):
    qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
    logits = jnp.einsum("hsd,htd->hst", qf, kf) * self.scale
    return jnp.einsum("hst,htd->hsd", jax.nn.softmax(logits, axis=-1), vf)

  def _assert_reference_is_sane(self, reference, q, k, v):
    """Guards against a harness bug that would make both sides equally wrong."""
    self.assertTrue(bool(jnp.all(jnp.isfinite(reference))), "padded reference is not finite")
    self.assertGreater(float(jnp.max(jnp.abs(reference))), 0.0, "padded reference is all zeros")
    dense = self._dense_reference(q, k, v)
    diff = float(jnp.max(jnp.abs(reference[:, : self.q_len] - dense)))
    self.assertLess(diff, 5e-2, f"padded reference disagrees with dense softmax: {diff=}")

  def _compare(self, *, unpad_q, unpad_kv, use_fixed_m=True, uniform_fixed_m=None, bkv_compute=None, dirty=True):
    """Core assertion: unpadded inputs reproduce padded inputs bit-for-bit.

    `uniform_fixed_m` defaults to tracking `use_fixed_m` rather than to a bare
    `True`, because `uniform_fixed_m=True` with `use_fixed_m=False` is rejected
    by the kernel at construction time. Defaulting it to `True` made the online
    cases abort before reaching the device -- they looked like failures of the
    kernel when they were failures of this harness.
    """
    if uniform_fixed_m is None:
      uniform_fixed_m = use_fixed_m
    q_in, k_in, v, q_raw, k_raw = self._inputs()
    mk, k_mean = self._metadata(q_in, k_in)

    reference = self._run(
        self._pad_seq(q_in, self.q_padded_len),
        self._pad_seq(k_in, self.kv_padded_len),
        self._pad_seq(v, self.kv_padded_len),
        mk,
        k_mean,
        use_fixed_m=use_fixed_m,
        uniform_fixed_m=uniform_fixed_m,
        bkv_compute=bkv_compute,
    )
    jax.block_until_ready(reference)
    self._assert_reference_is_sane(reference, q_raw, k_raw, v)

    if dirty:
      _dirty_device_memory()

    candidate = self._run(
        q_in if unpad_q else self._pad_seq(q_in, self.q_padded_len),
        k_in if unpad_kv else self._pad_seq(k_in, self.kv_padded_len),
        v if unpad_kv else self._pad_seq(v, self.kv_padded_len),
        mk,
        k_mean,
        use_fixed_m=use_fixed_m,
        uniform_fixed_m=uniform_fixed_m,
        bkv_compute=bkv_compute,
    )
    jax.block_until_ready(candidate)

    self.assertEqual(reference.shape, candidate.shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(candidate))), "unpadded run produced non-finite values")
    self.assertTrue(
        bool(jnp.array_equal(reference, candidate)),
        "unpadded inputs changed the result -- the kernel is reading past the end "
        f"of the array (max abs delta {float(jnp.max(jnp.abs(reference - candidate)))})",
    )
    return reference, candidate

  # ------------------------------------------------------------ C3a: K and V

  def test_kv_unpadded_fixed_m(self):
    """C3a gate. Padded K/V rows are sliced away, so removing them must be a no-op."""
    self._compare(unpad_q=False, unpad_kv=True)

  def test_kv_unpadded_hybrid(self):
    """C3a on the hybrid kernel, whose tail goes through `_last_online`."""
    self._compare(unpad_q=False, unpad_kv=True, uniform_fixed_m=False)

  def test_kv_unpadded_online(self):
    """C3a on the pure online path -- the simplest probe of the DMA question."""
    self._compare(unpad_q=False, unpad_kv=True, use_fixed_m=False)

  def test_kv_unpadded_multi_iteration_tail(self):
    """C3a with bkv_compute < bkv, exercising the fori_loop + ragged remainder."""
    self._compare(unpad_q=False, unpad_kv=True, bkv_compute=self.bq // 2)

  def test_kv_unpadded_online_multi_iteration_tail(self):
    """Same, on the online path.

    The online path tracks a *running* max instead of a pinned one, so
    `_last_online` is genuinely different code from `_last_fixed` even though
    both slice the tail the same way. It gets its own ragged-remainder case
    rather than inheriting confidence from the fixed-m result.
    """
    self._compare(unpad_q=False, unpad_kv=True, use_fixed_m=False, bkv_compute=self.bq // 2)

  # ---------------------------------------------------------------- C3b: Q

  def test_q_unpadded_fixed_m(self):
    """C3b gate. Relies on the norm-vector padding in `_metadata`."""
    self._compare(unpad_q=True, unpad_kv=False)

  def test_q_unpadded_hybrid(self):
    self._compare(unpad_q=True, unpad_kv=False, uniform_fixed_m=False)

  def test_q_unpadded_online(self):
    self._compare(unpad_q=True, unpad_kv=False, use_fixed_m=False)

  # ------------------------------------------------------------ both halves

  def test_all_unpadded_fixed_m(self):
    self._compare(unpad_q=True, unpad_kv=True)

  def test_all_unpadded_hybrid(self):
    self._compare(unpad_q=True, unpad_kv=True, uniform_fixed_m=False)

  # ------------------------------------------------------- adversarial cases

  def test_all_unpadded_is_stable_across_repeats(self):
    """A single cold run can pass by luck on zeroed memory; N runs cannot.

    Between repeats the allocator pool is re-poisoned with NaN. If the kernel
    were reading past the end of the array, the value it reads would change
    from run to run and at least one repeat would diverge.
    """
    first = None
    for i in range(4):
      _dirty_device_memory(num_buffers=4)
      _, candidate = self._compare(unpad_q=True, unpad_kv=True, dirty=False)
      if first is None:
        first = candidate
      else:
        self.assertTrue(
            bool(jnp.array_equal(first, candidate)),
            f"unpadded run {i} differs from run 0 -- result depends on memory "
            "outside the array, which is the signature of an unclipped DMA",
        )

  def test_all_unpadded_survives_interleaved_kernel_invocation(self):
    """Runs a differently-shaped kernel first so VMEM holds real, non-zero data.

    `_dirty_device_memory` poisons HBM; this poisons the VMEM scratch buffers
    that a ragged block DMA would only partially overwrite. The interleaved call
    uses a large-magnitude value tensor so any leakage is numerically obvious
    rather than lost in rounding.
    """
    q_in, k_in, v, _, _ = self._inputs()
    mk, k_mean = self._metadata(q_in, k_in)
    loud = (jnp.ones_like(v) * 64).astype(v.dtype)
    jax.block_until_ready(
        self._run(
            self._pad_seq(q_in, self.q_padded_len),
            self._pad_seq(k_in, self.kv_padded_len),
            self._pad_seq(loud, self.kv_padded_len),
            mk,
            k_mean,
            use_fixed_m=True,
            uniform_fixed_m=True,
        )
    )
    self._compare(unpad_q=True, unpad_kv=True, dirty=False)


if __name__ == "__main__":
  unittest.main()
