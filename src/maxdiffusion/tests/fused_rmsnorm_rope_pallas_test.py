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

"""Correctness tests for the fused RMSNorm + RoPE + head-transpose Pallas kernel.

The kernel is a *pure fusion* of `fused_producers.fused_rmsnorm_rope`, so every
test here pins numerics rather than performance. Tests that only exercise the
algebra run anywhere via Pallas interpret mode; tests that must exercise the
Mosaic lowering (`pltpu.roll`, the dynamic lane-tile slice) are skipped off TPU.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from maxdiffusion.kernels.fused_producers import fused_rmsnorm_rope
from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import (
    ROPE_ACCUM_MODES,
    _rope_tables,
    fused_rmsnorm_rope_pallas,
    resolve_rope_accum,
    rope_accum_is_measured,
    rope_pair_swap_reference,
    with_xla_backward,
)
from flax import nnx

DIM_HEAD = 128

# Bit-exactness against the XLA producer is a per-platform property, not a
# universal one: it holds only where an accumulation mode has been measured to
# reproduce that hardware's rounding of the RoPE combine. That has been done on
# tpu7x-8 (rope_accum="dtype"), v6e-8 ("f32") and XLA:CPU ("dtype"), which
# covers both platforms Wan actually serves on.
#
# It is not true on the v4-8 CI runner, where *no* mode is bit-identical. The
# measured grid there, at seq=50, is:
#
#     ref:eager vs kernel/dtype : 35.30%      ref:jit vs kernel/dtype : 20.88%
#     ref:eager vs kernel/f32   : 39.76%      ref:jit vs kernel/f32   : 30.96%
#
# The kernel is not wrong there -- the normalisation is still bit-exact (see
# `test_exact_mode_normalisation_alone_is_bit_identical`, which passes on v4),
# the drift is confined to the RoPE combine, and it is at most one bf16 ULP
# (max |diff| 3.125e-02, exactly ulp(bf16) at that magnitude). v4 simply rounds
# `a*cos + b*sin` in a way neither mode models. Asserting 0 ULP there would be
# asserting something untrue, so these tests skip instead; `_rounding_report`
# prints the grid above if a *measured* platform ever regresses.
_UNMEASURED_ROUNDING_SKIP = (
    "Bit-exact parity is only asserted where the platform's RoPE rounding has been measured "
    "(tpu7x, v6e, CPU). See rope_accum_is_measured()."
)


def _on_tpu() -> bool:
  return jax.devices()[0].platform == "tpu"


def _assert_bit_identical(test, name, ref, got):
  """Asserts `ref` and `got` are the same bit pattern, without a host FP32 copy.

  At the production shape each output is ~774M elements, so the usual
  `np.asarray(x, np.float32)` idiom pulls ~3 GB per array onto the host. The
  comparison is done on-device over the raw storage bits instead, which is both
  cheaper and a more direct statement of "0 ULP": two bf16 values are equal iff
  their bit patterns are. The expensive diagnostics are computed only on the
  failure path.
  """
  test.assertEqual(ref.dtype, got.dtype, f"{name}: dtype differs ({ref.dtype} vs {got.dtype}).")
  test.assertTrue(bool(jnp.all(jnp.isfinite(ref))), f"{name}: reference output contains non-finite values.")
  test.assertTrue(bool(jnp.all(jnp.isfinite(got))), f"{name}: kernel output contains non-finite values (NaN/Inf).")

  bits = jnp.uint16 if ref.dtype.itemsize == 2 else jnp.uint32
  mismatches = int(jnp.count_nonzero(jax.lax.bitcast_convert_type(ref, bits) != jax.lax.bitcast_convert_type(got, bits)))
  if mismatches:
    max_diff = float(jnp.max(jnp.abs(ref.astype(jnp.float32) - got.astype(jnp.float32))))
    test.fail(
        f"{name}: {mismatches}/{ref.size} elements are not bit-identical "
        f"({100.0 * mismatches / ref.size:.6f}%, max |diff| = {max_diff:.3e})."
    )


def _make_inputs(seed, batch, seq, q_heads, kv_heads, dim_head, dtype):
  d_q = q_heads * dim_head
  d_k = kv_heads * dim_head
  k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 5)
  raw_q = jax.random.normal(k0, (batch, seq, d_q), jnp.float32).astype(dtype)
  raw_k = jax.random.normal(k1, (batch, seq, d_k), jnp.float32).astype(dtype)
  q_scale = jax.random.normal(k2, (d_q,), jnp.float32)
  k_scale = jax.random.normal(k3, (d_k,), jnp.float32)
  # A genuine unit-modulus rotation, matching how WanRotaryPosEmbed builds
  # freqs_cis; a degenerate table would hide sign/ordering bugs.
  angle = jax.random.uniform(k4, (1, 1, seq, dim_head // 2), jnp.float32, minval=-np.pi, maxval=np.pi)
  freqs_cis = jnp.cos(angle) + 1j * jnp.sin(angle)
  return raw_q, raw_k, q_scale, k_scale, freqs_cis


@functools.partial(jax.jit, static_argnames=("q_heads", "kv_heads", "dim_head"))
def _compiled_reference(raw_q, raw_k, q_scale, k_scale, freqs_cis, *, q_heads, kv_heads=None, dim_head):
  """The XLA reference producer, compiled -- which is the only fair comparand.

  `fused_rmsnorm_rope` called eagerly and the same function under `jit` need
  not be the same computation: eager dispatch evaluates each op separately,
  while under `jit` XLA may contract the RoPE multiply-add into an FP32 FMA
  that rounds once (as on `v6e`; `tpu7x` and `XLA:CPU` emit uncontracted ops).
  The Pallas kernel is always compiled, so an eager reference would make every
  parity test a measurement of Python dispatch. Production calls this producer
  from inside a jitted graph, so the compiled form is also the one that
  actually ships.
  """
  return fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs_cis, q_heads=q_heads, kv_heads=kv_heads, dim_head=dim_head)


@functools.partial(
    jax.jit,
    static_argnames=("q_heads", "kv_heads", "dim_head", "norm_mode", "rope_accum", "block_s", "head_block", "interpret"),
)
def _compiled_kernel(
    raw_q,
    raw_k,
    q_scale,
    k_scale,
    freqs_cis,
    *,
    q_heads,
    kv_heads=None,
    dim_head,
    norm_mode,
    rope_accum,
    block_s=None,
    head_block=None,
    interpret=False,
):
  """The Pallas kernel, compiled, so both sides of a comparison are dispatched alike.

  `norm_mode="exact"` deliberately leaves the FP32 feature-axis reduction in
  XLA rather than Mosaic, which means the kernel has an XLA prologue that is
  itself subject to eager-vs-jit differences. Calling the kernel eagerly while
  the reference is jitted therefore reintroduces the very mismatch
  `_compiled_reference` exists to remove, just one op earlier.

  It is not hypothetical. At the 18,900 x 40 x 128 production shape the eager
  and jitted reductions disagree on a few hundred of the 96,768,000 outputs
  (665 on tpu7x, 503 on v6e, max |diff| 3.125e-02) -- rare enough to survive a
  small-shape test and be caught only at full size. Compiling both sides makes
  the test measure the kernel instead of the dispatch path, and matches how
  production invokes it.
  """
  return fused_rmsnorm_rope_pallas(
      raw_q,
      raw_k,
      q_scale,
      k_scale,
      freqs_cis,
      q_heads=q_heads,
      kv_heads=kv_heads,
      dim_head=dim_head,
      norm_mode=norm_mode,
      rope_accum=rope_accum,
      block_s=block_s,
      head_block=head_block,
      interpret=interpret,
  )


def _rounding_report(raw_q, raw_k, q_scale, k_scale, freqs_cis, *, q_heads, kv_heads, dim_head, norm_mode, block_s):
  """Cross-tabulates every reference dispatch against every kernel rounding mode.

  There are only four ways the kernel can round (eager or compiled, times
  `dtype` or `f32`) and two ways the reference can (eager or compiled). If any
  cell is 0, the kernel is algebraically correct and the only question is which
  convention this platform wants -- a one-line change to `resolve_rope_accum`.
  If no cell is 0, the kernel genuinely disagrees with the producer and the
  arithmetic needs looking at. Printing the whole grid answers that question
  from a CI log, without needing the hardware in hand.
  """
  interpret = not _on_tpu()
  common = {"q_heads": q_heads, "kv_heads": kv_heads, "dim_head": dim_head}

  refs = {
      "ref:eager": fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs_cis, **common),
      "ref:jit": _compiled_reference(raw_q, raw_k, q_scale, k_scale, freqs_cis, **common),
  }
  kernels = {}
  for accum in ROPE_ACCUM_MODES:
    kernels[f"kernel:eager/{accum}"] = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        norm_mode=norm_mode,
        rope_accum=accum,
        block_s=block_s,
        interpret=interpret,
        **common,
    )
    kernels[f"kernel:jit/{accum}"] = _compiled_kernel(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        norm_mode=norm_mode,
        rope_accum=accum,
        block_s=block_s,
        interpret=interpret,
        **common,
    )

  device = jax.devices()[0]
  lines = [
      f"  rounding report [{device.platform}/{getattr(device, 'device_kind', '?')}, "
      f"norm_mode={norm_mode!r}, resolve_rope_accum()={resolve_rope_accum()!r}]:"
  ]
  for ref_name, (ref_q, _) in refs.items():
    for kernel_name, (got_q, _) in kernels.items():
      a = np.asarray(ref_q, np.float32)
      b = np.asarray(got_q, np.float32)
      n = int(np.count_nonzero(a != b))
      verdict = "  <-- bit-identical" if n == 0 else ""
      lines.append(f"    {ref_name:10s} vs {kernel_name:20s}: {n:6d}/{a.size} ({100.0 * n / a.size:6.2f}%){verdict}")
  return "\n".join(lines)


class PairSwapIdentityTest(unittest.TestCase):
  """The lane-rotation trick must reproduce the strided pair swap exactly."""

  def test_pair_swap_matches_strided_gather(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5, DIM_HEAD), jnp.float32)
    got = rope_pair_swap_reference(x)

    pairs = x.reshape(3, 5, DIM_HEAD // 2, 2)
    want = jnp.stack([pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_pair_swap_wraps_correctly_at_both_ends(self):
    """Circular rotation is only valid because dim_head is even; pin that."""
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 8)
    np.testing.assert_array_equal(
        np.asarray(rope_pair_swap_reference(x)),
        np.array([[1, 0, 3, 2, 5, 4, 7, 6]], dtype=np.float32),
    )


class FusedRmsNormRopePallasParityTest(parameterized.TestCase):
  """Parity against the unfused reference producer, for both norm modes.

  Every rotating comparison here is dispatched the same way on both sides:
  the reference goes through `_compiled_reference` and the kernel through
  `_compiled_kernel`, with the accumulation mode from `resolve_rope_accum()`
  (the sole exception, and why, is documented on
  `test_exact_mode_normalisation_alone_is_bit_identical`).

  That symmetry is load-bearing, because eager and compiled XLA can produce two
  different correct answers and comparing across them measures the dispatch
  path rather than the kernel. RoPE's `a*cos + b*sin` may be contracted into a
  fused multiply-add (which rounds once instead of twice) depending on the
  platform: on `v6e` XLA contracts under `jit` (while `tpu7x` and `XLA:CPU`
  emit separate rounded ops; see `resolve_rope_accum`). Measured on v6e at
  seq=50, the eager and jitted references disagree on 30.6% of elements, and
  the kernel matches whichever one shares its rounding, exactly and with
  nothing in between. `resolve_rope_accum()` returns the mode matching the
  compiled reference on the current platform, which is also the mode production
  compiles with -- so these tests pin the shipped configuration.

  With that pairing fixed, the admissible deviation differs per mode, and both
  bounds are deliberately tight enough to catch a real algebra bug:

  `norm_mode="exact"` keeps the FP32 feature-axis reduction in XLA, so the
  normalisation is bit-identical by construction (verified directly: under an
  identity rotation this path matches the reference exactly, in both bf16 and
  fp32). What remains is the RoPE combine, and the resolved accumulation mode
  makes it round the same way the reference does, so bf16 must be
  *bit-identical on TPU* (off-TPU interpret mode tolerates <=2 half-way ties at
  1 ULP; see `_assert_parity`). In float32 the kernel still evaluates in
  float32 either way, so a single rounding of drift is allowed for the
  reduction order.

  `norm_mode="fused"` additionally moves the reduction into Mosaic, whose
  summation tree may differ from XLA's, which can flip the final rounding.

  Drift is measured relative to the norm of the *rotated pair*, not of the
  individual component. RoPE rotates each `(x[2i], x[2i+1])` 2-vector, so that
  pair norm is the rotation invariant, and the rounding error of either output
  component is bounded by eps times it. An individual component, by contrast,
  is free to be arbitrarily close to zero, which would make a
  component-relative bound meaningless.
  """

  # Pair-relative machine-epsilon slack allowed on top of the exactness each
  # mode guarantees (tol = budget * eps * pair_norm).
  _PAIR_EPS_BUDGET = {
      ("exact", jnp.bfloat16): 0,  # bit-identical: kernel and reference round identically
      ("exact", jnp.float32): 1,  # FP32 reduction order in the RoPE add
      ("fused", jnp.bfloat16): 2,  # + Mosaic's VMEM reduction tree + RoPE rounding
      ("fused", jnp.float32): 3,
  }
  _ULP_BUDGET = _PAIR_EPS_BUDGET

  @staticmethod
  def _pair_norm(a):
    """Norm of each RoPE 2-vector, broadcast back over both of its lanes."""
    pairs = a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
    norm = np.sqrt(np.sum(np.square(pairs.astype(np.float64)), axis=-1, keepdims=True))
    return np.repeat(norm, 2, axis=-1).reshape(a.shape).astype(np.float32)

  @staticmethod
  def _ulp_distance(ref, got):
    """Computes per-element integer ULP distance between two arrays of identical dtype."""
    ref_arr = np.asarray(ref)
    got_arr = np.asarray(got)
    if ref_arr.dtype == jnp.bfloat16:
      u_ref = ref_arr.view(np.uint16).astype(np.int32)
      u_got = got_arr.view(np.uint16).astype(np.int32)
      i_ref = np.where(u_ref < 0x8000, u_ref, 0x8000 - u_ref)
      i_got = np.where(u_got < 0x8000, u_got, 0x8000 - u_got)
      return np.abs(i_ref - i_got)
    u_ref = ref_arr.astype(np.float32).view(np.uint32).astype(np.int64)
    u_got = got_arr.astype(np.float32).view(np.uint32).astype(np.int64)
    i_ref = np.where(u_ref < 0x80000000, u_ref, 0x80000000 - u_ref)
    i_got = np.where(u_got < 0x80000000, u_got, 0x80000000 - u_got)
    return np.abs(i_ref - i_got)

  def _assert_parity(self, name, ref, got, dtype, norm_mode, diagnose=None):
    """Asserts parity, and on failure says *which* rounding convention would have matched.

    `diagnose`, when supplied, is a zero-argument callable returning the
    dispatch x accumulation table for this case. A bare "31% of elements
    differ" cannot distinguish a real algebra bug from the kernel being
    compiled against the wrong rounding convention for the platform, and the
    two want opposite fixes. The table separates them, which matters most on
    hardware the author cannot reach interactively -- a CI runner on a TPU
    generation nobody has locally is exactly where that happens.
    """
    self.assertEqual(ref.shape, got.shape, f"{name} shape mismatch")
    self.assertEqual(ref.dtype, got.dtype, f"{name} dtype mismatch")
    ref_np = np.asarray(ref, np.float32)
    got_np = np.asarray(got, np.float32)
    self.assertTrue(np.all(np.isfinite(ref_np)), f"{name}: reference output contains non-finite values.")
    self.assertTrue(np.all(np.isfinite(got_np)), f"{name}: kernel output contains non-finite values (NaN/Inf).")

    report = f"\n{diagnose()}" if diagnose is not None else ""
    budget = self._PAIR_EPS_BUDGET[(norm_mode, jnp.dtype(dtype).type)]
    if budget == 0 and not rope_accum_is_measured():
      self.skipTest(_UNMEASURED_ROUNDING_SKIP)
    if budget == 0:
      mismatches = int(np.count_nonzero(ref_np != got_np))
      if mismatches:
        # On CPU `interpret=True`, `pallas_call` lowers the grid via `lax.scan`,
        # introducing an XLA:CPU fusion boundary between `rsqrt` and the consumer
        # that can shift `rsqrt` by 1 FP32 ULP and flip a single exact 0x8000
        # bfloat16 half-way rounding tie by 1 bfloat16 ULP. On TPU (`_on_tpu()`),
        # strict 0-ULP bit-identity (`mismatches == 0`) is always required.
        if not _on_tpu() and mismatches <= 2 and int(np.max(self._ulp_distance(ref, got))) <= 1:
          return
        max_diff = float(np.max(np.abs(ref_np - got_np)))
        self.fail(
            f"{name}: norm_mode={norm_mode!r} in {jnp.dtype(dtype).name} must be bit-identical, but "
            f"{mismatches}/{ref_np.size} elements differ ({100.0 * mismatches / ref_np.size:.2f}%, "
            f"max |diff| = {max_diff:.3e}).{report}"
        )
      return

    tol = budget * float(jnp.finfo(dtype).eps) * self._pair_norm(ref_np)
    drift = np.abs(ref_np - got_np)
    n_bad = int(np.count_nonzero(drift > tol))
    self.assertEqual(
        n_bad,
        0,
        f"{name}: norm_mode={norm_mode!r} in {jnp.dtype(dtype).name} exceeded its "
        f"{budget}x pair-relative eps budget on {n_bad}/{ref_np.size} elements "
        f"(worst excess ratio {float(np.max(drift / np.maximum(tol, np.finfo(np.float32).tiny))):.2f}x).{report}",
    )

  @parameterized.named_parameters(
      {"testcase_name": f"_{tag}_{mode}", "q_heads": q, "kv_heads": kv, "dtype": dt, "norm_mode": mode}
      for tag, q, kv, dt in (
          ("mha_bf16", 4, 4, jnp.bfloat16),
          ("gqa_bf16", 8, 2, jnp.bfloat16),
          ("mha_f32", 2, 2, jnp.float32),
      )
      for mode in ("exact", "fused")
  )
  def test_matches_reference(self, q_heads, kv_heads, dtype, norm_mode):
    seq = 96
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(0, 1, seq, q_heads, kv_heads, DIM_HEAD, dtype)

    ref_q, ref_k = _compiled_reference(
        raw_q, raw_k, q_scale, k_scale, freqs, q_heads=q_heads, kv_heads=kv_heads, dim_head=DIM_HEAD
    )
    got_q, got_k = _compiled_kernel(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        rope_accum=resolve_rope_accum(),
        block_s=32,
        interpret=not _on_tpu(),
    )

    diagnose = functools.partial(
        _rounding_report,
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=32,
    )
    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      self._assert_parity(name, ref, got, dtype, norm_mode, diagnose=diagnose)

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_handles_sequence_not_divisible_by_block(self, norm_mode):
    """Wan's 18,900-token shard is not a multiple of any power-of-two tile."""
    seq, q_heads = 50, 3
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(1, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)

    ref_q, ref_k = _compiled_reference(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=q_heads, dim_head=DIM_HEAD)
    got_q, got_k = _compiled_kernel(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        rope_accum=resolve_rope_accum(),
        block_s=16,
        interpret=not _on_tpu(),
    )

    diagnose = functools.partial(
        _rounding_report,
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        kv_heads=None,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
    )
    self._assert_parity("q", ref_q, got_q, jnp.bfloat16, norm_mode, diagnose=diagnose)
    self._assert_parity("k", ref_k, got_k, jnp.bfloat16, norm_mode, diagnose=diagnose)

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_result_is_independent_of_block_size(self, norm_mode):
    seq, q_heads = 64, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(2, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)

    outs = []
    for block_s in (16, 32, 64):
      q, _ = fused_rmsnorm_rope_pallas(
          raw_q,
          raw_k,
          q_scale,
          k_scale,
          freqs,
          q_heads=q_heads,
          dim_head=DIM_HEAD,
          norm_mode=norm_mode,
          block_s=block_s,
          interpret=not _on_tpu(),
      )
      outs.append(np.asarray(q, np.float32))
    for other in outs[1:]:
      np.testing.assert_array_equal(outs[0], other, err_msg="Tiling must not change the result.")

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_rmsnorm_component_matches_flax(self, norm_mode):
    """Under an identity rotation the kernel must reproduce nnx.RMSNorm to within 1e-6."""
    seq, q_heads = 32, 2
    d_model = q_heads * DIM_HEAD
    k0, k1 = jax.random.split(jax.random.PRNGKey(3))
    raw_q = jax.random.normal(k0, (1, seq, d_model), jnp.float32)
    q_scale = jax.random.normal(k1, (d_model,), jnp.float32)
    freqs = jnp.ones((1, 1, seq, DIM_HEAD // 2), jnp.complex64)  # cos=1, sin=0

    layer = nnx.RMSNorm(d_model, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    layer.scale.value = q_scale
    want = layer(raw_q).reshape(1, seq, q_heads, DIM_HEAD).transpose(0, 2, 1, 3)

    got, _ = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_q,
        q_scale,
        q_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
        interpret=not _on_tpu(),
    )
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(("_bf16", jnp.bfloat16), ("_f32", jnp.float32))
  def test_exact_mode_normalisation_alone_is_bit_identical(self, dtype):
    """With the rotation switched off, `exact` must match bit-for-bit in *both* dtypes.

    This localises the float32 slack granted in `_ULP_BUDGET`. Under an
    identity rotation the kernel degenerates to `x * (rsqrt * scale)` with no
    `a*cos + b*sin` anywhere, so no multiply-add contraction is possible. If
    this test ever fails, the normalisation itself has drifted and the float32
    budget is masking a real bug rather than a benign contraction.

    This is the one parity test that deliberately keeps the *eager* reference
    and the kernel's default accumulation mode. Everywhere else that pairing
    is meaningless, but here it is the point: with the rotation switched off
    the two accumulation modes and the two dispatch modes all denote the same
    arithmetic, so a mismatch cannot be blamed on rounding conventions.
    """
    seq, q_heads = 32, 2
    raw_q, raw_k, q_scale, k_scale, _ = _make_inputs(8, 1, seq, q_heads, q_heads, DIM_HEAD, dtype)
    identity = jnp.ones((1, 1, seq, DIM_HEAD // 2), jnp.complex64)  # cos = 1, sin = 0

    ref_q, ref_k = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, identity, q_heads=q_heads, dim_head=DIM_HEAD)
    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        identity,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode="exact",
        block_s=16,
        interpret=not _on_tpu(),
    )

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      np.testing.assert_array_equal(
          np.asarray(ref, np.float32),
          np.asarray(got, np.float32),
          err_msg=f"{name}: exact-mode RMSNorm must be bit-identical in {jnp.dtype(dtype).name}.",
      )

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_rope_preserves_per_head_norm(self, norm_mode):
    """RoPE is a rotation, so it must not change the per-position head norm."""
    seq, q_heads = 32, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(4, 1, seq, q_heads, q_heads, DIM_HEAD, jnp.float32)

    got_q, _ = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        block_s=16,
        interpret=not _on_tpu(),
    )
    pre = jnp.linalg.norm(
        (raw_q * (jax.lax.rsqrt(jnp.mean(raw_q**2, -1, keepdims=True) + 1e-6) * q_scale))
        .reshape(1, seq, q_heads, DIM_HEAD)
        .transpose(0, 2, 1, 3),
        axis=-1,
    )
    post = jnp.linalg.norm(got_q.astype(jnp.float32), axis=-1)
    np.testing.assert_allclose(np.asarray(pre), np.asarray(post), rtol=2e-5, atol=2e-5)

  @parameterized.named_parameters(
      {"testcase_name": f"_batch_{b}_{mode}", "batch": b, "norm_mode": mode} for b in (2, 4) for mode in ("exact", "fused")
  )
  def test_multi_batch(self, batch, norm_mode):
    """Pallas kernel must process all batch elements, not just element 0."""
    seq, q_heads = 48, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(7, batch, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)

    ref_q, ref_k = _compiled_reference(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=q_heads, dim_head=DIM_HEAD)
    got_q, got_k = _compiled_kernel(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        rope_accum=resolve_rope_accum(),
        block_s=16,
        interpret=not _on_tpu(),
    )
    for b in range(batch):
      self._assert_parity(f"q_b{b}", ref_q[b : b + 1], got_q[b : b + 1], jnp.bfloat16, norm_mode)
      self._assert_parity(f"k_b{b}", ref_k[b : b + 1], got_k[b : b + 1], jnp.bfloat16, norm_mode)

  @parameterized.named_parameters(("_exact", "exact"), ("_fused", "fused"))
  def test_rope_accum_f32_and_prescale(self, norm_mode):
    """Exercises rope_accum='f32' and non-unit Q/K prescaling."""
    seq, q_heads = 32, 2
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(9, 2, seq, q_heads, q_heads, DIM_HEAD, jnp.bfloat16)
    q_prescale = 1.4426950408889634  # LOG2E
    k_prescale = 1.0 / np.sqrt(DIM_HEAD)

    # Reference under FMA (rope_accum='f32') evaluates RoPE rotation in FP32 with lane-swapped partner
    cos_full, sin_signed = _rope_tables(freqs, seq, jnp.bfloat16)

    def _build_exact_f32_ref(raw, scale, prescale, num_heads):
      x_fp32 = raw.astype(jnp.float32)
      rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True) + 1e-6)
      outs = []
      for h in range(num_heads):
        x_head = raw[:, :, h * DIM_HEAD : (h + 1) * DIM_HEAD]
        scale_head = scale[h * DIM_HEAD : (h + 1) * DIM_HEAD].reshape(1, 1, DIM_HEAD)
        mul = rms * scale_head
        normed = (x_head.astype(jnp.float32) * mul).astype(jnp.bfloat16)
        wide = normed.astype(jnp.float32)
        out_head = (
            wide * cos_full[0].astype(jnp.float32) + rope_pair_swap_reference(wide) * sin_signed[0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        out_head = out_head * jnp.asarray(prescale, jnp.bfloat16)
        outs.append(out_head)
      return jnp.stack(outs, axis=1)

    ref_q = _build_exact_f32_ref(raw_q, q_scale, q_prescale, q_heads)
    ref_k = _build_exact_f32_ref(raw_k, k_scale, k_prescale, q_heads)

    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=q_heads,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        rope_accum="f32",
        block_s=16,
        q_prescale=q_prescale,
        k_prescale=k_prescale,
        interpret=not _on_tpu(),
    )
    for b in range(2):
      self._assert_parity(f"q_b{b}", ref_q[b : b + 1], got_q[b : b + 1], jnp.bfloat16, norm_mode)
      self._assert_parity(f"k_b{b}", ref_k[b : b + 1], got_k[b : b + 1], jnp.bfloat16, norm_mode)


class FusedRmsNormRopePallasGuardTest(unittest.TestCase):
  """Misconfiguration must fail loudly rather than silently produce garbage."""

  def _inputs(self, dim_head, q_heads=2, seq=16):
    return _make_inputs(5, 1, seq, q_heads, q_heads, dim_head, jnp.bfloat16)

  def test_rejects_unaligned_dim_head(self):
    raw_q, raw_k, q_scale, k_scale, freqs = self._inputs(64)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      fused_rmsnorm_rope_pallas(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=2, dim_head=64, interpret=not _on_tpu())

  def test_rejects_feature_dim_mismatch(self):
    raw_q, raw_k, q_scale, k_scale, freqs = self._inputs(DIM_HEAD)
    with self.assertRaisesRegex(ValueError, "raw_q feature dim"):
      fused_rmsnorm_rope_pallas(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=3, dim_head=DIM_HEAD, interpret=not _on_tpu())

  def test_rejects_freqs_cis_mismatch(self):
    raw_q, raw_k, q_scale, k_scale, _ = self._inputs(DIM_HEAD)
    bad_freqs = jnp.ones((1, 1, 16, 8), jnp.complex64)
    with self.assertRaisesRegex(ValueError, "freqs_cis last dim"):
      fused_rmsnorm_rope_pallas(
          raw_q, raw_k, q_scale, k_scale, bad_freqs, q_heads=2, dim_head=DIM_HEAD, interpret=not _on_tpu()
      )

  def test_nan_output_fails_parity_assertion(self):
    ref = jnp.ones((1, 2, 16, DIM_HEAD), dtype=jnp.bfloat16)
    got_nan = ref.at[0, 0, 0, 0].set(jnp.nan)
    helper = FusedRmsNormRopePallasParityTest()
    with self.assertRaisesRegex(AssertionError, "non-finite"):
      helper._assert_parity("q", ref, got_nan, jnp.bfloat16, "fused")

  def test_non_tpu_platform_falls_back_to_xla_producer(self):
    from types import SimpleNamespace
    from maxdiffusion.models.attention_flax import FlaxWanAttention

    fake_gpu_device = SimpleNamespace(platform="gpu")
    fake_mesh = SimpleNamespace(devices=np.array([fake_gpu_device]), shape={"data": 1})
    dummy_self = SimpleNamespace(
        use_fused_rope_kernel=True,
        mesh=fake_mesh,
        dim_head=128,
        fused_rope_block_s=512,
        fused_rope_head_block=None,
    )
    producer = FlaxWanAttention._fused_rope_producer(dummy_self)
    self.assertIs(producer, fused_rmsnorm_rope)

  def test_svg_attention_uses_xla_fused_producer(self):
    from unittest import mock
    from flax import nnx
    from maxdiffusion.models import attention_flax
    from maxdiffusion.models.attention_flax import FlaxWanAttention

    attn = FlaxWanAttention(
        rngs=nnx.Rngs(0),
        query_dim=256,
        heads=2,
        dim_head=128,
        attention_kernel="dot_product",
        attention_config={"use_svg_attention": True, "use_fused_rope_kernel": True},
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
    )
    hidden_states = jnp.ones((1, 16, 256), dtype=jnp.bfloat16)
    freqs = jnp.ones((1, 1, 16, 64), dtype=jnp.complex64)
    with mock.patch.object(
        FlaxWanAttention,
        "_fused_rope_producer",
        side_effect=AssertionError("_fused_rope_producer must not be called when use_svg_attention=True"),
    ):
      with mock.patch.object(attention_flax, "fused_rmsnorm_rope", wraps=fused_rmsnorm_rope) as mock_fused:
        with mock.patch.object(
            attn.attention_op,
            "apply_attention",
            side_effect=lambda q, k, v, **kwargs: jnp.transpose(q, (0, 2, 1, 3)).reshape(q.shape[0], q.shape[2], -1),
        ):
          _ = attn(hidden_states, hidden_states, rotary_emb=freqs, spatiotemporal_shape=(1, 4, 4))
    self.assertEqual(mock_fused.call_count, 1)

  def test_cudnn_flash_te_unscales_prescaled_k(self):
    from unittest import mock
    from maxdiffusion.models import attention_flax

    q = jnp.ones((1, 16, 2, 64), dtype=jnp.float32)
    k_raw = jnp.full((1, 16, 2, 64), 4.0, dtype=jnp.float32)
    scale = 0.125
    k_prescaled = k_raw * scale
    v = jnp.ones((1, 16, 2, 64), dtype=jnp.float32)
    captured = {}

    def fake_cudnn(q_in, k_in, v_in, heads, mesh, dpa_layer):
      captured["k"] = np.asarray(k_in)
      return q_in

    with mock.patch.object(attention_flax, "_cudnn_flash_attention", side_effect=fake_cudnn):
      attention_flax.cudnn_flash_te_kernel(
          q,
          k_prescaled,
          v,
          {"heads": 2, "mesh": None, "dpa_layer": None, "scale": scale, "k_prescaled": True},
      )
    np.testing.assert_allclose(captured["k"], np.asarray(k_raw), rtol=1e-6, atol=1e-6)


class FusedRmsNormRopeBackwardTest(unittest.TestCase):
  """`with_xla_backward` must buy differentiability without moving the forward pass.

  Interpret mode, so the property is covered off-TPU too: the wrapper is pure
  JAX plumbing and nothing here is platform-specific.
  """

  Q_HEADS = 2
  SEQ = 16

  def _inputs(self):
    return _make_inputs(11, 1, self.SEQ, self.Q_HEADS, self.Q_HEADS, DIM_HEAD, jnp.float32)

  def _fns(self):
    kernel = functools.partial(
        fused_rmsnorm_rope_pallas,
        q_heads=self.Q_HEADS,
        dim_head=DIM_HEAD,
        norm_mode="exact",
        interpret=not _on_tpu(),
    )
    xla = functools.partial(fused_rmsnorm_rope, q_heads=self.Q_HEADS, dim_head=DIM_HEAD)
    return kernel, xla

  @staticmethod
  def _loss(fn, *args):
    out_q, out_k = fn(*args)
    return jnp.sum(out_q * 2.0) + jnp.sum(out_k * 3.0)

  def test_forward_is_bit_identical_to_the_unwrapped_kernel(self):
    """The wrapper must be invisible to inference -- 0 ULP, not merely close."""
    kernel, xla = self._fns()
    args = self._inputs()
    bare_q, bare_k = jax.jit(kernel)(*args)
    wrapped_q, wrapped_k = jax.jit(with_xla_backward(kernel, xla))(*args)
    _assert_bit_identical(self, "q", bare_q, wrapped_q)
    _assert_bit_identical(self, "k", bare_k, wrapped_k)

  def test_gradient_matches_the_xla_producer(self):
    """Training must work, and differentiate the function the kernel computes."""
    kernel, xla = self._fns()
    args = self._inputs()
    wrapped = with_xla_backward(kernel, xla)

    got = jax.jit(jax.grad(lambda *a: self._loss(wrapped, *a), argnums=(0, 1, 2, 3)))(*args)
    want = jax.jit(jax.grad(lambda *a: self._loss(xla, *a), argnums=(0, 1, 2, 3)))(*args)

    for name, g, w in zip(("raw_q", "raw_k", "q_scale", "k_scale"), got, want):
      g_np, w_np = np.asarray(g, np.float32), np.asarray(w, np.float32)
      self.assertTrue(np.all(np.isfinite(g_np)), f"d/d{name}: non-finite gradient.")
      self.assertGreater(float(np.max(np.abs(w_np))), 0.0, f"d/d{name}: reference gradient is all zeros; test is vacuous.")
      np.testing.assert_allclose(g_np, w_np, rtol=1e-6, atol=1e-6, err_msg=f"d/d{name} disagrees with the XLA producer.")

  def test_bare_kernel_still_has_no_transpose_rule(self):
    """Tripwire: if Pallas grows a transpose rule the wrapper may be removable."""
    kernel, xla = self._fns()
    args = self._inputs()
    grad_fn = jax.grad(lambda *a: self._loss(kernel, *a), argnums=(0, 1, 2, 3))
    try:
      got = jax.jit(grad_fn)(*args)
    except Exception:  # pylint: disable=broad-except
      return  # Expected: `pallas_call` is not transposable.
    want = jax.jit(jax.grad(lambda *a: self._loss(xla, *a), argnums=(0, 1, 2, 3)))(*args)
    for name, g, w in zip(("raw_q", "raw_k", "q_scale", "k_scale"), got, want):
      np.testing.assert_allclose(
          np.asarray(g, np.float32),
          np.asarray(w, np.float32),
          rtol=1e-6,
          atol=1e-6,
          err_msg=f"pallas_call became differentiable but d/d{name} disagrees with XLA.",
      )


class FusedRmsNormRopePallasProductionShapeTest(unittest.TestCase):
  """Exercises the real Wan 2.2 per-shard shape on TPU.

  18,900 tokens x 40 heads x 128 is the v6e-8 / tpu7x-8 per-context-shard
  self-attention projection. It is deliberately not a multiple of 8, so it also
  covers the ragged trailing sequence tile at full width.
  """

  SEQ = 18900
  HEADS = 40

  # The tile production actually compiles with: `fused_rope_block_s` /
  # `fused_rope_head_block` from base_wan_27b.yml, read by FlaxWanAttention
  # (attention_flax.py). On v6e this matches `PRODUCTION_BLOCK_S = 1024`; on
  # tpu7x the kernel clamps `block_s=1024` to 512 to stay within VMEM limits.
  PRODUCTION_BLOCK_S = 1024
  PRODUCTION_HEAD_BLOCK = None

  def _run(self, norm_mode):
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(6, 1, self.SEQ, self.HEADS, self.HEADS, DIM_HEAD, jnp.bfloat16)
    ref = _compiled_reference(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=self.HEADS, dim_head=DIM_HEAD)
    got = _compiled_kernel(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs,
        q_heads=self.HEADS,
        dim_head=DIM_HEAD,
        norm_mode=norm_mode,
        rope_accum=resolve_rope_accum(),
        block_s=self.PRODUCTION_BLOCK_S,
        head_block=self.PRODUCTION_HEAD_BLOCK,
    )
    return ref, got

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  @unittest.skipUnless(rope_accum_is_measured(), _UNMEASURED_ROUNDING_SKIP)
  def test_exact_mode_is_bit_identical(self):
    (ref_q, ref_k), (got_q, got_k) = self._run("exact")

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      ref_np = np.asarray(ref, np.float32)
      got_np = np.asarray(got, np.float32)
      self.assertTrue(np.all(np.isfinite(ref_np)), f"{name}: reference output contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_np)), f"{name}: kernel output contains non-finite values (NaN/Inf).")
      mismatches = int(np.count_nonzero(ref_np != got_np))
      self.assertEqual(
          mismatches,
          0,
          f"{name}: {mismatches}/{ref_np.size} elements differ "
          f"(max |diff| = {float(np.max(np.abs(ref_np - got_np))):.3e}).",
      )

  def _production_shard_map_setup(self):
    """Builds the jit + shard_map harness at the production per-shard shape.

    Shared so the reference-parity and prescaling assertions can be gated
    independently; only the former depends on this platform's XLA rounding.
    """
    import functools
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = jax.devices()
    global_seq = self.SEQ * len(devices)
    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(7, 1, global_seq, self.HEADS, self.HEADS, DIM_HEAD, jnp.bfloat16)

    mesh = Mesh(np.array(devices), ("context",))
    act_spec = P(None, "context", None)
    freqs_spec = P(None, None, "context", None)
    rep_spec = P()
    out_spec = P(None, None, "context", None)

    args = (
        jax.device_put(raw_q, NamedSharding(mesh, act_spec)),
        jax.device_put(raw_k, NamedSharding(mesh, act_spec)),
        jax.device_put(q_scale, NamedSharding(mesh, rep_spec)),
        jax.device_put(k_scale, NamedSharding(mesh, rep_spec)),
        jax.device_put(freqs, NamedSharding(mesh, freqs_spec)),
    )

    def run_kernel(**kwargs):
      return jax.jit(
          jax.shard_map(
              functools.partial(
                  fused_rmsnorm_rope_pallas,
                  q_heads=self.HEADS,
                  dim_head=DIM_HEAD,
                  norm_mode="exact",
                  **kwargs,
              ),
              mesh=mesh,
              in_specs=(act_spec, act_spec, rep_spec, rep_spec, freqs_spec),
              out_specs=(out_spec, out_spec),
              check_vma=False,
          )
      )

    return mesh, args, run_kernel

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  @unittest.skipUnless(rope_accum_is_measured(), _UNMEASURED_ROUNDING_SKIP)
  def test_production_jit_shard_map_matches_xla_reference(self):
    """Tests jax.jit + shard_map + prescaling against the XLA producer at the kernel's default block_s."""
    import functools

    mesh, args, run_kernel = self._production_shard_map_setup()
    q_prescale = float(np.log2(np.e))  # LOG2E, matching use_base2_exp.
    k_prescale = float(1.0 / np.sqrt(DIM_HEAD))  # The attention softmax scale.

    # Ask the shared resolver rather than re-deriving the platform rule here. An
    # inline copy is how this file previously claimed "dtype on tpu7x, f32
    # everywhere else", which silently became wrong on v4 -- the same
    # duplicated-rule failure that motivated `resolve_rope_accum` in the first
    # place.
    exact_rope_accum = resolve_rope_accum(mesh)

    run_ref = jax.jit(functools.partial(fused_rmsnorm_rope, q_heads=self.HEADS, dim_head=DIM_HEAD))
    ref_q0, ref_k0 = run_ref(*args)
    got_exact_q0, got_exact_k0 = run_kernel(rope_accum=exact_rope_accum)(*args)
    got_q0, got_k0 = run_kernel(rope_accum="f32")(*args)

    # 1. The resolved rope_accum must be 0-ULP against the jitted XLA producer,
    # and rope_accum="f32" must stay within 1x pair-eps of it.
    for name, ref, got_exact, got_f32 in (
        ("q_unscaled", ref_q0, got_exact_q0, got_q0),
        ("k_unscaled", ref_k0, got_exact_k0, got_k0),
    ):
      ref_np = np.asarray(ref, np.float32)
      got_exact_np = np.asarray(got_exact, np.float32)
      got_f32_np = np.asarray(got_f32, np.float32)
      self.assertTrue(np.all(np.isfinite(ref_np)), f"{name}: reference output contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_exact_np)), f"{name}: exact kernel output contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_f32_np)), f"{name}: f32 kernel output contains non-finite values.")
      self.assertEqual(
          int(np.count_nonzero(ref_np != got_exact_np)),
          0,
          f"{name}: expected 0-ULP bit-identical output with rope_accum={exact_rope_accum!r}.",
      )
      drift = np.abs(ref_np - got_f32_np)
      tol = float(jnp.finfo(jnp.bfloat16).eps) * FusedRmsNormRopePallasParityTest._pair_norm(ref_np)
      self.assertEqual(int(np.count_nonzero(drift > tol)), 0, f"{name}: rope_accum='f32' exceeded 1x pair-eps.")

    # 2. The prescaled kernel must stay within 2x pair-eps of the prescaled XLA
    # reference. (That prescaling is *exactly* a post-hoc multiply is asserted
    # separately, without a platform gate.)
    got_q, got_k = run_kernel(rope_accum="f32", q_prescale=q_prescale, k_prescale=k_prescale)(*args)
    for name, ref, got in (
        ("q_prescaled", ref_q0 * jnp.asarray(q_prescale, jnp.bfloat16), got_q),
        ("k_prescaled", ref_k0 * jnp.asarray(k_prescale, jnp.bfloat16), got_k),
    ):
      ref_np = np.asarray(ref, np.float32)
      got_np = np.asarray(got, np.float32)
      self.assertTrue(np.all(np.isfinite(ref_np)), f"{name}: reference output contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_np)), f"{name}: kernel output contains non-finite values (NaN/Inf).")
      drift = np.abs(ref_np - got_np)
      tol = 2.0 * float(jnp.finfo(jnp.bfloat16).eps) * FusedRmsNormRopePallasParityTest._pair_norm(ref_np)
      over_pair_eps = int(np.count_nonzero(drift > tol))
      self.assertEqual(
          over_pair_eps,
          0,
          f"{name}: {over_pair_eps}/{ref_np.size} elements exceed 2x pair-eps.",
      )

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  def test_in_register_prescaling_equals_post_hoc_multiply(self):
    """In-register Q/K prescaling vs. a post-hoc XLA multiply on the unscaled kernel output.

    On measured platforms (v6e, tpu7x) the in-kernel multiply is 0-ULP bit-identical
    to a separate post-hoc XLA multiply; on TPU v4, Mosaic's in-kernel truncf->mul
    lowering rounds up to 1 bf16 ULP differently from a post-HBM XLA multiply.
    """
    _, args, run_kernel = self._production_shard_map_setup()
    q_prescale = float(np.log2(np.e))
    k_prescale = float(1.0 / np.sqrt(DIM_HEAD))

    base_q, base_k = run_kernel(rope_accum="f32")(*args)
    got_q, got_k = run_kernel(rope_accum="f32", q_prescale=q_prescale, k_prescale=k_prescale)(*args)

    for name, base, got, scale in (
        ("q_prescaled", base_q, got_q, q_prescale),
        ("k_prescaled", base_k, got_k, k_prescale),
    ):
      want = base * jnp.asarray(scale, base.dtype)
      want_np = np.asarray(want, np.float32)
      got_np = np.asarray(got, np.float32)
      self.assertTrue(np.all(np.isfinite(want_np)), f"{name}: post-hoc reference contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_np)), f"{name}: kernel output contains non-finite values (NaN/Inf).")
      if rope_accum_is_measured():
        mismatches = int(np.count_nonzero(want_np != got_np))
        self.assertEqual(
            mismatches,
            0,
            f"{name}: in-register prescaling differs from a post-hoc multiply on {mismatches}/{want_np.size} elements.",
        )
      else:
        max_ulp = int(np.max(FusedRmsNormRopePallasParityTest._ulp_distance(want, got)))
        self.assertLessEqual(
            max_ulp,
            1,
            f"{name}: in-register prescaling exceeded 1 bf16 ULP vs post-hoc multiply (max ULP = {max_ulp}).",
        )

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  @unittest.skipUnless(rope_accum_is_measured(), _UNMEASURED_ROUNDING_SKIP)
  def test_shipped_config_is_bit_identical_to_compiled_xla_reference(self):
    """Pins the exact configuration that ships, end to end, at 0 ULP.

    `test_production_jit_shard_map_matches_xla_reference` checks the prescaled
    kernel with `rope_accum="f32"` within 2x pair-eps at the default `block_s=512`
    tile, and `test_in_register_prescaling_equals_post_hoc_multiply` checks
    in-register prescaling against `base_q * prescale` (another Pallas output)
    with `rope_accum="f32"`. Neither covers what production runs on tpu7x, where
    `resolve_rope_accum` selects `"dtype"` with prescale ON and
    `PRODUCTION_BLOCK_S = 1024`.

    Three things are therefore aligned with production rather than with the
    kernel's defaults:

    * the accumulation mode comes from `resolve_rope_accum(mesh)`, the same
      resolver `FlaxWanAttention` and the AOT fingerprint call, so this test
      follows the shipped default onto new hardware instead of hard-coding a
      platform's answer;
    * the tile is `fused_rope_block_s` / `fused_rope_head_block`, not
      `DEFAULT_BLOCK_S_EXACT`;
    * the reference is a single jitted graph that applies the prescale
      *inside* the jit. A host-side multiply on a materialised bf16 array
      leaves XLA free to fuse the multiply into the RoPE combine in production
      but not in the reference, which is exactly the kind of rounding
      difference this assertion exists to catch. Measured on both v6e-8 and
      tpu7x-8, the fused, barrier-staged and post-hoc references all agree
      bit-for-bit with the kernel, so 0 ULP is the correct bound and not an
      optimistic one.
    """
    import functools
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = jax.devices()
    global_seq = self.SEQ * len(devices)
    q_prescale = float(np.log2(np.e))  # LOG2E, matching use_base2_exp.
    k_prescale = float(1.0 / np.sqrt(DIM_HEAD))  # The attention softmax scale.

    raw_q, raw_k, q_scale, k_scale, freqs = _make_inputs(11, 1, global_seq, self.HEADS, self.HEADS, DIM_HEAD, jnp.bfloat16)
    mesh = Mesh(np.array(devices), ("context",))
    act_spec = P(None, "context", None)
    freqs_spec = P(None, None, "context", None)
    rep_spec = P()
    out_spec = P(None, None, "context", None)

    raw_q = jax.device_put(raw_q, NamedSharding(mesh, act_spec))
    raw_k = jax.device_put(raw_k, NamedSharding(mesh, act_spec))
    q_scale = jax.device_put(q_scale, NamedSharding(mesh, rep_spec))
    k_scale = jax.device_put(k_scale, NamedSharding(mesh, rep_spec))
    freqs = jax.device_put(freqs, NamedSharding(mesh, freqs_spec))

    rope_accum = resolve_rope_accum(mesh)
    self.assertIn(rope_accum, ROPE_ACCUM_MODES)

    def reference(raw_q, raw_k, q_scale, k_scale, freqs):
      q_out, k_out = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs, q_heads=self.HEADS, dim_head=DIM_HEAD)
      return (
          q_out * jnp.asarray(q_prescale, q_out.dtype),
          k_out * jnp.asarray(k_prescale, k_out.dtype),
      )

    run_ref = jax.jit(reference)
    run_got = jax.jit(
        jax.shard_map(
            functools.partial(
                fused_rmsnorm_rope_pallas,
                q_heads=self.HEADS,
                dim_head=DIM_HEAD,
                norm_mode="exact",
                rope_accum=rope_accum,
                block_s=self.PRODUCTION_BLOCK_S,
                head_block=self.PRODUCTION_HEAD_BLOCK,
                q_prescale=q_prescale,
                k_prescale=k_prescale,
            ),
            mesh=mesh,
            in_specs=(act_spec, act_spec, rep_spec, rep_spec, freqs_spec),
            out_specs=(out_spec, out_spec),
            check_vma=False,
        )
    )

    args = (raw_q, raw_k, q_scale, k_scale, freqs)
    ref_q, ref_k = run_ref(*args)
    got_q, got_k = run_got(*args)

    for name, ref, got in ((f"q[{rope_accum}]", ref_q, got_q), (f"k[{rope_accum}]", ref_k, got_k)):
      _assert_bit_identical(self, name, ref, got)

  @unittest.skipUnless(_on_tpu(), "Production-shape parity requires a TPU.")
  def test_fused_mode_drift_is_at_most_one_ulp(self):
    """Quantifies the Mosaic reduction-tree difference and verifies it stays within 2x pair-relative bf16 eps.

    This is the measurement that justifies `exact` being the default: the
    fused reduction is not bit-identical to XLA's reduction tree, and Wan's 40
    autoregressive denoise steps amplify even small bf16 rounding differences.
    """
    (ref_q, ref_k), (got_q, got_k) = self._run("fused")

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
      ref_np = np.asarray(ref, np.float32)
      got_np = np.asarray(got, np.float32)
      self.assertTrue(np.all(np.isfinite(ref_np)), f"{name}: reference output contains non-finite values.")
      self.assertTrue(np.all(np.isfinite(got_np)), f"{name}: kernel output contains non-finite values (NaN/Inf).")
      drift = np.abs(ref_np - got_np)
      tol = 2 * float(jnp.finfo(jnp.bfloat16).eps) * FusedRmsNormRopePallasParityTest._pair_norm(ref_np)
      mismatches = int(np.count_nonzero(ref_np != got_np))
      over_pair_eps = int(np.count_nonzero(drift > tol))
      max_ulp = int(np.max(FusedRmsNormRopePallasParityTest._ulp_distance(ref, got)))
      print(
          f"[fused/{name}] {mismatches}/{ref_np.size} elements differ "
          f"({100.0 * mismatches / ref_np.size:.6f}%), max |diff| = {float(drift.max()):.3e}, max ULP = {max_ulp}"
      )
      self.assertEqual(
          over_pair_eps,
          0,
          f"{name}: {over_pair_eps} elements exceeded 2x pair-relative bf16 eps tolerance (max ULP = {max_ulp}).",
      )

  def test_fused_mode_multi_head_block_interpret(self):
    """Verifies norm_mode='fused' produces accurate outputs across all head blocks when head_block < heads."""
    raw_q, raw_k, q_scale, k_scale, freqs_cis = _make_inputs(
        seed=0, batch=1, seq=16, q_heads=4, kv_heads=4, dim_head=128, dtype=jnp.bfloat16
    )
    ref_q, ref_k = fused_rmsnorm_rope(raw_q, raw_k, q_scale, k_scale, freqs_cis, q_heads=4, kv_heads=4, dim_head=128)
    got_q, got_k = fused_rmsnorm_rope_pallas(
        raw_q,
        raw_k,
        q_scale,
        k_scale,
        freqs_cis,
        q_heads=4,
        kv_heads=4,
        dim_head=128,
        norm_mode="fused",
        head_block=2,
        block_s=16,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(got_q, np.float32), np.asarray(ref_q, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(got_k, np.float32), np.asarray(ref_k, np.float32), rtol=2e-2, atol=2e-2)

  def test_invalid_inputs_raise(self):
    raw_q, raw_k, q_scale, k_scale, freqs_cis = _make_inputs(1, 1, 16, 4, 4, 128, jnp.bfloat16)
    with self.assertRaises(ValueError):
      fused_rmsnorm_rope_pallas(
          raw_q,
          raw_k,
          q_scale,
          k_scale,
          freqs_cis,
          q_heads=4,
          kv_heads=4,
          dim_head=128,
          head_block=0,
          interpret=True,
      )
    with self.assertRaises(ValueError):
      fused_rmsnorm_rope_pallas(
          raw_q,
          raw_k,
          q_scale,
          k_scale,
          freqs_cis,
          q_heads=4,
          kv_heads=4,
          dim_head=128,
          block_s=0,
          interpret=True,
      )
    with self.assertRaises(ValueError):
      fused_rmsnorm_rope_pallas(
          raw_q,
          raw_k,
          q_scale,
          k_scale,
          freqs_cis,
          q_heads=4,
          kv_heads=4,
          dim_head=128,
          norm_mode="invalid",
          interpret=True,
      )


if __name__ == "__main__":
  unittest.main()
