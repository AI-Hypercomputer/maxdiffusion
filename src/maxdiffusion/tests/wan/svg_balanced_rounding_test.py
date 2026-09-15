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

Tests for SVG balanced boundary rounding.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.kernels import custom_svg_attention_dispatch as dispatch
from maxdiffusion.kernels import custom_svg_balanced_rounding_attention as balanced
from maxdiffusion.kernels import custom_svg_static_range_attention as static_range
from maxdiffusion.kernels import custom_svg_balanced_rounding_partial as padding


def _bs():
  return static_range.SVGBlockSizes(
      block_q=3328,
      block_kv=2816,
      block_kv_compute=256,
      block_kv_compute_in=256,
  )


def _band_width(density: float, n: int = 75600):
  if density >= 1.0:
    return n - 1
  return min(
      n - 1,
      int(math.ceil((n * (1.0 - math.sqrt(1.0 - density))) / 128.0)) * 128,
  )


def _budget(density: float, n: int = 75600, frame_size: int = 3600):
  _, _, bm, _, stats = balanced.build_boundary_stats(
      orig_q_seq_len=n,
      orig_kv_seq_len=n,
      block_sizes=_bs(),
      band_width=_band_width(density, n),
      frame_size=frame_size,
      include_first_frame=True,
  )
  _, active, report = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="global_balanced",
      budget_scale=1.0,
  )
  return active, report


def test_global_balanced_matches_exact_pair_budget_720p_operating_points():
  for density in (0.65, 0.50, 0.35, 0.20, 0.15):
    _, report = _budget(density)
    assert abs(report["budget_error_fraction"]) <= 0.02, (density, report)
    assert report["boundary_tiles_selected"] > 0
    assert report["rounded_boundary_pairs"] > 0
    assert report["exact_boundary_pairs"] > 0


def test_round_up_and_down_bracket_exact_boundary_budget():
  _, _, bm, _, stats = balanced.build_boundary_stats(
      orig_q_seq_len=75600,
      orig_kv_seq_len=75600,
      block_sizes=_bs(),
      band_width=22144,
      frame_size=3600,
      include_first_frame=True,
  )
  _, _, up = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="up",
  )
  _, _, down = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="down",
  )
  assert up["rounded_boundary_pairs"] >= up["exact_boundary_pairs"]
  assert down["rounded_boundary_pairs"] == 0


def test_selected_table_is_deterministic():
  _, _, bm, _, stats = balanced.build_boundary_stats(
      orig_q_seq_len=75600,
      orig_kv_seq_len=75600,
      block_sizes=_bs(),
      band_width=22144,
      frame_size=3600,
      include_first_frame=True,
  )
  a_table, a_active, a_report = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="global_balanced",
      budget_scale=1.0,
  )
  b_table, b_active, b_report = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="global_balanced",
      budget_scale=1.0,
  )
  np.testing.assert_array_equal(a_table, b_table)
  np.testing.assert_array_equal(a_active, b_active)
  assert a_report == b_report


def test_union_tail_preserves_physical_tiles_and_isolates_sequence_tail():
  n = 75600
  bs = _bs()
  fm, fa, bm, _, stats = balanced.build_boundary_stats(
      orig_q_seq_len=n,
      orig_kv_seq_len=n,
      block_sizes=bs,
      band_width=_band_width(0.15, n),
      frame_size=3600,
      include_first_frame=True,
  )
  sm, sa, _ = balanced.build_selected_boundary_table(
      stats=stats,
      qtiles=bm.shape[0],
      policy="global_balanced",
      budget_scale=1.0,
  )
  mm, ma, tm, ta = balanced.build_union_tail_tables(
      full_table=fm,
      full_active=fa,
      selected_boundary_table=sm,
      selected_boundary_active=sa,
      orig_q_seq_len=n,
      orig_kv_seq_len=n,
      block_sizes=bs,
  )

  original = {
      (qi, int(kj))
      for qi in range(fm.shape[0])
      for table, active in ((fm, fa), (sm, sa))
      for kj in table[qi, : int(active[qi])]
  }
  main = {(qi, int(kj)) for qi in range(mm.shape[0]) for kj in mm[qi, : int(ma[qi])]}
  tail = {(qi, int(kj)) for qi in range(tm.shape[0]) for kj in tm[qi, : int(ta[qi])]}

  assert main.isdisjoint(tail)
  assert main | tail == original
  assert len(main) == 115
  assert len(tail) == 6
  assert all((qi + 1) * bs.block_q > n or (kj + 1) * bs.block_kv > n for qi, kj in tail)
  assert all((qi + 1) * bs.block_q <= n and (kj + 1) * bs.block_kv <= n for qi, kj in main)


def test_production_builder_is_fixed_global_balanced():
  kernel = dispatch.make_svg_static_range_mha(
      block_sizes=_bs(),
      orig_q_seq_len=75600,
      orig_kv_seq_len=75600,
      band_width=14720,
      frame_size=3600,
      include_first_frame=True,
  )
  assert kernel.rounding_policy == "global_balanced"
  assert kernel.rounding_budget_scale == 1.0
  assert abs(kernel.rounding_budget["budget_error_fraction"]) < 0.02
  assert kernel.union_main_tiles > 0
  assert kernel.tail_cleanup_tiles > 0
  assert kernel.union_main_tiles + kernel.tail_cleanup_tiles == (kernel.full_tiles + kernel.selected_boundary_tiles)


def test_exact_reference_builder_remains_available():
  kernel = static_range.make_svg_exact_static_range_mha(
      block_sizes=_bs(),
      orig_q_seq_len=75600,
      orig_kv_seq_len=75600,
      band_width=14720,
      frame_size=3600,
      include_first_frame=True,
  )
  assert kernel.full_tiles > 0
  assert kernel.boundary_tiles > 0


@pytest.mark.parametrize("n", [384, 401], ids=["aligned", "padded"])
@pytest.mark.parametrize("dense_support", [False, True], ids=["sparse", "density_one"])
@pytest.mark.parametrize("use_base2_exp", [False, True], ids=["exp", "exp2"])
def test_production_kernel_matches_rounded_support(n, dense_support, use_base2_exp):
  """Check compiled main/tail execution against its real-token support."""
  if jax.default_backend() != "tpu":
    pytest.skip("Requires TPU compilation and execution")
  block = 128
  padded = math.ceil(n / block) * block
  band = n - 1 if dense_support else block
  bs = static_range.SVGBlockSizes(block_q=block, block_kv=block, block_kv_compute=block, block_kv_compute_in=block)
  support_args = {
      "block_sizes": bs,
      "orig_q_seq_len": n,
      "orig_kv_seq_len": n,
      "band_width": band,
      "frame_size": 128,
      "include_first_frame": True,
  }
  full, full_active, boundary, _, stats = balanced.build_boundary_stats(**support_args)
  selected, selected_active, _ = balanced.build_selected_boundary_table(
      stats=stats, qtiles=boundary.shape[0], policy="global_balanced", budget_scale=1.0
  )
  mask = np.zeros((n, n), dtype=bool)
  for table, active in ((full, full_active), (selected, selected_active)):
    for qi in range(table.shape[0]):
      for ki in table[qi, : int(active[qi])]:
        mask[qi * block : (qi + 1) * block, int(ki) * block : (int(ki) + 1) * block] = True
  assert mask.any(axis=1).all()
  assert mask.all() if dense_support else not mask.all()

  rng = np.random.default_rng(41)
  q, k, v = (jnp.asarray(rng.normal(size=(1, n, 128)), dtype=jnp.bfloat16) for _ in range(3))
  # Match the BF16 input scaling used by dispatch before forming the FP32 oracle.
  k = k * (128**-0.5)
  q = q * math.log2(math.e) if use_base2_exp else q
  logits = jnp.einsum("hqd,hkd->hqk", q.astype(jnp.float32), k.astype(jnp.float32))
  if use_base2_exp:
    logits = logits * math.log(2)
  weights = jax.nn.softmax(jnp.where(jnp.asarray(mask)[None], logits, -jnp.inf), axis=-1)
  expected = jnp.einsum("hqk,hkd->hqd", weights, v.astype(jnp.float32))
  # Large finite padding makes accidental inclusion visible without NaN propagation.
  inputs = [jnp.pad(x, ((0, 0), (0, padded - n), (0, 0)), constant_values=64) for x in (q, k, v)]
  kernel = dispatch.make_svg_static_range_mha(**support_args, use_base2_exp=use_base2_exp)
  assert kernel.union_main_tiles > 0
  assert (kernel.tail_cleanup_tiles > 0) == (n != padded)
  actual = jnp.swapaxes(jax.jit(kernel)(*inputs), 1, 2)[:, :n, :].astype(jnp.float32)
  actual, expected = np.asarray(actual), np.asarray(expected)
  assert np.isfinite(actual).all()
  relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
  assert relative_error < 0.01, relative_error
  np.testing.assert_allclose(actual, expected, rtol=0, atol=0.05)


def test_padding_partial_rejects_unaligned_value_dimension():
  block = 128
  bs = static_range.SVGBlockSizes(block_q=block, block_kv=block, block_kv_compute=block, block_kv_compute_in=block)
  kernel = padding.make_padding_partial_from_table(
      table_np=np.zeros((1, 1), dtype=np.int32),
      active_np=np.ones((1,), dtype=np.int32),
      block_sizes=bs,
      orig_q_seq_len=127,
      orig_kv_seq_len=127,
      band_width=128,
      frame_size=128,
  )
  q = jnp.zeros((1, block, 128), dtype=jnp.bfloat16)
  v = jnp.zeros((1, block, padding.NUM_SUBLANES + 1), dtype=jnp.bfloat16)
  with pytest.raises(NotImplementedError, match="must be divisible"):
    kernel(q, q, v)
