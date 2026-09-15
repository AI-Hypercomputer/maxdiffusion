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

Comprehensive CPU and numerical reference tests for Sparse VideoGen (SVG).
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.transformers import svg_attention
from maxdiffusion.kernels import custom_svg_static_range_attention as static_kernel


def _numpy_upstream_get_attention_mask(mask_name: str, sample_mse_max_row: int, num_frame: int, frame_size: int):
  """Exact upstream mask reference from Sparse VideoGen."""
  seq_len = num_frame * frame_size
  block_size = 128
  block_thres = frame_size * 2
  num_block = math.ceil(seq_len / block_size)
  pixel_attn_mask = np.zeros((seq_len, seq_len), dtype=bool)
  pixel_attn_mask[:, :frame_size] = 1

  for i in range(num_block):
    for j in range(num_block):
      if abs(i - j) < block_thres // block_size:
        pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

  if mask_name == "spatial":
    attention_mask = pixel_attn_mask
  else:
    pixel_attn_mask = (
        pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
        .transpose(1, 0, 3, 2)
        .reshape(frame_size * num_frame, frame_size * num_frame)
    )
    attention_mask = pixel_attn_mask

  return attention_mask[:sample_mse_max_row]


def _require_tpu(test):
  """These tests call Pallas directly; CPU only supports interpret mode."""
  if jax.devices()[0].platform != "tpu":
    test.skipTest("Requires TPU: Pallas runs only in interpret mode on CPU")


class SVGAttentionUnitTest(unittest.TestCase):

  def test_01_sample_row_domain(self):
    F, H, W = 21, 45, 80
    L = F * H * W
    key = jax.random.PRNGKey(42)

    sampled_rows = jax.random.randint(key, (64,), minval=0, maxval=min(10000, L), dtype=jnp.int32)
    self.assertEqual(sampled_rows.shape, (64,))
    self.assertTrue(jnp.all(sampled_rows >= 0))
    self.assertTrue(jnp.all(sampled_rows < 10000))
    self.assertTrue(jnp.all(sampled_rows < L))

    L_small = 1280
    sampled_small = jax.random.randint(key, (64,), minval=0, maxval=min(10000, L_small), dtype=jnp.int32)
    self.assertTrue(jnp.all(sampled_small >= 0))
    self.assertTrue(jnp.all(sampled_small < 1280))

  def test_02_fixed_profiler_spatial_mask(self):
    F, H, W = 5, 16, 16
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    sampled_rows = jnp.array([0, 10, 127, 128, 255, 256, 500, 1000, 1279], dtype=jnp.int32)

    jax_spatial, _ = svg_attention.svg_probe_masks(sampled_rows, token_grid, block_size=128)
    np_spatial = _numpy_upstream_get_attention_mask("spatial", L, F, P)[np.asarray(sampled_rows), :]
    np.testing.assert_array_equal(np.asarray(jax_spatial), np_spatial)

  def test_03_fixed_profiler_temporal_mask(self):
    F, H, W = 5, 16, 16
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    sampled_rows = jnp.array([0, 10, 127, 128, 255, 256, 500, 1000, 1279], dtype=jnp.int32)

    _, jax_temporal = svg_attention.svg_probe_masks(sampled_rows, token_grid, block_size=128)
    np_temporal = _numpy_upstream_get_attention_mask("temporal", L, F, P)[np.asarray(sampled_rows), :]
    np.testing.assert_array_equal(np.asarray(jax_temporal), np_temporal)

  def test_04_first_frame_sink_semantics(self):
    F, H, W = 5, 16, 16
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    sampled_rows = jnp.array([600, 1000], dtype=jnp.int32)

    jax_spatial, jax_temporal = svg_attention.svg_probe_masks(sampled_rows, token_grid, block_size=128)
    self.assertTrue(np.all(np.asarray(jax_spatial[:, :P])))
    np_temporal = _numpy_upstream_get_attention_mask("temporal", L, F, P)[np.asarray(sampled_rows), :]
    np.testing.assert_array_equal(np.asarray(jax_temporal), np_temporal)

  def test_05_band_width_calculation(self):
    F, H, W = 21, 45, 80
    L = F * H * W
    token_grid = (F, H, W)

    w_25 = svg_attention.svg_execution_band_width(token_grid, 0.25)
    self.assertEqual(w_25, 10240)
    self.assertEqual(w_25 % 128, 0)
    self.assertEqual(svg_attention.svg_execution_band_width(token_grid, 1.0), L - 1)

  def test_06_density_boundary_conditions(self):
    token_grid = (5, 4, 4)
    self.assertEqual(svg_attention.svg_execution_band_width(token_grid, 1.0), 79)
    self.assertGreater(svg_attention.svg_execution_band_width(token_grid, 0.5), 0)
    with self.assertRaises(ValueError):
      svg_attention.svg_execution_band_width(token_grid, 0.0)
    with self.assertRaises(ValueError):
      svg_attention.svg_execution_band_width(token_grid, -0.1)
    with self.assertRaises(ValueError):
      svg_attention.svg_execution_band_width(token_grid, 1.05)

  def test_07_placement_permutation_and_inverse(self):
    F, H, W = 4, 3, 2
    L = F * H * W
    token_grid = (F, H, W)
    B, num_heads, D = 2, 4, 8

    rng = jax.random.PRNGKey(123)
    x = jax.random.normal(rng, (B, num_heads, L, D))

    is_temporal_spatial = jnp.zeros((B, num_heads), dtype=bool)
    placed_sp = svg_attention.svg_placement_permute(x, x, x, is_temporal_spatial, token_grid)[0]
    restored_sp = svg_attention.svg_placement_unpermute(placed_sp, is_temporal_spatial, token_grid)
    np.testing.assert_allclose(np.asarray(restored_sp), np.asarray(x), atol=1e-6)
    np.testing.assert_allclose(np.asarray(placed_sp), np.asarray(x), atol=1e-6)

    is_temporal_all = jnp.ones((B, num_heads), dtype=bool)
    placed_tp = svg_attention.svg_placement_permute(x, x, x, is_temporal_all, token_grid)[0]
    restored_tp = svg_attention.svg_placement_unpermute(placed_tp, is_temporal_all, token_grid)
    np.testing.assert_allclose(np.asarray(restored_tp), np.asarray(x), atol=1e-6)

  def test_08_mixed_per_head_permutation(self):
    F, H, W = 3, 2, 2
    L = F * H * W
    token_grid = (F, H, W)
    B, num_heads, D = 1, 4, 8

    q = jax.random.normal(jax.random.PRNGKey(1), (B, num_heads, L, D))
    k = jax.random.normal(jax.random.PRNGKey(2), (B, num_heads, L, D))
    v = jax.random.normal(jax.random.PRNGKey(3), (B, num_heads, L, D))
    is_temporal = jnp.array([[False, True, True, False]])

    q_placed, _, _ = svg_attention.svg_placement_permute(q, k, v, is_temporal, token_grid)
    forward_idx, _ = svg_attention.svg_token_major_indices(token_grid)
    np.testing.assert_allclose(np.asarray(q_placed[:, 0]), np.asarray(q[:, 0]))
    np.testing.assert_allclose(np.asarray(q_placed[:, 3]), np.asarray(q[:, 3]))
    np.testing.assert_allclose(np.asarray(q_placed[:, 1]), np.asarray(jnp.take(q[:, 1], forward_idx, axis=1)))
    np.testing.assert_allclose(np.asarray(q_placed[:, 2]), np.asarray(jnp.take(q[:, 2], forward_idx, axis=1)))
    q_restored = svg_attention.svg_placement_unpermute(q_placed, is_temporal, token_grid)
    np.testing.assert_allclose(np.asarray(q_restored), np.asarray(q), atol=1e-6)

  def test_09_routing_argmin_oracle(self):
    F, H, W = 3, 2, 2
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    B, num_heads, D = 1, 2, 16

    key = jax.random.PRNGKey(10)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, num_heads, L, D))
    k = jax.random.normal(k2, (B, num_heads, L, D))
    v = jax.random.normal(k3, (B, num_heads, L, D))

    scale = 1.0 / math.sqrt(D)
    jax_is_temporal = svg_attention.svg_profile_temporal_heads(
        q, k, v, token_grid, query_count=12, profile_key=k4, scale=scale, sample_max_row=12
    )

    sampled_rows = np.arange(12, dtype=np.int32)
    sampled_q = np.asarray(q)[:, :, sampled_rows, :]
    key_np = np.asarray(k)
    val_np = np.asarray(v)
    scores = np.einsum("bhqd,bhkd->bhqk", sampled_q, key_np) * scale
    dense_w = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    dense_w /= np.sum(dense_w, axis=-1, keepdims=True)
    golden = np.einsum("bhqk,bhkd->bhqd", dense_w, val_np)

    sp_mask = _numpy_upstream_get_attention_mask("spatial", 12, F, P)
    tp_mask = _numpy_upstream_get_attention_mask("temporal", 12, F, P)

    def mask_out(mask):
      masked_scores = np.where(mask[None, None, :, :], scores, -1e9)
      weights = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
      weights /= np.sum(weights, axis=-1, keepdims=True)
      return np.einsum("bhqk,bhkd->bhqd", weights, val_np)

    sp_out = mask_out(sp_mask)
    tp_out = mask_out(tp_mask)
    sp_err = np.mean((sp_out - golden) ** 2, axis=(-2, -1))
    tp_err = np.mean((tp_out - golden) ** 2, axis=(-2, -1))
    np.testing.assert_array_equal(np.asarray(jax_is_temporal), tp_err < sp_err)

  def test_09b_base2_execution_scale_is_softmax_equivalent(self):
    logits = jax.random.normal(jax.random.PRNGKey(91), (4, 17), dtype=jnp.float32)
    natural = jax.nn.softmax(logits, axis=-1)
    log2e = math.log2(math.e)
    exp2_weights = jnp.exp2(logits * log2e - jnp.max(logits * log2e, axis=-1, keepdims=True))
    exp2_weights /= jnp.sum(exp2_weights, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(natural), np.asarray(exp2_weights), rtol=2e-6, atol=2e-6)

  def test_inactive_static_step_with_traced_layer(self):
    for step in (0, 40):

      def active(layer):
        result = svg_attention.is_svg_active(
            step_index=step, layer_index=layer, start_step=11, end_step=40, start_layer=1, end_layer=40
        )
        self.assertIs(result, False)
        return result

      self.assertFalse(bool(jax.jit(active)(jnp.asarray(2))))

    active = jax.jit(
        lambda step, layer: svg_attention.is_svg_active(
            step_index=step, layer_index=layer, start_step=11, end_step=40, start_layer=1, end_layer=40
        )
    )
    self.assertTrue(bool(active(jnp.asarray(11), jnp.asarray(1))))
    self.assertFalse(bool(active(jnp.asarray(10), jnp.asarray(1))))

  def test_10_step_scheduling_exact_counts(self):
    num_inference_steps = 40
    active_start_step = 12
    active_end_step = 40
    step_is_sparse = [(active_start_step <= s < active_end_step) for s in range(num_inference_steps)]
    self.assertEqual(sum(not s for s in step_is_sparse), 12)
    self.assertEqual(sum(step_is_sparse), 28)
    self.assertFalse(step_is_sparse[11])
    self.assertTrue(step_is_sparse[12])

  def test_11_layer_scheduling_exact_counts(self):
    num_layers = 40
    active_start_layer = 1
    active_end_layer = 40
    layer_is_sparse = [(active_start_layer <= l < active_end_layer) for l in range(num_layers)]
    self.assertEqual(sum(not l for l in layer_is_sparse), 1)
    self.assertEqual(sum(layer_is_sparse), 39)
    self.assertFalse(layer_is_sparse[0])
    self.assertTrue(layer_is_sparse[39])

  def test_12_static_tile_classifier_properties(self):
    q_seq_len = 1000
    kv_seq_len = 1000
    bq = 128
    bkv = 128
    band_width = 256
    frame_size = 128
    include_first_frame = True

    full_map, full_act, boundary_map, bnd_act = static_kernel._classify_tiles(
        q_seq_len, kv_seq_len, bq, bkv, band_width, frame_size, include_first_frame
    )
    q_tiles = math.ceil(q_seq_len / bq)
    kv_tiles = math.ceil(kv_seq_len / bkv)
    for qi in range(q_tiles):
      full_set = set(full_map[qi, : full_act[qi]])
      bnd_set = set(boundary_map[qi, : bnd_act[qi]])
      self.assertEqual(len(full_set.intersection(bnd_set)), 0)
      q0 = qi * bq
      for kj in full_set:
        k0 = kj * bkv
        self.assertTrue(q0 + bq <= q_seq_len)
        self.assertTrue(k0 + bkv <= kv_seq_len)
        sink_full = include_first_frame and (k0 + bkv - 1 < frame_size)
        local_full = max(abs(q0 - (k0 + bkv - 1)), abs((q0 + bq - 1) - k0)) <= band_width
        self.assertTrue(sink_full or local_full)
      q1 = min(q_seq_len, q0 + bq) - 1
      band0 = max(0, q0 - band_width)
      band1 = min(kv_seq_len - 1, q1 + band_width)
      expected_live = set(range(band0 // bkv, min(kv_tiles, (band1 // bkv) + 1)))
      if include_first_frame:
        expected_live.update(range(0, min(kv_tiles, ((frame_size - 1) // bkv) + 1)))
      self.assertEqual(full_set.union(bnd_set), expected_live)

  def test_13_single_execution_matches_dual_kernel_reference(self):
    F, H, W = 3, 2, 2
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    B, num_heads, D = 1, 4, 8

    key = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, num_heads, L, D))
    k = jax.random.normal(k2, (B, num_heads, L, D))
    v = jax.random.normal(k3, (B, num_heads, L, D))
    scale = 1.0 / math.sqrt(D)
    is_temporal = jnp.array([[False, True, True, False]])
    band_width = 3

    common_mask = (jnp.arange(L)[None, :] < P) | (jnp.abs(jnp.arange(L)[:, None] - jnp.arange(L)[None, :]) <= band_width)
    q_placed, k_placed, v_placed = svg_attention.svg_placement_permute(q, k, v, is_temporal, token_grid)
    scores_placed = jnp.einsum("bhqd,bhkd->bhqk", q_placed, k_placed) * scale
    logits_placed = jnp.where(common_mask[None, None, :, :], scores_placed, -1e9)
    weights_placed = jax.nn.softmax(logits_placed, axis=-1)
    out_placed = jnp.einsum("bhqk,bhkd->bhqd", weights_placed, v_placed)
    out_single_exec = svg_attention.svg_placement_unpermute(out_placed, is_temporal, token_grid)

    scores_sp = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    logits_sp = jnp.where(common_mask[None, None, :, :], scores_sp, -1e9)
    weights_sp = jax.nn.softmax(logits_sp, axis=-1)
    out_spatial = jnp.einsum("bhqk,bhkd->bhqd", weights_sp, v)
    tm_idx = (jnp.arange(L) % P) * F + (jnp.arange(L) // P)
    temporal_mask = (tm_idx[None, :] < P) | (jnp.abs(tm_idx[:, None] - tm_idx[None, :]) <= band_width)
    logits_tp = jnp.where(temporal_mask[None, None, :, :], scores_sp, -1e9)
    weights_tp = jax.nn.softmax(logits_tp, axis=-1)
    out_temporal = jnp.einsum("bhqk,bhkd->bhqd", weights_tp, v)
    out_bruteforce = jnp.where(is_temporal[:, :, None, None], out_temporal, out_spatial)
    np.testing.assert_allclose(np.asarray(out_single_exec), np.asarray(out_bruteforce), rtol=1e-5, atol=1e-5)

  def test_14_exact_sparse_placed_matches_dense_masked_reference(self):
    _require_tpu(self)
    F, H, W = 5, 16, 16
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    B, num_heads, D = 1, 4, 128

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, num_heads, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (B, num_heads, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, num_heads, L, D), dtype=jnp.bfloat16)
    scale = 1.0 / math.sqrt(D)
    test_cases = [
        ("all_spatial", jnp.zeros((B, num_heads), dtype=bool)),
        ("all_temporal", jnp.ones((B, num_heads), dtype=bool)),
        ("mixed", jnp.array([[False, True, True, False]])),
    ]
    band_width = 256
    block_sizes = static_kernel.SVGBlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    scores_fp32 = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    sp_mask = (jnp.arange(L)[None, :] < P) | (jnp.abs(jnp.arange(L)[:, None] - jnp.arange(L)[None, :]) <= band_width)
    tm_idx = (jnp.arange(L) % P) * F + (jnp.arange(L) // P)
    tp_mask = (tm_idx[None, :] < P) | (jnp.abs(tm_idx[:, None] - tm_idx[None, :]) <= band_width)

    for case_name, is_temporal in test_cases:
      with self.subTest(case=case_name):
        head_masks = jnp.where(is_temporal[:, :, None, None], tp_mask[None, None, :, :], sp_mask[None, None, :, :])
        logits = jnp.where(head_masks, scores_fp32, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        out_ref = jnp.einsum("bhqk,bhkd->bhqd", weights, v.astype(jnp.float32)).astype(jnp.bfloat16)

        q_placed, k_placed, v_placed = svg_attention.svg_placement_permute(q, k, v, is_temporal, token_grid)
        q_local = q_placed * math.log2(math.e)
        k_scaled = k_placed * scale
        svg_kernel = static_kernel.make_svg_exact_static_range_mha(
            block_sizes=block_sizes,
            orig_q_seq_len=L,
            orig_kv_seq_len=L,
            band_width=band_width,
            frame_size=P,
            include_first_frame=True,
            use_base2_exp=True,
        )
        vmapped_svg = jax.vmap(svg_kernel, in_axes=(0, 0, 0))
        out_placed = vmapped_svg(q_local, k_scaled, v_placed)
        out_placed = jnp.swapaxes(out_placed, 2, 3)
        out_actual = svg_attention.svg_placement_unpermute(out_placed, is_temporal, token_grid)
        rel_l2 = jnp.linalg.norm(out_actual.astype(jnp.float32) - out_ref.astype(jnp.float32)) / jnp.linalg.norm(
            out_ref.astype(jnp.float32)
        )
        max_abs = jnp.max(jnp.abs(out_actual.astype(jnp.float32) - out_ref.astype(jnp.float32)))
        self.assertLess(float(rel_l2), 0.01, f"{case_name} rel_l2 {float(rel_l2):.6f} exceeds tolerance")
        self.assertLess(float(max_abs), 0.05, f"{case_name} max_abs {float(max_abs):.6f} exceeds tolerance")

  def test_15_density_one_matches_dense_reference(self):
    _require_tpu(self)
    F, H, W = 4, 16, 16
    P = H * W
    L = F * P
    token_grid = (F, H, W)
    B, num_heads, D = 1, 2, 128

    key = jax.random.PRNGKey(777)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, num_heads, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (B, num_heads, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, num_heads, L, D), dtype=jnp.bfloat16)
    scale = 1.0 / math.sqrt(D)
    is_temporal = jnp.array([[False, True]])
    band_width = svg_attention.svg_execution_band_width(token_grid, 1.0)
    self.assertEqual(band_width, L - 1)

    scores_fp32 = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    weights = jax.nn.softmax(scores_fp32, axis=-1)
    out_ref = jnp.einsum("bhqk,bhkd->bhqd", weights, v.astype(jnp.float32)).astype(jnp.bfloat16)

    block_sizes = static_kernel.SVGBlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    q_placed, k_placed, v_placed = svg_attention.svg_placement_permute(q, k, v, is_temporal, token_grid)
    q_local = q_placed * math.log2(math.e)
    k_scaled = k_placed * scale
    svg_kernel = static_kernel.make_svg_exact_static_range_mha(
        block_sizes=block_sizes,
        orig_q_seq_len=L,
        orig_kv_seq_len=L,
        band_width=band_width,
        frame_size=P,
        include_first_frame=True,
        use_base2_exp=True,
    )
    vmapped_svg = jax.vmap(svg_kernel, in_axes=(0, 0, 0))
    out_placed = vmapped_svg(q_local, k_scaled, v_placed)
    out_placed = jnp.swapaxes(out_placed, 2, 3)
    out_actual = svg_attention.svg_placement_unpermute(out_placed, is_temporal, token_grid)
    rel_l2 = jnp.linalg.norm(out_actual.astype(jnp.float32) - out_ref.astype(jnp.float32)) / jnp.linalg.norm(
        out_ref.astype(jnp.float32)
    )
    self.assertLess(float(rel_l2), 0.01)

  def test_svg_cache_check_without_loaded_low_noise_transformer(self):
    from unittest.mock import Mock
    from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2

    pipeline = WanPipeline2_2.__new__(WanPipeline2_2)
    pipeline.use_svg_attention = False
    pipeline.low_noise_transformer = None
    pipeline._prepare_model_inputs = Mock(side_effect=RuntimeError("reached input preparation"))
    with self.assertRaisesRegex(RuntimeError, "reached input preparation"):
      pipeline(prompt="test")
    pipeline._prepare_model_inputs.assert_called_once()

  def test_svg_cache_incompatibility_fail_closed(self):
    from types import SimpleNamespace
    from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import run_inference_2_1
    from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2

    for cache_flag in ["use_cfg_cache", "use_magcache"]:
      cfg = SimpleNamespace(use_svg_attention=True)
      with self.assertRaises(ValueError):
        run_inference_2_1(
            graphdef=None,
            sharded_state=None,
            rest_of_state=None,
            latents=jnp.zeros((1, 16, 21, 45, 80), dtype=jnp.bfloat16),
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=5.0,
            num_inference_steps=40,
            scheduler=None,
            scheduler_state=None,
            config=cfg,
            use_cfg_cache=(cache_flag == "use_cfg_cache"),
            use_magcache=(cache_flag == "use_magcache"),
        )

    pipeline_22 = WanPipeline2_2.__new__(WanPipeline2_2)
    pipeline_22.use_svg_attention = True
    pipeline_22.low_noise_transformer = type("T", (), {"config": type("C", (), {"use_svg_attention": True})()})()
    for cache_flag in ["use_cfg_cache", "use_magcache"]:
      with self.assertRaises(ValueError):
        pipeline_22(
            prompt="test prompt",
            use_cfg_cache=(cache_flag == "use_cfg_cache"),
            use_magcache=(cache_flag == "use_magcache"),
        )


if __name__ == "__main__":
  unittest.main()
