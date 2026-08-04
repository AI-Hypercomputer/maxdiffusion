# Copyright 2026 Google LLC / The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive unit tests for the refactored attention strategies, dispatcher, utils, and backwards compatibility."""

import functools
import sys
import unittest
from unittest import mock

# Guard optional or TPU-only dependencies for CPU testing environments
for mod_name in [
    "aqt",
    "aqt.jax",
    "aqt.jax.v2",
    "aqt.jax.v2.flax",
    "aqt.jax.v2.config",
    "huggingface_hub",
    "huggingface_hub.constants",
]:
  if mod_name not in sys.modules:
    try:
      __import__(mod_name)
    except ImportError:
      sys.modules[mod_name] = mock.MagicMock()

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models import attention_utils
from maxdiffusion.models import attention_dispatch
from maxdiffusion.models import attention_strategies
from maxdiffusion.models import attention_flax
from maxdiffusion.models import unet_transformer_blocks_flax
from maxdiffusion.kernels.splash_attention import ring_attention_kernel


class AttentionStrategiesTest(unittest.TestCase):
  """Unit tests for attention strategies, registry, utils, and re-exports."""

  def test_kernel_registry_all_13_keys(self):
    """Verifies that KERNEL_REGISTRY contains all 13 standard kernel names."""
    expected_keys = {
        "dot_product",
        "ulysses",
        "ulysses_custom",
        "ulysses_custom_fixed_m",
        "ulysses_ring",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
        "ulysses_ring_custom_bidir",
        "flash",
        "tokamax_flash",
        "tokamax_ring",
        "tokamax_ring_custom",
        "cudnn_flash_te",
    }
    registered_keys = set(attention_dispatch.KERNEL_REGISTRY.keys())
    self.assertTrue(
        expected_keys.issubset(registered_keys),
        f"Missing registered keys: {expected_keys - registered_keys}",
    )

    # Test dynamic kernel registration
    @attention_dispatch.register_kernel("test_dummy_kernel")
    def dummy_kernel(q, k, v, context):
      return q

    self.assertIn("test_dummy_kernel", attention_dispatch.KERNEL_REGISTRY)
    self.assertEqual(attention_dispatch.KERNEL_REGISTRY["test_dummy_kernel"]("q", "k", "v", {}), "q")

  def test_backwards_compatibility_reexports(self):
    """Verifies that attention_flax and ring_attention_kernel re-export all expected symbols."""
    # From attention_utils
    self.assertIs(attention_flax._select_flash_block_sizes, attention_utils._select_flash_block_sizes)
    self.assertIs(attention_flax._coerce_tokamax_block_sizes, attention_utils._coerce_tokamax_block_sizes)
    self.assertIs(attention_flax._extract_custom_block_sizes, attention_utils._extract_custom_block_sizes)
    self.assertIs(attention_flax._pad_data_for_flash, attention_utils._pad_data_for_flash)
    self.assertIs(attention_flax.AttentionBlockSizes, attention_utils.AttentionBlockSizes)

    # From attention_dispatch
    self.assertIs(attention_flax.KERNEL_REGISTRY, attention_dispatch.KERNEL_REGISTRY)
    self.assertIs(attention_flax.register_kernel, attention_dispatch.register_kernel)
    self.assertIs(attention_flax._apply_attention, attention_dispatch._apply_attention)
    self.assertIs(attention_flax._apply_attention_dot, attention_dispatch._apply_attention_dot)
    self.assertIs(attention_flax._tpu_flash_attention, attention_dispatch._tpu_flash_attention)
    self.assertIs(attention_flax._ulysses_attention, attention_dispatch._ulysses_attention)
    self.assertIs(attention_flax._ulysses_ring_attention, attention_dispatch._ulysses_ring_attention)
    self.assertIs(attention_flax._ulysses_ring_custom_attention, attention_dispatch._ulysses_ring_custom_attention)

    # From unet_transformer_blocks_flax
    self.assertIs(attention_flax.FlaxAttention, unet_transformer_blocks_flax.FlaxAttention)
    self.assertIs(attention_flax.FlaxBasicTransformerBlock, unet_transformer_blocks_flax.FlaxBasicTransformerBlock)
    self.assertIs(attention_flax.FlaxTransformer2DModel, unet_transformer_blocks_flax.FlaxTransformer2DModel)
    self.assertIs(attention_flax.FlaxFeedForward, unet_transformer_blocks_flax.FlaxFeedForward)
    self.assertIs(attention_flax.FlaxGEGLU, unet_transformer_blocks_flax.FlaxGEGLU)

    # From custom_ring in ring_attention_kernel
    self.assertTrue(hasattr(ring_attention_kernel, "make_custom_ring_attention"))
    self.assertTrue(hasattr(ring_attention_kernel, "_custom_ring_attention_forward"))
    self.assertTrue(hasattr(ring_attention_kernel, "_custom_bidirectional_ring_forward"))

  def test_attention_block_sizes_adapter(self):
    """Tests the AttentionBlockSizes unified adapter methods."""
    bsizes = attention_utils.AttentionBlockSizes(
        block_q=2048,
        block_kv=1024,
        block_kv_compute=512,
    )
    self.assertEqual(bsizes.block_q, 2048)
    self.assertEqual(bsizes.block_kv, 1024)

    tokamax_bs = bsizes.to_tokamax()
    self.assertEqual(tokamax_bs.block_q, 2048)
    self.assertEqual(tokamax_bs.block_kv, 1024)

    custom_bs = bsizes.to_custom_splash()
    self.assertEqual(custom_bs.block_q, 2048)
    self.assertEqual(custom_bs.block_kv, 1024)

  def test_attention_utils_reshaping_and_padding(self):
    """Tests tensor head reshaping and sequence padding."""
    arr = jnp.arange(8 * 16 * 32, dtype=jnp.float32).reshape(8, 16, 32)
    heads = 4

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("batch", "heads"))
    with jax.set_mesh(mesh):
      reshaped = attention_utils._reshape_batch_dim_to_heads(arr, heads)
      restored = attention_utils._reshape_heads_to_batch_dim(reshaped, heads)
      np.testing.assert_allclose(arr, restored)

    # Test padding
    short_arr = jnp.ones((2, 4, 10, 16), dtype=jnp.float32)
    padded, kv_size, orig_len = attention_utils._pad_data_for_flash(short_arr, heads=4, flash_block_size=8)
    self.assertEqual(orig_len, 10)
    self.assertEqual(padded.shape[2], 16)  # Padded to next multiple of block_size=8

  def test_attention_strategies_instantiation_and_hooks(self):
    """Tests strategy object creation and pre-all-to-all Cauchy-Schwarz hooks."""
    single = attention_strategies.SingleShardStrategy(
        local_kernel=None,
        block_sizes=None,
        use_base2_exp=True,
    )
    self.assertTrue(single.use_base2_exp)

    ring_tokamax = attention_strategies.RingAttentionStrategy(
        block_sizes=None,
        backend="tokamax",
    )
    self.assertIsNone(ring_tokamax.pre_all_to_all_hook(jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,))))

    ring_custom_fixed = attention_strategies.RingAttentionStrategy(
        block_sizes=None,
        backend="custom",
        use_fixed_m=True,
        num_ring_shards=2,
    )
    # pre_all_to_all_hook computes qk_max row norm expression
    q = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
    k = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
    v = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("ring", "ulysses"))
    with jax.set_mesh(mesh):
      hook_res = ring_custom_fixed.pre_all_to_all_hook(q, k, v)
      self.assertIsNotNone(hook_res)

    ulysses = attention_strategies.UlyssesStrategy(
        ulysses_shards=2,
        num_ring_shards=1,
        inner_strategy=single,
    )
    self.assertEqual(ulysses.ulysses_shards, 2)
    self.assertIs(ulysses.inner_strategy, single)

  def test_apply_attention_dot_numerical_correctness(self):
    """Tests reference dot-product attention against explicit numpy calculation."""
    q_3d = np.random.normal(size=(1, 2, 16)).astype(np.float32)
    k_3d = np.random.normal(size=(1, 2, 16)).astype(np.float32)
    v_3d = np.random.normal(size=(1, 2, 16)).astype(np.float32)
    scale = 0.5

    out = attention_dispatch._apply_attention_dot(
        jnp.array(q_3d),
        jnp.array(k_3d),
        jnp.array(v_3d),
        dtype=jnp.float32,
        heads=2,
        dim_head=8,
        scale=scale,
        split_head_dim=True,
    )

    # Explicit numpy reference calculation
    q = q_3d.reshape(1, 2, 2, 8)
    k = k_3d.reshape(1, 2, 2, 8)
    v = v_3d.reshape(1, 2, 2, 8)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) * scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    ref_out = np.einsum("bhqk,bkhd->bqhd", weights, v).reshape(1, 2, 16)

    np.testing.assert_allclose(out, ref_out, atol=1e-2, rtol=5e-2)

  def test_check_attention_inputs_validation(self):
    """Tests shape assertion logic in _check_attention_inputs."""
    q = jnp.ones((1, 4, 2, 8))
    k = jnp.ones((1, 4, 2, 8))
    v = jnp.ones((1, 4, 2, 8))
    attention_dispatch._check_attention_inputs(q, k, v)  # Should not raise

    k_bad = jnp.ones((1, 4, 2, 16))
    with self.assertRaises(ValueError):
      attention_dispatch._check_attention_inputs(q, k_bad, v)

  def test_jax_memory_efficient_attention_indivisible_chunking(self):
    """Tests chunked attention when sequence lengths are not divisible by chunk sizes."""
    np.random.seed(42)
    q = jnp.array(np.random.normal(size=(1, 7, 2, 8)).astype(np.float32))
    k = jnp.array(np.random.normal(size=(1, 10, 2, 8)).astype(np.float32))
    v = jnp.array(np.random.normal(size=(1, 10, 2, 8)).astype(np.float32))

    out_chunked = attention_dispatch.jax_memory_efficient_attention(q, k, v, query_chunk_size=3, key_chunk_size=4)
    out_ref = attention_dispatch._apply_attention_dot(
        q, k, v, dtype=jnp.float32, heads=2, dim_head=8, scale=8.0**-0.5, split_head_dim=True
    ).reshape(1, 7, 2, 8)
    np.testing.assert_allclose(out_chunked, out_ref, atol=1e-5, rtol=1e-5)

  def test_custom_ring_attention_branches_coverage(self):
    """Tests custom ring attention bidirectional=True and fixed-m _lse_scan fallback branch."""
    from maxdiffusion.models.attention_strategies import custom_ring
    from maxdiffusion.kernels import custom_splash_attention as custom_splash

    q = jnp.ones((2, 16, 32), dtype=jnp.float32)
    k = jnp.ones((2, 16, 32), dtype=jnp.float32)
    v = jnp.ones((2, 16, 32), dtype=jnp.float32)
    bsizes = custom_splash._BlockSizes(block_q=16, block_kv=16, block_kv_compute=16, block_kv_compute_in=16)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1), ("context",))

    def run_fwd(q_in, k_in, v_in, bi, fm, fm_norms):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=jax.sharding.PartitionSpec(),
          out_specs=jax.sharding.PartitionSpec(),
          check_vma=False,
      )
      def _mapped(q_i, k_i, v_i, norms_i):
        return custom_ring._custom_ring_attention_forward(
            q_i,
            k_i,
            v_i,
            block_sizes=bsizes,
            orig_q_seq_len=16,
            orig_kv_seq_len=16,
            ring_size=1,
            bidirectional=bi,
            use_fixed_m=fm,
            fixed_m_norms=norms_i,
            use_base2_exp=True,
            use_experimental_scheduler=False,
            vmem_limit_bytes=None,
            mask_value=-1e30,
            ring_axis="context",
        )

      return _mapped(q_in, k_in, v_in, fm_norms)

    def mock_splash_forward_ring(q_i, k_i, v_i, *args, **kwargs):
      s_len, n_h, d = q_i.shape
      return jnp.ones((s_len, n_h, d)), jnp.zeros((s_len, n_h)), jnp.zeros((s_len, n_h))

    with (
        mock.patch.object(custom_splash, "splash_attention_forward_ring", side_effect=mock_splash_forward_ring),
        mock.patch.object(custom_splash, "_splash_attention_forward_ring", side_effect=mock_splash_forward_ring),
    ):
      # 1. Test bidirectional=True standard path
      fwd_bi = run_fwd(q, k, v, True, False, None)
      self.assertEqual(fwd_bi.shape, q.shape)

      # 2. Test _lse_scan branch of use_fixed_m (when fixed bound is exceeded)
      fwd_lse_scan = run_fwd(q, k, v, False, True, (jnp.array([1e6, 1e6]), jnp.array([1e6, 1e6])))
      self.assertEqual(fwd_lse_scan.shape, q.shape)

  def test_ring_attention_unsupported_backend_message(self):
    """Tests that RingAttentionStrategy raises an informative ValueError for unsupported backends."""
    ring_invalid = attention_strategies.RingAttentionStrategy(block_sizes=None, backend="invalid_backend")
    with self.assertRaisesRegex(ValueError, "Expected one of: 'custom', 'tokamax'"):
      ring_invalid(
          jnp.ones((1, 2, 4, 8)),
          jnp.ones((1, 2, 4, 8)),
          jnp.ones((1, 2, 4, 8)),
          q_seq_len=4,
          kv_seq_len=4,
      )

  def test_single_shard_strategy_signature_filtering(self):
    """Tests that SingleShardStrategy filters kwargs based on local_kernel signature."""

    def fake_kernel_no_orig_q(block_sizes=None):
      return lambda q, k, v, mask=None: q

    def fake_kernel_with_kwargs(block_sizes=None, **kwargs):
      self.assertIn("orig_q_seq_len", kwargs)
      self.assertIn("orig_kv_seq_len", kwargs)
      return lambda q, k, v, mask=None: q

    q = jnp.ones((1, 2, 4, 8))
    strategy1 = attention_strategies.SingleShardStrategy(block_sizes=None, local_kernel=fake_kernel_no_orig_q)
    out1 = strategy1(q, q, q, q_seq_len=2, kv_seq_len=2)
    self.assertEqual(out1.shape, q.shape)

    strategy2 = attention_strategies.SingleShardStrategy(block_sizes=None, local_kernel=fake_kernel_with_kwargs)
    out2 = strategy2(q, q, q, q_seq_len=2, kv_seq_len=2)
    self.assertEqual(out2.shape, q.shape)

  def test_q_scaling_equivalence(self):
    """Tests that scaling Q produces identical attention outputs to standard dot-product attention."""
    key = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=jnp.float32)
    query = jnp.array([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=jnp.float32)
    value = jnp.array([[[[2.0, 3.0], [4.0, 5.0]]]], dtype=jnp.float32)

    scale = 0.5
    # Standard DPA reference
    dpa = attention_strategies.DotProductAttentionStrategy(scale=scale)
    ref_out = dpa(query, key, value, q_seq_len=2, kv_seq_len=2)

    # Q-scaled reference manually
    q_scaled = query * scale
    dpa_unscaled = attention_strategies.DotProductAttentionStrategy(scale=1.0)
    q_scaled_out = dpa_unscaled(q_scaled, key, value, q_seq_len=2, kv_seq_len=2)

    np.testing.assert_allclose(ref_out, q_scaled_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
