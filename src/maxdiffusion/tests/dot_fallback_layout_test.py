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

"""Layout regression tests for the short-sequence dot-product fallback.

Sequences below `flash_min_seq_length` bypass the flash/ulysses kernels and
run `_apply_attention_dot`. Callers that apply rotary embeddings hand the
dispatcher `[B, H, S, D]` -- the dispatcher says so itself, reading the
sequence length from axis 2 when `ndim == 4` -- but the `split_head_dim` path
reshaped those tensors as if they were `[B, S, H*D]`.

Both layouts hold the same number of elements, so the reshape succeeded and
returned a wrong answer with no error. That silent corruption is what these
tests exist to prevent. They are pure JAX and run on CPU.
"""

import math
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.attention_flax import _apply_attention_dot

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# Kernel numerics grids and multi-device tests are skipped in CI; see
# end_to_end/tpu/run_wan_stack_tests.sh.
_SKIP_IN_GITHUB_ACTIONS = unittest.skipIf(
    IN_GITHUB_ACTIONS, "TPU kernel / multi-device test, skipped in GitHub Actions; run end_to_end/tpu/run_wan_stack_tests.sh"
)


def _call_dot(query, key, value, heads, dim_head, kv_heads=None):
  """Runs the fallback and returns [B, H, S, D].

  `_apply_attention_dot` emits the flat `[B, S, H*D]` form, so unflatten it
  here rather than at each call site.
  """
  out = _apply_attention_dot(
      query=query,
      key=key,
      value=value,
      dtype=jnp.float32,
      heads=heads,
      dim_head=dim_head,
      scale=1.0 / math.sqrt(dim_head),
      split_head_dim=True,
      float32_qk_product=True,
      use_memory_efficient_attention=False,
      attention_mask=None,
      kv_heads=kv_heads,
  )
  batch, seq, _ = out.shape
  return jnp.swapaxes(out.reshape(batch, seq, heads, dim_head), 1, 2)


def _reference_attention(query, key, value, scale):
  """Dense f32 attention on explicit [B, H, S, D] inputs."""
  q, k, v = (x.astype(jnp.float32) for x in (query, key, value))
  logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
  return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), v)


class DotFallbackLayoutTest(unittest.TestCase):
  """`_apply_attention_dot` must transpose, not reshape, 4-D inputs."""

  def setUp(self):
    super().setUp()
    self._matmul_precision_ctx = jax.default_matmul_precision("float32")
    self._matmul_precision_ctx.__enter__()

  def tearDown(self):
    self._matmul_precision_ctx.__exit__(None, None, None)
    super().tearDown()

  def test_zero_logits_return_per_head_token_means(self):
    """The reviewer's counterexample, reproduced exactly.

    One active head dimension, three tokens, two heads. Q = K = 0 makes every
    logit zero, so softmax is uniform and each head's output is the mean of
    its values over tokens:

        head 0 values [1, 2, 3]  -> 2
        head 1 values [10, 20, 30] -> 20

    Reinterpreting [B, H, S, D] as [B, S, H, D] instead walks the buffer
    [1, 2, 3, 10, 20, 30] as three token-pairs (1,2), (3,10), (20,30),
    producing [8, 14].
    """
    heads, seq, dim_head = 2, 3, 1
    shape = (1, heads, seq, dim_head)
    query = jnp.zeros(shape, jnp.float32)
    key = jnp.zeros(shape, jnp.float32)
    value = jnp.array([[[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]]], dtype=jnp.float32)
    self.assertEqual(value.shape, shape)

    out = np.asarray(_call_dot(query, key, value, heads, dim_head))

    np.testing.assert_allclose(out[0, 0], 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 20.0, rtol=1e-5, atol=1e-5)
    # The exact wrong answer the reshape produced.
    self.assertFalse(np.allclose(out[0, 0], 8.0))
    self.assertFalse(np.allclose(out[0, 1], 14.0))

  def test_matches_reference_on_random_4d_inputs(self):
    heads, seq, dim_head = 4, 8, 16
    shape = (2, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(1), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(2), shape, jnp.float32)

    out = np.asarray(_call_dot(query, key, value, heads, dim_head))
    expected = np.asarray(_reference_attention(query, key, value, 1.0 / math.sqrt(dim_head)))
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

  def test_three_dim_inputs_agree_with_four_dim(self):
    """The flat [B, S, H*D] contract must keep working, and agree."""
    heads, seq, dim_head = 4, 8, 16
    batch = 2
    shape = (batch, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(3), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(4), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(5), shape, jnp.float32)

    def flatten(x):
      return jnp.swapaxes(x, 1, 2).reshape(batch, seq, heads * dim_head)

    out_4d = np.asarray(_call_dot(query, key, value, heads, dim_head))
    out_3d = np.asarray(_call_dot(flatten(query), flatten(key), flatten(value), heads, dim_head))
    np.testing.assert_allclose(out_4d, out_3d, rtol=1e-5, atol=1e-5)

  def test_gqa_repeat_still_applies_on_4d(self):
    """Head repetition must happen after the transpose, on the head axis."""
    heads, kv_heads, seq, dim_head = 4, 2, 8, 16
    q = jax.random.normal(jax.random.PRNGKey(6), (1, heads, seq, dim_head), jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(7), (1, kv_heads, seq, dim_head), jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(8), (1, kv_heads, seq, dim_head), jnp.float32)

    out = np.asarray(_call_dot(q, k, v, heads, dim_head, kv_heads=kv_heads))
    expected = np.asarray(
        _reference_attention(
            q,
            jnp.repeat(k, heads // kv_heads, axis=1),
            jnp.repeat(v, heads // kv_heads, axis=1),
            1.0 / math.sqrt(dim_head),
        )
    )
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

  def test_qk_prescaled_with_base2_matches_unscaled(self):
    """When Q is scaled by log2(e) and K is scaled by scale, dot attention with
    qk_prescaled=True and use_base2_exp=True must match to within 1e-5."""
    heads, seq, dim_head = 4, 8, 16
    shape = (2, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(10), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(11), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(12), shape, jnp.float32)
    scale = 1.0 / math.sqrt(dim_head)
    log2e = math.log2(math.e)

    with jax.default_matmul_precision("float32"):
      out_unscaled = _apply_attention_dot(
          query=query,
          key=key,
          value=value,
          dtype=jnp.float32,
          heads=heads,
          dim_head=dim_head,
          scale=scale,
          split_head_dim=True,
          float32_qk_product=True,
          use_memory_efficient_attention=False,
          qk_prescaled=False,
          use_base2_exp=False,
      )

      q_prescaled = query * log2e
      k_prescaled = key * scale
      out_prescaled = _apply_attention_dot(
          query=q_prescaled,
          key=k_prescaled,
          value=value,
          dtype=jnp.float32,
          heads=heads,
          dim_head=dim_head,
          scale=scale,
          split_head_dim=True,
          float32_qk_product=True,
          use_memory_efficient_attention=False,
          qk_prescaled=True,
          use_base2_exp=True,
      )
    np.testing.assert_allclose(np.asarray(out_prescaled), np.asarray(out_unscaled), rtol=1e-5, atol=1e-5)

  def test_k_prescaled_matches_unscaled(self):
    """When only K is prescaled by scale, dot attention with k_prescaled=True
    must match to within 1e-5 without double-scaling."""
    heads, seq, dim_head = 4, 8, 16
    shape = (2, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(13), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(14), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(15), shape, jnp.float32)
    scale = 1.0 / math.sqrt(dim_head)

    out_unscaled = _apply_attention_dot(
        query=query,
        key=key,
        value=value,
        dtype=jnp.float32,
        heads=heads,
        dim_head=dim_head,
        scale=scale,
        split_head_dim=True,
        float32_qk_product=True,
        use_memory_efficient_attention=False,
        qk_prescaled=False,
        k_prescaled=False,
    )

    k_prescaled_arr = key * scale
    out_k_prescaled = _apply_attention_dot(
        query=query,
        key=k_prescaled_arr,
        value=value,
        dtype=jnp.float32,
        heads=heads,
        dim_head=dim_head,
        scale=scale,
        split_head_dim=True,
        float32_qk_product=True,
        use_memory_efficient_attention=False,
        qk_prescaled=False,
        k_prescaled=True,
    )
    np.testing.assert_allclose(np.asarray(out_k_prescaled), np.asarray(out_unscaled), rtol=1e-5, atol=1e-5)

  def test_uncached_unaffected_by_cross_attn_prescale_kv(self):
    """wan_cross_attn_prescale_kv=True prescales cached cross-attn K in compute_kv, without affecting uncached or cached outputs."""
    from flax import nnx
    from maxdiffusion.models.attention_flax import FlaxWanAttention

    heads, seq_q, seq_kv, dim_head = 4, 8, 6, 16
    dim = heads * dim_head
    hidden_states = jax.random.normal(jax.random.PRNGKey(16), (2, seq_q, dim), jnp.float32)
    encoder_hidden_states = jax.random.normal(jax.random.PRNGKey(17), (2, seq_kv, dim), jnp.float32)
    scale = 1.0 / math.sqrt(dim_head)

    def build(prescale_kv):
      return FlaxWanAttention(
          rngs=nnx.Rngs(0),
          query_dim=dim,
          heads=heads,
          dim_head=dim_head,
          attention_kernel="dot_product",
          is_self_attention=False,
          dtype=jnp.float32,
          weights_dtype=jnp.float32,
          attention_config={"wan_cross_attn_prescale_kv": prescale_kv},
      )

    attn0 = build(False)
    self.assertFalse(attn0.cross_attn_prescale_kv)
    out_uncached_0 = attn0(hidden_states, encoder_hidden_states)
    kv0 = attn0.compute_kv(encoder_hidden_states)
    out_cached_0 = attn0(hidden_states, cached_kv=kv0)

    attn1 = build(True)
    self.assertTrue(attn1.cross_attn_prescale_kv)
    out_uncached_1 = attn1(hidden_states, encoder_hidden_states)
    kv1 = attn1.compute_kv(encoder_hidden_states)
    out_cached_1 = attn1(hidden_states, cached_kv=kv1)

    # Cached K in kv1["text"][0] is prescaled by scale compared to kv0["text"][0]
    np.testing.assert_allclose(np.asarray(kv1["text"][0]), np.asarray(kv0["text"][0] * scale), rtol=1e-6, atol=1e-6)
    # Uncached outputs are identical, and cached outputs with prescaled K match unprescaled within float32 precision
    np.testing.assert_allclose(np.asarray(out_uncached_1), np.asarray(out_uncached_0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_cached_1), np.asarray(out_cached_0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_cached_0), np.asarray(out_uncached_0), rtol=1e-5, atol=1e-5)

  def test_memory_efficient_attention_with_prescaled_k(self):
    """When use_memory_efficient_attention=True, k_prescaled=True must not double-scale."""
    heads, seq, dim_head = 4, 8, 16
    shape = (2, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(19), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(20), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(21), shape, jnp.float32)
    scale = 1.0 / math.sqrt(dim_head)

    out_unscaled = _apply_attention_dot(
        query=query,
        key=key,
        value=value,
        dtype=jnp.float32,
        heads=heads,
        dim_head=dim_head,
        scale=scale,
        split_head_dim=True,
        float32_qk_product=True,
        use_memory_efficient_attention=True,
        qk_prescaled=False,
        k_prescaled=False,
    )
    out_prescaled = _apply_attention_dot(
        query=query,
        key=key * scale,
        value=value,
        dtype=jnp.float32,
        heads=heads,
        dim_head=dim_head,
        scale=scale,
        split_head_dim=True,
        float32_qk_product=True,
        use_memory_efficient_attention=True,
        qk_prescaled=False,
        k_prescaled=True,
    )
    np.testing.assert_allclose(np.asarray(out_prescaled), np.asarray(out_unscaled), rtol=1e-5, atol=1e-5)

  def test_cfg_before_unpatchify_is_bit_identical(self):
    """Applying CFG on packed tokens before unpatchify_tokens is 0-ULP bit-identical to CFG after unpatchify_tokens."""
    import types
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    batch, frames, height, width = 2, 3, 4, 4
    p_t, p_h, p_w = 1, 2, 2
    out_dim = 16
    tokens = (frames // p_t) * (height // p_h) * (width // p_w)
    patch_dim = p_t * p_h * p_w * out_dim

    pred_cond = jax.random.normal(jax.random.PRNGKey(30), (batch, tokens, patch_dim), jnp.bfloat16)
    pred_uncond = jax.random.normal(jax.random.PRNGKey(31), (batch, tokens, patch_dim), jnp.bfloat16)
    guidance_scale = 4.0

    dummy = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=(p_t, p_h, p_w)))

    cfg_tokens = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    before = WanModel.unpatchify_tokens(dummy, cfg_tokens, frames, height, width)

    cond_vol = WanModel.unpatchify_tokens(dummy, pred_cond, frames, height, width)
    uncond_vol = WanModel.unpatchify_tokens(dummy, pred_uncond, frames, height, width)
    after = uncond_vol + guidance_scale * (cond_vol - uncond_vol)

    np.testing.assert_array_equal(np.asarray(before, np.float32), np.asarray(after, np.float32))


def _bf16_cfg_reroute_may_round_differently() -> bool:
  """True on TPUs where the bf16 CFG reroute is not bit-identical (bit-identical on CPU and v6e)."""
  kinds = [d.device_kind.lower() for d in jax.devices() if d.platform == "tpu"]
  return bool(kinds) and not all("v6" in k for k in kinds)


@_SKIP_IN_GITHUB_ACTIONS
class CfgBeforeUnpatchifyCompiledParityTest(unittest.TestCase):
  """wan_cfg_before_unpatchify and the Wan 2.1 no-cache CFG reroute, compiled through a real WanModel.

  The eager test above only checks unpatchify_tokens. Here both paths go
  through `transformer_forward_pass` under jit (with the production mesh and
  logical axis rules), so XLA is free to fuse the CFG combine with the
  unpatchify transpose. The result must still be bit-identical, for p_t == 1
  (the collapsed reshape) and p_t == 2, in f32 everywhere and in bf16 on CPU
  and v6e; in bf16 on tpu7x it is within one bf16 ULP. The reroute is also
  checked against `transformer_forward_pass_full_cfg`, the path Wan 2.1 no-cache
  CFG used before.
  """

  def test_compiled_paths_are_bit_identical(self):
    import os
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
    from maxdiffusion.pipelines.wan import wan_pipeline

    pyconfig.initialize(
        [None, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "base_wan_14b.yml")],
        unittest=True,
    )
    config = pyconfig.config
    mesh = Mesh(create_device_mesh(config), config.mesh_axes)

    def forward(patch_size, dtype, cfg_before_unpatchify):
      model = WanModel(
          rngs=nnx.Rngs(0),
          patch_size=patch_size,
          num_attention_heads=2,
          attention_head_dim=16,
          in_channels=4,
          out_channels=4,
          text_dim=32,
          freq_dim=32,
          ffn_dim=64,
          num_layers=2,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=dtype,
          scan_layers=False,
          wan_cfg_before_unpatchify=cfg_before_unpatchify,
      )
      graphdef, state, rest = nnx.split(model, nnx.Param, ...)
      latents = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 4, 8, 8), dtype)
      embeds = jax.random.normal(jax.random.PRNGKey(2), (2, 8, 32), dtype)
      timestep = jnp.full((2,), 500.0)
      out = wan_pipeline.transformer_forward_pass(
          graphdef, state, rest, latents, timestep, embeds, do_classifier_free_guidance=True, guidance_scale=5.0
      )
      full_cfg, _, _ = wan_pipeline.transformer_forward_pass_full_cfg(
          graphdef, state, rest, jnp.concatenate([latents] * 2), timestep, embeds, guidance_scale=5.0
      )
      return np.asarray(out, np.float32), np.asarray(full_cfg, np.float32)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      for patch_size in ((1, 2, 2), (2, 2, 2)):
        for dtype in (jnp.float32, jnp.bfloat16):
          with self.subTest(patch_size=patch_size, dtype=jnp.dtype(dtype).name):
            before, full_cfg = forward(patch_size, dtype, True)
            after, _ = forward(patch_size, dtype, False)
            self.assertEqual(before.shape, (1, 4, 4, 8, 8))
            if dtype == jnp.float32 or not _bf16_cfg_reroute_may_round_differently():
              np.testing.assert_array_equal(before, after)
              np.testing.assert_array_equal(before, full_cfg)
            else:
              # tpu7x fuses the bf16 CFG combine differently on the two paths, so
              # it is rounded at a different point: ~30% of elements differ by
              # one bf16 ULP of the output scale (measured on tpu7x-8).
              atol = 2**-7 * float(np.max(np.abs(before)))
              np.testing.assert_allclose(before, after, rtol=2**-7, atol=atol)
              np.testing.assert_allclose(before, full_cfg, rtol=2**-7, atol=atol)


class DispatcherLayoutContractTest(unittest.TestCase):
  """The threshold check and the dot path must agree on where S lives."""

  def test_seq_len_axis_for_4d_is_the_third_axis(self):
    """`_apply_attention` reads S from axis 2 for 4-D, i.e. [B, H, S, D].

    That is the convention `_apply_attention_dot` now honours by transposing.
    If the dispatcher ever moves to [B, S, H, D], the transpose becomes wrong
    and this assertion should be revisited alongside it.
    """
    query = jnp.zeros((1, 4, 128, 8), jnp.float32)  # B, H, S, D
    seq_len_idx = 2 if query.ndim == 4 else 1
    self.assertEqual(query.shape[seq_len_idx], 128)


if __name__ == "__main__":
  unittest.main()
