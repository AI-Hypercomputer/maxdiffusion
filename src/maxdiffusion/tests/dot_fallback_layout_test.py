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
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models import attention_flax
from maxdiffusion.models.attention_flax import _apply_attention, _apply_attention_dot


def _call_dot(query, key, value, heads, dim_head, kv_heads=None, split_head_dim=True):
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
      split_head_dim=split_head_dim,
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
    """Both the [B, S, H, D] and the batched-heads ([B*H, S, D]) paths."""
    heads, seq, dim_head = 4, 8, 16
    shape = (2, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(1), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(2), shape, jnp.float32)
    expected = np.asarray(_reference_attention(query, key, value, 1.0 / math.sqrt(dim_head)))

    for split_head_dim in (True, False):
      with self.subTest(split_head_dim=split_head_dim):
        out = np.asarray(_call_dot(query, key, value, heads, dim_head, split_head_dim=split_head_dim))
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

    expected = np.asarray(
        _reference_attention(
            q,
            jnp.repeat(k, heads // kv_heads, axis=1),
            jnp.repeat(v, heads // kv_heads, axis=1),
            1.0 / math.sqrt(dim_head),
        )
    )
    for split_head_dim in (True, False):
      with self.subTest(split_head_dim=split_head_dim):
        out = np.asarray(_call_dot(q, k, v, heads, dim_head, kv_heads=kv_heads, split_head_dim=split_head_dim))
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

  def test_rejects_kv_heads_that_do_not_divide_heads(self):
    """A non-integer repeat factor or non-positive head count must raise ValueError."""
    heads, kv_heads, seq, dim_head = 4, 3, 8, 16
    q = jnp.zeros((1, heads, seq, dim_head), jnp.float32)
    kv = jnp.zeros((1, kv_heads, seq, dim_head), jnp.float32)
    for split_head_dim in (True, False):
      with self.subTest(split_head_dim=split_head_dim):
        with self.assertRaisesRegex(ValueError, "multiple of kv_heads"):
          _call_dot(q, kv, kv, heads, dim_head, kv_heads=kv_heads, split_head_dim=split_head_dim)
        with self.assertRaisesRegex(ValueError, "positive integers"):
          _call_dot(q, kv, kv, heads, dim_head, kv_heads=0, split_head_dim=split_head_dim)
        with self.assertRaisesRegex(ValueError, "positive integers"):
          _call_dot(q, kv, kv, heads, dim_head, kv_heads=-2, split_head_dim=split_head_dim)


class DispatcherLayoutContractTest(unittest.TestCase):
  """The threshold check and the dot path must agree on where S lives."""

  def test_dispatcher_reads_seq_len_from_axis_2_and_dot_path_honours_it(self):
    """`_apply_attention` reads S from axis 2 for 4-D inputs, i.e. [B, H, S, D].

    The shape is chosen so the two readings disagree: H = 128 is above the
    threshold while S = 32 is below it. Reading S from axis 2 routes to the
    dot-product fallback, which must then interpret the same tensor as
    [B, H, S, D] (the transpose, not the reshape). Reading S from axis 1 would
    try to run the configured flash kernel instead and never reach the spy.
    """
    batch, heads, seq, dim_head = 1, 128, 32, 8
    flash_min_seq_length = 64
    shape = (batch, heads, seq, dim_head)
    query = jax.random.normal(jax.random.PRNGKey(9), shape, jnp.float32)
    key = jax.random.normal(jax.random.PRNGKey(10), shape, jnp.float32)
    value = jax.random.normal(jax.random.PRNGKey(11), shape, jnp.float32)
    scale = 1.0 / math.sqrt(dim_head)

    real_kernel = attention_flax.KERNEL_REGISTRY["dot_product"]
    spy = mock.Mock(side_effect=real_kernel)
    with mock.patch.dict(attention_flax.KERNEL_REGISTRY, {"dot_product": spy}):
      out = _apply_attention(
          query,
          key,
          value,
          heads=heads,
          dim_head=dim_head,
          split_head_dim=True,
          float32_qk_product=True,
          attention_kernel="ulysses_custom",
          flash_min_seq_length=flash_min_seq_length,
          use_memory_efficient_attention=False,
          scale=scale,
          dtype=jnp.float32,
          mesh=None,
          axis_names_q=None,
          axis_names_kv=None,
          flash_block_sizes=None,
          dpa_layer=None,
      )

    spy.assert_called_once()
    seen_query = spy.call_args.args[0]
    self.assertEqual(seen_query.shape, shape)
    self.assertLess(seen_query.shape[2], flash_min_seq_length)
    self.assertGreaterEqual(seen_query.shape[1], flash_min_seq_length)
    out = jnp.swapaxes(jnp.reshape(out, (batch, seq, heads, dim_head)), 1, 2)
    expected = _reference_attention(query, key, value, scale)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
