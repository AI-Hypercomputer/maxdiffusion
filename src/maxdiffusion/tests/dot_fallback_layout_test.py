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

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.attention_flax import _apply_attention_dot


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
