# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

import unittest
from absl.testing import parameterized
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from . import base


def test_device_matches(devices: list[str]) -> bool:
  """Returns True if the test device matches any of the given devices."""
  return any(d.lower() in jax.devices()[0].device_kind.lower() for d in devices)


def thread_unsafe_test_class():
  """Decorator that marks a TestCase class as thread-hostile."""

  def f(klass):
    assert issubclass(klass, unittest.TestCase), type(klass)
    klass.thread_hostile = True
    return klass

  return f


class SplashAttentionTestCase(parameterized.TestCase):
  """Base class for SplashAttention tests."""

  INTERPRET = False

  def setUp(self):
    if self.INTERPRET and not test_device_matches(["cpu"]):
      self.skipTest("Interpret mode only supported on CPU")

    super().setUp()

  def _assert_array_equal(self, x, y, **kwargs):
    if x is None or y is None:
      self.assertIsNone(x)
      self.assertIsNone(y)
      return

    self.assertTrue(jnp.isfinite(x).all())
    self.assertTrue(jnp.isfinite(y).all())

    if x.dtype == np.dtype(jnp.bfloat16):
      x = x.astype(np.float32)
    if y.dtype == np.dtype(jnp.bfloat16):
      y = y.astype(np.float32)

    self.assertEqual(x.dtype, y.dtype)
    self.assertTupleEqual(x.shape, y.shape)
    np.testing.assert_array_equal(x, y, **kwargs)

  def _assert_allclose(self, x, y, **kwargs):
    if x.dtype == np.dtype(jnp.bfloat16):
      x = x.astype(np.float32)
    if y.dtype == np.dtype(jnp.bfloat16):
      y = y.astype(np.float32)
    self.assertEqual(x.dtype, y.dtype)
    self.assertTupleEqual(x.shape, y.shape)
    np.testing.assert_allclose(x, y, **kwargs)

  def assert_allclose_mcjax(self, x, y, *, rtol, atol):
    """`allclose` that is safe under multi-controller (multi-host) JAX.

    Some tests build their device mesh from a subset of the global devices --
    e.g. `jax.devices()[:ring_size]`, which all live on process 0 -- so the
    result `jax.Array`s are only addressable on that one process. Pulling them
    to host with `np.testing.assert_allclose` (as `_assert_allclose` does)
    raises `RuntimeError: ... spans non-addressable devices` on every *other*
    process, failing the test on all but one host.

    Instead, evaluate the comparison on-device (works on all processes, no host
    fetch), read the scalar verdict only on the owning process (process 0, which
    holds `jax.devices()[:ring_size]`), and broadcast it to every process with a
    single collective. Every process runs the same two `broadcast_one_to_all`
    calls in the same order, so there is no collective-participation mismatch /
    deadlock. The result is identical on all hosts and genuinely reflects the
    owner's computation.

    Only use this for tests whose mesh is a subset of one process's devices; for
    fully-sharded (every-process-owns-a-shard) arrays use `_assert_allclose`.
    """
    if x.dtype == np.dtype(jnp.bfloat16):
      x = x.astype(jnp.float32)
    if y.dtype == np.dtype(jnp.bfloat16):
      y = y.astype(jnp.float32)
    self.assertTupleEqual(x.shape, y.shape)
    ok = jnp.all(jnp.abs(x - y) <= atol + rtol * jnp.abs(y))
    max_err = jnp.max(jnp.abs(x - y))
    is_owner = jax.process_index() == 0
    local_ok = np.asarray(ok) if is_owner else np.array(True)
    local_err = np.asarray(max_err) if is_owner else np.array(0.0, np.float32)
    global_err = float(multihost_utils.broadcast_one_to_all(local_err))
    self.assertTrue(
        bool(multihost_utils.broadcast_one_to_all(local_ok)),
        f"arrays differ: max abs err {global_err:.3e} exceeds rtol={rtol} atol={atol}",
    )


def create_segment_ids(seq_len: int, num_breaks: int = 2) -> base.SegmentIds:
  break_indices = np.random.choice(range(1, seq_len), num_breaks, replace=False)
  idxs = np.zeros(seq_len, dtype=np.int32)
  idxs[break_indices] = 1

  idxs = np.cumsum(idxs, dtype=np.int32)
  return base.SegmentIds(q=idxs, kv=idxs)
