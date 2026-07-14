# Copyright 2026 Google LLC
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

"""Tests for the per-shape AOT executable cache (CPU backend)."""

import functools
import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion import aot_cache


@functools.partial(aot_cache.cached_jit, static_argnames=("flag",))
def _toy_fn(x, y, flag=False):
  return x @ y + (1.0 if flag else 0.0)


class AotCacheTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self._mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    self._a = jnp.ones((8, 8))
    self._b = jnp.eye(8)
    # Reset process-global install state between tests.
    aot_cache._STATE.enabled = False
    for entry in aot_cache._REGISTRY:
      entry._compiled.clear()
      entry._pending.clear()

  def tearDown(self):
    aot_cache._STATE.enabled = False
    self._tmp.cleanup()

  def _install(self):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh)
    aot_cache.wait_for_loads()

  def test_disabled_is_plain_jit(self):
    result = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(result, self._a @ self._b + 1.0)
    self.assertFalse(aot_cache._STATE.enabled)

  def test_record_save_hit_roundtrip(self):
    self._install()
    first = _toy_fn(self._a, self._b, True)  # miss -> jit + record
    self.assertEqual(aot_cache.save_pending(), 1)
    hit = _toy_fn(self._a, self._b, True)  # compiled hit
    np.testing.assert_allclose(np.asarray(first), np.asarray(hit))

  def test_reload_from_disk(self):
    self._install()
    first = _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()
    entry = next(e for e in aot_cache._REGISTRY if e.name.endswith("._toy_fn"))
    entry._compiled.clear()  # simulate a fresh process
    entry.load_from_disk()
    self.assertTrue(entry._compiled)
    reloaded = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(first), np.asarray(reloaded))

  def test_static_value_gets_own_executable(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()
    # Different static value -> different signature -> jit fallback, correct.
    off = _toy_fn(self._a, self._b, False)
    np.testing.assert_allclose(off, self._a @ self._b)

  def test_new_shape_falls_back_and_saves(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    small = _toy_fn(jnp.ones((4, 8)), jnp.ones((8, 4)), True)
    self.assertEqual(small.shape, (4, 4))
    self.assertEqual(aot_cache.save_pending(), 1)  # only the new shape

  def test_warmup_mode_compiles_without_executing(self):
    self._install()
    with aot_cache.warmup_mode():
      warm = _toy_fn(self._a, self._b, True)
    # Zeros prove the fn body never ran (real output would be a@b+1).
    self.assertEqual(warm.shape, (8, 8))
    np.testing.assert_allclose(np.asarray(warm), np.zeros((8, 8)))
    # The signature was compiled during warmup and is serializable.
    self.assertEqual(aot_cache.save_pending(), 1)
    # Outside warmup mode the compiled executable returns real values.
    real = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(real), self._a @ self._b + 1.0)

  def test_warmup_mode_disabled_cache_executes_normally(self):
    with aot_cache.warmup_mode():  # cache not installed -> no-op
      result = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)

  def test_traced_call_inlines_like_nested_jit(self):
    # I2V's denoise loop invokes wrapped fns inside lax.cond branches; a
    # deserialized executable cannot be applied to tracers. The wrapper
    # must inline (plain nested-jit behavior) and never crash or record.
    self._install()
    _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()  # compiled entry exists for this signature

    def branch_true(x):
      return _toy_fn(x, self._b, True)

    def branch_false(x):
      return x

    result = jax.lax.cond(True, branch_true, branch_false, self._a)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    self.assertEqual(aot_cache.save_pending(), 0)  # nothing recorded

  def test_signature_deterministic_across_processes(self):
    # Signatures live in filenames; a process-dependent component (e.g.
    # object addresses inside a GraphDef repr) would make every restart
    # miss its own cache. Compute the same signature in two interpreters.
    import subprocess
    import sys

    snippet = "\n".join((
        "import os",
        "os.environ['JAX_PLATFORMS'] = 'cpu'",
        "import jax",
        "import jax.numpy as jnp",
        "from flax import nnx",
        "from maxdiffusion import aot_cache",
        "",
        "class T(nnx.Module):",
        "  def __init__(self, rngs):",
        "    self.lin = nnx.Linear(4, 4, rngs=rngs)",
        "",
        "graphdef, state = nnx.split(T(nnx.Rngs(0)))",
        "sig = aot_cache._dynamic_signature(",
        "    (graphdef, state.to_pure_dict(), jnp.ones((2, 4))), {})",
        "print(sig)",
    ))
    outs = [
        subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        ).stdout.strip()
        for _ in range(2)
    ]
    self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
  unittest.main()
