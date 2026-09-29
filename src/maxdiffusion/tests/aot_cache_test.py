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

  def test_empty_install_disables_and_clears_a_previous_cache(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertTrue(aot_cache._STATE.enabled)

    aot_cache.install("", meta={}, mesh=None)

    self.assertFalse(aot_cache._STATE.enabled)
    self.assertEqual(aot_cache._STATE.cache_dir, "")
    for entry in aot_cache._REGISTRY:
      self.assertFalse(entry._compiled)
      self.assertFalse(entry._pending)

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

  def test_graphdef_svg_config_changes_dynamic_signature_and_prevents_reuse(self):
    from flax import nnx

    class ToyExpert(nnx.Module):

      def __init__(self, use_svg: bool, density: float):
        self.use_svg_attention = use_svg
        self.svg_spatial_density = density
        self.config = {"attention_config": {"use_svg_attention": use_svg, "svg_spatial_density": density}}
        self.w = nnx.Param(jnp.ones((8, 8)))

      def __call__(self, x):
        scale = self.svg_spatial_density if self.use_svg_attention else 1.0
        return (x @ self.w[...]) * scale

    @aot_cache.cached_jit
    def forward(graphdef, state, x):
      model = nnx.merge(graphdef, state)
      return model(x)

    self._install()
    gd_dense, state_dense = nnx.split(ToyExpert(False, 1.0))
    gd_svg50, state_svg50 = nnx.split(ToyExpert(True, 0.5))
    gd_svg25, state_svg25 = nnx.split(ToyExpert(True, 0.25))

    sig_dense = aot_cache._dynamic_signature((gd_dense, state_dense, self._a), {})
    sig_svg50 = aot_cache._dynamic_signature((gd_svg50, state_svg50, self._a), {})
    sig_svg25 = aot_cache._dynamic_signature((gd_svg25, state_svg25, self._a), {})
    self.assertEqual(len({sig_dense, sig_svg50, sig_svg25}), 3)

    # Cache dense executable
    out_dense = forward(gd_dense, state_dense, self._a)
    self.assertEqual(aot_cache.save_pending(), 1)
    np.testing.assert_allclose(np.asarray(out_dense), np.full((8, 8), 8.0))

    # Call SVG expert with identical input shapes; must NOT reuse dense executable
    out_svg50 = forward(gd_svg50, state_svg50, self._a)
    np.testing.assert_allclose(np.asarray(out_svg50), np.full((8, 8), 4.0))
    self.assertEqual(aot_cache.save_pending(), 1)

  def test_extract_svg_meta_changes_fingerprint_on_svg_changes(self):
    from types import SimpleNamespace

    cfg_dense = SimpleNamespace(use_svg_attention=False, svg_spatial_density=1.0)
    cfg_svg50 = SimpleNamespace(use_svg_attention=True, svg_spatial_density=0.5)
    cfg_svg25 = SimpleNamespace(use_svg_attention=True, svg_spatial_density=0.25)

    meta_dense = aot_cache.extract_svg_meta(cfg_dense)
    meta_svg50 = aot_cache.extract_svg_meta(cfg_svg50)
    meta_svg25 = aot_cache.extract_svg_meta(cfg_svg25)

    fp_dense = aot_cache._metadata_fingerprint(meta_dense)
    fp_svg50 = aot_cache._metadata_fingerprint(meta_svg50)
    fp_svg25 = aot_cache._metadata_fingerprint(meta_svg25)
    self.assertEqual(len({fp_dense, fp_svg50, fp_svg25}), 3)

  def test_step_array_preserves_dynamic_signature_across_steps(self):
    sigs = {
        aot_cache._dynamic_signature((self._a,), {"svg_step_index": jnp.asarray(step, dtype=jnp.int32)})
        for step in range(40)
    }
    self.assertEqual(len(sigs), 1)


if __name__ == "__main__":
  unittest.main()
