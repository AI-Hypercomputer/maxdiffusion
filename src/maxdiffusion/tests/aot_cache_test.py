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

from maxdiffusion import aot_cache, wan_runtime_options


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
    wan_runtime_options.reset()
    for entry in aot_cache._REGISTRY:
      entry._compiled.clear()
      entry._pending.clear()

  def tearDown(self):
    aot_cache._STATE.enabled = False
    wan_runtime_options.reset()
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
        "from maxdiffusion import aot_cache",
        "from flax import nnx",
        "",
        "class T(nnx.Module):",
        "  def __init__(self, rngs):",
        "    self.lin = nnx.Linear(4, 4, rngs=rngs)",
        "    self.tags = {'alpha', 'beta', 'gamma'}",
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
            env={
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "PYTHONPATH": os.pathsep.join(sys.path),
                "PYTHONHASHSEED": str(seed),
            },
        ).stdout.strip()
        for seed in (1, 42)
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

  def test_graphdef_static_attribute_changes_dynamic_signature(self):
    """Toggling static attributes on nnx.GraphDef (such as use_k_centering) changes _dynamic_signature."""
    from flax import nnx

    class DummyBlock(nnx.Module):

      def __init__(self, use_k_centering: bool):
        self.attention_config = {"use_k_centering": use_k_centering}

    gd_on, _ = nnx.split(DummyBlock(use_k_centering=True))
    gd_off, _ = nnx.split(DummyBlock(use_k_centering=False))

    sig_on = aot_cache._dynamic_signature((gd_on, jnp.ones((2, 4))), {})
    sig_off = aot_cache._dynamic_signature((gd_off, jnp.ones((2, 4))), {})
    self.assertNotEqual(sig_on, sig_off)

  def test_wan_aot_metadata_includes_use_k_centering(self):
    """Toggling use_k_centering changes the Wan AOT metadata fingerprint."""
    import types
    from maxdiffusion import generate_wan

    cfg_on = types.SimpleNamespace(use_k_centering=True, attention="ulysses_ring_custom_fixed_m")
    cfg_off = types.SimpleNamespace(use_k_centering=False, attention="ulysses_ring_custom_fixed_m")

    meta_on = generate_wan._build_wan_aot_metadata(cfg_on, self._mesh, "rev1")
    meta_off = generate_wan._build_wan_aot_metadata(cfg_off, self._mesh, "rev1")

    self.assertEqual(meta_on["use_k_centering"], "True")
    self.assertEqual(meta_off["use_k_centering"], "False")
    self.assertNotEqual(
        aot_cache._metadata_fingerprint(meta_on),
        aot_cache._metadata_fingerprint(meta_off),
    )

  def test_wan_aot_metadata_includes_compiler_flags(self):
    """Raw LIBTPU_INIT_ARGS/XLA_FLAGS and runtime versions are included in the metadata."""
    import types
    from unittest import mock
    from maxdiffusion import generate_wan

    cfg = types.SimpleNamespace(attention="ulysses_ring_custom_fixed_m")

    def metadata(env):
      with mock.patch.dict(os.environ, env, clear=False):
        return generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")

    meta1 = metadata({"LIBTPU_INIT_ARGS": "--a=1 --b=true", "XLA_FLAGS": ""})
    meta2 = metadata({"LIBTPU_INIT_ARGS": "--a=1 --b=false", "XLA_FLAGS": ""})
    meta3 = metadata({"LIBTPU_INIT_ARGS": "--a=1 --b=true", "XLA_FLAGS": "--xla_c=2"})

    self.assertNotEqual(aot_cache._metadata_fingerprint(meta1), aot_cache._metadata_fingerprint(meta2))
    self.assertNotEqual(aot_cache._metadata_fingerprint(meta1), aot_cache._metadata_fingerprint(meta3))
    self.assertIn("jaxlib", meta1)
    self.assertIn("flax", meta1)
    self.assertIn("platform_version", meta1)
    self.assertIn("default_matmul_precision", meta1)
    self.assertIn("default_prng_impl", meta1)

  def test_wan_aot_metadata_includes_runtime_switches(self):
    """Runtime optimization env switches must change the Wan AOT metadata fingerprint."""
    import os
    import types
    from unittest import mock
    from maxdiffusion import generate_wan

    cfg = types.SimpleNamespace(attention="ulysses_ring_custom_fixed_m")
    base_meta = generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")
    base_fp = aot_cache._metadata_fingerprint(base_meta)

    switches = [
        ("WAN_ROPE_NORM_MODE", "fused"),
        ("WAN_FUSE_QK_PRESCALE", "0"),
        ("WAN_SPLASH_TRANSPOSE_OUT", "1"),
        ("WAN_CFG_BEFORE_UNPATCHIFY", "0"),
        ("WAN_CROSS_ATTN_PRESCALE_KV", "1"),
    ]
    for env_var, new_val in switches:
      with mock.patch.dict(os.environ, {env_var: new_val}):
        switched_meta = generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")
        switched_fp = aot_cache._metadata_fingerprint(switched_meta)
        self.assertNotEqual(
            base_fp,
            switched_fp,
            f"Changing {env_var} to {new_val} did not invalidate the Wan AOT fingerprint.",
        )

  def test_wan_aot_metadata_includes_resolved_rope_accum(self):
    """WAN_ROPE_ACCUM changes the lowered graph's rounding and must key the executable.

    This is deliberately NOT folded into the generic switches list above. The
    fingerprint records the *resolved* mode, whose default is
    platform-dependent ("dtype" on tpu7x and CPU, "f32" on v6e and other TPUs). Asserting on a
    fixed literal would be vacuous on whichever platform already defaults to
    it -- e.g. WAN_ROPE_ACCUM="f32" is a no-op on v6e. So the override is
    chosen to be the opposite of whatever this platform resolves to.
    """
    import os
    import types
    from unittest import mock
    from maxdiffusion import generate_wan
    from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import resolve_rope_accum

    cfg = types.SimpleNamespace(attention="ulysses_ring_custom_fixed_m")

    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("WAN_ROPE_ACCUM", None)
      default_mode = resolve_rope_accum(self._mesh)
      base_meta = generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")

    self.assertIn(default_mode, ("dtype", "f32"))
    self.assertEqual(base_meta["wan_rope_accum"], default_mode)

    other_mode = "f32" if default_mode == "dtype" else "dtype"
    with mock.patch.dict(os.environ, {"WAN_ROPE_ACCUM": other_mode}):
      switched_meta = generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")
    self.assertEqual(switched_meta["wan_rope_accum"], other_mode)
    self.assertNotEqual(
        aot_cache._metadata_fingerprint(base_meta),
        aot_cache._metadata_fingerprint(switched_meta),
        f"WAN_ROPE_ACCUM={other_mode} (platform default {default_mode}) did not invalidate the Wan AOT fingerprint.",
    )

    # An explicit override equal to the platform default describes the same
    # executable and must NOT force a recompile.
    with mock.patch.dict(os.environ, {"WAN_ROPE_ACCUM": default_mode}):
      same_meta = generate_wan._build_wan_aot_metadata(cfg, self._mesh, "rev1")
    self.assertEqual(
        aot_cache._metadata_fingerprint(base_meta),
        aot_cache._metadata_fingerprint(same_meta),
        f"WAN_ROPE_ACCUM={default_mode} matches the platform default and must reuse the cache.",
    )

  def test_resolve_rope_accum_rejects_unknown_mode(self):
    """An unrecognised WAN_ROPE_ACCUM must fail loudly rather than silently defaulting."""
    import os
    from unittest import mock
    from maxdiffusion.kernels.fused_rmsnorm_rope_pallas import resolve_rope_accum

    with mock.patch.dict(os.environ, {"WAN_ROPE_ACCUM": "fp32"}):
      with self.assertRaises(ValueError):
        resolve_rope_accum(self._mesh)

  def test_wan_source_hash_includes_shared_modules_and_prefers_content_hash_over_commit(self):
    """Verifies _compute_wan_source_hash hashes shared modules and prefers content hash over commit."""
    import types
    from unittest import mock
    from maxdiffusion import generate_wan

    base_hash = generate_wan._compute_wan_source_hash()
    self.assertIsNotNone(base_hash)
    self.assertTrue(base_hash.startswith("src:"))

    # Simulate modifying models/normalization_flax.py or models/embeddings_flax.py
    orig_open = open

    def patched_open(path, *args, **kwargs):
      f = orig_open(path, *args, **kwargs)
      if str(path).endswith(("normalization_flax.py", "embeddings_flax.py")) and "rb" in args:
        content = f.read()
        f.close()
        import io

        return io.BytesIO(content + b"\n# modified for fingerprint test\n")
      return f

    generate_wan._compute_wan_source_hash.cache_clear()
    with mock.patch("builtins.open", side_effect=patched_open):
      mod_hash = generate_wan._compute_wan_source_hash()
    generate_wan._compute_wan_source_hash.cache_clear()

    self.assertNotEqual(base_hash, mod_hash)

    # Content hash is preferred, reusable, and not invalidated by dirty status
    cfg = types.SimpleNamespace(aot_build_revision=None)
    rev = generate_wan._resolve_wan_aot_source_revision(cfg, commit_hash="commit_dirty-dirty")
    self.assertEqual(rev, base_hash)
    self.assertTrue(generate_wan._is_reusable_aot_revision(rev))

    # When source hash is unavailable, clean commit hashes are reusable while
    # "<sha>-dirty" commit hashes from get_git_commit_hash() are rejected.
    self.assertTrue(generate_wan._is_reusable_aot_revision("abc1234"))
    self.assertFalse(generate_wan._is_reusable_aot_revision("abc1234-dirty"))
    self.assertFalse(generate_wan._is_reusable_aot_revision("dirty:abc1234"))
    self.assertFalse(generate_wan._is_reusable_aot_revision("unversioned:abc1234"))

    # Explicit aot_build_revision takes highest precedence
    cfg_explicit = types.SimpleNamespace(aot_build_revision="build-explicit-123")
    self.assertEqual(generate_wan._resolve_wan_aot_source_revision(cfg_explicit), "build-explicit-123")

  def test_different_nnx_modules_and_unordered_sets_in_dynamic_signature(self):
    from flax import nnx

    class ModMul(nnx.Module):

      def __init__(self):
        self.scale = 2
        self.tags = {"alpha", "beta", "gamma"}

      def __call__(self, x):
        return x * self.scale

    class ModAdd(nnx.Module):

      def __init__(self):
        self.scale = 2
        self.tags = {"gamma", "alpha", "beta"}

      def __call__(self, x):
        return x + self.scale

    gdef_mul, state_mul = nnx.split(ModMul())
    gdef_add, state_add = nnx.split(ModAdd())
    x = jnp.array(3.0, dtype=jnp.float32)

    sig_mul = aot_cache._dynamic_signature((gdef_mul, state_mul, x), {})
    sig_add = aot_cache._dynamic_signature((gdef_add, state_add, x), {})
    self.assertNotEqual(sig_mul, sig_add)

    # Unordered sets inside static attributes must format canonically
    self.assertEqual(
        aot_cache._format_static_val({"a", "b", "c"}),
        "set({'a','b','c'})",
    )

    self._install()
    fn = aot_cache.cached_jit(lambda g, s, val: nnx.merge(g, s)(val))
    out_mul = fn(gdef_mul, state_mul, x)
    aot_cache.save_pending()
    out_add = fn(gdef_add, state_add, x)
    self.assertAlmostEqual(float(out_mul), 6.0)
    self.assertAlmostEqual(float(out_add), 5.0)

    # Same module class with different static attributes must not collide in _sig_cache
    mod_ten = ModMul()
    mod_ten.scale = 10
    gdef_ten, state_ten = nnx.split(mod_ten)
    out_ten = fn(gdef_ten, state_ten, x)
    self.assertAlmostEqual(float(out_ten), 30.0)

  def test_compiled_call_failure_raises_loudly_without_fallback(self):
    """When compiled(flat) fails, it must raise loudly instead of silently recompiling via JIT."""
    from unittest import mock

    self._install()

    @aot_cache.cached_jit
    def fn(x):
      return x + 1.0

    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    # Warmup and compile
    with aot_cache.warmup_mode():
      fn(x)

    # Corrupt the compiled entry so compiled(flat) raises
    sig = next(iter(fn._compiled.keys()))
    bad_compiled = mock.MagicMock(side_effect=RuntimeError("simulated execution error"))
    bad_compiled.input_shardings = fn._compiled[sig].input_shardings
    fn._compiled[sig] = bad_compiled

    with self.assertRaises(RuntimeError) as ctx:
      fn(x)
    self.assertIn("simulated execution error", str(ctx.exception))

  def test_warmup_mode_with_tempdir_install_returns_zeros(self):
    """Installing a throwaway dir (as generate_wan does when persistence is disabled) enables zero-execution warmup."""
    import tempfile

    ephemeral_dir = tempfile.mkdtemp(prefix="aot_ephemeral_")
    aot_cache.install(ephemeral_dir, meta={"test": "warmup"}, mesh=self._mesh)
    self.assertTrue(aot_cache._STATE.enabled)

    with aot_cache.warmup_mode():
      res = _toy_fn(self._a, self._b, True)
      self.assertEqual(res.shape, (8, 8))
      self.assertTrue(jnp.all(res == 0))

    # Outside warmup, real execution runs and returns computed values
    real = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(real), self._a @ self._b + 1.0)

  def test_bytecode_digest_distinguishes_constants(self):
    """Bytecode digest must distinguish functions that differ only by constant values."""

    def f1(x):
      return x * 2.0

    def f2(x):
      return x * 3.0

    val1 = aot_cache._format_static_val(f1)
    val2 = aot_cache._format_static_val(f2)
    self.assertNotEqual(val1, val2)

  def test_graphdef_memoization_caches_by_id(self):
    """GraphDef statics extraction must memoize results by id(obj) without re-traversal."""
    from unittest import mock
    from flax import nnx

    class SimpleMod(nnx.Module):

      def __init__(self):
        self.param = 42

    mod = SimpleMod()
    gdef, _ = nnx.split(mod)

    aot_cache._GRAPHDEF_MEMO.clear()
    out1 = aot_cache._extract_graphdef_statics(gdef)
    self.assertIn(id(gdef), aot_cache._GRAPHDEF_MEMO)
    self.assertEqual(len(out1), 1)

    # Second call uses cached digest in _GRAPHDEF_MEMO without calling _format_static_val
    with mock.patch(
        "maxdiffusion.aot_cache._format_static_val",
        wraps=aot_cache._format_static_val,
    ) as spy_fmt:
      out2 = aot_cache._extract_graphdef_statics(gdef)
      self.assertEqual(spy_fmt.call_count, 0)
    self.assertEqual(out1, out2)

  def test_use_k_centering_defaults_to_auto(self):
    """use_k_centering must default to 'auto' in pyconfig and WanTransformerBlock."""
    from maxdiffusion import pyconfig
    from maxdiffusion.models.wan.transformers.transformer_wan import WanTransformerBlock
    from flax import nnx

    pyconfig.initialize([None, "src/maxdiffusion/configs/base_wan_14b.yml", "run_name=test_k_centering"])
    self.assertEqual(pyconfig.config.use_k_centering, "auto")

    # Check WanTransformerBlock default attention_config
    block = WanTransformerBlock(rngs=nnx.Rngs(0), dim=64, num_heads=4, ffn_dim=128, cross_attn_norm=True)
    self.assertEqual(block.attn1.attention_op.use_k_centering, "auto")

  def test_video_output_path_preserves_prefix_and_gcs_fallback(self):
    """format_video_output_path must preserve prefix and handle directory vs local naming."""
    from maxdiffusion.generate_wan import format_video_output_path

    # With output_dir
    path_with_prefix = format_video_output_path("/tmp/test_wan_out", "wan_run", 42, 0, "prefix_")
    self.assertEqual(path_with_prefix, "/tmp/test_wan_out/prefix_wan_run_42_0.mp4")

    path_no_prefix = format_video_output_path("/tmp/test_wan_out", "wan_run", 42, 1)
    self.assertEqual(path_no_prefix, "/tmp/test_wan_out/wan_run_42_1.mp4")

    # Without output_dir (or GCS)
    path_empty_dir = format_video_output_path("", "wan_run", 42, 0, "prefix_")
    self.assertEqual(path_empty_dir, "prefix_wan_output_42_0.mp4")

    path_gcs_dir = format_video_output_path("gs://bucket/dir", "wan_run", 42, 2, "my_")
    self.assertEqual(path_gcs_dir, "my_wan_output_42_2.mp4")

  def test_float32_qk_product_gate(self):
    """_apply_attention_dot must use float32 preferred_element_type only when float32_qk_product=True."""
    from maxdiffusion.models.attention_flax import _apply_attention_dot

    q = jnp.ones((1, 4, 4, 16), dtype=jnp.bfloat16)
    k = jnp.ones((1, 4, 4, 16), dtype=jnp.bfloat16)
    v = jnp.ones((1, 4, 4, 16), dtype=jnp.bfloat16)

    def fn_false(q, k, v):
      return _apply_attention_dot(
          q,
          k,
          v,
          dtype=jnp.bfloat16,
          heads=4,
          dim_head=16,
          scale=0.25,
          split_head_dim=True,
          float32_qk_product=False,
          use_memory_efficient_attention=False,
      )

    def fn_true(q, k, v):
      return _apply_attention_dot(
          q,
          k,
          v,
          dtype=jnp.bfloat16,
          heads=4,
          dim_head=16,
          scale=0.25,
          split_head_dim=True,
          float32_qk_product=True,
          use_memory_efficient_attention=False,
      )

    jaxpr_false = jax.make_jaxpr(fn_false)(q, k, v)
    jaxpr_true = jax.make_jaxpr(fn_true)(q, k, v)

    preferred_false = [
        eq.params.get("preferred_element_type") for eq in jaxpr_false.eqns if eq.primitive.name == "dot_general"
    ]
    preferred_true = [
        eq.params.get("preferred_element_type") for eq in jaxpr_true.eqns if eq.primitive.name == "dot_general"
    ]

    # The first dot_general is QK product; second is (QK)V product.
    self.assertEqual(preferred_false[0], jnp.dtype("bfloat16"))
    self.assertEqual(preferred_true[0], jnp.dtype("float32"))

  def test_graphdef_memo_prevents_gc_address_reuse_collision(self):
    """_GRAPHDEF_MEMO must retain object references to prevent id() reuse collisions after GC."""
    from flax import nnx
    from maxdiffusion.aot_cache import _extract_graphdef_statics

    class Dummy(nnx.Module):

      def __init__(self, scale: float):
        self.scale = scale

    seen_digests = set()
    for i in range(100):
      m = Dummy(scale=float(i))
      graphdef, _ = nnx.split(m)
      statics = _extract_graphdef_statics(graphdef)
      digest = statics[0]
      self.assertNotIn(digest, seen_digests, f"Collision detected at iter {i}: {digest}")
      seen_digests.add(digest)

  def test_pending_stores_shape_dtype_structs_and_clear_pending_empties(self):
    """_pending must store ShapeDtypeStructs (not live jax.Arrays) and clear_pending() must empty it."""
    self._install()
    _toy_fn(self._a, self._b, True)
    entry = next(e for e in aot_cache._REGISTRY if e.name.endswith("._toy_fn"))
    self.assertEqual(len(entry._pending), 1)
    leaves, _, _ = next(iter(entry._pending.values()))
    for leaf in leaves:
      self.assertIsInstance(leaf, jax.ShapeDtypeStruct)
      self.assertNotIsInstance(leaf, jax.Array)
    # Re-lowering from ShapeDtypeStructs in save_pending succeeds even when _compiled is empty
    entry._compiled.clear()
    self.assertEqual(aot_cache.save_pending(), 1)

    # Record another shape and verify clear_pending() discards it
    _toy_fn(jnp.ones((4, 8)), jnp.ones((8, 4)), True)
    self.assertEqual(len(entry._pending), 1)
    aot_cache.clear_pending()
    self.assertEqual(len(entry._pending), 0)

  def test_nested_warmup_mode_restores_previous_state(self):
    """Nested warmup_mode() must restore outer warmup_only state on exit."""
    self._install()
    self.assertFalse(aot_cache.in_warmup())
    with aot_cache.warmup_mode():
      self.assertTrue(aot_cache.in_warmup())
      with aot_cache.warmup_mode():
        self.assertTrue(aot_cache.in_warmup())
      self.assertTrue(aot_cache.in_warmup())
    self.assertFalse(aot_cache.in_warmup())

  def test_format_static_val_canonicalizes_use_default_values_and_frozensets(self):
    """_format_static_val must sort diffusers _use_default_values lists and _format_const must sort frozensets."""
    d1 = {"_use_default_values": ["wan_seq_pad", "wan_patch_embed_mode"], "attention": "flash"}
    d2 = {"_use_default_values": ["wan_patch_embed_mode", "wan_seq_pad"], "attention": "flash"}
    self.assertEqual(aot_cache._format_static_val(d1), aot_cache._format_static_val(d2))
    self.assertEqual(
        aot_cache._format_const(frozenset({"b", "a", "c"})),
        "frozenset({'a','b','c'})",
    )


if __name__ == "__main__":
  unittest.main()
