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

CPU tests for the tile-size e2e verification pass (maxdiffusion/utils/tile_e2e_verify.py).
Transformers are tiny real WanModels; the pipeline call is faked, so nothing here needs a TPU.
"""

import ast
import csv
import itertools
import os
import re
import tempfile
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion import aot_cache, max_utils
from maxdiffusion.max_utils import CustomFlashBlockSizes
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
from maxdiffusion.utils import tile_e2e_verify as tev

OLD = CustomFlashBlockSizes(block_q=256, block_kv=256, block_kv_compute=256, block_kv_compute_in=256)
NEW = CustomFlashBlockSizes(block_q=512, block_kv=128, block_kv_compute=128, block_kv_compute_in=128, block_q_sub=128)
BASELINE = {"block_q": 256, "block_kv": 256, "block_kv_compute": 256, "block_kv_compute_in": 256}
# What generate_wan parsers read: the LAST '[tile-search] using' line.
USING_RE = re.compile(
    r"^\[tile-search\] using block_q=(\d+) block_kv=(\d+) block_kv_compute=(\d+) "
    r"block_q_sub=(\d+|auto) block_q_outer=(\d+|none) \(e2e-verified ([\d.]+) s/step\)$"
)
CANDIDATES = [  # proxy-ranked, best first
    {"block_q": 512, "block_kv": 256, "block_kv_compute": 256, "block_q_sub": 128, "block_q_outer": None, "proxy_ms": 1.0},
    {"block_q": 1024, "block_kv": 512, "block_kv_compute": 256, "block_q_sub": None, "block_q_outer": 2, "proxy_ms": 1.1},
    {"block_q": 2048, "block_kv": 512, "block_kv_compute": 512, "block_q_sub": None, "block_q_outer": None, "proxy_ms": 1.2},
]
_UNIQUE = itertools.count()


def _mesh():
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1, 1)), ("data", "fsdp", "tensor"))


def _tiny_wan(block_sizes, rngs=None, scan_layers=True):
  return WanModel(
      rngs=rngs or nnx.Rngs(0),
      num_attention_heads=2,
      attention_head_dim=8,
      in_channels=4,
      out_channels=4,
      text_dim=16,
      freq_dim=8,
      ffn_dim=32,
      num_layers=2,
      rope_max_seq_len=32,
      flash_block_sizes=block_sizes,
      mesh=_mesh(),
      attention="dot_product",
      scan_layers=scan_layers,
  )


def _op_block_sizes(model):
  """Every distinct flash_block_sizes held by a module of `model`."""
  return {
      vars(n)["flash_block_sizes"]
      for _, n in nnx.iter_graph(model)
      if isinstance(n, nnx.Module) and "flash_block_sizes" in vars(n)
  }


class _Config:
  """Mimics pyconfig.HyperParameters: attribute reads (AttributeError if missing), get_keys()."""

  def __init__(self, **keys):
    self._keys = dict(keys)

  def __getattr__(self, name):
    try:
      return self._keys[name]
    except KeyError:
      raise AttributeError(name) from None

  def get_keys(self):
    return self._keys


def _config(**overrides):
  keys = {"attention": "ulysses_ring_custom", "flash_block_sizes": dict(BASELINE)}
  keys.update(overrides)
  return _Config(**keys)


def _wan22_pipeline(config):
  """Stand-in WAN 2.2 pipeline: two tiny transformers built from the config's block sizes."""
  block_sizes = max_utils.get_flash_block_sizes(config)
  return types.SimpleNamespace(
      high_noise_transformer=_tiny_wan(block_sizes, nnx.Rngs(1)),
      low_noise_transformer=_tiny_wan(block_sizes, nnx.Rngs(2)),
  )


class _FakeRunner:
  """run_steps stand-in: checks that every transformer carries one and the same block sizes,
  then returns `seconds_by_bq[block_q] * n` (or raises a VMEM OOM for block_q in `fail_bq`)."""

  def __init__(self, config, pipeline, seconds_by_bq, *, fail_bq=(), compile_on_calls=()):
    self.config, self.pipeline = config, pipeline
    self.seconds_by_bq, self.fail_bq, self.compile_on_calls = seconds_by_bq, set(fail_bq), set(compile_on_calls)
    self.calls = []

  def __call__(self, n):
    sizes = set()
    for attr in tev.pipeline_transformer_attrs(self.pipeline):
      model = getattr(self.pipeline, attr)
      sizes |= _op_block_sizes(model) | {model.config["flash_block_sizes"]}
    assert sizes == {max_utils.get_flash_block_sizes(self.config)}, sizes
    (block_sizes,) = sizes
    self.calls.append((n, block_sizes.block_q))
    if len(self.calls) in self.compile_on_calls:  # a jit-cache miss inside this call
      k = next(_UNIQUE)
      jax.block_until_ready(jax.jit(lambda x: x * 2.0 + k)(jnp.ones(3)))
    if block_sizes.block_q in self.fail_bq:
      raise RuntimeError("RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem.\n" + "HLO dump line\n" * 50)
    return self.seconds_by_bq[block_sizes.block_q] * n


def _verify(config, pipeline, runner, **kwargs):
  logs = []
  kwargs.setdefault("k", 3)
  kwargs.setdefault("steps", 3)
  winner = tev.verify_candidates(config, pipeline, CANDIDATES, run_steps=runner, log=logs.append, **kwargs)
  return winner, logs


def _read_csv(out_dir):
  with open(os.path.join(out_dir, tev.CSV_NAME), newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


class RebuildTest(unittest.TestCase):

  def test_retags_every_attention_op_and_the_config_record(self):
    for scan_layers in (True, False):
      with self.subTest(scan_layers=scan_layers):
        orig = _tiny_wan(OLD, scan_layers=scan_layers)
        clone = tev.rebuild_with_block_sizes(orig, NEW)
        self.assertEqual(_op_block_sizes(clone), {NEW})
        self.assertEqual(clone.config["flash_block_sizes"], NEW)
        self.assertEqual(_op_block_sizes(orig), {OLD})  # the original is untouched
        self.assertEqual(orig.config["flash_block_sizes"], OLD)

  def test_shares_the_weights(self):
    orig = _tiny_wan(OLD)
    clone = tev.rebuild_with_block_sizes(orig, NEW)
    leaves, clone_leaves = jax.tree_util.tree_leaves(nnx.state(orig)), jax.tree_util.tree_leaves(nnx.state(clone))
    self.assertEqual(len(leaves), len(clone_leaves))
    self.assertTrue(all(a is b for a, b in zip(leaves, clone_leaves)))

  def test_has_the_aot_signature_of_a_model_built_with_the_new_sizes(self):
    rngs = nnx.Rngs(0)
    orig, fresh = _tiny_wan(OLD, rngs), _tiny_wan(NEW, rngs)
    clone = tev.rebuild_with_block_sizes(orig, NEW)

    def signature(model):
      return aot_cache._dynamic_signature((nnx.graphdef(model),), {})  # pylint: disable=protected-access

    self.assertEqual(signature(clone), signature(fresh))
    self.assertNotEqual(signature(clone), signature(orig))

  def test_same_sizes_keep_the_jit_cache_key(self):
    orig = _tiny_wan(OLD)
    self.assertEqual(nnx.graphdef(tev.rebuild_with_block_sizes(orig, OLD)), nnx.graphdef(orig))
    self.assertNotEqual(nnx.graphdef(tev.rebuild_with_block_sizes(orig, NEW)), nnx.graphdef(orig))

  def test_jitted_steps_see_the_new_sizes_and_compute_the_same_function(self):
    orig = _tiny_wan(OLD)
    clone = tev.rebuild_with_block_sizes(orig, NEW)
    seen = []

    @jax.jit
    def forward(graphdef, params, rest, inputs):
      model = nnx.merge(graphdef, params, rest)
      seen.append(_op_block_sizes(model))
      return model(*inputs)

    inputs = (
        jax.random.normal(jax.random.key(1), (1, 4, 2, 4, 4)),  # latents (B, C, F, H, W)
        jnp.array([500.0]),  # timestep
        jax.random.normal(jax.random.key(2), (1, 5, 16)),  # text embeddings
    )
    with _mesh():
      out_orig = forward(*nnx.split(orig, nnx.Param, ...), inputs)
      out_clone = forward(*nnx.split(clone, nnx.Param, ...), inputs)
    self.assertEqual(seen, [{OLD}, {NEW}])  # a distinct static graphdef -> its own trace
    np.testing.assert_array_equal(np.asarray(out_orig), np.asarray(out_clone))  # dot_product ignores sizes

  def test_rejects_a_model_without_configured_block_sizes(self):
    with self.assertRaises(ValueError):
      tev.rebuild_with_block_sizes(nnx.Linear(2, 2, rngs=nnx.Rngs(0)), NEW)


class VerifyCandidatesTest(unittest.TestCase):

  def test_commits_the_fastest_e2e_candidate_not_the_proxy_top1(self):
    config = _config()
    pipeline = _wan22_pipeline(config)
    originals = (pipeline.high_noise_transformer, pipeline.low_noise_transformer)
    runner = _FakeRunner(config, pipeline, {512: 5.0, 1024: 4.0, 2048: 4.5})
    with tempfile.TemporaryDirectory() as out_dir:
      winner, logs = _verify(config, pipeline, runner, out_dir=out_dir)
      rows = _read_csv(out_dir)

    self.assertEqual(winner.index, 2)
    self.assertAlmostEqual(winner.s_per_step, 4.0)
    # Untimed pass + timed pass per candidate, each with the full 3-step schedule.
    self.assertEqual(runner.calls, [(3, 512), (3, 512), (3, 1024), (3, 1024), (3, 2048), (3, 2048)])
    # Config: candidate 2 applied on top of the BASELINE (candidate 1's block_q_sub=128 must not leak).
    self.assertNotIn("block_q_sub", config.flash_block_sizes)
    self.assertEqual(
        config.flash_block_sizes,
        max_utils.flash_block_sizes_for_candidate(
            BASELINE, config.attention, 1024, 512, 256, block_q_sub=None, block_q_outer=2
        ),
    )
    expected = max_utils.get_flash_block_sizes(config)
    for model, original in zip((pipeline.high_noise_transformer, pipeline.low_noise_transformer), originals):
      self.assertIsNot(model, original)
      self.assertEqual(_op_block_sizes(model), {expected})
      self.assertEqual(model.config["flash_block_sizes"], expected)
      self.assertEqual(_op_block_sizes(original), {OLD})
    self.assertIn(
        "[tile-e2e] cand 2/3: block_q=1024 block_kv=512 block_kv_compute=256 block_q_sub=auto block_q_outer=2 "
        "proxy=1.10 ms -> 4.000 s/step",
        logs,
    )
    self.assertIn(
        "[tile-e2e] cand 1/3: block_q=512 block_kv=256 block_kv_compute=256 block_q_sub=128 block_q_outer=none "
        "proxy=1.00 ms -> 5.000 s/step",
        logs,
    )
    match = USING_RE.match(logs[-1])
    self.assertIsNotNone(match, logs[-1])
    self.assertEqual(match.groups(), ("1024", "512", "256", "auto", "2", "4.000"))
    self.assertEqual(sum(line.startswith("[tile-search] using") for line in logs), 1)
    self.assertEqual([r["proxy_rank"] for r in rows], ["1", "2", "3"])
    self.assertEqual([r["e2e_rank"] for r in rows], ["3", "1", "2"])
    self.assertEqual([r["winner"] for r in rows], ["False", "True", "False"])
    self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
    self.assertEqual(rows[1]["block_q_outer"], "2")
    self.assertAlmostEqual(float(rows[1]["s_per_step"]), 4.0)
    self.assertAlmostEqual(float(rows[1]["denoise_s"]), 12.0)

  def test_a_failed_candidate_is_recorded_and_never_chosen(self):
    config = _config()
    pipeline = _wan22_pipeline(config)
    runner = _FakeRunner(config, pipeline, {512: 5.0, 1024: 4.0, 2048: 4.5}, fail_bq={1024})
    with tempfile.TemporaryDirectory() as out_dir:
      winner, logs = _verify(config, pipeline, runner, out_dir=out_dir)
      rows = _read_csv(out_dir)
    self.assertEqual(winner.index, 3)
    self.assertEqual(config.flash_block_sizes["block_q"], 2048)
    failed = [line for line in logs if line.startswith("[tile-e2e] cand 2/3")]
    self.assertEqual(len(failed), 1)
    self.assertIn("-> FAILED: RuntimeError: RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem.", failed[0])
    self.assertNotIn("HLO dump", failed[0])
    self.assertEqual([r["status"] for r in rows], ["ok", "failed", "ok"])
    self.assertEqual(rows[1]["e2e_rank"], "")
    self.assertEqual(rows[1]["s_per_step"], "")
    self.assertTrue(USING_RE.match(logs[-1]))

  def test_every_candidate_failing_raises_and_restores_config_and_pipeline(self):
    config = _config()
    pipeline = _wan22_pipeline(config)
    originals = (pipeline.high_noise_transformer, pipeline.low_noise_transformer)
    runner = _FakeRunner(config, pipeline, {}, fail_bq={512, 1024, 2048})
    with self.assertRaisesRegex(RuntimeError, "every tile-size candidate failed"):
      _verify(config, pipeline, runner)
    self.assertEqual(config.flash_block_sizes, BASELINE)
    self.assertIs(pipeline.high_noise_transformer, originals[0])
    self.assertIs(pipeline.low_noise_transformer, originals[1])

  def test_retimes_when_the_timed_pass_compiles(self):
    config = _config()
    pipeline = _wan22_pipeline(config)
    # Call 2 is candidate 1's first timed pass.
    runner = _FakeRunner(config, pipeline, {512: 5.0, 1024: 4.0}, compile_on_calls={2})
    winner, logs = _verify(config, pipeline, runner, k=2)
    self.assertEqual(runner.calls, [(3, 512)] * 3 + [(3, 1024)] * 2)
    self.assertEqual(winner.index, 2)
    self.assertIn("[tile-e2e]   cand 1/2 re-timed: the first timed pass traced/compiled", logs)

  def test_top_k_and_steps(self):
    config = _config()
    pipeline = _wan22_pipeline(config)
    runner = _FakeRunner(config, pipeline, {512: 5.0, 1024: 4.0, 2048: 1.0})
    winner, _ = _verify(config, pipeline, runner, k=2, steps=2)
    self.assertEqual(runner.calls, [(2, 512), (2, 512), (2, 1024), (2, 1024)])  # 2048 is outside the top-2
    self.assertEqual(winner.index, 2)
    with self.assertRaises(ValueError):
      _verify(config, pipeline, runner, steps=0)

  def test_wan21_pipeline_and_a_shared_module(self):
    config = _config()
    shared = _tiny_wan(max_utils.get_flash_block_sizes(config))
    pipeline = types.SimpleNamespace(transformer=shared, high_noise_transformer=shared, low_noise_transformer=None)
    runner = _FakeRunner(config, pipeline, {512: 3.0, 1024: 4.0, 2048: 4.5})
    winner, _ = _verify(config, pipeline, runner)
    self.assertEqual(winner.index, 1)
    self.assertIs(pipeline.transformer, pipeline.high_noise_transformer)  # still one module
    self.assertIsNot(pipeline.transformer, shared)
    self.assertIsNone(pipeline.low_noise_transformer)

  def test_counts_compile_events(self):
    k = next(_UNIQUE)
    with tev._count_compile_events() as events:  # pylint: disable=protected-access
      jax.block_until_ready(jax.jit(lambda x: x - k)(jnp.ones(2)))
    self.assertGreater(events[0], 0)
    seen = events[0]
    jax.block_until_ready(jax.jit(lambda x: x + k)(jnp.ones(2)))  # after exit: not counted
    self.assertEqual(events[0], seen)


class _FakeWanPipeline:
  """Callable pipeline stand-in with the WAN 2.2 transformer attributes and the trace contract."""

  def __init__(self, config, seconds_by_bq):
    block_sizes = max_utils.get_flash_block_sizes(config)
    self.high_noise_transformer = _tiny_wan(block_sizes, nnx.Rngs(1))
    self.low_noise_transformer = _tiny_wan(block_sizes, nnx.Rngs(2))
    self.config, self.seconds_by_bq, self.calls = config, seconds_by_bq, []

  def __call__(self, **kwargs):
    block_q = self.high_noise_transformer.config["flash_block_sizes"].block_q
    self.calls.append(dict(kwargs, enable_profiler=self.config.enable_profiler, block_q=block_q))
    trace = {"conditioning": 0.1, "denoise_total": self.seconds_by_bq[block_q] * kwargs["num_inference_steps"]}
    return ("latents", trace) if kwargs.get("output_type") == "latent" else ("video", trace)


def _wan22_t2v_config(out_dir, **overrides):
  """Every key generate_wan.call_pipeline (WAN 2.2 T2V) and make_pipeline_runner read."""
  keys = {
      "model_name": "wan2.2",
      "model_type": "T2V",
      "prompt": "a cat",
      "negative_prompt": "blurry",
      "global_batch_size_to_train_on": 1,
      "height": 480,
      "width": 832,
      "num_frames": 81,
      "num_inference_steps": 40,
      "guidance_scale_low": 3.0,
      "guidance_scale_high": 4.0,
      "use_cfg_cache": False,
      "use_sen_cache": False,
      "use_kv_cache": False,
      "use_magcache": False,
      "magcache_thresh": 0.0,
      "magcache_K": 0,
      "retention_ratio": 0.0,
      "enable_profiler": True,
      "tile_search_out": out_dir,
  }
  keys.update(overrides)
  return _config(**keys)


class MaybeVerifyTest(unittest.TestCase):

  def test_noop_cases(self):
    config = _config(tile_search_e2e_top_k=3)

    def must_not_run(*args, **kwargs):
      raise AssertionError("pipeline must not run")

    for candidates in (None, [], CANDIDATES[:1]):
      self.assertIsNone(tev.maybe_verify(config, None, candidates, call_pipeline=must_not_run, log=lambda _: None))
    off = _config(tile_search_e2e_top_k=1)
    self.assertIsNone(tev.maybe_verify(off, None, CANDIDATES, call_pipeline=must_not_run, log=lambda _: None))
    self.assertEqual(off.flash_block_sizes, BASELINE)

  def test_with_generate_wan_call_pipeline(self):
    try:
      from maxdiffusion import generate_wan  # pylint: disable=import-outside-toplevel
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"cannot import generate_wan here: {type(e).__name__}: {e}"[:300])
    with tempfile.TemporaryDirectory() as out_dir:
      config = _wan22_t2v_config(out_dir, tile_search_e2e_top_k=2, tile_search_e2e_steps=2)
      pipeline = _FakeWanPipeline(config, {512: 5.0, 1024: 4.0, 2048: 1.0})
      logs = []
      winner = tev.maybe_verify(config, pipeline, CANDIDATES, call_pipeline=generate_wan.call_pipeline, log=logs.append)
      self.assertTrue(os.path.exists(os.path.join(out_dir, tev.CSV_NAME)))
    self.assertEqual(winner.index, 2)
    self.assertEqual([c["block_q"] for c in pipeline.calls], [512, 512, 1024, 1024])
    for call in pipeline.calls:
      self.assertEqual(call["output_type"], "latent")  # no VAE decode
      self.assertEqual(call["num_inference_steps"], 2)
      self.assertEqual(call["prompt"], ["a cat"])
      self.assertEqual(call["negative_prompt"], ["blurry"])
      self.assertFalse(call["enable_profiler"])  # never profile the verification passes
    self.assertTrue(config.enable_profiler)  # restored
    self.assertEqual(config.flash_block_sizes["block_q"], 1024)
    self.assertTrue(USING_RE.match(logs[-1]))

  def test_generate_wan_verifies_after_load_and_before_aot_install(self):
    # Parsed, not imported: the ordering must hold even where generate_wan's deps don't import.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tev.__file__))), "generate_wan.py")
    with open(path, encoding="utf-8") as f:
      (run_def,) = [n for n in ast.parse(f.read()).body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    first_call = {}  # callee source -> first line it is called on
    for node in ast.walk(run_def):
      if isinstance(node, ast.Call):
        callee = ast.unparse(node.func)
        first_call[callee] = min(first_call.get(callee, node.lineno), node.lineno)
    order = [
        first_call["maybe_tune_block_sizes"],
        first_call["lora_loader.load_lora_weights"],
        first_call["tile_e2e_verify.maybe_verify"],
        first_call["aot_cache.install"],
    ]
    self.assertEqual(order, sorted(order))
    self.assertIn("candidates = maybe_tune_block_sizes(config)", ast.unparse(run_def))


if __name__ == "__main__":
  unittest.main()
