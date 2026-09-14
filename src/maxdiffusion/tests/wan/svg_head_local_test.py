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

Synthetic wiring tests on eight CPU devices; these are not TPU benchmarks.
"""

import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxdiffusion.models.wan.transformers.svg_head_local import exchange_local, inference_only

from maxdiffusion.models.wan.transformers import svg_attention as svg


@pytest.mark.parametrize("routing", ["mixed", "spatial", "temporal"])
def test_exchange_preserves_head_routes_and_attention(routing):
  if len(jax.devices()) != 8 or any(d.platform != "cpu" for d in jax.devices()):
    pytest.skip("Requires eight CPU devices")
  mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("data", "context"))
  ps = P("data", None, "context", None)
  grid = (3, 2, 4)
  rng = np.random.default_rng(41)
  inputs = tuple(
      jax.device_put(rng.normal(size=(4, 8, 24, 4)).astype(np.float32), NamedSharding(mesh, ps)) for _ in range(3)
  )
  # Different patterns within and between head blocks and data replicas.
  route = np.arange(32).reshape(4, 8) % 3 == 1
  if routing != "mixed":
    route[:] = routing == "temporal"
  route = jax.device_put(route, NamedSharding(mesh, P("data", None)))
  place = partial(svg.svg_placement_permute, token_grid=grid)
  restore = partial(svg.svg_placement_unpermute, token_grid=grid)

  def core(q, k, v):
    # Position-dependent attention makes missing/wrong placement observable.
    score = jnp.einsum("bhid,bhjd->bhij", q, k) * 0.5
    pos = jnp.arange(q.shape[2])
    mask = jnp.abs(pos[:, None] - pos[None, :]) <= 3
    return jnp.einsum("bhij,bhjd->bhid", jax.nn.softmax(jnp.where(mask, score, -jnp.inf)), v)

  def reference(q, k, v, r):
    return restore(core(*place(q, k, v, r)), r)

  def candidate(q, k, v, r):
    return exchange_local(
        q, k, v, r, mesh=mesh, qspec=ps, kvspec=ps, ulysses_axis="context", place=place, restore=restore, core=core
    )

  expected = jax.jit(reference)(*inputs, route)
  executable = jax.jit(candidate).lower(*inputs, route).compile()
  actual = executable(*inputs, route)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  assert actual.sharding.is_equivalent_to(NamedSharding(mesh, ps), 4)
  assert "all-to-all" in executable.as_text().lower()
  if routing == "mixed":
    # Sensitivity control: wrong routing must not accidentally pass this test.
    wrong = jax.jit(reference)(*inputs, jnp.logical_not(route))
    assert np.max(np.abs(np.asarray(actual - wrong))) > 0.1


def test_reject_head_sharded_inputs():
  with pytest.raises(ValueError, match="unsharded heads"):
    exchange_local(
        None,
        None,
        None,
        None,
        mesh=None,
        qspec=P("data", "ulysses", None, None),
        kvspec=P("data", "ulysses", None, None),
        ulysses_axis="context",
        place=None,
        restore=None,
        core=None,
    )


@pytest.mark.parametrize("transform", [jax.grad, lambda f: lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))])
def test_inference_rejects_derivatives(transform):
  with pytest.raises(NotImplementedError, match="inference only"):
    transform(lambda x: inference_only(x).sum())(jnp.ones((2,)))


def test_static_inactive_layer_skips_dynamic_step():
  def call(step):
    assert svg.is_svg_active(step, 0, start_step=0, end_step=40, start_layer=1, end_layer=40) is False
    return step

  assert jax.jit(call)(3) == 3


@pytest.mark.parametrize("base2", [False, True])
def test_pallas_scratch_matches_dense(monkeypatch, base2):
  from jax.experimental import pallas as pl
  from maxdiffusion.kernels import custom_svg_attention_dispatch as dispatch
  from maxdiffusion.kernels import custom_svg_static_range_attention as kernel

  original = pl.pallas_call
  monkeypatch.setattr(pl, "pallas_call", lambda *a, **kw: original(*a, **dict(kw, interpret=True)))
  rng = np.random.default_rng(12)
  n = 257
  q, k, v = [jnp.asarray(rng.normal(size=(2, 384, 128)).astype(np.float32) * 0.1) for _ in range(3)]
  # Padding must not contribute, even when its values dominate real tokens.
  k = k.at[:, n:].set(100)
  v = v.at[:, n:].set(100)
  blocks = kernel.SVGBlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
  call = dispatch.make_svg_static_range_mha(
      block_sizes=blocks,
      orig_q_seq_len=n,
      orig_kv_seq_len=n,
      band_width=n,
      frame_size=1,
      use_base2_exp=base2,
  )
  # TPU lowers float32 matmuls to bf16 passes by default, which perturbs the
  # reference below by ~1e-2 and would make this exactness check test the
  # hardware default rather than the kernel. Pin precision instead of loosening
  # the tolerance; on CPU this is already the effective behaviour.
  with jax.default_matmul_precision("highest"):
    actual = call(q * (np.log2(np.e) if base2 else 1), k, v).transpose(0, 2, 1)
    scores = jnp.einsum("hid,hjd->hij", q[:, :n], k[:, :n])
    expected = jnp.einsum("hij,hjd->hid", jax.nn.softmax(scores), v[:, :n])
  np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# --- End-to-end dispatcher tests ------------------------------------------
#
# These drive `_apply_attention` itself, so they cover the parts the unit
# tests above cannot: the SVG gate, logical->mesh axis resolution, the
# all_to_all, route sharding, flash padding and the final unpad. Pallas runs
# in interpret mode, so they check semantics only, never performance.

_AXIS_RULES = (
    ("activation_batch", "data"),
    ("activation_length", "context"),
    ("activation_heads", None),
    ("activation_kv", None),
    ("activation_self_attn_heads", None),
    ("activation_self_attn_q_length", "context"),
    ("activation_self_attn_kv_length", "context"),
)
_GRID = (5, 2, 4)  # 40 tokens: not a multiple of the 16-wide q block.
_HEADS, _DIM = 8, 4  # dim < 128 forces head-dim padding inside the kernel.


def _axis_names():
  from maxdiffusion.common_types import BATCH, D_KV, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, SELF_ATTN_Q_LENGTH

  return (BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV), (BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV)


def _interpret_pallas(monkeypatch):
  from jax.experimental import pallas as pl

  original = pl.pallas_call
  monkeypatch.setattr(pl, "pallas_call", lambda *a, **kw: original(*a, **dict(kw, interpret=True)))


def _apply_svg(
    monkeypatch,
    mesh,
    qkv,
    route,
    *,
    band_width,
    scale,
    base2=False,
    block=16,
    lower=False,
    kernel="ulysses_custom",
    ulysses_shards=None,
    **overrides,
):
  """Run the production dispatcher on the head-local SVG path."""
  import flax.linen as nn
  from maxdiffusion.models import attention_flax

  monkeypatch.setattr(svg, "svg_profile_temporal_heads", lambda *a, **kw: jnp.asarray(route))
  axis_names_q, axis_names_kv = _axis_names()
  cfg = {
      "use_svg_attention": True,
      "profile_seed": 0,
      "profile_query_count": 4,
      "band_width": band_width,
      "custom_flash_block_sizes": {
          "block_q": block,
          "block_kv": block,
          "block_kv_compute": block,
          "block_kv_compute_in": block,
          "heads_per_tile": 1,
      },
  }
  cfg.update(overrides)

  def run(q, k, v):
    return attention_flax._apply_attention(
        query=q,
        key=k,
        value=v,
        heads=_HEADS,
        dim_head=_DIM,
        split_head_dim=True,
        float32_qk_product=False,
        attention_kernel=kernel,
        flash_min_seq_length=0,
        use_memory_efficient_attention=False,
        scale=scale,
        dtype=jnp.float32,
        mesh=mesh,
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        flash_block_sizes=None,
        dpa_layer=None,
        use_base2_exp=base2,
        ulysses_shards=mesh.shape["context"] if ulysses_shards is None else ulysses_shards,
        spatiotemporal_config=cfg,
        spatiotemporal_shape=_GRID,
    )

  with mesh, nn.logical_axis_rules(_AXIS_RULES):
    if lower:
      return jax.jit(run).lower(*qkv).as_text(debug_info=True)
    return jax.jit(run)(*qkv)


def _dense_reference(q, k, v, scale):
  scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
  out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)
  return jnp.transpose(out, (0, 2, 1, 3)).reshape(out.shape[0], out.shape[2], -1)


def _random_qkv(batch, seed=7):
  rng = np.random.default_rng(seed)
  n = int(np.prod(_GRID))
  return tuple(jnp.asarray(rng.normal(size=(batch, _HEADS, n, _DIM)).astype(np.float32)) for _ in range(3))


@pytest.mark.parametrize("base2", [False, True])
def test_dispatcher_full_support_matches_dense(monkeypatch, base2):
  """Density 1.0 must reproduce plain softmax attention exactly.

  Full support makes place/restore an exact inverse pair, so any surviving
  difference is a real defect: a missing or doubled logit scale, padding
  tokens leaking into the softmax, or a mis-stitched all_to_all.
  """
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("data", "context"))
  q, k, v = _random_qkv(2)
  route = np.arange(16).reshape(2, 8) % 3 == 1
  # See test_pallas_scratch_matches_dense: TPU's default bf16 matmul passes
  # perturb the dense reference well past this tolerance, so pin the precision
  # rather than weaken the exactness requirement.
  with jax.default_matmul_precision("highest"):
    actual = _apply_svg(monkeypatch, mesh, (q, k, v), route, band_width=int(np.prod(_GRID)), scale=0.37, base2=base2)
    expected = _dense_reference(q, k, v, 0.37)
  np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("routing", ["mixed", "spatial", "temporal"])
def test_dispatcher_sparse_is_sharding_invariant(monkeypatch, routing):
  """Head-local execution must not depend on how heads are distributed.

  The band is narrow and the sequence is ragged, so this exercises partial
  tiles, the union tail and the log-sum-exp merge under a real all_to_all.
  """
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  route = np.arange(8).reshape(1, 8) % 3 == 1
  if routing != "mixed":
    route[:] = routing == "temporal"
  qkv = _random_qkv(1, seed=19)
  sharded = Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "context"))
  single = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "context"))
  kwargs = {"band_width": 8, "scale": 0.25}
  actual = _apply_svg(monkeypatch, sharded, qkv, route, **kwargs)
  expected = _apply_svg(monkeypatch, single, qkv, route, **kwargs)
  # A narrow band must actually drop mass; otherwise this test is vacuous.
  assert np.max(np.abs(np.asarray(expected - _dense_reference(*qkv, 0.25)))) > 1e-3
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_dispatcher_batch_fold_matches_unfolded(monkeypatch):
  """Folding batch into heads is a pure layout change, not a semantic one."""
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  qkv = _random_qkv(2, seed=23)
  route = np.arange(16).reshape(2, 8) % 4 < 2
  kwargs = {"band_width": 8, "scale": 0.25}
  # data=1 folds (batch>1, nothing shards batch); data=2 cannot.
  folded = _apply_svg(monkeypatch, Mesh(np.array(jax.devices()).reshape(1, 8), ("data", "context")), qkv, route, **kwargs)
  unfolded = _apply_svg(monkeypatch, Mesh(np.array(jax.devices()).reshape(2, 4), ("data", "context")), qkv, route, **kwargs)
  np.testing.assert_allclose(folded, unfolded, rtol=1e-5, atol=1e-5)
  # The two batch entries have different routes, so folding must not mix them.
  assert np.max(np.abs(np.asarray(folded[0] - folded[1]))) > 1e-3


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"attention_kernel": "flash"}, "custom Ulysses attention backend"),
        ({"ulysses_attention_chunks": 2}, "chunked Ulysses"),
        ({"global_stride": 4}, "external or periodic masks"),
        ({"spatiotemporal_shape": None}, "matched self-attention QKV"),
        ({"heads_per_tile": 2}, "heads_per_tile=1"),
    ],
)
def test_dispatcher_rejects_unsupported_configurations(monkeypatch, overrides, error):
  """Unsupported combinations must fail loudly rather than silently run dense."""
  import flax.linen as nn
  from maxdiffusion.models import attention_flax

  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  monkeypatch.setattr(svg, "svg_profile_temporal_heads", lambda *a, **kw: jnp.zeros((1, _HEADS), bool))
  mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("data", "context"))
  axis_names_q, axis_names_kv = _axis_names()
  block = {"block_q": 16, "block_kv": 16, "block_kv_compute": 16, "block_kv_compute_in": 16, "heads_per_tile": 1}
  if "heads_per_tile" in overrides:
    block["heads_per_tile"] = overrides.pop("heads_per_tile")
  cfg = {
      "use_svg_attention": True,
      "profile_seed": 0,
      "profile_query_count": 4,
      "band_width": 8,
      "custom_flash_block_sizes": block,
  }
  cfg.update({key: overrides.pop(key) for key in ("global_stride",) if key in overrides})
  kwargs = {
      "heads": _HEADS,
      "dim_head": _DIM,
      "split_head_dim": True,
      "float32_qk_product": False,
      "attention_kernel": "ulysses_custom",
      "flash_min_seq_length": 0,
      "use_memory_efficient_attention": False,
      "scale": 0.25,
      "dtype": jnp.float32,
      "mesh": mesh,
      "axis_names_q": axis_names_q,
      "axis_names_kv": axis_names_kv,
      "flash_block_sizes": None,
      "dpa_layer": None,
      "ulysses_shards": 4,
      "spatiotemporal_config": cfg,
      "spatiotemporal_shape": _GRID,
  }
  kwargs.update(overrides)
  q, k, v = _random_qkv(1)
  with mesh, nn.logical_axis_rules(_AXIS_RULES):
    with pytest.raises(ValueError, match=error):
      jax.eval_shape(lambda a, b, c: attention_flax._apply_attention(query=a, key=b, value=c, **kwargs), q, k, v)


def test_realized_density_is_recorded_in_profile_scope(monkeypatch):
  """A profile must show the realized density, not just that SVG was entered.

  Density is not the requested fraction: tile rounding, the anchor and ragged
  edges all move it. Recording the executed tile count is what makes an XProf
  trace sufficient to prove the sparse path really ran sparsely.
  """
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "context"))
  qkv = _random_qkv(1, seed=31)
  route = np.zeros((1, _HEADS), bool)
  sparse = _apply_svg(monkeypatch, mesh, qkv, route, band_width=8, scale=0.25, lower=True)
  dense = _apply_svg(monkeypatch, mesh, qkv, route, band_width=int(np.prod(_GRID)), scale=0.25, lower=True)
  pattern = r"svg_kernel_c_tiles(\d+)of(\d+)_d[0-9.]+"
  (sparse_exec, total), *_ = re.findall(pattern, sparse) or [(None, None)]
  (dense_exec, _), *_ = re.findall(pattern, dense) or [(None, None)]
  assert sparse_exec and dense_exec, "realized density missing from the profile scope"
  assert int(dense_exec) == int(total), "full support must execute every tile"
  assert int(sparse_exec) < int(dense_exec), "a narrow band must execute fewer tiles"


def test_dynamic_step_index_stays_traced():
  """A traced step index must yield a traced predicate, not a Python bool.

  The step index cannot be static: it is a non-static argument of the jitted
  transformer pass, so only the layer dimension can short-circuit statically.
  """
  captured = {}

  def call(step):
    captured["active"] = svg.is_svg_active(step, 3, start_step=0, end_step=40, start_layer=1, end_layer=40)
    return step

  jax.jit(call)(3)
  assert isinstance(captured["active"], jax.Array) and not isinstance(captured["active"], bool)


@pytest.mark.parametrize("kernel", ["ulysses_ring_custom", "ulysses_ring_custom_fixed_m"])
def test_svg_runs_under_a_ring_configured_dense_backend(monkeypatch, kernel):
  """SVG must not force the dense arm to give up its ring split.

  The validated recipe pairs a Ring2/Ulysses2 dense arm with pure-Ulysses
  sparse attention. `ulysses_shards` configures only the dense arm, which
  re-meshes the context axis inside its own shard_map; head-local SVG always
  exchanges over the whole context axis. A gate demanding
  `ulysses_shards == context` rejected that recipe outright, and because
  lax.cond traces both branches it did so even on dense-scheduled steps.
  """
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "context"))
  qkv = _random_qkv(1, seed=17)
  route = np.arange(_HEADS).reshape(1, _HEADS) % 2 == 0
  common = {"band_width": 8, "scale": 0.31, "kernel": kernel}
  ring2 = _apply_svg(monkeypatch, mesh, qkv, route, ulysses_shards=2, **common)
  ring1 = _apply_svg(monkeypatch, mesh, qkv, route, ulysses_shards=4, **common)
  # Same sparse result either way: proof that the sparse arm ignores the dense
  # arm's ring/ulysses split rather than merely tolerating it.
  np.testing.assert_array_equal(np.asarray(ring2), np.asarray(ring1))
  reference = _apply_svg(monkeypatch, mesh, qkv, route, band_width=8, scale=0.31, kernel="ulysses_custom")
  np.testing.assert_allclose(np.asarray(ring2), np.asarray(reference), rtol=2e-5, atol=2e-5)


def test_svg_block_sizes_override_the_dense_tiling(monkeypatch):
  """A configured SVG tiling must reach the kernel and change its geometry."""
  if len(jax.devices()) != 8:
    pytest.skip("Requires eight devices")
  _interpret_pallas(monkeypatch)
  mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "context"))
  qkv = _random_qkv(1, seed=23)
  route = np.zeros((1, _HEADS), bool)
  pattern = r"svg_kernel_c_tiles(\d+)of(\d+)_d[0-9.]+"

  def tiles(block):
    text = _apply_svg(monkeypatch, mesh, qkv, route, band_width=8, scale=0.25, block=block, lower=True)
    found = re.findall(pattern, text)
    assert found, "sparse kernel did not run"
    return tuple(int(x) for x in found[0])

  coarse_exec, coarse_total = tiles(16)
  fine_exec, fine_total = tiles(8)
  assert fine_total > coarse_total, "a smaller tile must produce more tiles overall"
  assert fine_exec != coarse_exec, "the configured tiling must change what the kernel executes"
