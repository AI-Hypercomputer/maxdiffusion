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

Tests for Sol-Attn (arXiv:2607.24027): routing math, the dense reference, the
Pallas TPU kernel, sharded execution, and model/pipeline config plumbing.
"""

import dataclasses
from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy.stats import norm

from maxdiffusion import aot_cache
from maxdiffusion import common_types as ct
from maxdiffusion.kernels.sol_attention import pooling
from maxdiffusion.kernels.sol_attention import reference
from maxdiffusion.kernels.sol_attention import routing
from maxdiffusion.kernels.sol_attention import sol_attention_kernel
from maxdiffusion.models import sol_attention

ON_TPU = jax.devices()[0].platform == "tpu"
requires_tpu = pytest.mark.skipif(not ON_TPU, reason="Pallas TPU kernel requires a TPU.")


def _rel_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _video_like_qkv(heads, seq_len, head_dim, *, seed=0, smooth=64, noise=0.5, dtype=jnp.float32):
  """Tokens that are locally correlated (like video latents) so pooling matters."""
  k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
  base = jax.random.normal(k1, (heads, seq_len // smooth + 1, head_dim))
  base = jnp.repeat(base, smooth, axis=1)[:, :seq_len]
  q = base + noise * jax.random.normal(k2, (heads, seq_len, head_dim))
  k = base + noise * jax.random.normal(k3, (heads, seq_len, head_dim))
  v = jax.random.normal(k4, (heads, seq_len, head_dim))
  return q.astype(dtype), k.astype(dtype), v.astype(dtype)


# --- Routing -----------------------------------------------------------------


def test_block_sizes_validation():
  with pytest.raises(ValueError, match="multiple of 128"):
    routing.SolBlockSizes(block_q=96)
  assert routing.SolBlockSizes(block_q=256, block_kv=384).seq_multiple == 768
  assert routing.padded_length(1000, routing.SolBlockSizes(256, 128)) == 1024


def test_pool_blocks_ignores_padding():
  x = jnp.arange(5 * 2, dtype=jnp.float32).reshape(1, 5, 2)
  # Pad 5 tokens to 8 with block 4: block 0 averages tokens 0..3, block 1 only token 4.
  pooled = routing.pool_blocks(routing.pad_sequence(x, 8), seq_len=5, block=4)
  np.testing.assert_allclose(pooled[0, 0], x[0, :4].mean(axis=0))
  np.testing.assert_allclose(pooled[0, 1], x[0, 4])
  np.testing.assert_array_equal(routing.block_counts(5, 8, 4), [4, 1])


def test_threshold_matches_pooled_key_moments():
  """Row mean/std equal the pooled-key moment form of Eq. (5)."""
  rng = np.random.default_rng(0)
  q_pooled = rng.normal(size=(3, 7, 16))
  k_pooled = rng.normal(size=(3, 11, 16))
  scale = 0.25
  scores = routing.block_proxy_scores(jnp.asarray(q_pooled), jnp.asarray(k_pooled), scale)
  _, tau = routing.threshold_route(scores, jnp.full((11,), 128), beta=1.3)

  qs = q_pooled * scale
  mean_k = k_pooled.mean(axis=1)  # [H, d]
  second_k = np.einsum("hkd,hke->hde", k_pooled, k_pooled) / k_pooled.shape[1]  # [H, d, d]
  mu = np.einsum("hqd,hd->hq", qs, mean_k)
  var = np.einsum("hqd,hde,hqe->hq", qs, second_k, qs) - mu**2
  np.testing.assert_allclose(tau, mu + 1.3 * np.sqrt(var), rtol=1e-5, atol=1e-5)


def test_threshold_density_tracks_gaussian_tail():
  """For Gaussian proxy logits the selected density is ~ 1 - Phi(beta) (Fig. 3)."""
  scores = jax.random.normal(jax.random.PRNGKey(1), (4, 64, 2048))
  counts = jnp.full((2048,), 128)
  for beta in (0.5, 1.0, 1.5):
    selected, _ = routing.threshold_route(scores, counts, beta)
    assert abs(float(selected.mean()) - (1.0 - norm.cdf(beta))) < 0.01


def test_threshold_skips_padding_and_honors_always_exact():
  scores = jnp.array([[[5.0, 0.0, 0.0, 9.0]]])
  counts = jnp.array([128, 128, 128, 0])  # last block is pure padding
  selected, _ = routing.threshold_route(scores, counts, beta=1.0)
  np.testing.assert_array_equal(selected[0, 0], [True, False, False, False])
  always = jnp.array([False, True, False, True])
  selected, _ = routing.threshold_route(scores, counts, beta=1.0, always_exact=always)
  np.testing.assert_array_equal(selected[0, 0], [True, True, False, False])


def test_compact_selection_layout():
  selected = jnp.array([[[False, True, False, True, True], [False] * 5]])
  rows = routing.compact_selection(selected, row_width=8)
  assert rows.shape == (1, 2, 8)
  np.testing.assert_array_equal(rows[0, 0, :4], [3, 1, 3, 4])
  assert int(rows[0, 1, 0]) == 0


# --- Dense reference ---------------------------------------------------------


def test_reference_selecting_everything_is_dense():
  q, k, v = _video_like_qkv(2, 300, 32)
  bs = routing.SolBlockSizes(128, 128)
  nq = nk = routing.padded_length(300, bs) // 128
  out = reference.sol_attention_reference(
      q, k, v, scale=32**-0.5, beta=0.0, block_sizes=bs, selected=jnp.ones((2, nq, nk), bool)
  )
  np.testing.assert_allclose(out, reference.dense_attention_reference(q, k, v, scale=32**-0.5), atol=1e-5)


def test_zeroth_order_term_is_exact_for_constant_key_blocks():
  """With constant keys per block, Eqs. (9)-(10) reproduce dense attention even
  when every block is approximated, including the partial last block."""
  seq_len, block = 300, 128
  rng = np.random.default_rng(3)
  block_keys = rng.normal(size=(2, 3, 16)).astype(np.float32)
  k = jnp.asarray(np.repeat(block_keys, block, axis=1)[:, :seq_len])
  q = jnp.asarray(rng.normal(size=(2, seq_len, 16)).astype(np.float32))
  v = jnp.asarray(rng.normal(size=(2, seq_len, 16)).astype(np.float32))
  bs = routing.SolBlockSizes(128, 128)
  out = reference.sol_attention_reference(q, k, v, scale=0.3, beta=0.0, block_sizes=bs, selected=jnp.zeros((2, 3, 3), bool))
  np.testing.assert_allclose(out, reference.dense_attention_reference(q, k, v, scale=0.3), atol=1e-5)


def test_approximate_correction_beats_keep_or_drop():
  """Same routing, with and without the pooled correction (Fig. 9)."""
  q, k, v = _video_like_qkv(2, 1024, 64, seed=5, noise=0.3)
  q = q * 3.0  # sharper attention, so dropping blocks actually hurts
  dense = reference.dense_attention_reference(q, k, v, scale=64**-0.5)
  bs = routing.SolBlockSizes(128, 128)
  for beta in (0.5, 1.0, 1.5):
    sol = reference.sol_attention_reference(q, k, v, scale=64**-0.5, beta=beta, block_sizes=bs)
    drop = reference.sol_attention_reference(q, k, v, scale=64**-0.5, beta=beta, block_sizes=bs, approximate=False)
    assert _rel_l2(sol, dense) < _rel_l2(drop, dense)


# --- Pallas kernel -----------------------------------------------------------


@requires_tpu
@pytest.mark.parametrize(
    "heads,seq_len,block_q,block_kv,beta",
    [
        (2, 1000, 128, 128, 1.0),
        (3, 3000, 256, 128, 1.0),
        (2, 4100, 256, 256, 0.5),
        (2, 2048, 512, 128, 1.5),
    ],
)
def test_kernel_matches_reference(heads, seq_len, block_q, block_kv, beta):
  q, k, v = _video_like_qkv(heads, seq_len, 128, dtype=jnp.bfloat16)
  bs = routing.SolBlockSizes(block_q, block_kv)
  scale = 128**-0.5
  padded = routing.padded_length(seq_len, bs)
  qp, kp, vp = (routing.pad_sequence(x, padded) for x in (q, k, v))
  # Route once and feed both implementations so threshold ties cannot differ.
  route = routing.compute_routing(qp, kp, vp, seq_len=seq_len, scale=scale, beta=beta, block_sizes=bs)
  out = sol_attention_kernel.sol_attention_from_routing(qp, kp, vp, route, seq_len=seq_len, scale=scale, block_sizes=bs)
  want = reference.sol_attention_reference(q, k, v, scale=scale, beta=beta, block_sizes=bs, selected=route.selected)
  assert 0.0 < float(route.density()) < 1.0
  assert _rel_l2(out[:, :seq_len], want) < 1e-2


@requires_tpu
@pytest.mark.parametrize("group_size,num_buffers", [(1, 2), (2, 4), (3, 6), (4, 12)])
def test_kernel_pipelining_variants(group_size, num_buffers):
  """Grouped hot loop, leftover blocks, and the ragged tail epilogue."""
  q, k, v = _video_like_qkv(2, 2500, 128, seed=3, dtype=jnp.bfloat16)
  bs = routing.SolBlockSizes(256, 128)
  padded = routing.padded_length(2500, bs)
  qp, kp, vp = (routing.pad_sequence(x, padded) for x in (q, k, v))
  route = routing.compute_routing(qp, kp, vp, seq_len=2500, scale=0.09, beta=0.3, block_sizes=bs)
  # Force the ragged tail block into every row so the epilogue always runs.
  sel = route.selected.at[:, :, 2500 // 128].set(True)
  route = dataclasses.replace(route, selected=sel)
  out = sol_attention_kernel.sol_attention_from_routing(
      qp, kp, vp, route, seq_len=2500, scale=0.09, block_sizes=bs, group_size=group_size, num_buffers=num_buffers
  )
  want = reference.sol_attention_reference(q, k, v, scale=0.09, beta=0.3, block_sizes=bs, selected=sel)
  assert _rel_l2(out[:, :2500], want) < 1e-2


def test_invalid_pipelining_is_rejected():
  with pytest.raises(ValueError, match="multiple of group_size"):
    sol_attention_kernel._resolve_pipelining(128, group_size=4, num_buffers=6)  # pylint: disable=protected-access
  assert sol_attention_kernel._resolve_pipelining(256, None, None) == (4, 8)  # pylint: disable=protected-access


@requires_tpu
def test_kernel_handles_empty_and_full_selection():
  q, k, v = _video_like_qkv(1, 700, 128, dtype=jnp.bfloat16)
  bs = routing.SolBlockSizes(128, 128)
  padded = routing.padded_length(700, bs)
  qp, kp, vp = (routing.pad_sequence(x, padded) for x in (q, k, v))
  route = routing.compute_routing(qp, kp, vp, seq_len=700, scale=0.1, beta=1.0, block_sizes=bs)
  for fill in (False, True):
    sel = jnp.full_like(route.selected, fill)
    forced = dataclasses.replace(route, selected=sel)
    out = sol_attention_kernel.sol_attention_from_routing(qp, kp, vp, forced, seq_len=700, scale=0.1, block_sizes=bs)
    want = reference.sol_attention_reference(q, k, v, scale=0.1, beta=1.0, block_sizes=bs, selected=sel)
    assert _rel_l2(out[:, :700], want) < 1e-2


@requires_tpu
@pytest.mark.parametrize("seq_len,block_q,block_kv", [(9000, 256, 128), (20000, 1024, 256)])
def test_unpadded_path_matches_reference(seq_len, block_q, block_kv):
  """Fused pooling + unpadded kernel (ragged tail read as a shifted window)."""
  q, k, v = _video_like_qkv(2, seq_len, 128, seed=11, dtype=jnp.bfloat16)
  bs = routing.SolBlockSizes(block_q, block_kv)
  route = sol_attention_kernel.compute_routing_unpadded(q, k, v, scale=128**-0.5, beta=1.0, block_sizes=bs)
  out = sol_attention_kernel.sol_attention_from_routing(q, k, v, route, seq_len=seq_len, scale=128**-0.5, block_sizes=bs)
  padded = routing.padded_length(seq_len, bs)
  nq, nk = route.selected.shape[1:]
  sel = jnp.pad(route.selected, ((0, 0), (0, padded // block_q - nq), (0, padded // block_kv - nk)))
  want = reference.sol_attention_reference(q, k, v, scale=128**-0.5, beta=1.0, block_sizes=bs, selected=sel)
  assert _rel_l2(out, want) < 1e-2


@requires_tpu
def test_fused_pooling_matches_reference_pooling():
  q, k, v = _video_like_qkv(2, 9000, 128, seed=2, dtype=jnp.bfloat16)
  q_pooled, k_pooled, v_pooled, k_radius = pooling.pool_qkv(q, k, v, block_q=256, block_kv=128)
  padded = routing.padded_length(9000, routing.SolBlockSizes(256, 128))
  qp, kp, vp = (routing.pad_sequence(x, padded) for x in (q, k, v))
  want_k = routing.pool_blocks(kp, 9000, 128)[:, : k_pooled.shape[1]]
  np.testing.assert_allclose(q_pooled, routing.pool_blocks(qp, 9000, 256)[:, : q_pooled.shape[1]], rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(k_pooled, want_k, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(v_pooled, routing.pool_blocks(vp, 9000, 128)[:, : v_pooled.shape[1]], rtol=1e-5, atol=1e-5)
  want_r = routing.block_radius(kp, routing.pool_blocks(kp, 9000, 128), 9000, 128)[:, : k_radius.shape[1]]
  np.testing.assert_allclose(k_radius, want_r, rtol=1e-4, atol=1e-4)


@requires_tpu
def test_fixed_offset_survives_outlier_keys():
  """A few keys far above their block mean must neither overflow nor be lost."""
  q, k, v = _video_like_qkv(1, 2048, 128, seed=4, dtype=jnp.float32)
  direction = q[0, 100] / jnp.linalg.norm(q[0, 100])
  k = k.at[0, 77].set(60.0 * direction).at[0, 1500].set(-60.0 * direction)
  q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))
  bs = routing.SolBlockSizes(256, 128)
  route = routing.compute_routing(q, k, v, seq_len=2048, scale=0.3, beta=1.0, block_sizes=bs)
  out = sol_attention_kernel.sol_attention_from_routing(q, k, v, route, seq_len=2048, scale=0.3, block_sizes=bs)
  want = reference.sol_attention_reference(q, k, v, scale=0.3, beta=1.0, block_sizes=bs, selected=route.selected)
  assert np.all(np.isfinite(np.asarray(out, np.float32)))
  assert _rel_l2(out, want) < 1e-2


@requires_tpu
@pytest.mark.parametrize("head_chunks", [1, 2])
def test_sharded_ulysses_matches_single_device(head_chunks):
  devices = jax.devices()
  if len(devices) < 2:
    pytest.skip("Requires multiple TPU devices.")
  mesh = Mesh(np.array(devices).reshape(1, -1), ("data", "context"))
  cp = len(devices)
  batch, heads, seq_len, head_dim = 2, 2 * cp, 128 * cp * 3, 128
  rng = jax.random.PRNGKey(7)
  q, k, v = (jax.random.normal(r, (batch, heads, seq_len, head_dim), jnp.bfloat16) for r in jax.random.split(rng, 3))
  cfg = sol_attention.sparse_config(
      SimpleNamespace(
          sol_beta=1.0,
          sol_block_q=256,
          sol_block_kv=128,
          sol_vmem_limit_bytes=None,
          sol_log_density=False,
      )
  )
  context = {
      "spatiotemporal_config": cfg,
      "heads": heads,
      "mesh": mesh,
      "scale": head_dim**-0.5,
      "attention_mask": None,
      "ulysses_attention_chunks": head_chunks,
      "axis_names_q": (ct.BATCH, ct.SELF_ATTN_HEAD, ct.SELF_ATTN_Q_LENGTH, ct.D_KV),
      "axis_names_kv": (ct.BATCH, ct.SELF_ATTN_HEAD, ct.SELF_ATTN_KV_LENGTH, ct.D_KV),
  }
  rules = tuple(tuple(r) for r in ct.ULYSSES_ATTENTION_AXIS_RULES) + (
      ("activation_batch", "data"),
      ("activation_length", "context"),
      ("activation_heads", None),
  )
  spec = NamedSharding(mesh, P("data", None, "context", None))
  with mesh, nn.logical_axis_rules(rules):
    sharded = jax.jit(lambda q, k, v: sol_attention.sol_sharded_attention(q, k, v, context))(
        *(jax.device_put(x, spec) for x in (q, k, v))
    )
  local = sol_attention_kernel.sol_attention(
      q, k, v, scale=head_dim**-0.5, beta=1.0, block_sizes=routing.SolBlockSizes(256, 128)
  )
  local = local.transpose(0, 2, 1, 3).reshape(batch, seq_len, heads * head_dim)
  np.testing.assert_allclose(np.asarray(sharded, np.float32), np.asarray(local, np.float32), atol=2e-2, rtol=2e-2)


# --- Config plumbing ---------------------------------------------------------


def test_init_sol_config_defaults_when_disabled():
  cfg = sol_attention.init_sol_config({"sol_beta": 2.0}, default_num_layers=30)
  assert cfg["use_sol_attention"] is False
  assert cfg["sol_beta"] == sol_attention.SOL_ATTENTION_DEFAULTS["sol_beta"]
  assert cfg["sol_num_layers"] == 30
  # Cross-attention never uses Sol.
  cfg = sol_attention.init_sol_config({"use_sol_attention": True}, 30, is_self_attention=False)
  assert cfg["use_sol_attention"] is False


def test_init_sol_config_validation():
  cfg = sol_attention.init_sol_config({"use_sol_attention": True, "sol_beta": 1.5, "sol_block_q": 512}, 40)
  assert cfg["use_sol_attention"] and cfg["sol_beta"] == 1.5 and cfg["sol_block_q"] == 512
  with pytest.raises(ValueError, match="mutually exclusive"):
    sol_attention.init_sol_config({"use_sol_attention": True, "use_svg_attention": True}, 40)
  with pytest.raises(ValueError, match="multiple of 128"):
    sol_attention.init_sol_config({"use_sol_attention": True, "sol_block_kv": 64}, 40)


def test_apply_sol_or_dense_follows_warmup_schedule():
  sol = SimpleNamespace(**sol_attention.init_sol_config({"use_sol_attention": True}, 40))
  sol.sol_active_start_step, sol.sol_active_end_step = 10, 50
  sol.sol_active_start_layer, sol.sol_active_end_layer = 1, 40
  calls = []

  def run_attention(**kwargs):
    calls.append(kwargs.get("sparse_config_override"))
    return jnp.zeros(())

  sol_attention.apply_sol_or_dense(sol, run_attention, layer_index=0, step_index=20, timestep=None)
  sol_attention.apply_sol_or_dense(sol, run_attention, layer_index=5, step_index=5, timestep=None)
  sol_attention.apply_sol_or_dense(sol, run_attention, layer_index=5, step_index=20, timestep=None)
  assert calls[0] is None and calls[1] is None
  assert calls[2]["use_sol_attention"] and calls[2]["beta"] == 1.0
  with pytest.raises(ValueError, match="Incomplete explicit Sol-Attn step schedule"):
    sol.sol_active_end_step = -1
    sol_attention.apply_sol_or_dense(sol, run_attention, layer_index=5, step_index=20, timestep=None)


def test_pipeline_and_aot_plumbing_is_inert_when_disabled():
  dense = SimpleNamespace(use_sol_attention=False, sol_beta=1.0)
  assert not sol_attention.attention_config_from_pyconfig(dense, 40)
  assert not aot_cache.extract_sol_meta(dense)
  enabled = SimpleNamespace(use_sol_attention=True, sol_beta=0.8, sol_block_q=512, sol_active_start_step=10)
  attn_cfg = sol_attention.attention_config_from_pyconfig(enabled, 40)
  assert attn_cfg["use_sol_attention"] and attn_cfg["sol_beta"] == 0.8 and attn_cfg["sol_num_layers"] == 40
  assert sol_attention.init_sol_config(attn_cfg, 40)["sol_block_q"] == 512
  meta_a = aot_cache.extract_sol_meta(enabled)
  meta_b = aot_cache.extract_sol_meta(SimpleNamespace(**{**vars(enabled), "sol_beta": 1.2}))
  assert meta_a and meta_a != meta_b
