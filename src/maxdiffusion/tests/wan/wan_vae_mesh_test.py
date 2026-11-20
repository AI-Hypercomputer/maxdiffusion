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

CPU checks for the batch-aware VAE mesh selection and the batch-sharded
cross-attention KV cache (helpers, and compute_kv_cache against the eager vmap
it replaced).
"""

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.models.wan.transformers import transformer_wan
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel


def _cfg(batch_size):
  return types.SimpleNamespace(global_batch_size_to_train_on=batch_size)


@pytest.mark.parametrize(
    "batch_size, total_devices, expected",
    [
        (0, 8, 8),
        (1, 8, 8),
        (2, 8, 8),
        (3, 8, 8),
        (4, 8, 8),
        (8, 8, 1),
        (12, 8, 8),
        (16, 8, 1),
        (4, 4, 1),
        (1, 1, 1),
    ],
)
def test_auto_vae_spatial_batch_parallel(batch_size, total_devices, expected):
  """With the gate on, B % N == 0 picks pure data parallelism, otherwise full spatial."""
  assert WanPipeline._auto_vae_spatial(_cfg(batch_size), total_devices, batch_parallel_vae=True) == expected


@pytest.mark.parametrize("batch_size", [0, 1, 2, 8, 16])
def test_auto_vae_spatial_default_is_full_spatial(batch_size):
  """Without the gate (i2v / vace / animate) auto always resolves to the old value."""
  assert WanPipeline._auto_vae_spatial(_cfg(batch_size), 8) == 8
  assert WanPipeline._auto_vae_spatial(_cfg(batch_size), 8, batch_parallel_vae=False) == 8


def test_only_t2v_pipelines_enable_batch_parallel_vae():
  """Guard against the gate leaking into pipelines that VAE-encode at batch != global batch."""
  pipeline_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipelines", "wan")
  enabled = set()
  for name in sorted(os.listdir(pipeline_dir)):
    if not name.endswith(".py"):
      continue
    with open(os.path.join(pipeline_dir, name)) as f:
      if "batch_parallel_vae=True" in f.read():
        enabled.add(name)
  assert enabled == {"wan_pipeline_2_1.py", "wan_pipeline_2_2.py"}


def test_decode_latents_rejects_indivisible_batch_on_batch_parallel_mesh():
  """_decode_latents_to_video must fail fast instead of silently decoding replicated."""
  all_devices = jax.devices()
  if len(all_devices) < 2:
    pytest.skip("Requires at least 2 JAX devices to build a redundant > 1 VAE mesh")
  vae_mesh = Mesh(np.array(all_devices).reshape(-1, 1), ("redundant", "vae_spatial"))
  dummy_pipeline = types.SimpleNamespace(vae_mesh=vae_mesh)
  latents = jnp.zeros((1, 16, 1, 8, 8))
  with pytest.raises(ValueError, match="divisible by the 'redundant' mesh axis"):
    WanPipeline._decode_latents_to_video(dummy_pipeline, latents)


@pytest.fixture
def mesh_and_rules():
  all_devices = jax.devices()
  if len(all_devices) < 2:
    pytest.skip("Requires at least 2 JAX devices to test multi-shard batch sharding")
  usable = (len(all_devices) // 2) * 2
  devices = np.array(all_devices[:usable]).reshape(2, 1, -1, 1)
  mesh = Mesh(devices, ("data", "fsdp", "context", "tensor"))
  rules = (("activation_batch", ("data", "fsdp")),)
  return mesh, rules


def test_batch_sharding_divisible(mesh_and_rules):
  mesh, rules = mesh_and_rules
  with mesh, nn_partitioning.axis_rules(rules):
    sharding = transformer_wan._batch_sharding(mesh, ndim=3, batch_axis=0, batch_size=4)
  assert sharding is not None
  assert sharding.spec == P(("data", "fsdp"), None, None)


def test_batch_sharding_not_divisible_returns_none(mesh_and_rules):
  """B=3 on data*fsdp=2 must skip the constraint instead of raising inside the jit."""
  mesh, rules = mesh_and_rules
  with mesh, nn_partitioning.axis_rules(rules):
    assert transformer_wan._batch_sharding(mesh, ndim=3, batch_axis=0, batch_size=3) is None


def test_batch_sharding_without_rules_or_mesh(mesh_and_rules):
  mesh, _ = mesh_and_rules
  assert transformer_wan._batch_sharding(None, ndim=3, batch_axis=0) is None
  with mesh, nn_partitioning.axis_rules(()):
    assert transformer_wan._batch_sharding(mesh, ndim=3, batch_axis=0) is None


def test_shard_kv_cache_on_batch_layouts(mesh_and_rules):
  """[B,S,D] shards on axis 0, stacked [L,B,S,D] on axis 1; odd batches stay replicated."""
  mesh, rules = mesh_and_rules
  cache = {
      "text": (jnp.zeros((4, 16, 8)), jnp.zeros((2, 4, 16, 8))),
      "odd": (jnp.zeros((3, 16, 8)), jnp.zeros((2, 3, 16, 8))),
  }
  with mesh, nn_partitioning.axis_rules(rules):
    out = jax.jit(lambda c: transformer_wan._shard_kv_cache_on_batch(c, mesh))(cache)
  k, v = out["text"]
  # JAX canonicalises the spec of the returned array (size-1 'fsdp' is dropped),
  # so compare semantically rather than by literal PartitionSpec.
  assert k.sharding.is_equivalent_to(NamedSharding(mesh, P(("data", "fsdp"), None, None)), k.ndim)
  assert v.sharding.is_equivalent_to(NamedSharding(mesh, P(None, ("data", "fsdp"), None, None)), v.ndim)
  assert k.addressable_shards[0].data.shape[0] == 2
  assert v.addressable_shards[0].data.shape[1] == 2
  k_odd, v_odd = out["odd"]
  assert k_odd.shape == (3, 16, 8) and v_odd.shape == (2, 3, 16, 8)
  assert k_odd.sharding.is_equivalent_to(NamedSharding(mesh, P()), k_odd.ndim)


# Production logical axis rules (configs/base_wan_27b.yml), so the tiny model
# below is constrained the same way the real one is.
_WAN_LOGICAL_AXIS_RULES = (
    ("batch", ("data", "fsdp")),
    ("activation_batch", ("data", "fsdp")),
    ("activation_self_attn_heads", ("context", "tensor")),
    ("activation_cross_attn_q_length", ("context", "tensor")),
    ("activation_length", "context"),
    ("activation_heads", "tensor"),
    ("mlp", "tensor"),
    ("embed", ("context", "fsdp")),
    ("heads", "tensor"),
    ("norm", "tensor"),
    ("conv_batch", ("data", "context", "fsdp")),
    ("out_channels", "tensor"),
    ("conv_out", "context"),
)


def _eager_vmap_kv_cache(model, encoder_hidden_states):
  """The eager nnx.vmap implementation that _scanned_compute_kv replaced."""
  timestep = jnp.zeros((encoder_hidden_states.shape[0],), dtype=jnp.int32)
  _, _, encoder_hidden_states, _, encoder_attention_mask = model.condition_embedder(timestep, encoder_hidden_states, None)

  @nnx.vmap(in_axes=(0, None, None), out_axes=0, transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"})
  def _compute_kv(block, enc_states, enc_mask):
    return block.compute_kv(enc_states, enc_mask)

  return _compute_kv(model.blocks, encoder_hidden_states, encoder_attention_mask)


def _assert_close(actual, expected, dtype):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  if dtype == jnp.float32:
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
  else:
    # Scan and vmap may be fused differently on TPU, so allow one bf16 ULP of the output scale.
    np.testing.assert_allclose(actual, expected, rtol=2**-7, atol=2**-7 * float(np.max(np.abs(expected))))


@pytest.mark.parametrize(
    "dtype, batch_size",
    [(jnp.float32, 4), (jnp.bfloat16, 4), (jnp.float32, 3)],
    ids=["f32_divisible", "bf16_divisible", "f32_not_divisible"],
)
def test_scanned_kv_cache_matches_eager_vmap(mesh_and_rules, dtype, batch_size):
  """compute_kv_cache (jitted, scanned, batch-sharded) equals the old eager vmap, and the forward pass accepts it."""
  mesh, _ = mesh_and_rules
  num_layers = 2
  with mesh, nn_partitioning.axis_rules(_WAN_LOGICAL_AXIS_RULES):
    model = WanModel(
        rngs=nnx.Rngs(0),
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=64,
        num_layers=num_layers,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=dtype,
        scan_layers=True,
    )
    encoder_hidden_states = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 8, 32), dtype)
    expected = _eager_vmap_kv_cache(model, encoder_hidden_states)
    kv_cache, encoder_attention_mask = model.compute_kv_cache(encoder_hidden_states)

    assert set(kv_cache) == set(expected)
    data_shards = mesh.shape["data"] * mesh.shape["fsdp"]
    for name, expected_kv in expected.items():
      for actual, want in zip(kv_cache[name], expected_kv):
        assert actual.shape == want.shape and actual.shape[:2] == (num_layers, batch_size)
        assert actual.dtype == want.dtype
        _assert_close(actual, want, dtype)
        if batch_size % data_shards == 0:
          # Stacked [L, B, S, H*D]: each device holds only its own slice of the batch.
          assert actual.addressable_shards[0].data.shape[1] == batch_size // data_shards
        else:
          assert actual.addressable_shards[0].data.shape[1] == batch_size

    hidden_states = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 4, 4, 8, 8), dtype)
    timestep = jnp.full((batch_size,), 500.0)
    uncached = model(hidden_states, timestep, encoder_hidden_states, return_dict=False)
    cached = model(
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict=False,
        kv_cache=kv_cache,
        encoder_attention_mask=encoder_attention_mask,
    )
    _assert_close(cached[0], uncached[0], dtype)
