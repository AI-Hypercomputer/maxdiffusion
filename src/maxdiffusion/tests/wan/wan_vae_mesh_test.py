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
cross-attention KV cache helpers.
"""

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.models.wan.transformers import transformer_wan


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
