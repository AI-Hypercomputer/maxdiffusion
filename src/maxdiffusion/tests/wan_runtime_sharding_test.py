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

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from jax.sharding import Mesh, PartitionSpec as P

from ..pipelines.wan.wan_pipeline import WanPipeline


class RuntimeDataShardingTest(unittest.TestCase):

  def _make_pipeline(self) -> WanPipeline:
    device = jax.devices("cpu")[0]
    pipeline = WanPipeline.__new__(WanPipeline)
    pipeline.mesh = Mesh(np.array([device]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    pipeline.config = SimpleNamespace(data_sharding=[["data", "fsdp", "context", "tensor"]])
    return pipeline

  def test_batch_shard_count_supports_composite_batch_axes(self):
    mesh = mock.Mock()
    mesh.shape = {"data": 2, "fsdp": 4, "context": 8, "tensor": 1}

    self.assertEqual(
        WanPipeline._batch_shard_count(mesh, [["data", "fsdp"], None, None]),
        8,
    )

  def test_runtime_data_sharding_uses_configured_spec_when_batch_is_shardable(self):
    pipeline = self._make_pipeline()

    with mock.patch.object(WanPipeline, "_batch_shard_count", return_value=1):
      sharding = pipeline.get_runtime_data_sharding(batch_dim_size=1)

    self.assertEqual(sharding.spec, P(("data", "fsdp", "context", "tensor")))

  def test_runtime_data_sharding_falls_back_to_replicated_when_batch_is_not_shardable(self):
    pipeline = self._make_pipeline()

    with mock.patch.object(WanPipeline, "_batch_shard_count", return_value=8):
      sharding = pipeline.get_runtime_data_sharding(batch_dim_size=1)

    self.assertEqual(sharding.spec, P())


if __name__ == "__main__":
  unittest.main()
