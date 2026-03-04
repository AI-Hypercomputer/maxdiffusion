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

"""Tests for partitioning splash_attention."""

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from . import base
from . import splash_attention_kernel as splash
from . import splash_attention_mask as mask_lib
from . import splash_attention_test_utils as test_utils


PartitionSpec = jax.sharding.PartitionSpec
P = jax.P
partial = functools.partial

jax.config.parse_flags_with_absl()


class PallasBaseTest(test_utils.SplashAttentionTestCase):
  INTERPRET = False

  def setUp(self):
    super().setUp()
    if not test_utils.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU.")

    if len(jax.devices()) < 4:
      self.skipTest("This test requires at least 4 devices.")


class SplashAttentionShardingTest(PallasBaseTest):

  def setUp(self):
    self.skipTest("no sharding on runners")
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.product(
      topology=[(2, 2), (1, 4), (4, 1)],
      num_heads=[2, 16],
      dtype=[jnp.bfloat16],
      is_segmented=[False, True],
      is_dynamic_mask=[False, True],
  )
  def test_manual_partitioning_mha_fwd(
      self, topology, num_heads, dtype, is_segmented, is_dynamic_mask
  ):
    # TODO: Re-enable once dynamic masks are fixed.
    if is_dynamic_mask:
      self.skipTest("Dynamic masks not supported.")

    k1, k2, k3 = random.split(random.key(0), 3)
    seq_len = 1024
    head_dim = 128

    head_shards, q_seq_shards = topology
    num_devices = math.prod(topology)

    if head_shards > num_heads:
      self.skipTest(
          f"This test requires {num_heads} heads, but has only"
          f" {head_shards} head shards available."
      )

    if len(jax.devices()) < num_devices:
      self.skipTest(
          f"This test requires {num_devices} devices, but has only"
          f" {len(jax.devices())} devices available."
      )

    q = random.uniform(k1, (num_heads, seq_len, head_dim), dtype=dtype)
    k = random.uniform(k2, (num_heads, seq_len, head_dim), dtype=dtype)
    v = random.uniform(k3, (num_heads, seq_len, head_dim), dtype=dtype)

    mask = mask_lib.make_causal_mask((seq_len, seq_len))
    if is_dynamic_mask:
      mask = jnp.array(mask)

    if is_segmented:
      segment_ids = test_utils.create_segment_ids(seq_len)
      segment_ids_spec = base.SegmentIds(
          q=PartitionSpec("q_seq" if q_seq_shards > 1 else None),
          kv=PartitionSpec(None),
      )
    else:
      segment_ids = segment_ids_spec = None

    devices = np.asarray(jax.devices()[:num_devices]).reshape(
        head_shards, q_seq_shards
    )

    mesh = jax.sharding.Mesh(devices, ("heads", "q_seq"))
    q_spec = PartitionSpec(
        "heads" if head_shards > 1 else None,
        "q_seq" if q_seq_shards > 1 else None,
    )
    mask_spec = PartitionSpec("q_seq" if q_seq_shards > 1 else None)
    kv_spec = PartitionSpec("heads" if head_shards > 1 else None, None)

    if is_dynamic_mask:
      kernel, kernel_spec = splash.make_dynamic_splash_mha(
          mask, mesh=mesh, mask_spec=mask_spec
      )
    else:
      kernel = splash.make_splash_mha(mask, q_seq_shards=q_seq_shards)
      kernel_spec = kernel.manual_sharding_spec(
          jax.sharding.NamedSharding(mesh, mask_spec)
      )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
            segment_ids_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def f(kernel, q, k, v, segment_ids):
      return kernel(q, k, v, segment_ids)

    out = f(kernel, q, k, v, segment_ids)
    out_ref = base.attention_reference(q, k, v, mask, segment_ids, is_mqa=False)
    self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

  @parameterized.product(
      topology=[(2, 2), (1, 4), (4, 1)],
      num_heads=[2, 4],
      dtype=[jnp.bfloat16],
      is_segmented=[False, True],
      is_dynamic_mask=[False, True],
  )
  def test_manual_partitioning_mha_bwd(
      self, topology, num_heads, dtype, is_segmented, is_dynamic_mask
  ):
    # TODO: Re-enable once dynamic masks are fixed.
    if is_dynamic_mask:
      self.skipTest("Dynamic masks not supported.")

    assert num_heads % 2 == 0
    k1, k2, k3, k4 = random.split(random.key(0), 4)
    seq_len = 1024
    head_dim = 128

    head_shards, q_seq_shards = topology
    num_devices = math.prod(topology)

    if head_shards > num_heads:
      self.skipTest(
          f"This test requires {num_heads} heads, but has only"
          f" {head_shards} head shards available."
      )

    q = random.uniform(k1, (num_heads, seq_len, head_dim), dtype=dtype)
    k = random.uniform(k2, (num_heads, seq_len, head_dim), dtype=dtype)
    v = random.uniform(k3, (num_heads, seq_len, head_dim), dtype=dtype)

    mask = mask_lib.make_causal_mask((seq_len, seq_len))
    if is_dynamic_mask:
      mask = jnp.array(mask)

    if is_segmented:
      segment_ids = test_utils.create_segment_ids(seq_len)
      segment_ids_spec = base.SegmentIds(
          q=PartitionSpec("q_seq" if q_seq_shards > 1 else None),
          kv=PartitionSpec(None),
      )
    else:
      segment_ids = segment_ids_spec = None

    devices = np.asarray(jax.devices()[:num_devices]).reshape(
        head_shards, q_seq_shards
    )

    mesh = jax.sharding.Mesh(devices, ("heads", "q_seq"))
    q_spec = PartitionSpec(
        "heads" if head_shards > 1 else None,
        "q_seq" if q_seq_shards > 1 else None,
    )
    mask_spec = PartitionSpec("q_seq" if q_seq_shards > 1 else None)
    kv_spec = PartitionSpec("heads" if head_shards > 1 else None, None)

    if is_dynamic_mask:
      kernel, kernel_spec = splash.make_dynamic_splash_mha(
          mask, mesh=mesh, mask_spec=mask_spec
      )
    else:
      kernel = splash.make_splash_mha(mask, q_seq_shards=q_seq_shards)
      kernel_spec = kernel.manual_sharding_spec(
          jax.sharding.NamedSharding(mesh, mask_spec)
      )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
            segment_ids_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def f(kernel, q, k, v, segment_ids):
      return kernel(q, k, v, segment_ids)

    f_ref = partial(base.attention_reference, is_mqa=False)

    out, out_vjp = jax.vjp(f, kernel, q, k, v, segment_ids)
    out_ref, out_vjp_ref = jax.vjp(f_ref, q, k, v, mask, segment_ids)
    self._assert_allclose(out, out_ref, rtol=5e-3, atol=5e-3)

    do = random.uniform(k4, out.shape, dtype=out.dtype)
    _, dq, dk, dv, _ = out_vjp(do)
    dq_ref, dk_ref, dv_ref, _, _ = out_vjp_ref(do.astype(jnp.float32))

    self._assert_allclose(dq, dq_ref, atol=8e-2, rtol=1e-2)
    self._assert_allclose(dk, dk_ref, atol=8e-2, rtol=2e-2)
    self._assert_allclose(dv, dv_ref, atol=8e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
