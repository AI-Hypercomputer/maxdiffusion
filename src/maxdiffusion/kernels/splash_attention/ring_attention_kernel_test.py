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
"""Tests for ring attention."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from . import base
from . import ring_attention_kernel
from . import splash_attention_kernel as splash
from . import splash_attention_mask as mask_lib
from . import splash_attention_test_utils as test_utils

P = jax.sharding.PartitionSpec
partial = functools.partial

jax.config.parse_flags_with_absl()


class RingAttentionTest(test_utils.SplashAttentionTestCase):

  def setUp(self):
    self.skipTest("no sharding on runners")
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

    if len(jax.devices()) < 4:
      self.skipTest("This test requires at least 4 devices.")

    super().setUp()

  @parameterized.product(
      ring_size=[2],
      num_heads=[1],
      head_dim=[128, 256],
      dtype=[jnp.bfloat16],
      is_mqa=[False, True],
      is_segmented=[False, True],
      mask_type=["FULL", "CAUSAL"],
  )
  def test_ring_attention(
      self,
      ring_size,
      num_heads,
      head_dim,
      dtype,
      is_mqa,
      is_segmented,
      mask_type,
  ):
    if len(jax.devices()) < ring_size:
      self.skipTest(f"This test requires {ring_size} devices, but has only" f" {len(jax.devices())} devices available.")

    # Mesh Creation and Input Generation
    ring_axis = "ring"
    devices = np.asarray(jax.devices()[:ring_size]).reshape(1, ring_size)
    mesh = jax.sharding.Mesh(devices, ("heads", ring_axis))
    seq_len = 1024 * ring_size

    k1, k2, k3, k4 = random.split(random.key(0), 4)
    scale = head_dim**-0.5
    q = random.normal(k1, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    if is_mqa:
      k = random.normal(k2, (seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (seq_len, head_dim), dtype=dtype) * scale
    else:
      k = random.normal(k2, (num_heads, seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    do = random.normal(k4, q.shape, dtype=dtype) * scale

    if mask_type == "CAUSAL":
      mask = mask_lib.make_causal_mask((seq_len, seq_len))
    elif mask_type == "FULL":
      mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    else:
      raise ValueError(f"Unsupported mask type: {mask_type}")

    if is_segmented:
      segment_ids = test_utils.create_segment_ids(seq_len)
      segment_ids_spec = base.SegmentIds(q=P(ring_axis), kv=P(ring_axis))
    else:
      segment_ids = segment_ids_spec = None

    # For ring attention, sequence dimension is sharded over 'ring' axis
    q_spec = P(None, ring_axis, None)
    kv_spec = P(ring_axis, None) if is_mqa else q_spec

    splash_config = splash.SplashConfig.get_default()
    splash_config = dataclasses.replace(
        splash_config,
        use_base2_exp=False,
        fuse_reciprocal=True,
        # TODO: Change fuse_reciprocal behavior for ring attention
        # so we do the reciprocal after ring
    )

    ring_kernel = ring_attention_kernel.make_ring_attention(
        mask,
        is_mqa=is_mqa,
        ring_axis=ring_axis,
        config=splash_config,
        save_residuals=False,
        q_seq_shards=ring_size,
        kv_seq_shards=ring_size,
    )
    kernel_spec = ring_kernel.manual_sharding_spec()

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
    def ring_attn(ring_kernel, q, k, v, segment_ids):
      out = ring_kernel(q, k, v, segment_ids)
      return out

    ring_attn_ref = partial(base.attention_reference, is_mqa=is_mqa)

    with self.subTest("fwd"):
      out = ring_attn(ring_kernel, q, k, v, segment_ids)
      out_ref = ring_attn_ref(q, k, v, mask[:, :], segment_ids)
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

    with self.subTest("bwd"):
      out, out_vjp = jax.vjp(ring_attn, ring_kernel, q, k, v, segment_ids)
      out_ref, out_vjp_ref = jax.vjp(ring_attn_ref, q, k, v, mask[:, :], segment_ids)
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

      _, dq, dk, dv, _ = out_vjp(do)
      dq_ref, dk_ref, dv_ref, _, _ = out_vjp_ref(do.astype(jnp.float32))

      self._assert_allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dv, dv_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
