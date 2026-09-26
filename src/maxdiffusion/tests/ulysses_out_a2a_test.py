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

"""Tests for the inverse Ulysses exchange (`wan_ulysses_out_a2a`).

Inside the shard_map each shard holds [B, H/n, ..., S] (its heads, full
sequence) and must end with [B, H, ..., S/n] (every head, its sequence chunk).
Seen globally that is a pure resharding, so the output must equal the input
exactly, in both modes. Needs >= 2 devices; on CPU run with
XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from maxdiffusion import wan_runtime_options
from maxdiffusion.models import attention_flax


class UlyssesSeqToHeadsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    wan_runtime_options.reset()
    self.addCleanup(wan_runtime_options.reset)
    n = len(jax.devices())
    if n < 2:
      self.skipTest("needs >= 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU)")
    self.n = 4 if n >= 4 else 2
    self.mesh = Mesh(np.array(jax.devices()[: self.n]), ("context",))

  def _exchange(self, mode, x, seq_axis):
    in_spec = P(None, "context", *([None] * (x.ndim - 2)))
    out_spec = [None] * x.ndim
    out_spec[seq_axis] = "context"
    fn = jax.shard_map(
        lambda t: attention_flax._ulysses_seq_to_heads(t, axis_name="context", seq_axis=seq_axis, num_shards=self.n),
        mesh=self.mesh,
        in_specs=(in_spec,),
        out_specs=P(*out_spec),
        check_vma=False,
    )
    with mock.patch.dict(os.environ, {"WAN_ULYSSES_OUT_A2A": mode}):
      return jax.jit(fn)(x)

  def test_both_modes_are_a_pure_reshard(self):
    heads, d, seq = 2 * self.n, 128, 24 * self.n
    shapes = {3: (1, heads, d, seq), 2: (1, heads, seq, d)}  # [B,H,D,S] (kernel O^T) and [B,H,S,D]
    for seq_axis, shape in shapes.items():
      x = jax.random.normal(jax.random.PRNGKey(seq_axis), shape, jnp.float32).astype(jnp.bfloat16)
      for mode in attention_flax.ULYSSES_OUT_A2A_MODES:
        with self.subTest(seq_axis=seq_axis, mode=mode):
          got = self._exchange(mode, x, seq_axis)
          self.assertEqual(got.shape, x.shape)
          np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(x, np.float32))

  def test_rejects_unknown_mode(self):
    x = jnp.zeros((1, 2 * self.n, 8 * self.n, 128), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "wan_ulysses_out_a2a"):
      self._exchange("bogus", x, 2)


if __name__ == "__main__":
  unittest.main()
