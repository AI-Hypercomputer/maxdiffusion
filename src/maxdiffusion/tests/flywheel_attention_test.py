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

"""Tests for attention='flywheel' and attention='ulysses_flywheel'.

The guard and wiring tests need neither flywheel-tpu nor a TPU: every
unsupported input must raise before the optional import. The numerical tests
need flywheel-tpu on a TPU the kernel supports (v6e or tpu7x) and compare
against an fp32 softmax reference; the multi-device ones also need 4 devices.
"""
# pylint: disable=protected-access
import importlib.util
import math
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh
import numpy as np

from .. import common_types
from ..models import attention_flax
from ..tpu_utils import TpuType, get_tpu_type

HAS_FLYWHEEL = importlib.util.find_spec("flywheel_tpu") is not None
FLYWHEEL_TPU = get_tpu_type() in (TpuType.TPU_V6_LITE, TpuType.TPU_7X)
_requires_flywheel = unittest.skipUnless(HAS_FLYWHEEL and FLYWHEEL_TPU, "needs flywheel-tpu on TPU v6e or tpu7x")
_requires_4_devices = unittest.skipIf(len(jax.devices()) < 4, "needs 4 devices for context parallelism")

MESH_AXES = ("data", "fsdp", "context", "tensor")
Q_NAMES = (attention_flax.BATCH, attention_flax.SELF_ATTN_HEAD, attention_flax.SELF_ATTN_Q_LENGTH, attention_flax.D_KV)
KV_NAMES = (attention_flax.BATCH, attention_flax.SELF_ATTN_HEAD, attention_flax.SELF_ATTN_KV_LENGTH, attention_flax.D_KV)
# The rules pyconfig adds for attention='flywheel' (as for 'flash') and for 'ulysses*'.
FLASH_RULES = tuple(tuple(rule) for rule in common_types.SEQUENCE_PARALLEL_AXIS_RULES) + ((common_types.BATCH, "data"),)
ULYSSES_RULES = tuple(tuple(rule) for rule in common_types.ULYSSES_ATTENTION_AXIS_RULES) + ((common_types.BATCH, "data"),)


def _mesh(num_context):
  devices = np.array(jax.devices()[:num_context]).reshape(1, 1, num_context, 1)
  return Mesh(devices, MESH_AXES)


def _inputs(batch, seq, heads, head_dim, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  shape = (batch, seq, heads * head_dim)
  return tuple(jax.random.normal(k, shape, jnp.float32).astype(jnp.bfloat16) for k in keys)


def _reference(query, key, value, heads, scale):
  """fp32 softmax attention on [B, S, H*D] inputs; returns [B, S, H*D]."""
  b, s, hd = query.shape
  q, k, v = (x.astype(jnp.float32).reshape(b, -1, heads, hd // heads).transpose(0, 2, 1, 3) for x in (query, key, value))
  probs = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale, axis=-1)
  return jnp.einsum("bhqk,bhkd->bhqd", probs, v).transpose(0, 2, 1, 3).reshape(b, s, hd)


def _rel_err(out, ref):
  return float(jnp.linalg.norm(out.astype(jnp.float32) - ref) / jnp.linalg.norm(ref))


class FlywheelGuardTest(unittest.TestCase):
  """Unsupported inputs raise before flywheel_tpu is imported, so these run anywhere."""

  def setUp(self):
    self.x = jnp.zeros((1, 256, 2 * 128), jnp.bfloat16)

  def _call(self, mesh=None, rules=FLASH_RULES, x=None, **kwargs):
    mesh = mesh if mesh is not None else AbstractMesh((1, 1, 1, 1), MESH_AXES)
    x = self.x if x is None else x
    with nn_partitioning.axis_rules(rules):
      return attention_flax._flywheel_attention(
          x, x, x, heads=2, mesh=mesh, axis_names_q=Q_NAMES, axis_names_kv=KV_NAMES, scale=1.0, **kwargs
      )

  def test_kernels_are_registered(self):
    for name in attention_flax.FLYWHEEL_KERNELS:
      self.assertIn(name, attention_flax.KERNEL_REGISTRY)

  def test_attention_mask_is_rejected(self):
    for use_ulysses in (False, True):
      with self.assertRaisesRegex(NotImplementedError, "attention_mask"):
        self._call(attention_mask=jnp.ones((1, 256), jnp.bool_), use_ulysses=use_ulysses)

  def test_causal_is_rejected(self):
    with self.assertRaisesRegex(NotImplementedError, "non-causal"):
      self._call(is_causal=True)

  def test_gqa_is_rejected(self):
    with self.assertRaisesRegex(NotImplementedError, "GQA"):
      self._call(kv_heads=1)

  def test_non_bfloat16_is_rejected(self):
    with self.assertRaisesRegex(TypeError, "bfloat16"):
      self._call(x=self.x.astype(jnp.float32))

  def test_sequence_sharded_kv_is_rejected_without_ulysses(self):
    with self.assertRaisesRegex(ValueError, "ulysses_flywheel"):
      self._call(mesh=AbstractMesh((1, 1, 2, 1), MESH_AXES), rules=ULYSSES_RULES)

  def test_ulysses_rejects_indivisible_heads(self):
    with self.assertRaisesRegex(ValueError, "divisible"):
      self._call(mesh=AbstractMesh((1, 1, 3, 1), MESH_AXES), rules=ULYSSES_RULES, use_ulysses=True)

  def test_ulysses_rejects_mismatched_ulysses_shards(self):
    with self.assertRaisesRegex(ValueError, "ulysses_shards=4"):
      self._call(mesh=AbstractMesh((1, 1, 2, 1), MESH_AXES), rules=ULYSSES_RULES, use_ulysses=True, ulysses_shards=4)


class FlywheelWanWiringTest(unittest.TestCase):
  """FlaxWanAttention runs self-attention on FlyWheel and keeps cross-attention on Splash."""

  def _build(self, attention_kernel, is_self_attention):
    with _mesh(1), nn_partitioning.axis_rules(FLASH_RULES):
      return attention_flax.FlaxWanAttention(
          rngs=nnx.Rngs(0),
          query_dim=256,
          heads=2,
          dim_head=128,
          attention_kernel=attention_kernel,
          mesh=_mesh(1),
          is_self_attention=is_self_attention,
      )

  def test_self_attention_uses_flywheel_and_cross_attention_uses_flash(self):
    for kernel in attention_flax.FLYWHEEL_KERNELS:
      self.assertEqual(self._build(kernel, is_self_attention=True).attention_op.attention_kernel, kernel)
      self.assertEqual(self._build(kernel, is_self_attention=False).attention_op.attention_kernel, "flash")

  def test_mesh_is_required(self):
    with self.assertRaises(ValueError):
      attention_flax.FlaxWanAttention(
          rngs=nnx.Rngs(0), query_dim=256, heads=2, dim_head=128, attention_kernel="flywheel", mesh=None
      )


@_requires_flywheel
class FlywheelNumericsTest(unittest.TestCase):
  """Sharded FlyWheel attention against an fp32 reference on the same bf16 inputs."""

  # bf16 inputs against an fp32 reference: the observed error is ~3e-3.
  TOL = 1e-2

  def _check(self, *, num_context=1, use_ulysses=False, chunks=1, batch=2, seq=1024, heads=4, head_dim=128):
    """Runs the sharded backend on random bf16 inputs and compares with the fp32 reference."""
    query, key, value = _inputs(batch, seq, heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    mesh = _mesh(num_context)
    with mesh, nn_partitioning.axis_rules(ULYSSES_RULES if use_ulysses else FLASH_RULES):
      out = jax.jit(
          lambda q, k, v: attention_flax._flywheel_attention(
              q,
              k,
              v,
              heads=heads,
              mesh=mesh,
              axis_names_q=Q_NAMES,
              axis_names_kv=KV_NAMES,
              scale=scale,
              use_ulysses=use_ulysses,
              ulysses_attention_chunks=chunks,
          )
      )(query, key, value)
    ref = _reference(query, key, value, heads, scale)
    self.assertEqual(out.shape, ref.shape)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertLess(_rel_err(out, ref), self.TOL)

  def test_single_device(self):
    self._check()

  def test_single_device_ulysses(self):
    self._check(use_ulysses=True)

  def test_unaligned_length_does_not_attend_padding(self):
    # 600 is not a multiple of 128: the kernel must not attend the rows it pads to.
    self._check(seq=600)

  def test_single_batch_row(self):
    self._check(batch=1, heads=3)

  def test_k_prescaled_matches_scale_argument(self):
    query, key, value = _inputs(2, 1024, 4, 128)
    scale = 1.0 / math.sqrt(128)
    mesh = _mesh(1)
    outs = []
    with mesh, nn_partitioning.axis_rules(FLASH_RULES):
      for k, prescaled in ((key, False), ((key.astype(jnp.float32) * scale).astype(jnp.bfloat16), True)):
        context = {
            "heads": 4,
            "mesh": mesh,
            "axis_names_q": Q_NAMES,
            "axis_names_kv": KV_NAMES,
            "scale": scale,
            "k_prescaled": prescaled,
            "attention_mask": None,
        }
        outs.append(attention_flax.KERNEL_REGISTRY["flywheel"](query, k, value, context))
    self.assertLess(_rel_err(outs[1], outs[0].astype(jnp.float32)), self.TOL)

  def test_gradient_is_not_supported(self):
    query, key, value = _inputs(1, 256, 2, 128)
    mesh = _mesh(1)

    def loss(q):
      out = attention_flax._flywheel_attention(
          q, key, value, heads=2, mesh=mesh, axis_names_q=Q_NAMES, axis_names_kv=KV_NAMES, scale=1.0
      )
      return out.astype(jnp.float32).sum()

    with mesh, nn_partitioning.axis_rules(FLASH_RULES):
      with self.assertRaises(NotImplementedError):
        jax.grad(loss)(query)

  def test_wan_self_attention_matches_flash(self):
    """Same FlaxWanAttention weights and inputs, attention='flash' vs attention='flywheel'."""
    heads, head_dim, seq = 4, 128, 1560
    hidden = jax.random.normal(jax.random.PRNGKey(1), (2, seq, heads * head_dim), jnp.float32).astype(jnp.bfloat16)
    angles = jax.random.uniform(jax.random.PRNGKey(2), (1, 1, seq, head_dim // 2), jnp.float32, 0.0, 2 * math.pi)
    rotary_emb = jax.lax.complex(jnp.cos(angles), jnp.sin(angles))
    mesh = _mesh(1)
    outs = {}
    with mesh, nn_partitioning.axis_rules(FLASH_RULES):
      for kernel in ("flash", "flywheel"):
        attn = attention_flax.FlaxWanAttention(
            rngs=nnx.Rngs(0),
            query_dim=heads * head_dim,
            heads=heads,
            dim_head=head_dim,
            attention_kernel=kernel,
            mesh=mesh,
            flash_min_seq_length=0,
            dtype=jnp.bfloat16,
            weights_dtype=jnp.bfloat16,
        )
        outs[kernel] = nnx.jit(lambda m, h: m(hidden_states=h, rotary_emb=rotary_emb))(attn, hidden)
    self.assertLess(_rel_err(outs["flywheel"], outs["flash"].astype(jnp.float32)), self.TOL)

  @_requires_4_devices
  def test_context_parallel(self):
    # q is sequence-sharded 4 ways and K/V replicated; 1022 % 4 != 0 exercises the context padding.
    for seq in (1024, 1022):
      with self.subTest(seq=seq):
        self._check(num_context=4, seq=seq)

  @_requires_4_devices
  def test_context_parallel_ulysses(self):
    for seq, chunks in ((1024, 1), (1022, 1), (1024, 2)):
      with self.subTest(seq=seq, chunks=chunks):
        self._check(num_context=4, use_ulysses=True, chunks=chunks, heads=8, seq=seq)


if __name__ == "__main__":
  unittest.main()
