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

"""CPU checks for the attention-config guards across TPU topologies.

Guard 1: a non-ring ulysses kernel must REJECT a ulysses_shards it cannot honour.
Guard 2: a ring kernel must WARN when it degenerates to R=1, and the remedy it
         suggests must be valid for the actual mesh and head count.
Guard 3: a real ring (R>1) must REJECT a sequence length the context mesh axis
         does not divide, and R=1 must never be rejected for it.
"""
import unittest
from unittest import mock

from maxdiffusion.models import attention_flax

# Context-parallel degrees reachable on real slices: v5e/v6e-8 (CP up to 8),
# v6e-16 / v7x-16 (CP up to 16), and larger multi-host slices.
TOPOLOGIES = [1, 2, 4, 8, 16, 32, 64]

# Wan 2.2 T2V-A14B has 40 attention heads, which is NOT a power of two -- this
# is precisely why a naive "context_shards // 2" suggestion is unsafe.
WAN_HEADS = 40


class ImplicitUlyssesDegreeGuardTest(unittest.TestCase):

  def test_unset_is_allowed_on_every_topology(self):
    for cp in TOPOLOGIES:
      with self.subTest(cp=cp):
        attention_flax._validate_implicit_ulysses_degree(-1, cp, "ulysses_custom")
        attention_flax._validate_implicit_ulysses_degree(0, cp, "ulysses_custom")
        attention_flax._validate_implicit_ulysses_degree(None, cp, "ulysses_custom")

  def test_matching_request_is_allowed_on_every_topology(self):
    for cp in TOPOLOGIES:
      with self.subTest(cp=cp):
        attention_flax._validate_implicit_ulysses_degree(cp, cp, "ulysses_custom")

  def test_mismatched_request_is_rejected_on_every_topology(self):
    for cp in TOPOLOGIES:
      for requested in {1, 2, cp // 2, cp * 2} - {cp, 0}:
        if requested <= 0:
          continue
        with self.subTest(cp=cp, requested=requested):
          with self.assertRaises(ValueError) as ctx:
            attention_flax._validate_implicit_ulysses_degree(requested, cp, "ulysses_custom_fixed_m_per_q_block")
          msg = str(ctx.exception)
          self.assertIn(f"ulysses_shards={requested}", msg)
          self.assertIn(f"context_shards={cp}", msg)

  def test_rejection_names_the_offending_kernel(self):
    with self.assertRaises(ValueError) as ctx:
      attention_flax._validate_implicit_ulysses_degree(2, 4, "ulysses_custom_fixed_m_per_q_block")
    self.assertIn("ulysses_custom_fixed_m_per_q_block", str(ctx.exception))


class RealRingSuggestionTest(unittest.TestCase):

  def test_suggestion_is_valid_for_every_topology(self):
    """Whatever U we suggest must satisfy every constraint the ring enforces."""
    for cp in TOPOLOGIES:
      with self.subTest(cp=cp, heads=WAN_HEADS):
        u = attention_flax._largest_ulysses_shards_for_real_ring(cp, WAN_HEADS, WAN_HEADS)
        if u is None:
          # Only legitimate when no divisor below cp works.
          self.assertTrue(
              all(cp % c != 0 or WAN_HEADS % c != 0 for c in range(1, cp)),
              f"returned None for cp={cp} despite a valid candidate existing",
          )
          continue
        self.assertLess(u, cp)
        self.assertEqual(cp % u, 0, "suggested U must divide the context shard count")
        self.assertEqual(WAN_HEADS % u, 0, "suggested U must divide the head count")
        self.assertGreater(cp // u, 1, "suggested U must leave a real ring R>1")

  def test_no_suggestion_for_single_shard(self):
    self.assertIsNone(attention_flax._largest_ulysses_shards_for_real_ring(1, WAN_HEADS, WAN_HEADS))

  def test_prefers_smallest_real_ring(self):
    # cp=8, heads=40 -> U=4 gives R=2, the cheapest real ring.
    self.assertEqual(attention_flax._largest_ulysses_shards_for_real_ring(8, 40, 40), 4)
    # cp=16, heads=40 -> U=16 and U=8 both divide 16, but only 8 divides 40.
    self.assertEqual(attention_flax._largest_ulysses_shards_for_real_ring(16, 40, 40), 8)
    # cp=32, heads=40 -> largest common divisor below 32 is 8, giving R=4.
    self.assertEqual(attention_flax._largest_ulysses_shards_for_real_ring(32, 40, 40), 8)

  def test_respects_asymmetric_kv_heads(self):
    # GQA-style: 40 query heads but 8 KV heads restricts U to divisors of 8.
    u = attention_flax._largest_ulysses_shards_for_real_ring(16, 40, 8)
    self.assertEqual(8 % u, 0)
    self.assertEqual(40 % u, 0)
    self.assertEqual(16 % u, 0)


class DegenerateRingWarningTest(unittest.TestCase):

  def setUp(self):
    attention_flax._WARNED_ONCE.clear()

  def test_warns_on_every_topology_when_degenerate(self):
    for cp in TOPOLOGIES:
      with self.subTest(cp=cp):
        attention_flax._WARNED_ONCE.clear()
        with mock.patch.object(attention_flax.max_logging, "log") as log:
          attention_flax._warn_if_ring_is_degenerate(1, cp, cp, heads=WAN_HEADS, kv_heads=WAN_HEADS)
        log.assert_called_once()
        msg = log.call_args[0][0]
        self.assertIn("R=1", msg)
        self.assertIn("Do NOT report this as a ring-attention result", msg)

  def test_suggested_remedy_in_message_is_actionable(self):
    for cp in [2, 4, 8, 16, 32]:
      with self.subTest(cp=cp):
        attention_flax._WARNED_ONCE.clear()
        with mock.patch.object(attention_flax.max_logging, "log") as log:
          attention_flax._warn_if_ring_is_degenerate(1, cp, cp, heads=WAN_HEADS, kv_heads=WAN_HEADS)
        msg = log.call_args[0][0]
        expected_u = attention_flax._largest_ulysses_shards_for_real_ring(cp, WAN_HEADS, WAN_HEADS)
        self.assertIn(f"ulysses_shards={expected_u}", msg)
        self.assertIn(f"R={cp // expected_u}", msg)

  def test_single_shard_mesh_does_not_suggest_zero(self):
    """Regression: context_shards//2 would have advised the impossible U=0."""
    with mock.patch.object(attention_flax.max_logging, "log") as log:
      attention_flax._warn_if_ring_is_degenerate(1, 1, 1, heads=WAN_HEADS, kv_heads=WAN_HEADS)
    msg = log.call_args[0][0]
    self.assertNotIn("ulysses_shards=0", msg)
    self.assertIn("only one context shard", msg)

  def test_silent_for_real_ring_on_every_topology(self):
    for cp in [2, 4, 8, 16, 32]:
      u = attention_flax._largest_ulysses_shards_for_real_ring(cp, WAN_HEADS, WAN_HEADS)
      with self.subTest(cp=cp, u=u):
        attention_flax._WARNED_ONCE.clear()
        with mock.patch.object(attention_flax.max_logging, "log") as log:
          attention_flax._warn_if_ring_is_degenerate(cp // u, u, cp, heads=WAN_HEADS, kv_heads=WAN_HEADS)
        log.assert_not_called()

  def test_warns_only_once_per_configuration(self):
    with mock.patch.object(attention_flax.max_logging, "log") as log:
      for _ in range(80):  # ~40 layers x 2 transformers
        attention_flax._warn_if_ring_is_degenerate(1, 4, 4, heads=WAN_HEADS, kv_heads=WAN_HEADS)
    log.assert_called_once()

  def test_distinct_configurations_each_warn(self):
    with mock.patch.object(attention_flax.max_logging, "log") as log:
      attention_flax._warn_if_ring_is_degenerate(1, 4, 4, heads=WAN_HEADS, kv_heads=WAN_HEADS)
      attention_flax._warn_if_ring_is_degenerate(1, 8, 8, heads=WAN_HEADS, kv_heads=WAN_HEADS)
    self.assertEqual(log.call_count, 2)


class RingSequenceDivisibilityGuardTest(unittest.TestCase):
  """`_validate_ring_sequence_divisibility`: the R>1 `S % context_shards` guard.

  `_reshape_data_for_flash` zero-pads Q/K/V to a multiple of the context shard
  count. R=1 masks that padding (the kernel gets the true `orig_kv_seq_len`); a
  real ring would attend to it as real keys, so the wrapper rejects it. The
  guard is pure arithmetic and the wrapper calls it before any kernel is built.
  """

  # Wan 2.2 latent lengths from the base_wan_27b.yml recipe comments: 480p/81f
  # is 21 * 30 * 52, 720p/81f is 21 * 45 * 80. Neither is a multiple of 16/32.
  WAN_480P_81F = 32760
  WAN_720P_81F = 75600

  def _real_rings(self, cp):
    """(U, R) pairs with R > 1 reachable on a `cp`-way context axis."""
    return [(u, cp // u) for u in range(1, cp) if cp % u == 0]

  def test_r1_never_raises_on_every_topology(self):
    for cp in TOPOLOGIES:
      for seq in (cp * 8 + 1, 514, self.WAN_480P_81F, self.WAN_720P_81F):
        with self.subTest(cp=cp, seq=seq):
          attention_flax._validate_ring_sequence_divisibility(seq, seq, 1, cp)

  def test_real_ring_accepts_divisible_sequences_on_every_topology(self):
    for cp in TOPOLOGIES:
      for u, r in self._real_rings(cp):
        for seq in (cp * 8, cp * 37):
          with self.subTest(cp=cp, u=u, r=r, seq=seq):
            attention_flax._validate_ring_sequence_divisibility(seq, seq, r, cp)

  def test_real_ring_rejects_non_divisible_sequences_on_every_topology(self):
    for cp in TOPOLOGIES:
      for u, r in self._real_rings(cp):
        for rem in sorted({1, cp // 2, cp - 1} - {0}):
          seq = cp * 8 + rem
          with self.subTest(cp=cp, u=u, r=r, seq=seq):
            with self.assertRaises(ValueError) as ctx:
              attention_flax._validate_ring_sequence_divisibility(seq, seq, r, cp)
            msg = str(ctx.exception)
            self.assertIn("divisible by context_shards", msg)
            self.assertIn(f"context_shards={cp}", msg)
            self.assertIn(f"orig_q_seq_len={seq}", msg)
            self.assertIn(f"orig_kv_seq_len={seq}", msg)
            self.assertIn(f"R={r}", msg)
            # The remedies must be actionable for this mesh.
            self.assertIn(f"ulysses_shards={cp}", msg)
            self.assertIn(f"multiple of {cp}", msg)

  def test_q_and_kv_lengths_are_checked_independently(self):
    cp, r = 4, 2
    with self.assertRaisesRegex(ValueError, "divisible by context_shards"):
      attention_flax._validate_ring_sequence_divisibility(512, 514, r, cp)
    with self.assertRaisesRegex(ValueError, "divisible by context_shards"):
      attention_flax._validate_ring_sequence_divisibility(514, 512, r, cp)
    attention_flax._validate_ring_sequence_divisibility(512, 512, r, cp)

  def test_documented_wan_topologies(self):
    """The recipes this restriction actually touches (previously silently wrong, now an error)."""
    # 480p/81f at CP=16 (v7x-16, ulysses_shards=8 -> R=2): 32760 % 16 = 8.
    with self.assertRaisesRegex(ValueError, "divisible by context_shards"):
      attention_flax._validate_ring_sequence_divisibility(self.WAN_480P_81F, self.WAN_480P_81F, 2, 16)
    # 720p/81f at CP=32 (U=8 -> R=4): 75600 % 32 = 16.
    with self.assertRaisesRegex(ValueError, "divisible by context_shards"):
      attention_flax._validate_ring_sequence_divisibility(self.WAN_720P_81F, self.WAN_720P_81F, 4, 32)
    # 480p/81f at CP=8 (U=4 -> R=2) divides: 32760 % 8 = 0.
    attention_flax._validate_ring_sequence_divisibility(self.WAN_480P_81F, self.WAN_480P_81F, 2, 8)
    # 720p/81f at CP=16 (U=8 -> R=2) divides: 75600 % 16 = 0.
    attention_flax._validate_ring_sequence_divisibility(self.WAN_720P_81F, self.WAN_720P_81F, 2, 16)
    # Any of them at R=1 (ulysses_shards == CP) is fine.
    attention_flax._validate_ring_sequence_divisibility(self.WAN_480P_81F, self.WAN_480P_81F, 1, 16)
    attention_flax._validate_ring_sequence_divisibility(self.WAN_720P_81F, self.WAN_720P_81F, 1, 32)


class KernelClassificationDriftTest(unittest.TestCase):
  """Every registered ulysses-ring kernel must be classified for tile sizing.

  `local_tiled_seq_len` silently falls through to `return full_seq` for any
  attention name it does not recognise, which yields a mesh-independent (and
  therefore wrong) tile length. A newly registered ring kernel must not be able
  to slip through that default.
  """

  def test_all_registered_ulysses_ring_kernels_are_classified(self):
    from maxdiffusion.utils import tile_size_grid_search as tsgs
    from maxdiffusion.utils import wan_block_benchmark as wbb

    registered = {name for name in attention_flax.KERNEL_REGISTRY if name.startswith("ulysses_ring")}
    self.assertTrue(registered, "expected at least one registered ulysses_ring kernel")
    missing = registered - tsgs.ULYSSES_RING_ATTENTION_KERNELS
    self.assertEqual(
        missing,
        set(),
        f"these ulysses_ring kernels are missing from ULYSSES_RING_ATTENTION_KERNELS and would "
        f"get a mesh-independent tile length: {sorted(missing)}",
    )
    missing_wbb = registered - wbb._RING_VARIANTS
    self.assertEqual(
        missing_wbb,
        set(),
        f"these ulysses_ring kernels are missing from wan_block_benchmark._RING_VARIANTS: {sorted(missing_wbb)}",
    )

  def test_classified_kernels_scale_with_topology(self):
    from maxdiffusion.utils.tile_size_grid_search import local_tiled_seq_len

    full_seq = 6144
    for attention in sorted(attention_flax.KERNEL_REGISTRY):
      if not attention.startswith("ulysses_ring"):
        continue
      for cp in [2, 4, 8, 16]:
        u = attention_flax._largest_ulysses_shards_for_real_ring(cp, WAN_HEADS, WAN_HEADS)
        with self.subTest(attention=attention, cp=cp, u=u):
          local = local_tiled_seq_len(full_seq, attention, context_shards=cp, ulysses_shards=u)
          # Ulysses gathers u chunks of the context-local sequence.
          self.assertEqual(local, (full_seq // cp) * u)
          self.assertLess(
              local,
              full_seq,
              "a sharded mesh must tile less than the full sequence",
          )


class KCenteringResolutionTest(unittest.TestCase):
  """`use_k_centering="auto"` must pick the cheap choice per attention path."""

  def test_auto_is_on_for_non_ring_and_off_for_ring(self):
    for value in ("auto", "AUTO", None):
      self.assertTrue(attention_flax.resolve_k_centering(value, ring=False))
      self.assertFalse(attention_flax.resolve_k_centering(value, ring=True))

  def test_explicit_values_are_honoured_on_both_paths(self):
    for ring in (False, True):
      self.assertTrue(attention_flax.resolve_k_centering(True, ring=ring))
      self.assertFalse(attention_flax.resolve_k_centering(False, ring=ring))
      self.assertTrue(attention_flax.resolve_k_centering("true", ring=ring))
      self.assertFalse(attention_flax.resolve_k_centering("False", ring=ring))

  def test_rejects_unknown_strings(self):
    with self.assertRaises(ValueError):
      attention_flax.resolve_k_centering("sometimes", ring=False)


class GQAUnsupportedKernelsGuardTest(unittest.TestCase):
  """Kernels that do not support GQA must fail loudly when kv_heads != heads."""

  def test_flash_and_non_custom_ulysses_reject_gqa(self):
    base_kwargs = {
        "query": None,
        "key": None,
        "value": None,
        "heads": 8,
        "mesh": None,
        "axis_names_q": None,
        "axis_names_kv": None,
        "flash_block_sizes": None,
        "dtype": None,
        "kv_heads": 2,
    }
    for fn, extra in [
        (attention_flax._tpu_flash_attention, {"attention_kernel": "flash"}),
        (attention_flax._ulysses_attention, {"use_custom_kernel": False}),
        (attention_flax._ulysses_ring_attention, {}),
    ]:
      with self.subTest(fn=fn.__name__):
        with self.assertRaises(NotImplementedError):
          fn(**base_kwargs, **extra)


if __name__ == "__main__":
  unittest.main()
