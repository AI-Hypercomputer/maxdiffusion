"""CPU checks for the attention-config guards across TPU topologies.

Guard 1: a non-ring ulysses kernel must REJECT a ulysses_shards it cannot honour.
Guard 2: a ring kernel must WARN when it degenerates to R=1, and the remedy it
         suggests must be valid for the actual mesh and head count.
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


class KernelClassificationDriftTest(unittest.TestCase):
  """Every registered ulysses-ring kernel must be classified for tile sizing.

  `local_tiled_seq_len` silently falls through to `return full_seq` for any
  attention name it does not recognise, which yields a mesh-independent (and
  therefore wrong) tile length. A newly registered ring kernel must not be able
  to slip through that default.
  """

  def test_all_registered_ulysses_ring_kernels_are_classified(self):
    from maxdiffusion.utils import tile_size_grid_search as tsgs

    registered = {name for name in attention_flax.KERNEL_REGISTRY if name.startswith("ulysses_ring")}
    self.assertTrue(registered, "expected at least one registered ulysses_ring kernel")
    missing = registered - tsgs.ULYSSES_RING_ATTENTION_KERNELS
    self.assertEqual(
        missing,
        set(),
        f"these ulysses_ring kernels are missing from ULYSSES_RING_ATTENTION_KERNELS and would "
        f"get a mesh-independent tile length: {sorted(missing)}",
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
          self.assertLess(local, full_seq, "a sharded mesh must tile less than the full sequence")


if __name__ == "__main__":
  unittest.main()
