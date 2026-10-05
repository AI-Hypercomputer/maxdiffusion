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

Attention-kernel family registry: the ONE place the kernel-name sets live.

pyconfig (sharding rules), attention_flax (cross-attention remap + iperm
dispatch), the tile-size search and the per-model block benchmarks all key
behaviour off "which family is this kernel in". Those used to be five
hand-maintained copies of the same lists, which drifted. Adding a kernel now
means adding it to the right set below; every consumer follows.

Deliberately stdlib-only: pyconfig and attention_flax import it, and it must
not drag either of them (or jax) into the other.
"""

# Pure ring kernels: the ring spans every context shard, so one local kernel
# invocation tiles ceil(full_seq / context_shards) tokens.
PURE_RING_ATTENTION_KERNELS = frozenset({
    "tokamax_ring",
    "tokamax_ring_custom",
})

# Internal-permutation ("iperm") kernels. With more than one ring shard they
# run ONE pallas_call for the whole ring and keep a whole-shard-resident Q
# block in VMEM, walked in `block_q_sub`-row chunks (optionally split into
# `block_q_outer`-row launches). Their VMEM footprint is therefore NOT the
# per-(bq, bkv) tile model the other kernels use -- see the iperm section of
# utils/tile_size_grid_search.py.
INTERNAL_PERM_KERNELS = frozenset({
    "ulysses_ring_custom_iperm",
    "ulysses_ring_custom_iperm_fixed_m",
    "ulysses_ring_custom_iperm_fixed_m_nocond",
    "ulysses_ring_custom_iperm_fixed_m_hybrid",
})

# Ulysses x ring (USP) kernels: heads all-to-all over `ulysses_shards`, then a
# ring over context_shards / ulysses_shards.
ULYSSES_RING_ATTENTION_KERNELS = (
    frozenset({
        "ulysses_ring",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
        "ulysses_ring_custom_fixed_m_per_q_block",
        "ulysses_ring_custom_bidir",
    })
    | INTERNAL_PERM_KERNELS
)

# Pure-Ulysses custom splash kernels (no ring).
ULYSSES_CUSTOM_ATTENTION_KERNELS = frozenset({
    "ulysses_custom",
    "ulysses_custom_fixed_m",
    "ulysses_custom_fixed_m_per_q_block",
})

# Self-attention-only kernels: cross-attention under these is remapped to flash
# with a local (unsharded) KV.
CROSS_ATTENTION_REMAPPED_TO_FLASH_KERNELS = (
    PURE_RING_ATTENTION_KERNELS | ULYSSES_RING_ATTENTION_KERNELS | ULYSSES_CUSTOM_ATTENTION_KERNELS
)


def auto_block_q_sub(block_q_outer: int, lane: int = 128) -> int:
  """The iperm dispatch's default `block_q_sub` for a given outer Q block.

  Largest `lane`-multiple divisor of `block_q_outer` that is <= max(lane,
  floor_lane(block_q_outer / 4)) -- i.e. at least ~4 sub-chunks so the score
  tile stays well under VMEM while the resident Q block amortises the KV
  stream. attention_flax's iperm dispatch calls THIS function, and the tile
  search uses it to know what an unset `block_q_sub` will resolve to, so the
  two can never disagree.
  """
  ceiling = max(lane, (block_q_outer // 4) // lane * lane)
  q_sub = lane
  d = lane
  while d <= block_q_outer:
    if block_q_outer % d == 0:
      if d <= ceiling:
        q_sub = d
      else:
        break
    d += lane
  return q_sub
