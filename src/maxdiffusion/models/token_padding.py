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

"""Static spec for masked padding tokens in WAN self-attention.

Inputs are built in natural order (`num_real` real tokens followed by padding).
`layout` selects the physical sequence order inside the transformer blocks:
  - "end": padding stays at the sequence end (per-chunk valid lengths differ).
  - "shard": each ring chunk gets `num_real / num_chunks` real tokens followed
    by its share of padding, so every chunk shares the unpadded valid prefix.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

LAYOUT_END = "end"
LAYOUT_SHARD = "shard"
_LAYOUTS = (LAYOUT_END, LAYOUT_SHARD)


class TokenPadding(NamedTuple):
  """Static spec of masked padding tokens in the self-attention sequence."""

  num_real: int
  num_total: int
  layout: str = LAYOUT_END
  num_chunks: int = 1
  mask: bool = True

  @property
  def num_pad(self) -> int:
    return self.num_total - self.num_real

  def validate(self) -> "TokenPadding":
    if self.layout not in _LAYOUTS:
      raise ValueError(f"TokenPadding.layout must be one of {_LAYOUTS}, got {self.layout!r}.")
    if not 0 < self.num_real <= self.num_total:
      raise ValueError(f"TokenPadding needs 0 < num_real <= num_total, got {self.num_real}, {self.num_total}.")
    if self.num_chunks < 1 or self.num_total % self.num_chunks:
      raise ValueError(f"num_total={self.num_total} must split into num_chunks={self.num_chunks} equal ring chunks.")
    if self.layout == LAYOUT_SHARD and (self.num_real % self.num_chunks or self.num_pad % self.num_chunks):
      raise ValueError(
          f"layout='shard' needs num_real={self.num_real} and num_pad={self.num_pad} "
          f"divisible by num_chunks={self.num_chunks}."
      )
    return self

  @property
  def chunk_len(self) -> int:
    return self.num_total // self.num_chunks

  def natural_to_physical(self) -> Optional[np.ndarray]:
    """Gather indices `perm`: physical[j] = natural[perm[j]]. None means identity."""
    if self.layout == LAYOUT_END:
      return None
    real_per_chunk = self.num_real // self.num_chunks
    pad_per_chunk = self.num_pad // self.num_chunks
    parts = []
    for c in range(self.num_chunks):
      parts.append(np.arange(c * real_per_chunk, (c + 1) * real_per_chunk))
      parts.append(np.arange(self.num_real + c * pad_per_chunk, self.num_real + (c + 1) * pad_per_chunk))
    return np.concatenate(parts).astype(np.int32)

  def physical_to_natural(self) -> Optional[np.ndarray]:
    """Inverse of `natural_to_physical` (natural[i] = physical[inv[i]])."""
    perm = self.natural_to_physical()
    if perm is None:
      return None
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv

  def physical_valid_mask(self) -> np.ndarray:
    """Bool [num_total]: True where the PHYSICAL token is a real token."""
    perm = self.natural_to_physical()
    natural_index = np.arange(self.num_total) if perm is None else perm
    return natural_index < self.num_real

  def chunk_valid_lengths(self) -> Tuple[int, ...]:
    """Valid-prefix length of each ring chunk in physical order."""
    if self.layout == LAYOUT_SHARD:
      return (self.num_real // self.num_chunks,) * self.num_chunks
    c = self.chunk_len
    return tuple(int(min(max(self.num_real - i * c, 0), c)) for i in range(self.num_chunks))
