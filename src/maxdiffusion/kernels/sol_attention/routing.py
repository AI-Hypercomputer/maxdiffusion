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

Sol-Attn block pooling and query-dependent threshold routing.

Implements Sec. 3.1 of "Sol-Attn: Accelerating Video Generation Inference via
On-the-Fly Attention Sparsification" (arXiv:2607.24027):

  s_ij  = mean(Q_i) . mean(K_j) * scale                      (Eq. 3)
  tau_i = mu_i + beta * sigma_i,  S_i = {j : s_ij > tau_i}   (Eq. 4)

The functions here are pure JAX, model agnostic, and shared by the Pallas TPU
kernel and the dense reference so that both see bit-identical routing.

TPU note: the paper fuses routing into the GPU kernel to avoid materializing
the N x N block-proxy map in HBM. With TPU-friendly blocks (>=128 tokens) the
map is tiny (e.g. 592 x 592 per head for Wan 2.1 14B 720p), so it is computed
here in a cheap XLA pre-pass. mu_i / sigma_i are taken directly from the
materialized row, which equals the exact moment estimator of Eq. (5).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax
import jax.numpy as jnp

# Finite "minus infinity" keeps exp() and max() NaN-free on TPU.
MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


@dataclasses.dataclass(frozen=True)
class SolBlockSizes:
  """Token granularity of Sol-Attn.

  Attributes:
    block_q: Query tokens per routing block (and per kernel query tile).
    block_kv: Key/value tokens per block. Selected blocks are attended
      exactly; unselected blocks are approximated by their pooled key.
  """

  block_q: int = 1024
  block_kv: int = 256

  def __post_init__(self):
    for name in ("block_q", "block_kv"):
      value = getattr(self, name)
      if value <= 0 or value % 128:
        raise ValueError(f"Sol-Attn {name} must be a positive multiple of 128, got {value}.")

  @property
  def seq_multiple(self) -> int:
    return math.lcm(self.block_q, self.block_kv)


def padded_length(seq_len: int, block_sizes: SolBlockSizes) -> int:
  multiple = block_sizes.seq_multiple
  return -(-seq_len // multiple) * multiple


def pad_sequence(x: jax.Array, padded_len: int) -> jax.Array:
  """Zero-pads axis -2 (sequence) of `x` to `padded_len`."""
  pad = padded_len - x.shape[-2]
  if pad == 0:
    return x
  widths = [(0, 0)] * x.ndim
  widths[-2] = (0, pad)
  return jnp.pad(x, widths)


def block_counts(seq_len: int, padded_len: int, block: int) -> jax.Array:
  """Number of real (non-padding) tokens in each block, shape [padded_len // block]."""
  starts = jnp.arange(padded_len // block, dtype=jnp.int32) * block
  return jnp.clip(seq_len - starts, 0, block)


def pool_blocks(x: jax.Array, seq_len: int, block: int) -> jax.Array:
  """Mean over the real tokens of each block.

  Args:
    x: Zero-padded tensor [..., padded_len, d].
    seq_len: Number of real tokens.
    block: Block size along the sequence.

  Returns:
    float32 tensor [..., padded_len // block, d]. Fully padded blocks are 0.
  """
  *lead, padded_len, d = x.shape
  sums = x.astype(jnp.float32).reshape(*lead, padded_len // block, block, d).sum(axis=-2)
  counts = block_counts(seq_len, padded_len, block).astype(jnp.float32)
  return sums / jnp.maximum(counts, 1.0)[:, None]


def block_proxy_scores(q_pooled: jax.Array, k_pooled: jax.Array, scale: float) -> jax.Array:
  """Pre-softmax block-proxy logits s_ij (Eq. 3), shape [..., Nq, Nk]."""
  return (
      jnp.einsum(
          "...qd,...kd->...qk",
          q_pooled.astype(jnp.float32),
          k_pooled.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )


def threshold_route(
    scores: jax.Array,
    kv_counts: jax.Array,
    beta: float,
    always_exact: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
  """Query-dependent threshold routing (Eq. 4).

  Args:
    scores: Block-proxy logits [..., Nq, Nk].
    kv_counts: Real tokens per key block [Nk]; blocks with 0 tokens are padding
      and are neither selected nor approximated.
    beta: Standardized cutoff. Under the Gaussian reference the selected
      density is 1 - Phi(beta) (beta=1.0 -> ~16% density, ~84% sparsity).
    always_exact: Optional bool [Nk] (or broadcastable to [..., Nq, Nk]) of key
      blocks that always take the exact branch, e.g. text/sink tokens in
      MMDiT-style joint attention.

  Returns:
    (selected, tau): bool [..., Nq, Nk] selection mask and float32 [..., Nq]
    thresholds.
  """
  valid = kv_counts > 0
  num_valid = jnp.maximum(jnp.sum(valid), 1).astype(jnp.float32)
  masked = jnp.where(valid, scores, 0.0)
  mu = jnp.sum(masked, axis=-1) / num_valid
  centered = jnp.where(valid, scores - mu[..., None], 0.0)
  sigma = jnp.sqrt(jnp.sum(centered * centered, axis=-1) / num_valid)
  tau = mu + beta * sigma
  selected = jnp.logical_and(scores > tau[..., None], valid)
  if always_exact is not None:
    selected = jnp.logical_or(selected, jnp.logical_and(always_exact, valid))
  return selected, tau


def compact_selection(selected: jax.Array, row_width: int) -> jax.Array:
  """Packs a selection mask into per-row index lists for the kernel.

  Row layout: [count, j_0, j_1, ..., j_{count-1}, <don't care>...], padded to
  `row_width` (>= Nk + 1) int32 entries. Selected indices are ascending so the
  kernel walks HBM in order.
  """
  nk = selected.shape[-1]
  if row_width < nk + 1:
    raise ValueError(f"row_width={row_width} must be >= Nk + 1 = {nk + 1}.")
  j = jnp.arange(nk, dtype=jnp.int32)
  # Selected indices sort first; the modulo recovers the index for both groups.
  order = jnp.sort(jnp.where(selected, j, nk + j), axis=-1) % nk
  counts = jnp.sum(selected, axis=-1, dtype=jnp.int32)
  rows = jnp.concatenate([counts[..., None], order], axis=-1)
  widths = [(0, 0)] * (rows.ndim - 1) + [(0, row_width - nk - 1)]
  return jnp.pad(rows, widths)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SolRouting:
  """Everything the exact/approximate branches need, derived from Q and K.

  Shapes use H for the flattened batch*heads axis.
  """

  selected: jax.Array  # bool [H, Nq, Nk]
  tau: jax.Array  # f32 [H, Nq]
  kv_counts: jax.Array  # int32 [Nk]
  k_pooled: jax.Array  # f32 [H, Nk, d]   mean key per block
  v_pooled: jax.Array  # f32 [H, Nk, d]   mean value per block

  def density(self) -> jax.Array:
    """Fraction of (query block, real key block) pairs computed exactly."""
    valid = (self.kv_counts > 0).astype(jnp.float32)
    return jnp.sum(self.selected * valid) / (jnp.sum(valid) * self.selected.shape[0] * self.selected.shape[1])


def compute_routing(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    seq_len: int,
    scale: float,
    beta: float,
    block_sizes: SolBlockSizes,
    always_exact: Optional[jax.Array] = None,
) -> SolRouting:
  """Pools Q/K/V into blocks and routes key blocks per query block.

  Args:
    q, k, v: Zero-padded [H, padded_len, d] tensors.
    seq_len: Number of real tokens.
    scale: Attention logit scale.
    beta: Standardized cutoff (see `threshold_route`).
    block_sizes: Block granularity.
    always_exact: Optional bool [Nk] of key blocks forced into the exact branch.
  """
  q_pooled = pool_blocks(q, seq_len, block_sizes.block_q)
  k_pooled = pool_blocks(k, seq_len, block_sizes.block_kv)
  v_pooled = pool_blocks(v, seq_len, block_sizes.block_kv)
  kv_counts = block_counts(seq_len, k.shape[-2], block_sizes.block_kv)
  scores = block_proxy_scores(q_pooled, k_pooled, scale)
  selected, tau = threshold_route(scores, kv_counts, beta, always_exact)
  return SolRouting(
      selected=selected,
      tau=tau,
      kv_counts=kv_counts,
      k_pooled=k_pooled,
      v_pooled=v_pooled,
  )


def approx_bias(routing: SolRouting) -> jax.Array:
  """Additive logit bias for the zeroth-order approximate branch.

  For an unselected block j with c_j real tokens, Eqs. (9)-(10) contribute
  c_j * exp(q . Kbar_j) to the denominator and exp(q . Kbar_j) * sum(V_j) to the
  numerator, i.e. one pooled key with logit q . Kbar_j + log(c_j) and value
  mean(V_j). Selected and padding blocks get MASK_VALUE (no approx term).

  Returns:
    float32 [H, Nq, Nk].
  """
  log_counts = jnp.log(jnp.maximum(routing.kv_counts, 1).astype(jnp.float32))
  keep = jnp.logical_and(jnp.logical_not(routing.selected), routing.kv_counts > 0)
  return jnp.where(keep, log_counts, MASK_VALUE)
