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

Fused single-pass block pooling of Q, K and V for Sol-Attn on TPU.

XLA lowers the reshape-and-mean pooling of `routing.pool_blocks` to a relayout
copy plus a reduction per tensor (~0.4 ms each for Wan 2.1 14B shapes on v6e).
This kernel reads each tensor once, straight from the unpadded [H, L, d]
layout, and writes the three block means.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_SUBLANES = 8


def slab_rows(block_q: int, block_kv: int) -> int:
  """Rows pooled per grid step; each output gets >= 8 pooled rows (sublane tiling)."""
  return math.lcm(NUM_SUBLANES * block_q, NUM_SUBLANES * block_kv)


def _pool_kernel(
    q_ref, k_ref, v_ref, qo_ref, ko_ref, vo_ref, kr_ref, *, seq_len: int, tile: int, block_q: int, block_kv: int
):
  """Block means of one `tile`-row slab of q (block_q) and k/v (block_kv), plus key block radii."""
  start = pl.program_id(1) * tile
  rows = start + lax.broadcasted_iota(jnp.int32, (tile, 1), 0)
  valid = rows < seq_len  # The last slab may run past the sequence end.

  def pool(x_ref, out_ref, block):
    x = jnp.where(valid, x_ref[...].astype(jnp.float32), 0.0)
    sums = x.reshape(tile // block, block, x.shape[-1]).sum(axis=1)
    block_start = start + lax.broadcasted_iota(jnp.int32, (tile // block, 1), 0) * block
    counts = jnp.clip(seq_len - block_start, 1, block).astype(jnp.float32)
    mean = sums / counts
    out_ref[...] = mean
    return x, mean

  pool(q_ref, qo_ref, block_q)
  k, k_mean = pool(k_ref, ko_ref, block_kv)
  pool(v_ref, vo_ref, block_kv)
  dev = k.reshape(tile // block_kv, block_kv, k.shape[-1]) - k_mean[:, None, :]
  dist2 = jnp.where(valid.reshape(tile // block_kv, block_kv, 1), dev * dev, 0.0).sum(axis=2, keepdims=True)
  kr_ref[...] = jnp.sqrt(dist2.max(axis=1))


def pool_qkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    block_q: int,
    block_kv: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Means of q over block_q tokens and of k, v over block_kv tokens.

  Args:
    q, k, v: Unpadded [H, L, d] tensors.
    block_q, block_kv: Pooling block sizes (multiples of 128).
    interpret: Run in Pallas interpret mode.

  Returns:
    float32 (q means [H, ceil(L/block_q), d], k and v means [H, ceil(L/block_kv), d],
    k radii [H, ceil(L/block_kv)]; see `routing.block_radius`). A ragged last
    block averages only its real tokens.
  """
  num_heads, seq_len, head_dim = q.shape
  tile = slab_rows(block_q, block_kv)
  if seq_len < tile:
    raise ValueError(f"pool_qkv needs at least {tile} tokens, got {seq_len}.")
  nq, nk = -(-seq_len // block_q), -(-seq_len // block_kv)
  in_spec = pl.BlockSpec((None, tile, head_dim), lambda h, i: (h, i, 0))
  kernel = functools.partial(_pool_kernel, seq_len=seq_len, tile=tile, block_q=block_q, block_kv=block_kv)
  q_pooled, k_pooled, v_pooled, k_radius = pl.pallas_call(
      kernel,
      grid=(num_heads, -(-seq_len // tile)),
      in_specs=[in_spec, in_spec, in_spec],
      out_specs=[
          pl.BlockSpec((None, tile // block_q, head_dim), lambda h, i: (h, i, 0)),
          pl.BlockSpec((None, tile // block_kv, head_dim), lambda h, i: (h, i, 0)),
          pl.BlockSpec((None, tile // block_kv, head_dim), lambda h, i: (h, i, 0)),
          pl.BlockSpec((None, tile // block_kv, 1), lambda h, i: (h, i, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((num_heads, nq, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((num_heads, nk, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((num_heads, nk, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((num_heads, nk, 1), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
      name="sol_pool_qkv",
  )(q, k, v)
  return q_pooled, k_pooled, v_pooled, k_radius[..., 0]
