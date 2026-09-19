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

"""Fused RMSNorm + RoPE + head-transposition Pallas kernel for TPU.

Motivation (HLO/LLO profile, Wan 2.2 27B, 40 denoise steps)
-----------------------------------------------------------
The unfused producer in `fused_producers.fused_rmsnorm_rope` is three separate
HBM passes over the same 193 MB activation per tensor per layer:

  1. `multiply_subtract_fusion` -- FP32 RMSNorm over the 5120-wide feature axis.
  2. `reshape.1483`-`reshape.1486` -- the `[B, S, H*D] -> [B, H, S, D]` head
     transposition. This is *not* a free bitcast: `[S, H*D]` is tiled
     `(8, 128)` over `(S, H*D)` while `[H, S, D]` is tiled `(8, 128)` over
     `(S, D)`, so XLA emits a physical relayout copy (0.35 ms x 4 ops per
     2 steps on 16 cores = ~1.75 s per 40-step denoise).
  3. The RoPE elementwise chain, which reads the transposed tensor back.

XLA cannot fuse the three because RMSNorm reduces along the feature axis (which
spans *all* heads) while RoPE and the attention kernel need a head-major
layout. Commuting RoPE ahead of the transpose in pure JAX (experiment
`exp1_rope_pre_transpose`) is bit-exact but 3.8-4.3 s *slower*, because
broadcasting `cos`/`sin` across an interior `heads` axis destroys sublane
broadcast coalescing.

Two normalisation modes
-----------------------
`norm_mode="exact"` (default)
    The FP32 `mean(x**2)` reduction stays in XLA and only its `[B, S, 1]`
    result is handed to the kernel, which then does scale + RoPE + transpose in
    a single pass. Every floating-point operation is then *the same HLO op in
    the same order* as the reference, so the output is bit-identical by
    construction. Costs one extra streaming read of the activation.

`norm_mode="fused"`
    The reduction also happens inside the kernel, so the activation is read
    exactly once. Mosaic's reduction tree over the 5120-wide feature axis does
    not necessarily match XLA's, which can flip the final bf16 rounding: at the
    Wan shard shape this is a <=1 ULP difference on a very small fraction of
    elements. Use only where a hash-identical video is not required.

Numerical contract
------------------
Everything other than the reduction tree is bit-exact by construction:

  * RMSNorm keeps Flax's association `x * (rsqrt(var + eps) * scale)`. Folding
    left-to-right as `(x * rsqrt) * scale` rounds differently.
  * RoPE is evaluated in the activation dtype, exactly like the reference, via

        out[2i]   = q[2i]   * cos[i] + q[2i+1] * (-sin[i])
        out[2i+1] = q[2i+1] * cos[i] + q[2i]   * (+sin[i])

    `a + (-b) == a - b` and `a + b == b + a` hold exactly in IEEE-754, so this
    reproduces the reference's `q0*cos - q1*sin` / `q0*sin + q1*cos` exactly.
  * The `cos`/`sin` tables are rounded to the activation dtype *before* the
    lane duplication, so each lane holds precisely the value the reference
    multiplies by.

The pairwise `(q[2i], q[2i+1]) -> (q[2i+1], q[2i])` swap uses two circular lane
rotations and an even-lane select rather than a strided `x[..., 0::2]` gather,
which Mosaic cannot lower efficiently.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128

# Sequence tile sizes. Both paths now stream a full-width `[block_s, feature]`
# row tile, so VMEM scales as `block_s * feature * dtype_size` (times two for
# double buffering). The FP32 temporaries differ: "exact" builds them after the
# per-head slice so they are `[block_s, dim_head]`, while "fused" must hold a
# `[block_s, feature]` FP32 intermediate for the reduction. Hence the smaller
# default tile for "fused". Tuned on v6e at seq=18900, heads=40, dim_head=128.
DEFAULT_BLOCK_S_EXACT = 512
DEFAULT_BLOCK_S_FUSED = 256

# v6e and tpu7x both expose 128 MiB of VMEM per core. 64 MiB leaves ample room
# for the fused path's FP32 temporaries without starving the compiler.
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024

NORM_MODES = ("exact", "fused")
ROPE_ACCUM_MODES = ("dtype", "f32")


def _apply_rope(normed: jax.Array, cos: jax.Array, sin_signed: jax.Array, accum: str) -> jax.Array:
  """Combines `normed * cos + swap(normed) * sin` under a chosen rounding mode.

  The reference is `q0*cos - q1*sin`. Whether that rounds twice (once per
  product, then once for the sum) or only once depends on whether XLA contracts
  it into a fused multiply-add -- which it does under `jit` but not in eager
  execution. Both roundings are legitimate; only one matches any given
  reference build, so the caller must be able to pick.

    `accum="dtype"`: round each product to the activation dtype, then add.
        Matches an uncontracted reference.
    `accum="f32"`:   evaluate both products and the sum in FP32 and round once.
        Matches an FMA-contracted reference.
  """
  if accum == "f32":
    wide = normed.astype(jnp.float32)
    out = wide * cos.astype(jnp.float32) + _pair_swap(wide) * sin_signed.astype(jnp.float32)
    return out.astype(normed.dtype)
  return normed * cos + _pair_swap(normed) * sin_signed


def _pair_swap(x: jax.Array) -> jax.Array:
  """Swaps adjacent lane pairs: `[a0, a1, a2, a3, ...] -> [a1, a0, a3, a2, ...]`.

  Implemented with two circular rotations and an even-lane select. The strided
  `x[..., 0::2]` / `x[..., 1::2]` gather that the reference relies on XLA to
  handle is a lane-stride-2 relayout which Mosaic lowers very poorly, whereas
  `tpu.DynamicRotate` is a single cheap lane rotation.

  On TPU 7x, tpu.DynamicRotate requires 32-bit data, so we ensure 32-bit width
  during rotation.

  The rotation is circular, but because `dim_head` is even the wrap-around
  lanes land exactly where the swap needs them:
    * lane 0 takes `roll_left[0] = x[1]`, the partner of lane 0.
    * lane D-1 takes `roll_right[D-1] = x[D-2]`, the partner of lane D-1.
  """
  orig_dtype = x.dtype
  if orig_dtype != jnp.float32 and orig_dtype != jnp.int32:
    x = x.astype(jnp.float32)
  dim = x.shape[-1]
  axis = x.ndim - 1
  roll_right = pltpu.roll(x, 1, axis)  # roll_right[i] = x[i - 1]
  roll_left = pltpu.roll(x, dim - 1, axis)  # roll_left[i]  = x[i + 1]
  lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis)
  res = jnp.where(jax.lax.rem(lane, 2) == 0, roll_left, roll_right)
  return res.astype(orig_dtype)


def _rope_tables(freqs_cis: jax.Array, seq_len: int, dtype: jnp.dtype) -> Tuple[jax.Array, jax.Array]:
  """Expands `freqs_cis` into lane-aligned `cos` / signed-`sin` tables.

  Args:
    freqs_cis: Complex rotary embedding, shape `[1, 1, S, dim_head // 2]`.
    seq_len: Number of sequence positions to keep.
    dtype: Activation dtype. The tables are rounded to it *before* the
      duplication so every lane holds exactly the value the reference
      implementation multiplies by.

  Returns:
    `(cos_full, sin_signed)`, each `[1, seq_len, dim_head]`, with
    `cos_full[..., 2i] == cos_full[..., 2i+1] == cos[i]`,
    `sin_signed[..., 2i] == -sin[i]` and `sin_signed[..., 2i+1] == +sin[i]`.
  """
  cos = jnp.real(freqs_cis)[0, :, :seq_len, :].astype(dtype)
  sin = jnp.imag(freqs_cis)[0, :, :seq_len, :].astype(dtype)
  cos_full = jnp.repeat(cos, 2, axis=-1)
  sin_full = jnp.repeat(sin, 2, axis=-1)
  # [-1, +1, -1, +1, ...]; scaling a float by +-1 is exact.
  sign = jnp.tile(jnp.array([-1.0, 1.0], dtype=dtype), (sin_full.shape[-1] // 2,))
  return cos_full, sin_full * sign


def _exact_kernel(x_ref, scale_ref, rsqrt_ref, cos_ref, sin_ref, o_ref, *, dim_head: int, rope_accum: str, head_block: int):
  """Scale + RoPE + transpose for one `(sequence tile, head block)` step.

  The activation tile is fetched at *full feature width* and the per-head slice
  is taken in VMEM. Slicing the head out of HBM instead (via the input
  `BlockSpec`) would read `dim_head * 2 = 256` contiguous bytes per row out of
  a 10,240-byte row, i.e. a strided burst pattern that runs at a fraction of
  HBM speed. At full width the read is sequential, and because the block index
  does not depend on the head, Pallas fetches each tile exactly once.

  `head_block` heads are emitted per grid step. With one head per step the grid
  is `ceil(S/block_s) * heads` (2960 steps at the Wan shape) and each output DMA
  is only `block_s * dim_head` elements; the per-step overhead then dominates,
  costing ~45% over emitting all heads at once. The loop is unrolled in Python
  so every store addresses a statically known sub-block.

  The FP32 temporaries are created *after* the slice, so they are
  `[block_s, dim_head]` rather than `[block_s, heads * dim_head]`.
  """
  blk = pl.program_id(1)
  for i in range(head_block):
    # Pure lane-tile selection: `dim_head` is a multiple of NUM_LANES, so this
    # picks whole lane tiles and needs no relayout. `pl.multiple_of` supplies
    # the alignment fact Mosaic cannot infer from a dynamic product.
    offset = pl.multiple_of((blk * head_block + i) * dim_head, dim_head)
    x = x_ref[0, :, pl.ds(offset, dim_head)]

    # Flax's association: fold the scale into the reciprocal before applying it.
    mul = rsqrt_ref[0] * scale_ref[i, 0].astype(jnp.float32)
    normed = (x.astype(jnp.float32) * mul).astype(x.dtype)
    o_ref[0, i] = _apply_rope(normed, cos_ref[0], sin_ref[0], rope_accum)


def _fused_kernel(
    x_ref, scale_ref, cos_ref, sin_ref, o_ref, xn_ref, *, dim_head: int, eps: float, rope_accum: str, head_block: int
):
  """As `_exact_kernel`, but also computes the feature-axis reduction in VMEM."""
  blk = pl.program_id(1)

  @pl.when(blk == 0)
  def _normalize_tile():
    # Computed once per sequence tile and reused by every head, so the
    # feature-wide reduction is paid once rather than `heads` times.
    x = x_ref[0]
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    mul = jax.lax.rsqrt(var + eps) * scale_ref[0].astype(jnp.float32)
    xn_ref[...] = (x_f32 * mul).astype(x.dtype)

  for i in range(head_block):
    offset = pl.multiple_of((blk * head_block + i) * dim_head, dim_head)
    normed = xn_ref[:, pl.ds(offset, dim_head)]
    o_ref[0, i] = _apply_rope(normed, cos_ref[0], sin_ref[0], rope_accum)


def _run_exact(
    x, scale, cos_full, sin_signed, *, heads, dim_head, eps, block_s, vmem_limit_bytes, interpret, rope_accum, head_block
):
  batch, seq_len, feature = x.shape
  block_s = min(block_s, seq_len)

  # The reduction is left to XLA, expressed exactly as the reference expresses
  # it. That makes it bit-identical *provided* XLA emits the same reduction for
  # both graphs; see the module docstring on why that is checked empirically
  # rather than assumed.
  x_f32 = x.astype(jnp.float32)
  rsqrt_val = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)

  return pl.pallas_call(
      functools.partial(_exact_kernel, dim_head=dim_head, rope_accum=rope_accum, head_block=head_block),
      grid=(pl.cdiv(seq_len, block_s), heads // head_block),
      in_specs=[
          # Full-width and head-invariant: one sequential fetch per tile.
          pl.BlockSpec((1, block_s, feature), lambda s, h: (0, s, 0)),
          # Mosaic requires the second-minor block dimension to be a multiple
          # of 8 or to equal the array's. `scale` is carried as
          # `[heads, 1, dim_head]` so the unit axis satisfies the latter; a
          # `[heads, dim_head]` layout with a `(1, dim_head)` block does not
          # lower at all.
          pl.BlockSpec((head_block, 1, dim_head), lambda s, h: (h, 0, 0)),
          pl.BlockSpec((1, block_s, 1), lambda s, h: (0, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda s, h: (0, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda s, h: (0, s, 0)),
      ],
      out_specs=pl.BlockSpec((1, head_block, block_s, dim_head), lambda s, h: (0, h, s, 0)),
      out_shape=jax.ShapeDtypeStruct((batch, heads, seq_len, dim_head), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes
      ),
      interpret=interpret,
  )(x, scale.reshape(heads, 1, dim_head), rsqrt_val, cos_full, sin_signed)


def _run_fused(
    x, scale, cos_full, sin_signed, *, heads, dim_head, eps, block_s, vmem_limit_bytes, interpret, rope_accum, head_block
):
  batch, seq_len, feature = x.shape
  block_s = min(block_s, seq_len)

  # The input block index does not depend on the head-block axis, so Pallas
  # fetches each `[block_s, H*D]` tile from HBM exactly once and every head
  # reuses the normalised scratch.
  return pl.pallas_call(
      functools.partial(_fused_kernel, dim_head=dim_head, eps=eps, rope_accum=rope_accum, head_block=head_block),
      grid=(pl.cdiv(seq_len, block_s), heads // head_block),
      in_specs=[
          pl.BlockSpec((1, block_s, feature), lambda s, h: (0, s, 0)),
          pl.BlockSpec((1, feature), lambda s, h: (0, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda s, h: (0, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda s, h: (0, s, 0)),
      ],
      out_specs=pl.BlockSpec((1, head_block, block_s, dim_head), lambda s, h: (0, h, s, 0)),
      out_shape=jax.ShapeDtypeStruct((batch, heads, seq_len, dim_head), x.dtype),
      scratch_shapes=[pltpu.VMEM((block_s, feature), x.dtype)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes
      ),
      interpret=interpret,
  )(x, scale.reshape(1, feature), cos_full, sin_signed)


def fused_rmsnorm_rope_pallas(
    raw_q: jax.Array,
    raw_k: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    freqs_cis: jax.Array,
    q_heads: int = 40,
    kv_heads: int | None = None,
    dim_head: int = 128,
    eps: float = 1e-6,
    heads: int | None = None,
    norm_mode: str = "exact",
    rope_accum: str = "dtype",
    block_s: int | None = None,
    head_block: int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Fused FP32 RMSNorm + RoPE + head transposition on TPU via Pallas.

  Drop-in replacement for `fused_producers.fused_rmsnorm_rope`.

  Args:
    raw_q: Raw query projection, `[B, Sq, q_heads * dim_head]`.
    raw_k: Raw key projection, `[B, Sk, kv_heads * dim_head]`.
    q_norm_scale: RMSNorm scale for the query, `[q_heads * dim_head]`.
    k_norm_scale: RMSNorm scale for the key, `[kv_heads * dim_head]`.
    freqs_cis: Complex rotary embedding, `[1, 1, S, dim_head // 2]`.
    q_heads: Number of query heads.
    kv_heads: Number of key/value heads (defaults to `q_heads` for MHA).
    dim_head: Per-head dimension. Must be a multiple of 128 and even.
    eps: RMSNorm epsilon.
    heads: Deprecated alias for `q_heads`.
    norm_mode: `"exact"` leaves the FP32 feature-axis reduction to XLA, written
      exactly as the reference writes it; `"fused"` folds it into the kernel,
      reading the activation once but changing the summation order. Measured at
      the Wan shape, `"fused"` alters ~0.001% of elements by up to 3e-2, which
      is far too coarse for a hash-equality bar.
    rope_accum: Rounding of the RoPE combine. `"dtype"` rounds each product to
      the activation dtype before summing; `"f32"` keeps both products and the
      sum in FP32 and rounds once. The reference matches `"dtype"` when its
      multiply-adds are left uncontracted and `"f32"` when XLA contracts them
      into FMAs, which depends on the surrounding graph. Pick whichever
      reproduces the build you must match.
    block_s: Sequence tile size; defaults per `norm_mode`.
    head_block: Number of heads emitted per grid step. Must divide the head
      count of each tensor. Defaults to all of them, which measured fastest:
      one head per step leaves a 2960-step grid whose per-step overhead costs
      ~45%. Lower it only if VMEM is tight, since the output tile is
      `head_block * block_s * dim_head`.
    vmem_limit_bytes: Scoped VMEM budget handed to Mosaic.
    interpret: Run in Pallas interpret mode (for CPU tests).

  Returns:
    `(q_out, k_out)` of shapes `[B, q_heads, Sq, dim_head]` and
    `[B, kv_heads, Sk, dim_head]`.
  """
  if heads is not None:
    q_heads = heads
  kv_heads = q_heads if kv_heads is None else kv_heads

  if norm_mode not in NORM_MODES:
    raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}.")
  if rope_accum not in ROPE_ACCUM_MODES:
    raise ValueError(f"rope_accum must be one of {ROPE_ACCUM_MODES}, got {rope_accum!r}.")
  if dim_head % NUM_LANES != 0:
    raise ValueError(f"fused_rmsnorm_rope_pallas requires dim_head to be a multiple of {NUM_LANES}, got {dim_head}.")
  if dim_head % 2 != 0:
    raise ValueError(f"RoPE requires an even dim_head, got {dim_head}.")
  if head_block is not None:
    for name, n in (("q_heads", q_heads), ("kv_heads", kv_heads)):
      if n % head_block != 0:
        raise ValueError(f"head_block ({head_block}) must divide {name} ({n}).")

  _, seq_q, feature_q = raw_q.shape
  _, seq_k, feature_k = raw_k.shape
  if feature_q != q_heads * dim_head:
    raise ValueError(f"raw_q feature dim ({feature_q}) must equal q_heads ({q_heads}) * dim_head ({dim_head})")
  if feature_k != kv_heads * dim_head:
    raise ValueError(f"raw_k feature dim ({feature_k}) must equal kv_heads ({kv_heads}) * dim_head ({dim_head})")
  if freqs_cis.shape[-1] * 2 != dim_head:
    raise ValueError(f"freqs_cis last dim ({freqs_cis.shape[-1]}) must be dim_head // 2 ({dim_head // 2}).")

  if block_s is None:
    block_s = DEFAULT_BLOCK_S_EXACT if norm_mode == "exact" else DEFAULT_BLOCK_S_FUSED
  runner = _run_exact if norm_mode == "exact" else _run_fused

  cos_q, sin_q = _rope_tables(freqs_cis, seq_q, raw_q.dtype)
  if seq_k == seq_q and raw_k.dtype == raw_q.dtype:
    cos_k, sin_k = cos_q, sin_q
  else:
    cos_k, sin_k = _rope_tables(freqs_cis, seq_k, raw_k.dtype)

  common = {
      "dim_head": dim_head,
      "eps": eps,
      "block_s": block_s,
      "vmem_limit_bytes": vmem_limit_bytes,
      "interpret": interpret,
      "rope_accum": rope_accum,
  }
  q_out = runner(
      raw_q, q_norm_scale, cos_q, sin_q, heads=q_heads, head_block=q_heads if head_block is None else head_block, **common
  )
  k_out = runner(
      raw_k, k_norm_scale, cos_k, sin_k, heads=kv_heads, head_block=kv_heads if head_block is None else head_block, **common
  )
  return q_out, k_out


def rope_pair_swap_reference(x: jax.Array) -> jax.Array:
  """Pure-JAX twin of `_pair_swap`, used to pin the rotation identity in tests."""
  axis = x.ndim - 1
  roll_right = jnp.roll(x, 1, axis=axis)
  roll_left = jnp.roll(x, -1, axis=axis)
  lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis)
  return jnp.where(lane % 2 == 0, roll_left, roll_right)
