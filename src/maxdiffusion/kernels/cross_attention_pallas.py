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

"""Fused short-KV cross-attention Pallas kernel for TPU.

Why
---
Wan's text cross-attention attends ~19K video tokens per shard to 512 cached
text tokens. That KV length is below `flash_min_seq_length`, so the layer runs
as XLA dot-product attention (`_apply_attention_dot`). XLA materialises the
`[B, heads, Sq, 512]` attention scores (in bf16 under Wan's default
`float32_qk_product: False`, or FP32 when `float32_qk_product: True`) and
softmax probabilities in HBM and streams them across QK^T, softmax, and PV,
while relayouting `[B, S, H*D]` to `[B, S, H, D]` and back around the two
einsums. At 720p/81 frames that is 3.94 ms per layer on v6e and 1.81 ms on
tpu7x, nearly all of it HBM traffic.

Here the whole KV of a block of heads stays resident in VMEM and each query
tile makes one pass: QK^T on the MXU, softmax in registers, PV on the MXU. HBM
sees Q once, K/V once per head block and the output once, all in the flat
`[B, S, H*D]` layout, so the relayouts disappear as well.

Numerics
--------
Scores and the softmax state stay in FP32 inside the kernel (accumulated in FP32
on the MXU), whereas Wan's default XLA fallback (`float32_qk_product: False`)
rounds the QK^T logits to bf16 before running a bf16 softmax. The kernel result
is within bf16 rounding of an FP32-score reference (and at least as close to
that reference as the XLA path), but not bit-identical, because:

  * the softmax is normalised after PV: the unnormalised probabilities are
    rounded to bf16 for the MXU and the FP32 PV product is divided by their row
    sum, where XLA rounds the normalised probabilities;
  * with `sum_mode="mxu"` the row sum also comes from the MXU, by appending a
    block of ones to each head of V. On a 256-wide MXU with `dim_head == 128`
    the extra columns are free, and the weights actually applied to V then sum
    to one up to FP32 accumulation;
  * the exponent is evaluated as `exp2((s - m) * scale * log2(e))`.

`max_mode="tile"` shifts every row of a query tile by the tile-wide maximum
instead of its own. Softmax is shift-invariant, so this is exact unless a row's
maximum sits more than ~87 (in scaled-logit units) below the tile's, where its
exponentials underflow. It exists to measure what the per-row reduction costs;
production uses `"row"`.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128
LOG2E = math.log2(math.e)

# Tile sizes, swept at the Wan shard shape (q [1, 18900, 5120] unpadded or
# [1, 18944, 5120] with `wan_seq_pad="lane"`, kv 512 tokens). `block_q` is an
# upper bound: the rows are split into `cdiv(Sq, block_q)` equal tiles rounded
# up to a multiple of 16, so with `DEFAULT_BLOCK_Q = 2400` both 18900 and 18944
# rows run as 8 tiles of 2368 instead of 7 full tiles plus a mostly empty
# eighth. Per-call time for 40 heads:
#   tpu7x: 0.546 ms at 1024 x 8 heads, 0.501 ms at 2368 x 8, 0.487 ms at 2368 x 10
#   v6e:   0.567 ms at 1024 x 8 heads, 0.514 ms at 2368 x 8, 0.512 ms at 2368 x 10
# against 1.81 ms (tpu7x) and 3.94 ms (v6e) for the XLA path inside the model.
DEFAULT_BLOCK_Q = 2400
DEFAULT_HEAD_BLOCK = 10
# v6e exposes 128 MiB of VMEM per core and tpu7x exposes 64 MiB per core; a
# 64 MiB default fits both platforms.
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
# The full KV of a head block is VMEM-resident, so this kernel is only meant
# for short KV sequences such as Wan's 512 text tokens.
MAX_KV_LEN = 4096

MAX_MODES = ("row", "tile")
SUM_MODES = ("mxu", "vpu")

# q @ k^T: contract the last axis of both operands.
_NT_DIMS = (((1,), (1,)), ((), ()))
# p @ v: standard matmul.
_NN_DIMS = (((1,), (0,)), ((), ()))


def _cross_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    *,
    head_block: int,
    dim_head: int,
    exp2_scale: float,
    max_mode: str,
    sum_mode: str,
    seq_len: int,
    block_q: int,
):
  ones = jnp.ones((v_ref.shape[1], dim_head), dtype=v_ref.dtype) if sum_mode == "mxu" else None
  row_valid = None
  if max_mode == "tile" and seq_len % block_q != 0:
    # Rows past the end of the ragged last tile hold stale VMEM. Per-row
    # softmax never mixes rows, but a tile-wide maximum would read them.
    rows = pl.program_id(2) * block_q + jax.lax.broadcasted_iota(jnp.int32, (block_q, 1), 0)
    row_valid = rows < seq_len

  for h in range(head_block):
    q = q_ref[0, :, pl.ds(h * dim_head, dim_head)]
    k = k_ref[0, :, pl.ds(h * dim_head, dim_head)]
    s = jax.lax.dot_general(q, k, _NT_DIMS, preferred_element_type=jnp.float32)

    if max_mode == "row":
      m = jnp.max(s, axis=-1, keepdims=True)
    else:
      s_for_max = s if row_valid is None else jnp.where(row_valid, s, -jnp.inf)
      m = jnp.max(jnp.max(s_for_max, axis=0, keepdims=True), axis=1, keepdims=True)
    p = jnp.exp2((s - m) * exp2_scale)

    v = v_ref[0, :, pl.ds(h * dim_head, dim_head)]
    if sum_mode == "mxu":
      v_aug = jnp.concatenate([v, ones], axis=-1)
      acc = jax.lax.dot_general(p.astype(v.dtype), v_aug, _NN_DIMS, preferred_element_type=jnp.float32)
      num, den = acc[:, :dim_head], acc[:, dim_head:]
    else:
      den = jnp.sum(p, axis=-1, keepdims=True)
      num = jax.lax.dot_general(p.astype(v.dtype), v, _NN_DIMS, preferred_element_type=jnp.float32)
    o_ref[0, :, pl.ds(h * dim_head, dim_head)] = (num / den).astype(o_ref.dtype)


def append_ones_per_head(v: jax.Array, heads: int, dim_head: int) -> jax.Array:
  """`[B, T, H*D]` -> `[B, T, H*2D]` with `dim_head` ones appended to each head."""
  batch, seq_kv, _ = v.shape
  v4 = v.reshape(batch, seq_kv, heads, dim_head)
  return jnp.concatenate([v4, jnp.ones_like(v4)], axis=-1).reshape(batch, seq_kv, heads * 2 * dim_head)


def cross_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    heads: int,
    dim_head: int = 128,
    scale: float | None = None,
    k_prescaled: bool = False,
    block_q: int = DEFAULT_BLOCK_Q,
    head_block: int = DEFAULT_HEAD_BLOCK,
    max_mode: str = "row",
    sum_mode: str = "mxu",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> jax.Array:
  """Unmasked multi-head attention against a short, fully VMEM-resident KV.

  Computes `softmax(q_h @ k_h^T * scale) @ v_h` for every head `h`, reading and
  writing the flat `[B, S, heads * dim_head]` layout directly.

  Args:
    q: Queries, `[B, Sq, heads * dim_head]`.
    k: Keys, `[B, Skv, heads * dim_head]`.
    v: Values, `[B, Skv, heads * dim_head]`.
    heads: Number of heads (MHA only; K/V carry the same heads as Q).
    dim_head: Per-head dimension; must be a multiple of 128.
    scale: Softmax scale; defaults to `dim_head ** -0.5`.
    k_prescaled: `k` was already multiplied by `scale`, so it is not reapplied.
    block_q: Upper bound on query rows per grid step. The rows are split into
      `cdiv(Sq, block_q)` equal 16-aligned tiles; the last may be ragged.
    head_block: Heads per grid step; must divide `heads`.
    max_mode: `"row"` (per-row softmax shift) or `"tile"` (see module docs).
    sum_mode: `"mxu"` takes the softmax denominator from the PV matmul via a
      ones-augmented V; `"vpu"` reduces it on the vector unit.
    vmem_limit_bytes: Scoped VMEM budget handed to Mosaic.
    interpret: Run in Pallas interpret mode (for CPU tests).

  Returns:
    Attention output, `[B, Sq, heads * dim_head]`, in `q.dtype`.
  """
  if max_mode not in MAX_MODES:
    raise ValueError(f"max_mode must be one of {MAX_MODES}, got {max_mode!r}.")
  if sum_mode not in SUM_MODES:
    raise ValueError(f"sum_mode must be one of {SUM_MODES}, got {sum_mode!r}.")
  if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
    raise ValueError(f"expected flat [B, S, H*D] inputs, got q={q.shape} k={k.shape} v={v.shape}.")
  if dim_head % NUM_LANES != 0:
    raise ValueError(f"dim_head must be a multiple of {NUM_LANES}, got {dim_head}.")
  if heads % head_block != 0:
    raise ValueError(f"head_block ({head_block}) must divide heads ({heads}).")
  batch, seq_q, feature = q.shape
  if feature != heads * dim_head:
    raise ValueError(f"q feature dim ({feature}) must equal heads ({heads}) * dim_head ({dim_head}).")
  if k.shape != v.shape or k.shape[0] != batch or k.shape[2] != feature:
    raise ValueError(f"k and v must both be [{batch}, Skv, {feature}], got k={k.shape} v={v.shape}.")
  seq_kv = k.shape[1]
  if seq_kv > MAX_KV_LEN:
    raise ValueError(f"KV length {seq_kv} exceeds MAX_KV_LEN ({MAX_KV_LEN}); use a flash kernel instead.")
  if seq_kv % 8 != 0:
    raise ValueError(f"KV length ({seq_kv}) must be a multiple of 8.")

  if scale is None:
    scale = dim_head**-0.5
  exp2_scale = (1.0 if k_prescaled else float(scale)) * LOG2E
  if block_q <= 0 or block_q % 8 != 0:
    raise ValueError(f"block_q ({block_q}) must be a positive multiple of 8.")
  if block_q >= seq_q:
    if seq_q % 8 != 0:
      raise ValueError(f"seq_q ({seq_q}) must be a multiple of 8 when block_q ({block_q}) >= seq_q.")
    block_q = seq_q
  else:
    # Same tile count, but spread evenly: 16-row alignment matches bf16
    # sublane packing and never exceeds `block_q` by more than 15 rows.
    even = pl.cdiv(seq_q, pl.cdiv(seq_q, block_q))
    block_q = min(seq_q, pl.cdiv(even, 16) * 16)
    if block_q % 8 != 0:
      raise ValueError(f"resolved block_q ({block_q}) must be a multiple of 8.")

  v_width = 2 * dim_head if sum_mode == "mxu" else dim_head
  num_q_blocks = pl.cdiv(seq_q, block_q)
  padded_rows = num_q_blocks * block_q
  itemsize = jnp.dtype(q.dtype).itemsize
  cost = pl.CostEstimate(
      flops=2 * batch * heads * padded_rows * seq_kv * (dim_head + v_width),
      transcendentals=batch * heads * padded_rows * seq_kv,
      bytes_accessed=itemsize * batch * 2 * (seq_q + seq_kv) * feature,
  )

  kernel = functools.partial(
      _cross_attention_kernel,
      head_block=head_block,
      dim_head=dim_head,
      exp2_scale=exp2_scale,
      max_mode=max_mode,
      sum_mode=sum_mode,
      seq_len=seq_q,
      block_q=block_q,
  )
  # The query-tile axis is innermost, so the K/V block index is unchanged
  # across it and each head block's KV is fetched from HBM once.
  return pl.pallas_call(
      kernel,
      grid=(batch, heads // head_block, num_q_blocks),
      in_specs=[
          pl.BlockSpec((1, block_q, head_block * dim_head), lambda b, h, i: (b, i, h)),
          pl.BlockSpec((1, seq_kv, head_block * dim_head), lambda b, h, i: (b, 0, h)),
          pl.BlockSpec((1, seq_kv, head_block * dim_head), lambda b, h, i: (b, 0, h)),
      ],
      out_specs=pl.BlockSpec((1, block_q, head_block * dim_head), lambda b, h, i: (b, i, h)),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes
      ),
      cost_estimate=cost,
      interpret=interpret,
  )(q, k, v)


def cross_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    heads: int,
    dim_head: int = 128,
    scale: float | None = None,
    k_prescaled: bool = False,
) -> jax.Array:
  """FP32 golden reference: exact softmax attention, rounded once at the end."""
  if scale is None:
    scale = dim_head**-0.5
  batch, seq_q, _ = q.shape
  seq_kv = k.shape[1]
  q4 = q.reshape(batch, seq_q, heads, dim_head).astype(jnp.float32)
  k4 = k.reshape(batch, seq_kv, heads, dim_head).astype(jnp.float32)
  v4 = v.reshape(batch, seq_kv, heads, dim_head).astype(jnp.float32)
  s = jnp.einsum("bqhd,bkhd->bhqk", q4, k4, precision=jax.lax.Precision.HIGHEST)
  if not k_prescaled:
    s = s * scale
  p = jax.nn.softmax(s, axis=-1)
  o = jnp.einsum("bhqk,bkhd->bqhd", p, v4, precision=jax.lax.Precision.HIGHEST)
  return o.reshape(batch, seq_q, heads * dim_head).astype(q.dtype)
