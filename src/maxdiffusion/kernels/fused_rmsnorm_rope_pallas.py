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

Why
---
The unfused producer in `fused_producers.fused_rmsnorm_rope` makes three HBM
passes over the same 193 MB activation, per tensor per layer: FP32 RMSNorm over
the 5120-wide feature axis; the `[B, S, H*D] -> [B, H, S, D]` head transposition;
then the RoPE elementwise chain reading the result back.

The transposition is not a free bitcast -- `[S, H*D]` is tiled `(8, 128)` over
`(S, H*D)` while `[H, S, D]` is tiled over `(S, D)` -- so XLA emits a physical
relayout copy: 0.35 ms x 4 ops per 2 steps on 16 cores, ~1.75 s per 40-step
denoise.

XLA cannot fuse the three, because RMSNorm reduces along the feature axis (which
spans *all* heads) while RoPE and attention need a head-major layout. Commuting
RoPE ahead of the transpose in pure JAX (measured in an out-of-tree experiment)
is bit-exact but 3.8-4.3 s *slower*: broadcasting `cos`/`sin` across an interior
`heads` axis destroys sublane broadcast coalescing.

Normalisation modes
-------------------
`norm_mode="exact"` (default)
    The FP32 `mean(x**2)` reduction stays in XLA; the kernel gets only its
    `[B, S, 1]` result, then does scale + RoPE + transpose in one pass. Every op
    is the same HLO op in the same order as the reference, so the kernel is
    bit-identical to the *separately jitted* `fused_rmsnorm_rope` producer
    (asserted at 0 ULP in the tests). That is a kernel-level guarantee only:
    inside the full 40-layer graph XLA fuses the unfused producer with its
    neighbours and rounds it differently, so end-to-end output is equivalent
    but not identical (v6e 720p/81f, same seed: 53.8 dB PSNR after 1 denoise
    step, 34.1 dB after 40; see `_fused_rope_producer`). Costs one extra
    streaming read.

`norm_mode="fused"`
    The reduction moves into the kernel, so the activation is read exactly once.
    Mosaic's reduction tree need not match XLA's, which can flip the final bf16
    rounding: <=1 ULP on a small fraction of elements at the Wan shard shape.
    Use only where a hash-identical video is not required.

Numerical contract
------------------
Apart from the reduction tree and the RoPE combine's rounding (selected by
`rope_accum`, see `resolve_rope_accum`), every op matches the reference:

  * RMSNorm keeps Flax's association `x * (rsqrt(var + eps) * scale)`; folding
    left-to-right as `(x * rsqrt) * scale` rounds differently.
  * RoPE is evaluated either in the activation dtype (`rope_accum="dtype"`) or
    in FP32 with a single rounding (`rope_accum="f32"`) as
    `out[2i]   = q[2i]  *cos[i] + q[2i+1]*(-sin[i])` and
    `out[2i+1] = q[2i+1]*cos[i] + q[2i]  *(+sin[i])`, which reproduces the
    reference's rounding on measured platforms (`v6e`, `tpu7x`, `XLA:CPU`):
    `a + (-b) == a - b` and `a + b == b + a` hold in IEEE-754.
  * `cos`/`sin` are rounded to the activation dtype *before* lane duplication,
    so each lane holds precisely the value the reference multiplies by.

The pairwise `(q[2i], q[2i+1]) -> (q[2i+1], q[2i])` swap uses two circular lane
rotations and an even-lane select, not a strided `x[..., 0::2]` gather, which
Mosaic cannot lower efficiently.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion import wan_runtime_options

NUM_LANES = 128

# Kernel-level default sequence tile sizes (production overrides this with
# `fused_rope_block_s = 1024` from the Wan YAMLs; in `fused` mode `block_s` is
# capped at 256). Both paths stream a full-width `[block_s, feature]` row tile,
# so VMEM scales as `block_s * feature * dtype_size` (times two for double
# buffering). The FP32 temporaries differ: "exact" builds them after the
# per-head slice so they are `[block_s, dim_head]`, while "fused" must hold a
# `[block_s, feature]` FP32 intermediate for the reduction.
DEFAULT_BLOCK_S_EXACT = 512
DEFAULT_BLOCK_S_FUSED = 256

# v6e exposes 128 MiB of VMEM per core and tpu7x exposes 64 MiB. 64 MiB fits
# both (the full tpu7x budget).
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024

NORM_MODES = ("exact", "fused")
ROPE_ACCUM_MODES = ("dtype", "f32")


# Rounding conventions that have actually been measured, as
# (device_kind substring, accumulation mode). A platform is on this list only
# if the kernel was shown to be bit-identical to the XLA producer under that
# mode at the production shape. Anything absent gets a best-effort default and
# is reported as unmeasured by `rope_accum_is_measured`, because the honest
# claim there is "close", not "identical": on TPU v4, for instance, no mode is
# bit-identical -- the closest is "dtype" against a compiled reference, which
# still differs on 20.9% of elements by up to one bf16 ULP.
_MEASURED_TPU_ROUNDING = (
    ("7x", "dtype"),
    ("v6", "f32"),
)
_UNMEASURED_TPU_DEFAULT = "f32"


def _rope_accum_devices(mesh=None):
  """Returns the device list backing the resolution, or None if no backend."""
  try:
    return list(mesh.devices.flat) if mesh is not None else jax.devices()
  except RuntimeError:
    return None


def resolve_rope_accum(mesh=None) -> str:
  """Returns the RoPE accumulation mode this process will actually compile with.

  The default is platform-dependent, because the mode has to match however the
  local compiler chooses to round the reference's RoPE multiply-add. Matching
  that contraction is what yields 0-ULP output against the XLA producer.

    * tpu7x: XLA emits a native bf16 multiply-add -> "dtype".
    * TPU v6: XLA contracts the multiply-add into an FP32 FMA -> "f32".
    * Off TPU: XLA:CPU emits the multiply and the add as separate rounded
      ops -> "dtype". Measured, not assumed: on CPU the eager and jitted
      references agree to 0 ULP and only "dtype" matches either. This branch is
      reached only by Pallas interpret-mode tests, since production falls back
      to the unfused XLA producer off TPU.
    * Other TPU generations: "f32" as a best effort. See
      `rope_accum_is_measured` before relying on bit-exactness there.

  A non-"auto" `wan_rope_accum` config value (legacy env: `WAN_ROPE_ACCUM`)
  overrides all of it.

  This resolver is shared by the production call site and by the AOT cache
  fingerprint. Keep it that way: the mode changes the lowered graph, so a
  duplicated copy of this rule that drifts from the real one would let an
  executable compiled under one rounding mode be served for another.

  Args:
    mesh: Mesh whose devices determine the platform default. Falls back to
      `jax.devices()` when None. Pass the same mesh the kernel runs under.

  Returns:
    One of `ROPE_ACCUM_MODES`.
  """
  override = wan_runtime_options.get("wan_rope_accum")
  if override != "auto":
    if override not in ROPE_ACCUM_MODES:
      raise ValueError(f"wan_rope_accum must be 'auto' or one of {ROPE_ACCUM_MODES}, got {override!r}")
    return override
  devices = _rope_accum_devices(mesh)
  if devices is None:
    # No backend (e.g. metadata built before device init); assume the
    # conservative FP32 contraction.
    return _UNMEASURED_TPU_DEFAULT
  if not any(getattr(d, "platform", "") == "tpu" for d in devices):
    return "dtype"
  kinds = [getattr(d, "device_kind", "").lower() for d in devices]
  for fragment, mode in _MEASURED_TPU_ROUNDING:
    if any(fragment in kind for kind in kinds):
      return mode
  return _UNMEASURED_TPU_DEFAULT


def rope_accum_is_measured(mesh=None) -> bool:
  """Whether `resolve_rope_accum` is returning a convention verified on this platform.

  False means the kernel still computes the right value but may not be
  *bit-identical* to the XLA producer, because no accumulation mode has been
  shown to reproduce this hardware's rounding. Tests that assert 0 ULP should
  consult this and skip rather than assert something untrue; callers that need
  exact reproducibility across a kernel/producer switch should treat it as a
  warning that they do not have it here.

  An explicit (non-"auto") `wan_rope_accum` counts as measured: setting it is an assertion
  by the caller that they know which convention this platform wants, and
  silently relaxing a bound underneath that would defeat the override.
  """
  if wan_runtime_options.get("wan_rope_accum") != "auto":
    return True
  devices = _rope_accum_devices(mesh)
  if devices is None:
    return False
  if not any(getattr(d, "platform", "") == "tpu" for d in devices):
    # Only XLA:CPU was measured; a GPU backend has not been.
    return all(getattr(d, "platform", "") == "cpu" for d in devices)
  kinds = [getattr(d, "device_kind", "").lower() for d in devices]
  return any(fragment in kind for fragment, _ in _MEASURED_TPU_ROUNDING for kind in kinds)


def with_xla_backward(fused_fn, xla_fn):
  """Makes a Pallas producer differentiable by transposing an equivalent XLA graph.

  `pallas_call` has no transpose rule, so a fused producer in a training graph
  fails at `jax.grad` time, from inside the autodiff machinery and with no
  indication that the fusion is what broke. This keeps the kernel on the forward
  pass and routes the backward pass through `xla_fn`, which must compute the same
  function by unfused primitives. The substitution is exact wherever the two
  forwards agree -- the property `rope_accum_is_measured` tracks. It costs one
  extra unfused forward per step, since the backward re-runs it to linearise.

  Args:
    fused_fn: Callable over differentiable array arguments only (bind static
      configuration with `functools.partial` first).
    xla_fn: Same signature and same mathematics, differentiable.

  Returns:
    A callable equivalent to `fused_fn` whose VJP is that of `xla_fn`.
  """

  @jax.custom_vjp
  def wrapped(*args):
    return fused_fn(*args)

  def _fwd(*args):
    return fused_fn(*args), args

  def _bwd(residual, cotangents):
    return jax.vjp(xla_fn, *residual)[1](cotangents)

  wrapped.defvjp(_fwd, _bwd)
  return wrapped


def _apply_rope(
    normed: jax.Array,
    cos: jax.Array,
    sin_signed: jax.Array,
    accum: str,
    even_mask: jax.Array | None = None,
) -> jax.Array:
  """Combines `normed * cos + swap(normed) * sin` under a chosen rounding mode.

  The reference is `q0*cos - q1*sin`. Whether XLA contracts that into an FP32
  fused multiply-add (rounding once) or emits separate rounded ops is
  platform-dependent: `v6e` contracts under `jit`, while `tpu7x` and `XLA:CPU`
  do not (and eager vs `jit` can also differ; see `resolve_rope_accum`). Both
  roundings are legitimate; only one matches any given reference build, so the
  caller must be able to pick.

    `accum="dtype"`: round each product to the activation dtype, then add.
        Matches an uncontracted reference.
    `accum="f32"`:   evaluate both products and the sum in FP32 and round once.
        Matches an FMA-contracted reference.
  """
  if accum == "f32":
    wide = normed.astype(jnp.float32)
    out = wide * cos.astype(jnp.float32) + _pair_swap(wide, even_mask=even_mask) * sin_signed.astype(jnp.float32)
    return out.astype(normed.dtype)
  return normed * cos + _pair_swap(normed, even_mask=even_mask) * sin_signed


def _pair_swap(x: jax.Array, even_mask: jax.Array | None = None) -> jax.Array:
  """Swaps adjacent lane pairs: `[a0, a1, a2, a3, ...] -> [a1, a0, a3, a2, ...]`.

  Implemented with two circular rotations and an even-lane select. The strided
  `x[..., 0::2]` / `x[..., 1::2]` gather that the reference relies on XLA to
  handle is a lane-stride-2 relayout which Mosaic lowers very poorly, whereas
  `tpu.DynamicRotate` is a single cheap lane rotation.

  Rotation is done at 32-bit width on all platforms (required by
  `tpu.DynamicRotate` on TPU 7x); the round-trip cast is exact.

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
  if even_mask is None:
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis)
    even_mask = jax.lax.rem(lane, 2) == 0
  res = jnp.where(even_mask, roll_left, roll_right)
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


def _exact_kernel(
    x_ref,
    scale_ref,
    rsqrt_ref,
    cos_ref,
    sin_ref,
    o_ref,
    *,
    dim_head: int,
    rope_accum: str,
    head_block: int,
    prescale: float = 1.0,
):
  """Scale + RoPE + transpose for one `(batch, sequence tile, head block)` step.

  The activation tile is fetched at *full feature width* and the per-head slice
  is taken in VMEM. Slicing the head out of HBM instead (via the input
  `BlockSpec`) would read `dim_head * 2 = 256` contiguous bytes per row out of
  a 10,240-byte row, i.e. a strided burst pattern that runs at a fraction of
  HBM speed. At full width the read is sequential, and because the block index
  does not depend on the head, Pallas fetches each tile exactly once.

  `head_block` heads are emitted per grid step. With one head per step the grid
  is `B * ceil(S/block_s) * heads` and each output DMA is only `block_s * dim_head`
  elements; the per-step overhead then dominates, costing ~45% over emitting
  all heads at once. The loop is unrolled in Python so every store addresses a
  statically known sub-block.

  The FP32 temporaries are created *after* the slice, so they are
  `[block_s, dim_head]` rather than `[block_s, heads * dim_head]`.
  """
  blk = pl.program_id(2)
  rsqrt_val = rsqrt_ref[0]
  cos = cos_ref[0].astype(jnp.float32) if rope_accum == "f32" else cos_ref[0]
  sin = sin_ref[0].astype(jnp.float32) if rope_accum == "f32" else sin_ref[0]
  lane = jax.lax.broadcasted_iota(jnp.int32, cos.shape, cos.ndim - 1)
  even_mask = jax.lax.rem(lane, 2) == 0
  prescale_val = jnp.asarray(prescale, o_ref.dtype) if prescale != 1.0 else None

  for i in range(head_block):
    # Pure lane-tile selection: `dim_head` is a multiple of NUM_LANES, so this
    # picks whole lane tiles and needs no relayout. `pl.multiple_of` supplies
    # the alignment fact Mosaic cannot infer from a dynamic product.
    offset = pl.multiple_of((blk * head_block + i) * dim_head, dim_head)
    x = x_ref[0, :, pl.ds(offset, dim_head)]

    # Flax's association: fold the scale into the reciprocal before applying it.
    mul = rsqrt_val * scale_ref[i, 0].astype(jnp.float32)
    normed = (x.astype(jnp.float32) * mul).astype(x.dtype)
    out = _apply_rope(normed, cos, sin, rope_accum, even_mask=even_mask)
    if prescale_val is not None:
      # Intentionally multiply in out.dtype (bfloat16) after _apply_rope so that
      # in-register prescaling is 0-ULP bit-identical on measured platforms
      # (v6e, tpu7x) to the unfused attention path (`query * LOG2E` and
      # `k * scale` on the bfloat16 RoPE output).
      out = out * prescale_val
    o_ref[0, i] = out


def _fused_kernel(
    x_ref,
    scale_ref,
    cos_ref,
    sin_ref,
    o_ref,
    *,
    dim_head: int,
    eps: float,
    rope_accum: str,
    head_block: int,
    prescale: float = 1.0,
):
  """As `_exact_kernel`, but also computes the feature-axis reduction in VMEM."""
  blk = pl.program_id(2)
  x_full_f32 = x_ref[0].astype(jnp.float32)
  var = jnp.mean(x_full_f32 * x_full_f32, axis=-1, keepdims=True)
  inv_rms = jax.lax.rsqrt(var + eps)
  cos = cos_ref[0].astype(jnp.float32) if rope_accum == "f32" else cos_ref[0]
  sin = sin_ref[0].astype(jnp.float32) if rope_accum == "f32" else sin_ref[0]
  lane = jax.lax.broadcasted_iota(jnp.int32, cos.shape, cos.ndim - 1)
  even_mask = jax.lax.rem(lane, 2) == 0
  prescale_val = jnp.asarray(prescale, o_ref.dtype) if prescale != 1.0 else None

  for i in range(head_block):
    offset = pl.multiple_of((blk * head_block + i) * dim_head, dim_head)
    x = x_ref[0, :, pl.ds(offset, dim_head)]
    mul = inv_rms * scale_ref[i, 0].astype(jnp.float32)
    normed = (x.astype(jnp.float32) * mul).astype(x.dtype)
    out = _apply_rope(normed, cos, sin, rope_accum, even_mask=even_mask)
    if prescale_val is not None:
      out = out * prescale_val
    o_ref[0, i] = out


def _run_exact(
    x,
    scale,
    cos_full,
    sin_signed,
    *,
    heads,
    dim_head,
    eps,
    block_s,
    vmem_limit_bytes,
    interpret,
    rope_accum,
    head_block,
    prescale: float = 1.0,
):
  batch, seq_len, feature = x.shape
  block_s = min(block_s, seq_len)

  # The reduction is left to XLA, expressed exactly as the reference expresses
  # it. That makes it bit-identical *provided* XLA emits the same reduction for
  # both graphs; see `rope_accum_is_measured` on why that is checked empirically
  # rather than assumed.
  x_f32 = x.astype(jnp.float32)
  rsqrt_val = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)

  return pl.pallas_call(
      functools.partial(_exact_kernel, dim_head=dim_head, rope_accum=rope_accum, head_block=head_block, prescale=prescale),
      grid=(batch, pl.cdiv(seq_len, block_s), heads // head_block),
      in_specs=[
          # Full-width and head-invariant: one sequential fetch per tile.
          pl.BlockSpec((1, block_s, feature), lambda b, s, h: (b, s, 0)),
          # Mosaic requires the second-minor block dimension to be a multiple
          # of 8 or to equal the array's. `scale` is carried as
          # `[heads, 1, dim_head]` so the unit axis satisfies the latter; a
          # `[heads, dim_head]` layout with a `(1, dim_head)` block does not
          # lower at all.
          pl.BlockSpec((head_block, 1, dim_head), lambda b, s, h: (h, 0, 0)),
          pl.BlockSpec((1, block_s, 1), lambda b, s, h: (b, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda b, s, h: (0, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda b, s, h: (0, s, 0)),
      ],
      out_specs=pl.BlockSpec((1, head_block, block_s, dim_head), lambda b, s, h: (b, h, s, 0)),
      out_shape=jax.ShapeDtypeStruct((batch, heads, seq_len, dim_head), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes
      ),
      interpret=interpret,
  )(x, scale.reshape(heads, 1, dim_head), rsqrt_val, cos_full, sin_signed)


def _run_fused(
    x,
    scale,
    cos_full,
    sin_signed,
    *,
    heads,
    dim_head,
    eps,
    block_s,
    vmem_limit_bytes,
    interpret,
    rope_accum,
    head_block,
    prescale: float = 1.0,
):
  batch, seq_len, feature = x.shape
  block_s = min(block_s, seq_len)

  # The input block index does not depend on the head-block axis, so Pallas
  # fetches each `[block_s, H*D]` tile from HBM once per `(b, s)` tile and
  # computes the `[block_s, 1]` inverse-RMS factor in-register without any
  # `[block_s, feature]` VMEM scratch buffer.
  return pl.pallas_call(
      functools.partial(
          _fused_kernel, dim_head=dim_head, eps=eps, rope_accum=rope_accum, head_block=head_block, prescale=prescale
      ),
      grid=(batch, pl.cdiv(seq_len, block_s), heads // head_block),
      in_specs=[
          pl.BlockSpec((1, block_s, feature), lambda b, s, h: (b, s, 0)),
          pl.BlockSpec((head_block, 1, dim_head), lambda b, s, h: (h, 0, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda b, s, h: (0, s, 0)),
          pl.BlockSpec((1, block_s, dim_head), lambda b, s, h: (0, s, 0)),
      ],
      out_specs=pl.BlockSpec((1, head_block, block_s, dim_head), lambda b, s, h: (b, h, s, 0)),
      out_shape=jax.ShapeDtypeStruct((batch, heads, seq_len, dim_head), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes
      ),
      interpret=interpret,
  )(x, scale.reshape(heads, 1, dim_head), cos_full, sin_signed)


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
    k_eps: float | None = None,
    heads: int | None = None,
    norm_mode: str = "exact",
    rope_accum: str = "dtype",
    block_s: int | None = None,
    head_block: int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
    q_prescale: float = 1.0,
    k_prescale: float = 1.0,
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
    eps: RMSNorm epsilon for `raw_q` (and `raw_k` when `k_eps` is None).
    k_eps: Optional separate RMSNorm epsilon for `raw_k`.
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
    block_s: Sequence tile size; defaults per `norm_mode` (512 for `"exact"`,
      256 for `"fused"`, and clamped to at most 256 in `"fused"` mode).
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
  effective_k_eps = eps if k_eps is None else k_eps

  if norm_mode not in NORM_MODES:
    raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}.")
  if rope_accum not in ROPE_ACCUM_MODES:
    raise ValueError(f"rope_accum must be one of {ROPE_ACCUM_MODES}, got {rope_accum!r}.")
  if dim_head % NUM_LANES != 0:
    raise ValueError(f"fused_rmsnorm_rope_pallas requires dim_head to be a multiple of {NUM_LANES}, got {dim_head}.")
  if dim_head % 2 != 0:
    raise ValueError(f"RoPE requires an even dim_head, got {dim_head}.")
  if head_block is not None:
    if head_block <= 0:
      raise ValueError(f"head_block must be a positive integer, got {head_block}.")
    for name, n in (("q_heads", q_heads), ("kv_heads", kv_heads)):
      if n % head_block != 0:
        raise ValueError(f"head_block ({head_block}) must divide {name} ({n}).")
  if block_s is not None and block_s <= 0:
    raise ValueError(f"block_s must be a positive integer, got {block_s}.")

  _, seq_q, feature_q = raw_q.shape
  _, seq_k, feature_k = raw_k.shape
  if feature_q != q_heads * dim_head:
    raise ValueError(f"raw_q feature dim ({feature_q}) must equal q_heads ({q_heads}) * dim_head ({dim_head})")
  if feature_k != kv_heads * dim_head:
    raise ValueError(f"raw_k feature dim ({feature_k}) must equal kv_heads ({kv_heads}) * dim_head ({dim_head})")
  if freqs_cis.shape[-1] * 2 != dim_head:
    raise ValueError(f"freqs_cis last dim ({freqs_cis.shape[-1]}) must be dim_head // 2 ({dim_head // 2}).")
  if freqs_cis.shape[2] < max(seq_q, seq_k):
    raise ValueError(
        f"freqs_cis sequence dim ({freqs_cis.shape[2]}) must be at least max(seq_q, seq_k) ({max(seq_q, seq_k)})."
    )

  if block_s is None:
    block_s = DEFAULT_BLOCK_S_EXACT if norm_mode == "exact" else DEFAULT_BLOCK_S_FUSED
  elif norm_mode == "fused":
    block_s = min(block_s, DEFAULT_BLOCK_S_FUSED)
  runner = _run_exact if norm_mode == "exact" else _run_fused

  cos_q, sin_q = _rope_tables(freqs_cis, seq_q, raw_q.dtype)
  if seq_k == seq_q and raw_k.dtype == raw_q.dtype:
    cos_k, sin_k = cos_q, sin_q
  else:
    cos_k, sin_k = _rope_tables(freqs_cis, seq_k, raw_k.dtype)

  common = {
      "dim_head": dim_head,
      "block_s": block_s,
      "vmem_limit_bytes": vmem_limit_bytes,
      "interpret": interpret,
      "rope_accum": rope_accum,
  }
  q_out = runner(
      raw_q,
      q_norm_scale,
      cos_q,
      sin_q,
      heads=q_heads,
      head_block=q_heads if head_block is None else head_block,
      eps=eps,
      prescale=q_prescale,
      **common,
  )
  k_out = runner(
      raw_k,
      k_norm_scale,
      cos_k,
      sin_k,
      heads=kv_heads,
      head_block=kv_heads if head_block is None else head_block,
      eps=effective_k_eps,
      prescale=k_prescale,
      **common,
  )
  return q_out, k_out


def rope_pair_swap_reference(x: jax.Array) -> jax.Array:
  """Pure-JAX twin of `_pair_swap`, used to pin the rotation identity in tests."""
  axis = x.ndim - 1
  roll_right = jnp.roll(x, 1, axis=axis)
  roll_left = jnp.roll(x, -1, axis=axis)
  lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis)
  return jnp.where(lane % 2 == 0, roll_left, roll_right)
