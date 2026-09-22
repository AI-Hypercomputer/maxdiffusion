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

Static-range SVG attention with exact and tile-rounded support.

Mask behavior is selected at trace time to keep full tiles branch-free.

This module is a leaf of the SVG kernel stack: it depends only on the dense
splash-attention primitives and must not import the balanced-rounding layer or
the dispatch layer. The production entry points live in
`custom_svg_attention_dispatch`.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion.kernels import custom_splash_attention as dense_custom

NUM_SUBLANES = dense_custom.NUM_SUBLANES
NT_DIM_NUMBERS = dense_custom.NT_DIM_NUMBERS
DEFAULT_MASK_VALUE = dense_custom.DEFAULT_MASK_VALUE


@dataclasses.dataclass(frozen=True)
class SVGBlockSizes:
  block_q: int = 3328
  block_kv: int = 2048
  block_kv_compute: int = 256
  block_kv_compute_in: int = 256


def _classify_tiles(q_seq_len, kv_seq_len, bq, bkv, band_width, frame_size, include_first_frame):
  q_tiles = math.ceil(q_seq_len / bq)
  kv_tiles = math.ceil(kv_seq_len / bkv)
  sink_last = (frame_size - 1) // bkv if include_first_frame else -1
  full_rows, boundary_rows = [], []

  for qi in range(q_tiles):
    q0 = qi * bq
    q1 = min(q_seq_len, q0 + bq) - 1
    band0 = max(0, q0 - band_width)
    band1 = min(kv_seq_len - 1, q1 + band_width)
    first, last = band0 // bkv, band1 // bkv
    live = set(range(first, last + 1))
    if include_first_frame:
      live.update(range(0, sink_last + 1))

    full, boundary = [], []
    for kj in sorted(x for x in live if 0 <= x < kv_tiles):
      k0 = kj * bkv
      k1 = min(kv_seq_len, k0 + bkv) - 1
      q_real_full = q0 + bq <= q_seq_len
      k_real_full = k0 + bkv <= kv_seq_len
      sink_full = include_first_frame and k1 < frame_size
      local_full = max(abs(q0 - k1), abs(q1 - k0)) <= band_width
      (full if q_real_full and k_real_full and (sink_full or local_full) else boundary).append(kj)
    full_rows.append(full)
    boundary_rows.append(boundary)

  def pack(rows):
    width = max(1, max((len(r) for r in rows), default=0))
    table = np.zeros((q_tiles, width), np.int32)
    active = np.zeros((q_tiles,), np.int32)
    for qi, row in enumerate(rows):
      if row:
        table[qi, : len(row)] = row
        active[qi] = len(row)
    return table, active

  fm, fa = pack(full_rows)
  bm, ba = pack(boundary_rows)
  return fm, fa, bm, ba


def _partial_kernel(
    kv_map_ref,
    active_counts_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    lse_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_mode,
    mask_value,
    grid_width,
    bq,
    bkv,
    bkv_compute,
    bkv_compute_in,
    head_dim_v,
    q_seq_len,
    kv_seq_len,
    band_width,
    frame_size,
    include_first_frame,
    use_base2_exp,
):
  """Run one static tile-table partial.

  `mask_mode` is a Python/static argument, so Pallas specializes three distinct
  kernels rather than carrying a runtime tile-kind branch through FULL work:
  `none` for exact FULL tiles, `padding` for rounded boundary rectangles, and
  `exact` for the exact SVG boundary predicate.
  """
  if mask_mode not in ("none", "padding", "exact"):
    raise ValueError(f"unknown SVG mask_mode {mask_mode!r}")

  float32 = jnp.float32
  repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem:
    raise NotImplementedError(f"head_dim_v={head_dim_v} must divide {NUM_SUBLANES}")

  exp = jnp.exp2 if use_base2_exp else jnp.exp
  log = jnp.log2 if use_base2_exp else jnp.log
  qi = pl.program_id(1)
  slot = pl.program_id(2)
  active = slot < active_counts_ref[qi]

  @pl.when(slot == 0)
  def init():
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

  def body(ci, _):
    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[:]
    q = q_ref[...]
    off = ci * bkv_compute
    sl = pl.ds(off, bkv_compute)
    kc = k_ref[sl, :]
    vc = v_ref[sl, :]
    qk = lax.dot_general(kc, q, NT_DIM_NUMBERS, preferred_element_type=float32)

    if mask_mode != "none":
      kj = kv_map_ref[qi, slot]
      qids = qi * bq + jnp.arange(bq, dtype=jnp.int32)[None, :]
      kids = kj * bkv + off + jnp.arange(bkv_compute, dtype=jnp.int32)[:, None]
      valid = (qids < q_seq_len) & (kids < kv_seq_len)
      if mask_mode == "exact":
        local = jnp.abs(qids - kids) <= band_width
        anchor = (kids < frame_size) if include_first_frame else jnp.zeros_like(local)
        valid = valid & (local | anchor)
      qk = jnp.where(valid, qk, jnp.asarray(mask_value, qk.dtype))

    for i in range(0, bkv_compute, bkv_compute_in):
      z = qk[i : i + bkv_compute_in]
      vv = vc[i : i + bkv_compute_in]
      mc = z.max(axis=0)[None, :]
      mn = jnp.maximum(m_prev, mc)
      p = exp(z - mn[0:1])
      lc = p.sum(axis=0, keepdims=True)
      alpha = exp(m_prev - mn)
      ln = lc + alpha * l_prev
      oc = lax.dot_general(
          vv,
          p.astype(q_ref.dtype),
          (((0,), (0,)), ((), ())),
          preferred_element_type=float32,
      )
      o_prev = alpha[0:1] * o_prev + oc
      m_prev, l_prev = mn, ln

    m_scratch_ref[...] = m_prev
    l_scratch_ref[...] = l_prev
    o_scratch_ref[:] = o_prev

  @pl.when(active)
  def run():
    lax.fori_loop(0, bkv // bkv_compute, body, None, unroll=True)

  @pl.when(slot == grid_width - 1)
  def finish():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    has = l > 0
    safe = jnp.where(has, l, jnp.ones_like(l))
    inv = jnp.tile(1.0 / safe, (repeats, 1))
    out = o_scratch_ref[...] * inv
    o_ref[...] = jnp.where(jnp.tile(has, (repeats, 1)), out, jnp.zeros_like(out)).astype(o_ref.dtype)
    lse_ref[...] = jnp.where(
        has,
        m + log(safe),
        jnp.asarray(mask_value, m.dtype),
    ).astype(lse_ref.dtype)


def _make_partial(
    *,
    table_np,
    active_np,
    mask_mode="none",
    exact_boundary_mask=None,
    block_sizes,
    orig_q_seq_len,
    orig_kv_seq_len,
    band_width,
    frame_size,
    include_first_frame,
    bkv_compute_in=None,
    use_base2_exp=True,
    use_experimental_scheduler=False,
    vmem_limit_bytes=None,
):
  """Build a specialized partial for one static tile table.

  `exact_boundary_mask` remains as a compatibility alias for older callers.
  """
  if exact_boundary_mask is not None:
    mask_mode = "exact" if exact_boundary_mask else "none"

  bq = int(block_sizes.block_q)
  bkv = int(block_sizes.block_kv)
  bc = int(block_sizes.block_kv_compute)
  bci = int(bkv_compute_in if bkv_compute_in is not None else block_sizes.block_kv_compute_in)
  if bkv % bc or bc % bci:
    raise ValueError(f"invalid blocks bkv={bkv}, bc={bc}, bci={bci}")

  kv_map = jnp.asarray(table_np)
  active = jnp.asarray(active_np)
  height, width = table_np.shape

  def partial(q, k, v):
    heads, _, dq = q.shape
    dv = v.shape[-1]
    if heads != k.shape[0]:
      raise NotImplementedError("static-range SVG kernel supports MHA only")

    def qmap(h, i, j, *refs):
      del j, refs
      return (h, i, 0)

    def kvmap(h, i, j, kvref, *refs):
      del refs
      return (h, kvref[i, j], 0)

    def outmap(h, i, j, *refs):
      del j, refs
      return (h, 0, i)

    outs = pl.pallas_call(
        functools.partial(
            _partial_kernel,
            mask_mode=str(mask_mode),
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=width,
            bq=bq,
            bkv=bkv,
            bkv_compute=bc,
            bkv_compute_in=bci,
            head_dim_v=dv,
            q_seq_len=int(orig_q_seq_len),
            kv_seq_len=int(orig_kv_seq_len),
            band_width=int(band_width),
            frame_size=int(frame_size),
            include_first_frame=bool(include_first_frame),
            use_base2_exp=bool(use_base2_exp),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec((None, bq, dq), qmap),
                pl.BlockSpec((None, bkv, dq), kvmap),
                pl.BlockSpec((None, bkv, dv), kvmap),
            ],
            out_specs=[
                pl.BlockSpec((None, dv, bq), outmap),
                pl.BlockSpec((None, NUM_SUBLANES, bq), outmap),
            ],
            scratch_shapes=[
                pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
                pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
                pltpu.VMEM((dv, bq), jnp.float32),
            ],
            grid=(heads, height, width),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
            disable_bounds_checks=True,
            skip_device_barrier=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((heads, dv, orig_q_seq_len), q.dtype),
            jax.ShapeDtypeStruct((heads, NUM_SUBLANES, orig_q_seq_len), jnp.float32),
        ],
    )(kv_map, active, q, k, v)
    return outs[-2], outs[-1][:, 0, :]

  partial.boundary_tiles = int(active_np.sum())
  partial.boundary_table_shape = tuple(table_np.shape)
  return partial


def make_svg_exact_static_range_mha(
    *,
    block_sizes,
    orig_q_seq_len,
    orig_kv_seq_len,
    band_width,
    frame_size,
    include_first_frame=True,
    bkv_compute_in=None,
    use_base2_exp=True,
    use_experimental_scheduler=False,
    vmem_limit_bytes=None,
):
  """Build the exact two-pass SVG reference implementation."""
  bq, bkv = int(block_sizes.block_q), int(block_sizes.block_kv)
  fm, fa, bm, ba = _classify_tiles(
      orig_q_seq_len,
      orig_kv_seq_len,
      bq,
      bkv,
      int(band_width),
      int(frame_size),
      bool(include_first_frame),
  )
  common = {
      "block_sizes": block_sizes,
      "orig_q_seq_len": orig_q_seq_len,
      "orig_kv_seq_len": orig_kv_seq_len,
      "band_width": band_width,
      "frame_size": frame_size,
      "include_first_frame": include_first_frame,
      "bkv_compute_in": bkv_compute_in,
      "use_base2_exp": use_base2_exp,
      "use_experimental_scheduler": use_experimental_scheduler,
      "vmem_limit_bytes": vmem_limit_bytes,
  }
  full = _make_partial(table_np=fm, active_np=fa, mask_mode="none", **common)
  boundary = _make_partial(table_np=bm, active_np=ba, mask_mode="exact", **common)

  def attention(q, k, v):
    of, lf = full(q, k, v)
    ob, lb = boundary(q, k, v)
    m = jnp.maximum(lf, lb)
    exp = jnp.exp2 if use_base2_exp else jnp.exp
    wf, wb = exp(lf - m), exp(lb - m)
    den = wf + wb
    return (of.astype(jnp.float32) * (wf / den)[:, None, :] + ob.astype(jnp.float32) * (wb / den)[:, None, :]).astype(
        q.dtype
    )

  attention.full_tiles = int(fa.sum())
  attention.boundary_tiles = int(ba.sum())
  attention.full_table_shape = tuple(fm.shape)
  attention.boundary_table_shape = tuple(bm.shape)
  attention.physical_tiles = attention.full_tiles + attention.boundary_tiles
  return attention
