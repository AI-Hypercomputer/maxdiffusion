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

Pallas partial attention with padding-only masks for sequence-edge tiles.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxdiffusion.kernels import custom_splash_attention as dense_custom

NUM_SUBLANES = dense_custom.NUM_SUBLANES
NT_DIM_NUMBERS = dense_custom.NT_DIM_NUMBERS
MASK_VALUE = dense_custom.DEFAULT_MASK_VALUE


def _build_padding_metadata(table_np, active_np, *, q_seq_len, kv_seq_len, bq, bkv):
  q_valid = np.zeros(table_np.shape, dtype=np.int32)
  k_valid = np.zeros(table_np.shape, dtype=np.int32)
  for qi in range(table_np.shape[0]):
    q0 = qi * bq
    qv = max(0, min(bq, q_seq_len - q0))
    for slot in range(int(active_np[qi])):
      kj = int(table_np[qi, slot])
      k0 = kj * bkv
      q_valid[qi, slot] = qv
      k_valid[qi, slot] = max(0, min(bkv, kv_seq_len - k0))
  return q_valid, k_valid


def _padding_kernel(
    kv_map_ref,
    active_ref,
    qvalid_ref,
    kvalid_ref,
    q_ref,
    k_ref,
    v_ref,
    out_ref,
    lse_ref,
    m_ref,
    l_ref,
    oacc_ref,
    *,
    grid_width,
    bq,
    bkv,
    bc,
    bci,
    dv,
    use_base2_exp,
):
  qi = pl.program_id(1)
  slot = pl.program_id(2)
  active = slot < active_ref[qi]
  exp = jnp.exp2 if use_base2_exp else jnp.exp
  log = jnp.log2 if use_base2_exp else jnp.log

  @pl.when(slot == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, MASK_VALUE)
    l_ref[...] = jnp.zeros_like(l_ref)
    oacc_ref[...] = jnp.zeros_like(oacc_ref)

  @pl.when(active)
  def run():
    q = q_ref[...]
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    o_prev = oacc_ref[...]

    def body(ci, carry):
      m_c, l_c, o_c = carry
      off = ci * bc
      sl = pl.ds(off, bc)
      kc = k_ref[sl, :]
      vc = v_ref[sl, :]
      scores = lax.dot_general(
          kc,
          q,
          NT_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )

      qc = jnp.arange(bq, dtype=jnp.int32)[None, :]
      kr = off + jnp.arange(bc, dtype=jnp.int32)[:, None]
      valid = (qc < qvalid_ref[qi, slot]) & (kr < kvalid_ref[qi, slot])
      scores = jnp.where(valid, scores, jnp.asarray(MASK_VALUE, scores.dtype))

      for inner in range(0, bc, bci):
        z = scores[inner : inner + bci]
        vv = vc[inner : inner + bci]
        m_chunk = z.max(axis=0)[None, :]
        m_new = jnp.maximum(m_c, m_chunk)
        p = exp(z - m_new[0:1])
        l_chunk = p.sum(axis=0, keepdims=True)
        alpha = exp(m_c - m_new)
        l_new = l_chunk + alpha * l_c
        o_chunk = lax.dot_general(
            vv,
            p.astype(q_ref.dtype),
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        o_c = alpha[0:1] * o_c + o_chunk
        m_c, l_c = m_new, l_new
      return m_c, l_c, o_c

    m_prev, l_prev, o_prev = lax.fori_loop(
        0,
        bkv // bc,
        body,
        (m_prev, l_prev, o_prev),
        unroll=True,
    )
    m_ref[...] = m_prev
    l_ref[...] = l_prev
    oacc_ref[...] = o_prev

  @pl.when(slot == grid_width - 1)
  def finish():
    l = l_ref[...]
    m = m_ref[...]
    has_value = l > 0
    safe_l = jnp.where(has_value, l, jnp.ones_like(l))
    inv_l = jnp.tile(1.0 / safe_l, (dv // NUM_SUBLANES, 1))
    out = oacc_ref[...] * inv_l
    out_ref[...] = jnp.where(
        jnp.tile(has_value, (dv // NUM_SUBLANES, 1)),
        out,
        jnp.zeros_like(out),
    ).astype(out_ref.dtype)
    lse_ref[...] = jnp.where(
        has_value,
        m + log(safe_l),
        jnp.asarray(MASK_VALUE, m.dtype),
    ).astype(lse_ref.dtype)


def make_padding_partial_from_table(
    *,
    table_np,
    active_np,
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
  del band_width, frame_size, include_first_frame
  bq = int(block_sizes.block_q)
  bkv = int(block_sizes.block_kv)
  bc = int(block_sizes.block_kv_compute)
  bci = int(bkv_compute_in if bkv_compute_in is not None else block_sizes.block_kv_compute_in)
  if bkv % bc or bc % bci:
    raise ValueError(f"invalid blocks bkv={bkv}, bc={bc}, bci={bci}")

  q_valid, k_valid = _build_padding_metadata(
      table_np,
      active_np,
      q_seq_len=int(orig_q_seq_len),
      kv_seq_len=int(orig_kv_seq_len),
      bq=bq,
      bkv=bkv,
  )
  scalars = tuple(jnp.asarray(x) for x in (table_np, active_np, q_valid, k_valid))
  qtiles, width = table_np.shape

  def partial(q, k, v):
    heads, _, dq = q.shape
    dv = v.shape[-1]
    if dv % NUM_SUBLANES:
      raise NotImplementedError(f"head_dim_v={dv} must be divisible by {NUM_SUBLANES}")

    def qmap(h, qi, slot, *refs):
      del slot, refs
      return (h, qi, 0)

    def kvmap(h, qi, slot, kvmap_ref, *refs):
      del refs
      return (h, kvmap_ref[qi, slot], 0)

    def outmap(h, qi, slot, *refs):
      del slot, refs
      return (h, 0, qi)

    outs = pl.pallas_call(
        functools.partial(
            _padding_kernel,
            grid_width=width,
            bq=bq,
            bkv=bkv,
            bc=bc,
            bci=bci,
            dv=dv,
            use_base2_exp=bool(use_base2_exp),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
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
            grid=(heads, qtiles, width),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_experimental_scheduler},
            disable_bounds_checks=True,
            skip_device_barrier=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((heads, dv, qtiles * bq), q.dtype),
            jax.ShapeDtypeStruct((heads, NUM_SUBLANES, qtiles * bq), jnp.float32),
        ],
    )(*scalars, q, k, v)
    return (
        outs[-2][:, :, :orig_q_seq_len],
        outs[-1][:, 0, :orig_q_seq_len],
    )

  partial.boundary_tiles = int(active_np.sum())
  partial.boundary_table_shape = tuple(table_np.shape)
  return partial
