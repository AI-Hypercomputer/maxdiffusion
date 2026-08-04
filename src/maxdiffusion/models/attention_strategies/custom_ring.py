# Copyright 2026 Google LLC / The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Splash Ring Attention implementation (fixed-m, bidirectional, and standard scan loops)."""

from functools import partial
from typing import Any
import jax
from jax import lax
import jax.numpy as jnp
from maxdiffusion.kernels import custom_splash_attention as custom_splash
from maxdiffusion.kernels.splash_attention import base


def _custom_bidirectional_ring_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    block_sizes: Any,
    orig_q_seq_len: int,
    orig_kv_seq_len: int,
    use_base2_exp: bool,
    use_experimental_scheduler: bool,
    vmem_limit_bytes: int | None,
    mask_value: float,
    ring_axis: str,
) -> jax.Array:
  """Wrap-free (bidirectional) ring attention for a NON-wrapping ring axis."""
  axis_size = lax.axis_size(ring_axis)
  idx = lax.axis_index(ring_axis)
  exp_fn = jnp.exp2 if use_base2_exp else jnp.exp

  def _attn(kc, vc):
    o, m, l = custom_splash.splash_attention_forward_ring(
        q,
        kc,
        vc,
        block_sizes,
        q_seq_len=orig_q_seq_len,
        kv_seq_len=orig_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return o.astype(jnp.float32), m.astype(jnp.float32), l.astype(jnp.float32)

  def _merge(m, l, o, mc, lc, oc, valid):
    mc = jnp.where(valid, mc, mask_value)
    lc = jnp.where(valid, lc, 0.0)
    oc = jnp.where(valid, oc, 0.0)
    m_next = jnp.maximum(m, mc)
    alpha = exp_fn(m - m_next)
    beta = exp_fn(mc - m_next)
    return (
        m_next,
        alpha * l + beta * lc,
        alpha[..., None] * o + beta[..., None] * oc,
    )

  o, m, l = _attn(k, v)

  shift_r = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, i + 1) for i in range(axis_size - 1)],
  )
  shift_l = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, i - 1) for i in range(1, axis_size)],
  )

  kr, vr = shift_r(k), shift_r(v)
  kl, vl = shift_l(k), shift_l(v)

  def body(carry, t):
    m, l, o, kr, vr, kl, vl = carry
    valid_r = (idx - t) >= 0
    valid_l = (idx + t) <= (axis_size - 1)
    kr_s, vr_s = jnp.where(valid_r, kr, k), jnp.where(valid_r, vr, v)
    kl_s, vl_s = jnp.where(valid_l, kl, k), jnp.where(valid_l, vl, v)
    o_r, m_r, l_r = _attn(kr_s, vr_s)
    m, l, o = _merge(m, l, o, m_r, l_r, o_r, valid_r)
    o_l, m_l, l_l = _attn(kl_s, vl_s)
    m, l, o = _merge(m, l, o, m_l, l_l, o_l, valid_l)
    kr_n, vr_n = shift_r(kr), shift_r(vr)
    kl_n, vl_n = shift_l(kl), shift_l(vl)
    return (m, l, o, kr_n, vr_n, kl_n, vl_n), None

  (_, l_final, o_final, *_), _ = lax.scan(
      body,
      (m, l, o, kr, vr, kl, vl),
      xs=jnp.arange(1, axis_size),
      length=axis_size - 1,
      unroll=True,
  )

  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  return (o_final * l_inv[..., None]).astype(q.dtype)


def _custom_ring_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    block_sizes: Any,
    orig_q_seq_len: int,
    orig_kv_seq_len: int,
    use_base2_exp: bool,
    use_experimental_scheduler: bool,
    vmem_limit_bytes: int | None,
    mask_value: float,
    ring_axis: str,
    ring_size: int | None = None,
    perm: list[tuple[int, int]] | None = None,
    bidirectional: bool = False,
    use_fixed_m: bool = False,
    fixed_m_norms: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Forward-only ring attention using the custom dense splash kernel."""
  axis_size = lax.axis_size(ring_axis)
  if bidirectional:
    if perm is not None or (ring_size is not None and ring_size != axis_size):
      raise ValueError(
          "bidirectional (wrap-free) ring requires perm=None and ring_size==axis_size "
          "(it operates on the full real ring axis)."
      )
    if use_fixed_m:
      raise NotImplementedError("fixed-m is not yet supported on the bidirectional ring path.")
    return _custom_bidirectional_ring_forward(
        q,
        k,
        v,
        block_sizes=block_sizes,
        orig_q_seq_len=orig_q_seq_len,
        orig_kv_seq_len=orig_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
        mask_value=mask_value,
        ring_axis=ring_axis,
    )
  if ring_size is None:
    ring_size = axis_size
  if perm is None:
    perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]

  shift = partial(lax.ppermute, axis_name=ring_axis, perm=perm)

  exp_fn = jnp.exp2 if use_base2_exp else jnp.exp

  num_q_heads = q.shape[0]
  head_dim_v = v.shape[-1]

  if use_fixed_m:
    if fixed_m_norms is None:
      raise ValueError("use_fixed_m on the ring path requires fixed_m_norms=(qn_max, mk_h).")
    log_fn = jnp.log2 if use_base2_exp else jnp.log
    qn_max, mk_h_init = fixed_m_norms
    tiny = jnp.finfo(jnp.float32).tiny
    lse_init = -1e30

    if mk_h_init.ndim == 2:
      mk_all = mk_h_init
    else:
      mk_all = lax.all_gather(mk_h_init, ring_axis)  # (axis_size, heads)
    my_ring_index = lax.axis_index(ring_axis)

    mk_global = mk_all.max(axis=0)  # (heads,)
    all_fixed_global = jnp.all(qn_max * mk_global <= custom_splash._FIXED_M_RING_SAFE_BOUND)

    def _accumulate_scan(_):
      mk_arr = jnp.stack([mk_global, jnp.ones_like(mk_global)])
      o_sum = jnp.zeros((num_q_heads, orig_q_seq_len, head_dim_v), jnp.float32)
      l_sum = jnp.zeros((num_q_heads, orig_q_seq_len), jnp.float32)
      k_current, v_current = k, v
      for hop in range(ring_size):
        is_last_hop = hop == ring_size - 1
        if not is_last_hop:
          k_next = shift(k_current)
          v_next = shift(v_current)
        o_curr, _, l_curr = custom_splash.splash_attention_forward_ring(
            q,
            k_current,
            v_current,
            block_sizes,
            q_seq_len=orig_q_seq_len,
            kv_seq_len=orig_kv_seq_len,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
            vmem_limit_bytes=vmem_limit_bytes,
            use_fixed_m=True,
            mk=mk_arr,
            uniform_fixed_m=True,
        )
        o_sum = o_sum + o_curr.astype(jnp.float32)
        l_sum = l_sum + l_curr.astype(jnp.float32)
        if not is_last_hop:
          k_current, v_current = k_next, v_next
      l_inv = jnp.where(l_sum == 0.0, 0.0, 1.0 / l_sum)
      return (o_sum * l_inv[..., None]).astype(q.dtype)

    def fixed_body(carry, hop, is_last_hop):
      o_run, lse_run, k_current, v_current = carry
      if is_last_hop:
        k_next, v_next = k_current, v_current
      else:
        k_next = shift(k_current)
        v_next = shift(v_current)

      mk_h = jax.lax.dynamic_index_in_dim(mk_all, (my_ring_index - hop) % axis_size, keepdims=False)
      fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_RING_SAFE_BOUND).astype(jnp.float32)
      mk_arr = jnp.stack([mk_h, fixed_ok])

      o_curr, m_curr, l_curr = custom_splash.splash_attention_forward_ring(
          q,
          k_current,
          v_current,
          block_sizes,
          q_seq_len=orig_q_seq_len,
          kv_seq_len=orig_kv_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          use_fixed_m=True,
          mk=mk_arr,
      )
      m_curr = m_curr.astype(jnp.float32)
      l_curr = l_curr.astype(jnp.float32)
      o_curr = o_curr.astype(jnp.float32)

      l_safe = jnp.maximum(l_curr, tiny)
      lse_curr = jnp.where(l_curr > 0.0, m_curr + log_fn(l_safe), -jnp.inf)
      o_norm = o_curr / l_safe[..., None]

      lse_new = jnp.maximum(lse_run, lse_curr)
      w_run = exp_fn(lse_run - lse_new)
      w_curr = exp_fn(lse_curr - lse_new)
      denom = w_run + w_curr
      o_new = (w_run[..., None] * o_run + w_curr[..., None] * o_norm) / denom[..., None]
      return (o_new, lse_new + log_fn(denom), k_next, v_next), None

    fixed_init = (
        jnp.zeros((num_q_heads, orig_q_seq_len, head_dim_v), jnp.float32),
        jnp.full((num_q_heads, orig_q_seq_len), lse_init, jnp.float32),
        k,
        v,
    )

    def _lse_scan(_):
      carry = fixed_init
      for hop in range(ring_size):
        carry, _ = fixed_body(carry, hop, hop == ring_size - 1)
      return carry[0].astype(q.dtype)

    return lax.cond(all_fixed_global, _accumulate_scan, _lse_scan, None)

  o_init = jnp.zeros((num_q_heads, orig_q_seq_len, head_dim_v), jnp.float32)
  l_init = jnp.zeros((num_q_heads, orig_q_seq_len), jnp.float32)
  m_init = jnp.full((num_q_heads, orig_q_seq_len), mask_value, jnp.float32)

  m_final, l_final, o_final = m_init, l_init, o_init
  k_current, v_current = k, v
  for hop in range(ring_size):
    if hop != ring_size - 1:
      k_next = shift(k_current)
      v_next = shift(v_current)

    o_curr, m_curr, l_curr = custom_splash.splash_attention_forward_ring(
        q,
        k_current,
        v_current,
        block_sizes,
        q_seq_len=orig_q_seq_len,
        kv_seq_len=orig_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    m_curr = m_curr.astype(jnp.float32)
    l_curr = l_curr.astype(jnp.float32)
    o_curr = o_curr.astype(jnp.float32)

    m_next = jnp.maximum(m_final, m_curr)
    alpha = exp_fn(m_final - m_next)
    beta = exp_fn(m_curr - m_next)
    l_final = alpha * l_final + beta * l_curr
    o_final = alpha[..., None] * o_final + beta[..., None] * o_curr
    m_final = m_next
    if hop != ring_size - 1:
      k_current, v_current = k_next, v_next

  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  out = (o_final * l_inv[..., None]).astype(q.dtype)
  return out


def make_custom_ring_attention(
    *,
    block_sizes: Any,
    orig_q_seq_len: int,
    orig_kv_seq_len: int,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    vmem_limit_bytes: int | None = None,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    ring_axis: str = "context",
    ring_size: int | None = None,
    perm: list[tuple[int, int]] | None = None,
    bidirectional: bool = False,
    use_fixed_m: bool = False,
    fixed_m_norms: tuple[jax.Array, jax.Array] | None = None,
):
  """Builds a forward-only ring-attention callable around the custom kernel."""

  def _ring(q, k, v):
    return _custom_ring_attention_forward(
        q,
        k,
        v,
        block_sizes=block_sizes,
        orig_q_seq_len=orig_q_seq_len,
        orig_kv_seq_len=orig_kv_seq_len,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        vmem_limit_bytes=vmem_limit_bytes,
        mask_value=mask_value,
        ring_axis=ring_axis,
        ring_size=ring_size,
        perm=perm,
        bidirectional=bidirectional,
        use_fixed_m=use_fixed_m,
        fixed_m_norms=fixed_m_norms,
    )

  return _ring
