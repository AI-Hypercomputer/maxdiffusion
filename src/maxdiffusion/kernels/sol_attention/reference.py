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

Dense O(L^2)-memory reference for Sol-Attn.

Materializes the full score matrix, so it is only meant for tests and small
sequence lengths. It evaluates Eqs. (9)-(10) of arXiv:2607.24027 literally:
one softmax over [exact scores of selected blocks | zeroth-order pooled scores
of unselected blocks].
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from maxdiffusion.kernels.sol_attention import routing as sol_routing


def _flatten_heads(x: jax.Array) -> jax.Array:
  return x.reshape(-1, *x.shape[-2:])


def sol_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    scale: float,
    beta: float,
    block_sizes: sol_routing.SolBlockSizes = sol_routing.SolBlockSizes(),
    always_exact: Optional[jax.Array] = None,
    approximate: bool = True,
    selected: Optional[jax.Array] = None,
) -> jax.Array:
  """Sol-Attn on [..., L, d] tensors (any leading batch/head dims).

  Args:
    q, k, v: Query/key/value [..., L, d] self-attention tensors.
    scale: Attention logit scale.
    beta: Threshold cutoff, see `routing.threshold_route`.
    block_sizes: Block granularity.
    always_exact: Optional bool [Nk] of key blocks forced into the exact branch.
    approximate: If False, unselected blocks are dropped instead of
      approximated (standard keep-or-drop block-sparse attention). Used for the
      approximate-correction ablation (Fig. 9 of the paper).
    selected: Optional precomputed bool [prod(...), Nq, Nk] selection that
      overrides threshold routing.

  Returns:
    Output [..., L, d] in q.dtype.
  """
  lead = q.shape[:-2]
  seq_len, head_dim = q.shape[-2:]
  padded_len = sol_routing.padded_length(seq_len, block_sizes)
  qp, kp, vp = (sol_routing.pad_sequence(_flatten_heads(x), padded_len) for x in (q, k, v))
  routing = sol_routing.compute_routing(
      qp,
      kp,
      vp,
      seq_len=seq_len,
      scale=scale,
      beta=beta,
      block_sizes=block_sizes,
      always_exact=always_exact,
  )
  if selected is not None:
    routing = sol_routing.SolRouting(
        selected=selected,
        tau=routing.tau,
        kv_counts=routing.kv_counts,
        k_pooled=routing.k_pooled,
        v_pooled=routing.v_pooled,
    )
  bq, bkv = block_sizes.block_q, block_sizes.block_kv

  hp = jax.lax.Precision.HIGHEST
  qf, kf, vf = (x.astype(jnp.float32) for x in (qp, kp, vp))
  scores = jnp.einsum("hqd,hkd->hqk", qf, kf, precision=hp) * scale
  token_selected = jnp.repeat(jnp.repeat(routing.selected, bq, axis=1), bkv, axis=2)
  key_valid = jnp.arange(padded_len) < seq_len
  exact_logits = jnp.where(jnp.logical_and(token_selected, key_valid), scores, sol_routing.MASK_VALUE)

  if approximate:
    approx_logits = jnp.einsum("hqd,hkd->hqk", qf, routing.k_pooled, precision=hp) * scale
    approx_logits = approx_logits + jnp.repeat(sol_routing.approx_bias(routing), bq, axis=1)
    logits = jnp.concatenate([exact_logits, approx_logits], axis=-1)
    values = jnp.concatenate([vf, routing.v_pooled], axis=-2)
  else:
    logits, values = exact_logits, vf

  m = jnp.max(logits, axis=-1, keepdims=True)
  p = jnp.where(logits > 0.5 * sol_routing.MASK_VALUE, jnp.exp(logits - m), 0.0)
  out = jnp.einsum("hqk,hkd->hqd", p, values, precision=hp) / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
  return out[:, :seq_len].reshape(*lead, seq_len, head_dim).astype(q.dtype)


def dense_attention_reference(q: jax.Array, k: jax.Array, v: jax.Array, *, scale: float) -> jax.Array:
  """Plain softmax attention on [..., L, d] tensors in float32."""
  hp = jax.lax.Precision.HIGHEST
  qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
  p = jax.nn.softmax(jnp.einsum("...qd,...kd->...qk", qf, kf, precision=hp) * scale, axis=-1)
  return jnp.einsum("...qk,...kd->...qd", p, vf, precision=hp).astype(q.dtype)
