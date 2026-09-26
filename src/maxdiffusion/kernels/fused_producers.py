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

"""Optimized fused producers for Wan Attention."""

from typing import Tuple

import jax
import jax.numpy as jnp


def fused_rmsnorm_rope(
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
    k_eps: float | None = None,
) -> Tuple[jax.Array, jax.Array]:
  """Fusion-friendly FP32 RMSNorm + BF16 RoPE + Head Transposition producer.

  Performs FP32 RMSNorm normalization for maximum stability, casts to input dtype
  (e.g. BF16), and applies RoPE rotation and head transposition in BF16 precision,
  avoiding excess FP32 VPU/VMEM cycles on long sequence lengths. Fully supports GQA
  where q_heads != kv_heads.

  Args:
    raw_q: Raw query projection of shape [B, Sq, Dq] (where Dq = q_heads * dim_head).
    raw_k: Raw key projection of shape [B, Sk, Dk] (where Dk = kv_heads * dim_head).
    q_norm_scale: RMSNorm scale parameter for query of shape [Dq].
    k_norm_scale: RMSNorm scale parameter for key of shape [Dk].
    freqs_cis: Complex rotary embedding tensor of shape [1, 1, S, dim_head // 2].
    q_heads: Number of query attention heads.
    kv_heads: Number of key/value attention heads (defaults to q_heads for MHA).
    dim_head: Dimension of each attention head.
    eps: Epsilon for query RMSNorm numerical stability (and key if k_eps is None).
    heads: Deprecated alias for q_heads.
    k_eps: Optional separate epsilon for key RMSNorm numerical stability.

  Returns:
    Transposed and RoPE-rotated (q_out, k_out) of shapes [B, q_heads, Sq, dim_head]
    and [B, kv_heads, Sk, dim_head].
  """
  if heads is not None:
    q_heads = heads
  kv_heads = q_heads if kv_heads is None else kv_heads
  effective_k_eps = eps if k_eps is None else k_eps
  B, Sq, Dq = raw_q.shape
  _, Sk, Dk = raw_k.shape

  if Dq != q_heads * dim_head:
    raise ValueError(f"raw_q feature dim ({Dq}) must equal q_heads ({q_heads}) * dim_head ({dim_head})")
  if Dk != kv_heads * dim_head:
    raise ValueError(f"raw_k feature dim ({Dk}) must equal kv_heads ({kv_heads}) * dim_head ({dim_head})")

  # 1. FP32 RMSNorm for stability, then cast directly to target activation dtype.
  #
  # Association matters: Flax's `_normalize` computes `mul = rsqrt(var + eps)`,
  # then `mul *= scale`, then `y = x * mul` -- i.e. x * (rsqrt * scale). Folding
  # left-to-right as (x * rsqrt) * scale rounds differently and makes this path
  # drift from `nnx.RMSNorm` bit-for-bit. Keep the parenthesisation below in
  # step with Flax so the fused producer stays a pure fusion, not a numerical
  # change.
  q_fp32 = raw_q.astype(jnp.float32)
  q_rms = jax.lax.rsqrt(jnp.mean(jnp.square(q_fp32), axis=-1, keepdims=True) + eps)
  q_norm = (q_fp32 * (q_rms * q_norm_scale.astype(jnp.float32))).astype(raw_q.dtype)

  k_fp32 = raw_k.astype(jnp.float32)
  k_rms = jax.lax.rsqrt(jnp.mean(jnp.square(k_fp32), axis=-1, keepdims=True) + effective_k_eps)
  k_norm = (k_fp32 * (k_rms * k_norm_scale.astype(jnp.float32))).astype(raw_k.dtype)

  # 2. Reshape and transpose to [B, heads, S, dim_head]
  q_h = q_norm.reshape(B, Sq, q_heads, dim_head).transpose(0, 2, 1, 3)
  k_h = k_norm.reshape(B, Sk, kv_heads, dim_head).transpose(0, 2, 1, 3)

  # 3. Direct RoPE with freqs_cis [1, 1, S, dim_head // 2] in input dtype
  cos = jnp.real(freqs_cis).astype(raw_q.dtype)
  sin = jnp.imag(freqs_cis).astype(raw_q.dtype)
  cos_q, sin_q = cos[:, :, :Sq, :], sin[:, :, :Sq, :]
  cos_k, sin_k = cos[:, :, :Sk, :], sin[:, :, :Sk, :]

  q_pairs = q_h.reshape(B, q_heads, Sq, -1, 2)
  q_0, q_1 = q_pairs[..., 0], q_pairs[..., 1]
  q_out_0 = q_0 * cos_q - q_1 * sin_q
  q_out_1 = q_0 * sin_q + q_1 * cos_q
  q_out = jnp.stack([q_out_0, q_out_1], axis=-1).reshape(B, q_heads, Sq, dim_head)

  k_pairs = k_h.reshape(B, kv_heads, Sk, -1, 2)
  k_0, k_1 = k_pairs[..., 0], k_pairs[..., 1]
  k_out_0 = k_0 * cos_k - k_1 * sin_k
  k_out_1 = k_0 * sin_k + k_1 * cos_k
  k_out = jnp.stack([k_out_0, k_out_1], axis=-1).reshape(B, kv_heads, Sk, dim_head)

  return q_out, k_out
