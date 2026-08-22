"""Optimized fused producers for Wan Transformer Block and Attention."""

from typing import Tuple
import jax
import jax.numpy as jnp


def fused_ln_adaln(
    x: jax.Array,
    scale_msa: jax.Array,
    shift_msa: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
  """Fusion-friendly FP32 LayerNorm + AdaLN scale/shift modulation producer."""
  x_fp32 = x.astype(jnp.float32)
  mean = jnp.mean(x_fp32, axis=-1, keepdims=True)
  diff = x_fp32 - mean
  var = jnp.mean(jnp.square(diff), axis=-1, keepdims=True)
  x_ln = diff * jax.lax.rsqrt(var + eps)
  scale_term = 1.0 + scale_msa.astype(jnp.float32)
  x_mod = (x_ln * scale_term + shift_msa.astype(jnp.float32)).astype(x.dtype)
  return x_mod


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
    eps: Epsilon for RMSNorm numerical stability.
    heads: Deprecated alias for q_heads.

  Returns:
    Transposed and RoPE-rotated (q_out, k_out) of shapes [B, q_heads, Sq, dim_head]
    and [B, kv_heads, Sk, dim_head].
  """
  if heads is not None:
    q_heads = heads
  kv_heads = q_heads if kv_heads is None else kv_heads
  B, Sq, Dq = raw_q.shape
  _, Sk, Dk = raw_k.shape

  if Dq != q_heads * dim_head:
    raise ValueError(f"raw_q feature dim ({Dq}) must equal q_heads ({q_heads}) * dim_head ({dim_head})")
  if Dk != kv_heads * dim_head:
    raise ValueError(f"raw_k feature dim ({Dk}) must equal kv_heads ({kv_heads}) * dim_head ({dim_head})")

  # 1. FP32 RMSNorm for stability, then cast directly to target activation dtype
  q_fp32 = raw_q.astype(jnp.float32)
  q_rms = jax.lax.rsqrt(jnp.mean(jnp.square(q_fp32), axis=-1, keepdims=True) + eps)
  q_norm = (q_fp32 * q_rms * q_norm_scale.astype(jnp.float32)).astype(raw_q.dtype)

  k_fp32 = raw_k.astype(jnp.float32)
  k_rms = jax.lax.rsqrt(jnp.mean(jnp.square(k_fp32), axis=-1, keepdims=True) + eps)
  k_norm = (k_fp32 * k_rms * k_norm_scale.astype(jnp.float32)).astype(raw_k.dtype)

  # 2. Reshape and transpose to [B, heads, S, dim_head]
  q_h = q_norm.reshape(B, Sq, q_heads, dim_head).transpose(0, 2, 1, 3)
  k_h = k_norm.reshape(B, Sk, kv_heads, dim_head).transpose(0, 2, 1, 3)

  # 3. Direct RoPE with freqs_cis [1, 1, S, dim_head // 2] in input dtype
  cos = jnp.real(freqs_cis).astype(raw_q.dtype)
  sin = jnp.imag(freqs_cis).astype(raw_q.dtype)

  q_pairs = q_h.reshape(B, q_heads, Sq, -1, 2)
  q_0, q_1 = q_pairs[..., 0], q_pairs[..., 1]
  q_out_0 = q_0 * cos - q_1 * sin
  q_out_1 = q_0 * sin + q_1 * cos
  q_out = jnp.concatenate([q_out_0[..., None], q_out_1[..., None]], axis=-1).reshape(B, q_heads, Sq, dim_head)

  k_pairs = k_h.reshape(B, kv_heads, Sk, -1, 2)
  k_0, k_1 = k_pairs[..., 0], k_pairs[..., 1]
  k_out_0 = k_0 * cos - k_1 * sin
  k_out_1 = k_0 * sin + k_1 * cos
  k_out = jnp.concatenate([k_out_0[..., None], k_out_1[..., None]], axis=-1).reshape(B, kv_heads, Sk, dim_head)

  return q_out, k_out
