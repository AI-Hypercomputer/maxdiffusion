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

"""Pure Flax NNX Implementation of PRXPixel Transformer (Photoroom/prxpixel-t2i)."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class FlaxPRXPixelConfig:
  """Configuration parameters for PRXPixel Transformer."""

  def __init__(
      self,
      in_channels: int = 3,
      patch_size: int = 16,
      context_in_dim: int = 2048,
      hidden_size: int = 3584,
      mlp_ratio: float = 3.5,
      num_heads: int = 28,
      depth: int = 24,
      axes_dim: Tuple[int, int] = (64, 64),
      theta: int = 10000,
      time_factor: float = 1000.0,
      time_max_period: int = 10000,
      dtype: Any = jnp.float32,
      attention_kernel: str = "dot_product",
  ):
    self.in_channels = in_channels
    self.patch_size = patch_size
    self.out_channels = in_channels * patch_size * patch_size
    self.context_in_dim = context_in_dim
    self.hidden_size = hidden_size
    self.mlp_ratio = mlp_ratio
    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)  # 12544
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads  # 128
    self.depth = depth
    self.axes_dim = list(axes_dim)
    self.theta = theta
    self.time_factor = time_factor
    self.time_max_period = time_max_period
    self.dtype = dtype
    self.attention_kernel = attention_kernel


# -----------------------------------------------------------------------------
# Spatial Patchification Helpers
# -----------------------------------------------------------------------------


def img2seq_flax(img: jnp.ndarray, patch_size: int = 16) -> jnp.ndarray:
  """Flattens an image tensor into a sequence of non-overlapping patches.

  Input:  (B, C, H, W)
  Output: (B, (H//p)*(W//p), C * p * p)
  """
  b, c, h, w = img.shape
  p = patch_size
  gh, gw = h // p, w // p
  img = img.reshape(b, c, gh, p, gw, p)
  # Einsum "nchpwq->nhwcpq"
  img = jnp.transpose(img, (0, 2, 4, 1, 3, 5))
  img = img.reshape(b, gh * gw, c * p * p)
  return img


def seq2img_flax(seq: jnp.ndarray, patch_size: int, shape: Tuple[int, int, int, int]) -> jnp.ndarray:
  """Reconstructs an image tensor from a sequence of patches.

  Input:  (B, L, C * p * p)
  Output: (B, C, H, W)
  """
  b, c, h, w = shape
  p = patch_size
  gh, gw = h // p, w // p
  seq = seq.reshape(b, gh, gw, c, p, p)
  # Einsum "nhwcpq->nchpwq"
  seq = jnp.transpose(seq, (0, 3, 1, 4, 2, 5))
  seq = seq.reshape(b, c, h, w)
  return seq


def get_image_ids_flax(batch_size: int, height: int, width: int, patch_size: int = 16) -> jnp.ndarray:
  """Generates 2D patch coordinate indices of shape (batch_size, num_patches, 2)."""
  gh, gw = height // patch_size, width // patch_size
  row_ids = jnp.arange(gh)[:, None]
  col_ids = jnp.arange(gw)[None, :]
  row_grid = jnp.broadcast_to(row_ids, (gh, gw))
  col_grid = jnp.broadcast_to(col_ids, (gh, gw))
  img_ids = jnp.stack([row_grid, col_grid], axis=-1).reshape(gh * gw, 2)
  return jnp.broadcast_to(img_ids[None, :, :], (batch_size, gh * gw, 2))


# -----------------------------------------------------------------------------
# Positional Embeddings & 2D RoPE
# -----------------------------------------------------------------------------


def get_timestep_embedding_flax(
    timesteps: jnp.ndarray,
    embedding_dim: int = 256,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: int = 10000,
) -> jnp.ndarray:
  """Generates sinusoidal embeddings for timesteps or resolution dimensions."""
  half_dim = embedding_dim // 2
  exponent = -math.log(max_period) * jnp.arange(0, half_dim, dtype=jnp.float32)
  exponent = exponent / (half_dim - downscale_freq_shift)
  freqs = jnp.exp(exponent)
  args = timesteps[:, None].astype(jnp.float32) * freqs[None, :] * scale

  if flip_sin_to_cos:
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  else:
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
  return emb


class NNXPRXEmbedND(nnx.Module):
  """N-Dimensional Rotary Positional Embedding (RoPE) for 2D Spatial Patches."""

  def __init__(self, dim: int = 128, theta: int = 10000, axes_dim: List[int] = (64, 64)):
    self.dim = dim
    self.theta = theta
    self.axes_dim = list(axes_dim)

  def rope(self, pos: jnp.ndarray, dim: int, theta: int) -> jnp.ndarray:
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta**scale)
    out = pos[..., None].astype(jnp.float32) * omega[None, :]
    cos = jnp.cos(out)
    sin = jnp.sin(out)
    # Stack [cos, -sin, sin, cos] into shape (..., dim/2, 2, 2)
    m00 = cos
    m01 = -sin
    m10 = sin
    m11 = cos
    row0 = jnp.stack([m00, m01], axis=-1)
    row1 = jnp.stack([m10, m11], axis=-1)
    mat = jnp.stack([row0, row1], axis=-2)
    return mat

  def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
    n_axes = ids.shape[-1]
    emb = jnp.concatenate([self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(n_axes)], axis=-3)
    return emb[:, None, :, :, :, :]  # Shape: (B, 1, L_img, dim/2, 2, 2)


def apply_rope_flax(xq: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
  """Applies 2D RoPE rotation matrices to queries or keys.

  xq:        (B, num_heads, L, head_dim)
  freqs_cis: (B, 1, L, head_dim/2, 2, 2)
  """
  b, h, l, d = xq.shape
  xq_pairs = xq.astype(jnp.float32).reshape(b, h, l, d // 2, 2)  # (B, H, L, D/2, 2)
  # Matrix multiply [2, 2] with [2]: y_0 = m00 * x0 + m01 * x1, y_1 = m10 * x0 + m11 * x1
  m00 = freqs_cis[..., 0, 0]
  m01 = freqs_cis[..., 0, 1]
  m10 = freqs_cis[..., 1, 0]
  m11 = freqs_cis[..., 1, 1]

  x0 = xq_pairs[..., 0]
  x1 = xq_pairs[..., 1]

  y0 = m00 * x0 + m01 * x1
  y1 = m10 * x0 + m11 * x1

  out = jnp.stack([y0, y1], axis=-1).reshape(b, h, l, d)
  return out.astype(xq.dtype)


# -----------------------------------------------------------------------------
# Conditioning & Modulation Modules
# -----------------------------------------------------------------------------


class NNXMLPEmbedder(nnx.Module):
  """Two-layer MLP for timestep or resolution embeddings: in_layer -> SiLU -> out_layer."""

  def __init__(self, in_dim: int, hidden_dim: int, dtype: Any = jnp.float32, rngs: nnx.Rngs = None):
    self.in_layer = nnx.Linear(in_dim, hidden_dim, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.out_layer = nnx.Linear(hidden_dim, hidden_dim, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.out_layer(jax.nn.silu(self.in_layer(x)))


class NNXModulation(nnx.Module):
  """Modulation network generating 6D chunked parameters (shift, scale, gate) for attention & MLP."""

  def __init__(self, dim: int, dtype: Any = jnp.float32, rngs: nnx.Rngs = None):
    self.dim = dim
    self.lin = nnx.Linear(dim, 6 * dim, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)

  def __call__(self, vec: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
    out = self.lin(jax.nn.silu(vec))[:, None, :]  # (B, 1, 6 * dim)
    chunks = jnp.split(out, 6, axis=-1)
    mod_attn = (chunks[0], chunks[1], chunks[2])  # shift, scale, gate
    mod_mlp = (chunks[3], chunks[4], chunks[5])   # shift, scale, gate
    return mod_attn, mod_mlp


# -----------------------------------------------------------------------------
# Attention & Transformer Block
# -----------------------------------------------------------------------------


class NNXRMSNorm(nnx.Module):
  """Affine RMSNorm with scale parameter."""

  def __init__(self, dim: int, eps: float = 1e-6, dtype: Any = jnp.float32, rngs: nnx.Rngs = None):
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + self.eps)
    return (normed * self.weight).astype(x.dtype)


class NNXLayerNorm(nnx.Module):
  """Affine-free LayerNorm."""

  def __init__(self, eps: float = 1e-6):
    self.eps = eps

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x.astype(jnp.float32), axis=-1, keepdims=True)
    variance = jnp.var(x.astype(jnp.float32), axis=-1, keepdims=True)
    normed = (x.astype(jnp.float32) - mean) * jax.lax.rsqrt(variance + self.eps)
    return normed.astype(x.dtype)


class NNXPRXAttention(nnx.Module):
  """Asymmetric Cross-Attention for PRXPixel: Image Queries attend to Joint [Text; Image] Keys/Values."""

  def __init__(self, config: FlaxPRXPixelConfig, rngs: nnx.Rngs = None):
    self.config = config
    self.heads = config.num_heads
    self.head_dim = config.head_dim
    self.hidden_size = config.hidden_size
    dtype = config.dtype

    self.img_qkv_proj = nnx.Linear(config.hidden_size, 3 * config.hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.norm_q = NNXRMSNorm(self.head_dim, eps=1e-6, dtype=dtype, rngs=rngs)
    self.norm_k = NNXRMSNorm(self.head_dim, eps=1e-6, dtype=dtype, rngs=rngs)

    self.txt_kv_proj = nnx.Linear(config.hidden_size, 2 * config.hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.norm_added_k = NNXRMSNorm(self.head_dim, eps=1e-6, dtype=dtype, rngs=rngs)

    self.to_out = nnx.List([
        nnx.Linear(config.hidden_size, config.hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)
    ])

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
      image_rotary_emb: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    b, l_img, _ = hidden_states.shape
    _, l_txt, _ = encoder_hidden_states.shape

    # 1. Image QKV projection
    img_qkv = self.img_qkv_proj(hidden_states)
    img_qkv = img_qkv.reshape(b, l_img, 3, self.heads, self.head_dim)
    # Permute to (3, B, H, L_img, D)
    img_q = jnp.transpose(img_qkv[:, :, 0], (0, 2, 1, 3))  # (B, H, L_img, D)
    img_k = jnp.transpose(img_qkv[:, :, 1], (0, 2, 1, 3))
    img_v = jnp.transpose(img_qkv[:, :, 2], (0, 2, 1, 3))

    # Apply Q/K RMSNorm
    img_q = self.norm_q(img_q)
    img_k = self.norm_k(img_k)

    # 2. Text KV projection
    txt_kv = self.txt_kv_proj(encoder_hidden_states)
    txt_kv = txt_kv.reshape(b, l_txt, 2, self.heads, self.head_dim)
    txt_k = jnp.transpose(txt_kv[:, :, 0], (0, 2, 1, 3))  # (B, H, L_txt, D)
    txt_v = jnp.transpose(txt_kv[:, :, 1], (0, 2, 1, 3))
    txt_k = self.norm_added_k(txt_k)

    # 3. Apply 2D RoPE to Image Queries & Keys
    if image_rotary_emb is not None:
      img_q = apply_rope_flax(img_q, image_rotary_emb)
      img_k = apply_rope_flax(img_k, image_rotary_emb)

    # 4. Concatenate [Text; Image] Keys and Values
    k = jnp.concatenate([txt_k, img_k], axis=2)  # (B, H, L_txt + L_img, D)
    v = jnp.concatenate([txt_v, img_v], axis=2)

    # 5. Attention Computation
    scale = 1.0 / math.sqrt(self.head_dim)
    q_f = img_q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)

    # scores: (B, H, L_img, L_txt + L_img)
    scores = jnp.matmul(q_f, jnp.swapaxes(k_f, -1, -2)) * scale

    if attention_mask is not None:
      # Joint mask: [attention_mask (B, L_txt), ones (B, L_img)]
      ones_img = jnp.ones((b, l_img), dtype=jnp.bool_)
      joint_mask = jnp.concatenate([attention_mask.astype(jnp.bool_), ones_img], axis=-1)
      joint_mask = joint_mask[:, None, None, :]  # (B, 1, 1, L_txt + L_img)
      scores = jnp.where(joint_mask, scores, -1e4)

    attn_probs = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(attn_probs, v_f).astype(self.config.dtype)  # (B, H, L_img, D)

    # Transpose to (B, L_img, H * D)
    attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(b, l_img, self.heads * self.head_dim)
    output = self.to_out[0](attn_out)
    return output


class NNXPRXBlock(nnx.Module):
  """Single PRX Transformer Block with AdaLN Modulation, Cross-Attention, and GELU-tanh MLP."""

  def __init__(self, config: FlaxPRXPixelConfig, rngs: nnx.Rngs = None):
    self.config = config
    dtype = config.dtype
    hidden_size = config.hidden_size
    mlp_hidden_dim = config.mlp_hidden_dim

    self.img_pre_norm = NNXLayerNorm(eps=1e-6)
    self.attention = NNXPRXAttention(config=config, rngs=rngs)

    self.post_attention_layernorm = NNXLayerNorm(eps=1e-6)
    self.gate_proj = nnx.Linear(hidden_size, mlp_hidden_dim, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.up_proj = nnx.Linear(hidden_size, mlp_hidden_dim, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.down_proj = nnx.Linear(mlp_hidden_dim, hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs)

    self.modulation = NNXModulation(hidden_size, dtype=dtype, rngs=rngs)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      temb: jnp.ndarray,
      image_rotary_emb: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    mod_attn, mod_mlp = self.modulation(temb)
    attn_shift, attn_scale, attn_gate = mod_attn
    mlp_shift, mlp_scale, mlp_gate = mod_mlp

    # Attention Sub-Layer
    hidden_states_mod = (1 + attn_scale) * self.img_pre_norm(hidden_states) + attn_shift
    attn_out = self.attention(
        hidden_states=hidden_states_mod,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = hidden_states + attn_gate * attn_out

    # MLP Sub-Layer with GELU tanh approximation
    x = (1 + mlp_scale) * self.post_attention_layernorm(hidden_states) + mlp_shift
    gate = jax.nn.gelu(self.gate_proj(x), approximate=True)
    up = self.up_proj(x)
    mlp_out = self.down_proj(gate * up)
    hidden_states = hidden_states + mlp_gate * mlp_out

    return hidden_states


class NNXFinalLayer(nnx.Module):
  """Final Layer: Adaptive LayerNorm Modulation -> Linear(3584, 768)."""

  def __init__(self, hidden_size: int = 3584, out_channels: int = 768, dtype: Any = jnp.float32, rngs: nnx.Rngs = None):
    self.norm_final = NNXLayerNorm(eps=1e-6)
    self.linear = nnx.Linear(hidden_size, out_channels, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)
    # adaLN_modulation: SiLU -> Linear(hidden_size, 2 * hidden_size)
    self.adaLN_modulation = nnx.List([
        nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)
    ])

  def __call__(self, x: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    mod = self.adaLN_modulation[0](jax.nn.silu(vec))[:, None, :]
    chunks = jnp.split(mod, 2, axis=-1)
    shift, scale = chunks[0], chunks[1]
    normed = (1 + scale) * self.norm_final(x) + shift
    return self.linear(normed)


class NNXResolutionEmbedder(nnx.Module):
  """Resolution conditioning: height + width sinusoidal embeddings -> MLP."""

  def __init__(self, in_dim: int = 256, hidden_dim: int = 3584, dtype: Any = jnp.float32, rngs: nnx.Rngs = None):
    self.mlp = NNXMLPEmbedder(in_dim=in_dim, hidden_dim=hidden_dim, dtype=dtype, rngs=rngs)

  def __call__(self, height: int, width: int, batch_size: int) -> jnp.ndarray:
    h_emb = get_timestep_embedding_flax(jnp.array([float(height)]), embedding_dim=128, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)
    w_emb = get_timestep_embedding_flax(jnp.array([float(width)]), embedding_dim=128, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)
    res_emb = jnp.concatenate([h_emb, w_emb], axis=-1)
    return self.mlp(jnp.broadcast_to(res_emb, (batch_size, 256)))


# -----------------------------------------------------------------------------
# Full Transformer Model
# -----------------------------------------------------------------------------


class NNXPRXPixelTransformer2DModel(nnx.Module):
  """Complete PRXPixel 2D Transformer in Flax NNX."""

  def __init__(self, config: FlaxPRXPixelConfig, rngs: nnx.Rngs = None):
    self.config = config
    dtype = config.dtype

    self.pe_embedder = NNXPRXEmbedND(dim=config.head_dim, theta=config.theta, axes_dim=config.axes_dim)
    # Two-layer image input bottleneck
    self.img_in = nnx.List([
        nnx.Linear(768, 768, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs),
        nnx.Linear(768, config.hidden_size, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs),
    ])

    self.txt_in = nnx.Linear(config.context_in_dim, config.hidden_size, use_bias=True, dtype=dtype, param_dtype=dtype, rngs=rngs)
    self.time_in = NNXMLPEmbedder(in_dim=256, hidden_dim=config.hidden_size, dtype=dtype, rngs=rngs)
    self.resolution_embedder = NNXResolutionEmbedder(in_dim=256, hidden_dim=config.hidden_size, dtype=dtype, rngs=rngs)

    self.blocks = nnx.List([NNXPRXBlock(config=config, rngs=rngs) for _ in range(config.depth)])
    self.final_layer = NNXFinalLayer(hidden_size=config.hidden_size, out_channels=config.out_channels, dtype=dtype, rngs=rngs)

  def _compute_timestep_embedding(self, timestep: jnp.ndarray) -> jnp.ndarray:
    raw_time = get_timestep_embedding_flax(
        timesteps=timestep,
        embedding_dim=256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
        scale=self.config.time_factor,
        max_period=self.config.time_max_period,
    )
    return self.time_in(raw_time)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      timestep: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    b, c, h, w = hidden_states.shape

    # 1. Text projection
    txt = self.txt_in(encoder_hidden_states)

    # 2. Pixel patchify & two-layer bottleneck
    patches = img2seq_flax(hidden_states, patch_size=self.config.patch_size)
    img = self.img_in[0](patches)
    img = self.img_in[1](img)

    # 3. 2D Positional Embeddings
    img_ids = get_image_ids_flax(b, h, w, patch_size=self.config.patch_size)
    pe = self.pe_embedder(img_ids)

    # 4. Conditioning vector = Timestep
    vec = self._compute_timestep_embedding(timestep)

    # 5. Transformer Blocks
    for block in self.blocks:
      img = block(
          hidden_states=img,
          encoder_hidden_states=txt,
          temb=vec,
          image_rotary_emb=pe,
          attention_mask=attention_mask,
      )

    # 6. Final Layer & unpatchify
    final_patches = self.final_layer(img, vec)
    output = seq2img_flax(final_patches, patch_size=self.config.patch_size, shape=(b, c, h, w))
    return output
