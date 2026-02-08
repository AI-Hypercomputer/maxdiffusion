"""
Copyright 2025 Google LLC

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

from typing import Any, Dict, Optional, Tuple, Union
from flax import nnx
import jax
import jax.numpy as jnp
from ... import common_types
from ..attention_flax import NNXAttentionOp

Array = common_types.Array
Mesh = common_types.Mesh
DType = common_types.DType


def apply_rotary_emb(x: Array, freqs: Tuple[Array, Array]) -> Array:
  """
  Applies Interleaved RoPE to input x.
  Logic matches LTX-2 PyTorch: pairs neighbors [-x2, x1].

  Args:
      x: Input tensor [..., D]
      freqs: Tuple of (cos, sin), broadcasting to [..., D]
  """
  cos, sin = freqs

  # 1. Reshape to pair neighbors: [..., D] -> [..., D//2, 2]
  # This corresponds to "rearrange(..., (d r) -> ... d r, r=2)"
  x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

  # 2. Split into components
  # x_real = x[..., 0], x_imag = x[..., 1]
  x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]

  # 3. Rotate [-x2, x1]
  # Corresponds to "stack((-t2, t1))"
  x_rotated = jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)

  # 4. Apply frequencies (Float32 for stability)
  out = x.astype(jnp.float32) * cos + x_rotated.astype(jnp.float32) * sin

  return out.astype(x.dtype)


def apply_split_rotary_emb(x: Array, freqs: Tuple[Array, Array]) -> Array:
  """
  Applies Split RoPE to input x.
  Logic matches Diffusers apply_split_rotary_emb.

  Args:
      x: Input tensor.
         If ndim=3 [B, S, D], it will be reshaped to satisfy cos/sin shapes if needed.
      freqs: Tuple of (cos, sin).
             Expected to be [B, H, S, D//2] if coming from LTX2RotaryPosEmbed(split).
  """
  cos, sin = freqs

  x_dtype = x.dtype
  needed_reshape = False
  original_shape = x.shape

  if x.ndim != 4 and cos.ndim == 4:
    b = x.shape[0]
    h, s, r = cos.shape[1], cos.shape[2], cos.shape[3]
    x = x.reshape(b, s, h, -1).transpose(0, 2, 1, 3)
    needed_reshape = True

  last_dim = x.shape[-1]
  r = last_dim // 2

  split_x = x.reshape(*x.shape[:-1], 2, r)

  first_x = split_x[..., 0, :]
  second_x = split_x[..., 1, :]

  cos_u = jnp.expand_dims(cos, axis=-2)
  sin_u = jnp.expand_dims(sin, axis=-2)

  out = split_x * cos_u

  out_first = out[..., 0, :] - second_x * sin_u.squeeze(-2)
  out_second = out[..., 1, :] + first_x * sin_u.squeeze(-2)

  out = jnp.stack([out_first, out_second], axis=-2)
  out = out.reshape(*out.shape[:-2], last_dim)

  if needed_reshape:
    out = out.transpose(0, 2, 1, 3).reshape(original_shape)

  return out.astype(x_dtype)


class LTX2RotaryPosEmbed(nnx.Module):
  """
  Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.
  Matches logic of LTX2AudioVideoRotaryPosEmbed from Diffusers.
  """

  def __init__(
      self,
      dim: int,
      patch_size: int = 1,
      patch_size_t: int = 1,
      base_num_frames: int = 20,
      base_height: int = 2048,
      base_width: int = 2048,
      sampling_rate: int = 16000,
      hop_length: int = 160,
      scale_factors: Tuple[int, ...] = (8, 32, 32),
      theta: float = 10000.0,
      causal_offset: int = 1,
      modality: str = "video",
      double_precision: bool = True,
      rope_type: str = "interleaved",
      num_attention_heads: int = 32,
  ):
    self.dim = dim
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.base_num_frames = base_num_frames
    self.base_height = base_height
    self.base_width = base_width
    self.sampling_rate = sampling_rate
    self.hop_length = hop_length
    self.scale_factors = scale_factors
    self.theta = theta
    self.causal_offset = causal_offset
    self.modality = modality
    self.double_precision = double_precision
    self.rope_type = rope_type
    self.num_attention_heads = num_attention_heads

    if self.rope_type not in ["interleaved", "split"]:
      raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")

    if self.modality not in ["video", "audio"]:
      raise ValueError(f"Modality {modality} is not supported. Supported modalities are `video` and `audio`.")

    self.audio_latents_per_second = float(sampling_rate) / float(hop_length) / float(scale_factors[0])

  def prepare_video_coords(
      self,
      batch_size: int,
      num_frames: int,
      height: int,
      width: int,
      fps: float = 24.0,
  ) -> Array:
    # 1. Generate grid coordinates for each spatiotemporal dimension
    grid_f = jnp.arange(0, num_frames, self.patch_size_t, dtype=jnp.float32)
    grid_h = jnp.arange(0, height, self.patch_size, dtype=jnp.float32)
    grid_w = jnp.arange(0, width, self.patch_size, dtype=jnp.float32)

    # indexing='ij' ensures (frames, height, width) order
    grid = jnp.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
    grid = jnp.stack(grid, axis=0)  # [3, N_F, N_H, N_W]

    # 2. Get patch boundaries
    patch_size_arr = jnp.array((self.patch_size_t, self.patch_size, self.patch_size), dtype=grid.dtype)
    patch_size_delta = patch_size_arr.reshape(3, 1, 1, 1)
    patch_ends = grid + patch_size_delta

    # Combine start and end coordinates
    latent_coords = jnp.stack([grid, patch_ends], axis=-1)  # [3, N_F, N_H, N_W, 2]
    latent_coords = latent_coords.transpose(1, 2, 3, 0, 4)  # [N_F, N_H, N_W, 3, 2]
    latent_coords = latent_coords.reshape(-1, 3, 2)  # [num_patches, 3, 2]
    latent_coords = jnp.expand_dims(latent_coords, 0)  # [1, num_patches, 3, 2]
    latent_coords = jnp.tile(latent_coords, (batch_size, 1, 1, 1))  # [B, num_patches, 3, 2]

    latent_coords = jnp.stack([grid, patch_ends], axis=-1)  # [3, N_F, N_H, N_W, 2]
    latent_coords = latent_coords.reshape(3, -1, 2)  # [3, num_patches, 2]
    latent_coords = jnp.expand_dims(latent_coords, 0)  # [1, 3, num_patches, 2]
    latent_coords = jnp.tile(latent_coords, (batch_size, 1, 1, 1))  # [B, 3, num_patches, 2]

    # 3. Calculate pixel space coords
    scale_tensor = jnp.array(self.scale_factors, dtype=latent_coords.dtype)
    scale_tensor = scale_tensor.reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_tensor

    # Causal clamp logic
    # pixel_coords[:, 0, ...] selects Frame dimension.
    # pixel_coords shape: [B, 3, num_patches, 2] -> dim 1 is (F, H, W)
    frame_coords = pixel_coords[:, 0, ...]
    frame_coords = jnp.clip(frame_coords + self.causal_offset - self.scale_factors[0], a_min=0)
    pixel_coords = pixel_coords.at[:, 0, ...].set(frame_coords / fps)

    return pixel_coords

  def prepare_audio_coords(
      self,
      batch_size: int,
      num_frames: int,
      shift: int = 0,
  ) -> Array:
    # 1. Generate coordinates in frame (time) dimension
    grid_f = jnp.arange(shift, num_frames + shift, self.patch_size_t, dtype=jnp.float32)

    # 2. Start timestamps
    audio_scale_factor = self.scale_factors[0]
    grid_start_mel = grid_f * audio_scale_factor
    grid_start_mel = jnp.clip(grid_start_mel + self.causal_offset - audio_scale_factor, a_min=0)
    grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

    # 3. End timestamps
    grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
    grid_end_mel = jnp.clip(grid_end_mel + self.causal_offset - audio_scale_factor, a_min=0)
    grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

    # Stack [num_patches, 2]
    audio_coords = jnp.stack([grid_start_s, grid_end_s], axis=-1)
    # [num_patches, 2] -> [B, num_patches, 2]
    audio_coords = jnp.expand_dims(audio_coords, 0)
    audio_coords = jnp.tile(audio_coords, (batch_size, 1, 1))
    # [B, 1, num_patches, 2]
    audio_coords = jnp.expand_dims(audio_coords, 1)

    return audio_coords

  def prepare_coords(self, *args, **kwargs):
    if self.modality == "video":
      return self.prepare_video_coords(*args, **kwargs)
    elif self.modality == "audio":
      return self.prepare_audio_coords(*args, **kwargs)
    return None

  def __call__(self, coords: Array) -> Tuple[Array, Array]:
    # coords: [B, num_pos_dims, num_patches, 2]
    num_pos_dims = coords.shape[1]

    # 1. Midpoint
    if coords.ndim == 4:
      coords_start = coords[..., 0]
      coords_end = coords[..., 1]
      coords = (coords_start + coords_end) / 2.0  # [B, num_pos_dims, num_patches]

    # 2. Fractions
    if self.modality == "video":
      max_positions = jnp.array((self.base_num_frames, self.base_height, self.base_width), dtype=coords.dtype)
    elif self.modality == "audio":
      max_positions = jnp.array((self.base_num_frames,), dtype=coords.dtype)

    max_positions = max_positions[:num_pos_dims]
    max_positions = max_positions.reshape(1, num_pos_dims, 1)
    grid = coords / max_positions

    grid = grid.transpose(0, 2, 1)

    num_rope_elems = num_pos_dims * 2

    # 3. Frequencies
    freqs_dtype = jnp.float64 if self.double_precision else jnp.float32
    # linspace 0..1
    steps = self.dim // num_rope_elems
    pow_indices = jnp.power(self.theta, jnp.linspace(0.0, 1.0, steps, dtype=freqs_dtype))
    freqs = (pow_indices * jnp.pi / 2.0).astype(jnp.float32)  # [D//2K]

    # 4. Outer product
    freqs = (jnp.expand_dims(grid, -1) * 2 - 1) * freqs

    # Flatten last two dims: K, S -> K*S = dim//2
    freqs = freqs.reshape(*freqs.shape[:2], -1)

    # 5. Cos/Sin
    cos_freqs = jnp.cos(freqs)
    sin_freqs = jnp.sin(freqs)

    if self.rope_type == "interleaved":
      # repeat interleave: [c1, c2] -> [c1, c1, c2, c2]
      cos_freqs = jnp.repeat(cos_freqs, 2, axis=-1)
      sin_freqs = jnp.repeat(sin_freqs, 2, axis=-1)

      # Padding if needed
      if self.dim % num_rope_elems != 0:
        curr_dim = cos_freqs.shape[-1]
        pad_amt = self.dim - curr_dim
        if pad_amt > 0:
          cos_padding = jnp.ones((*cos_freqs.shape[:-1], pad_amt), dtype=cos_freqs.dtype)
          sin_padding = jnp.zeros((*sin_freqs.shape[:-1], pad_amt), dtype=sin_freqs.dtype)
          cos_freqs = jnp.concatenate([cos_padding, cos_freqs], axis=-1)
          sin_freqs = jnp.concatenate([sin_padding, sin_freqs], axis=-1)

    elif self.rope_type == "split":
      # Cos/Sin
      cos_freq = jnp.cos(freqs)
      sin_freq = jnp.sin(freqs)

      curr_dim = cos_freq.shape[-1]
      expected_dim = self.dim // 2
      pad_size = expected_dim - curr_dim

      if pad_size > 0:
        cos_padding = jnp.ones((*cos_freq.shape[:-1], pad_size), dtype=cos_freq.dtype)
        sin_padding = jnp.zeros((*sin_freq.shape[:-1], pad_size), dtype=sin_freq.dtype)
        cos_freq = jnp.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = jnp.concatenate([sin_padding, sin_freq], axis=-1)

      b = cos_freq.shape[0]
      s = cos_freq.shape[1]
      h = self.num_attention_heads

      cos_freqs = cos_freq.reshape(b, s, h, -1).transpose(0, 2, 1, 3)
      sin_freqs = sin_freq.reshape(b, s, h, -1).transpose(0, 2, 1, 3)

    return cos_freqs, sin_freqs


class LTX2Attention(nnx.Module):

  def __init__(
      self,
      query_dim: int,
      heads: int,
      dim_head: int,
      context_dim: Optional[int] = None,
      dropout: float = 0.0,
      bias: bool = True,  # LTX-2 uses bias=True for projections
      out_bias: bool = True,
      rngs: nnx.Rngs = None,
      mesh: Mesh = None,
      eps: float = 1e-6,
      dtype: DType = jnp.float32,
      attention_kernel: str = "flash",
      rope_type: str = "interleaved",
  ):
    self.heads = heads
    self.rope_type = rope_type
    self.dim_head = dim_head
    self.inner_dim = dim_head * heads
    self.dropout_rate = dropout

    # 1. Projections
    self.to_q = nnx.Linear(query_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)

    # Handle Self vs Cross Attention input dims
    kv_dim = context_dim if context_dim is not None else query_dim
    self.to_k = nnx.Linear(kv_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)
    self.to_v = nnx.Linear(kv_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)

    # 2. Normalization (Applied to full inner_dim, NOT per-head)
    self.norm_q = nnx.RMSNorm(
        self.inner_dim, epsilon=eps, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=True, rngs=rngs
    )
    self.norm_k = nnx.RMSNorm(
        self.inner_dim, epsilon=eps, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=True, rngs=rngs
    )

    # 3. Output
    self.to_out = nnx.Linear(self.inner_dim, query_dim, use_bias=out_bias, rngs=rngs, dtype=dtype)

    if self.dropout_rate > 0:
      self.dropout_layer = nnx.Dropout(self.dropout_rate, rngs=rngs)
    else:
      self.dropout_layer = None

    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=dim_head**-0.5,
        heads=heads,
        dim_head=dim_head,
        dtype=dtype,
    )

  def __call__(
      self,
      hidden_states: Array,
      encoder_hidden_states: Optional[Array] = None,
      attention_mask: Optional[Array] = None,
      rotary_emb: Optional[Tuple[Array, Array]] = None,
      k_rotary_emb: Optional[Tuple[Array, Array]] = None,
  ) -> Array:
    # Determine context (Self or Cross)
    context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    # 1. Project
    query = self.to_q(hidden_states)
    key = self.to_k(context)
    value = self.to_v(context)

    # 2. Norm (Full Inner Dimension)
    query = self.norm_q(query)
    key = self.norm_k(key)

    # 3. Apply RoPE to tensors of shape [B, S, InnerDim]
    # Frequencies are shape [B, S, InnerDim]
    # 3. Apply RoPE
    if rotary_emb is not None:
      if hasattr(self, "rope_type") and self.rope_type == "split":
        # Split RoPE: passing full freqs [B, H, S, D//2]
        # apply_split_rotary_emb handles reshaping query/key

        query = apply_split_rotary_emb(query, rotary_emb)

        if k_rotary_emb is not None:
          key = apply_split_rotary_emb(key, k_rotary_emb)
        elif encoder_hidden_states is None:
          key = apply_split_rotary_emb(key, rotary_emb)

      else:
        # Interleaved (Default)
        query = apply_rotary_emb(query, rotary_emb)
        if k_rotary_emb is not None:
          key = apply_rotary_emb(key, k_rotary_emb)
        elif encoder_hidden_states is None:
          key = apply_rotary_emb(key, rotary_emb)

    # 4. Attention
    # NNXAttentionOp expects flattened input [B, S, InnerDim] for flash kernel
    attn_output = self.attention_op.apply_attention(query=query, key=key, value=value, attention_mask=attention_mask)

    # 7. Output Projection
    hidden_states = self.to_out(attn_output)

    if self.dropout_layer is not None:
      hidden_states = self.dropout_layer(hidden_states)

    return hidden_states
