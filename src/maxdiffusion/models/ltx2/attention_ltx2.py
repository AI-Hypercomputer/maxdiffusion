"""Copyright 2025 Google LLC

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


def apply_interleaved_rotary_emb(x: Array, freqs: Tuple[Array, Array]) -> Array:
  """Apply interleaved rotary embeddings.

  x: [B, S, H, D] or [B, S, D] depending on usage, but typically [B, S, H, D] in
  attention. freqs: (cos, sin)
  """
  cos, sin = freqs
  # x shape assumption: [..., D]
  # In PyTorch: x.unflatten(2, (-1, 2)).unbind(-1) -> [..., D//2, 2] -> unbind -> x_real, x_imag
  # We assume the last dimension is the head dimension D.

  # Reshape to [..., D//2, 2]
  x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
  x_real = x_reshaped[..., 0]
  x_imag = x_reshaped[..., 1]

  # Rotated: [-x_imag, x_real]
  x_rotated = jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)

  # Cast to float32 for high precision rotation, then back to x.dtype
  x_float = x.astype(jnp.float32)
  x_rotated_float = x_rotated.astype(jnp.float32)

  # cos, sin shapes needed generally matching x
  # cos, sin are usually [..., D] or broadcastable

  out = x_float * cos + x_rotated_float * sin
  return out.astype(x.dtype)


def apply_split_rotary_emb(x: Array, freqs: Tuple[Array, Array]) -> Array:
  """Apply split rotary embeddings."""
  cos, sin = freqs
  x_dtype = x.dtype

  # In PyTorch code, they handle some reshaping if x.ndim != 4 and cos.ndim == 4.
  # We will assume inputs are already aligned or broadcastable for simplicity in JAX,
  # but we might need to handle the specific logic if relevant.
  # The PyTorch code splits the last dim (2*r) into (d=2, r).
  # Then first_x = split_x[..., :1, :] (real part ish), second_x = split_x[..., 1:, :] (imag part ish)
  # But wait, split_x is (..., 2, r).
  # first_x is (..., 1, r), second_x is (..., 1, r).

  last = x.shape[-1]
  r = last // 2

  # (..., 2, r)
  split_x = x.reshape(*x.shape[:-1], 2, r).astype(jnp.float32)
  first_x = split_x[..., :1, :]
  second_x = split_x[..., 1:, :]

  # cos, sin need to broadcast to (..., 1, r)
  # They are passed as arguments.
  # In PyTorch: cos_u = cos.unsqueeze(-2)

  cos_u = jnp.expand_dims(cos, -2)
  sin_u = jnp.expand_dims(sin, -2)

  out = split_x * cos_u
  # first_out = out[..., :1, :]
  # second_out = out[..., 1:, :]

  # first_out.addcmul_(-sin_u, second_x) -> first_out = first_out - sin_u * second_x
  # second_out.addcmul_(sin_u, first_x)  -> second_out = second_out + sin_u * first_x

  first_out = out[..., 0:1, :] - sin_u * second_x
  second_out = (
      out[..., 1:2, :] + sin_u * first_x
  )  # Note: first_x from original split_x logic was index 0

  out = jnp.concatenate([first_out, second_out], axis=1)  # (..., 2, r)
  out = out.reshape(*x.shape[:-1], last)

  return out.astype(x_dtype)


class LTX2AudioVideoRotaryPosEmbed(nnx.Module):

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
    self.rope_type = rope_type
    self.base_num_frames = base_num_frames
    self.num_attention_heads = num_attention_heads
    self.base_height = base_height
    self.base_width = base_width
    self.sampling_rate = sampling_rate
    self.hop_length = hop_length
    # Calculate latent rate for audio
    self.audio_latents_per_second = (
        float(sampling_rate) / float(hop_length) / float(scale_factors[0])
    )
    self.scale_factors = scale_factors
    self.theta = theta
    self.causal_offset = causal_offset
    self.modality = modality
    if self.modality not in ["video", "audio"]:
      raise ValueError(
          f"Modality {modality} is not supported. Supported modalities are"
          " `video` and `audio`."
      )
    self.double_precision = double_precision

  def prepare_video_coords(
      self,
      batch_size: int,
      num_frames: int,
      height: int,
      width: int,
      fps: float = 24.0,
  ) -> Array:
    # Use linspace or arange.
    # To minimize stack/reshape, we can use broadcasting.

    # grid_f: [N_F]
    grid_f = jnp.arange(0, num_frames, self.patch_size_t, dtype=jnp.float32)
    grid_h = jnp.arange(0, height, self.patch_size, dtype=jnp.float32)
    grid_w = jnp.arange(0, width, self.patch_size, dtype=jnp.float32)

    # Meshgrid produces [N_F, N_H, N_W] for each
    # We want [3, N_F, N_H, N_W]
    grid = jnp.stack(
        jnp.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), axis=0
    )

    # Patch ends
    patch_size_arr = jnp.array(
        [self.patch_size_t, self.patch_size, self.patch_size], dtype=grid.dtype
    ).reshape(3, 1, 1, 1)

    # [3, N_F, N_H, N_W]
    patch_ends = grid + patch_size_arr

    # We want to flatten spatial dims completely.
    # grid: [3, N_F, N_H, N_W]
    # Flatten last 3 dims -> [3, P]
    grid_flat = grid.reshape(3, -1)
    patch_ends_flat = patch_ends.reshape(3, -1)

    # latent_coords: [3, P, 2] -> [start, end]
    # Interleave start/end coordinates?
    # The original return shape was [B, 3, P, 2].

    latent_coords = jnp.stack(
        [grid_flat, patch_ends_flat], axis=-1
    )  # [3, P, 2]

    # Reorder to [3, P, 2] is fine.

    # Expand batch: [B, 3, P, 2]
    # We can broadcast later but let's match return signature.
    latent_coords = jnp.broadcast_to(
        latent_coords[None, ...], (batch_size, 3, latent_coords.shape[1], 2)
    )

    # Scale to pixel coords
    scale_tensor = jnp.array(self.scale_factors).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_tensor

    # Causal offset for time dim (index 0)
    # pixel_coords[:, 0, ...] : [B, P, 2]
    # We can do this in-place or via updates

    t_coords = pixel_coords[:, 0, :, :]
    t_coords = jnp.clip(
        t_coords + self.causal_offset - self.scale_factors[0], a_min=0.0
    )
    t_coords = t_coords / fps

    pixel_coords = pixel_coords.at[:, 0, :, :].set(t_coords)

    return pixel_coords

  def prepare_audio_coords(
      self,
      batch_size: int,
      num_frames: int,
      shift: int = 0,
  ) -> Array:
    grid_f = jnp.arange(
        shift, num_frames + shift, self.patch_size_t, dtype=jnp.float32
    )

    audio_scale_factor = self.scale_factors[0]

    # Calculate start/end in one go
    # starts = grid_f
    # ends = grid_f + patch_size_t

    # [P, 2]
    grid_f_2 = jnp.stack([grid_f, grid_f + self.patch_size_t], axis=-1)

    grid_mel = grid_f_2 * audio_scale_factor
    grid_mel = jnp.clip(
        grid_mel + self.causal_offset - audio_scale_factor, a_min=0.0
    )
    grid_s = grid_mel * self.hop_length / self.sampling_rate

    # grid_s: [P, 2]
    # Output: [B, 1, P, 2]
    audio_coords = jnp.broadcast_to(grid_s, (batch_size, 1) + grid_s.shape)

    return audio_coords

  def __call__(self, coords: Array) -> Tuple[Array, Array]:
    # coords: [B, num_pos_dims, num_patches, 2] or similar
    if coords.ndim == 4:
      coords = (coords[..., 0] + coords[..., 1]) / 2.0
      # coords: [B, num_pos_dims, num_patches]

    # Normalize by max positions
    if self.modality == "video":
      max_positions = jnp.array(
          [self.base_num_frames, self.base_height, self.base_width]
      )
    elif self.modality == "audio":
      max_positions = jnp.array([self.base_num_frames])
    else:
      raise ValueError(f"Unknown modality: {self.modality}")

    # grid = coords / max_positions
    # coords [B, 3, P] (if video)
    # max [3]
    # grid [B, 3, P]
    max_positions_b = max_positions[None, :, None]
    grid = coords / max_positions_b

    # Transpose to [B, P, 3] for easier dot product
    grid = grid.transpose(0, 2, 1)

    num_rope_elems = grid.shape[-1] * 2

    freqs_dtype = jnp.float64 if self.double_precision else jnp.float32

    # linspace steps = dim // num_rope_elems
    steps = self.dim // num_rope_elems

    pow_indices = jnp.power(
        self.theta, jnp.linspace(0.0, 1.0, steps, dtype=freqs_dtype)
    )
    freqs = (pow_indices * jnp.pi / 2.0).astype(jnp.float32)

    # grid: [B, P, ndim]
    # freqs: [steps]
    # Outer product implicitly?
    # We want [B, P, ndim, steps] -> [B, P, steps, ndim] -> flatten

    # grid * 2 - 1
    grid_scaled = grid.astype(jnp.float32) * 2 - 1

    # [B, P, ndim, 1] * [1, 1, 1, steps] -> [B, P, ndim, steps]
    freqs_out = grid_scaled[..., None] * freqs[None, None, None, :]

    # Transpose to [B, P, steps, ndim] and flatten
    freqs_out = freqs_out.transpose(0, 1, 3, 2).reshape(
        grid.shape[0], grid.shape[1], -1
    )

    if self.rope_type == "interleaved":
      cos_freqs = jnp.repeat(jnp.cos(freqs_out), 2, axis=-1)
      sin_freqs = jnp.repeat(jnp.sin(freqs_out), 2, axis=-1)

      if self.dim % num_rope_elems != 0:
        pad_len = self.dim - cos_freqs.shape[-1]
        pad_shape = list(cos_freqs.shape)
        pad_shape[-1] = pad_len
        cos_padding = jnp.ones(pad_shape, dtype=cos_freqs.dtype)
        sin_padding = jnp.zeros(pad_shape, dtype=sin_freqs.dtype)
        cos_freqs = jnp.concatenate([cos_padding, cos_freqs], axis=-1)
        sin_freqs = jnp.concatenate([sin_padding, sin_freqs], axis=-1)

    elif self.rope_type == "split":
      cos_freq = jnp.cos(freqs_out)
      sin_freq = jnp.sin(freqs_out)

      expected_half_dim = self.dim // 2
      current_dim = cos_freq.shape[-1]
      pad_size = expected_half_dim - current_dim
      if pad_size > 0:
        pad_shape = list(cos_freq.shape)
        pad_shape[-1] = pad_size
        cos_padding = jnp.ones(pad_shape, dtype=cos_freq.dtype)
        sin_padding = jnp.zeros(pad_shape, dtype=sin_freq.dtype)
        cos_freq = jnp.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = jnp.concatenate([sin_padding, sin_freq], axis=-1)

      # Reshape for multi-head
      B, T, D_half = cos_freq.shape
      H = self.num_attention_heads

      cos_freq = cos_freq.reshape(B, T, H, -1)
      sin_freq = sin_freq.reshape(B, T, H, -1)

      cos_freqs = cos_freq.transpose(0, 2, 1, 3)  # [B, H, T, D_head//2]
      sin_freqs = sin_freq.transpose(0, 2, 1, 3)

    else:
      raise ValueError(f"Unknown rope_type: {self.rope_type}")

    return cos_freqs, sin_freqs


class LTX2Attention(nnx.Module):
  def __init__(
      self,
      query_dim: int,
      heads: int,
      dim_head: int,
      dropout: float = 0.0,
      bias: bool = False,
      added_kv_proj_dim: Optional[int] = None,
      added_proj_bias: bool = True,
      out_bias: bool = True,
      rngs: nnx.Rngs = None,
      mesh: Mesh = None,
      eps: float = 1e-5,
      rope_type: str = "interleaved",
      dtype: DType = jnp.float32,
  ):
    self.inner_dim = dim_head * heads
    self.query_dim = query_dim
    self.out_dim = query_dim
    self.heads = heads
    self.dim_head = dim_head
    self.dropout_rate = dropout
    self.rope_type = rope_type
    self.added_kv_proj_dim = added_kv_proj_dim

    self.to_q = nnx.Linear(
        query_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype
    )
    self.to_k = nnx.Linear(
        query_dim,
        self.inner_dim,
        use_bias=bias,
        rngs=rngs,
        dtype=dtype,
    )
    self.to_v = nnx.Linear(
        query_dim,
        self.inner_dim,
        use_bias=bias,
        rngs=rngs,
        dtype=dtype,
    )

    # QK Norm
    self.norm_q = nnx.RMSNorm(
        self.dim_head,
        epsilon=eps,
        dtype=dtype,
        use_scale=True,
        rngs=rngs,
    )
    self.norm_k = nnx.RMSNorm(
        self.dim_head,
        epsilon=eps,
        dtype=dtype,
        use_scale=True,
        rngs=rngs,
    )

    if added_kv_proj_dim is not None:
      self.add_q_proj = nnx.Linear(
          added_kv_proj_dim,
          self.inner_dim,
          use_bias=added_proj_bias,
          rngs=rngs,
          dtype=dtype,
      )
      self.add_k_proj = nnx.Linear(
          added_kv_proj_dim,
          self.inner_dim,
          use_bias=added_proj_bias,
          rngs=rngs,
          dtype=dtype,
      )
      self.add_v_proj = nnx.Linear(
          added_kv_proj_dim,
          self.inner_dim,
          use_bias=added_proj_bias,
          rngs=rngs,
          dtype=dtype,
      )
      self.norm_added_q = nnx.RMSNorm(
          self.dim_head,
          epsilon=eps,
          dtype=dtype,
          use_scale=True,
          rngs=rngs,
      )
      self.norm_added_k = nnx.RMSNorm(
          self.dim_head,
          epsilon=eps,
          dtype=dtype,
          use_scale=True,
          rngs=rngs,
      )
      self.to_add_out = nnx.Linear(
          self.inner_dim, query_dim, use_bias=out_bias, rngs=rngs, dtype=dtype
      )

    self.to_out = nnx.Linear(
        self.inner_dim, self.out_dim, use_bias=out_bias, rngs=rngs, dtype=dtype
    )

    if self.dropout_rate > 0:
      self.dropout_layer = nnx.Dropout(self.dropout_rate, rngs=rngs)
    else:
      self.dropout_layer = None

    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel="flash",
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
      image_rotary_emb: Optional[Tuple[Array, Array]] = None,
  ) -> Array:

    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)

    query = self.norm_q(query.reshape(*query.shape[:-1], self.heads, self.dim_head))
    key = self.norm_k(key.reshape(*key.shape[:-1], self.heads, self.dim_head))
    value = value.reshape(*value.shape[:-1], self.heads, self.dim_head)

    if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
      encoder_query = self.add_q_proj(encoder_hidden_states)
      encoder_key = self.add_k_proj(encoder_hidden_states)
      encoder_value = self.add_v_proj(encoder_hidden_states)

      encoder_query = self.norm_added_q(
          encoder_query.reshape(
              *encoder_query.shape[:-1], self.heads, self.dim_head
          )
      )
      encoder_key = self.norm_added_k(
          encoder_key.reshape(*encoder_key.shape[:-1], self.heads, self.dim_head)
      )
      encoder_value = encoder_value.reshape(
          *encoder_value.shape[:-1], self.heads, self.dim_head
      )

      query = jnp.concatenate([encoder_query, query], axis=1)
      key = jnp.concatenate([encoder_key, key], axis=1)
      value = jnp.concatenate([encoder_value, value], axis=1)

    # Reshape for RoPE and Attention
    query = query.reshape(query.shape[0], query.shape[1], -1)
    key = key.reshape(key.shape[0], key.shape[1], -1)
    value = value.reshape(value.shape[0], value.shape[1], -1)

    if image_rotary_emb is not None:
      if self.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, image_rotary_emb)
        key = apply_interleaved_rotary_emb(key, image_rotary_emb)
      elif self.rope_type == "split":
        query = apply_split_rotary_emb(query, image_rotary_emb)
        key = apply_split_rotary_emb(key, image_rotary_emb)

    attn_output = self.attention_op.apply_attention(
        query=query, key=key, value=value, attention_mask=attention_mask
    )

    if (
        encoder_hidden_states is not None
        and self.added_kv_proj_dim is not None
    ):
      encoder_attn_output = attn_output[
          :, : encoder_hidden_states.shape[1], :
      ]
      hidden_states = attn_output[:, encoder_hidden_states.shape[1] :, :]
      encoder_attn_output = self.to_add_out(encoder_attn_output)
    else:
      hidden_states = attn_output

    hidden_states = self.to_out(hidden_states)

    if self.dropout_layer is not None:
      hidden_states = self.dropout_layer(hidden_states)

    if (
        encoder_hidden_states is not None
        and self.added_kv_proj_dim is not None
    ):
      return hidden_states, encoder_attn_output
    else:
      return hidden_states
