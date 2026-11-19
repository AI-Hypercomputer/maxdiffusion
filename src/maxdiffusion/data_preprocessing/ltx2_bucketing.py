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

"""3D spatial-temporal video + 1D audio bucketing, normalization, patchification,
per-sample RoPE coordinate generation, and mesh-sharded batch iterators for LTX-2.
"""

from collections import deque
from dataclasses import dataclass
from glob import glob
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


ArrayLike = Union[np.ndarray, jax.Array]


# ==============================================================================
# 1. Video & Audio Latent Normalization and Patchification
# ==============================================================================


def _xp(arr: ArrayLike):
  return jnp if isinstance(arr, jax.Array) else np


def normalize_video_latents(
    latents: ArrayLike,
    latents_mean: ArrayLike,
    latents_std: ArrayLike,
    scaling_factor: float = 1.0,
    channel_last: bool = False,
) -> ArrayLike:
  """Normalizes 4D `(C, F, H, W)` or 5D `(B, C, F, H, W)` video latents before patchify.

  Matches `LTX2Pipeline._normalize_latents` (`ltx2_pipeline.py:L1509-L1518`).
  When `channel_last=True` (`(..., F, H, W, C)` direct VAE output), normalizes
  along the trailing channel axis.
  """
  xp = _xp(latents)
  if latents.ndim not in (4, 5):
    raise ValueError(f"Video latents must be 4D or 5D for normalization, got shape {latents.shape}.")

  mean = xp.asarray(latents_mean, dtype=latents.dtype)
  std = xp.asarray(latents_std, dtype=latents.dtype)
  if channel_last:
    bshape = (1,) * (latents.ndim - 1) + (-1,)
  elif latents.ndim == 5:
    bshape = (1, -1, 1, 1, 1)
  else:
    bshape = (-1, 1, 1, 1)

  mean = mean.reshape(bshape)
  std = std.reshape(bshape)
  return (latents - mean) * scaling_factor / std


def denormalize_video_latents(
    latents: ArrayLike,
    latents_mean: ArrayLike,
    latents_std: ArrayLike,
    scaling_factor: float = 1.0,
    channel_last: bool = False,
) -> ArrayLike:
  """Denormalizes 4D/5D video latents (`ltx2_pipeline.py:L1521-L1530`)."""
  xp = _xp(latents)
  if latents.ndim not in (4, 5):
    raise ValueError(f"Video latents must be 4D or 5D for denormalization, got shape {latents.shape}.")

  mean = xp.asarray(latents_mean, dtype=latents.dtype)
  std = xp.asarray(latents_std, dtype=latents.dtype)
  if channel_last:
    bshape = (1,) * (latents.ndim - 1) + (-1,)
  elif latents.ndim == 5:
    bshape = (1, -1, 1, 1, 1)
  else:
    bshape = (-1, 1, 1, 1)

  mean = mean.reshape(bshape)
  std = std.reshape(bshape)
  return latents * std / scaling_factor + mean


def normalize_audio_latents(
    packed_audio_latents: ArrayLike,
    latents_mean: ArrayLike,
    latents_std: ArrayLike,
) -> ArrayLike:
  """Normalizes packed `(T, C*F_mel)` or `(B, T, C*F_mel)` audio latents *after* patchify.

  Matches `LTX2Pipeline.prepare_audio_latents` (`ltx2_pipeline.py:L1616-L1620, L1788-L1800`)
  where `audio_vae.latents_mean` / `latents_std` have shape `(base_channels,)` (`C * F_mel`).
  """
  xp = _xp(packed_audio_latents)
  if packed_audio_latents.ndim not in (2, 3):
    raise ValueError(f"Packed audio latents must be 2D or 3D, got shape {packed_audio_latents.shape}.")
  mean = xp.asarray(latents_mean, dtype=packed_audio_latents.dtype)
  std = xp.asarray(latents_std, dtype=packed_audio_latents.dtype)
  return (packed_audio_latents - mean) / std


def denormalize_audio_latents(
    packed_audio_latents: ArrayLike,
    latents_mean: ArrayLike,
    latents_std: ArrayLike,
) -> ArrayLike:
  """Denormalizes packed `(T, C*F_mel)` or `(B, T, C*F_mel)` audio latents before unpatchify."""
  xp = _xp(packed_audio_latents)
  if packed_audio_latents.ndim not in (2, 3):
    raise ValueError(f"Packed audio latents must be 2D or 3D, got shape {packed_audio_latents.shape}.")
  mean = xp.asarray(latents_mean, dtype=packed_audio_latents.dtype)
  std = xp.asarray(latents_std, dtype=packed_audio_latents.dtype)
  return (packed_audio_latents * std) + mean


def patchify_video_latents(
    latents: ArrayLike,
    patch_size: int = 1,
    patch_size_t: int = 1,
    channel_last: bool = False,
) -> ArrayLike:
  """Packs 4D `(C, F, H, W)` or 5D `(B, C, F, H, W)` video latents into patch sequences.

  Matches `LTX2Pipeline._pack_latents` (`ltx2_pipeline.py:L1464-L1482`):
  flattens sequence in `(N_F, N_H, N_W)` order and patch channels in
  `(C, patch_size_t, patch_size, patch_size)` order.
  """
  xp = _xp(latents)
  squeeze_batch = False
  if latents.ndim == 4:
    latents = xp.expand_dims(latents, axis=0)
    squeeze_batch = True
  elif latents.ndim != 5:
    raise ValueError(f"Video latents must be 4D or 5D, got shape {latents.shape}.")

  if channel_last:
    # (B, F, H, W, C) -> (B, C, F, H, W)
    latents = xp.transpose(latents, (0, 4, 1, 2, 3))

  batch_size, channels, num_frames, height, width = latents.shape
  if num_frames % patch_size_t != 0 or height % patch_size != 0 or width % patch_size != 0:
    raise ValueError(
        f"Video latent shape (F={num_frames}, H={height}, W={width}) must be divisible by "
        f"(patch_size_t={patch_size_t}, patch_size={patch_size}, patch_size={patch_size})."
    )

  n_f = num_frames // patch_size_t
  n_h = height // patch_size
  n_w = width // patch_size
  packed = latents.reshape(batch_size, channels, n_f, patch_size_t, n_h, patch_size, n_w, patch_size)
  packed = xp.transpose(packed, (0, 2, 4, 6, 1, 3, 5, 7)).reshape(
      batch_size, n_f * n_h * n_w, channels * patch_size_t * patch_size * patch_size
  )
  return packed[0] if squeeze_batch else packed


def unpatchify_video_latents(
    packed: ArrayLike,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    channel_last: bool = False,
) -> ArrayLike:
  """Unpacks 2D `(S, D)` or 3D `(B, S, D)` video patches back to 4D/5D latents."""
  xp = _xp(packed)
  squeeze_batch = False
  if packed.ndim == 2:
    packed = xp.expand_dims(packed, axis=0)
    squeeze_batch = True
  elif packed.ndim != 3:
    raise ValueError(f"Packed video latents must be 2D or 3D, got shape {packed.shape}.")

  batch_size = packed.shape[0]
  n_f = num_frames // patch_size_t
  n_h = height // patch_size
  n_w = width // patch_size
  latents = packed.reshape(batch_size, n_f, n_h, n_w, -1, patch_size_t, patch_size, patch_size)
  latents = xp.transpose(latents, (0, 4, 1, 5, 2, 6, 3, 7)).reshape(batch_size, -1, num_frames, height, width)
  if channel_last:
    latents = xp.transpose(latents, (0, 2, 3, 4, 1))
  return latents[0] if squeeze_batch else latents


def patchify_audio_latents(
    audio_latents: ArrayLike,
    patch_size: Optional[int] = None,
    patch_size_t: Optional[int] = None,
    layout: str = "BCFM",
) -> ArrayLike:
  """Packs 3D/4D audio latents into `(T', D)` or `(B, T', D)` tokens.

  Supports:
    - `layout="BCFM"`: `(B, C, T, F_mel)` / `(C, T, F_mel)` matching `LTX2Pipeline._pack_audio_latents`
      (`ltx2_pipeline.py:L1647-L1680`).
    - `layout="BTFC"`: `(B, T, F_mel, C)` / `(T, F_mel, C)` matching
      `FlaxLTX2AudioAudioPatchifier.patchify` (`autoencoder_kl_ltx2_audio.py:L421-L432`).
  """
  xp = _xp(audio_latents)
  squeeze_batch = False
  if audio_latents.ndim == 3:
    audio_latents = xp.expand_dims(audio_latents, axis=0)
    squeeze_batch = True
  elif audio_latents.ndim != 4:
    raise ValueError(f"Audio latents must be 3D or 4D, got shape {audio_latents.shape}.")

  if layout == "BTFC":
    # (B, T, F_mel, C) -> (B, C, T, F_mel)
    audio_latents = xp.transpose(audio_latents, (0, 3, 1, 2))
  elif layout != "BCFM":
    raise ValueError(f"Unsupported audio layout {layout!r}. Expected 'BCFM' or 'BTFC'.")

  batch_size, channels, latent_length, latent_mel_bins = audio_latents.shape
  if patch_size is not None and patch_size_t is not None:
    if latent_length % patch_size_t != 0 or latent_mel_bins % patch_size != 0:
      raise ValueError(
          f"Audio latent shape (T={latent_length}, F_mel={latent_mel_bins}) must be divisible by "
          f"(patch_size_t={patch_size_t}, patch_size={patch_size})."
      )
    n_t = latent_length // patch_size_t
    n_m = latent_mel_bins // patch_size
    packed = audio_latents.reshape(batch_size, channels, n_t, patch_size_t, n_m, patch_size)
    packed = xp.transpose(packed, (0, 2, 4, 1, 3, 5)).reshape(batch_size, n_t * n_m, channels * patch_size_t * patch_size)
  else:
    packed = xp.transpose(audio_latents, (0, 2, 1, 3)).reshape(batch_size, latent_length, channels * latent_mel_bins)

  return packed[0] if squeeze_batch else packed


def unpatchify_audio_latents(
    packed: ArrayLike,
    latent_length: int,
    num_mel_bins: int,
    num_channels: int,
    patch_size: Optional[int] = None,
    patch_size_t: Optional[int] = None,
    layout: str = "BCFM",
) -> ArrayLike:
  """Unpacks 2D `(S, D)` or 3D `(B, S, D)` audio tokens back to 3D/4D audio latents."""
  xp = _xp(packed)
  squeeze_batch = False
  if packed.ndim == 2:
    packed = xp.expand_dims(packed, axis=0)
    squeeze_batch = True
  elif packed.ndim != 3:
    raise ValueError(f"Packed audio latents must be 2D or 3D, got shape {packed.shape}.")

  batch_size = packed.shape[0]
  if patch_size is not None and patch_size_t is not None:
    n_t = latent_length // patch_size_t
    n_m = num_mel_bins // patch_size
    latents = packed.reshape(batch_size, n_t, n_m, num_channels, patch_size_t, patch_size)
    latents = xp.transpose(latents, (0, 3, 1, 4, 2, 5)).reshape(batch_size, num_channels, latent_length, num_mel_bins)
  else:
    latents = packed.reshape(batch_size, latent_length, num_channels, num_mel_bins)
    latents = xp.transpose(latents, (0, 2, 1, 3))

  if layout == "BTFC":
    latents = xp.transpose(latents, (0, 2, 3, 1))
  elif layout != "BCFM":
    raise ValueError(f"Unsupported audio layout {layout!r}. Expected 'BCFM' or 'BTFC'.")

  return latents[0] if squeeze_batch else latents


# ==============================================================================
# 2. Per-Sample 3D Video & 1D Audio RoPE Coordinate Generation
# ==============================================================================


def compute_video_coords_1sample(
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    patch_size: int = 1,
    patch_size_t: int = 1,
    scale_factors: Tuple[int, int, int] = (8, 32, 32),
    causal_offset: int = 1,
) -> np.ndarray:
  """Computes exact per-sample 3D RoPE coordinates of shape `[3, S_vid, 2]` (`float32`).

  Matches `LTX2RotaryPosEmbed.prepare_video_coords(1, num_frames, height, width, fps=fps)[0]`
  (`attention_ltx2.py:L154-L194`).
  """
  if fps <= 0:
    raise ValueError(f"fps must be positive, got {fps}.")
  grid_f = np.arange(0, num_frames, patch_size_t, dtype=np.float32)
  grid_h = np.arange(0, height, patch_size, dtype=np.float32)
  grid_w = np.arange(0, width, patch_size, dtype=np.float32)

  grid = np.stack(np.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), axis=0)  # [3, N_F, N_H, N_W]
  patch_delta = np.array((patch_size_t, patch_size, patch_size), dtype=np.float32).reshape(3, 1, 1, 1)
  patch_ends = grid + patch_delta

  latent_coords = np.stack([grid, patch_ends], axis=-1).reshape(3, -1, 2)  # [3, S_vid, 2]
  scale_arr = np.array(scale_factors, dtype=np.float32).reshape(3, 1, 1)
  pixel_coords = latent_coords * scale_arr
  pixel_coords[0] = np.maximum(pixel_coords[0] + np.float32(causal_offset) - np.float32(scale_factors[0]), 0.0) / np.float32(
      fps
  )
  return pixel_coords


def compute_audio_coords_1sample(
    audio_num_frames: int,
    patch_size_t: int = 1,
    audio_scale_factor: int = 4,
    causal_offset: int = 1,
    hop_length: int = 160,
    sampling_rate: int = 16000,
    shift: int = 0,
) -> np.ndarray:
  """Computes exact per-sample 1D audio RoPE coordinates of shape `[1, S_aud, 2]` (`float32`).

  Matches `LTX2RotaryPosEmbed.prepare_audio_coords(1, audio_num_frames, shift=shift)[0]`
  (`attention_ltx2.py:L196-L224`).
  """
  if audio_num_frames <= 0:
    return np.zeros((1, 0, 2), dtype=np.float32)

  grid_f = np.arange(shift, audio_num_frames + shift, patch_size_t, dtype=np.float32)
  scale_f = np.float32(audio_scale_factor)
  offset_f = np.float32(causal_offset)
  sec_per_mel = np.float32(hop_length) / np.float32(sampling_rate)

  start_mel = np.maximum(grid_f * scale_f + offset_f - scale_f, 0.0)
  end_mel = np.maximum((grid_f + np.float32(patch_size_t)) * scale_f + offset_f - scale_f, 0.0)

  start_s = start_mel * sec_per_mel
  end_s = end_mel * sec_per_mel
  return np.stack([start_s, end_s], axis=-1)[None, :, :]  # [1, S_aud, 2]


def encode_with_ltx2_connectors(
    connectors: Any,
    prompt_embeds: ArrayLike,
    prompt_attention_mask: ArrayLike,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Runs `LTX2AudioVideoGemmaTextEncoder` to produce register-filled `(video_embeds, audio_embeds, mask)`.

  Matches `LTX2Pipeline._run_connectors` (`ltx2_pipeline.py:L1825-L1828, L2066-L2071`),
  ensuring training conditioning matches serve-time register-filled embeddings.
  """
  mask_bool = jnp.asarray(prompt_attention_mask, dtype=jnp.bool_)
  return connectors(prompt_embeds, mask_bool)


# ==============================================================================
# 3. Bucket Spec, Alignment & Deterministic Bucket Selection
# ==============================================================================


def align_bucket_length(
    length: int,
    context_parallelism: int = 1,
    block_multiple: int = 128,
    ring_shards: int = 1,
    scale_block_by_context: bool = False,
) -> int:
  """Rounds `max(1, length)` up to satisfy context-parallel, ring, and Splash tile divisibility.

  - Default (`scale_block_by_context=False`): aligns to
    `lcm(context_parallelism, block_multiple * ring_shards)`, guaranteeing
    `seq_len % context_parallelism == 0` (`attention_flax.py:L345`) and block divisibility per ring shard.
  - When `scale_block_by_context=True`: aligns to `context_parallelism * block_multiple` so local
    `seq_len // context_parallelism` is already a multiple of `block_multiple` inside local `flash`.
  """
  if context_parallelism <= 0 or block_multiple <= 0 or ring_shards <= 0:
    raise ValueError(
        f"context_parallelism ({context_parallelism}), block_multiple ({block_multiple}), "
        f"and ring_shards ({ring_shards}) must be positive."
    )
  target = max(1, int(length))
  if scale_block_by_context:
    unit = context_parallelism * block_multiple
  else:
    unit = math.lcm(context_parallelism, block_multiple * ring_shards)
  return ((target + unit - 1) // unit) * unit


@dataclass(frozen=True)
class LTX2BucketSpec:
  """Static 3-stream `(b_vid, b_aud, b_txt)` sequence-length bucket specification."""

  b_vid: int
  b_aud: int
  b_txt: int

  def __post_init__(self):
    if self.b_vid <= 0 or self.b_aud <= 0 or self.b_txt <= 0:
      raise ValueError(f"Bucket capacities must be positive, got {self}.")

  def as_tuple(self) -> Tuple[int, int, int]:
    return (self.b_vid, self.b_aud, self.b_txt)

  def aligned(
      self,
      context_parallelism: int = 1,
      vid_block_multiple: int = 128,
      aud_block_multiple: int = 8,
      txt_block_multiple: int = 128,
      ring_shards: int = 1,
      scale_vid_block_by_context: bool = False,
  ) -> "LTX2BucketSpec":
    return LTX2BucketSpec(
        b_vid=align_bucket_length(
            self.b_vid,
            context_parallelism=context_parallelism,
            block_multiple=vid_block_multiple,
            ring_shards=ring_shards,
            scale_block_by_context=scale_vid_block_by_context,
        ),
        b_aud=align_bucket_length(
            self.b_aud,
            context_parallelism=context_parallelism,
            block_multiple=aud_block_multiple,
            ring_shards=ring_shards,
            scale_block_by_context=False,
        ),
        b_txt=align_bucket_length(
            self.b_txt,
            context_parallelism=context_parallelism,
            block_multiple=txt_block_multiple,
            ring_shards=1,
            scale_block_by_context=False,
        ),
    )

  def validate_for_mesh(self, context_parallelism: int = 1, ring_shards: int = 1) -> None:
    """Validates sequence divisibility for `hidden_states`, `audio_hidden_states`, and `encoder_hidden_states`."""
    if context_parallelism < 1 or ring_shards < 1:
      raise ValueError(f"Invalid context_parallelism={context_parallelism} or ring_shards={ring_shards}.")
    if context_parallelism > 1:
      if self.b_vid % context_parallelism != 0:
        raise ValueError(f"Bucket b_vid={self.b_vid} must be divisible by context_parallelism={context_parallelism}.")
      if self.b_txt % context_parallelism != 0:
        raise ValueError(
            f"Bucket b_txt={self.b_txt} must be divisible by context_parallelism={context_parallelism} "
            "because encoder_hidden_states is sharded on 'context' (transformer_ltx2.py:L657)."
        )
      if self.b_aud % context_parallelism != 0:
        raise ValueError(
            f"Bucket b_aud={self.b_aud} must be divisible by context_parallelism={context_parallelism} "
            f"(ring_shards={ring_shards}) for audio self-attention context sharding."
        )


def select_bucket(
    s_vid: int,
    s_aud: int,
    s_txt: int,
    buckets: Sequence[LTX2BucketSpec],
    policy: str = "error",
    align_kwargs: Optional[Mapping[str, Any]] = None,
) -> Optional[LTX2BucketSpec]:
  """Selects the smallest `LTX2BucketSpec` satisfying `(b_vid >= s_vid, b_aud >= s_aud, b_txt >= s_txt)`."""
  sorted_buckets = sorted(buckets, key=lambda b: (b.b_vid + b.b_aud + b.b_txt, b.b_vid, b.b_aud, b.b_txt))
  eff_aud = max(1, s_aud)
  eff_txt = max(1, s_txt)
  for bucket in sorted_buckets:
    if bucket.b_vid >= s_vid and bucket.b_aud >= eff_aud and bucket.b_txt >= eff_txt:
      return bucket

  if policy == "drop":
    return None
  if policy == "pad_to_aligned":
    kw = dict(align_kwargs or {})
    return LTX2BucketSpec(b_vid=max(1, s_vid), b_aud=eff_aud, b_txt=eff_txt).aligned(**kw)
  if policy == "error":
    raise ValueError(
        f"Sample token lengths (s_vid={s_vid}, s_aud={s_aud}, s_txt={s_txt}) exceed all configured buckets: "
        f"{[b.as_tuple() for b in sorted_buckets]}."
    )
  raise ValueError(f"Unknown bucket overflow policy {policy!r}. Expected 'error', 'drop', or 'pad_to_aligned'.")


# ==============================================================================
# 4. Sample Representation, Right-Padding Collation & Mesh Sharding
# ==============================================================================


@dataclass
class LTX2Sample:
  """Unpadded variable-shape joint audio-video (or video-only) training sample."""

  video_latents: ArrayLike
  encoder_hidden_states: ArrayLike
  latent_shape: Optional[Tuple[int, int, int]] = None  # Required when video_latents is already 2D packed (S_vid, D_vid)
  audio_latents: Optional[ArrayLike] = None
  audio_encoder_hidden_states: Optional[ArrayLike] = None
  encoder_attention_mask: Optional[ArrayLike] = None
  audio_encoder_attention_mask: Optional[ArrayLike] = None
  fps: float = 24.0
  has_audio: bool = True
  is_normalized: bool = True
  video_channel_last: bool = False
  audio_layout: str = "BCFM"


def _infer_sample_token_counts(
    sample: LTX2Sample,
    patch_size: int = 1,
    patch_size_t: int = 1,
    audio_patch_size: Optional[int] = None,
    audio_patch_size_t: Optional[int] = None,
) -> Tuple[int, int, int]:
  v = sample.video_latents
  if v.ndim == 2:
    if sample.latent_shape is None:
      raise ValueError("LTX2Sample.latent_shape=(F, H, W) is required when video_latents is already 2D packed.")
    f_lat, h_lat, w_lat = sample.latent_shape
    expected_s = (f_lat // patch_size_t) * (h_lat // patch_size) * (w_lat // patch_size)
    if v.shape[0] != expected_s:
      raise ValueError(
          f"2D video_latents length {v.shape[0]} != expected {expected_s} from latent_shape={sample.latent_shape}."
      )
    s_vid = v.shape[0]
  elif v.ndim == 4:
    if sample.video_channel_last:
      f_lat, h_lat, w_lat, _ = v.shape
    else:
      _, f_lat, h_lat, w_lat = v.shape
    s_vid = (f_lat // patch_size_t) * (h_lat // patch_size) * (w_lat // patch_size)
  else:
    raise ValueError(f"LTX2Sample.video_latents must be 2D packed or 4D unbatched, got shape {v.shape}.")

  if not sample.has_audio or sample.audio_latents is None or sample.audio_latents.shape[0] == 0:
    s_aud = 0
  else:
    a = sample.audio_latents
    if a.ndim == 2:
      s_aud = a.shape[0]
    elif a.ndim == 3:
      if sample.audio_layout == "BTFC":
        t_aud, f_mel, _ = a.shape
      else:
        _, t_aud, f_mel = a.shape
      if audio_patch_size is not None and audio_patch_size_t is not None:
        s_aud = (t_aud // audio_patch_size_t) * (f_mel // audio_patch_size)
      else:
        s_aud = t_aud
    else:
      raise ValueError(f"LTX2Sample.audio_latents must be 2D packed or 3D unbatched, got shape {a.shape}.")

  s_txt = sample.encoder_hidden_states.shape[0]
  return s_vid, s_aud, s_txt


def _validate_connector_registers_for_text_mode(
    text_padding_mode: str,
    connector_num_learnable_registers: Optional[int],
    connectors: Optional[Any],
) -> None:
  if text_padding_mode not in ("post_connector", "right_pad_masked"):
    raise ValueError(f"text_padding_mode must be 'post_connector' or 'right_pad_masked', got {text_padding_mode!r}.")
  if text_padding_mode == "right_pad_masked":
    num_regs = connector_num_learnable_registers
    if num_regs is None and connectors is not None:
      vid_conn = getattr(connectors, "video_embeddings_connector", connectors)
      num_regs = getattr(vid_conn, "num_learnable_registers", 0)
    if num_regs is not None and num_regs > 0:
      raise ValueError(
          f"text_padding_mode='right_pad_masked' is incompatible with num_learnable_registers={num_regs} > 0 "
          "(embeddings_connector_ltx2.py:L142-L182 replaces padding with learnable registers and emits an all-True "
          "mask at serve time). Pre-encode with connectors and use text_padding_mode='post_connector'."
      )


def collate_ltx2_bucket_batch(
    samples: Sequence[LTX2Sample],
    bucket: LTX2BucketSpec,
    *,
    patch_size: int = 1,
    patch_size_t: int = 1,
    vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
    causal_offset: int = 1,
    audio_in_channels: int = 128,
    audio_patch_size: Optional[int] = None,
    audio_patch_size_t: Optional[int] = None,
    audio_rope_patch_size_t: int = 1,
    audio_scale_factor: int = 4,
    audio_hop_length: int = 160,
    audio_sampling_rate: int = 16000,
    text_padding_mode: str = "post_connector",
    connector_num_learnable_registers: Optional[int] = None,
    connectors: Optional[Any] = None,
    video_latents_mean: Optional[ArrayLike] = None,
    video_latents_std: Optional[ArrayLike] = None,
    video_scaling_factor: float = 1.0,
    audio_latents_mean: Optional[ArrayLike] = None,
    audio_latents_std: Optional[ArrayLike] = None,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
  """Packs and right-pads heterogeneous `LTX2Sample`s into static `(b_vid, b_aud, b_txt)` arrays."""
  if not samples:
    raise ValueError("collate_ltx2_bucket_batch requires at least 1 sample.")

  _validate_connector_registers_for_text_mode(text_padding_mode, connector_num_learnable_registers, connectors)

  batch_size = len(samples)
  b_vid, b_aud, b_txt = bucket.b_vid, bucket.b_aud, bucket.b_txt
  out_dtype = np.dtype(dtype)

  vid_list: List[np.ndarray] = []
  aud_list: List[np.ndarray] = []
  txt_list: List[np.ndarray] = []
  atxt_list: List[np.ndarray] = []
  vcoords_list: List[np.ndarray] = []
  acoords_list: List[np.ndarray] = []
  vmask = np.zeros((batch_size, b_vid), dtype=np.bool_)
  amask = np.zeros((batch_size, b_aud), dtype=np.bool_)
  tmask = np.zeros((batch_size, b_txt), dtype=np.bool_)
  atmask = np.zeros((batch_size, b_txt), dtype=np.bool_)
  mod_mask = np.zeros((batch_size, 1, 1), dtype=np.float32)

  for i, sample in enumerate(samples):
    # 1. Video normalization (before patchify) + 3D patchify + per-sample 3D coords.
    v_raw = np.asarray(sample.video_latents)
    if v_raw.ndim == 4:
      if sample.video_channel_last:
        v_raw = np.transpose(v_raw, (3, 0, 1, 2))
      _, f_lat, h_lat, w_lat = v_raw.shape
      if not sample.is_normalized:
        if video_latents_mean is None or video_latents_std is None:
          raise ValueError("video_latents_mean and video_latents_std are required when sample.is_normalized=False.")
        v_raw = normalize_video_latents(
            v_raw, video_latents_mean, video_latents_std, video_scaling_factor, channel_last=False
        )
      v_packed = patchify_video_latents(v_raw, patch_size=patch_size, patch_size_t=patch_size_t, channel_last=False)
    elif v_raw.ndim == 2:
      if not sample.is_normalized:
        raise ValueError("2D pre-packed video_latents must already be normalized (is_normalized=True).")
      if sample.latent_shape is None:
        raise ValueError("sample.latent_shape=(F, H, W) is required when video_latents is 2D.")
      f_lat, h_lat, w_lat = sample.latent_shape
      v_packed = v_raw
    else:
      raise ValueError(f"Unsupported video_latents rank {v_raw.ndim} (expected 2D or 4D).")

    s_vid, d_vid = v_packed.shape
    if s_vid > b_vid:
      raise ValueError(f"Sample {i} video length {s_vid} exceeds bucket b_vid={b_vid}.")

    v_coords = compute_video_coords_1sample(
        f_lat,
        h_lat,
        w_lat,
        fps=sample.fps,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        scale_factors=vae_scale_factors,
        causal_offset=causal_offset,
    )
    if v_coords.shape[1] != s_vid:
      raise ValueError(f"Sample {i} video_coords length {v_coords.shape[1]} != packed video length {s_vid}.")

    v_pad = np.zeros((b_vid, d_vid), dtype=out_dtype)
    v_pad[:s_vid] = v_packed.astype(out_dtype)
    vid_list.append(v_pad)

    vc_pad = np.zeros((3, b_vid, 2), dtype=np.float32)
    vc_pad[:, :s_vid, :] = v_coords
    vcoords_list.append(vc_pad)
    vmask[i, :s_vid] = True

    # 2. Audio patchify + normalization (after patchify) + per-sample 1D coords.
    has_audio = bool(sample.has_audio and sample.audio_latents is not None and sample.audio_latents.shape[0] > 0)
    ac_pad = np.zeros((1, b_aud, 2), dtype=np.float32)
    if has_audio:
      a_raw = np.asarray(sample.audio_latents)
      if a_raw.ndim == 3:
        t_aud = a_raw.shape[0] if sample.audio_layout == "BTFC" else a_raw.shape[1]
        a_packed = patchify_audio_latents(
            a_raw, patch_size=audio_patch_size, patch_size_t=audio_patch_size_t, layout=sample.audio_layout
        )
      elif a_raw.ndim == 2:
        a_packed = a_raw
        t_aud = a_packed.shape[0] * audio_rope_patch_size_t
      else:
        raise ValueError(f"Unsupported audio_latents rank {a_raw.ndim} (expected 2D or 3D).")

      if not sample.is_normalized:
        if audio_latents_mean is None or audio_latents_std is None:
          raise ValueError("audio_latents_mean and audio_latents_std are required when sample.is_normalized=False.")
        a_packed = normalize_audio_latents(a_packed, audio_latents_mean, audio_latents_std)

      s_aud, d_aud = a_packed.shape
      if s_aud > b_aud:
        raise ValueError(f"Sample {i} audio length {s_aud} exceeds bucket b_aud={b_aud}.")

      a_coords = compute_audio_coords_1sample(
          t_aud,
          patch_size_t=audio_rope_patch_size_t,
          audio_scale_factor=audio_scale_factor,
          causal_offset=causal_offset,
          hop_length=audio_hop_length,
          sampling_rate=audio_sampling_rate,
      )
      if a_coords.shape[1] != s_aud:
        raise ValueError(f"Sample {i} audio_coords length {a_coords.shape[1]} != packed audio length {s_aud}.")

      a_pad = np.zeros((b_aud, d_aud), dtype=out_dtype)
      a_pad[:s_aud] = a_packed.astype(out_dtype)
      aud_list.append(a_pad)

      ac_pad[:, :s_aud, :] = a_coords
      amask[i, :s_aud] = True
      mod_mask[i, 0, 0] = 1.0
    else:
      d_aud = aud_list[0].shape[1] if aud_list else audio_in_channels
      aud_list.append(np.zeros((b_aud, d_aud), dtype=out_dtype))
      mod_mask[i, 0, 0] = 0.0

    acoords_list.append(ac_pad)

    # 3. Text embeddings & masks (`post_connector` vs `right_pad_masked`).
    t_enc = np.asarray(sample.encoder_hidden_states)
    at_enc = np.asarray(
        sample.audio_encoder_hidden_states
        if sample.audio_encoder_hidden_states is not None
        else sample.encoder_hidden_states
    )
    if t_enc.ndim != 2 or at_enc.ndim != 2:
      raise ValueError(f"Text embeddings must be 2D (L_txt, D), got {t_enc.shape} and {at_enc.shape}.")
    if t_enc.shape[0] != at_enc.shape[0]:
      raise ValueError(
          f"encoder_hidden_states length {t_enc.shape[0]} != audio_encoder_hidden_states length {at_enc.shape[0]}."
      )

    s_txt = t_enc.shape[0]
    if text_padding_mode == "post_connector":
      if s_txt != b_txt:
        raise ValueError(
            f"Sample {i} has L_txt={s_txt} != bucket.b_txt={b_txt} under text_padding_mode='post_connector'. "
            "Post-connector embeddings have fixed register-filled length; pass matching b_txt or explicitly opt into "
            "text_padding_mode='right_pad_masked' when num_learnable_registers == 0."
        )
      txt_list.append(t_enc.astype(out_dtype))
      atxt_list.append(at_enc.astype(out_dtype))
      if sample.encoder_attention_mask is not None:
        em = np.asarray(sample.encoder_attention_mask, dtype=np.bool_)
        if em.shape != (b_txt,):
          raise ValueError(f"sample.encoder_attention_mask must have shape ({b_txt},), got {em.shape}.")
        tmask[i] = em
      else:
        tmask[i, :] = True

      if sample.audio_encoder_attention_mask is not None:
        aem = np.asarray(sample.audio_encoder_attention_mask, dtype=np.bool_)
        if aem.shape != (b_txt,):
          raise ValueError(f"sample.audio_encoder_attention_mask must have shape ({b_txt},), got {aem.shape}.")
        atmask[i] = aem
      else:
        atmask[i] = tmask[i]
    else:
      if s_txt > b_txt:
        raise ValueError(f"Sample {i} text length {s_txt} exceeds bucket b_txt={b_txt}.")
      t_pad = np.zeros((b_txt, t_enc.shape[1]), dtype=out_dtype)
      t_pad[:s_txt] = t_enc.astype(out_dtype)
      txt_list.append(t_pad)

      at_pad = np.zeros((b_txt, at_enc.shape[1]), dtype=out_dtype)
      at_pad[:s_txt] = at_enc.astype(out_dtype)
      atxt_list.append(at_pad)

      if sample.encoder_attention_mask is not None:
        em = np.asarray(sample.encoder_attention_mask, dtype=np.bool_)
        if em.shape != (s_txt,):
          raise ValueError(f"sample.encoder_attention_mask must have shape ({s_txt},), got {em.shape}.")
        tmask[i, :s_txt] = em
      else:
        tmask[i, :s_txt] = True

      if sample.audio_encoder_attention_mask is not None:
        aem = np.asarray(sample.audio_encoder_attention_mask, dtype=np.bool_)
        if aem.shape != (s_txt,):
          raise ValueError(f"sample.audio_encoder_attention_mask must have shape ({s_txt},), got {aem.shape}.")
        atmask[i, :s_txt] = aem
      else:
        atmask[i, :s_txt] = tmask[i, :s_txt]

  # Normalize retro-filled video-only audio width if sample 0 had no audio but a later sample did.
  d_aud_final = max(a.shape[1] for a in aud_list)
  aud_list = [a if a.shape[1] == d_aud_final else np.zeros((b_aud, d_aud_final), dtype=out_dtype) for a in aud_list]

  return {
      "hidden_states": np.stack(vid_list, axis=0),
      "audio_hidden_states": np.stack(aud_list, axis=0),
      "encoder_hidden_states": np.stack(txt_list, axis=0),
      "audio_encoder_hidden_states": np.stack(atxt_list, axis=0),
      "video_coords": np.stack(vcoords_list, axis=0),
      "audio_coords": np.stack(acoords_list, axis=0),
      "video_attention_mask": vmask,
      "audio_attention_mask": amask,
      "encoder_attention_mask": tmask,
      "audio_encoder_attention_mask": atmask,
      "modality_mask": mod_mask,
      "bucket_key": bucket.as_tuple(),
  }


def get_ltx2_batch_shardings(
    mesh: Mesh,
    bucket: LTX2BucketSpec,
) -> Dict[str, NamedSharding]:
  """Returns exact `NamedSharding`s matching `LTX2VideoTransformerBlock` entry constraints."""
  cp = mesh.shape.get("context", 1)
  batch_axis = (
      ("data", "fsdp")
      if ("data" in mesh.axis_names and "fsdp" in mesh.axis_names)
      else ("data" if "data" in mesh.axis_names else ("fsdp" if "fsdp" in mesh.axis_names else None))
  )
  vid_seq_axis = "context" if ("context" in mesh.axis_names and bucket.b_vid % cp == 0) else None
  txt_seq_axis = "context" if ("context" in mesh.axis_names and bucket.b_txt % cp == 0) else None

  return {
      "hidden_states": NamedSharding(mesh, P(batch_axis, vid_seq_axis, None)),
      "audio_hidden_states": NamedSharding(mesh, P(batch_axis, None, None)),
      "encoder_hidden_states": NamedSharding(mesh, P(batch_axis, txt_seq_axis, None)),
      "audio_encoder_hidden_states": NamedSharding(mesh, P(batch_axis, None, None)),
      "video_coords": NamedSharding(mesh, P(batch_axis, None, vid_seq_axis, None)),
      "audio_coords": NamedSharding(mesh, P(batch_axis, None, None, None)),
      "video_attention_mask": NamedSharding(mesh, P(batch_axis, None)),
      "audio_attention_mask": NamedSharding(mesh, P(batch_axis, None)),
      "encoder_attention_mask": NamedSharding(mesh, P(batch_axis, None)),
      "audio_encoder_attention_mask": NamedSharding(mesh, P(batch_axis, None)),
      "modality_mask": NamedSharding(mesh, P(batch_axis, None, None)),
  }


def shard_ltx2_batch_for_mesh(
    batch: Mapping[str, Any],
    mesh: Mesh,
    *,
    input_is_process_local: bool = False,
) -> Dict[str, Any]:
  """Places collated batch arrays onto `mesh` without `multihost_dataloading`'s `B >= num_devices` restriction.

  Multi-host contract:
    - `input_is_process_local=False` (default): `batch` holds the global `(global_B, ...)` batch on every process
      (e.g. single-host or deterministically seeded synthetic iterators); uses `jax.make_array_from_callback`,
      which supports `'context'` spanning hosts.
    - `input_is_process_local=True`: `batch` holds this process's batch slice `(global_B // process_count, ...)`
      with host-complete sequence dimensions (`'context'` intra-host); passes explicit `global_shape` to
      `jax.make_array_from_process_local_data`.
  """
  bucket_tuple = batch["bucket_key"]
  bucket = LTX2BucketSpec(*bucket_tuple)
  dp_shards = mesh.shape.get("data", 1) * mesh.shape.get("fsdp", 1)
  local_b = int(batch["hidden_states"].shape[0])
  global_b = local_b * jax.process_count() if input_is_process_local else local_b
  if global_b % dp_shards != 0:
    raise ValueError(f"Global batch size {global_b} must be divisible by data * fsdp = {dp_shards}.")

  shardings = get_ltx2_batch_shardings(mesh, bucket)
  out: Dict[str, Any] = {"bucket_key": bucket_tuple}
  for key, val in batch.items():
    if key == "bucket_key":
      continue
    arr = np.asarray(val)
    sharding = shardings[key]
    if input_is_process_local:
      global_shape = (global_b,) + arr.shape[1:]
      out[key] = jax.make_array_from_process_local_data(sharding, arr, global_shape=global_shape)
    else:
      out[key] = jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])
  return out


# ==============================================================================
# 5. Pre-Encoded & Synthetic Multi-Bucket Batch Iterators
# ==============================================================================


def _coerce_to_sample(item: Union[LTX2Sample, Mapping[str, Any]]) -> LTX2Sample:
  if isinstance(item, LTX2Sample):
    return item
  if isinstance(item, Mapping):
    latent_shape = item.get("latent_shape", None)
    if latent_shape is not None:
      latent_shape = tuple(int(x) for x in latent_shape)
    return LTX2Sample(
        video_latents=item["video_latents"],
        encoder_hidden_states=item["encoder_hidden_states"],
        latent_shape=latent_shape,
        audio_latents=item.get("audio_latents", None),
        audio_encoder_hidden_states=item.get("audio_encoder_hidden_states", None),
        encoder_attention_mask=item.get("encoder_attention_mask", None),
        audio_encoder_attention_mask=item.get("audio_encoder_attention_mask", None),
        fps=float(item.get("fps", 24.0)),
        has_audio=bool(item.get("has_audio", item.get("audio_latents", None) is not None)),
        is_normalized=bool(item.get("is_normalized", True)),
        video_channel_last=bool(item.get("video_channel_last", False)),
        audio_layout=str(item.get("audio_layout", "BCFM")),
    )
  raise TypeError(f"Expected LTX2Sample or Mapping, got {type(item).__name__}.")


def iter_preencoded_npz_samples(npz_dir: str, repeat: bool = False) -> Iterator[LTX2Sample]:
  """Streams pre-encoded `.npz` sample files sorted deterministically by filename."""
  paths = sorted(glob(os.path.join(npz_dir, "*.npz")))
  if not paths:
    raise FileNotFoundError(f"No .npz files found in {npz_dir!r}.")
  while True:
    for path in paths:
      with np.load(path, allow_pickle=False) as data:
        d: Dict[str, Any] = {k: data[k] for k in data.files}
      if "fps" in d:
        d["fps"] = float(np.asarray(d["fps"]).item())
      if "has_audio" in d:
        d["has_audio"] = bool(np.asarray(d["has_audio"]).item())
      if "is_normalized" in d:
        d["is_normalized"] = bool(np.asarray(d["is_normalized"]).item())
      yield _coerce_to_sample(d)
    if not repeat:
      break


class LTX2BucketedBatchIterator:
  """Routes variable-shape `LTX2Sample`s into per-bucket FIFO queues and emits static-bucket batches."""

  def __init__(
      self,
      sample_source: Iterable[Union[LTX2Sample, Mapping[str, Any]]],
      buckets: Sequence[LTX2BucketSpec],
      batch_size: int,
      *,
      mesh: Optional[Mesh] = None,
      overflow_policy: str = "error",
      align_kwargs: Optional[Mapping[str, Any]] = None,
      drop_remainder: bool = True,
      input_is_process_local: bool = False,
      collate_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if not buckets:
      raise ValueError("buckets must be non-empty.")

    self.sample_source = sample_source
    self.buckets = tuple(buckets)
    self.batch_size = batch_size
    self.mesh = mesh
    self.overflow_policy = overflow_policy
    self.align_kwargs = dict(align_kwargs or {})
    self.drop_remainder = drop_remainder
    self.input_is_process_local = input_is_process_local
    self.collate_kwargs = dict(collate_kwargs or {})

    if self.mesh is not None:
      cp = self.mesh.shape.get("context", 1)
      for b in self.buckets:
        b.validate_for_mesh(context_parallelism=cp)

    self._reset_state()

  def _reset_state(self):
    self._iter = iter(self.sample_source)
    self._queues: Dict[LTX2BucketSpec, deque] = {}
    self._exhausted = False

  def __iter__(self):
    self._reset_state()
    return self

  def _emit(self, bucket: LTX2BucketSpec, count: int) -> Dict[str, Any]:
    q = self._queues[bucket]
    chunk = [q.popleft() for _ in range(count)]
    batch = collate_ltx2_bucket_batch(chunk, bucket, **self.collate_kwargs)
    if self.mesh is not None:
      batch = shard_ltx2_batch_for_mesh(batch, self.mesh, input_is_process_local=self.input_is_process_local)
    return batch

  def __next__(self) -> Dict[str, Any]:
    for bucket, q in self._queues.items():
      if len(q) >= self.batch_size:
        return self._emit(bucket, self.batch_size)

    while not self._exhausted:
      try:
        raw = next(self._iter)
      except StopIteration:
        self._exhausted = True
        break

      sample = _coerce_to_sample(raw)
      s_vid, s_aud, s_txt = _infer_sample_token_counts(
          sample,
          patch_size=self.collate_kwargs.get("patch_size", 1),
          patch_size_t=self.collate_kwargs.get("patch_size_t", 1),
          audio_patch_size=self.collate_kwargs.get("audio_patch_size", None),
          audio_patch_size_t=self.collate_kwargs.get("audio_patch_size_t", None),
      )
      bucket = select_bucket(s_vid, s_aud, s_txt, self.buckets, policy=self.overflow_policy, align_kwargs=self.align_kwargs)
      if bucket is None:
        continue
      if self.mesh is not None:
        bucket.validate_for_mesh(context_parallelism=self.mesh.shape.get("context", 1))
      q = self._queues.setdefault(bucket, deque())
      q.append(sample)
      if len(q) >= self.batch_size:
        return self._emit(bucket, self.batch_size)

    if not self.drop_remainder:
      for bucket, q in self._queues.items():
        if len(q) > 0:
          return self._emit(bucket, len(q))

    raise StopIteration


@dataclass(frozen=True)
class LTX2SyntheticShapeProfile:
  """Latent `(F_lat, H_lat, W_lat, T_aud, L_txt)` shape profile mapped into a bucket."""

  latent_frames: int
  latent_height: int
  latent_width: int
  audio_frames: int
  text_length: int
  fps: float = 24.0


class LTX2SyntheticBucketedIterator:
  """Zero-download deterministic multi-bucket iterator for CI, HLO sharding & convergence gates."""

  def __init__(
      self,
      buckets: Sequence[LTX2BucketSpec],
      global_batch_size: int,
      *,
      in_channels: int = 128,
      audio_in_channels: int = 128,
      caption_channels: int = 3840,
      audio_caption_channels: Optional[int] = None,
      patch_size: int = 1,
      patch_size_t: int = 1,
      vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
      audio_scale_factor: int = 4,
      text_padding_mode: str = "post_connector",
      connector_num_learnable_registers: Optional[int] = None,
      video_only_prob: float = 0.0,
      num_unique_batches_per_bucket: Optional[int] = None,
      num_steps: Optional[int] = None,
      seed: int = 0,
      dtype: np.dtype = np.float32,
      mesh: Optional[Mesh] = None,
      bucket_profiles: Optional[Mapping[LTX2BucketSpec, Sequence[LTX2SyntheticShapeProfile]]] = None,
  ):
    if not buckets:
      raise ValueError("buckets must be non-empty.")
    if global_batch_size <= 0:
      raise ValueError(f"global_batch_size must be positive, got {global_batch_size}.")
    if not 0.0 <= video_only_prob <= 1.0:
      raise ValueError(f"video_only_prob must be in [0, 1], got {video_only_prob}.")

    _validate_connector_registers_for_text_mode(text_padding_mode, connector_num_learnable_registers, None)

    self.buckets = tuple(buckets)
    self.global_batch_size = global_batch_size
    self.in_channels = in_channels
    self.audio_in_channels = audio_in_channels
    self.caption_channels = caption_channels
    self.audio_caption_channels = audio_caption_channels or caption_channels
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.vae_scale_factors = vae_scale_factors
    self.audio_scale_factor = audio_scale_factor
    self.text_padding_mode = text_padding_mode
    self.video_only_prob = video_only_prob
    self.num_unique_batches_per_bucket = num_unique_batches_per_bucket
    self.num_steps = num_steps
    self.seed = seed
    self.dtype = np.dtype(dtype)
    self.mesh = mesh
    self.bucket_profiles = dict(bucket_profiles or {})

    if self.mesh is not None:
      cp = self.mesh.shape.get("context", 1)
      for b in self.buckets:
        b.validate_for_mesh(context_parallelism=cp)

    self._step = 0

  def __iter__(self):
    self._step = 0
    return self

  def _default_profiles_for_bucket(self, bucket: LTX2BucketSpec) -> List[LTX2SyntheticShapeProfile]:
    if bucket in self.bucket_profiles and self.bucket_profiles[bucket]:
      return list(self.bucket_profiles[bucket])

    # Build two heterogeneous (F, H, W) profiles <= bucket.b_vid so padding & heterogeneous coords are exercised.
    max_patches = bucket.b_vid
    # Profile A: full or ~75% capacity landscape; Profile B: ~60% capacity portrait.
    h_a = max(1, int(math.isqrt(max(1, max_patches // 2))))
    w_a = max(1, max_patches // (2 * h_a))
    while 2 * h_a * w_a > max_patches and w_a > 1:
      w_a -= 1
    f_a = max(1, min(4, max_patches // max(1, h_a * w_a)))

    w_b = max(1, h_a - 1)
    h_b = max(1, min(max_patches // max(1, w_b), h_a + 1))
    f_b = max(1, min(3, max_patches // max(1, h_b * w_b)))

    aud_a = max(1, int(bucket.b_aud * 0.75))
    aud_b = max(1, int(bucket.b_aud * 0.5))
    txt_a = bucket.b_txt if self.text_padding_mode == "post_connector" else max(1, int(bucket.b_txt * 0.75))
    txt_b = bucket.b_txt if self.text_padding_mode == "post_connector" else max(1, int(bucket.b_txt * 0.5))

    return [
        LTX2SyntheticShapeProfile(
            latent_frames=f_a * self.patch_size_t,
            latent_height=h_a * self.patch_size,
            latent_width=w_a * self.patch_size,
            audio_frames=aud_a,
            text_length=txt_a,
            fps=24.0,
        ),
        LTX2SyntheticShapeProfile(
            latent_frames=f_b * self.patch_size_t,
            latent_height=h_b * self.patch_size,
            latent_width=w_b * self.patch_size,
            audio_frames=aud_b,
            text_length=txt_b,
            fps=16.0,
        ),
    ]

  def __next__(self) -> Dict[str, Any]:
    if self.num_steps is not None and self._step >= self.num_steps:
      raise StopIteration

    step_idx = self._step
    self._step += 1

    bucket_idx = step_idx % len(self.buckets)
    bucket = self.buckets[bucket_idx]
    occurrence = step_idx // len(self.buckets)
    if self.num_unique_batches_per_bucket is not None and self.num_unique_batches_per_bucket > 0:
      occurrence = occurrence % self.num_unique_batches_per_bucket

    rng = np.random.default_rng(self.seed + bucket_idx * 100_003 + occurrence * 1_009)
    profiles = self._default_profiles_for_bucket(bucket)

    samples: List[LTX2Sample] = []
    vid_c = self.in_channels // (self.patch_size_t * self.patch_size * self.patch_size)
    for b_i in range(self.global_batch_size):
      prof = profiles[b_i % len(profiles)]
      vid_latents = rng.standard_normal((vid_c, prof.latent_frames, prof.latent_height, prof.latent_width)).astype(
          self.dtype
      )

      has_audio = bool(rng.random() >= self.video_only_prob)
      aud_latents = (
          rng.standard_normal((prof.audio_frames, self.audio_in_channels)).astype(self.dtype) if has_audio else None
      )

      txt_len = bucket.b_txt if self.text_padding_mode == "post_connector" else prof.text_length
      enc_states = rng.standard_normal((txt_len, self.caption_channels)).astype(self.dtype)
      aenc_states = rng.standard_normal((txt_len, self.audio_caption_channels)).astype(self.dtype)

      samples.append(
          LTX2Sample(
              video_latents=vid_latents,
              audio_latents=aud_latents,
              encoder_hidden_states=enc_states,
              audio_encoder_hidden_states=aenc_states,
              fps=prof.fps,
              has_audio=has_audio,
              is_normalized=True,
          )
      )

    batch = collate_ltx2_bucket_batch(
        samples,
        bucket,
        patch_size=self.patch_size,
        patch_size_t=self.patch_size_t,
        vae_scale_factors=self.vae_scale_factors,
        audio_in_channels=self.audio_in_channels,
        audio_scale_factor=self.audio_scale_factor,
        text_padding_mode=self.text_padding_mode,
        dtype=self.dtype,
    )
    if self.mesh is not None:
      batch = shard_ltx2_batch_for_mesh(batch, self.mesh, input_is_process_local=False)
    return batch
