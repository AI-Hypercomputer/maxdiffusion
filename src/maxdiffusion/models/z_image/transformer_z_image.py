"""JAX/NNX implementation of the Diffusers Z-Image transformer.

Z-Image and Z-Image-Turbo use the same denoiser.  The model deliberately
keeps the variable-length image/text representation from Diffusers: it avoids
padding an image before patchification and only pads the joint sequence when a
batch contains different resolutions or prompt lengths.
"""

from typing import Optional, Sequence

import math

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.common_types import BlockSizes
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.attention_flax import NNXAttentionOp


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


def _linear(
    rngs,
    in_features,
    out_features,
    *,
    use_bias=True,
    dtype=jnp.float32,
    weights_dtype=jnp.float32,
    kernel_axes=("embed", "heads"),
    bias_axes=("heads",),
):
  """Create a logically sharded NNX dense layer for the distributed denoiser."""
  return nnx.Linear(
      in_features,
      out_features,
      use_bias=use_bias,
      rngs=rngs,
      dtype=dtype,
      param_dtype=weights_dtype,
      kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), kernel_axes),
      bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), bias_axes),
  )


@flax.struct.dataclass
class ZImageTransformer2DModelOutput:
  sample: list[jax.Array]


class ZImageTimestepEmbedder(nnx.Module):
  """Sinusoidal timestep embedding followed by the upstream two-layer MLP."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      out_size: int,
      mid_size: Optional[int] = None,
      frequency_embedding_size: int = 256,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.frequency_embedding_size = frequency_embedding_size
    mid_size = out_size if mid_size is None else mid_size
    self.mlp_in = _linear(
        rngs,
        frequency_embedding_size,
        mid_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=("embed", "mlp"),
        bias_axes=("mlp",),
    )
    self.mlp_out = _linear(
        rngs,
        mid_size,
        out_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=("mlp", "embed"),
        bias_axes=("embed",),
    )

  @staticmethod
  def timestep_embedding(timestep: jax.Array, dim: int, max_period: float = 10000.0) -> jax.Array:
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = timestep[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
    if dim % 2:
      embedding = jnp.concatenate((embedding, jnp.zeros_like(embedding[:, :1])), axis=-1)
    return embedding

  def __call__(self, timestep: jax.Array) -> jax.Array:
    x = self.timestep_embedding(timestep, self.frequency_embedding_size)
    return self.mlp_out(nnx.silu(self.mlp_in(x)))


class ZImageFeedForward(nnx.Module):
  """The bias-free SwiGLU MLP used by every Z-Image transformer block."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      hidden_dim: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.w1 = _linear(
        rngs, dim, hidden_dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype, kernel_axes=("embed", "mlp")
    )
    self.w2 = _linear(
        rngs, hidden_dim, dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype, kernel_axes=("mlp", "embed")
    )
    self.w3 = _linear(
        rngs, dim, hidden_dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype, kernel_axes=("embed", "mlp")
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class ZImageAttention(nnx.Module):
  """Z-Image self-attention backed by MaxDiffusion's shared attention operator."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      heads: int,
      qk_norm: bool = True,
      eps: float = 1e-5,
      attention_kernel: str = "dot_product",
      mesh: Optional[jax.sharding.Mesh] = None,
      flash_block_sizes: Optional[BlockSizes] = None,
      flash_min_seq_length: int = 4096,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    if dim % heads:
      raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
    self.heads = heads
    self.dim_head = dim // heads
    self.qk_norm = qk_norm
    self.attention_kernel = attention_kernel
    self.flash_min_seq_length = flash_min_seq_length
    self.to_q = _linear(rngs, dim, dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype)
    self.to_k = _linear(rngs, dim, dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype)
    self.to_v = _linear(rngs, dim, dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype)
    self.to_out = _linear(
        rngs, dim, dim, use_bias=False, dtype=dtype, weights_dtype=weights_dtype, kernel_axes=("heads", "embed")
    )
    if qk_norm:
      self.norm_q = nnx.RMSNorm(
          self.dim_head, epsilon=eps, use_scale=True, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32
      )
      self.norm_k = nnx.RMSNorm(
          self.dim_head, epsilon=eps, use_scale=True, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32
      )
    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=self.dim_head**-0.5,
        heads=heads,
        dim_head=self.dim_head,
        split_head_dim=True,
        float32_qk_product=False,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
    )

  @staticmethod
  def _apply_rope(x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
    """Apply complex RoPE; x is B,S,H,D and frequencies B,S,D/2."""
    x_float = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    real, imag = x_float[..., 0], x_float[..., 1]
    cos, sin = jnp.real(freqs_cis)[:, :, None], jnp.imag(freqs_cis)[:, :, None]
    out = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1)
    return out.reshape(x.shape).astype(x.dtype)

  def __call__(
      self, hidden_states: jax.Array, freqs_cis: jax.Array, attention_mask: Optional[jax.Array] = None
  ) -> jax.Array:
    batch, length, _ = hidden_states.shape
    q = self.to_q(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    k = self.to_k(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    v = self.to_v(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    if self.qk_norm:
      q = self.norm_q(q).astype(hidden_states.dtype)
      k = self.norm_k(k).astype(hidden_states.dtype)
    q = self._apply_rope(q, freqs_cis)
    k = self._apply_rope(k, freqs_cis)
    # The local dot-product path is deliberately used for CPU parity tests.
    # Production TPU execution uses NNXAttentionOp below (Flash/Splash/etc.).
    if self.attention_kernel == "dot_product" or length < self.flash_min_seq_length:
      scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
      scores = scores * (self.dim_head**-0.5)
      if attention_mask is not None:
        scores = jnp.where(attention_mask[:, None, None, :], scores, -jnp.inf)
      output = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1).astype(v.dtype), v).reshape(batch, length, -1)
    else:
      q, k, v = (item.reshape(batch, length, -1) for item in (q, k, v))
      output = self.attention_op.apply_attention(q, k, v, attention_mask=attention_mask)
    return self.to_out(output)


def _select_per_token(noisy: jax.Array, clean: jax.Array, noise_mask: jax.Array) -> jax.Array:
  return jnp.where(noise_mask[..., None] == 1, noisy[:, None, :], clean[:, None, :])


class ZImageTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      layer_id: int,
      dim: int,
      n_heads: int,
      norm_eps: float,
      qk_norm: bool,
      modulation: bool = True,
      attention_kernel: str = "dot_product",
      mesh: Optional[jax.sharding.Mesh] = None,
      flash_block_sizes: Optional[BlockSizes] = None,
      flash_min_seq_length: int = 4096,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.layer_id = layer_id
    self.modulation = modulation
    self.attention = ZImageAttention(
        rngs,
        dim,
        n_heads,
        qk_norm,
        attention_kernel=attention_kernel,
        mesh=mesh,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    self.feed_forward = ZImageFeedForward(rngs, dim, int(dim / 3 * 8), dtype=dtype, weights_dtype=weights_dtype)
    norm_args = dict(epsilon=norm_eps, use_scale=True, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32)
    self.attention_norm1 = nnx.RMSNorm(dim, **norm_args)
    self.ffn_norm1 = nnx.RMSNorm(dim, **norm_args)
    self.attention_norm2 = nnx.RMSNorm(dim, **norm_args)
    self.ffn_norm2 = nnx.RMSNorm(dim, **norm_args)
    if modulation:
      self.adaln_modulation = _linear(
          rngs,
          min(dim, ADALN_EMBED_DIM),
          4 * dim,
          dtype=dtype,
          weights_dtype=weights_dtype,
          kernel_axes=("embed", "mlp"),
          bias_axes=("mlp",),
      )

  def __call__(
      self,
      x: jax.Array,
      freqs_cis: jax.Array,
      attention_mask: Optional[jax.Array] = None,
      adaln_input: Optional[jax.Array] = None,
      noise_mask: Optional[jax.Array] = None,
      adaln_noisy: Optional[jax.Array] = None,
      adaln_clean: Optional[jax.Array] = None,
  ) -> jax.Array:
    if not self.modulation:
      attn_out = self.attention(self.attention_norm1(x), freqs_cis, attention_mask)
      x = x + self.attention_norm2(attn_out)
      return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    if noise_mask is None:
      modulation = self.adaln_modulation(adaln_input)
      scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(modulation[:, None], 4, axis=-1)
      scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
      gate_msa, gate_mlp = jnp.tanh(gate_msa), jnp.tanh(gate_mlp)
    else:
      noisy = self.adaln_modulation(adaln_noisy)
      clean = self.adaln_modulation(adaln_clean)
      noisy_parts = jnp.split(noisy, 4, axis=-1)
      clean_parts = jnp.split(clean, 4, axis=-1)
      scale_msa = 1.0 + _select_per_token(noisy_parts[0], clean_parts[0], noise_mask)
      gate_msa = _select_per_token(jnp.tanh(noisy_parts[1]), jnp.tanh(clean_parts[1]), noise_mask)
      scale_mlp = 1.0 + _select_per_token(noisy_parts[2], clean_parts[2], noise_mask)
      gate_mlp = _select_per_token(jnp.tanh(noisy_parts[3]), jnp.tanh(clean_parts[3]), noise_mask)

    attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis, attention_mask)
    x = x + gate_msa * self.attention_norm2(attn_out)
    return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))


class ZImageFinalLayer(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      out_channels: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    self.norm_final = nnx.LayerNorm(
        hidden_size,
        epsilon=1e-6,
        use_bias=False,
        use_scale=False,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    self.linear = _linear(
        rngs,
        hidden_size,
        out_channels,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=("embed", "out_channels"),
        bias_axes=("out_channels",),
    )
    self.adaln_modulation = _linear(
        rngs,
        min(hidden_size, ADALN_EMBED_DIM),
        hidden_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=("embed", "mlp"),
        bias_axes=("mlp",),
    )

  def __call__(
      self,
      x: jax.Array,
      c: Optional[jax.Array] = None,
      noise_mask: Optional[jax.Array] = None,
      c_noisy: Optional[jax.Array] = None,
      c_clean: Optional[jax.Array] = None,
  ) -> jax.Array:
    if noise_mask is None:
      scale = 1.0 + self.adaln_modulation(nnx.silu(c))[:, None]
    else:
      scale = 1.0 + _select_per_token(
          self.adaln_modulation(nnx.silu(c_noisy)), self.adaln_modulation(nnx.silu(c_clean)), noise_mask
      )
    return self.linear(self.norm_final(x) * scale)


class ZImageRopeEmbedder:
  """CPU-precomputed multi-axis RoPE frequencies, identical to Diffusers' RopeEmbedder."""

  def __init__(
      self, theta: float = 256.0, axes_dims: Sequence[int] = (32, 48, 48), axes_lens: Sequence[int] = (1536, 512, 512)
  ):
    if len(axes_dims) != len(axes_lens):
      raise ValueError("axes_dims and axes_lens must have the same length.")
    self.axes_dims = tuple(axes_dims)
    self.axes_lens = tuple(axes_lens)
    freqs = []
    for dim, length in zip(self.axes_dims, self.axes_lens):
      angular = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
      phase = np.arange(length, dtype=np.float32)[:, None] * angular[None]
      freqs.append(np.exp(1j * phase).astype(np.complex64))
    self.freqs_cis = tuple(freqs)

  def __call__(self, ids: jax.Array) -> jax.Array:
    if ids.ndim != 3 or ids.shape[-1] != len(self.axes_dims):
      raise ValueError("ids must have shape (batch, sequence, number_of_axes).")
    return jnp.concatenate([jnp.asarray(freq)[ids[..., axis]] for axis, freq in enumerate(self.freqs_cis)], axis=-1)


class ZImageTransformer2DModel(nnx.Module, ConfigMixin):
  """Z-Image / Z-Image-Turbo denoiser.

  The public call signature mirrors Diffusers for the base text-to-image model.
  `x` and `cap_feats` are lists so prompts and images can have individual
  lengths/resolutions; this is important for the upstream checkpoint format.
  """

  config_name = "config.json"

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      all_patch_size: Sequence[int] = (2,),
      all_f_patch_size: Sequence[int] = (1,),
      in_channels: int = 16,
      dim: int = 3840,
      n_layers: int = 30,
      n_refiner_layers: int = 2,
      n_heads: int = 30,
      n_kv_heads: int = 30,
      norm_eps: float = 1e-5,
      qk_norm: bool = True,
      cap_feat_dim: int = 2560,
      rope_theta: float = 256.0,
      t_scale: float = 1000.0,
      axes_dims: Sequence[int] = (32, 48, 48),
      axes_lens: Sequence[int] = (1536, 512, 512),
      attention_kernel: str = "dot_product",
      mesh: Optional[jax.sharding.Mesh] = None,
      flash_block_sizes: Optional[BlockSizes] = None,
      flash_min_seq_length: int = 4096,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      **unused_kwargs,
  ):
    del n_kv_heads, unused_kwargs
    if tuple(all_patch_size) != (2,) or tuple(all_f_patch_size) != (1,):
      raise NotImplementedError("MaxDiffusion currently supports Z-Image's released 2x2x1 patching only.")
    if dim // n_heads != sum(axes_dims):
      raise ValueError("Z-Image head_dim must equal sum(axes_dims).")
    self.in_channels = in_channels
    self.out_channels = in_channels
    self.dim = dim
    self.n_heads = n_heads
    self.t_scale = t_scale
    self.patch_size, self.f_patch_size = 2, 1
    patch_dim = self.patch_size * self.patch_size * self.f_patch_size * in_channels
    self.x_embedder = _linear(rngs, patch_dim, dim, dtype=dtype, weights_dtype=weights_dtype)
    self.final_layer = ZImageFinalLayer(rngs, dim, patch_dim, dtype=dtype, weights_dtype=weights_dtype)
    self.t_embedder = ZImageTimestepEmbedder(
        rngs, min(dim, ADALN_EMBED_DIM), mid_size=1024, dtype=dtype, weights_dtype=weights_dtype
    )
    self.cap_embedder_norm = nnx.RMSNorm(
        cap_feat_dim, epsilon=norm_eps, use_scale=True, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32
    )
    self.cap_embedder = _linear(rngs, cap_feat_dim, dim, dtype=dtype, weights_dtype=weights_dtype)
    self.x_pad_token = nnx.Param(jnp.zeros((1, dim), dtype=weights_dtype))
    self.cap_pad_token = nnx.Param(jnp.zeros((1, dim), dtype=weights_dtype))

    block_args = dict(
        dim=dim,
        n_heads=n_heads,
        norm_eps=norm_eps,
        qk_norm=qk_norm,
        attention_kernel=attention_kernel,
        mesh=mesh,
        flash_block_sizes=flash_block_sizes,
        flash_min_seq_length=flash_min_seq_length,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    self.noise_refiner = nnx.List(
        [ZImageTransformerBlock(rngs, 1000 + index, modulation=True, **block_args) for index in range(n_refiner_layers)]
    )
    self.context_refiner = nnx.List(
        [ZImageTransformerBlock(rngs, index, modulation=False, **block_args) for index in range(n_refiner_layers)]
    )
    self.layers = nnx.List([ZImageTransformerBlock(rngs, index, modulation=True, **block_args) for index in range(n_layers)])
    self.rope_embedder = ZImageRopeEmbedder(rope_theta, axes_dims, axes_lens)

  @staticmethod
  def _patchify(image: jax.Array) -> tuple[jax.Array, tuple[int, int, int], tuple[int, int, int]]:
    channels, frames, height, width = image.shape
    if frames % 1 or height % 2 or width % 2:
      raise ValueError("Z-Image latents must be divisible by the released 1x2x2 patch size.")
    ft, ht, wt = frames, height // 2, width // 2
    patches = image.reshape(channels, ft, 1, ht, 2, wt, 2).transpose(1, 3, 5, 2, 4, 6, 0)
    return patches.reshape(ft * ht * wt, -1), (frames, height, width), (ft, ht, wt)

  @staticmethod
  def _coordinate_grid(size: tuple[int, int, int], start: tuple[int, int, int]) -> jax.Array:
    return jnp.stack(
        jnp.meshgrid(*[jnp.arange(s, s + n, dtype=jnp.int32) for s, n in zip(start, size)], indexing="ij"), axis=-1
    ).reshape(-1, 3)

  def _pad(
      self, feature: jax.Array, grid_size: tuple[int, int, int], start: tuple[int, int, int]
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    original = feature.shape[0]
    pad = (-original) % SEQ_MULTI_OF
    positions = self._coordinate_grid(grid_size, start)
    if pad:
      feature = jnp.concatenate((feature, jnp.repeat(feature[-1:], pad, axis=0)))
      positions = jnp.concatenate((positions, jnp.repeat(jnp.zeros((1, 3), dtype=jnp.int32), pad, axis=0)))
    return feature, positions, jnp.arange(original + pad) >= original

  def _prepare(
      self, features: list[jax.Array], positions: list[jax.Array], pad_masks: list[jax.Array], pad_token: jax.Array
  ):
    lengths = [feature.shape[0] for feature in features]
    max_length = max(lengths)
    padded_features, padded_freqs, attention_mask = [], [], []
    for feature, position, pad_mask in zip(features, positions, pad_masks):
      feature = jnp.where(pad_mask[:, None], pad_token, feature)
      frequency = self.rope_embedder(position[None])[0]
      missing = max_length - feature.shape[0]
      padded_features.append(jnp.pad(feature, ((0, missing), (0, 0))))
      padded_freqs.append(jnp.pad(frequency, ((0, missing), (0, 0))))
      attention_mask.append(jnp.arange(max_length) < feature.shape[0])
    mask = jnp.stack(attention_mask)
    return (
        jnp.stack(padded_features),
        jnp.stack(padded_freqs),
        None if all(length == max_length for length in lengths) else mask,
        lengths,
    )

  def __call__(
      self,
      x: list[jax.Array],
      t: jax.Array,
      cap_feats: list[jax.Array],
      return_dict: bool = True,
      patch_size: int = 2,
      f_patch_size: int = 1,
      **unused_kwargs,
  ):
    del unused_kwargs
    if patch_size != 2 or f_patch_size != 1:
      raise NotImplementedError("Only Z-Image's released 2x2x1 patching is supported.")
    if len(x) != len(cap_feats):
      raise ValueError("x and cap_feats must have one item per batch element.")
    adaln = self.t_embedder(t * self.t_scale).astype(x[0].dtype)
    image_features, image_positions, image_masks, sizes = [], [], [], []
    caption_features, caption_positions, caption_masks = [], [], []
    for image, caption in zip(x, cap_feats):
      caption, cap_position, cap_mask = self._pad(caption, (caption.shape[0], 1, 1), (1, 0, 0))
      patch, size, tokens = self._patchify(image)
      patch, image_position, image_mask = self._pad(patch, tokens, (caption.shape[0] + 1, 0, 0))
      image_features.append(patch)
      image_positions.append(image_position)
      image_masks.append(image_mask)
      caption_features.append(caption)
      caption_positions.append(cap_position)
      caption_masks.append(cap_mask)
      sizes.append(size)

    image_lengths = [item.shape[0] for item in image_features]
    image_embeddings = self.x_embedder(jnp.concatenate(image_features, axis=0))
    image_embeddings = list(jnp.split(image_embeddings, jnp.cumsum(jnp.asarray(image_lengths[:-1]))))
    image_embeddings, image_freqs, image_attention_mask, image_lengths = self._prepare(
        image_embeddings, image_positions, image_masks, self.x_pad_token[...]
    )
    for block in self.noise_refiner:
      image_embeddings = block(image_embeddings, image_freqs, image_attention_mask, adaln)

    caption_lengths = [item.shape[0] for item in caption_features]
    captions = self.cap_embedder(self.cap_embedder_norm(jnp.concatenate(caption_features, axis=0)))
    captions = list(jnp.split(captions, jnp.cumsum(jnp.asarray(caption_lengths[:-1]))))
    captions, caption_freqs, caption_attention_mask, caption_lengths = self._prepare(
        captions, caption_positions, caption_masks, self.cap_pad_token[...]
    )
    for block in self.context_refiner:
      captions = block(captions, caption_freqs, caption_attention_mask)

    joint_lengths = [image_length + caption_length for image_length, caption_length in zip(image_lengths, caption_lengths)]
    joint_max = max(joint_lengths)
    joint, joint_freqs, joint_mask = [], [], []
    for index, length in enumerate(joint_lengths):
      value = jnp.concatenate((image_embeddings[index, : image_lengths[index]], captions[index, : caption_lengths[index]]))
      freqs = jnp.concatenate((image_freqs[index, : image_lengths[index]], caption_freqs[index, : caption_lengths[index]]))
      joint.append(jnp.pad(value, ((0, joint_max - length), (0, 0))))
      joint_freqs.append(jnp.pad(freqs, ((0, joint_max - length), (0, 0))))
      joint_mask.append(jnp.arange(joint_max) < length)
    hidden_states, freqs_cis = jnp.stack(joint), jnp.stack(joint_freqs)
    joint_attention_mask = None if all(length == joint_max for length in joint_lengths) else jnp.stack(joint_mask)
    for block in self.layers:
      hidden_states = block(hidden_states, freqs_cis, joint_attention_mask, adaln)
    hidden_states = self.final_layer(hidden_states, c=adaln)

    outputs = []
    for index, (frames, height, width) in enumerate(sizes):
      tokens = frames * (height // 2) * (width // 2)
      patches = hidden_states[index, :tokens].reshape(frames, height // 2, width // 2, 1, 2, 2, self.out_channels)
      outputs.append(patches.transpose(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, frames, height, width))
    return ZImageTransformer2DModelOutput(sample=outputs) if return_dict else (outputs,)
