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
"""JAX/NNX implementation of the Z-Image transformer."""

from typing import Optional, Sequence

import math

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.common_types import (
    BATCH,
    BlockSizes,
    D_KV,
    SELF_ATTN_HEAD,
    SELF_ATTN_KV_LENGTH,
    SELF_ATTN_Q_LENGTH,
)
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.attention_flax import NNXAttentionOp
from .logical_sharding_z_image import get_sharding_specs, ZImageDiTShardingSpecs

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
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
  ):
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
    self.frequency_embedding_size = frequency_embedding_size
    mid_size = out_size if mid_size is None else mid_size
    self.mlp_in = _linear(
        rngs,
        frequency_embedding_size,
        mid_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.emb_linear_1_kernel,
        bias_axes=self.sharding_specs.emb_linear_1_bias,
    )
    self.mlp_out = _linear(
        rngs,
        mid_size,
        out_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.emb_linear_2_kernel,
        bias_axes=self.sharding_specs.emb_linear_2_bias,
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
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
  ):
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
    self.w1 = _linear(
        rngs,
        dim,
        hidden_dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.net_0_kernel,
        bias_axes=self.sharding_specs.net_0_bias,
    )
    self.w2 = _linear(
        rngs,
        hidden_dim,
        dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.net_2_kernel,
        bias_axes=self.sharding_specs.net_2_bias,
    )
    self.w3 = _linear(
        rngs,
        dim,
        hidden_dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.net_0_kernel,
        bias_axes=self.sharding_specs.net_0_bias,
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
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
      attention_config: Optional[dict] = None,
  ):
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        "ulysses_attention_chunks": 1,
        **(attention_config or {}),
    }
    if dim % heads:
      raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
    self.heads = heads
    self.dim_head = dim // heads
    self.qk_norm = qk_norm
    self.dtype = dtype
    self.attention_kernel = attention_kernel
    self.flash_min_seq_length = flash_min_seq_length
    self.to_q = _linear(
        rngs,
        dim,
        dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.qkv_kernel,
        bias_axes=self.sharding_specs.qkv_bias,
    )
    self.to_k = _linear(
        rngs,
        dim,
        dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.qkv_kernel,
        bias_axes=self.sharding_specs.qkv_bias,
    )
    self.to_v = _linear(
        rngs,
        dim,
        dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.qkv_kernel,
        bias_axes=self.sharding_specs.qkv_bias,
    )
    self.to_out = _linear(
        rngs,
        dim,
        dim,
        use_bias=False,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.out_kernel,
        bias_axes=self.sharding_specs.out_bias,
    )
    if qk_norm:
      norm_scale_init = nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale)
      self.norm_q = nnx.RMSNorm(
          self.dim_head,
          epsilon=eps,
          use_scale=True,
          rngs=rngs,
          dtype=jnp.float32,
          param_dtype=jnp.float32,
          scale_init=norm_scale_init,
      )
      self.norm_k = nnx.RMSNorm(
          self.dim_head,
          epsilon=eps,
          use_scale=True,
          rngs=rngs,
          dtype=jnp.float32,
          param_dtype=jnp.float32,
          scale_init=norm_scale_init,
      )
    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=self.dim_head**-0.5,
        heads=heads,
        dim_head=self.dim_head,
        split_head_dim=True,
        float32_qk_product=False,
        # Z-Image attends over one joint image+caption sequence, so every
        # block is self-attention. Naming the axes as such is what lets the
        # sequence-parallel / ring rules shard queries while keeping keys and
        # values whole; the generic names would shard both and each device
        # would attend only to its own slice.
        axis_names_q=(BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV),
        axis_names_kv=(BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV),
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        # Only the tokamax kernels read these; plain "flash" ignores them.
        use_base2_exp=attention_config["use_base2_exp"],
        use_experimental_scheduler=attention_config["use_experimental_scheduler"],
        ulysses_shards=attention_config["ulysses_shards"],
        ulysses_attention_chunks=attention_config["ulysses_attention_chunks"],
    )

  @staticmethod
  def _apply_rope(
      x: jax.Array,
      freqs: tuple[jax.Array, jax.Array],
      out_dtype: jnp.dtype,
  ) -> jax.Array:
    """Apply real-valued RoPE; x is B,S,H,D and freqs is a (cos, sin) tuple of real arrays.

    Rotates in float32 and returns `out_dtype`, so q and k reach the
    attention kernel at the same dtype as v.
    """
    cos, sin = freqs

    if cos.ndim == 3:
      cos = cos[:, :, None, :]
    if sin.ndim == 3:
      sin = sin[:, :, None, :]

    x_float = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    real, imag = x_float[..., 0], x_float[..., 1]
    out = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1)
    return out.reshape(x.shape).astype(out_dtype)

  def __call__(
      self,
      hidden_states: jax.Array,
      freqs: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    batch, length, _ = hidden_states.shape
    q = self.to_q(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    k = self.to_k(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    v = self.to_v(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    if self.qk_norm:
      q = self.norm_q(q)
      k = self.norm_k(k)
    q = self._apply_rope(q, freqs, self.dtype)
    k = self._apply_rope(k, freqs, self.dtype)
    if self.attention_kernel == "dot_product" or length < self.flash_min_seq_length:
      scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
      scores = scores * (self.dim_head**-0.5)
      if attention_mask is not None:
        scores = jnp.where(attention_mask[:, None, None, :], scores, -jnp.inf)
      output = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1).astype(v.dtype), v).reshape(batch, length, -1)
    else:
      # NNXAttentionOp expects shape (batch, heads, sequence, dim_head) for 4D inputs.
      # Since our local q, k, v are (batch, sequence, heads, dim_head), transposing them
      # to (batch, heads, sequence, dim_head) bypasses any 3D flattening/unflattening.
      q_4d = jnp.transpose(q, (0, 2, 1, 3))
      k_4d = jnp.transpose(k, (0, 2, 1, 3))
      v_4d = jnp.transpose(v, (0, 2, 1, 3))
      output = self.attention_op.apply_attention(q_4d, k_4d, v_4d, attention_mask=attention_mask)
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
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
      attention_config: Optional[dict] = None,
  ):
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
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
        sharding_specs=self.sharding_specs,
        attention_config=attention_config,
    )
    self.feed_forward = ZImageFeedForward(
        rngs,
        dim,
        int(dim / 3 * 8),
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    norm_scale_init = nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale)
    norm_args = {
        "epsilon": norm_eps,
        "use_scale": True,
        "rngs": rngs,
        "dtype": jnp.float32,
        "param_dtype": jnp.float32,
        "scale_init": norm_scale_init,
    }
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
          kernel_axes=self.sharding_specs.adaln_kernel,
          bias_axes=self.sharding_specs.adaln_bias,
      )

  def __call__(
      self,
      x: jax.Array,
      freqs: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array] = None,
      adaln_input: Optional[jax.Array] = None,
      noise_mask: Optional[jax.Array] = None,
      adaln_noisy: Optional[jax.Array] = None,
      adaln_clean: Optional[jax.Array] = None,
  ) -> jax.Array:
    out_dtype = x.dtype
    if not self.modulation:
      attn_out = self.attention(self.attention_norm1(x), freqs, attention_mask)
      x = x + self.attention_norm2(attn_out)
      out = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
      return out.astype(out_dtype)

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

    attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs, attention_mask)
    x = x + gate_msa * self.attention_norm2(attn_out)
    out = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    return out.astype(out_dtype)


class ZImageFinalLayer(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      out_channels: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
  ):
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
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
        kernel_axes=self.sharding_specs.out_embed_kernel,
        bias_axes=self.sharding_specs.out_embed_bias,
    )
    self.adaln_modulation = _linear(
        rngs,
        min(hidden_size, ADALN_EMBED_DIM),
        hidden_size,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.adaln_kernel,
        bias_axes=self.sharding_specs.adaln_bias,
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
          self.adaln_modulation(nnx.silu(c_noisy)),
          self.adaln_modulation(nnx.silu(c_clean)),
          noise_mask,
      )
    return self.linear(self.norm_final(x) * scale)


class ZImageRopeEmbedder:
  """CPU-precomputed multi-axis RoPE frequencies returning real-valued cos and sin arrays directly."""

  def __init__(
      self,
      theta: float = 256.0,
      axes_dims: Sequence[int] = (32, 48, 48),
      axes_lens: Sequence[int] = (1536, 512, 512),
  ):
    if len(axes_dims) != len(axes_lens):
      raise ValueError("axes_dims and axes_lens must have the same length.")
    self.axes_dims = tuple(axes_dims)
    self.axes_lens = tuple(axes_lens)

    cos_list, sin_list = [], []
    for dim, length in zip(self.axes_dims, self.axes_lens):
      angular = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
      phase = np.arange(length, dtype=np.float32)[:, None] * angular[None]
      cos_list.append(np.cos(phase).astype(np.float32))
      sin_list.append(np.sin(phase).astype(np.float32))

    self.cos = tuple(cos_list)
    self.sin = tuple(sin_list)

  def __call__(self, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    if ids.ndim != 3 or ids.shape[-1] != len(self.axes_dims):
      raise ValueError("ids must have shape (batch, sequence, number_of_axes).")
    cos = jnp.concatenate([jnp.asarray(c)[ids[..., axis]] for axis, c in enumerate(self.cos)], axis=-1)
    sin = jnp.concatenate([jnp.asarray(s)[ids[..., axis]] for axis, s in enumerate(self.sin)], axis=-1)
    return cos, sin


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
      sharding_specs: Optional[ZImageDiTShardingSpecs] = None,
      scan_layers: bool = True,
      attention_config: Optional[dict] = None,
      **unused_kwargs,
  ):
    del n_kv_heads, unused_kwargs
    if tuple(all_patch_size) != (2,) or tuple(all_f_patch_size) != (1,):
      raise NotImplementedError("MaxDiffusion currently supports Z-Image's released 2x2x1 patching only.")
    if dim // n_heads != sum(axes_dims):
      raise ValueError("Z-Image head_dim must equal sum(axes_dims).")
    if sharding_specs is None:
      sharding_specs = get_sharding_specs("default", "z_image_dit")
    self.sharding_specs = sharding_specs
    self.scan_layers = scan_layers
    self.n_layers = n_layers
    self.n_refiner_layers = n_refiner_layers

    self.in_channels = in_channels
    self.out_channels = in_channels
    self.dim = dim
    self.n_heads = n_heads
    self.t_scale = t_scale
    self.patch_size, self.f_patch_size = 2, 1
    patch_dim = self.patch_size * self.patch_size * self.f_patch_size * in_channels
    self.x_embedder = _linear(
        rngs,
        patch_dim,
        dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.embed_kernel,
        bias_axes=self.sharding_specs.embed_bias,
    )
    self.final_layer = ZImageFinalLayer(
        rngs,
        dim,
        patch_dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    self.t_embedder = ZImageTimestepEmbedder(
        rngs,
        min(dim, ADALN_EMBED_DIM),
        mid_size=1024,
        dtype=dtype,
        weights_dtype=weights_dtype,
        sharding_specs=self.sharding_specs,
    )
    norm_scale_init = nnx.with_partitioning(nnx.initializers.ones_init(), self.sharding_specs.norm_scale)
    self.cap_embedder_norm = nnx.RMSNorm(
        cap_feat_dim,
        epsilon=norm_eps,
        use_scale=True,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=norm_scale_init,
    )
    self.cap_embedder = _linear(
        rngs,
        cap_feat_dim,
        dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        kernel_axes=self.sharding_specs.embed_kernel,
        bias_axes=self.sharding_specs.embed_bias,
    )
    self.x_pad_token = nnx.Param(jnp.zeros((1, dim), dtype=weights_dtype))
    self.cap_pad_token = nnx.Param(jnp.zeros((1, dim), dtype=weights_dtype))

    block_args = {
        "dim": dim,
        "n_heads": n_heads,
        "norm_eps": norm_eps,
        "qk_norm": qk_norm,
        "attention_kernel": attention_kernel,
        "mesh": mesh,
        "flash_block_sizes": flash_block_sizes,
        "flash_min_seq_length": flash_min_seq_length,
        "dtype": dtype,
        "weights_dtype": weights_dtype,
        "sharding_specs": self.sharding_specs,
        "attention_config": attention_config,
    }
    if self.scan_layers:

      @nnx.split_rngs(splits=n_refiner_layers)
      @nnx.vmap(
          in_axes=0,
          out_axes=0,
          axis_size=n_refiner_layers,
          transform_metadata={nnx.PARTITION_NAME: "layers"},
      )
      def init_noise_refiner(rngs):
        return ZImageTransformerBlock(rngs, 1000, modulation=True, **block_args)

      @nnx.split_rngs(splits=n_refiner_layers)
      @nnx.vmap(
          in_axes=0,
          out_axes=0,
          axis_size=n_refiner_layers,
          transform_metadata={nnx.PARTITION_NAME: "layers"},
      )
      def init_context_refiner(rngs):
        return ZImageTransformerBlock(rngs, 0, modulation=False, **block_args)

      @nnx.split_rngs(splits=n_layers)
      @nnx.vmap(
          in_axes=0,
          out_axes=0,
          axis_size=n_layers,
          transform_metadata={nnx.PARTITION_NAME: "layers"},
      )
      def init_layers(rngs):
        return ZImageTransformerBlock(rngs, 0, modulation=True, **block_args)

      self.noise_refiner = init_noise_refiner(rngs)
      self.context_refiner = init_context_refiner(rngs)
      self.layers = init_layers(rngs)
    else:
      self.noise_refiner = nnx.List(
          [ZImageTransformerBlock(rngs, 1000 + index, modulation=True, **block_args) for index in range(n_refiner_layers)]
      )
      self.context_refiner = nnx.List(
          [ZImageTransformerBlock(rngs, index, modulation=False, **block_args) for index in range(n_refiner_layers)]
      )
      self.layers = nnx.List(
          [ZImageTransformerBlock(rngs, index, modulation=True, **block_args) for index in range(n_layers)]
      )
    self.rope_embedder = ZImageRopeEmbedder(rope_theta, axes_dims, axes_lens)

  @staticmethod
  def _patchify(
      image: jax.Array,
  ) -> tuple[jax.Array, tuple[int, int, int], tuple[int, int, int]]:
    channels, frames, height, width = image.shape
    if frames % 1 or height % 2 or width % 2:
      raise ValueError("Z-Image latents must be divisible by the released 1x2x2 patch size.")
    ft, ht, wt = frames, height // 2, width // 2
    patches = image.reshape(channels, ft, 1, ht, 2, wt, 2).transpose(1, 3, 5, 2, 4, 6, 0)
    return patches.reshape(ft * ht * wt, -1), (frames, height, width), (ft, ht, wt)

  @staticmethod
  def _coordinate_grid(size: tuple[int, int, int], start: tuple[int, int, int]) -> jax.Array:
    grid = np.stack(
        np.meshgrid(
            *[np.arange(s, s + n, dtype=np.int32) for s, n in zip(start, size)],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return jnp.asarray(grid)

  def _pad(
      self,
      feature: jax.Array,
      grid_size: tuple[int, int, int],
      start: tuple[int, int, int],
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    for axis, (s, n, max_len) in enumerate(zip(start, grid_size, self.rope_embedder.axes_lens)):
      if s + n > max_len:
        raise ValueError(
            f"Sequence position coordinates along axis {axis} start={s}, size={n} (end={s + n}) "
            f"exceed max axis length limit {max_len}. Please increase axes_lens[{axis}] in ZImageRopeEmbedder."
        )
    original = feature.shape[0]
    pad = (-original) % SEQ_MULTI_OF
    total = original + pad
    positions = self._coordinate_grid(grid_size, start)
    if pad:
      feature = jnp.concatenate((feature, jnp.repeat(feature[-1:], pad, axis=0)))
      positions = jnp.concatenate((positions, jnp.repeat(jnp.zeros((1, 3), dtype=jnp.int32), pad, axis=0)))
    return feature, positions[:total], jnp.arange(total) >= original

  @staticmethod
  def _stack_and_pad_sequences(
      features: list[jax.Array],
      positions: list[jax.Array],
      masks: list[jax.Array],
      max_len: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Stack features into 3D batch arrays, padding sequence dimension if lengths differ in the batch."""
    lengths = [f.shape[0] for f in features]
    if all(l == max_len for l in lengths):
      return (
          jnp.stack(features, axis=0),
          jnp.stack(positions, axis=0),
          jnp.stack(masks, axis=0),
      )

    features_padded, positions_padded, masks_padded = [], [], []
    for feat, pos, mask in zip(features, positions, masks):
      missing = max_len - feat.shape[0]
      if missing == 0:
        features_padded.append(feat)
        positions_padded.append(pos)
        masks_padded.append(mask)
      else:
        features_padded.append(jnp.pad(feat, ((0, missing), (0, 0))))
        positions_padded.append(jnp.pad(pos, ((0, missing), (0, 0))))
        masks_padded.append(jnp.pad(mask, (0, missing), constant_values=True))

    return (
        jnp.stack(features_padded, axis=0),
        jnp.stack(positions_padded, axis=0),
        jnp.stack(masks_padded, axis=0),
    )

  @staticmethod
  def _batch_padding_mask(lengths: list[int], max_len: int) -> Optional[jax.Array]:
    """True where a token is real for its batch element, None when nothing is padded.

    Covers only the padding that squares up a ragged batch; a sequence's own
    pad tokens stay visible to attention, as upstream leaves them.
    """
    if all(length == max_len for length in lengths):
      return None
    return jnp.stack([jnp.arange(max_len) < length for length in lengths])

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

    # 1. Pad and patchify each element locally (on its device)
    caption_features, caption_positions, caption_masks = [], [], []
    image_features, image_positions, image_masks = [], [], []
    sizes = []

    for image, caption in zip(x, cap_feats):
      # Captions are laid out over their padded length, and the image grid
      # starts one past the end of that block -- both upstream conventions.
      caption_len = caption.shape[0] + (-caption.shape[0]) % SEQ_MULTI_OF
      caption_padded, cap_position, cap_mask = self._pad(caption, (caption_len, 1, 1), (1, 0, 0))
      patch, size, tokens = self._patchify(image)
      patch_padded, image_position, image_mask = self._pad(patch, tokens, (caption_len + 1, 0, 0))

      caption_features.append(caption_padded)
      caption_positions.append(cap_position)
      caption_masks.append(cap_mask)
      image_features.append(patch_padded)
      image_positions.append(image_position)
      image_masks.append(image_mask)
      sizes.append(size)

    # 2. Find maximum sequence lengths statically (Python values).
    # Text caption lengths are fixed at 512, and image sequence lengths are uniform
    # across the batch unless multi-resolution inputs are used, so padding usually does not occur.
    image_lengths = [item.shape[0] for item in image_features]
    caption_lengths = [item.shape[0] for item in caption_features]
    max_image_len = max(image_lengths)
    max_caption_len = max(caption_lengths)

    # 3. Stack into 3D batch arrays (pad only if sequence lengths differ in the batch)
    (
        caption_features_stacked,
        caption_positions_stacked,
        caption_masks_stacked,
    ) = self._stack_and_pad_sequences(caption_features, caption_positions, caption_masks, max_caption_len)

    (
        image_features_stacked,
        image_positions_stacked,
        image_masks_stacked,
    ) = self._stack_and_pad_sequences(image_features, image_positions, image_masks, max_image_len)

    # 4. Embed images and captions in parallel
    image_embeddings = self.x_embedder(image_features_stacked)
    image_embeddings = jnp.where(image_masks_stacked[..., None], self.x_pad_token[...], image_embeddings)

    captions = self.cap_embedder(self.cap_embedder_norm(caption_features_stacked))
    captions = jnp.where(caption_masks_stacked[..., None], self.cap_pad_token[...], captions)

    # 5. Rope frequencies
    image_cos, image_sin = self.rope_embedder(image_positions_stacked)
    caption_cos, caption_sin = self.rope_embedder(caption_positions_stacked)

    # 6. Refiner blocks
    image_attention_mask = self._batch_padding_mask(image_lengths, max_image_len)
    if self.scan_layers:
      # Blocks are dtype-preserving, so scan's carry-invariance check is
      # what keeps this stack agreeing with the unrolled one below.
      def noise_scan_fn(carry, block):
        return block(carry, (image_cos, image_sin), image_attention_mask, adaln), None

      image_embeddings, _ = nnx.scan(
          noise_scan_fn,
          length=self.n_refiner_layers,
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, 0),
      )(image_embeddings, self.noise_refiner)
    else:
      for block in self.noise_refiner:
        image_embeddings = block(image_embeddings, (image_cos, image_sin), image_attention_mask, adaln)

    caption_attention_mask = self._batch_padding_mask(caption_lengths, max_caption_len)
    if self.scan_layers:

      def context_scan_fn(carry, block):
        return block(carry, (caption_cos, caption_sin), caption_attention_mask), None

      captions, _ = nnx.scan(
          context_scan_fn,
          length=self.n_refiner_layers,
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, 0),
      )(captions, self.context_refiner)
    else:
      for block in self.context_refiner:
        captions = block(captions, (caption_cos, caption_sin), caption_attention_mask)

    # 7. Joint sequence construction (no physical cross-device communication)
    joint = jnp.concatenate((image_embeddings, captions), axis=1)
    joint_cos = jnp.concatenate((image_cos, caption_cos), axis=1)
    joint_sin = jnp.concatenate((image_sin, caption_sin), axis=1)
    freqs = (joint_cos, joint_sin)

    # 8. Model layers. Image and caption blocks stay padded separately
    # rather than packed back to back; attention is permutation invariant
    # given the mask and each token carries its own RoPE frequencies.
    if image_attention_mask is None and caption_attention_mask is None:
      joint_attention_mask = None
    else:
      batch = joint.shape[0]
      joint_attention_mask = jnp.concatenate(
          (
              image_attention_mask if image_attention_mask is not None else jnp.ones((batch, max_image_len), dtype=bool),
              caption_attention_mask
              if caption_attention_mask is not None
              else jnp.ones((batch, max_caption_len), dtype=bool),
          ),
          axis=1,
      )

    if self.scan_layers:

      def layers_scan_fn(carry, block):
        return block(carry, freqs, joint_attention_mask, adaln), None

      joint, _ = nnx.scan(
          layers_scan_fn,
          length=self.n_layers,
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, 0),
      )(joint, self.layers)
    else:
      for block in self.layers:
        joint = block(joint, freqs, joint_attention_mask, adaln)
    hidden_states = self.final_layer(joint, c=adaln)

    # 9. Unpatchify outputs
    outputs = []
    for index, (frames, height, width) in enumerate(sizes):
      tokens = frames * (height // 2) * (width // 2)
      patches = hidden_states[index, :tokens].reshape(frames, height // 2, width // 2, 1, 2, 2, self.out_channels)
      outputs.append(patches.transpose(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, frames, height, width))
    return ZImageTransformer2DModelOutput(sample=outputs) if return_dict else (outputs,)
