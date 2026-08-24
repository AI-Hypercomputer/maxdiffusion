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

import math
from typing import Any, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

from .constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR

# Segment id handed to the splash kernel for positions that must not attend to
# real tokens: both sequence padding and the zero rows we add to round the
# sequence up to a whole number of blocks.
_PAD_SEGMENT = 0
_VALID_SEGMENT = 1


@dataclass(frozen=True)
class Ideogram4Config:
  """Geometry of the Ideogram 4 transformer plus the runtime knobs wired from
  ``base_ideogram.yml`` (``attention``, ``weights_dtype``, ``activations_dtype``,
  ``flash_block_*``). The defaults reproduce the reference fp32 model.

  Frozen, with only hashable fields: the transformer keeps this on ``self``, so
  it ends up in the nnx graphdef, which is a static argument to the jitted
  denoise step. An unhashable field (e.g. a dict of block sizes) would raise
  there, and a non-frozen dataclass is unhashable because it defines ``__eq__``.
  """

  emb_dim: int = 4608
  num_heads: int = 18
  in_channels: int = 128
  llm_features_dim: int = 53248
  adanln_dim: int = 512
  rope_theta: int = 5000000
  mrope_section: Tuple[int, int, int] = (24, 20, 20)
  intermediate_size: int = 12288
  norm_eps: float = 1e-5
  num_layers: int = 34
  patch_size: int = 2

  # Runtime configuration, populated from the maxdiffusion config.
  attention: str = "dot_product"
  weights_dtype: Any = jnp.float32
  activations_dtype: Any = jnp.float32
  precision: Any = None
  # Splash block sizes. head_dim is 256 here, which is large enough that the
  # library defaults pick a tile that spills VMEM and runs an order of magnitude
  # slower than the naive einsum -- always benchmark after changing these.
  flash_block_q: int = 1024
  flash_block_kv: int = 1024


def _pad_to_multiple(x: jax.Array, multiple: int, axis: int) -> jax.Array:
  pad = (-x.shape[axis]) % multiple
  if pad == 0:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, pad)
  return jnp.pad(x, pad_width)


def _splash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    block_q: int,
    block_kv: int,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> jax.Array:
  """Flash (splash) self-attention over a block-diagonal segment mask.

  ``q``/``k``/``v`` are ``(B, H, L, D)`` with the softmax scale already folded
  into ``q``; ``segment_ids`` is ``(B, L)`` valued in ``{_PAD_SEGMENT,
  _VALID_SEGMENT}``. Splash masks every pair whose q and kv segment ids differ,
  which is exactly the ``segment_ids[:, :, None] == segment_ids[:, None, :]``
  mask the dense path builds -- without ever materializing the L x L score
  matrix.
  """
  seq_len = q.shape[2]
  block = max(block_q, block_kv)
  padded_len = seq_len + (-seq_len) % block

  q_p = _pad_to_multiple(q, block, axis=2)
  k_p = _pad_to_multiple(k, block, axis=2)
  v_p = _pad_to_multiple(v, block, axis=2)
  # Rows added by padding join the _PAD_SEGMENT group. That group is never
  # empty (it contains at least these rows), so no query row ends up with an
  # all-masked softmax, which would produce NaNs.
  seg_p = jnp.pad(segment_ids, ((0, 0), (0, padded_len - seq_len)), constant_values=_PAD_SEGMENT)

  num_heads = q.shape[1]
  mask = splash_attention_mask.MultiHeadMask(
      masks=(splash_attention_mask.FullMask(_shape=(padded_len, padded_len)),) * num_heads
  )
  kernel = splash_attention_kernel.make_splash_mha(
      mask=mask,
      head_shards=1,
      q_seq_shards=1,
      block_sizes=splash_attention_kernel.BlockSizes(
          block_q=min(block_q, padded_len),
          block_kv=min(block_kv, padded_len),
          block_kv_compute=min(block_kv, padded_len),
      ),
  )
  seg = splash_attention_kernel.SegmentIds(q=seg_p, kv=seg_p)
  vmapped = jax.vmap(kernel, in_axes=(0, 0, 0, 0))

  if mesh is not None and mesh.devices.size > 1:
    # Mosaic kernels cannot be auto-partitioned ("Please wrap the call in a
    # shard_map"), and a with_sharding_constraint to replicated is not enough --
    # XLA still tries to partition the custom call. Activations here are
    # replicated (only params are sharded), and num_heads=18 does not divide the
    # 8-device mesh anyway, so every device runs the same attention, exactly as
    # the dense einsum path does.
    replicated = jax.sharding.PartitionSpec()
    vmapped = jax.shard_map(
        vmapped,
        mesh=mesh,
        in_specs=(replicated,) * 4,
        out_specs=replicated,
        check_vma=False,
    )

  out = vmapped(q_p, k_p, v_p, seg)
  return out[:, :, :seq_len, :]


class Ideogram4MRoPE(nnx.Module):

  def __init__(self, head_dim: int, base: int, mrope_section: Tuple[int, int, int]):
    self.head_dim = head_dim
    self.mrope_section = mrope_section
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    self.inv_freq = nnx.Variable(inv_freq)

  def __call__(self, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # position_ids: (B, L, 3)
    inv_freq = self.inv_freq.value

    freqs_axes = []
    for i in range(3):
      pos_axis = position_ids[..., i].astype(jnp.float32)
      f = jnp.einsum("i, bl -> bli", inv_freq, pos_axis)
      freqs_axes.append(f)

    freqs_t = freqs_axes[0]

    # Interleave logic
    # In PyTorch:
    # for axis, offset in ((1, 1), (2, 2)):
    #   length = self.mrope_section[axis] * 3
    #   idx = torch.arange(offset, length, 3)
    #   freqs_t[..., idx] = freqs_axes[axis][..., idx]

    # In JAX, we can create an array of indices.
    inv_freq_size = freqs_t.shape[-1]
    indices = jnp.arange(inv_freq_size)

    cond_h = (indices % 3 == 1) & (indices < self.mrope_section[1] * 3)
    cond_w = (indices % 3 == 2) & (indices < self.mrope_section[2] * 3)

    freqs_t = jnp.where(cond_h, freqs_axes[1], freqs_t)
    freqs_t = jnp.where(cond_w, freqs_axes[2], freqs_t)

    emb = jnp.concatenate([freqs_t, freqs_t], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: jax.Array) -> jax.Array:
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array) -> Tuple[jax.Array, jax.Array]:
  cos = jnp.expand_dims(cos, axis=1)
  sin = jnp.expand_dims(sin, axis=1)
  q_embed = (q * cos) + (_rotate_half(q) * sin)
  k_embed = (k * cos) + (_rotate_half(k) * sin)
  return q_embed, k_embed


class Ideogram4Attention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      num_heads: int,
      eps: float = 1e-5,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      precision=None,
      attention: str = "dot_product",
      flash_block_q: int = 1024,
      flash_block_kv: int = 1024,
      mesh: Optional[jax.sharding.Mesh] = None,
  ):
    self.mesh = mesh
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.dtype = dtype
    self.attention = attention
    self.block_q = flash_block_q
    self.block_kv = flash_block_kv

    self.qkv = nnx.Linear(
        hidden_size,
        hidden_size * 3,
        use_bias=False,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
    )
    self.norm_q = nnx.RMSNorm(self.head_dim, epsilon=eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    self.norm_k = nnx.RMSNorm(self.head_dim, epsilon=eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    self.o = nnx.Linear(
        hidden_size,
        hidden_size,
        use_bias=False,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
    )

  def __call__(
      self,
      x: jax.Array,
      attn_mask: Optional[jax.Array],
      splash_segment_ids: Optional[jax.Array],
      cos: jax.Array,
      sin: jax.Array,
  ) -> jax.Array:
    batch_size, seq_len, _ = x.shape

    qkv = self.qkv(x)
    qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))

    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]

    q = self.norm_q(q)
    k = self.norm_k(k)

    # Transpose to (B, num_heads, L, head_dim)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    scale = 1.0 / math.sqrt(self.head_dim)

    if self.attention == "flash":
      # Splash applies no softmax scale of its own, so fold it into q first.
      out = _splash_attention(
          (q * scale).astype(self.dtype),
          k.astype(self.dtype),
          v.astype(self.dtype),
          splash_segment_ids,
          self.block_q,
          self.block_kv,
          self.mesh,
      )
    else:
      attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
      attn_weights = jnp.where(attn_mask, attn_weights, -1e10)
      attn_weights = jax.nn.softmax(attn_weights, axis=-1)
      out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, self.hidden_size))
    return self.o(out)


class Ideogram4MLP(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, dim: int, hidden_dim: int, dtype=jnp.float32, param_dtype=jnp.float32, precision=None):
    kwargs = dict(use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision)
    self.w1 = nnx.Linear(
        dim, hidden_dim, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")), **kwargs
    )
    self.w2 = nnx.Linear(
        hidden_dim, dim, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")), **kwargs
    )
    self.w3 = nnx.Linear(
        dim, hidden_dim, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")), **kwargs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      norm_eps: float,
      adanln_dim: int,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      precision=None,
      attention: str = "dot_product",
      flash_block_q: int = 1024,
      flash_block_kv: int = 1024,
      mesh: Optional[jax.sharding.Mesh] = None,
  ):
    self.attention = Ideogram4Attention(
        rngs,
        hidden_size,
        num_heads,
        eps=1e-5,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        attention=attention,
        flash_block_q=flash_block_q,
        flash_block_kv=flash_block_kv,
        mesh=mesh,
    )
    self.feed_forward = Ideogram4MLP(
        rngs, hidden_size, intermediate_size, dtype=dtype, param_dtype=param_dtype, precision=precision
    )

    norm_kwargs = dict(epsilon=norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    self.attention_norm1 = nnx.RMSNorm(hidden_size, **norm_kwargs)
    self.ffn_norm1 = nnx.RMSNorm(hidden_size, **norm_kwargs)
    self.attention_norm2 = nnx.RMSNorm(hidden_size, **norm_kwargs)
    self.ffn_norm2 = nnx.RMSNorm(hidden_size, **norm_kwargs)

    self.adaln_modulation = nnx.Linear(
        adanln_dim,
        4 * hidden_size,
        use_bias=True,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "embed")),
    )

  def __call__(
      self,
      x: jax.Array,
      attn_mask: Optional[jax.Array],
      splash_segment_ids: Optional[jax.Array],
      cos: jax.Array,
      sin: jax.Array,
      adaln_input: jax.Array,
  ) -> jax.Array:
    mod = self.adaln_modulation(adaln_input)

    # mod is split into 4 parts
    hidden_size = x.shape[-1]
    scale_msa = mod[..., 0 * hidden_size : 1 * hidden_size]
    gate_msa = mod[..., 1 * hidden_size : 2 * hidden_size]
    scale_mlp = mod[..., 2 * hidden_size : 3 * hidden_size]
    gate_mlp = mod[..., 3 * hidden_size : 4 * hidden_size]

    gate_msa = jnp.tanh(gate_msa)
    gate_mlp = jnp.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp

    attn_out = self.attention(
        self.attention_norm1(x) * scale_msa,
        attn_mask=attn_mask,
        splash_segment_ids=splash_segment_ids,
        cos=cos,
        sin=sin,
    )
    x = x + gate_msa * self.attention_norm2(attn_out)
    x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    return x


def _sinusoidal_embedding(t: jax.Array, dim: int, scale: float = 1e4) -> jax.Array:
  t = t.astype(jnp.float32)
  half = dim // 2
  freq = math.log(scale) / (half - 1)
  freq = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -freq)
  emb = jnp.expand_dims(t, -1) * freq
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
  if dim % 2 == 1:
    emb = jnp.pad(emb, ((0, 0), (0, 1)))
  return emb


class Ideogram4EmbedScalar(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      input_range: Tuple[float, float],
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      precision=None,
  ):
    self.dim = dim
    self.range_min, self.range_max = input_range
    kwargs = dict(use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision)
    self.mlp_in = nnx.Linear(dim, dim, **kwargs)
    self.mlp_out = nnx.Linear(dim, dim, **kwargs)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
    emb = _sinusoidal_embedding(scaled, self.dim)
    emb = emb.astype(self.mlp_in.dtype)
    emb = jax.nn.silu(self.mlp_in(emb))
    return self.mlp_out(emb)


class Ideogram4FinalLayer(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      out_channels: int,
      adanln_dim: int,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      precision=None,
  ):
    self.norm_final = nnx.LayerNorm(
        hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    kwargs = dict(use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision)
    self.linear = nnx.Linear(
        hidden_size,
        out_channels,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", None)),
        **kwargs,
    )
    self.adaln_modulation = nnx.Linear(adanln_dim, hidden_size, **kwargs)

  def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
    scale = 1.0 + self.adaln_modulation(jax.nn.silu(c))
    return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: Any, dtype=None, param_dtype=None, mesh=None):
    self.config = config
    self.mesh = mesh
    # ``dtype``/``param_dtype`` arguments stay for callers that want to override
    # the config; otherwise the config (fed from base_ideogram.yml) decides.
    self.dtype = dtype if dtype is not None else getattr(config, "activations_dtype", jnp.float32)
    param_dtype = param_dtype if param_dtype is not None else getattr(config, "weights_dtype", jnp.float32)
    self.param_dtype = param_dtype
    precision = getattr(config, "precision", None)
    attention = getattr(config, "attention", "dot_product")
    self.attention = attention
    flash_block_q = getattr(config, "flash_block_q", 1024)
    flash_block_kv = getattr(config, "flash_block_kv", 1024)

    dtype = self.dtype
    head_dim = config.emb_dim // config.num_heads
    lin_kwargs = dict(use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision)

    self.input_proj = nnx.Linear(
        config.in_channels,
        config.emb_dim,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "embed")),
        **lin_kwargs,
    )
    self.llm_cond_norm = nnx.RMSNorm(config.llm_features_dim, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    self.llm_cond_proj = nnx.Linear(
        config.llm_features_dim,
        config.emb_dim,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "embed")),
        **lin_kwargs,
    )

    self.t_embedding = Ideogram4EmbedScalar(
        rngs, config.emb_dim, input_range=(0.0, 1.0), dtype=dtype, param_dtype=param_dtype, precision=precision
    )
    self.adaln_proj = nnx.Linear(config.emb_dim, config.adanln_dim, **lin_kwargs)

    self.embed_image_indicator = nnx.Embed(2, config.emb_dim, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

    self.rotary_emb = Ideogram4MRoPE(
        head_dim=head_dim,
        base=config.rope_theta,
        mrope_section=config.mrope_section,
    )

    self.layers = nnx.List(
        [
            Ideogram4TransformerBlock(
                rngs,
                hidden_size=config.emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                norm_eps=config.norm_eps,
                adanln_dim=config.adanln_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                attention=attention,
                flash_block_q=flash_block_q,
                flash_block_kv=flash_block_kv,
                mesh=mesh,
            )
            for _ in range(config.num_layers)
        ]
    )

    self.final_layer = Ideogram4FinalLayer(
        rngs,
        hidden_size=config.emb_dim,
        out_channels=config.in_channels,
        adanln_dim=config.adanln_dim,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
    )

  def __call__(
      self,
      llm_features: Optional[jax.Array],
      x: jax.Array,
      t: jax.Array,
      position_ids: jax.Array,
      segment_ids: jax.Array,
      indicator: jax.Array,
  ) -> jax.Array:
    x = x.astype(self.dtype)
    t = t.astype(self.dtype)

    indicator = indicator.astype(jnp.int32)

    llm_token_mask = jnp.expand_dims((indicator == LLM_TOKEN_INDICATOR).astype(self.dtype), -1)
    output_image_mask = jnp.expand_dims((indicator == OUTPUT_IMAGE_INDICATOR).astype(self.dtype), -1)

    x = x * output_image_mask

    x = self.input_proj(x) * output_image_mask

    t_cond = self.t_embedding(t)
    if t.ndim == 1:
      t_cond = jnp.expand_dims(t_cond, 1)

    adaln_input = jax.nn.silu(self.adaln_proj(t_cond))

    h = x
    if llm_features is not None:
      # llm_features covers only the text prefix, not the whole sequence. The
      # image positions have llm_token_mask == 0, so projecting them is
      # provably a no-op -- and llm_cond_proj is a 53248 -> 4608 matmul, so
      # running it over the 4096 image tokens as well used to cost ~2 TFLOP per
      # step per branch and a 940 MB activation. The unconditional branch has no
      # LLM tokens at all and passes llm_features=None.
      text_len = llm_features.shape[1]
      text_mask = llm_token_mask[:, :text_len]
      llm_features = llm_features.astype(self.dtype) * text_mask
      llm_features = self.llm_cond_norm(llm_features)
      llm_features = self.llm_cond_proj(llm_features) * text_mask
      h = h.at[:, :text_len].add(llm_features)

    image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).astype(jnp.int32))
    h = h + image_indicator_embedding

    cos, sin = self.rotary_emb(position_ids)
    cos = cos.astype(self.dtype)
    sin = sin.astype(self.dtype)

    # The attention mask is a pure function of segment_ids, so it is invariant
    # across the 34 layers (and across denoise steps). Build it once here rather
    # than rebuilding an (B, 1, L, L) tensor inside every block.
    if self.attention == "flash":
      attn_mask = None
      splash_segment_ids = jnp.where(segment_ids > 0, _VALID_SEGMENT, _PAD_SEGMENT).astype(jnp.int32)
    else:
      attn_mask = jnp.expand_dims(segment_ids, axis=2) == jnp.expand_dims(segment_ids, axis=1)
      attn_mask = jnp.expand_dims(attn_mask, axis=1)  # (B, 1, L, L)
      splash_segment_ids = None

    for layer in self.layers:
      h = layer(
          h,
          attn_mask=attn_mask,
          splash_segment_ids=splash_segment_ids,
          cos=cos,
          sin=sin,
          adaln_input=adaln_input,
      )

    out = self.final_layer(h, c=adaln_input)
    return out.astype(jnp.float32)
