import math
from typing import Tuple, Optional, Any
import jax
import jax.numpy as jnp
from flax import nnx

class Ideogram4RMSNorm(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dim: int, eps: float = 1e-6, dtype=jnp.float32):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * self.weight.value).astype(self.dtype)

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
    def __init__(self, rngs: nnx.Rngs, hidden_size: int, num_heads: int, eps: float = 1e-5, dtype=jnp.float32):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dtype = dtype

        self.qkv = nnx.Linear(
            hidden_size, hidden_size * 3, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.norm_q = Ideogram4RMSNorm(rngs, self.head_dim, eps=eps, dtype=dtype)
        self.norm_k = Ideogram4RMSNorm(rngs, self.head_dim, eps=eps, dtype=dtype)
        self.o = nnx.Linear(
            hidden_size, hidden_size, use_bias=False, rngs=rngs, dtype=dtype
        )

    def __call__(self, x: jax.Array, segment_ids: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
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

        # Block-diagonal mask from segment ids
        attn_mask = jnp.expand_dims(segment_ids, axis=2) == jnp.expand_dims(segment_ids, axis=1)
        attn_mask = jnp.expand_dims(attn_mask, axis=1) # (B, 1, L, L)
        
        # JAX scaled dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        # apply mask
        attn_weights = jnp.where(attn_mask, attn_weights, -1e10)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, self.hidden_size))
        return self.o(out)

class Ideogram4MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dim: int, hidden_dim: int, dtype=jnp.float32):
        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs, dtype=dtype)
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs, dtype=dtype)
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))

class Ideogram4TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float, adanln_dim: int, dtype=jnp.float32):
        self.attention = Ideogram4Attention(rngs, hidden_size, num_heads, eps=1e-5, dtype=dtype)
        self.feed_forward = Ideogram4MLP(rngs, hidden_size, intermediate_size, dtype=dtype)

        self.attention_norm1 = Ideogram4RMSNorm(rngs, hidden_size, eps=norm_eps, dtype=dtype)
        self.ffn_norm1 = Ideogram4RMSNorm(rngs, hidden_size, eps=norm_eps, dtype=dtype)
        self.attention_norm2 = Ideogram4RMSNorm(rngs, hidden_size, eps=norm_eps, dtype=dtype)
        self.ffn_norm2 = Ideogram4RMSNorm(rngs, hidden_size, eps=norm_eps, dtype=dtype)

        self.adaln_modulation = nnx.Linear(adanln_dim, 4 * hidden_size, use_bias=True, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array, segment_ids: jax.Array, cos: jax.Array, sin: jax.Array, adaln_input: jax.Array) -> jax.Array:
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
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )
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
    def __init__(self, rngs: nnx.Rngs, dim: int, input_range: Tuple[float, float], dtype=jnp.float32):
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = nnx.Linear(dim, dim, use_bias=True, rngs=rngs, dtype=dtype)
        self.mlp_out = nnx.Linear(dim, dim, use_bias=True, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.astype(self.mlp_in.dtype)
        emb = jax.nn.silu(self.mlp_in(emb))
        return self.mlp_out(emb)

class Ideogram4FinalLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, hidden_size: int, out_channels: int, adanln_dim: int, dtype=jnp.float32):
        self.norm_final = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype
        )
        self.linear = nnx.Linear(hidden_size, out_channels, use_bias=True, rngs=rngs, dtype=dtype)
        self.adaln_modulation = nnx.Linear(adanln_dim, hidden_size, use_bias=True, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        scale = 1.0 + self.adaln_modulation(jax.nn.silu(c))
        return self.linear(self.norm_final(x) * scale)

class Ideogram4Transformer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: Any, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nnx.Linear(config.in_channels, config.emb_dim, use_bias=True, rngs=rngs, dtype=dtype)
        self.llm_cond_norm = Ideogram4RMSNorm(rngs, config.llm_features_dim, eps=1e-6, dtype=dtype)
        self.llm_cond_proj = nnx.Linear(config.llm_features_dim, config.emb_dim, use_bias=True, rngs=rngs, dtype=dtype)
        
        self.t_embedding = Ideogram4EmbedScalar(rngs, config.emb_dim, input_range=(0.0, 1.0), dtype=dtype)
        self.adaln_proj = nnx.Linear(config.emb_dim, config.adanln_dim, use_bias=True, rngs=rngs, dtype=dtype)

        self.embed_image_indicator = nnx.Embed(2, config.emb_dim, rngs=rngs, dtype=dtype)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = [
            Ideogram4TransformerBlock(
                rngs,
                hidden_size=config.emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                norm_eps=config.norm_eps,
                adanln_dim=config.adanln_dim,
                dtype=dtype,
            )
            for _ in range(config.num_layers)
        ]

        self.final_layer = Ideogram4FinalLayer(
            rngs,
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
            dtype=dtype,
        )

    def __call__(
        self,
        llm_features: jax.Array,
        x: jax.Array,
        t: jax.Array,
        position_ids: jax.Array,
        segment_ids: jax.Array,
        indicator: jax.Array,
    ) -> jax.Array:
        batch_size, seq_len, in_channels = x.shape
        
        x = x.astype(self.dtype)
        t = t.astype(self.dtype)
        llm_features = llm_features.astype(self.dtype)

        indicator = indicator.astype(jnp.int32)
        LLM_TOKEN_INDICATOR = 1 # Update with actual constant later
        OUTPUT_IMAGE_INDICATOR = 2 # Update with actual constant later
        
        from maxdiffusion.models.ideogram.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
        
        llm_token_mask = jnp.expand_dims((indicator == LLM_TOKEN_INDICATOR).astype(self.dtype), -1)
        output_image_mask = jnp.expand_dims((indicator == OUTPUT_IMAGE_INDICATOR).astype(self.dtype), -1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.ndim == 1:
            t_cond = jnp.expand_dims(t_cond, 1)
            
        adaln_input = jax.nn.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).astype(jnp.int32))
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.astype(self.dtype)
        sin = sin.astype(self.dtype)

        for layer in self.layers:
            h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

        out = self.final_layer(h, c=adaln_input)
        return out.astype(jnp.float32)
