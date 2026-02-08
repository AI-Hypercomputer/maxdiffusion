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


def apply_rotary_emb(x: Array, freqs: Tuple[Array, Array]) -> Array:
    """Apply rotary embeddings to input x."""
    cos, sin = freqs
    # x shape: [B, S, H, D]
    # cos/sin shape: [B, S, 1, D]
    
    # Standard interleaved rotation: [-x2, x1]
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]
    
    x_rotated = jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)
    
    out = x.astype(jnp.float32) * cos + x_rotated.astype(jnp.float32) * sin
    return out.astype(x.dtype)


class LTX2RotaryPosEmbed(nnx.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, ids: Array) -> Tuple[Array, Array]:
        """
        Generates RoPE frequencies.
        Args:
            ids: [B, S, Num_Axes]
                 - For Video 3D: Num_Axes=3 (T, H, W)
                 - For Audio 1D: Num_Axes=1 (T)
                 - For Temporal-Only: Pass ids[:, :, 0:1] (Slice to keep only Time)
        """
        num_axes = ids.shape[-1]
        dim_per_axis = self.dim // num_axes
        
        freq_indices = jnp.arange(0, dim_per_axis, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.theta ** (freq_indices / dim_per_axis))
        
        freqs_list = []
        for i in range(num_axes):
            axis_pos = ids[..., i] 
            freqs = jnp.einsum('bs,d->bsd', axis_pos, inv_freq)
            freqs_list.append(freqs)
            
        emb = jnp.concatenate(freqs_list, axis=-1)
        
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        cos = jnp.repeat(cos, 2, axis=-1)
        sin = jnp.repeat(sin, 2, axis=-1)
        
        # Add head dim: [B, S, 1, D]
        return cos[:, :, None, :], sin[:, :, None, :]


class LTX2Attention(nnx.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        out_bias: bool = True,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        eps: float = 1e-6,
        dtype: DType = jnp.float32,
        attention_kernel: str = "flash",
    ):
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.dropout_rate = dropout

        self.to_q = nnx.Linear(query_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)
        
        kv_dim = context_dim if context_dim is not None else query_dim
        self.to_k = nnx.Linear(kv_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)
        self.to_v = nnx.Linear(kv_dim, self.inner_dim, use_bias=bias, rngs=rngs, dtype=dtype)

        # Norm over full inner_dim (Fix #2)
        self.norm_q = nnx.RMSNorm(self.inner_dim, epsilon=eps, dtype=dtype, use_scale=True, rngs=rngs)
        self.norm_k = nnx.RMSNorm(self.inner_dim, epsilon=eps, dtype=dtype, use_scale=True, rngs=rngs)

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
        
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # 1. Project
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        # 2. Norm
        query = self.norm_q(query)
        key = self.norm_k(key)

        # 3. Reshape for RoPE [B, S, H, D]
        query = query.reshape(*query.shape[:-1], self.heads, self.dim_head)
        key = key.reshape(*key.shape[:-1], self.heads, self.dim_head)
        value = value.reshape(*value.shape[:-1], self.heads, self.dim_head)

        # 4. Apply RoPE
        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            
            if k_rotary_emb is not None:
                key = apply_rotary_emb(key, k_rotary_emb)
            elif encoder_hidden_states is None: # Self-Attention
                key = apply_rotary_emb(key, rotary_emb)

        # 5. Flatten back for AttentionOp (Fix #1)
        # [B, S, H, D] -> [B, S, H*D]
        query = query.reshape(*query.shape[:-2], self.inner_dim)
        key = key.reshape(*key.shape[:-2], self.inner_dim)
        value = value.reshape(*value.shape[:-2], self.inner_dim)

        # 6. Attention
        attn_output = self.attention_op.apply_attention(
            query=query, key=key, value=value, attention_mask=attention_mask
        )

        # attn_output is already [B, S, H*D], no reshape needed before output proj
        hidden_states = self.to_out(attn_output)

        if self.dropout_layer is not None:
            hidden_states = self.dropout_layer(hidden_states)

        return hidden_states
