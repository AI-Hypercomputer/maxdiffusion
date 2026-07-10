# Copyright 2026 The MaxDiffusion Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
# Qwen3 Configuration
# -----------------------------------------------------------------------------

class FlaxQwen3Config:
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 40960,
        dtype = jnp.float32,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype

# -----------------------------------------------------------------------------
# Core Model Layers
# -----------------------------------------------------------------------------

class FlaxQwen3RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        scale = self.param("weight", nn.initializers.ones, (self.dim,), self.dtype)
        normed = x_float * jax.lax.rsqrt(variance + self.eps)
        return (normed.astype(self.dtype)) * scale

class FlaxQwen3MLP(nn.Module):
    config: FlaxQwen3Config

    @nn.compact
    def __call__(self, x):
        gate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name="gate_proj",
        )
        up_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name="up_proj",
        )
        down_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name="down_proj",
        )

        return down_proj(jax.nn.silu(gate_proj(x)) * up_proj(x))

# -----------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# -----------------------------------------------------------------------------

def precompute_qwen3_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 1000000.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precomputes the cosine and sine tables for RoPE.
    Matches standard Llama/Qwen half-half rotation layout.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    
    # Concatenate [freqs, freqs] to match Hugging Face's rotate_half layout
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (max_seq_len, head_dim)
    
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin

def apply_qwen3_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies RoPE to Q and K tensors.
    q shape: (batch, seq_len, num_heads, head_dim)
    k shape: (batch, seq_len, num_kv_heads, head_dim)
    cos, sin shape: (seq_len, head_dim)
    """
    # Reshape cos/sin to (1, seq_len, 1, head_dim) for broadcasting
    cos = cos[jnp.newaxis, :, jnp.newaxis, :]
    sin = sin[jnp.newaxis, :, jnp.newaxis, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# -----------------------------------------------------------------------------
# Self Attention (Grouped Query Attention)
# -----------------------------------------------------------------------------

class FlaxQwen3Attention(nn.Module):
    config: FlaxQwen3Config

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        cos_table: Optional[jnp.ndarray] = None,
        sin_table: Optional[jnp.ndarray] = None,
    ):
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V
        # Output sizes: Q: 4096, K: 1024, V: 1024
        q_proj = nn.Dense(
            self.config.num_attention_heads * self.config.head_dim,
            use_bias=False,
            dtype=self.config.dtype,
            name="q_proj",
        )
        k_proj = nn.Dense(
            self.config.num_key_value_heads * self.config.head_dim,
            use_bias=False,
            dtype=self.config.dtype,
            name="k_proj",
        )
        v_proj = nn.Dense(
            self.config.num_key_value_heads * self.config.head_dim,
            use_bias=False,
            dtype=self.config.dtype,
            name="v_proj",
        )
        o_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name="o_proj",
        )

        # QK-Norm Layers (Head-wise, sharing scale weights of size head_dim = 128)
        q_norm = FlaxQwen3RMSNorm(
            dim=self.config.head_dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="q_norm",
        )
        k_norm = FlaxQwen3RMSNorm(
            dim=self.config.head_dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="k_norm",
        )

        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)

        # 2. Reshape to heads first: (batch, seq_len, num_heads, head_dim)
        q = q.reshape((batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim))
        k = k.reshape((batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim))
        v = v.reshape((batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim))

        # Apply QK-Norm head-wise (normalizes over the last axis of size 128)
        q = q_norm(q)
        k = k_norm(k)

        # 3. Apply RoPE
        if cos_table is not None and sin_table is not None:
            # Extract cos/sin for the current sequence length
            cos = cos_table[:seq_len, :]
            sin = sin_table[:seq_len, :]
            q, k = apply_qwen3_rotary_pos_emb(q, k, cos, sin)

        # 4. Repeat KV heads to match Query heads (GQA)
        gqa_ratio = self.config.num_attention_heads // self.config.num_key_value_heads
        if gqa_ratio > 1:
            k = jnp.repeat(k, gqa_ratio, axis=-2)
            v = jnp.repeat(v, gqa_ratio, axis=-2)

        # 5. Transpose to (batch, num_heads, seq_len, head_dim) for attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # 6. Compute attention logits in float32
        q_f = q.astype(jnp.float32)
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)

        scores = jnp.matmul(q_f, jnp.transpose(k_f, (0, 1, 3, 2))) / math.sqrt(self.config.head_dim)

        # 7. Apply causal attention mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal_mask, scores, -1e4)

        # 8. Apply padding attention mask if provided
        if attention_mask is not None:
            p_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :].astype(jnp.bool_)
            scores = jnp.where(p_mask, scores, -1e4)

        # 9. Softmax & Weighted Sum in float32
        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(probs, v_f).astype(self.config.dtype)

        # 10. Reshape back and project out: (batch, seq_len, hidden_size)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, -1))
        return o_proj(out)

# -----------------------------------------------------------------------------
# Decoder Block Layer
# -----------------------------------------------------------------------------

class FlaxQwen3DecoderLayer(nn.Module):
    config: FlaxQwen3Config

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        cos_table: Optional[jnp.ndarray] = None,
        sin_table: Optional[jnp.ndarray] = None,
    ):
        # input_layernorm
        input_layernorm = FlaxQwen3RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="input_layernorm",
        )
        # self_attn
        self_attn = FlaxQwen3Attention(
            config=self.config,
            name="self_attn",
        )
        # post_attention_layernorm
        post_attention_layernorm = FlaxQwen3RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="post_attention_layernorm",
        )
        # mlp
        mlp = FlaxQwen3MLP(
            config=self.config,
            name="mlp",
        )

        # Self-Attention block (with residual)
        attn_out = self_attn(
            input_layernorm(x),
            attention_mask=attention_mask,
            cos_table=cos_table,
            sin_table=sin_table,
        )
        x = x + attn_out

        # MLP block (with residual)
        mlp_out = mlp(post_attention_layernorm(x))
        x = x + mlp_out

        return x

# -----------------------------------------------------------------------------
# Full Transformer Model
# -----------------------------------------------------------------------------

class FlaxQwen3Model(nn.Module):
    config: FlaxQwen3Config

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Runs the full Qwen3-4B model.
        Returns:
            last_hidden_state: Output of the final layer (batch, seq_len, 2560)
            all_hidden_states: List of activations from every layer, including token embeddings (length 37)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token Embeddings
        embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=self.config.hidden_size**-0.5),
            dtype=self.config.dtype,
            name="embed_tokens",
        )
        hidden_states = embed_tokens(input_ids)
        
        # Track all layer activations (including embedding layer)
        all_hidden_states = [hidden_states]

        # 2. Precompute RoPE cos/sin tables
        cos_table, sin_table = precompute_qwen3_freqs_cis(
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            theta=self.config.rope_theta,
        )

        # 3. Stacked Decoder Layers
        for i in range(self.config.num_hidden_layers):
            layer = FlaxQwen3DecoderLayer(
                config=self.config,
                name=f"layers_{i}",
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                cos_table=cos_table,
                sin_table=sin_table,
            )
            all_hidden_states.append(hidden_states)

        # 4. Final RMSNorm
        norm = FlaxQwen3RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="norm",
        )
        hidden_states = norm(hidden_states)
        
        # Keep all_hidden_states as the raw outputs of the layers, do not overwrite with final norm.

        return hidden_states, all_hidden_states

# -----------------------------------------------------------------------------
# Weight Mapping & Conversion Utilities
# -----------------------------------------------------------------------------

def load_and_convert_qwen3_weights(
    safetensors_path: str, jax_params: dict, config: FlaxQwen3Config
) -> dict:
    """
    Loads PyTorch weights from a safetensors file or directory of shards,
    and converts them to our JAX parameter dictionary in-place.
    """
    import glob
    import os
    from safetensors.torch import load_file
    import torch

    torch_weights: dict = {}
    if os.path.isdir(safetensors_path):
        # Find all safetensors shards
        shards = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        print(f"Loading sharded Qwen3 weights from directory: {safetensors_path} (Found {len(shards)} shards)...")
        for shard in sorted(shards):
            print(f"Loading shard: {shard}...")
            torch_weights.update(load_file(shard, device="cpu"))
    else:
        # Single file path
        print(f"Loading Qwen3 weights from file: {safetensors_path}...")
        torch_weights = load_file(safetensors_path, device="cpu")
    print("PyTorch weights loaded successfully. Starting JAX parameter mapping...")

    # Helper to transpose and cast weight
    def get_w(name: str, transpose: bool = True) -> np.ndarray:
        nonlocal torch_weights
        if name not in torch_weights:
            raise KeyError(f"Weight '{name}' not found in PyTorch safetensors!")
        t = torch_weights[name]
        # Transpose linear layer weights (2D tensors) from (out, in) to (in, out)
        if len(t.shape) == 2 and transpose:
            t = t.T
        return t.to(torch.float32).numpy()

    # Create mutable copy of JAX params to populate
    import flax
    flat_params = flax.traverse_util.flatten_dict(jax_params)
    converted_flat = {}

    for k, v in flat_params.items():
        # Reconstruct path string for debugging/matching
        path_str = ".".join(k)

        # 1. Token Embeddings
        if k[0] == "embed_tokens" and k[1] == "embedding":
            converted_flat[k] = get_w("model.embed_tokens.weight", transpose=False)

        # 2. Decoder Layer Normalizations (RMSNorm)
        elif "input_layernorm" in path_str and k[-1] == "weight":
            layer_idx = k[0].split("_")[1]
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.input_layernorm.weight")

        elif "post_attention_layernorm" in path_str and k[-1] == "weight":
            layer_idx = k[0].split("_")[1]
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

        # 3. Attention Projections & QK-Norm
        elif "self_attn" in path_str and k[-1] == "kernel":
            layer_idx = k[0].split("_")[1]
            proj_name = k[2] # q_proj, k_proj, v_proj, o_proj
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.{proj_name}.weight")

        elif "self_attn" in path_str and "q_norm" in path_str and k[-1] == "weight":
            layer_idx = k[0].split("_")[1]
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.q_norm.weight")

        elif "self_attn" in path_str and "k_norm" in path_str and k[-1] == "weight":
            layer_idx = k[0].split("_")[1]
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.self_attn.k_norm.weight")

        # 4. MLP Block
        elif "mlp" in path_str and k[-1] == "kernel":
            layer_idx = k[0].split("_")[1]
            proj_name = k[2] # gate_proj, up_proj, down_proj
            converted_flat[k] = get_w(f"model.layers.{layer_idx}.mlp.{proj_name}.weight")

        # 5. Final RMSNorm
        elif k[0] == "norm" and k[1] == "weight":
            converted_flat[k] = get_w("model.norm.weight")

        else:
            print(f"WARNING: JAX parameter '{path_str}' did not match any PyTorch weights!")
            converted_flat[k] = np.zeros(v.shape, dtype=np.float32) if hasattr(v, 'shape') and not isinstance(v, np.ndarray) else v

    # Clean up PyTorch memory immediately
    del torch_weights
    import gc
    gc.collect()

    res = flax.traverse_util.unflatten_dict(converted_flat)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf,
        res
    )
