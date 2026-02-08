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

import unittest
import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import pandas as pd

# Set JAX to use float32 for precision checks
jax.config.update("jax_default_matmul_precision", "float32")

# ==========================================
# 1. PyTorch Reference Implementations
# ==========================================

class PytorchLTX2RotaryPosEmbed(torch.nn.Module):
    """
    Reference PyTorch implementation for LTX-2 RoPE Frequency Generation.
    Splits dim across axes, computes freqs, concatenates, and interleaves.
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, ids):
        # ids: [B, S, Num_Axes]
        num_axes = ids.shape[-1]
        dim_per_axis = self.dim // num_axes
        
        freqs_list = []
        # Standard RoPE frequencies: theta^(-2i/d)
        freq_indices = torch.arange(0, dim_per_axis, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (freq_indices / dim_per_axis))
        
        for i in range(num_axes):
            axis_pos = ids[..., i] # [B, S]
            # Outer product: [B, S, 1] * [1, 1, D/2] -> [B, S, D/2]
            freqs = torch.einsum('bs,d->bsd', axis_pos, inv_freq)
            freqs_list.append(freqs)
            
        # Concatenate axes -> [B, S, D/2]
        emb = torch.cat(freqs_list, dim=-1)
        
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        
        # Interleave: [c1, c2] -> [c1, c1, c2, c2]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        # Add head dim: [B, S, 1, D]
        return cos.unsqueeze(2), sin.unsqueeze(2)


def apply_rotary_emb_pt(x, cos, sin):
    """Standard PyTorch Interleaved RoPE application."""
    # x: [B, H, S, D] -> [B, H, S, D//2, 2]
    b, h, s, d = x.shape
    x_reshaped = x.view(b, h, s, d // 2, 2)
    x1, x2 = x_reshaped.unbind(-1)
    x_rotated = torch.stack((-x2, x1), dim=-1).view(b, h, s, d)
    
    # Cast to float32 for rotation parity
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    rot_f32 = x_rotated.to(torch.float32)
    cos_f32 = cos.to(torch.float32)
    sin_f32 = sin.to(torch.float32)
    
    out = x_f32 * cos_f32 + rot_f32 * sin_f32
    return out.to(orig_dtype)


class PytorchLTX2Attention(torch.nn.Module):
    """Reference LTX-2 Attention."""
    def __init__(self, query_dim, context_dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, bias=True), 
            torch.nn.Identity()
        )

    def forward(self, x, context=None, q_rope=None, k_rope=None):
        q = self.to_q(x)
        ctx = x if context is None else context
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        q_norm = self.q_norm(q)
        k_norm = self.k_norm(k)
        
        b, s_q, _ = q.shape
        _, s_kv, _ = k.shape
        
        # Reshape to [B, H, S, D]
        q_h = q_norm.view(b, s_q, self.heads, self.dim_head).transpose(1, 2)
        k_h = k_norm.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)
        v_h = v.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)

        if q_rope is not None:
             q_cos, q_sin = q_rope
             q_h = apply_rotary_emb_pt(q_h, q_cos, q_sin)
        
        if k_rope is not None:
             k_cos, k_sin = k_rope
             k_h = apply_rotary_emb_pt(k_h, k_cos, k_sin)

        out = torch.nn.functional.scaled_dot_product_attention(q_h, k_h, v_h, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, s_q, -1)
        
        return self.to_out(out)

# ==========================================
# 2. JAX Imports
# ==========================================
from ..models.ltx2.attention_ltx2 import LTX2Attention, LTX2RotaryPosEmbed

class LTX2AttentionTest(unittest.TestCase):
    
    def setUp(self):
        # Common Parameters
        self.B, self.S, self.D = 1, 16, 64
        self.heads = 4
        self.dim_head = 16
        self.context_dim = 64
        
        torch.manual_seed(0)
        self.rng = nnx.Rngs(0)
        self.np_x = np.random.randn(self.B, self.S, self.D).astype(np.float32)

    def _init_and_sync_models(self, dtype=jnp.bfloat16):
        """Initializes PyTorch (CPU) and JAX (TPU) and syncs weights."""
        
        pt_dtype = torch.float32 if dtype == jnp.float32 else torch.bfloat16
        pt_model = PytorchLTX2Attention(self.D, self.context_dim, self.heads, self.dim_head)
        pt_model.to(device="cpu", dtype=pt_dtype) 
        pt_model.eval()

        jax_model = LTX2Attention(
            query_dim=self.D, heads=self.heads, dim_head=self.dim_head, context_dim=self.context_dim,
            rngs=self.rng, attention_kernel="dot_product",
            dtype=dtype
        )

        def to_jax_dtype(arr): return jnp.array(arr).astype(dtype)

        def copy_linear(jax_layer, pt_layer):
            w_pt = pt_layer.weight.detach().float().numpy().T
            b_pt = pt_layer.bias.detach().float().numpy()
            jax_layer.kernel[...] = to_jax_dtype(w_pt)
            jax_layer.bias[...] = to_jax_dtype(b_pt)
        
        def copy_norm(jax_layer, pt_layer):
            w_pt = pt_layer.weight.detach().float().numpy()
            jax_layer.scale[...] = to_jax_dtype(w_pt)

        copy_linear(jax_model.to_q, pt_model.to_q)
        copy_linear(jax_model.to_k, pt_model.to_k)
        copy_linear(jax_model.to_v, pt_model.to_v)
        copy_linear(jax_model.to_out, pt_model.to_out[0])
        copy_norm(jax_model.norm_q, pt_model.q_norm)
        copy_norm(jax_model.norm_k, pt_model.k_norm)
        
        return pt_model, jax_model

    # ------------------------------------------
    # 1. RoPE Frequency Parity Test
    # ------------------------------------------
    def test_rope_frequency_parity(self):
        """
        Verifies that LTX2RotaryPosEmbed (JAX) generates the EXACT same 
        frequencies as the PyTorch reference for a given input ID set.
        """
        dim = 60 # Divisible by 3 for 3D test
        
        rope_pt = PytorchLTX2RotaryPosEmbed(dim=dim)
        rope_jax = LTX2RotaryPosEmbed(dim=dim)
        
        # Create random IDs [B, S, 3]
        np_ids = np.random.randint(0, 100, (2, 16, 3)).astype(np.float32)
        
        # Run PyTorch
        pt_cos, pt_sin = rope_pt(torch.from_numpy(np_ids))
        
        # Run JAX
        jax_cos, jax_sin = rope_jax(jnp.array(np_ids))
        
        # Compare
        pt_cos_np = pt_cos.numpy()
        jax_cos_np = np.array(jax_cos)
        
        # 1e-5 tolerance for freq generation math
        np.testing.assert_allclose(pt_cos_np, jax_cos_np, atol=1e-5)
        np.testing.assert_allclose(pt_sin.numpy(), np.array(jax_sin), atol=1e-5)
        print("\n[PASS] RoPE Frequency Generation matches PyTorch.")

    # ------------------------------------------
    # 2. Strict Parity Test (Full Model)
    # ------------------------------------------
    def test_parity_bf16_strict(self):
        """Checks if JAX(TPU) matches PyTorch(CPU) in BF16."""
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.bfloat16)
        
        pt_in = torch.from_numpy(self.np_x).to(device="cpu", dtype=torch.bfloat16)
        jax_in = jnp.array(self.np_x).astype(jnp.bfloat16)

        with torch.no_grad():
            pt_out = pt_model(pt_in)
            
        jax_out = jax_model(jax_in)
        
        pt_res = pt_out.float().numpy()
        jax_res = np.array(jax_out, dtype=np.float32)

        np.testing.assert_allclose(
            pt_res, jax_res, atol=2e-2, rtol=1e-2,
            err_msg="BF16 Parity Failed"
        )
        print("\n[PASS] BF16 Strict Parity Test passed.")

    # ------------------------------------------
    # 3. Layer-wise Stats (Corrected Shape Logic)
    # ------------------------------------------
    def test_layer_wise_stats(self):
        """Prints diagnostic stats for every layer."""
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.bfloat16)
        
        pt_in = torch.from_numpy(self.np_x).to(device="cpu", dtype=torch.bfloat16)
        jax_in = jnp.array(self.np_x).astype(jnp.bfloat16)

        # 1. Run PyTorch Step-by-Step
        with torch.no_grad():
            # Projections
            pt_q = pt_model.to_q(pt_in)
            pt_k = pt_model.to_k(pt_in)
            pt_v = pt_model.to_v(pt_in)
            
            # Norms
            pt_qn = pt_model.q_norm(pt_q)
            pt_kn = pt_model.k_norm(pt_k)
            
            # Attention Prep (Reshape -> Transpose)
            b, s, _ = pt_qn.shape
            pt_q_h = pt_qn.view(b, s, self.heads, self.dim_head).transpose(1, 2)
            pt_k_h = pt_kn.view(b, s, self.heads, self.dim_head).transpose(1, 2)
            pt_v_h = pt_v.view(b, s, self.heads, self.dim_head).transpose(1, 2)
            
            # Attention Op
            pt_attn_out = torch.nn.functional.scaled_dot_product_attention(pt_q_h, pt_k_h, pt_v_h)
            
            # Reshape Back
            pt_attn_flat = pt_attn_out.transpose(1, 2).reshape(b, s, -1)
            
            # Output
            pt_out = pt_model.to_out(pt_attn_flat)

        # 2. Run JAX Step-by-Step
        jax_q = jax_model.to_q(jax_in)
        jax_k = jax_model.to_k(jax_in)
        jax_v = jax_model.to_v(jax_in)
        
        jax_qn = jax_model.norm_q(jax_q)
        jax_kn = jax_model.norm_k(jax_k)

        # Attention Op: Pass [B, S, Inner_Dim] directly
        # The LTX2Attention.__call__ flattens inputs before calling apply_attention, 
        # so we pass the flattened (Inner_Dim) tensors here.
        jax_attn = jax_model.attention_op.apply_attention(jax_qn, jax_kn, jax_v)
        jax_out = jax_model.to_out(jax_attn)

        # 3. Print Comparison Table
        stats = []
        def add_stat(name, pt_t, jax_t):
            pt_val = pt_t.float().numpy() if isinstance(pt_t, torch.Tensor) else pt_t
            jax_val = np.array(jax_t, dtype=np.float32)
            stats.append({
                "Layer": name,
                "PT Mean": f"{pt_val.mean():.4f}",
                "JAX Mean": f"{jax_val.mean():.4f}",
                "PT Min": f"{pt_val.min():.4f}",
                "JAX Min": f"{jax_val.min():.4f}",
                "PT Max": f"{pt_val.max():.4f}",
                "JAX Max": f"{jax_val.max():.4f}",
                "Diff (Mean L1)": f"{np.abs(pt_val - jax_val).mean():.6f}"
            })

        add_stat("Query Proj", pt_q, jax_q)
        add_stat("Key Proj", pt_k, jax_k)
        add_stat("Value Proj", pt_v, jax_v)
        add_stat("Query Norm", pt_qn, jax_qn)
        add_stat("Key Norm", pt_kn, jax_kn)
        add_stat("Attn Output", pt_attn_flat, jax_attn)
        add_stat("Final Output", pt_out, jax_out)
        
        df = pd.DataFrame(stats)
        print("\n[DIAGNOSTIC] Layer-wise Stats (CPU vs TPU BFloat16):")
        print(df.to_string(index=False))
    # ------------------------------------------
    # 4. Cross-Attention + RoPE Integration
    # ------------------------------------------
    def test_cross_attn_rope_integration(self):
        """Verifies Video->Audio cross-attention with RoPE."""
        S_Q, S_KV = 16, 20
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.float32)

        np_x = np.random.randn(self.B, S_Q, self.D).astype(np.float32)