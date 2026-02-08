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
from jax.sharding import Mesh
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

# Set JAX to use float32 for higher precision checks
jax.config.update("jax_default_matmul_precision", "float32")

# ==========================================
# 1. PyTorch Reference Implementations
# ==========================================

class PytorchLTX2RotaryPosEmbed(torch.nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, ids):
        num_axes = ids.shape[-1]
        dim_per_axis = self.dim // num_axes
        
        freq_indices = torch.arange(0, dim_per_axis, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (freq_indices / dim_per_axis))
        
        freqs_list = []
        for i in range(num_axes):
            axis_pos = ids[..., i] 
            freqs = torch.einsum('bs,d->bsd', axis_pos, inv_freq)
            freqs_list.append(freqs)
            
        # Concatenate axes -> [B, S, D/2]
        emb = torch.cat(freqs_list, dim=-1)
        
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        
        # Interleave: [c1, c2] -> [c1, c1, c2, c2]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        # Return [B, S, InnerDim] to match JAX/LTX-2 global RoPE
        return cos, sin


def apply_rotary_emb_pt(x, cos, sin):
    """
    Standard PyTorch Interleaved RoPE application.
    Dimension-agnostic: Works for [B, S, D] or [B, H, S, D].
    """
    # 1. Reshape last dim to pairs: [..., D] -> [..., D//2, 2]
    shape = x.shape
    x_reshaped = x.view(*shape[:-1], -1, 2)
    
    # 2. Rotate: [-x2, x1]
    x1, x2 = x_reshaped.unbind(-1)
    x_rotated = torch.stack((-x2, x1), dim=-1).view(*shape)
    
    # 3. Apply Frequencies (Float32 for parity)
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    rot_f32 = x_rotated.to(torch.float32)
    cos_f32 = cos.to(torch.float32)
    sin_f32 = sin.to(torch.float32)
    
    out = x_f32 * cos_f32 + rot_f32 * sin_f32
    return out.to(orig_dtype)


class PytorchLTX2Attention(torch.nn.Module):
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

    def forward(self, x, context=None, q_rope=None, k_rope=None, mask=None):
        q = self.to_q(x)
        ctx = x if context is None else context
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        # Keep raw projections for test_layer_wise_stats
        q_raw, k_raw = q, k

        q_normed = self.q_norm(q)
        k_normed = self.k_norm(k)
        
        if q_rope is not None:
             q_cos, q_sin = q_rope
             q_normed = apply_rotary_emb_pt(q_normed, q_cos, q_sin)
        
        if k_rope is not None:
             k_cos, k_sin = k_rope
             k_normed = apply_rotary_emb_pt(k_normed, k_cos, k_sin)

        # Split Heads for Attention
        b, s_q, _ = q_normed.shape
        _, s_kv, _ = k_normed.shape
        q_h = q_normed.view(b, s_q, self.heads, self.dim_head).transpose(1, 2)
        k_h = k_normed.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)
        v_h = v.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_h, k_h, v_h, attn_mask=mask, dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(b, s_q, -1)
        return self.to_out(out), (q_raw, k_raw, v, q_normed, k_normed, out)

# ==========================================
# 2. JAX Imports & Test Suite
# ==========================================
from ..models.ltx2.attention_ltx2 import LTX2Attention, LTX2RotaryPosEmbed

class LTX2AttentionTest(unittest.TestCase):
    
    def setUp(self):
        # S=128 is preferred for TPU Flash Attention block sizes
        self.B, self.S, self.D = 1, 128, 64
        self.heads = 4
        self.dim_head = 16
        self.context_dim = 64
        
        torch.manual_seed(0)
        self.rng = nnx.Rngs(0)
        self.np_x = np.random.randn(self.B, self.S, self.D).astype(np.float32)

    def _init_and_sync_models(self, dtype=jnp.bfloat16):
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

    def test_shapes(self):
        model = LTX2Attention(64, 4, 16, 64, rngs=self.rng, attention_kernel="dot_product")
        
        x_vid = jnp.zeros((1, 128, 64))
        out_vid = model(x_vid)
        self.assertEqual(out_vid.shape, (1, 128, 64))
        
        x_aud = jnp.zeros((1, 32, 64))
        out_cross = model(x_vid, encoder_hidden_states=x_aud)
        self.assertEqual(out_cross.shape, (1, 128, 64)) 
        print("\n[PASS] Shape Tests Passed.")

    def test_rope_frequency_parity(self):
        dim = 60
        rope_pt = PytorchLTX2RotaryPosEmbed(dim=dim)
        rope_jax = LTX2RotaryPosEmbed(dim=dim)
        
        np_ids = np.random.randint(0, 100, (2, 16, 3)).astype(np.float32)
        pt_cos, pt_sin = rope_pt(torch.from_numpy(np_ids))
        jax_cos, jax_sin = rope_jax(jnp.array(np_ids))
        
        np.testing.assert_allclose(pt_cos.numpy(), np.array(jax_cos), atol=1e-5)
        np.testing.assert_allclose(pt_sin.numpy(), np.array(jax_sin), atol=1e-5)
        print("[PASS] RoPE Frequency Parity Verified.")

    def test_parity_bf16_strict(self):
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.bfloat16)
        
        pt_in = torch.from_numpy(self.np_x).to(device="cpu", dtype=torch.bfloat16)
        jax_in = jnp.array(self.np_x).astype(jnp.bfloat16)

        with torch.no_grad():
            pt_out, _ = pt_model(pt_in)
        
        jax_out = jax_model(jax_in)
        
        pt_res = pt_out.float().numpy()
        jax_res = np.array(jax_out, dtype=np.float32)

        np.testing.assert_allclose(
            pt_res, jax_res, atol=2e-2, rtol=1e-2,
            err_msg="BF16 Parity Failed"
        )
        print("\n[PASS] BF16 Strict Parity Test passed.")

    def test_layer_wise_stats(self):
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.bfloat16)
        
        pt_in = torch.from_numpy(self.np_x).to(device="cpu", dtype=torch.bfloat16)
        jax_in = jnp.array(self.np_x).astype(jnp.bfloat16)

        with torch.no_grad():
             pt_out, (pt_q, pt_k, pt_v, pt_qn, pt_kn, pt_attn) = pt_model(pt_in)

        jax_q = jax_model.to_q(jax_in)
        jax_k = jax_model.to_k(jax_in)
        jax_v = jax_model.to_v(jax_in)
        jax_qn = jax_model.norm_q(jax_q)
        jax_kn = jax_model.norm_k(jax_k)

        jax_attn = jax_model.attention_op.apply_attention(jax_qn, jax_kn, jax_v)
        jax_out = jax_model.to_out(jax_attn)

        stats = []
        def add_stat(name, pt_t, jax_t):
            if isinstance(pt_t, torch.Tensor):
                pt_val = pt_t.float().numpy()
            else:
                pt_val = pt_t
            jax_val = np.array(jax_t, dtype=np.float32)
            stats.append({
                "Layer": name,
                "PT Max": f"{pt_val.max():.4f}",
                "JAX Max": f"{jax_val.max():.4f}",
                "PT Mean": f"{pt_val.mean():.4f}",
                "JAX Mean": f"{jax_val.mean():.4f}",
                "PT Min": f"{pt_val.min():.4f}",
                "JAX Min": f"{jax_val.min():.4f}",
                "Diff (L1)": f"{np.abs(pt_val - jax_val).mean():.6f}"
            })

        add_stat("Query Proj", pt_q, jax_q)
        add_stat("Key Proj", pt_k, jax_k)
        add_stat("Value Proj", pt_v, jax_v)
        add_stat("Query Norm", pt_qn, jax_qn)
        add_stat("Key Norm", pt_kn, jax_kn)
        add_stat("Attn Output", pt_attn, jax_attn)
        add_stat("Final Output", pt_out, jax_out)
        
        df = pd.DataFrame(stats)
        print("\n[DIAGNOSTIC] Layer-wise Stats:")
        print(df.to_string(index=False))

    def test_cross_attn_rope_integration(self):
        S_Q, S_KV = 16, 20
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.float32)

        np_x = np.random.randn(self.B, S_Q, self.D).astype(np.float32)
        np_ctx = np.random.randn(self.B, S_KV, self.D).astype(np.float32)
        
        rope_gen_pt = PytorchLTX2RotaryPosEmbed(dim=64) # Gen [B, S, InnerDim]
        
        ids_q = torch.randint(0, 100, (self.B, S_Q, 1))
        ids_k = torch.randint(0, 100, (self.B, S_KV, 1))
        
        q_cos_pt, q_sin_pt = rope_gen_pt(ids_q.float())
        k_cos_pt, k_sin_pt = rope_gen_pt(ids_k.float())

        with torch.no_grad():
            pt_out, _ = pt_model(
                torch.from_numpy(np_x), 
                context=torch.from_numpy(np_ctx),
                q_rope=(q_cos_pt, q_sin_pt),
                k_rope=(k_cos_pt, k_sin_pt)
            )

        jax_q_rope = (jnp.array(q_cos_pt.numpy()), jnp.array(q_sin_pt.numpy()))
        jax_k_rope = (jnp.array(k_cos_pt.numpy()), jnp.array(k_sin_pt.numpy()))

        jax_out = jax_model(
            jnp.array(np_x),
            encoder_hidden_states=jnp.array(np_ctx),
            rotary_emb=jax_q_rope,
            k_rotary_emb=jax_k_rope
        )

        diff = np.abs(pt_out.numpy() - np.array(jax_out)).max()
        print(f"\n[Cross-Attn + RoPE] Max Diff: {diff:.6f}")
        np.testing.assert_allclose(pt_out.numpy(), np.array(jax_out), atol=1e-5)
        print("[PASS] Cross-Attention with RoPE Parity Verified.")

    def test_attention_mask_parity(self):
        S_flash = 512 
        np_x = np.random.randn(self.B, S_flash, self.D).astype(np.float32)
        pt_model, jax_model = self._init_and_sync_models(dtype=jnp.float32)
        
        devices = jax.devices()
        mesh = Mesh(np.array(devices).reshape(1, -1), ('data', 'context'))
        
        jax_model.attention_op.attention_kernel = "flash"
        jax_model.attention_op.mesh = mesh
        jax_model.attention_op.flash_block_sizes = splash_attention_kernel.BlockSizes(
            block_q=128, block_kv_compute=128, block_kv=128,
            block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128,
            block_q_dq=128, block_kv_dq=128,
        )

        mask_pattern_np = np.random.randint(0, 2, (self.B, S_flash)).astype(np.float32)
        pt_mask_additive = torch.from_numpy((1.0 - mask_pattern_np) * -1e9)[:, None, None, :]
        jax_mask_multiplicative = jnp.array(mask_pattern_np)

        with torch.no_grad():
            pt_out, _ = pt_model(torch.from_numpy(np_x), mask=pt_mask_additive)

        with mesh:
             jax_out = jax_model(
                jnp.array(np_x),
                attention_mask=jax_mask_multiplicative
            )

        diff = np.abs(pt_out.numpy() - np.array(jax_out)).max()
        print(f"\n[Mask Parity] Max Diff (Flash): {diff:.6f}")
        np.testing.assert_allclose(pt_out.numpy(), np.array(jax_out), atol=1e-4)
        print("[PASS] Attention Mask Parity Verified.")

if __name__ == "__main__":
    unittest.main()