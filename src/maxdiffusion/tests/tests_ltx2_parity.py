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

# --- 1. Reference PyTorch Model (Minimal LTX-2 Logic) ---
class PytorchLTX2Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        
        # LTX-2: RMSNorm on full inner_dim
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6)

        # LTX-2: Linear layers with bias=True
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, bias=True), 
            torch.nn.Identity()
        )

    def forward(self, x, context=None):
        q = self.to_q(x)
        ctx = x if context is None else context
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        # Norms (Key check for LTX-2 vs Flux)
        q_norm = self.q_norm(q)
        k_norm = self.k_norm(k)
        
        # Reshape
        b, s_q, _ = q.shape
        _, s_kv, _ = k.shape
        
        q_h = q_norm.view(b, s_q, self.heads, self.dim_head).transpose(1, 2)
        k_h = k_norm.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)
        v_h = v.view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)

        # Attention
        out = torch.nn.functional.scaled_dot_product_attention(q_h, k_h, v_h, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, s_q, -1)
        
        return self.to_out(out), (q, k, v, q_norm, k_norm, out) # Return intermediates

# --- 2. Import JAX Model ---
from ..models.ltx2.attention_ltx2 import LTX2Attention

class LTX2ParityTest(unittest.TestCase):
    
    def setUp(self):
        self.B, self.S, self.D = 1, 16, 64
        self.heads = 4
        self.dim_head = 16
        self.context_dim = 64
        
        torch.manual_seed(0)
        self.rng = nnx.Rngs(0)
        
        # Inputs
        self.np_x = np.random.randn(self.B, self.S, self.D).astype(np.float32)

    def _init_and_sync_models(self):
        """Initializes both models and copies PyTorch weights to JAX."""
        pt_model = PytorchLTX2Attention(self.D, self.context_dim, self.heads, self.dim_head)
        pt_model.eval()

        jax_model = LTX2Attention(
            query_dim=self.D, heads=self.heads, dim_head=self.dim_head, context_dim=self.context_dim,
            rngs=self.rng, attention_kernel="dot_product"
        )

        # Weight Copy Logic
        def copy_linear(jax_layer, pt_layer):
            jax_layer.kernel.value = jnp.array(pt_layer.weight.detach().numpy().T)
            jax_layer.bias.value = jnp.array(pt_layer.bias.detach().numpy())
        
        def copy_norm(jax_layer, pt_layer):
            jax_layer.scale.value = jnp.array(pt_layer.weight.detach().numpy())

        copy_linear(jax_model.to_q, pt_model.to_q)
        copy_linear(jax_model.to_k, pt_model.to_k)
        copy_linear(jax_model.to_v, pt_model.to_v)
        copy_linear(jax_model.to_out, pt_model.to_out[0])
        copy_norm(jax_model.norm_q, pt_model.q_norm)
        copy_norm(jax_model.norm_k, pt_model.k_norm)
        
        return pt_model, jax_model

    def test_parity_strict(self):
        """Standard Parity Test (Assertion)."""
        pt_model, jax_model = self._init_and_sync_models()
        
        with torch.no_grad():
            pt_out, _ = pt_model(torch.from_numpy(self.np_x))
            
        jax_out = jax_model(jnp.array(self.np_x))
        
        np.testing.assert_allclose(
            pt_out.numpy(), jax_out, atol=1e-5, 
            err_msg="Strict Parity Failed: Outputs mismatch > 1e-5"
        )
        print("\n[PASS] Strict Parity Test passed.")

    def test_layer_wise_stats(self):
        """Diagnostic Test: Prints Layer-wise stats."""
        pt_model, jax_model = self._init_and_sync_models()
        
        # 1. Run PyTorch (Get Intermediates)
        with torch.no_grad():
            pt_out, (pt_q, pt_k, pt_v, pt_qn, pt_kn, pt_attn) = pt_model(torch.from_numpy(self.np_x))

        # 2. Run JAX Step-by-Step (Manual Re-run to get intermediates)
        x = jnp.array(self.np_x)
        jax_q = jax_model.to_q(x)
        jax_k = jax_model.to_k(x) # Self-attn
        jax_v = jax_model.to_v(x)
        
        jax_qn = jax_model.norm_q(jax_q)
        jax_kn = jax_model.norm_k(jax_k)

        # JAX Reshape & Attn
        b, s, _ = jax_qn.shape
        q_h = jax_qn.reshape(b, s, self.heads, self.dim_head).reshape(b, s, -1)
        k_h = jax_kn.reshape(b, s, self.heads, self.dim_head).reshape(b, s, -1)
        v_h = jax_v.reshape(b, s, self.heads, self.dim_head).reshape(b, s, -1)
        
        jax_attn = jax_model.attention_op.apply_attention(q_h, k_h, v_h)
        jax_out = jax_model.to_out(jax_attn)

        # 3. Compare Stats
        stats = []
        def add_stat(name, pt_t, jax_t):
            pt_val = pt_t.numpy() if isinstance(pt_t, torch.Tensor) else pt_t
            jax_val = np.array(jax_t)
            stats.append({
                "Layer": name,
                "PT Mean": f"{pt_val.mean():.4f}",
                "JAX Mean": f"{jax_val.mean():.4f}",
                "Diff (Mean L1)": f"{np.abs(pt_val - jax_val).mean():.6f}"
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

if __name__ == "__main__":
    unittest.main()