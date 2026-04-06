# Potential Optimizations

## 1. Fix QKV Projection Sharding (Column-Parallel)

**Location:** `src/maxdiffusion/models/attention_flax.py` — `NNXAttentionBlock.__init__`

**Problem:**  
`hidden_states` last dim is constrained to `activation_heads` → `tensor` axis.  
Q/K/V kernels use `kernel_axes = ("embed", "heads")`, meaning the contracting dim maps to `embed` → `[context, fsdp]`.

This is a **sharding mismatch on the contracting dimension**: XLA must reshard before the matmul, adding unnecessary communication.

**Fix: Column-parallel QKV**  
Shard the weight *output* dim over `tensor` (heads), not the contracting dim. The output then naturally lands sharded over heads — exactly what attention needs — with no all-reduce in the forward pass.

```python
# Current (broken — contracting dim mismatch)
kernel_axes = ("embed", "heads")

# Fix: column-parallel — shard output over heads, leave contracting unrestricted
kernel_axes = (None, "heads")
# or if fsdp sharding of the contracting dim is desired:
kernel_axes = ("fsdp", "heads")
```

Also ensure `hidden_states` is not constrained to `activation_heads` on the embed dim before the QKV matmuls, or all-gather it first.

**Expected gain:** Eliminates a reshard/all-gather inserted by XLA on the contracting dimension for every Q, K, V projection in every attention block.
