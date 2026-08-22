# Code Review — `kernel_improv` (1d71782)

**Commit**: `feat(attention,wan): implement fixed-m splash and 2D ring attention with Wan2.2 optimizations`
**Author**: Rishabh Manoj — **2274 insertions, 488 deletions across 20 files**

---

## Executive Summary

This is a **large, multi-concern commit** touching four independent feature areas that should have been separate PRs:

1. **Per-Q-block fixed-m gating** in the Pallas splash kernel + ring attention
2. **Global Virtual K-Centering** (k_mean recentering across ring shards)
3. **Fused producers** for Wan transformer (LN+AdaLN, RMSNorm+RoPE, fused QKV GEMM)
4. **Wan infrastructure** (AOT cache revision safety, video export, batch-fold GQA guards)

The kernel math is **correct and well-derived**. The Pallas kernel changes are competent. But the integration layer is a **swamp** — and several issues would silently produce wrong results or blow up on real multi-host runs.

---

## 🔴 Critical Bugs

### 1. Fused QKV GEMM bypasses sharding constraints and breaks GQA

[attention_flax.py:L2879-L2918](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2879-L2918)

The self-attention path now manually concatenates `q_w, k_w, v_w` into a single `qkv_w` and runs `lax.dot_general` directly:
```python
qkv_w = jnp.concatenate([q_w, k_w, v_w], axis=-1)
qkv = jax.lax.dot_general(hidden_states, qkv_w, ...)
```

**Problems:**
- **This silently destroys sharding**: `self.query.kernel` and `self.key.kernel` carry NamedSharding from `nnx.Linear`'s parameter spec. Concatenating them along axis -1 produces an array whose sharding is either replicated or wrong. The `lax.dot_general` then runs on a potentially replicated weight on every device — **2-3x HBM blow-up and all-wrong FSDP gradients**.
- **GQA is broken**: When `kv_heads != q_heads`, `k_w` and `v_w` have shape `(D, kv_heads * dim_head)` while `q_w` has shape `(D, q_heads * dim_head)`. The concat works, but the `jnp.split` afterwards uses `[dim_q, dim_q + dim_k]` — **this is correct only if the split point arithmetic matches**. It does here, but there's zero runtime validation. One misconfigured `kv_heads` and you get silent garbage.
- **The original `self.query(hidden_states)` call goes through `nnx.Linear.__call__` which applies `lax.dot_general` with the module's precision spec**. The manual GEMM hardcodes `precision=self.precision` — which is only set in this commit (L2668) and could be `None`, changing behavior vs. the original.

> [!CAUTION]
> This will produce silently wrong results under FSDP. The fused GEMM must preserve the original weight sharding via `jax.lax.with_sharding_constraint` on `qkv_w`, or use `shard_map` to keep the matmul local.

### 2. `k_mean` auto-reduction in the ring path happens AFTER the Ulysses A2A — but the kernel consumes it PER KV-HEAD SHARD

[ring_attention_kernel.py:L915-L918](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L915-L918)

```python
k_mean_local = jnp.mean(k.astype(jnp.float32), axis=1)
k_mean = lax.pmean(k_mean_local, ring_axis)
```

This computes the mean over the **ring-local shard** only (axis=1 is the sequence dim). Then `pmean` averages across ring ranks. This gives you `E[mean_per_shard(k)]` which is only equal to `mean_global(k)` if every shard has the same sequence length. If there's padding, the means are biased by the zero-padded tokens.

In [attention_flax.py:L1522](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1522), the Ulysses path correctly computes on `real_key[:, :, :actual_kv_seq_len, :]` — **but the ring kernel path uses `k` which includes padding**. Under asymmetric padding the centering guarantee breaks and the fixed-m bound becomes unsound.

### 3. `_write_fixed_m` always executes `k_mean_ref` branch — the `None` guard is wrong

[custom_splash_attention.py:L155-L166](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L155-L166)

```python
def _write_fixed_m():
    m_base = mk_ref[0, h, i]
    if k_mean_ref is not None:  # ← this is ALWAYS True
```

`k_mean_ref` is a Pallas `Ref` (a `BlockSpec`-backed HBM reference). It is **never `None`** inside the kernel body — you always pass a `k_mean` operand (defaulting to zeros). So `if k_mean_ref is not None` is a dead branch and the `else: m_fixed = m_base` path never executes. The overhead is small (one dot product per Q-block), but it's misleading and should be gated by a `functools.partial` compile-time flag like `use_k_centering: bool`, not a ref-is-None check.

---

## 🟡 Significant Design Issues

### 4. `make_custom_ring_attention` — keyword-only `*` silently removed

[ring_attention_kernel.py:L1190](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L1190)

The diff shows `*` was removed from the function signature:
```diff
-def make_custom_ring_attention(
-    *,
+def make_custom_ring_attention(
```

All current callers use keyword args, so nothing breaks **today**. But this is a public factory function and the `*` existed to enforce keyword-only calling convention for clarity. Removing it is a gratuitous API regression.

### 5. `fused_rmsnorm_rope` RMSNorm is computed over the FULL flattened feature dim — not per-head

[fused_producers.py:L72-L74](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/fused_producers.py#L72-L74)

```python
q_rms = jax.lax.rsqrt(jnp.mean(jnp.square(q_fp32), axis=-1, keepdims=True) + eps)
q_norm = (q_fp32 * q_rms * q_norm_scale.astype(jnp.float32)).astype(raw_q.dtype)
```

`raw_q` has shape `[B, Sq, q_heads * dim_head]`. The RMSNorm is computed over the last axis, which is `q_heads * dim_head = 5120` for Wan 14B. The original `self.norm_q` is an `nnx.RMSNorm` with `num_features=dim_head` (128), applied **after** reshaping to per-head `[B, heads, Sq, dim_head]`. So the fused version normalizes over a 40x wider feature dimension. **This changes the normalization statistics and is mathematically not equivalent.**

> [!WARNING]
> The fused RMSNorm must be computed per-head (reshape to `[B, Sq, heads, dim_head]`, normalize over axis=-1, then reshape back) to match the original module's semantics.

### 6. Inline `from maxdiffusion.kernels.fused_producers import ...` inside `__call__`

[transformer_wan.py:L489](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L489), [transformer_wan.py:L532](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L532), [transformer_wan.py:L940](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L940)

Three inline `from ... import fused_ln_adaln` inside hot-path `__call__` methods. Python `import` uses a lock + dict lookup; inside a traced JAX function this is harmless for correctness but terrible for readability and breaks every linter's import-at-top rule. Move them to module level.

### 7. `is_self_attention` detection changed semantically

[attention_flax.py:L2867](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2867)

```python
is_self_attention = self.is_self_attention if hasattr(self, "is_self_attention") else (encoder_hidden_states is None)
```

The original code dynamically checked `encoder_hidden_states is None`. Now it reads a stored attribute. If the module is incorrectly initialized with `is_self_attention=True` but called with non-None `encoder_hidden_states`, the fused QKV path fires on cross-attention inputs, which is wrong. The `hasattr` fallback shows this was a bolt-on, not a planned design.

### 8. Cross-attention residual cast changed

[transformer_wan.py:L519](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L519)

```diff
-  hidden_states = hidden_states + attn_output
+  hidden_states = (hidden_states.astype(jnp.float32) + attn_output.astype(jnp.float32)).astype(hidden_states.dtype)
```

This changes the numerical behavior of cross-attention residuals (previously BF16 add, now FP32 add then downcast). This is probably an improvement for precision, but it's a **silent numerical diff** that should be called out and measured, not snuck in alongside kernel work.

---

## 🟢 What's Good

### 9. Fixed-m derivation and per-Q-block gating

[custom_splash_attention.py:L54-L88](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L54-L88)

`get_fixed_m_constants` is **clean, well-documented, and mathematically sound**. The dynamic `C(N) = 127 - ceil(log2(N)) - 8` derivation with the ring-specific halved safe window is correct. The per-Q-block extension that lets individual query tiles fall back to online softmax while the rest stay pinned is a genuinely good idea — it maximizes the fraction of tiles that avoid the rescale-and-divide overhead.

### 10. Squared-norm gating in the ring path

Moving from `qn * mk <= bound` to `qn_sq * mk_sq <= bound_sq` eliminates a `sqrt` from the hot-path gating check. Clean optimization.

### 11. `jnp.concatenate` instead of `jnp.stack` for RoPE

```diff
-  xq_out = jnp.stack([xq_out_0, xq_out_1], axis=-1).reshape(xq.shape)
+  xq_out = jnp.concatenate([xq_out_0[..., None], xq_out_1[..., None]], axis=-1).reshape(xq.shape)
```

This avoids a potential layout fragmentation on TPU — `stack` introduces a new axis that Mosaic may tileize differently from the final reshape target. `concatenate` keeps the rank stable. Good micro-optimization.

### 12. AOT cache revision safety

[generate_wan.py:L41-L60](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/generate_wan.py#L41-L60)

`_resolve_wan_aot_source_revision` / `_is_reusable_aot_revision` / `_non_reusable_aot_revision` form a clean guard against stale AOT cache hits during development. Using `uuid4().hex` as a non-reusable sentinel is correct and simple.

### 13. Test coverage

The test additions are **extensive** — 420+ lines for per-Q-block splash tests, 560+ lines for ring tests covering per-Q-block, GQA, multi-ring-size, and mixed eligibility. The tests properly compare against FP32 dense references and check tolerances. This is the strongest part of the commit.

---

## 🔵 Nits

| Location | Issue |
|---|---|
| [fused_producers.py:L3](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/fused_producers.py#L3) | `from typing import Tuple` — use `tuple` (Python 3.9+), `Tuple` is deprecated |
| [fused_producers.py:L35](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/fused_producers.py#L35) | `heads: int | None = None` deprecated alias param is dead weight in a brand new function. Just remove it. |
| [ring_attention_kernel.py:L927](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L927) | `jnp.repeat(k_mean, q_heads_per_kv_head, axis=0)` — this materializes a `(num_q_heads, D)` array when you only need the GQA broadcast at index time. The kernel already does `h // q_heads_per_kv_head`. Wasteful. |
| [generate_wan.py:L399](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/generate_wan.py#L399) | `import numpy as np` inside the `if jax.process_index() == 0` block — lazy imports inside conditional blocks make testing and static analysis harder |
| [attention_flax.py:L1084](file:///c:/Users/Window%2011/Miscellaneous/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1084) | `num_q_heads == num_kv_heads` guard on `fold_batch` — uses `num_q_heads` but this variable doesn't exist in this scope; the surrounding code uses `num_heads`. This is either a bug or a newly introduced variable I missed. |
| Several files | `# pylint: disable=protected-access` — there are 6+ of these accessing `custom_splash._` internals. Either make the API public or add proper re-exports. |

---

## Verdict

The kernel math is solid. The per-Q-block fixed-m and virtual k-centering are legitimate performance improvements for long-sequence diffusion attention.

**But the integration is rushed.** The fused QKV GEMM breaks sharding, the fused RMSNorm has wrong normalization axes, and several changes are smuggled in as "optimizations" without measuring or documenting the numerical impact. This needs to be split into at least 3 separate PRs with the FSDP and RMSNorm bugs fixed before landing.

---

## Recommended PR Split

### PR 1 — Per-Q-block fixed-m kernel + ring attention (pure kernel layer)

**Files:**
- `kernels/custom_splash_attention.py` — `get_fixed_m_constants`, `_flash_attention_kernel` (per-Q-block mk, k_mean_ref, fixed_m_recenter), `_splash_attention_forward`, `_splash_attention_forward_ring`, `make_splash_mha`
- `kernels/splash_attention/ring_attention_kernel.py` — squared-norm gating, `per_q_block` flag, `pregathered_mk`, Global Virtual K-Centering (`k_mean` pmean + centered Cauchy-Schwarz), accumulate vs LSE-merge `lax.cond`
- `models/attention_flax.py` — **only** `_compute_fixed_m_metadata` (per-Q-block path), new kernel registrations (`ulysses_ring_custom_fixed_m_per_q_block`, `ulysses_custom_fixed_m_per_q_block`), and the plumbing that passes `k_mean`/`per_q_block`/`kv_heads` through `_ulysses_attention` and `_ulysses_ring_custom_attention`
- `tests/custom_splash_fixed_m_test.py` — all new per-Q-block tests
- `tests/ring_fixed_m_test.py` — all new ring tests
- Config YAML changes (`base_wan_*.yml` attention kernel name updates)

**Bug fixes to include:**
- Fix `k_mean` padding bias: compute on `real_key[:, :, :actual_kv_seq_len, :]`, not padded `k`
- Replace the dead `if k_mean_ref is not None` in `_write_fixed_m` with a compile-time `use_k_centering: bool` partial flag
- Restore the `*` keyword-only marker on `make_custom_ring_attention`

---

### PR 2 — Wan fused producers (LN+AdaLN, RMSNorm+RoPE, fused QKV) — depends on PR 1

**Files:**
- `kernels/fused_producers.py` — **new file**, `fused_ln_adaln` and `fused_rmsnorm_rope`
- `models/wan/transformers/transformer_wan.py` — replace inline `norm1/norm3` + AdaLN with `fused_ln_adaln`, cross-attention FP32 residual cast, GELU scope fix, `proj_out` scope fix
- `models/attention_flax.py` — **only** the `FlaxWanAttention.__call__` changes: fused QKV GEMM path, `fused_rmsnorm_rope` integration, `is_self_attention` attribute, `concat` vs `stack` RoPE fix

**Bug fixes to include:**
- **Fix `fused_rmsnorm_rope`**: reshape to `[B, Sq, heads, dim_head]`, RMSNorm over `axis=-1` (dim_head), then reshape back — must match original per-head `nnx.RMSNorm`
- **Fix fused QKV GEMM sharding**: apply `jax.lax.with_sharding_constraint` on `qkv_w`, or drop manual GEMM and use `jnp.concatenate([self.query(x), self.key(x), self.value(x)])` to let XLA fuse while preserving sharding
- Move `from ... import fused_ln_adaln` to module-level imports
- Drop the `heads: int | None = None` deprecated alias — new function, nothing to deprecate
- Fix `from typing import Tuple` → `tuple`
- Document cross-attention FP32 residual cast with before/after PSNR measurement

---

### PR 3 — Wan infrastructure (AOT cache, video export, config) — independent

**Files:**
- `generate_wan.py` — AOT revision safety, video export guarded by `process_index() == 0`, `output_dir` path handling
- `aot_cache.py`, `max_utils.py`, `pyconfig.py`, `utils/export_utils.py`
- `tests/profiler_test.py`, `end_to_end/tpu/run_wan_fast_inference.sh`

**Fix:** Move `import numpy as np` to module level.

---

### Dependency Graph

```
PR 3 (infra) ─── independent, land anytime
PR 1 (kernel) ── land first
PR 2 (Wan fused) ── depends on PR 1 (uses k_mean, per_q_block, kv_heads plumbing)
```
