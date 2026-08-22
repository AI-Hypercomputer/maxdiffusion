# `kernel_improv` Branch Review

Reviewed against merge base `a327ebd4a99eba208f5a79127f7e637788b396ec`.

Commits in scope:

- `1d71782611d03c912eadc38de99cabe26ee21da3` - fixed-m Splash Attention, 2D ring attention, WAN fused producers, and WAN AOT integration.
- `199905ca585d808ea25e66bbfacfe69ed922aea1` - branch review document.

## Verdict

**Do not merge this branch in its current form.**

The branch contains supported-configuration crashes, incorrect GQA fallback behavior, unsafe persistent executable-cache identity, and unvalidated fixed-m numerical assumptions. The TPU performance claims also lack the multi-device HLO and profile evidence required to distinguish an actual optimization from a hidden reshard.

Any P1 finding below is independently merge-blocking.

## P1 findings

### 1. Fixed-m crashes when the attention head dimension is below 128

The integration computes `k_mean` from the unpadded key tensor and then pads Q/K feature dimensions to 128:

- [`attention_flax.py:928`](src/maxdiffusion/models/attention_flax.py#L928)
- [`attention_flax.py:930`](src/maxdiffusion/models/attention_flax.py#L930)

The custom Splash wrapper subsequently requires the unpadded mean width to equal the padded Q/K width:

- [`custom_splash_attention.py:568`](src/maxdiffusion/kernels/custom_splash_attention.py#L568)

This is a production-supported case, not a theoretical API corner. LTX2 audio defaults to a 64-wide attention head:

- [`transformer_ltx2.py:789`](src/maxdiffusion/models/ltx2/transformer_ltx2.py#L789)

Selecting `ulysses_custom_fixed_m` or its ring variant for LTX2 audio therefore raises before the Pallas kernel launches.

**Required fix:** pad `k_mean` consistently with K, or compute it after feature padding while still excluding padded sequence tokens. Add an integration test through `AttentionOp` with `head_dim=64`; the existing low-dimensional kernel test bypasses the failing production plumbing.

### 2. GQA/MQA is broken when Ulysses falls back to dot-product attention

The dispatcher sends short Ulysses workloads to dot-product attention:

- [`attention_flax.py:2142`](src/maxdiffusion/models/attention_flax.py#L2142)

The dot-product implementation reshapes Q, K, and V using the query-head count and does not receive `kv_heads`:

- [`attention_flax.py:1649`](src/maxdiffusion/models/attention_flax.py#L1649)
- [`attention_flax.py:1663`](src/maxdiffusion/models/attention_flax.py#L1663)

For example, with `Hq=8`, `Hkv=2`, and a sequence shorter than `flash_min_seq_length`, K/V are either reinterpreted with incorrect sequence/head dimensions or the reshape/dot fails. The new GQA test exercises fused RMSNorm/RoPE only; it never covers dispatcher fallback.

**Required fix:** pass `kv_heads` into the fallback, reshape K/V with that value, and implement the intended grouped-query head broadcast. Test both `split_head_dim` modes below the flash threshold.

### 3. Dirty source can reuse a stale WAN AOT executable

`generate_wan.run(..., commit_hash=...)` ignores its supplied revision and detects another revision internally:

- [`generate_wan.py:299`](src/maxdiffusion/generate_wan.py#L299)

The helper only runs `git rev-parse HEAD` from the process working directory:

- [`max_utils.py:362`](src/maxdiffusion/max_utils.py#L362)

It does not:

- anchor Git to the MaxDiffusion source checkout;
- detect modified tracked source;
- detect untracked source files; or
- return the `dirty:` identity expected by `_is_reusable_aot_revision`.

Consequently, editing a transformer or kernel without committing leaves the same reusable cache identity. Running MaxDiffusion from another Git repository can key the cache with that unrelated repository's HEAD. The `aot_build_revision` fallback can also be shadowed by this unrelated detection.

**Required fix:** use the LTX2 revision implementation as the baseline: resolve the package source root, invoke `git -C`, inspect tracked and relevant untracked source, honor the supplied `commit_hash`, and disable persistent reuse for dirty/unversioned trees.

### 4. The WAN AOT fingerprint omits graph-defining configuration

The persistent metadata currently excludes `ulysses_shards`, `ulysses_attention_chunks`, `use_base2_exp`, and the experimental scheduler setting:

- [`generate_wan.py:309`](src/maxdiffusion/generate_wan.py#L309)

The fast-inference script exposes Ulysses topology as a runtime knob while reusing the same per-model cache directory:

- [`run_wan_fast_inference.sh:75`](end_to_end/tpu/run_wan_fast_inference.sh#L75)
- [`run_wan_fast_inference.sh:98`](end_to_end/tpu/run_wan_fast_inference.sh#L98)

These values change the traced collective/kernel graph without necessarily changing input shapes. Graph/module configuration is intentionally absent from the dynamic signature, so a run with `ULYSSES_SHARDS=4` can select an executable serialized for `ULYSSES_SHARDS=2`.

**Required fix:** define one canonical, tested AOT metadata builder containing every static graph/compiler input. Add a test proving that changing each field changes the fingerprint.

### 5. The AOT fast signature cache aliases incompatible scalar avals

The new fast key records array shape/dtype but maps every non-array dynamic leaf to `None`:

- [`aot_cache.py:201`](src/maxdiffusion/aot_cache.py#L201)

It therefore omits Python scalar type and JAX weak type. Calling an AOT-wrapped function first with `scale=1` and then with `scale=1.0` selects the same cached signature even though the compiled executable expects different scalar avals.

The direct compiled call is outside the existing failure/fallback handler:

- [`aot_cache.py:226`](src/maxdiffusion/aot_cache.py#L226)

The result is an uncaught executable input mismatch instead of the advertised silent JIT fallback.

**Required fix:** include scalar type/value descriptors and array `weak_type` in the fast key, or cache the complete `_dynamic_signature` inputs. Put the fast compiled call behind the same safe fallback path and add int/float/weak-scalar regression tests.

### 6. Fixed-m's numerical safety proof relies on an unenforced value bound

The dynamic recentering derivation assumes `abs(V) <= 256`:

- [`custom_splash_attention.py:51`](src/maxdiffusion/kernels/custom_splash_attention.py#L51)

Neither the factory nor eligibility metadata checks this precondition. A minimal counterexample is:

- `N = 4096`
- `Q = 0`
- `K = 0`
- `V = 512`

The fixed path is eligible and chooses `C=107`. Its numerator reaches:

```text
4096 * 512 * 2^107 = 2^128
```

That overflows FP32 even though exact attention returns 512. Float16 inputs fail at much smaller values because the exponential weights are narrowed to `q_ref.dtype` before the output dot:

- [`custom_splash_attention.py:222`](src/maxdiffusion/kernels/custom_splash_attention.py#L222)

**Required fix:** either gate fixed-m using `max(abs(V))`, derive a conservative constant for the supported dtype/value domain, or validate and document a hard input contract. Add adversarial value/dtype tests rather than testing the algebra under its own assumed bound.

### 7. Fused QKV is not demonstrated to be a tensor-parallel TPU optimization

The WAN self-attention path concatenates three independently `("embed", "heads")`-sharded `D x D` kernels into `D x 3D`, performs one dot, and splits the result:

- [`attention_flax.py:2885`](src/maxdiffusion/models/attention_flax.py#L2885)

Contiguous partitions of the new `3D` axis do not match each projection's existing tensor-parallel shards. On TP>1, one of two things must happen:

1. XLA preserves the concatenation and introduces communication/materialization; or
2. XLA algebraically decomposes it back into separate projection dots, eliminating the claimed fusion.

The current single-device parity test cannot detect either outcome.

**Required evidence before merge:** dumped HLO showing the final partitioned program, communication-volume comparison, and an end-to-end profile on the target TPU topology. A performance PR without multi-device evidence is speculation wearing a benchmark costume.

## P2 findings

### 8. Public custom-Splash factory defaults now raise

`make_splash_mha` still declares `orig_kv_seq_len=None`, but it unconditionally computes fixed-m constants even for the online kernel:

- [`custom_splash_attention.py:872`](src/maxdiffusion/kernels/custom_splash_attention.py#L872)
- [`custom_splash_attention.py:888`](src/maxdiffusion/kernels/custom_splash_attention.py#L888)

`make_splash_mha(block_sizes)(q, k, v)` worked at the merge base and now raises during factory construction.

**Required fix:** compute constants only when fixed-m is enabled, or defer length inference until invocation.

### 9. The public ring fixed-m metadata contract is inconsistent with its implementation

The factory documents `fixed_m_norms=(qn_max, mk_h)`:

- [`ring_attention_kernel.py:1211`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L1211)

The implementation now consumes squared norms, and the default `per_q_block=True` expects query metadata with an additional block dimension:

- [`ring_attention_kernel.py:985`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L985)

A caller following the documentation can produce an underestimated bound and invalid scalar-prefetch metadata.

The automatic centering path also averages padded keys despite separately receiving `orig_kv_seq_len`:

- [`ring_attention_kernel.py:915`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L915)

**Required fix:** rename and validate squared metadata explicitly, preserve a compatible per-head default or version the API, and compute mean from real tokens only.

### 10. `FlaxWanAttention` can silently discard supplied encoder states

The constructor defaults `is_self_attention=True`, and the call path now trusts that stored flag instead of deriving cross-attention from a non-`None` encoder input:

- [`attention_flax.py:2629`](src/maxdiffusion/models/attention_flax.py#L2629)
- [`attention_flax.py:2870`](src/maxdiffusion/models/attention_flax.py#L2870)

A caller that supplies distinct `encoder_hidden_states` without also overriding the constructor flag silently receives self-attention. Internal WAN call sites set the flag correctly, but this is still an API regression.

**Required fix:** reject contradictory arguments or retain the previous runtime behavior.

### 11. The fast-inference script's controls and artifact handling are stale

The script documents `FIXEDM=0` as selecting online softmax, but defaults `ATTENTION` to a fixed-m kernel before consulting `FIXEDM`:

- [`run_wan_fast_inference.sh:29`](end_to_end/tpu/run_wan_fast_inference.sh#L29)
- [`run_wan_fast_inference.sh:59`](end_to_end/tpu/run_wan_fast_inference.sh#L59)

Therefore `FIXEDM=0` still runs fixed-m.

The generator now writes directly under `output_dir`, while the script searches the current directory for the obsolete `wan_output_*.mp4` pattern:

- [`generate_wan.py:395`](src/maxdiffusion/generate_wan.py#L395)
- [`run_wan_fast_inference.sh:118`](end_to_end/tpu/run_wan_fast_inference.sh#L118)

A normal run finds nothing. If a stale legacy artifact exists, the script can move and relabel the wrong video.

### 12. `fused_rmsnorm_rope` only works when Q and K sequence lengths match

The function presents separate `Sq` and `Sk` dimensions but applies one unsliced frequency tensor to both Q and K:

- [`fused_producers.py:84`](src/maxdiffusion/kernels/fused_producers.py#L84)
- [`fused_producers.py:96`](src/maxdiffusion/kernels/fused_producers.py#L96)

For `Sq != Sk`, one side fails broadcasting. Either enforce equal lengths in the API or accept/slice separate Q/K frequencies. Current tests use equal lengths exclusively.

## Review-document quality

The second commit's [`CODE_REVIEW_kernel_improv.md`](CODE_REVIEW_kernel_improv.md) should not ship as authoritative review material.

It contains machine-local `file:///C:/...` links and factual errors, including:

- claiming fused RMSNorm incorrectly normalizes over `inner_dim`, even though the original WAN modules instantiate `nnx.RMSNorm(num_features=self.inner_dim)`;
- claiming `num_q_heads` is undefined where it is defined earlier in the same function; and
- presenting compiler/sharding speculation as established wrong-result behavior.

Delete it or replace it with a reviewed, repository-portable report.

## Required validation before reconsideration

At minimum:

1. LTX2 audio fixed-m integration with `head_dim=64`.
2. GQA/MQA through short-sequence dot fallback in both layout modes.
3. Dirty-tree, unrelated-CWD, and explicit-build-revision AOT tests.
4. Fingerprint tests for every static graph/compiler option.
5. Dynamic Python/JAX scalar signature tests across dtype and weak type.
6. Fixed-m adversarial tests for V magnitude and supported dtypes.
7. Padded direct-ring automatic-centering tests.
8. Multi-device partitioned HLO and TPU profiles for fused QKV.
9. End-to-end script tests for fixed-m selection and output paths.

## Recommended PR split

1. Fixed-m numerical/kernel correctness and its tests.
2. Ulysses/ring/GQA integration and topology tests.
3. WAN fused producers with partitioned HLO and TPU measurements.
4. AOT cache identity/fingerprinting as an independently reviewable change.

The underlying ideas are worth pursuing. The current branch combines too many compiler, numerical, collective, caching, and integration changes without sufficient isolation or proof.
