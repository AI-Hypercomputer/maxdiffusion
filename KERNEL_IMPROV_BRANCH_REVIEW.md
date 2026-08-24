# `kernel_improv` Branch Review

> [!IMPORTANT]
> The branch was force-rewritten after the original review. The original findings are retained below for audit history,
> but their status is superseded by the [follow-up review of `7b483ee2`](#follow-up-review-2026-08-24).

Reviewed against merge base `a327ebd4a99eba208f5a79127f7e637788b396ec`.

Original commits in scope:

- `1d71782611d03c912eadc38de99cabe26ee21da3` - fixed-m Splash Attention, 2D ring attention, WAN fused producers, and WAN AOT integration.
- `199905ca585d808ea25e66bbfacfe69ed922aea1` - branch review document.

## Original verdict (superseded by follow-up)

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

---

## Follow-up review: 2026-08-24

Review target: `origin/kernel_improv` at `7b483ee247695af67f05411ddb005f1f4e82b2fe`, compared with merge base
`7cfb8801d41d1728f0232cd7ee59e900a151fecf` and the original feature revision `1d71782611d03c912eadc38de99cabe26ee21da3`.

### Follow-up verdict

**The branch is materially improved, but it is still not mergeable.**

The low-head-dimension integration, GQA fallback, and online factory regressions were repaired. The rewritten branch still
contains an immediately broken fast-inference entry point, unsafe WAN AOT identity/fingerprinting, an unenforced fixed-m
numerical precondition, and a public ring metadata contract capable of producing invalid bounds. The fused-QKV TPU
performance claim also remains unsupported by partitioned HLO or multi-device measurements.

Any P1 finding below is independently merge-blocking.

### P1 findings

#### 1. The fast-inference script crashes on its documented default invocation

The script enables nounset handling and then directly expands `ATTENTION` before giving it a default:

- [`run_wan_fast_inference.sh:34`](end_to_end/tpu/run_wan_fast_inference.sh#L34)
- [`run_wan_fast_inference.sh:59`](end_to_end/tpu/run_wan_fast_inference.sh#L59)

With `ATTENTION` unset, the documented invocation terminates immediately with:

```text
ATTENTION: unbound variable
```

The intended `FIXEDM=0` selection logic is otherwise correct, but it cannot execute on the normal default path.

**Required fix:** test `${ATTENTION:-}` rather than expanding `$ATTENTION` under `set -u`, and add a shell test covering
unset `ATTENTION`, `FIXEDM=0`, `FIXEDM=1`, and an explicit attention override.

#### 2. Dirty CLI source can still reuse clean-HEAD WAN AOT executables

The normal CLI obtains its provenance hash without dirty checking:

- [`generate_wan.py:478`](src/maxdiffusion/generate_wan.py#L478)
- [`generate_wan.py:485`](src/maxdiffusion/generate_wan.py#L485)

That non-`None` value is passed into `run`, which prefers it over the new dirty-aware fallback:

- [`generate_wan.py:302`](src/maxdiffusion/generate_wan.py#L302)

Therefore `get_git_commit_hash(check_dirty=True)` is never called during ordinary CLI execution. Editing a transformer or
kernel while retaining a clean-HEAD cache still permits stale executable reuse.

The helper also continues to run Git in the process working directory rather than against the MaxDiffusion source root:

- [`max_utils.py:384`](src/maxdiffusion/max_utils.py#L384)

Launching from another repository can key WAN AOT with that unrelated HEAD and shadow `aot_build_revision`.

**Required fix:** resolve the package source root, use `git -C`, perform the dirty/untracked-source check in the normal CLI
path, and test dirty, clean, packaged, and unrelated-CWD execution explicitly.

#### 3. The WAN AOT fingerprint still omits static graph and topology inputs

The metadata now contains Ulysses shard count, base-2 mode, and experimental scheduler mode:

- [`generate_wan.py:312`](src/maxdiffusion/generate_wan.py#L312)

It still omits graph-defining configuration including at least:

- `flash_min_seq_length`
- `mask_padding_tokens`
- `ulysses_attention_chunks`
- configured matmul precision
- logical axis rules and mesh-axis identity
- backend, TPU device kind, process count, and `jaxlib` version

Several of these values are embedded into the static NNX transformer graph:

- [`wan_pipeline.py:335`](src/maxdiffusion/pipelines/wan/wan_pipeline.py#L335)
- [`wan_pipeline.py:340`](src/maxdiffusion/pipelines/wan/wan_pipeline.py#L340)
- [`wan_pipeline.py:349`](src/maxdiffusion/pipelines/wan/wan_pipeline.py#L349)

For identical array shapes, moving `flash_min_seq_length` across the actual sequence length selects dot-product versus
Splash/Ulysses attention while remaining eligible for the same WAN AOT fingerprint. Changing
`ulysses_attention_chunks` likewise changes the collective graph without changing input shapes.

**Required fix:** build one canonical WAN metadata function equivalent in coverage to `ltx2_aot_metadata`, then test that
every static graph/compiler/topology field changes the fingerprint.

#### 4. Fixed-m still overflows for finite inputs outside an unenforced value bound

The dynamic recentering proof assumes `abs(V) <= 256`:

- [`custom_splash_attention.py:51`](src/maxdiffusion/kernels/custom_splash_attention.py#L51)

No factory or eligibility path validates that assumption. A minimal counterexample remains:

- `N = 4096`
- `Q = 0`
- `K = 0`
- BF16 `V = 512`

The tile is eligible and uses `C=107`, making the FP32 output numerator reach:

```text
4096 * 512 * 2^107 = 2^128
```

The accumulator overflows even though exact attention returns 512. Float16 is invalid at much smaller values because
the exponential weights are narrowed before the output dot:

- [`custom_splash_attention.py:222`](src/maxdiffusion/kernels/custom_splash_attention.py#L222)

**Required fix:** incorporate `max(abs(V))` into eligibility, derive a conservative constant for the complete supported
value/dtype domain, or enforce and document a hard input contract. Add adversarial tests that violate the current
assumption.

#### 5. The public ring fixed-m metadata contract remains unsafe

The factory documentation still advertises actual norms:

- [`ring_attention_kernel.py:1214`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L1214)

The implementation consumes squared, optionally per-Q-block norms:

- [`ring_attention_kernel.py:988`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L988)
- [`ring_attention_kernel.py:990`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L990)

A caller following the documented `(qn_max, mk_h)` contract gets `sqrt(qn * mk)` instead of `qn * mk`, underestimates
the fixed bound, and can accidentally broadcast one-dimensional metadata to `(heads, heads)` under the default
`per_q_block=True` behavior.

The merge-base implementation automatically detected a pre-gathered `(ring_size, heads)` K-norm table. The rewritten
implementation gathers it again unless callers discover and set `pregathered_mk=True`:

- [`ring_attention_kernel.py:1002`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L1002)
- [`ring_attention_kernel.py:1208`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L1208)

With the default, pre-gathered `(R, H)` metadata becomes `(R, R, H)` and no longer matches subsequent bound logic.

Automatic centering also continues to average padded K despite receiving `orig_kv_seq_len`:

- [`ring_attention_kernel.py:917`](src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L917)

That zero-diluted mean is inconsistent with real-token centered norm metadata and invalidates the claimed centered-row-max
proof for direct callers.

**Required fix:** version or repair the API, rename squared metadata explicitly, validate ranks/shapes, auto-detect
pre-gathered tables or remove compatibility, and compute global means from real-token sums and counts only.

#### 6. Fused QKV remains unsupported as a tensor-parallel TPU optimization

The WAN self-attention path still concatenates three independently `("embed", "heads")`-sharded kernels along the
tensor-parallel output dimension:

- [`attention_flax.py:2917`](src/maxdiffusion/models/attention_flax.py#L2917)
- [`attention_flax.py:2945`](src/maxdiffusion/models/attention_flax.py#L2945)

On TP greater than one, preserving global `[Q, K, V]` ordering requires redistribution/materialization around the
concatenation and split, unless XLA decomposes the operation back into separate projection dots. In the latter case the
claimed fusion disappears.

**Required evidence before merge:** partitioned HLO, collective-volume comparison, and an end-to-end profile on the target
TPU topology. Single-device value parity is not performance evidence.

### P2 findings

#### 7. AOT signatures still alias weak and strong JAX scalars

The Python `int` versus `float` fast-key collision is repaired. Array leaves are still keyed only by shape and dtype:

- [`aot_cache.py:98`](src/maxdiffusion/aot_cache.py#L98)
- [`aot_cache.py:202`](src/maxdiffusion/aot_cache.py#L202)

Weak `jnp.asarray(1)` and strong `jnp.asarray(1, dtype=jnp.int32)` therefore collide even though weak-type promotion can
lower a different graph. For example, addition to an `int8` array can retain `int8` for the weak scalar and promote to
`int32` for the strong scalar while presenting identical runtime scalar shape/dtype buffers.

**Required fix:** include `weak_type` in both the complete and fast signatures and keep compiled execution behind a safe
fallback path.

#### 8. `FlaxWanAttention` can still discard supplied encoder states silently

The constructor stores `is_self_attention=True` by default, and the call path trusts that stored value:

- [`attention_flax.py:2663`](src/maxdiffusion/models/attention_flax.py#L2663)
- [`attention_flax.py:2904`](src/maxdiffusion/models/attention_flax.py#L2904)

A caller passing distinct `encoder_hidden_states` without explicitly constructing the layer with
`is_self_attention=False` receives self-attention instead. Internal WAN call sites set the flag correctly, but the public
behavior remains a silent API regression.

#### 9. `fused_rmsnorm_rope` still requires equal Q/K sequence lengths implicitly

The function presents independent `Sq` and `Sk` dimensions but applies one unsliced frequency tensor to both sequences:

- [`fused_producers.py:85`](src/maxdiffusion/kernels/fused_producers.py#L85)
- [`fused_producers.py:90`](src/maxdiffusion/kernels/fused_producers.py#L90)
- [`fused_producers.py:96`](src/maxdiffusion/kernels/fused_producers.py#L96)

For unequal sequence lengths, either Q or K fails broadcasting. Enforce equal lengths in the API or accept/slice distinct
Q/K frequency tensors.

#### 10. `heads_per_tile` remains silently ignored for multi-rank custom ring

The value is extracted from the configured block sizes:

- [`attention_flax.py:1494`](src/maxdiffusion/models/attention_flax.py#L1494)

The `R > 1` ring factory cannot receive it:

- [`attention_flax.py:1621`](src/maxdiffusion/models/attention_flax.py#L1621)

The merge-base rejection was removed, so `heads_per_tile=2` appears accepted while still executing one head per program.
Restore an explicit rejection or implement the requested layout.

### Confirmed fixes since the original review

1. Fixed-m `k_mean` is padded consistently for head dimensions below 128. The stock LTX2 audio workload normally falls
   below `flash_min_seq_length` and uses dot-product attention, but the reachable custom-kernel configuration is now
   correctly handled.
2. Short-sequence GQA/MQA dot fallback reshapes and repeats K/V correctly in both supported layouts.
3. Online `make_splash_mha(block_sizes)` no longer computes fixed-m constants from a missing sequence length.
4. Python dynamic `int` versus `float` AOT fast-key collisions are distinguished.
5. Successful-run video discovery no longer moves and relabels the generated artifact.

No dedicated regression tests were added for these repairs.

### Follow-up validation performed

- `git diff --check 7cfb8801..7b483ee2` passed.
- All 13 changed Python files compiled successfully from the reviewed Git tree.
- JAX/TPU runtime tests were not executable in the review environment because JAX is not installed.
- No partitioned HLO or target-TPU profile was supplied for fused QKV.

### Follow-up conclusion

The rewrite fixes several concrete integration failures, but the branch still crosses kernel numerics, collective topology,
TP sharding, persistent executable identity, and shell integration without sufficient isolation or tests. Fix the P1 items
and supply the missing TPU evidence before requesting another merge review.
