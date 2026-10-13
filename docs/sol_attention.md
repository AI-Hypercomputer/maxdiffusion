# Sol-Attn: sparse self-attention for video/image DiTs

[Sol-Attn](https://arxiv.org/abs/2607.24027) is a training-free sparse
attention method. For every query block it picks the key/value blocks whose
block-proxy score `mean(Q_i) · mean(K_j)` exceeds a query-dependent threshold
`tau_i = mu_i + beta * sigma_i`, computes those blocks exactly, and replaces
every other block with its zeroth-order approximation (one pooled key with
weight `log(block size)`, value `mean(V_j)`). Both branches share one
online-softmax state, so skipped blocks still contribute to the softmax
numerator and denominator.

The expected exact-block density is about `1 - Phi(beta)`: `beta = 1.0` keeps
~16% of the blocks (~84% sparsity), close to the paper's Wan 2.1 setting.

## Using it with Wan 2.1 14B

```bash
python src/maxdiffusion/generate_wan.py src/maxdiffusion/configs/base_wan_14b.yml \
  attention=ulysses_custom ici_data_parallelism=2 ici_context_parallelism=4 \
  use_sol_attention=True sol_beta=1.0 sol_block_q=1024 sol_block_kv=256
```

| Key | Default | Meaning |
|---|---|---|
| `use_sol_attention` | `False` | Enable Sol-Attn for self-attention (cross-attention stays dense). Mutually exclusive with `use_svg_attention`. |
| `sol_beta` | `1.0` | Standardized threshold; higher is sparser. |
| `sol_block_q` | `1024` | Query routing block and kernel query tile, in tokens (multiple of 128). |
| `sol_block_kv` | `256` | Key/value block for exact compute and pooling (multiple of 128). |
| `sol_active_{start,end}_step` | `10, 50` | Sparse steps `[start, end)`; earlier steps are a dense warm-up (paper: first 20%). |
| `sol_active_{start,end}_layer` | `1, 40` | Sparse layers `[start, end)`; layer 0 stays dense. |
| `sol_dense_layer_fraction`, `sol_dense_timestep_fraction` | `0.0` | Fraction-based warm-up, used when no explicit interval is set. |
| `sol_log_density` | `False` | Print the measured exact-block density per call (debug). |

Sol-Attn works with any attention backend whose self-attention either shards
heads (`flash`) or the sequence over one mesh axis (`ulysses*`): in the latter
case an all-to-all gives each device full sequences for a subset of heads.

## Results on TPU v6e-8

Wan 2.1 14B T2V, 720p, 81 frames, 50 steps, `attention=ulysses_custom`,
`ici_data_parallelism=2 ici_context_parallelism=4`, one prompt, seed 12345.
The dense warm-up follows the paper (steps `[0, 10)` and layer 0 dense). PSNR
and SSIM compare each video with the dense video, as in Table 1 of the paper
(the paper reports PSNR 20.59 / SSIM 0.68 at 85% sparsity on H100).

| Config | Exact density | Generation time | Speedup | PSNR | SSIM |
|---|---|---|---|---|---|
| dense | 100% | 231.3 s | 1.00x | – | – |
| `sol_block_q=1024 sol_block_kv=256 sol_beta=1.0` (default) | ~17% | 160.6 s | 1.44x | 20.20 dB | 0.702 |
| `sol_block_q=1024 sol_block_kv=256 sol_beta=0.8` | – | 170.5 s | 1.36x | 20.60 dB | 0.735 |
| `sol_block_q=512 sol_block_kv=256 sol_beta=1.0` | – | 175.0 s | 1.32x | 20.99 dB | 0.755 |
| `sol_block_q=2048 sol_block_kv=512 sol_beta=1.0` | ~16% | 150.4 s* | 1.54x | 18.55 dB | 0.633 |

\* measured with `sol_log_density=True`, which adds a host callback per layer.

Per self-attention layer (all 8 chips, synthetic video-like inputs at the
same shapes), dense `ulysses_custom` takes 83.7 ms and Sol-Attn 36.6 ms at
`1024x256` (2.3x) and 23.4 ms at `2048x512` (3.6x). About 13 ms of each
layer is shared overhead (Ulysses all-to-alls and layout copies), so
end-to-end gains are bounded by Amdahl's law and the 20% dense warm-up.

## Code layout

```
kernels/sol_attention/
  routing.py               pooling, thresholds (Eq. 3-5), index compaction (pure JAX)
  sol_attention_kernel.py  Pallas TPU kernel + `sol_attention(q, k, v, ...)` entry point
  reference.py             dense O(L^2) oracle of Eqs. (9)-(10), for tests
models/
  sol_attention.py         config resolution, warm-up dispatch, sharded entry point
  sparse_schedule.py       step/layer warm-up schedule shared with SVG
```

`kernels/sol_attention` has no model dependencies: `sol_attention(q, k, v,
scale=..., beta=...)` takes `[..., L, d]` tensors and is usable on its own.

## TPU adaptation

The paper's GPU kernel fuses routing into the attention kernel to avoid
materializing the `N x N` block-proxy map. On TPU the design differs in a few
places, each driven by measurements on v6e:

* **Routing in a small XLA pre-pass.** With TPU-sized blocks the proxy map is
  tiny (e.g. 148 x 148 per head at 512-token blocks), so `mu_i` and `sigma_i`
  are taken directly from the materialized row (identical to the moment
  estimator of Eq. 5). Only compacted per-query-block index rows reach the
  kernel; the selection mask for the approximate branch is rebuilt on chip
  from them.
* **Exact branch via manual DMAs.** Keys/values stay in HBM; the kernel walks
  the selected blocks through a ring of async copies, so only routed blocks are
  read. The next grid step's index row is prefetched into SMEM.
* **Transposed score layout.** Scores are `[keys, queries]` so softmax
  reductions run across sublanes, as in the dense splash kernel.
* **Branch-free hot loop.** The ragged tail block (sequence length not a
  multiple of `sol_block_kv`) runs as a masked epilogue; a per-block branch
  cost ~40%.
* **Larger blocks than the paper's 64 x 64.** MXU efficiency and per-block
  overheads favor 512-2048 query tokens and 256-512 key tokens.
* **Grouped hot loop.** Selected blocks land in contiguous ring slots, so the
  loop fuses several of them into one ~1024-key score tile (one softmax
  rescale per tile), keeping routing at `sol_block_kv` granularity. This
  alone is 10-25% faster for 128-256 token blocks.

## Extending to other models

1. In the attention module constructor, call
   `sol_attention.init_sol_config(attention_config, num_layers, is_self_attention)`
   and `setattr` the result on the module.
2. In `__call__`, wrap the attention op:
   `sol_attention.apply_sol_or_dense(self, run_attention, layer_index, step_index, timestep)`.
   `run_attention(sparse_config_override=...)` must forward the override to
   `attention_flax._apply_attention(spatiotemporal_config=...)`.
3. In the pipeline, merge `sol_attention.attention_config_from_pyconfig(config, num_layers)`
   into the transformer's `attention_config`, and add
   `aot_cache.extract_sol_meta(config, pipeline)` to the AOT metadata.

For MMDiT-style joint attention (text and video tokens in one sequence), pass
`always_exact` to the kernel to keep text blocks in the exact branch, as the
paper does for HunyuanVideo.
