# 2D Context Parallelism Benchmark Results

## Setup
- **Model**: WAN 2.2 T2V (Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Hardware**: TPU v7-8 (4 chips x 2 cores = 8 devices)
- **Resolution**: 720x1280, 81 frames, 40 inference steps
- **Parallelism**: DP=2, CP=4 (per_device_batch_size=0.125, 1 video)
- **Flash block sizes**: block_q=2048, block_kv_compute=1024, block_kv=2048
- **VAE spatial sharding**: 8
- **Attention kernel**: tokamax splash (fused bwd kernel)

## Results

### Without XLA optimization flags

| Config | Compile (s) | Generation (s) | vs 2D Context |
|--------|------------|----------------|---------------|
| **2D Context (U=2, R=2)** | 228.8 | **208.0** | baseline |
| Flash (CP=4) | 250.9 | **231.3** | +11.2% slower |
| Ring (CP=4) | 258.1 | **237.1** | +14.0% slower |

### With XLA optimization flags (LIBTPU_INIT_ARGS)

| Config | Compile (s) | Generation (s) | vs 2D Context |
|--------|------------|----------------|---------------|
| **2D Context (U=2, R=2)** | 268.2 | **182.4** | baseline |
| Flash (CP=4) | 231.8 | **205.3** | +12.6% slower |
| Ring (CP=4) | -- | -- | not run |

### Key XLA flags used
```
--xla_tpu_enable_async_all_to_all=true
--xla_enable_async_collective_permute=true
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_overlap_compute_collective_tc=true
--xla_tpu_dvfs_p_state=7
--xla_tpu_scoped_vmem_limit_kib=65536
--xla_latency_hiding_scheduler_rerun=2
```
(plus additional scheduler/pipelining flags -- see benchmark script)

## Why 2D Context Parallelism Wins

TPU v7-8 topology: 4 chips with 2 cores each. Intra-chip bandwidth between
cores is **6x faster** than inter-chip ICI.

With U=2, R=2:
- **Ulysses all-to-all (U=2)**: Runs between the 2 cores on the same chip,
  using the 6x fast intra-chip link.
- **Ring ppermute (R=2)**: Only 1 rotation step across chips (R-1=1), halving
  ICI communication vs pure ring (CP=4, 3 rotation steps).

The async collective flags further help by overlapping the all-to-all and
ppermute communication with compute.

## Implementation

The 2D context parallelism is activated with:
```yaml
attention: 'ulysses_ring'
ici_data_parallelism: 2
ici_context_parallelism: 4
context_ulysses_parallelism: 2
context_ring_parallelism: 2
```

Files changed (git diff from main):
- `src/maxdiffusion/models/attention_flax.py` -- core `_2d_context_attention()` using tokamax splash
- `src/maxdiffusion/models/wan/transformers/transformer_wan.py` -- plumbing U/R params
- `src/maxdiffusion/pipelines/wan/wan_pipeline.py` -- config loading
- `src/maxdiffusion/pyconfig.py` -- axis rules and validation
- `src/maxdiffusion/common_types.py` -- ULYSSES_RING_ATTENTION_AXIS_RULES
- `src/maxdiffusion/configs/base_wan_14b.yml` -- config params
- `src/maxdiffusion/configs/base_wan_27b.yml` -- config params + block sizes
