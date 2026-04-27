#!/bin/bash
# Benchmark: DP=2, CP=4 on v7-8 (8 chips)
# Configs: flash, ring, ulysses, ring+ulysses (2D)
# Run ring+ulysses FIRST with video + profiler saved

set -e

REPO_DIR="/mnt/data/sagarchapara/workspace/maxdiffusion"
VENV="/mnt/data/sagarchapara/workspace/venv"
RESULTS_DIR="/mnt/data/sagarchapara/workspace/bench_dp2_cp4"

export HF_HOME="/mnt/data/sagarchapara/cache/huggingface"
export JAX_COMPILATION_CACHE_DIR="/mnt/data/sagarchapara/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_tpu_megacore_fusion_allow_ags=false \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_assign_all_reduce_scatter_layout=true'

source "${VENV}/bin/activate"
cd "${REPO_DIR}"

mkdir -p "${RESULTS_DIR}"

run_bench() {
    local name="$1"
    local attention="$2"
    local ulysses_par="$3"
    local ring_par="$4"
    local profiler="$5"
    local log_file="${RESULTS_DIR}/${name}.log"

    sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true

    echo "=========================================="
    echo "Running: ${name}"
    echo "  attention=${attention}, DP=2, CP=4, U=${ulysses_par}, R=${ring_par}"
    echo "  profiler=${profiler}"
    echo "=========================================="

    python src/maxdiffusion/generate_wan.py \
        src/maxdiffusion/configs/base_wan_27b.yml \
        run_name="${name}" \
        attention="${attention}" \
        ici_data_parallelism=2 \
        ici_fsdp_parallelism=1 \
        ici_context_parallelism=4 \
        ici_tensor_parallelism=1 \
        dcn_data_parallelism=1 \
        dcn_fsdp_parallelism=1 \
        dcn_context_parallelism=1 \
        dcn_tensor_parallelism=1 \
        context_ulysses_parallelism="${ulysses_par}" \
        context_ring_parallelism="${ring_par}" \
        vae_spatial=8 \
        height=720 \
        width=1280 \
        num_frames=81 \
        num_inference_steps=40 \
        per_device_batch_size=0.125 \
        enable_profiler="${profiler}" \
        base_output_directory="${RESULTS_DIR}" \
        scan_layers=True \
        flash_block_sizes='{"block_q":2048,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":2048,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"block_q_dq":2048,"block_kv_dq":2048,"use_fused_bwd_kernel":false}' \
        2>&1 | tee "${log_file}"

    # Move generated videos to results dir
    mkdir -p "${RESULTS_DIR}/${name}/videos"
    for f in wan_output_*.mp4; do
        [ -f "$f" ] && mv "$f" "${RESULTS_DIR}/${name}/videos/"
    done

    # Move profiler traces to results dir
    if [ -d "sdxl-model-finetuned/${name}/tensorboard" ]; then
        mv "sdxl-model-finetuned/${name}/tensorboard" "${RESULTS_DIR}/${name}/tensorboard"
    fi

    echo ""
    echo "Done: ${name} -> ${log_file}"
    echo ""
}

echo "============================================================"
echo "  WAN 2.2 T2V Inference Benchmark (720x1280, 81f, 40 steps)"
echo "  TPU v7-8 (8 chips), DP=2, CP=4"
echo "  Flash block sizes: 2048/1024/2048"
echo "  XLA optimization flags: ENABLED"
echo "============================================================"
echo ""

# 1. Ring+Ulysses (2D) — FIRST
run_bench "2d_u2_r2" "ulysses_ring" 2 2 True

# 2. Flash
run_bench "flash_dp2_cp4" "flash" 1 1 True

# 3. Ring
run_bench "ring_dp2_cp4" "ring" 1 1 True

# 4. Ulysses
run_bench "ulysses_dp2_cp4" "ulysses" 1 1 True

echo ""
echo "============================================================"
echo "  All benchmarks complete. Extracting results..."
echo "============================================================"
echo ""

python3 - <<'PYEOF'
import re, os, json

results_dir = "/mnt/data/sagarchapara/workspace/bench_dp2_cp4"
configs = ["2d_u2_r2", "flash_dp2_cp4", "ring_dp2_cp4", "ulysses_dp2_cp4"]

print("\n" + "=" * 70)
print("BENCHMARK RESULTS — DP=2, CP=4, 720x1280, 81f, 40 steps")
print("=" * 70)
print(f"{'Config':<22} {'Compile (s)':<15} {'Generation (s)':<15} {'Gen+Prof (s)':<15}")
print("-" * 70)

summary = {}
for name in configs:
    log_file = os.path.join(results_dir, f"{name}.log")
    if not os.path.exists(log_file):
        print(f"{name:<22} {'NO DATA':<15}")
        continue

    compile_time = gen_time = gen_prof_time = None
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"compile_time:\s*([0-9.]+)", line)
            if m: compile_time = float(m.group(1))
            m = re.search(r"generation_time:\s*([0-9.]+)", line)
            if m: gen_time = float(m.group(1))
            m = re.search(r"generation_time_with_profiler:\s*([0-9.]+)", line)
            if m: gen_prof_time = float(m.group(1))

    ct = f"{compile_time:.1f}" if compile_time else "—"
    gt = f"{gen_time:.1f}" if gen_time else "—"
    gpt = f"{gen_prof_time:.1f}" if gen_prof_time else "—"
    print(f"{name:<22} {ct:<15} {gt:<15} {gpt:<15}")
    summary[name] = {"compile_time": compile_time, "generation_time": gen_time, "generation_time_with_profiler": gen_prof_time}

print("-" * 70)
gen_times = {k: v["generation_time"] for k, v in summary.items() if v.get("generation_time")}
if gen_times:
    best = min(gen_times, key=gen_times.get)
    print(f"\nBest: {best} with generation_time = {gen_times[best]:.1f}s")
    baseline = gen_times.get("2d_u2_r2")
    if baseline:
        for name, gt in gen_times.items():
            if name != "2d_u2_r2":
                diff = (gt - baseline) / baseline * 100
                print(f"  {name} vs 2D: {diff:+.1f}%")

with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {results_dir}/summary.json")
PYEOF
