#!/bin/bash
# Benchmark sweep: 4 attention methods x 5 global batch sizes
# Methods: flash, tokamax_ring, ulysses, ulysses_ring (2D)
# Global batch sizes: 1, 2, 4, 8, 16
# TPU v7-8 (8 chips), DP=2, CP=4

set -eo pipefail

REPO_DIR="/mnt/data/sagarchapara/workspace/maxdiffusion"
VENV="/mnt/data/sagarchapara/workspace/venv"
RESULTS_DIR="/mnt/data/sagarchapara/workspace/bench_batch_sweep"

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

# Block sizes: simple for flash/ring, with layout hints for ulysses/2D
FLASH_BLOCKS='{"block_q":2048,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":2048,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"block_q_dq":2048,"block_kv_dq":2048,"use_fused_bwd_kernel":false}'
FLASH_BLOCKS_ULYSSES='{"block_q":2048,"block_kv_compute":1024,"block_kv":2048,"block_q_dkv":2048,"block_kv_dkv":2048,"block_kv_dkv_compute":1024,"block_q_dq":2048,"block_kv_dq":2048,"use_fused_bwd_kernel":false,"q_layout":"SEQ_MINOR","k_layout":"SEQ_MINOR","v_layout":"HEAD_DIM_MINOR"}'

SUMMARY_JSON="${RESULTS_DIR}/sweep_results.json"
echo '{}' > "${SUMMARY_JSON}"

run_bench() {
    local name="$1"
    local attention="$2"
    local ulysses_par="$3"
    local ring_par="$4"
    local block_sizes="$5"
    local per_device_bs="$6"
    local log_file="${RESULTS_DIR}/${name}.log"

    # Skip if already completed successfully
    if [ -f "${log_file}" ] && grep -q "generation_time" "${log_file}" 2>/dev/null; then
        echo "SKIPPING (already done): ${name}"
        return 0
    fi

    sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true

    echo "=========================================="
    echo "Running: ${name}"
    echo "  attention=${attention}, per_device_bs=${per_device_bs}"
    echo "  DP=2, CP=4, U=${ulysses_par}, R=${ring_par}"
    echo "=========================================="

    if python src/maxdiffusion/generate_wan.py \
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
        vae_spatial=1 \
        height=720 \
        width=1280 \
        num_frames=81 \
        num_inference_steps=40 \
        per_device_batch_size="${per_device_bs}" \
        enable_profiler=True \
        base_output_directory="${RESULTS_DIR}" \
        scan_layers=True \
        use_base2_exp=True \
        flash_block_sizes="${block_sizes}" \
        2>&1 | tee "${log_file}"; then
        echo "SUCCESS: ${name}"
    else
        echo "FAILED (likely OOM): ${name}" | tee -a "${log_file}"
    fi

    # Move generated videos to results dir
    mkdir -p "${RESULTS_DIR}/${name}/videos"
    for f in wan_output_*.mp4; do
        [ -f "$f" ] && mv "$f" "${RESULTS_DIR}/${name}/videos/"
    done

    # Move profiler traces to results dir
    if [ -d "sdxl-model-finetuned/${name}/tensorboard" ]; then
        mv "sdxl-model-finetuned/${name}/tensorboard" "${RESULTS_DIR}/${name}/tensorboard"
    fi

    echo "Done: ${name}"
    echo ""
}

echo "============================================================"
echo "  WAN 2.2 T2V Benchmark Sweep (720x1280, 81f, 40 steps)"
echo "  TPU v7-8 (8 chips), DP=2, CP=4"
echo "  Methods: flash, tokamax_ring, ulysses, ulysses_ring (2D)"
echo "  Global batch sizes: 1, 2, 4, 8, 16"
echo "============================================================"
echo ""

# Per-device batch sizes for global batch sizes on 8 chips:
# global=1 -> per_device=0.125
# global=2 -> per_device=0.25
# global=4 -> per_device=0.5
# global=8 -> per_device=1.0
# global=16 -> per_device=2.0

GLOBAL_SIZES=(1 2 4 8 16)
PER_DEVICE_SIZES=(0.125 0.25 0.5 1.0 2.0)

for i in "${!GLOBAL_SIZES[@]}"; do
    gbs="${GLOBAL_SIZES[$i]}"
    pbs="${PER_DEVICE_SIZES[$i]}"

    echo "============================================================"
    echo "  Global batch size: ${gbs} (per_device: ${pbs})"
    echo "============================================================"

    # 1. Flash
    run_bench "flash_gbs${gbs}" "flash" 1 1 "${FLASH_BLOCKS}" "${pbs}"

    # 2. Tokamax Ring
    run_bench "tokamax_ring_gbs${gbs}" "tokamax_ring" 1 1 "${FLASH_BLOCKS}" "${pbs}"

    # 3. Ulysses (with layout hints to avoid relayout)
    run_bench "ulysses_gbs${gbs}" "ulysses" 1 1 "${FLASH_BLOCKS_ULYSSES}" "${pbs}"

    # 4. 2D Ulysses+Ring (with layout hints to avoid relayout)
    run_bench "2d_u2r2_gbs${gbs}" "ulysses_ring" 2 2 "${FLASH_BLOCKS_ULYSSES}" "${pbs}"
done

echo ""
echo "============================================================"
echo "  All benchmarks complete. Extracting results..."
echo "============================================================"
echo ""

# Extract results into JSON
python3 - <<'PYEOF'
import re, os, json

results_dir = "/mnt/data/sagarchapara/workspace/bench_batch_sweep"
methods = ["flash", "tokamax_ring", "ulysses", "2d_u2r2"]
global_sizes = [1, 2, 4, 8, 16]

results = {}

for method in methods:
    results[method] = {}
    for gbs in global_sizes:
        name = f"{method}_gbs{gbs}"
        log_file = os.path.join(results_dir, f"{name}.log")
        if not os.path.exists(log_file):
            results[method][str(gbs)] = {"status": "missing"}
            continue

        compile_time = gen_time = None
        oom = False
        with open(log_file, "r") as f:
            content = f.read()
            if "RESOURCE_EXHAUSTED" in content or "Out of memory" in content or "OOM" in content:
                oom = True
            for line in content.split("\n"):
                m = re.search(r"compile_time:\s*([0-9.]+)", line)
                if m:
                    compile_time = float(m.group(1))
                m = re.search(r"generation_time:\s*([0-9.]+)", line)
                if m:
                    gen_time = float(m.group(1))

        if oom and gen_time is None:
            results[method][str(gbs)] = {"status": "OOM"}
        elif gen_time is not None:
            results[method][str(gbs)] = {
                "status": "ok",
                "compile_time": compile_time,
                "generation_time": gen_time,
            }
        else:
            results[method][str(gbs)] = {"status": "failed"}

output_file = os.path.join(results_dir, "sweep_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {output_file}")

# Print table
print("\n" + "=" * 80)
print("BENCHMARK RESULTS — DP=2, CP=4, 720x1280, 81f, 40 steps")
print("=" * 80)
header = f"{'Method':<18}"
for gbs in global_sizes:
    header += f"{'GBS=' + str(gbs):>12}"
print(header)
print("-" * 80)

for method in methods:
    row = f"{method:<18}"
    for gbs in global_sizes:
        entry = results[method].get(str(gbs), {})
        if entry.get("status") == "ok":
            row += f"{entry['generation_time']:>10.1f}s "
        elif entry.get("status") == "OOM":
            row += f"{'OOM':>12}"
        else:
            row += f"{'—':>12}"
    print(row)

print("-" * 80)
PYEOF
