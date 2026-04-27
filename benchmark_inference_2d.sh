#!/bin/bash
# Inference benchmark: compare attention strategies on v7-8 TPU (8 chips)
# WAN 2.2 T2V, 720x1280, 81 frames, 40 inference steps, 1 video

set -e

REPO_DIR="/mnt/data/sagarchapara/workspace/maxdiffusion"
VENV="/mnt/data/sagarchapara/workspace/venv"
RESULTS_DIR="/mnt/data/sagarchapara/workspace/inference_bench_results"

export HF_HOME="/mnt/data/sagarchapara/cache/huggingface"
export JAX_COMPILATION_CACHE_DIR="/mnt/data/sagarchapara/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

source "${VENV}/bin/activate"
cd "${REPO_DIR}"

mkdir -p "${RESULTS_DIR}"

run_inference_bench() {
    local name="$1"
    local attention="$2"
    local context_parallelism="$3"
    local ulysses_par="$4"
    local ring_par="$5"
    local log_file="${RESULTS_DIR}/${name}.log"

    echo "=========================================="
    echo "Running: ${name}"
    echo "  attention=${attention}, CP=${context_parallelism}, U=${ulysses_par}, R=${ring_par}"
    echo "=========================================="

    # Clear any stale lockfile from prior crash
    sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true

    python src/maxdiffusion/generate_wan.py \
        src/maxdiffusion/configs/base_wan_27b.yml \
        run_name="${name}" \
        attention="${attention}" \
        ici_data_parallelism=1 \
        ici_fsdp_parallelism=1 \
        ici_context_parallelism="${context_parallelism}" \
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
        enable_profiler=False \
        scan_layers=True \
        base_output_directory="" \
        2>&1 | tee "${log_file}"

    echo ""
    echo "Benchmark ${name} complete. Log: ${log_file}"
    echo ""
}

echo "============================================================"
echo "  WAN 2.2 T2V Inference Benchmark (720x1280, 81f, 40 steps)"
echo "  TPU v7-8 (8 chips), per_device_batch_size=0.125 (1 video)"
echo "============================================================"
echo ""

# 1. Flash baseline (CP=8, no ring/ulysses)
run_inference_bench "flash_cp8" "flash" 8 1 1

# 2. Ring attention (CP=8)
run_inference_bench "ring_cp8" "ring" 8 1 1

# 3. Ulysses attention (CP=8)
run_inference_bench "ulysses_cp8" "ulysses" 8 1 1

# 4. 2D context: Ulysses=2, Ring=4
run_inference_bench "2d_u2_r4" "ulysses_ring" 8 2 4

# 5. 2D context: Ulysses=4, Ring=2
run_inference_bench "2d_u4_r2" "ulysses_ring" 8 4 2

echo ""
echo "============================================================"
echo "  All benchmarks complete. Extracting results..."
echo "============================================================"
echo ""

# Extract timing from logs
python3 - <<'PYEOF'
import re
import os
import json

results_dir = "/mnt/data/sagarchapara/workspace/inference_bench_results"
configs = ["flash_cp8", "ring_cp8", "ulysses_cp8", "2d_u2_r4", "2d_u4_r2"]

print("\n" + "=" * 70)
print("INFERENCE BENCHMARK RESULTS")
print("=" * 70)
print(f"{'Config':<20} {'Compile (s)':<15} {'Generation (s)':<15}")
print("-" * 70)

summary = {}
for name in configs:
    log_file = os.path.join(results_dir, f"{name}.log")
    if not os.path.exists(log_file):
        print(f"{name:<20} {'NO DATA':<15}")
        continue

    compile_time = None
    gen_time = None
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"compile_time:\s*([0-9.]+)", line)
            if m:
                compile_time = float(m.group(1))
            m = re.search(r"generation_time:\s*([0-9.]+)", line)
            if m:
                gen_time = float(m.group(1))

    if compile_time is not None and gen_time is not None:
        print(f"{name:<20} {compile_time:<15.1f} {gen_time:<15.1f}")
        summary[name] = {"compile_time": compile_time, "generation_time": gen_time}
    else:
        print(f"{name:<20} {'PARSE ERROR':<15}")

print("-" * 70)
if summary:
    best = min(summary.items(), key=lambda x: x[1]["generation_time"])
    print(f"\nBest: {best[0]} with generation_time = {best[1]['generation_time']:.1f}s")

    for base in ["ring_cp8", "ulysses_cp8"]:
        if base in summary:
            base_time = summary[base]["generation_time"]
            for name, data in summary.items():
                if name.startswith("2d_"):
                    speedup = (base_time - data["generation_time"]) / base_time * 100
                    print(f"  {name} vs {base}: {speedup:+.1f}%")

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {results_dir}/summary.json")
PYEOF
