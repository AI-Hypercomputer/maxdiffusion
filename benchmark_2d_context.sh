#!/bin/bash
# Benchmark script for comparing attention parallelism strategies on v7-8 TPU
# Runs training with synthetic data for a few steps and records step times.

set -e

REPO_DIR="/mnt/data/sagarchapara/workspace/maxdiffusion"
VENV="/mnt/data/sagarchapara/workspace/venv"
BASE_CONFIG="src/maxdiffusion/configs/base_wan_14b.yml"
RESULTS_DIR="/mnt/data/sagarchapara/workspace/benchmark_results"
METRICS_DIR="${RESULTS_DIR}/metrics"

export HF_HOME="/mnt/data/sagarchapara/cache/huggingface"
export JAX_COMPILATION_CACHE_DIR="/mnt/data/sagarchapara/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

source "${VENV}/bin/activate"
cd "${REPO_DIR}"

mkdir -p "${RESULTS_DIR}" "${METRICS_DIR}"

NUM_STEPS=10  # steps to run (first step includes compilation, skip it)

run_benchmark() {
    local name="$1"
    local attention="$2"
    local context_parallelism="$3"
    local ulysses_par="$4"
    local ring_par="$5"
    local metrics_file="${METRICS_DIR}/${name}.txt"
    local log_file="${RESULTS_DIR}/${name}.log"

    echo "=========================================="
    echo "Running benchmark: ${name}"
    echo "  attention=${attention}, ici_context_parallelism=${context_parallelism}"
    echo "  context_ulysses_parallelism=${ulysses_par}, context_ring_parallelism=${ring_par}"
    echo "=========================================="

    python src/maxdiffusion/train_wan.py \
        src/maxdiffusion/configs/base_wan_14b.yml \
        run_name="${name}" \
        attention="${attention}" \
        dataset_type="synthetic" \
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
        max_train_steps="${NUM_STEPS}" \
        per_device_batch_size=1 \
        metrics_file="${metrics_file}" \
        write_metrics=True \
        enable_profiler=False \
        scan_layers=True \
        remat_policy="NONE" \
        checkpoint_every=-1 \
        save_final_checkpoint=False \
        skip_jax_distributed_system=False \
        base_output_directory="" \
        height=480 \
        width=832 \
        num_frames=81 \
        enable_ssim=False \
        2>&1 | tee "${log_file}"

    echo ""
    echo "Benchmark ${name} complete. Log: ${log_file}"
    echo ""
}

echo "============================================================"
echo "  2D Context Parallelism Benchmark Suite"
echo "  TPU v7-8 (8 chips)"
echo "  ${NUM_STEPS} training steps per config (step 0 = compilation)"
echo "============================================================"
echo ""

# 1. Pure Ring attention (context_parallelism=8)
run_benchmark "ring_cp8" "ring" 8 1 1

# 2. Pure Ulysses attention (context_parallelism=8)
run_benchmark "ulysses_cp8" "ulysses" 8 1 1

# 3. 2D context: Ulysses=2, Ring=4
run_benchmark "2d_u2_r4" "ulysses_ring" 8 2 4

# 4. 2D context: Ulysses=4, Ring=2
run_benchmark "2d_u4_r2" "ulysses_ring" 8 4 2

echo ""
echo "============================================================"
echo "  All benchmarks complete. Extracting results..."
echo "============================================================"
echo ""

# Extract step times from logs
python3 - <<'PYEOF'
import re
import os
import json

results_dir = "/mnt/data/sagarchapara/workspace/benchmark_results"
metrics_dir = os.path.join(results_dir, "metrics")
configs = ["ring_cp8", "ulysses_cp8", "2d_u2_r4", "2d_u4_r2"]

print("\n" + "=" * 70)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 70)
print(f"{'Config':<20} {'Avg Step (s)':<15} {'Min Step (s)':<15} {'TFLOPS/dev':<15}")
print("-" * 70)

summary = {}
for config_name in configs:
    metrics_file = os.path.join(metrics_dir, f"{config_name}.txt")
    if not os.path.exists(metrics_file):
        print(f"{config_name:<20} {'NO DATA':<15}")
        continue

    step_times = []
    tflops_vals = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "perf/step_time_seconds" in data.get("scalar", {}):
                    step_times.append(data["scalar"]["perf/step_time_seconds"])
                if "perf/per_device_tflops_per_sec" in data.get("scalar", {}):
                    tflops_vals.append(data["scalar"]["perf/per_device_tflops_per_sec"])
            except json.JSONDecodeError:
                # Try line-by-line key=value format
                pass

    if not step_times:
        # Try parsing from log file
        log_file = os.path.join(results_dir, f"{config_name}.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    m = re.search(r"step_time_seconds['\"]?\s*[:=]\s*([0-9.]+)", line)
                    if m:
                        step_times.append(float(m.group(1)))
                    m = re.search(r"per_device_tflops_per_sec['\"]?\s*[:=]\s*([0-9.]+)", line)
                    if m:
                        tflops_vals.append(float(m.group(1)))

    if step_times:
        # Skip first step (compilation)
        warmup = step_times[:1]
        steady = step_times[1:] if len(step_times) > 1 else step_times
        avg_time = sum(steady) / len(steady)
        min_time = min(steady)
        avg_tflops = sum(tflops_vals[1:]) / len(tflops_vals[1:]) if len(tflops_vals) > 1 else (tflops_vals[0] if tflops_vals else 0)
        print(f"{config_name:<20} {avg_time:<15.4f} {min_time:<15.4f} {avg_tflops:<15.2f}")
        summary[config_name] = {"avg_step_time": avg_time, "min_step_time": min_time, "avg_tflops": avg_tflops, "warmup_time": warmup[0] if warmup else 0}
    else:
        print(f"{config_name:<20} {'PARSE ERROR':<15}")

print("-" * 70)
if summary:
    best = min(summary.items(), key=lambda x: x[1]["avg_step_time"])
    print(f"\nBest config: {best[0]} with avg step time {best[1]['avg_step_time']:.4f}s")
    if "ring_cp8" in summary and "ulysses_cp8" in summary:
        ring_time = summary["ring_cp8"]["avg_step_time"]
        ulysses_time = summary["ulysses_cp8"]["avg_step_time"]
        for name, data in summary.items():
            if name.startswith("2d_"):
                speedup_vs_ring = (ring_time - data["avg_step_time"]) / ring_time * 100
                speedup_vs_ulysses = (ulysses_time - data["avg_step_time"]) / ulysses_time * 100
                print(f"{name}: {speedup_vs_ring:+.1f}% vs ring, {speedup_vs_ulysses:+.1f}% vs ulysses")

# Save summary
with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed results saved to {results_dir}/summary.json")
PYEOF
