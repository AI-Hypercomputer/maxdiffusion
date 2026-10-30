#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reproduces the MaxDiffusion Wan 2.2 T2V-A14B BF16 pretraining benchmark on
# Ironwood (tpu7x-2x4x4 / tpu7x-2x2x2) at 81x720x1280 (S=75,600 latent tokens).
#
# Verified results on tpu7x-2x2x2 (FSDP=4, CP=4; per-TensorCore step-time
# isomorphic to tpu7x-2x4x4 FSDP=16, CP=4 at per_device_batch_size=0.25):
#   - Stock tpu-recipes baseline (main @ eba2219a + MODE=baseline):
#       12.7255 s/step | 430.73 true TFLOP/s/TC (37.29% True MFU) | PJRT 21.03 GiB
#   - Phase E champion (cfd44a89):
#       11.7535 s/step | 466.35 true TFLOP/s/TC (40.38% True MFU) | PJRT 17.04 GiB
#   - Phase F champion (branch opt/wan22-train-tpu7x-diagon + MODE=optimized):
#       ~11.580 s/step on 2x2x2 (FSDP=4, Tier 1) | ~11.575 s/step on 2x4x4 (FSDP=16, Tier 2 raw_v)
#       473.1–473.5 true TFLOP/s/TC (40.96–41.00% True MFU) | PJRT 17.04 GiB (2x2x2) / 4.26 GiB (2x4x4)
#       (-173.3 ms/step same-node 40L A/B on FSDP<8; -178.2 ms/step 40L-eq on FSDP>=8;
#        ~-1.15 s/step = -9.0% step time / +9.9% throughput vs baseline, max_rel_diff<=0.00039)
#
# Usage:
#   BASE_OUTPUT_DIR=gs://<bucket>/wan22-repro \
#     bash end_to_end/tpu/run_wan22_tpu7x_train_repro.sh [optimized|baseline] [--dry-run] [key=value ...]
#
# XPK example (tpu7x-2x4x4, 32 chips / 64 TensorCores, FSDP=16, CP=4):
#   xpk workload create \
#     --cluster=<cluster> --project=<project> --zone=<zone> \
#     --workload=wan22-tpu7x-opt --tpu-type=tpu7x-2x4x4 \
#     --command="BASE_OUTPUT_DIR=gs://<bucket>/wan22-repro bash end_to_end/tpu/run_wan22_tpu7x_train_repro.sh optimized"
#
# Environment overrides:
#   BASE_OUTPUT_DIR         GCS or local output root (default: $HOME/wan22_tpu7x_repro)
#   JAX_CACHE_DIR           Persistent XLA compilation cache (default: $BASE_OUTPUT_DIR/jax_cache)
#   CONVERTED_WEIGHTS_DIR   Pre-converted Wan 2.2 weights dir (default: /dev/shm/converted_weights)
#   HF_HUB_CACHE            HuggingFace Hub cache dir (default: /dev/shm/hf_cache)
#   RUN_NAME                Run identifier (default: wan22-tpu7x-${MODE})
#   MAX_TRAIN_STEPS         Total steps to run (default: 51; steady-state median uses steps 25..48)
#   ICI_FSDP_PARALLELISM    FSDP axis size (-1 auto-fills devices_per_slice / 4: 16 on 2x4x4, 4 on 2x2x2)
#   ICI_CONTEXT_PARALLELISM Context/Ulysses axis size (default: 4)
#   WAN_TIER2_SAVE_QKV      Checkpoint tier selector for MODE=optimized (default: "auto"):
#                             "auto"  -> Tier 1 on FSDP<8 / <=2 hosts (2x2x2), Tier 2 ("raw_v") on FSDP>=8 / >=4 hosts (2x2x4, 2x4x4)
#                             "tier1" -> ["self_attn_output","self_attn","cross_attn","self_attn_proj","hidden_states","mlp_input","ffn_proj","ffn_out"]
#                             "raw_v" -> swaps "cross_attn" -> "raw_value_proj" (optimal when FSDP>=8 frees >=8.5 GiB/TC TrainState HBM)
#                             "vk"    -> adds "raw_key_proj" on top of "raw_v"
#                             "qkv"   -> adds "raw_key_proj","raw_query_proj" on top of "raw_v"
#   ENABLE_DIAGON           Set "true" to enable Google ML Diagnostics + on-demand XProf (default: false)
#   PROFILER_GCS_PATH       Diagon profile output path (default: $BASE_OUTPUT_DIR/diagon)
#   EXTRA_LIBTPU            Extra flags appended to LIBTPU_INIT_ARGS
#   DRY_RUN                 Set "1" (or pass --dry-run) to print the resolved command without running

set -euo pipefail

MODE="optimized"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    optimized | baseline)
      MODE="$arg"
      ;;
    --dry-run)
      DRY_RUN="1"
      ;;
    -h | --help)
      sed -n '16,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$HOME/wan22_tpu7x_repro}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$BASE_OUTPUT_DIR/jax_cache/}"
CONVERTED_WEIGHTS_DIR="${CONVERTED_WEIGHTS_DIR:-/dev/shm/converted_weights}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf_cache}"
RUN_NAME="${RUN_NAME:-wan22-tpu7x-${MODE}}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-51}"
ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:--1}"
ICI_CONTEXT_PARALLELISM="${ICI_CONTEXT_PARALLELISM:-4}"
ENABLE_DIAGON="${ENABLE_DIAGON:-false}"
PROFILER_GCS_PATH="${PROFILER_GCS_PATH:-$BASE_OUTPUT_DIR/diagon}"
LOG_DIR="${LOG_DIR:-/tmp/wan22_repro_${RUN_NAME}}"

# Tuned Ironwood (tpu7x) LIBTPU_INIT_ARGS from AI-Hypercomputer/tpu-recipes
# (training/ironwood/wan2.2-t2v-14b/bf16-tpu7x-2x4x4/xpk/run_recipe.sh).
RECIPE_LIBTPU_ARGS=(
  "--xla_tpu_bf16_emission_mode=NATIVE_EMISSION"
  "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true"
  "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true"
  "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
  "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true"
  "--xla_tpu_use_tc_device_shape_on_sc=true"
  "--xla_sc_disable_megacore_partitioning=true"
  "--xla_enable_async_all_gather=true"
  "--xla_max_concurrent_async_all_gathers=4"
  "--xla_tpu_enable_async_collective_fusion=false"
  "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
  "--xla_enable_async_all_reduce=true"
  "--xla_tpu_prefer_async_allgather_to_allreduce=true"
  "--xla_tpu_enable_sparse_core_collective_aggregator=true"
  "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3"
  "--xla_tpu_enable_all_gather_offload_tracing=true"
  "--xla_tpu_enable_async_all_to_all=true"
  "--xla_tpu_overlap_compute_collective_tc=true"
  "--xla_latency_hiding_scheduler_rerun=5"
  "--xla_tpu_enable_latency_hiding_layer_scheduler=true"
  "--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
  "--xla_tpu_scheduler_percent_shared_memory_limit=105"
  "--xla_tpu_rwb_fusion=false"
  "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
  "--xla_tpu_impure_enable_packed_bf16_math_ops=false"
  "--xla_tpu_scoped_vmem_limit_kib=65472"
  "--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true"
  "--xla_enable_transpose_trace=false"
)

DEFAULT_LIBTPU="${RECIPE_LIBTPU_ARGS[*]}"
if [[ "${MODE}" == "optimized" ]]; then
  DEFAULT_LIBTPU="${DEFAULT_LIBTPU} --xla_tpu_force_async_all_to_all=true"
fi
if [[ -n "${EXTRA_LIBTPU:-}" ]]; then
  DEFAULT_LIBTPU="${DEFAULT_LIBTPU} ${EXTRA_LIBTPU}"
fi
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-$DEFAULT_LIBTPU}"
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
export ENABLE_PJRT_COMPATIBILITY="${ENABLE_PJRT_COMPATIBILITY:-true}"

COMMON_ARGS=(
  "src/maxdiffusion/configs/base_wan_27b.yml"
  "model_name=wan2.2"
  "ulysses_shards=4"
  "mask_padding_tokens=False"
  "use_base2_exp=True"
  "weights_dtype=bfloat16"
  "activations_dtype=bfloat16"
  "guidance_scale_low=3.0"
  "guidance_scale_high=4.0"
  "boundary_ratio=0.875"
  "flow_shift=12.0"
  "fps=16"
  "skip_jax_distributed_system=False"
  "output_dir=${BASE_OUTPUT_DIR}/runs"
  "base_output_directory=${BASE_OUTPUT_DIR}/runs"
  "jax_cache_dir=${JAX_CACHE_DIR}"
  "converted_weights_dir=${CONVERTED_WEIGHTS_DIR}"
  "dataset_type=synthetic"
  "height=1280"
  "width=720"
  "num_frames=81"
  "num_inference_steps=40"
  "opt_enable_grad_global_norm_clipping=False"
  "max_train_steps=${MAX_TRAIN_STEPS}"
  "enable_profiler=False"
  "remat_policy=CUSTOM"
  "names_which_can_be_offloaded=[]"
  "flash_min_seq_length=0"
  "seed=123456789"
  "skip_first_n_steps_for_profiler=5"
  "profiler_steps=5"
  "per_device_batch_size=0.25"
  "ici_data_parallelism=1"
  "ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}"
  "ici_context_parallelism=${ICI_CONTEXT_PARALLELISM}"
  "ici_tensor_parallelism=1"
  "vae_spatial=1"
  "replicate_vae=True"
  "allow_split_physical_axes=True"
  "run_name=${RUN_NAME}"
)

if [[ "${ENABLE_DIAGON}" == "true" ]]; then
  COMMON_ARGS+=(
    "enable_ml_diagnostics=True"
    "enable_ondemand_xprof=True"
    "profiler_gcs_path=${PROFILER_GCS_PATH}"
  )
else
  COMMON_ARGS+=(
    "enable_ml_diagnostics=False"
    "enable_ondemand_xprof=False"
  )
fi

if [[ "${MODE}" == "optimized" ]]; then
  WAN_TIER2_SAVE_QKV="${WAN_TIER2_SAVE_QKV:-auto}"
  RESOLVED_TIER="${WAN_TIER2_SAVE_QKV}"
  if [[ "${RESOLVED_TIER}" == "auto" ]]; then
    if [[ "${ICI_FSDP_PARALLELISM}" != "-1" ]] && (( ICI_FSDP_PARALLELISM >= 8 )); then
      RESOLVED_TIER="raw_v"
    elif [[ "${ICI_FSDP_PARALLELISM}" != "-1" ]]; then
      RESOLVED_TIER="tier1"
    else
      # Count comma-separated hosts in TPU_WORKER_HOSTNAMES (>=4 hosts => >=32 TCs => FSDP>=8 at CP=4)
      IFS=',' read -r -a _tpu_hosts <<< "${TPU_WORKER_HOSTNAMES:-localhost}"
      if (( ${#_tpu_hosts[@]} >= 4 )); then
        RESOLVED_TIER="raw_v"
      else
        RESOLVED_TIER="tier1"
      fi
    fi
  fi
  case "${RESOLVED_TIER}" in
    tier1 | none)
      OPT_SAVED_NAMES='["self_attn_output","self_attn","cross_attn","self_attn_proj","hidden_states","mlp_input","ffn_proj","ffn_out"]'
      ;;
    raw_v)
      OPT_SAVED_NAMES='["self_attn_output","self_attn","self_attn_proj","hidden_states","mlp_input","ffn_proj","ffn_out","raw_value_proj"]'
      ;;
    vk)
      OPT_SAVED_NAMES='["self_attn_output","self_attn","self_attn_proj","hidden_states","mlp_input","ffn_proj","ffn_out","raw_value_proj","raw_key_proj"]'
      ;;
    qkv)
      OPT_SAVED_NAMES='["self_attn_output","self_attn","self_attn_proj","hidden_states","mlp_input","ffn_proj","ffn_out","raw_value_proj","raw_key_proj","raw_query_proj"]'
      ;;
    *)
      echo "ERROR: Invalid WAN_TIER2_SAVE_QKV='${WAN_TIER2_SAVE_QKV}' (expected: auto, tier1, raw_v, vk, qkv)" >&2
      exit 2
      ;;
  esac
  MODE_ARGS=(
    "attention=ulysses_custom_fixed_m_per_q_block"
    "use_k_centering=False"
    "ulysses_attention_chunks=5"
    "use_fused_rope_kernel=True"
    "names_which_can_be_saved=${OPT_SAVED_NAMES}"
    'flash_block_sizes={"block_q": 2816, "block_kv": 4096, "block_kv_compute": 4096, "block_kv_compute_in": 2048, "block_q_dkv": 2048, "block_kv_dkv": 4096, "block_kv_dkv_compute": 4096, "block_kv_dkv_compute_in": 512, "dq_reduction_steps": 2, "use_fused_bwd_kernel": true}'
  )
else
  MODE_ARGS=(
    "attention=ulysses_ring_custom"
    "ulysses_attention_chunks=2"
    "use_fused_rope_kernel=False"
    'names_which_can_be_saved=["attn_output","self_attn","cross_attn","hidden_states"]'
    'flash_block_sizes={"block_q": 2816, "block_kv": 4096, "block_kv_compute": 4096, "block_kv_compute_in": 4096, "block_q_dkv": 2048, "block_kv_dkv": 4096, "block_kv_dkv_compute": 4096, "block_kv_dkv_compute_in": 256, "dq_reduction_steps": 3, "use_fused_bwd_kernel": true}'
  )
fi

ALL_ARGS=("${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "# Mode: ${MODE}"
  echo "export LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS}'"
  printf 'python3 -u src/maxdiffusion/train_wan.py'
  for a in "${ALL_ARGS[@]}"; do
    printf ' \\\n  %q' "$a"
  done
  printf '\n'
  exit 0
fi

mkdir -p "${LOG_DIR}" "${CONVERTED_WEIGHTS_DIR}" "${HF_HUB_CACHE}"
TRAIN_LOG="${LOG_DIR}/train.log"
LOSS_JSON="${LOG_DIR}/loss_curve.json"

echo "[wan22-repro] Running MODE=${MODE} RUN_NAME=${RUN_NAME} -> ${TRAIN_LOG}"
python3 -u - "${ALL_ARGS[@]}" << 'PYWRAP' 2>&1 | tee "${TRAIN_LOG}"
import sys
import time
from absl import app
import jax
from jax.experimental import multihost_utils
from maxdiffusion import train_wan
from maxdiffusion.train_utils import transformer_engine_context


def _wrapped_main(argv):
  t0 = time.perf_counter()
  train_wan.main(argv)
  total_s = time.perf_counter() - t0
  peaks = []
  for d in jax.local_devices():
    ms = d.memory_stats() or {}
    if "peak_bytes_in_use" in ms:
      peaks.append(ms["peak_bytes_in_use"] / (1024**3))
  peak_gib = max(peaks) if peaks else 0.0
  if jax.process_index() == 0:
    print(f"WRAPPER_PEAK_HBM_GIB={peak_gib:.4f} WRAPPER_TOTAL_S={total_s:.2f}", flush=True)
  multihost_utils.sync_global_devices("end_of_training_barrier")


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(_wrapped_main)
PYWRAP

# Summarize steady-state step time (steps 25..48), True TFLOP/s/TC, and MFU on rank 0.
python3 - "${TRAIN_LOG}" "${LOSS_JSON}" "${RUN_NAME}" << 'PYSUM'
import json
import re
import statistics
import sys

log_path, loss_json_path, run_name = sys.argv[1], sys.argv[2], sys.argv[3]
# True non-remat algorithmic FLOPs per TensorCore at S=75,600 (per_device_batch_size=0.25, CP=4):
TRUE_TFLOPS_PER_TC = 5481.26
IRONWOOD_BF16_PEAK_TFLOPS_PER_TC = 1155.0

step_re = re.compile(
    r"completed step:\s*(\d+),\s*seconds:\s*([0-9.]+),\s*TFLOP/s/device:\s*([0-9.]+),\s*loss:\s*([-+0-9.eE]+)"
)
peak_re = re.compile(r"WRAPPER_PEAK_HBM_GIB=([0-9.]+)")
compile_re = re.compile(r"Pre-compiled per-expert training executables in ([0-9.]+)s")

steps = {}
peak_gib = 0.0
compile_s = 0.0
with open(log_path, encoding="utf-8", errors="replace") as f:
  for line in f:
    m = step_re.search(line)
    if m:
      steps[int(m.group(1))] = (float(m.group(2)), float(m.group(4)))
    mp = peak_re.search(line)
    if mp:
      peak_gib = float(mp.group(1))
    mc = compile_re.search(line)
    if mc:
      compile_s = float(mc.group(1))

if not steps:
  sys.exit(0)

steady = [steps[s][0] for s in sorted(steps) if 25 <= s <= 48]
if not steady:
  steady = [v[0] for k, v in sorted(steps.items())[min(5, len(steps) - 1) :]]

median_s = statistics.median(steady)
true_tflops_s = TRUE_TFLOPS_PER_TC / median_s
true_mfu_pct = 100.0 * true_tflops_s / IRONWOOD_BF16_PEAK_TFLOPS_PER_TC

blocks = [statistics.mean(steady[i : i + 6]) for i in range(0, len(steady) - 5, 6)]
block_cv_pct = (
    100.0 * statistics.pstdev(blocks) / statistics.mean(blocks) if len(blocks) >= 2 else 0.0
)

payload = {
    "run_name": run_name,
    "median_step_time_s": round(median_s, 4),
    "steady_step_times_s": steady,
    "peak_hbm_gib": round(peak_gib, 4),
    "compile_s": round(compile_s, 2),
    "true_tflops_per_sec": round(true_tflops_s, 2),
    "true_mfu_pct": round(true_mfu_pct, 2),
    "losses": [{"step": s, "loss": steps[s][1]} for s in sorted(steps)],
}
with open(loss_json_path, "w", encoding="utf-8") as out:
  json.dump(payload, out, indent=2)

print(
    f"SUMMARY run_name={run_name} steady_n={len(steady)} block_cv_pct={block_cv_pct:.3f} "
    f"true_tflops_per_s={true_tflops_s:.2f} true_mfu_pct={true_mfu_pct:.2f}"
)
print(f"METRIC step_time_s={median_s:.4f}")
print(f"METRIC peak_hbm_gib={peak_gib:.4f}")
print(f"METRIC compile_s={compile_s:.4f}")
print(f"METRIC true_tflops_per_s={true_tflops_s:.2f}")
print(f"[wan22-repro] Wrote metrics & 50-step loss curve to {loss_json_path}")
PYSUM
