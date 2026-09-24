#!/usr/bin/env bash
#
# Runner script for LTX-2 Mini-VBench (110 Prompts) Benchmark
# Runs Case 1 (Dense Ulysses) and Case 2 (SVG Ulysses) sequentially on jalwaniya-v6e-8.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate JAX TPU virtual environment
# shellcheck disable=SC1091
source /home/jalwaniya_google_com/maxdiffusion_venv/bin/activate

mkdir -p outputs/vbench_ltx2
mkdir -p /mnt/disks/external_disk/aot_cache_ltx2

COMMON_ARGS=(
  "src/maxdiffusion/configs/ltx2_video.yml"
  "prompt_file=benchmarks/vbench/prompts_110.txt"
  "base_output_directory=gs://jalwaniya-vbench-videos"
  "seed=12345"
  "num_inference_steps=40"
  "height=768"
  "width=1280"
  "num_frames=121"
  "fps=24"
  "attention=ulysses_custom"
  "ici_data_parallelism=1"
  "ici_context_parallelism=8"
  "per_device_batch_size=0.125"
  "global_batch_size_to_train_on=1"
  "enable_vae_tiling=True"
  "use_batched_text_encoder=True"
  "enable_profiler=False"
  "enable_ml_diagnostics=False"
  "aot_cache_dir=/mnt/disks/external_disk/aot_cache_ltx2"
)

echo "=========================================================================="
echo "Starting LTX-2 Mini-VBench (110 Prompts) Execution"
echo "  Target Resolution: 768x1280x121 (N=15,360 tokens)"
echo "  GCS Bucket:        gs://jalwaniya-vbench-videos"
echo "  Start Time:        $(date -u)"
echo "=========================================================================="

# -----------------------------------------------------------------------------
# Case 1: Dense Ulysses Attention (use_svg_attention=False)
# -----------------------------------------------------------------------------
RUN_NAME_1="ltx2-vbench-case1-dense"
LOG_FILE_1="outputs/vbench_ltx2/${RUN_NAME_1}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 1: Dense Ulysses Attention (${RUN_NAME_1})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_1}" \
  use_svg_attention=False 2>&1 | tee "${LOG_FILE_1}"

echo "Uploading benchmark metadata for ${RUN_NAME_1}..."
gcloud storage cp benchmarks/vbench/VBench_full_info_sub110.json \
  "gs://jalwaniya-vbench-videos/${RUN_NAME_1}/VBench_full_info_sub110.json"

COUNT_1=$(gcloud storage ls "gs://jalwaniya-vbench-videos/${RUN_NAME_1}/videos/*.mp4" 2>/dev/null | wc -l || echo 0)
echo "Case 1 Complete. Videos in GCS: ${COUNT_1}/110"
rm -f ltx2_video_output_*.mp4

# -----------------------------------------------------------------------------
# Case 2: SVG Ulysses Attention (use_svg_attention=True)
# -----------------------------------------------------------------------------
RUN_NAME_2="ltx2-vbench-case2-svg"
LOG_FILE_2="outputs/vbench_ltx2/${RUN_NAME_2}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 2: SVG Ulysses Attention (${RUN_NAME_2})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_2}" \
  use_svg_attention=True 2>&1 | tee "${LOG_FILE_2}"

echo "Uploading benchmark metadata for ${RUN_NAME_2}..."
gcloud storage cp benchmarks/vbench/VBench_full_info_sub110.json \
  "gs://jalwaniya-vbench-videos/${RUN_NAME_2}/VBench_full_info_sub110.json"

COUNT_2=$(gcloud storage ls "gs://jalwaniya-vbench-videos/${RUN_NAME_2}/videos/*.mp4" 2>/dev/null | wc -l || echo 0)
echo "Case 2 Complete. Videos in GCS: ${COUNT_2}/110"
rm -f ltx2_video_output_*.mp4

echo ""
echo "=========================================================================="
echo "All LTX-2 TPU generations finished!"
echo "  End Time: $(date -u)"
echo "  Case 1: gs://jalwaniya-vbench-videos/${RUN_NAME_1}/videos/ (${COUNT_1} videos)"
echo "  Case 2: gs://jalwaniya-vbench-videos/${RUN_NAME_2}/videos/ (${COUNT_2} videos)"
echo "=========================================================================="
