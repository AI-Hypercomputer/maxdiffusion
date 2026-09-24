#!/usr/bin/env bash
#
# Runner script for LTX-2 VABench Smoke Test (5 Prompts @ 768x1280x241, N=29,760 tokens)
# Runs Case 1 (Dense Ulysses) and Case 2 (SVG Ulysses) sequentially on jalwaniya-v6e-8.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate JAX TPU virtual environment
# shellcheck disable=SC1091
source /home/jalwaniya_google_com/maxdiffusion_venv/bin/activate

mkdir -p outputs/vabench_ltx2_241f
mkdir -p /mnt/disks/external_disk/aot_cache_ltx2

COMMON_ARGS=(
  "src/maxdiffusion/configs/ltx2_video.yml"
  "prompt_file=benchmarks/vabench/prompts_smoke5.txt"
  "base_output_directory=gs://jalwaniya-vbench-videos/vabench"
  "seed=12345"
  "num_inference_steps=40"
  "height=768"
  "width=1280"
  "num_frames=241"
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
  'flash_block_sizes={"block_q": 2048, "block_kv": 1280, "block_kv_compute": 1280, "block_q_dkv": 2048, "block_kv_dkv": 1280, "block_kv_dkv_compute": 1280, "use_fused_bwd_kernel": true}'
  "aot_cache_dir=/mnt/disks/external_disk/aot_cache_ltx2"
)

echo "=========================================================================="
echo "Starting LTX-2 VABench Smoke Test (5 Prompts)"
echo "  Target Resolution: 768x1280x241 (N=29,760 tokens, 10.04s @ 24fps + 24kHz audio)"
echo "  GCS Bucket:        gs://jalwaniya-vbench-videos/vabench"
echo "  Git Commit:        $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "  Start Time:        $(date -u)"
echo "=========================================================================="

# -----------------------------------------------------------------------------
# Case 1: Dense Ulysses Attention (use_svg_attention=False)
# -----------------------------------------------------------------------------
RUN_NAME_1="ltx2-vabench-dense-smoke5"
LOG_FILE_1="outputs/vabench_ltx2_241f/${RUN_NAME_1}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 1: Dense Ulysses Attention (${RUN_NAME_1})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_1}" \
  use_svg_attention=False 2>&1 | tee "${LOG_FILE_1}"

echo "Uploading benchmark metadata for ${RUN_NAME_1}..."
gcloud storage cp benchmarks/vabench/vabench_smoke5.json \
  "gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_1}/vabench_smoke5.json"

COUNT_1=$(gcloud storage ls "gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_1}/videos/*.mp4" 2>/dev/null | wc -l || echo 0)
echo "Case 1 Complete. Videos in GCS: ${COUNT_1}/5"
rm -f ltx2_video_output_*.mp4

# -----------------------------------------------------------------------------
# Case 2: SVG Ulysses Attention (use_svg_attention=True)
# -----------------------------------------------------------------------------
RUN_NAME_2="ltx2-vabench-svg-smoke5"
LOG_FILE_2="outputs/vabench_ltx2_241f/${RUN_NAME_2}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 2: SVG Ulysses Attention (${RUN_NAME_2})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_2}" \
  use_svg_attention=True 2>&1 | tee "${LOG_FILE_2}"

echo "Uploading benchmark metadata for ${RUN_NAME_2}..."
gcloud storage cp benchmarks/vabench/vabench_smoke5.json \
  "gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_2}/vabench_smoke5.json"

COUNT_2=$(gcloud storage ls "gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_2}/videos/*.mp4" 2>/dev/null | wc -l || echo 0)
echo "Case 2 Complete. Videos in GCS: ${COUNT_2}/5"
rm -f ltx2_video_output_*.mp4

echo ""
echo "=========================================================================="
echo "All LTX-2 VABench Smoke-5 TPU generations finished!"
echo "  End Time: $(date -u)"
echo "  Case 1: gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_1}/videos/ (${COUNT_1} videos)"
echo "  Case 2: gs://jalwaniya-vbench-videos/vabench/${RUN_NAME_2}/videos/ (${COUNT_2} videos)"
echo "=========================================================================="
