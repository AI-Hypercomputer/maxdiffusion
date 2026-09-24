#!/usr/bin/env bash
#
# Runner script for Phase 2: LTX-2 Mini-VABench (78 Stratified Prompts — 10% VABench)
# Resolution: 768x1280x241 (N=29,760 tokens, 10.04s @ 24fps + 24kHz audio)
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

# Pre-upload vabench_sub78.json metadata to GCS using Python storage client
python3 -c '
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("jalwaniya-vbench-videos")
for rn in ["ltx2-vabench-dense-241f-sub78", "ltx2-vabench-svg-241f-sub78"]:
    blob = bucket.blob(f"vabench/{rn}/vabench_sub78.json")
    blob.upload_from_filename("benchmarks/vabench/vabench_sub78.json")
print("Uploaded vabench_sub78.json for both Phase 2 cases via Python GCS client.")
'

COMMON_ARGS=(
  "src/maxdiffusion/configs/ltx2_video.yml"
  "prompt_file=benchmarks/vabench/prompts_78.txt"
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
echo "Starting Phase 2: LTX-2 Mini-VABench (78 Prompts — 10% Stratified Suite)"
echo "  Target Resolution: 768x1280x241 (N=29,760 tokens, 10.04s @ 24fps + 24kHz audio)"
echo "  GCS Bucket:        gs://jalwaniya-vbench-videos/vabench"
echo "  Git Commit:        $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "  Start Time:        $(date -u)"
echo "=========================================================================="

# -----------------------------------------------------------------------------
# Case 1: Dense Ulysses Attention (use_svg_attention=False)
# -----------------------------------------------------------------------------
RUN_NAME_1="ltx2-vabench-dense-241f-sub78"
LOG_FILE_1="outputs/vabench_ltx2_241f/${RUN_NAME_1}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 1: Dense Ulysses Attention (${RUN_NAME_1})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_1}" \
  use_svg_attention=False 2>&1 | tee "${LOG_FILE_1}"

echo "Case 1 (${RUN_NAME_1}) Complete at $(date -u)."
rm -f ltx2_video_output_*.mp4

# -----------------------------------------------------------------------------
# Case 2: SVG Ulysses Attention (use_svg_attention=True)
# -----------------------------------------------------------------------------
RUN_NAME_2="ltx2-vabench-svg-241f-sub78"
LOG_FILE_2="outputs/vabench_ltx2_241f/${RUN_NAME_2}.log"

echo ""
echo "--------------------------------------------------------------------------"
echo "Starting Case 2: SVG Ulysses Attention (${RUN_NAME_2})"
echo "--------------------------------------------------------------------------"
rm -f ltx2_video_output_*.mp4

python3 src/maxdiffusion/generate_ltx2.py "${COMMON_ARGS[@]}" \
  run_name="${RUN_NAME_2}" \
  use_svg_attention=True 2>&1 | tee "${LOG_FILE_2}"

echo "Case 2 (${RUN_NAME_2}) Complete at $(date -u)."
rm -f ltx2_video_output_*.mp4

echo ""
echo "=========================================================================="
echo "All Phase 2 LTX-2 Mini-VABench (78 Prompts) TPU generations finished!"
echo "  End Time: $(date -u)"
echo "=========================================================================="
