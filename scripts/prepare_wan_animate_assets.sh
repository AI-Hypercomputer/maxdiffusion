#!/usr/bin/env bash
set -euo pipefail

# Download official Wan2.2-Animate sample assets and stage them for local preprocessing/inference.
#
# Default sample: examples/mix/1/{1.mp4,1.jpeg}
# Alternative sample: --sample mov (examples/mov/1/{1.mp4,1.jpeg})
#
# If --run-preprocess is supplied, this script runs the official Wan2.2 preprocessing script
# (from a local Wan2.2 checkout) and emits preprocessed files under process_results.

SAMPLE="mix"
MODE="animate"
ASSETS_DIR="${WAN_ASSETS_DIR:-./wan_assets}"
TARGET_ROOT="./examples/wan_animate"
RUN_PREPROCESS=0
WAN22_ROOT="${WAN22_ROOT:-}"
CKPT_PATH="${CKPT_PATH:-}"
DOWNLOAD_PROCESS_CKPT=0
PROCESS_CKPT_REPO="Wan-AI/Wan2.2-Animate-14B"
DEFAULT_PROCESS_CKPT_ROOT="/mnt/data/maxdiffusion/checkpoints/Wan2.2-Animate-14B"
USE_FLUX=0

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_wan_animate_assets.sh [options]

Options:
  --sample <mix|mov>          Source sample folder in the official HF Space (default: mix)
  --mode <animate|replace>    Local target mode folder under examples/wan_animate (default: animate)
  --assets-dir <path>         Where to download original assets (default: ./wan_assets)
  --target-root <path>        Root folder for wan_animate examples (default: ./examples/wan_animate)
  --run-preprocess            Run official Wan2.2 preprocessing after staging raw files
  --wan22-root <path>         Local path to Wan2.2 repo (required with --run-preprocess)
  --ckpt-path <path>          Path to Wan2.2 preprocess checkpoint dir
  --download-process-ckpt     Download process_checkpoint/* from Wan-AI/Wan2.2-Animate-14B
  --process-ckpt-repo <repo>  Override checkpoint repo for --download-process-ckpt
  --use-flux                  Enable --use_flux during animate preprocessing (requires FLUX.1-Kontext-dev in ckpt path)
  -h, --help                  Show this help

Environment alternatives:
  WAN22_ROOT=/path/to/Wan2.2
  CKPT_PATH=/path/to/Wan2.2-Animate-14B/process_checkpoint
  WAN_ASSETS_DIR=/mnt/data/maxdiffusion/wan_assets
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample)
      SAMPLE="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --assets-dir)
      ASSETS_DIR="${2:-}"
      shift 2
      ;;
    --target-root)
      TARGET_ROOT="${2:-}"
      shift 2
      ;;
    --run-preprocess)
      RUN_PREPROCESS=1
      shift
      ;;
    --wan22-root)
      WAN22_ROOT="${2:-}"
      shift 2
      ;;
    --ckpt-path)
      CKPT_PATH="${2:-}"
      shift 2
      ;;
    --download-process-ckpt)
      DOWNLOAD_PROCESS_CKPT=1
      shift
      ;;
    --process-ckpt-repo)
      PROCESS_CKPT_REPO="${2:-}"
      shift 2
      ;;
    --use-flux)
      USE_FLUX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$SAMPLE" != "mix" && "$SAMPLE" != "mov" ]]; then
  echo "Invalid --sample '$SAMPLE'. Use 'mix' or 'mov'." >&2
  exit 1
fi

if [[ "$MODE" != "animate" && "$MODE" != "replace" ]]; then
  echo "Invalid --mode '$MODE'. Use 'animate' or 'replace'." >&2
  exit 1
fi

RAW_VIDEO_RELPATH="examples/${SAMPLE}/1/1.mp4"
RAW_IMAGE_RELPATH="examples/${SAMPLE}/1/1.jpeg"
RAW_DST_DIR="${TARGET_ROOT}/${MODE}"
RAW_VIDEO_DST="${RAW_DST_DIR}/video.mp4"
RAW_IMAGE_DST="${RAW_DST_DIR}/image.jpeg"
PROCESS_RESULTS_DIR="${RAW_DST_DIR}/process_results"

mkdir -p "${ASSETS_DIR}" "${RAW_DST_DIR}"

echo "Downloading official sample assets from Wan-AI/Wan2.2-Animate (space)..."
if command -v hf >/dev/null 2>&1; then
  hf download \
    --repo-type space \
    Wan-AI/Wan2.2-Animate \
    "${RAW_VIDEO_RELPATH}" \
    "${RAW_IMAGE_RELPATH}" \
    --local-dir "${ASSETS_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download \
    --repo-type space \
    Wan-AI/Wan2.2-Animate \
    "${RAW_VIDEO_RELPATH}" \
    "${RAW_IMAGE_RELPATH}" \
    --local-dir "${ASSETS_DIR}"
else
  echo "Neither 'hf' nor 'huggingface-cli' is available in PATH." >&2
  echo "Install huggingface_hub CLI, then retry." >&2
  exit 1
fi

cp -f "${ASSETS_DIR}/${RAW_VIDEO_RELPATH}" "${RAW_VIDEO_DST}"
cp -f "${ASSETS_DIR}/${RAW_IMAGE_RELPATH}" "${RAW_IMAGE_DST}"

echo "Staged raw assets:"
echo "  ${RAW_VIDEO_DST}"
echo "  ${RAW_IMAGE_DST}"

if [[ "${DOWNLOAD_PROCESS_CKPT}" -eq 1 ]]; then
  PROCESS_CKPT_ROOT="${DEFAULT_PROCESS_CKPT_ROOT}"
  mkdir -p "${PROCESS_CKPT_ROOT}"
  echo "Downloading preprocess checkpoint to: ${PROCESS_CKPT_ROOT}"

  if command -v hf >/dev/null 2>&1; then
    hf download \
      "${PROCESS_CKPT_REPO}" \
      --include "process_checkpoint/*" \
      --local-dir "${PROCESS_CKPT_ROOT}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download \
      "${PROCESS_CKPT_REPO}" \
      --include "process_checkpoint/*" \
      --local-dir "${PROCESS_CKPT_ROOT}"
  else
    echo "Neither 'hf' nor 'huggingface-cli' is available in PATH." >&2
    exit 1
  fi

  CKPT_PATH="${PROCESS_CKPT_ROOT}/process_checkpoint"
  echo "Checkpoint path set to: ${CKPT_PATH}"
fi

if [[ "${RUN_PREPROCESS}" -eq 1 ]]; then
  if [[ -z "${CKPT_PATH}" && -d "${DEFAULT_PROCESS_CKPT_ROOT}/process_checkpoint" ]]; then
    CKPT_PATH="${DEFAULT_PROCESS_CKPT_ROOT}/process_checkpoint"
  fi
  if [[ -z "${WAN22_ROOT}" || -z "${CKPT_PATH}" ]]; then
    echo "--run-preprocess requires --wan22-root and --ckpt-path (or WAN22_ROOT/CKPT_PATH env vars)." >&2
    exit 1
  fi
  if [[ ! -f "${WAN22_ROOT}/wan/modules/animate/preprocess/preprocess_data.py" ]]; then
    echo "Could not find official preprocess script at:" >&2
    echo "  ${WAN22_ROOT}/wan/modules/animate/preprocess/preprocess_data.py" >&2
    exit 1
  fi

  mkdir -p "${PROCESS_RESULTS_DIR}"
  echo "Running official Wan2.2 preprocessing..."

  PREPROCESS_CMD=(
    python "${WAN22_ROOT}/wan/modules/animate/preprocess/preprocess_data.py"
    --ckpt_path "${CKPT_PATH}"
    --video_path "${RAW_VIDEO_DST}"
    --refer_path "${RAW_IMAGE_DST}"
    --save_path "${PROCESS_RESULTS_DIR}"
    --resolution_area 1280 720
  )

  if [[ "${MODE}" == "animate" ]]; then
    PREPROCESS_CMD+=(--retarget_flag)
    if [[ "${USE_FLUX}" -eq 1 ]]; then
      PREPROCESS_CMD+=(--use_flux)
    fi
  else
    PREPROCESS_CMD+=(--iterations 3 --k 7 --w_len 1 --h_len 1 --replace_flag)
  fi

  "${PREPROCESS_CMD[@]}"

  echo "Preprocess outputs are under:"
  echo "  ${PROCESS_RESULTS_DIR}"
  echo "Expected files include pose/face controls (for example src_pose.mp4 and src_face.mp4)."
fi

echo "Done."
