#!/usr/bin/env bash
#
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
#
# ==============================================================================
# VBench GPU Evaluation Script
#
# Usage:
#   bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=<bucket> [KEY=VALUE ...]
#   bash benchmarks/vbench/run_gpu_eval.sh --ssh GCS_BUCKET=<bucket> GPU_NAME=<gpu-vm> [KEY=VALUE ...]
# ==============================================================================

set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:-}"
RUN_NAME="${RUN_NAME:-wan-inference-aug-31-1}"
GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-}"
GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-true}"
WORK_DIR_FROM_ARG=""
local_results_parent=""

VBENCH_REPO="${VBENCH_REPO:-https://github.com/Vchitect/VBench.git}"
VBENCH_BRANCH="${VBENCH_BRANCH:-master}"
MAXDIFFUSION_REPO="${MAXDIFFUSION_REPO:-https://github.com/AI-Hypercomputer/maxdiffusion.git}"
MAXDIFFUSION_BRANCH="${MAXDIFFUSION_BRANCH:-main}"
BENCHMARK_JSON="${BENCHMARK_JSON:-VBench_full_info_sub110.json}"
BENCHMARK_JSON_URL="${BENCHMARK_JSON_URL:-}"
BENCHMARK_JSON_PATH="${BENCHMARK_JSON_PATH:-}"
VBENCH_UTIL_PATH="${VBENCH_UTIL_PATH:-}"

GPU_NAME="${GPU_NAME:-}"
GPU_ZONE="${GPU_ZONE:-us-central1-a}"
GPU_PROJECT="${GPU_PROJECT:-}"
INTERNAL_IP="${INTERNAL_IP:-false}"
SSH_MODE=false

DIMENSIONS_TEXT="${DIMENSIONS:-}"
DIMENSIONS=()
if [[ -n "${DIMENSIONS_TEXT}" ]]; then
  # shellcheck disable=SC2206
  DIMENSIONS=(${DIMENSIONS_TEXT})
fi

usage() {
  cat <<EOF
Usage: $0 [--ssh] GCS_BUCKET=<bucket_name> [KEY=VALUE ...]

Required:
  GCS_BUCKET         GCS bucket containing generated videos.

Common options:
  RUN_NAME           Generation run name (default: ${RUN_NAME})
  BENCHMARK_JSON     Benchmark JSON file name (default: ${BENCHMARK_JSON})
  DIMENSIONS         Space-separated VBench dimensions (default: read from JSON)
  GCS_VIDEO_DIR      GCS video prefix (default: \${RUN_NAME}/videos)
  GCS_RESULTS_DIR    GCS results prefix (default: \${RUN_NAME}/vbench_results)
  UPLOAD_RESULTS     Upload results to GCS after evaluation (default: true; SSH mode uploads from the caller machine)
  WORK_DIR           Working directory on the GPU VM (default: \$HOME/vbench_evaluation)

Metadata options:
  BENCHMARK_JSON_PATH  Local benchmark JSON path.
  BENCHMARK_JSON_URL   URL fallback for benchmark JSON.
  MAXDIFFUSION_REPO    Repo fallback for benchmark JSON.
  MAXDIFFUSION_BRANCH  Branch fallback for benchmark JSON.

SSH options:
  --ssh              Copy this script to a GPU VM and run it there.
  GPU_NAME           GPU VM name.
  GPU_ZONE           GPU VM zone (default: ${GPU_ZONE})
  GPU_PROJECT        Optional GCP project for the GPU VM.
  INTERNAL_IP        Connect to GPU VM using internal IP (default: false).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

step() {
  echo "==> $*"
}

parse_args() {
  local arg key value
  for arg in "$@"; do
    case "${arg}" in
      --ssh)
        SSH_MODE=true
        ;;
      --help | -h)
        usage
        exit 0
        ;;
      WORK_DIR=*)
        WORK_DIR_FROM_ARG="${arg#*=}"
        export WORK_DIR="${WORK_DIR_FROM_ARG}"
        ;;
      DIMENSIONS=*)
        value="${arg#*=}"
        # shellcheck disable=SC2206
        DIMENSIONS=(${value})
        ;;
      *=*)
        key="${arg%%=*}"
        value="${arg#*=}"
        export "${key}=${value}"
        ;;
      *)
        if [[ -z "${GCS_BUCKET}" ]]; then
          GCS_BUCKET="${arg}"
        fi
        ;;
    esac
  done
}

absolute_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "File not found: ${path}"
  local dir base
  dir="$(cd "$(dirname "${path}")" && pwd)"
  base="$(basename "${path}")"
  printf '%s/%s\n' "${dir}" "${base}"
}

normalize_config() {
  [[ -n "${GCS_BUCKET}" ]] || die "GCS_BUCKET is required. Example: bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=my-bucket"

  GCS_BUCKET="${GCS_BUCKET#gs://}"
  GCS_BUCKET="${GCS_BUCKET%/}"
  GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-${RUN_NAME}/videos}"
  GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-${RUN_NAME}/vbench_results}"
  [[ "${UPLOAD_RESULTS}" == "true" || "${UPLOAD_RESULTS}" == "false" ]] || die "UPLOAD_RESULTS must be true or false."
  WORK_DIR="${WORK_DIR_FROM_ARG:-${WORK_DIR:-$HOME/vbench_evaluation}}"
  BENCHMARK_JSON_URL="${BENCHMARK_JSON_URL:-https://raw.githubusercontent.com/AI-Hypercomputer/maxdiffusion/${MAXDIFFUSION_BRANCH}/benchmarks/vbench/${BENCHMARK_JSON}}"

  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    BENCHMARK_JSON_PATH="$(absolute_file "${BENCHMARK_JSON_PATH}")"
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  UTIL_SCRIPT="${VBENCH_UTIL_PATH:-${SCRIPT_DIR}/gpu_eval_utils.py}"
  UTIL_SCRIPT="$(absolute_file "${UTIL_SCRIPT}")"

  JSON_DEST="${WORK_DIR}/${BENCHMARK_JSON}"
  DOWNLOAD_DIR="${WORK_DIR}/downloaded_videos"
  VBENCH_VIDEO_DIR="${WORK_DIR}/vbench_videos"
  RESULTS_DIR="${WORK_DIR}/evaluation_results"
}

print_config() {
  echo "=========================================================================="
  echo "Starting VBench evaluation on GPU VM"
  echo "  GCS Bucket:      gs://${GCS_BUCKET}"
  echo "  Video Source:    gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "  Results Target:  gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
  echo "  Working Dir:     ${WORK_DIR}"
  echo "  Benchmark JSON:  ${BENCHMARK_JSON}"
  echo "  Dimensions:      ${DIMENSIONS[*]:-(auto-extracted from ${BENCHMARK_JSON})}"
  echo "=========================================================================="
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

quote_command() {
  local command="$1"
  shift
  local arg
  for arg in "$@"; do
    command="${command} $(shell_quote "${arg}")"
  done
  printf '%s\n' "${command}"
}

run_over_ssh() {
  [[ -n "${GPU_NAME}" ]] || die "GPU_NAME must be specified when using --ssh mode."

  local source_script remote_script remote_util remote_command remote_json
  local gcloud_args=("--zone=${GPU_ZONE}")
  source_script="$(absolute_file "${BASH_SOURCE[0]}")"
  remote_script="/tmp/run_gpu_eval_${USER:-user}_$$.sh"
  remote_util="/tmp/gpu_eval_utils_${USER:-user}_$$.py"
  if [[ -n "${GPU_PROJECT}" ]]; then
    gcloud_args+=("--project=${GPU_PROJECT}")
  fi
  if [[ "${INTERNAL_IP}" == "true" ]]; then
    gcloud_args+=("--internal-ip")
  fi

  echo "=========================================================================="
  echo "Executing VBench evaluation remotely"
  echo "  GPU Name:    ${GPU_NAME}"
  echo "  GPU Zone:    ${GPU_ZONE}"
  echo "  GCS Bucket:  gs://${GCS_BUCKET}"
  echo "  Video Dir:   gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "  Results Dir: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
  echo "=========================================================================="

  local scp_script_cmd=("gcloud" "compute" "scp" "${source_script}" "${GPU_NAME}:${remote_script}" "${gcloud_args[@]}")
  "${scp_script_cmd[@]}"

  local scp_util_cmd=("gcloud" "compute" "scp" "${UTIL_SCRIPT}" "${GPU_NAME}:${remote_util}" "${gcloud_args[@]}")
  "${scp_util_cmd[@]}"

  local remote_args=(
    "GCS_BUCKET=${GCS_BUCKET}"
    "RUN_NAME=${RUN_NAME}"
    "GCS_VIDEO_DIR=${GCS_VIDEO_DIR}"
    "GCS_RESULTS_DIR=${GCS_RESULTS_DIR}"
    "VBENCH_REPO=${VBENCH_REPO}"
    "VBENCH_BRANCH=${VBENCH_BRANCH}"
    "MAXDIFFUSION_REPO=${MAXDIFFUSION_REPO}"
    "MAXDIFFUSION_BRANCH=${MAXDIFFUSION_BRANCH}"
    "BENCHMARK_JSON=${BENCHMARK_JSON}"
    "BENCHMARK_JSON_URL=${BENCHMARK_JSON_URL}"
    "VBENCH_UTIL_PATH=${remote_util}"
    "UPLOAD_RESULTS=false"
  )

  if [[ -n "${WORK_DIR_FROM_ARG}" ]]; then
    remote_args+=("WORK_DIR=${WORK_DIR_FROM_ARG}")
  fi
  if [[ ${#DIMENSIONS[@]} -gt 0 ]]; then
    remote_args+=("DIMENSIONS=${DIMENSIONS[*]}")
  fi
  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    remote_json="/tmp/${BENCHMARK_JSON}"
    local scp_json_cmd=("gcloud" "compute" "scp" "${BENCHMARK_JSON_PATH}" "${GPU_NAME}:${remote_json}" "${gcloud_args[@]}")
    "${scp_json_cmd[@]}"
    remote_args+=("BENCHMARK_JSON_PATH=${remote_json}")
  fi

  remote_command="$(quote_command "bash $(shell_quote "${remote_script}")" "${remote_args[@]}")"

  local ssh_cmd=("gcloud" "compute" "ssh" "${GPU_NAME}" "${gcloud_args[@]}")
  ssh_cmd+=("--command=${remote_command}")
  "${ssh_cmd[@]}"

  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    local remote_results_dir previous_results_dir
    if [[ -n "${WORK_DIR_FROM_ARG}" ]]; then
      remote_results_dir="${WORK_DIR_FROM_ARG%/}/evaluation_results"
      remote_results_dir="${remote_results_dir#~/}"
    else
      remote_results_dir="vbench_evaluation/evaluation_results"
    fi

    local_results_parent="$(mktemp -d "${TMPDIR:-/tmp}/vbench_results.XXXXXX")"
    trap 'rm -rf "${local_results_parent}"' EXIT
    step "Copying evaluation results from ${GPU_NAME}:${remote_results_dir}..."
    local scp_results_cmd=("gcloud" "compute" "scp" "--recurse" "${GPU_NAME}:${remote_results_dir}" "${local_results_parent}/" "${gcloud_args[@]}")
    "${scp_results_cmd[@]}"

    previous_results_dir="${RESULTS_DIR}"
    RESULTS_DIR="${local_results_parent}/evaluation_results"
    upload_results
    RESULTS_DIR="${previous_results_dir}"
    rm -rf "${local_results_parent}"
    trap - EXIT
  fi
}

python_supports_vbench() {
  "$1" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1
}

install_system_packages() {
  step "Ensuring build, git, and video codec system packages..."
  local packages=(git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0)

  if have sudo; then
    sudo apt-get update -y && sudo apt-get install -y "${packages[@]}" || true
  elif have apt-get; then
    apt-get update -y && apt-get install -y "${packages[@]}" || true
  fi
}

ensure_uv() {
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  if ! have uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || python3 -m pip install --user uv 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  fi
}

create_venv() {
  step "Setting up Python virtual environment (Python 3.10-3.12)..."
  ensure_uv

  if [[ -d venv ]] && ! python_supports_vbench venv/bin/python3; then
    echo "Recreating venv with compatible Python version."
    rm -rf venv
  fi

  if [[ ! -d venv ]]; then
    local candidate
    for candidate in python3 python3.12 python3.11 python3.10; do
      if have "${candidate}" && python_supports_vbench "${candidate}"; then
        "${candidate}" -m venv venv
        break
      fi
    done
  fi

  if [[ ! -d venv ]] && have uv; then
    uv venv venv --python 3.11 --seed || uv venv venv --python 3.12 --seed || uv venv venv --python 3.10 --seed
  fi

  [[ -d venv ]] || python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  python3 -m pip install --upgrade "pip<25" "setuptools<71" wheel packaging
}

sync_vbench_repo() {
  step "Syncing VBench repository..."
  if [[ ! -d VBench/.git ]]; then
    git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
  else
    git -C VBench fetch origin "${VBENCH_BRANCH}" || true
    git -C VBench checkout "${VBENCH_BRANCH}" || true
    git -C VBench pull origin "${VBENCH_BRANCH}" || true
  fi
}

show_gpu_status() {
  step "Checking NVIDIA GPU and driver status..."
  if have nvidia-smi; then
    nvidia-smi || true
  else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed on GPU VMs."
  fi
}

install_cuda_torch() {
  step "Installing PyTorch with CUDA support..."
  if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    python3 -m pip uninstall -y torch torchvision 2>/dev/null || true
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple || \
      python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple || \
      python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
  fi
  python3 -m pip install "numpy<2"
}

patch_vbench_sources() {
  step "Applying VBench compatibility patches..."
  git -C VBench checkout -- setup.py vbench/distributed.py evaluate.py vbench/__init__.py >/dev/null 2>&1 || true
  python3 "${UTIL_SCRIPT}" patch-vbench VBench
}

install_vbench_dependencies() {
  step "Installing VBench and evaluation dependencies..."
  patch_vbench_sources
  (cd VBench && python3 -m pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/cu121)
  python3 -m pip install --no-build-isolation git+https://github.com/openai/CLIP.git --extra-index-url https://download.pytorch.org/whl/cu121
  python3 -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' --extra-index-url https://download.pytorch.org/whl/cu121 || true
  python3 -m pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

  if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null && have nvidia-smi; then
    echo "Reinstalling CUDA-enabled PyTorch after dependency resolution..."
    python3 -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  fi

  python3 - <<'PY'
import torch
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"==> PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {device_count}")
PY
}

gcs_cp() {
  if have gcloud && gcloud storage cp "$@"; then
    return 0
  fi
  if have gsutil && gsutil -m cp "$@"; then
    return 0
  fi
  return 1
}

fetch_benchmark_json() {
  step "Locating ${BENCHMARK_JSON}..."
  mkdir -p "${WORK_DIR}"

  local repo_json
  repo_json="${REPO_ROOT}/benchmarks/vbench/${BENCHMARK_JSON}"

  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    cp "${BENCHMARK_JSON_PATH}" "${JSON_DEST}"
  elif [[ -f "${repo_json}" ]]; then
    cp "${repo_json}" "${JSON_DEST}"
  elif [[ -f "${JSON_DEST}" ]]; then
    echo "Using existing ${JSON_DEST}"
  else
    echo "Downloading ${BENCHMARK_JSON} from ${BENCHMARK_JSON_URL}..."
    curl -fLsS "${BENCHMARK_JSON_URL}" -o "${JSON_DEST}" 2>/dev/null || \
      python3 -c "import urllib.request; urllib.request.urlretrieve('${BENCHMARK_JSON_URL}', '${JSON_DEST}')" 2>/dev/null || \
      gcs_cp "gs://${GCS_BUCKET}/${RUN_NAME}/${BENCHMARK_JSON}" "${JSON_DEST}" 2>/dev/null || \
      fetch_benchmark_json_from_repo
  fi
}

fetch_benchmark_json_from_repo() {
  step "Cloning MaxDiffusion fallback metadata..."
  if [[ ! -d maxdiffusion/.git ]]; then
    git clone --depth 1 -b "${MAXDIFFUSION_BRANCH}" "${MAXDIFFUSION_REPO}" maxdiffusion
  fi
  cp "maxdiffusion/benchmarks/vbench/${BENCHMARK_JSON}" "${JSON_DEST}"
}

extract_dimensions() {
  if [[ ${#DIMENSIONS[@]} -eq 0 ]]; then
    step "Extracting dimensions from ${JSON_DEST}..."
    local extracted
    extracted="$(python3 "${UTIL_SCRIPT}" dimensions "${JSON_DEST}")"
    # shellcheck disable=SC2206
    DIMENSIONS=(${extracted})
  fi

  [[ ${#DIMENSIONS[@]} -gt 0 ]] || die "No VBench dimensions found. Pass DIMENSIONS=\"...\" or check ${JSON_DEST}."
  echo "Dimensions to evaluate: ${DIMENSIONS[*]}"
}

download_videos() {
  step "Downloading generated videos from gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
  mkdir -p "${DOWNLOAD_DIR}"
  gcs_cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${DOWNLOAD_DIR}/" || die "Failed to download videos from GCS."
}

prepare_videos() {
  step "Aligning video filenames for VBench..."
  mkdir -p "${VBENCH_VIDEO_DIR}"
  python3 "${UTIL_SCRIPT}" prepare-videos "${JSON_DEST}" "${DOWNLOAD_DIR}" "${VBENCH_VIDEO_DIR}"
}

run_vbench() {
  step "Running VBench evaluate.py..."
  mkdir -p "${RESULTS_DIR}"
  (
    cd "${WORK_DIR}/VBench"
    local num_gpus=1
    if have nvidia-smi; then
      num_gpus=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true) | wc -l)
    fi
    if [[ "${num_gpus}" -gt 1 ]]; then
      torchrun --nproc_per_node="${num_gpus}" evaluate.py \
        --videos_path "${VBENCH_VIDEO_DIR}" \
        --full_json_dir "${JSON_DEST}" \
        --output_path "${RESULTS_DIR}" \
        --dimension "${DIMENSIONS[@]}" \
        --mode vbench_standard
    else
      python3 evaluate.py \
        --videos_path "${VBENCH_VIDEO_DIR}" \
        --full_json_dir "${JSON_DEST}" \
        --output_path "${RESULTS_DIR}" \
        --dimension "${DIMENSIONS[@]}" \
        --mode vbench_standard
    fi
  )
}

upload_results() {
  step "Publishing results to gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/..."
  shopt -s nullglob
  local result_files=("${RESULTS_DIR}"/*)
  shopt -u nullglob

  [[ ${#result_files[@]} -gt 0 ]] || die "No result files found in ${RESULTS_DIR}."
  gcs_cp -r "${result_files[@]}" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/" || die "Failed to upload evaluation results to GCS."
}

run_local() {
  print_config
  mkdir -p "${WORK_DIR}"
  cd "${WORK_DIR}"

  install_system_packages
  create_venv
  sync_vbench_repo
  show_gpu_status
  install_cuda_torch
  install_vbench_dependencies
  fetch_benchmark_json
  extract_dimensions
  download_videos
  prepare_videos
  run_vbench
  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    upload_results
  else
    step "Skipping GCS result upload."
  fi

  echo ""
  echo "=========================================================================="
  echo "VBench evaluation finished successfully!"
  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    echo "Results published to: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"
  else
    echo "Results upload skipped."
  fi
  echo "Local results stored in: ${RESULTS_DIR}/"
  echo "=========================================================================="
}

parse_args "$@"
normalize_config

if [[ "${SSH_MODE}" == "true" ]]; then
  run_over_ssh
else
  run_local
fi
