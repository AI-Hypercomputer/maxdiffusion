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
# VBench TPU Video Generation Script
#
# Usage:
#   bash benchmarks/vbench/run_tpu_generation.sh GCS_BUCKET=<bucket> [KEY=VALUE ...]
#   bash benchmarks/vbench/run_tpu_generation.sh --ssh GCS_BUCKET=<bucket> TPU_NAME=<tpu-vm> [KEY=VALUE ...]
# ==============================================================================

set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:-}"
GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-}"
TPU_NAME="${TPU_NAME:-}"
TPU_ZONE="${TPU_ZONE:-}"
TPU_PROJECT="${TPU_PROJECT:-}"
SSH_MODE=false
EXPLICIT_ARGS=()

DEFAULT_FLASH_BLOCK_SIZES='{"block_q" : 3328, "block_kv_compute" : 256, "block_kv" : 2816, "block_kv_compute_in" : 256, "block_q_dkv": 3328, "block_kv_dkv" : 2816, "block_kv_dkv_compute" : 256, "block_q_dq" : 3328, "block_kv_dq" : 2816, "heads_per_tile" : 1}'

WAN_OVERRIDES=(
  "run_name|RUN_NAME|wan-inference"
  "attention|ATTENTION|ulysses_custom"
  "num_inference_steps|NUM_STEPS|40"
  "num_frames|NUM_FRAMES|81"
  "width|WIDTH|1280"
  "height|HEIGHT|720"
  "per_device_batch_size|PER_DEVICE_BATCH_SIZE|0.125"
  "vae_spatial|VAE_SPATIAL|4"
  "vae_decode_chunk|VAE_DECODE_CHUNK|4"
  "vae_weights_dtype|VAE_WEIGHTS_DTYPE|bfloat16"
  "vae_dtype|VAE_DTYPE|bfloat16"
  "text_encoder_dtype|TEXT_ENCODER_DTYPE|bfloat16"
  "compile_text_encoder|COMPILE_TEXT_ENCODER|true"
  "ici_data_parallelism|ICI_DATA_PARALLELISM|2"
  "ici_context_parallelism|ICI_CONTEXT_PARALLELISM|4"
  "fps|FPS|16"
  "use_kv_cache|USE_KV_CACHE|true"
  "use_base2_exp|USE_BASE2_EXP|true"
  "use_experimental_scheduler|USE_EXPERIMENTAL_SCHEDULER|true"
  "use_batched_text_encoder|USE_BATCHED_TEXT_ENCODER|true"
  "flash_block_sizes|FLASH_BLOCK_SIZES|"
  "prompt_file|PROMPT_FILE|./benchmarks/vbench/prompts_110.txt"
)

usage() {
  cat <<EOF
Usage: $0 [--ssh] GCS_BUCKET=<bucket_name> [KEY=VALUE ...]

Required:
  GCS_BUCKET         GCS bucket for generated videos.

Common options:
  RUN_NAME           Generation run name (default: wan-inference)
  GCS_VIDEO_DIR      GCS video prefix (default: \${RUN_NAME}/videos)
  PROMPT_FILE        Prompt file path (default: ./benchmarks/vbench/prompts_110.txt)
  CONFIG_FILE        WAN config file (default: src/maxdiffusion/configs/base_wan_27b.yml)
  EXTERNAL_DISK      Mounted disk root for large local files (default: /mnt/disks/external_disk)
  HF_CACHE_ROOT      Hugging Face cache root (default: \$EXTERNAL_DISK/hf_cache)
  HF_HOME            Hugging Face home directory (default: \$HF_CACHE_ROOT)
  HF_HUB_CACHE       Hugging Face Hub model cache (default: \$HF_HOME/hub)
  HF_XET_CACHE       Hugging Face Xet cache (default: \$HF_HOME/xet)
  TMPDIR             Temporary files directory (default: \$EXTERNAL_DISK/tmp)

SSH options:
  --ssh              Copy this script to a TPU VM and run it there.
  TPU_NAME           TPU VM name (required when using --ssh).
  TPU_ZONE           TPU VM zone.
  TPU_PROJECT        TPU VM project.
  REMOTE_DIR         MaxDiffusion repo path on the TPU VM (default: \$HOME/maxdiffusion)
  GIT_REPO           Repo to clone in SSH mode (default: https://github.com/AI-Hypercomputer/maxdiffusion.git).
  GIT_BRANCH         Branch to sync in SSH mode (default: main)
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

quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

set_default() {
  local var="$1" value="$2"
  [[ -n "${!var-}" ]] || printf -v "${var}" '%s' "${value}"
}

arg_was_explicit() {
  local explicit_arg
  for explicit_arg in "${EXPLICIT_ARGS[@]}"; do
    [[ "${explicit_arg}" == "$1" ]] && return 0
  done
  return 1
}

path_is_under_external_disk() {
  local path="${1%/}"
  [[ "${path}" == "${EXTERNAL_DISK}" || "${path}" == "${EXTERNAL_DISK}/"* ]]
}

ignore_inherited_path_outside_external_disk() {
  local var="$1"
  if [[ -n "${!var-}" ]] && ! arg_was_explicit "${var}" && ! path_is_under_external_disk "${!var}"; then
    unset "${var}"
  fi
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
      *=*)
        key="${arg%%=*}"
        value="${arg#*=}"
        EXPLICIT_ARGS+=("${key}")
        export "${key}=${value}"
        ;;
      *)
        [[ -n "${GCS_BUCKET}" ]] || GCS_BUCKET="${arg}"
        ;;
    esac
  done
}

normalize_config() {
  local key var value item
  [[ -n "${GCS_BUCKET}" ]] || die "GCS_BUCKET is required. Example: bash benchmarks/vbench/run_tpu_generation.sh GCS_BUCKET=my-bucket"

  GCS_BUCKET="${GCS_BUCKET#gs://}"
  GCS_BUCKET="${GCS_BUCKET%/}"
  set_default CONFIG_FILE "src/maxdiffusion/configs/base_wan_27b.yml"
  set_default FLASH_BLOCK_SIZES "${DEFAULT_FLASH_BLOCK_SIZES}"
  set_default TPU_NAME ""
  set_default TPU_ZONE ""
  set_default TPU_PROJECT ""
  set_default GIT_REPO "https://github.com/AI-Hypercomputer/maxdiffusion.git"
  set_default GIT_BRANCH "main"

  if [[ "${SSH_MODE}" == "true" ]]; then
    [[ -n "${TPU_NAME:-}" ]] || die "TPU_NAME is required when using --ssh."
  fi
  set_default EXTERNAL_DISK "/mnt/disks/external_disk"
  EXTERNAL_DISK="${EXTERNAL_DISK%/}"
  for var in HF_CACHE_ROOT HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE TRANSFORMERS_CACHE TMPDIR; do
    ignore_inherited_path_outside_external_disk "${var}"
  done
  set_default HF_CACHE_ROOT "${EXTERNAL_DISK}/hf_cache"
  set_default HF_HOME "${HF_CACHE_ROOT}"
  set_default HF_HUB_CACHE "${HF_HOME}/hub"
  set_default HF_XET_CACHE "${HF_HOME}/xet"
  set_default HF_ASSETS_CACHE "${HF_HOME}/assets"
  set_default HF_DATASETS_CACHE "${HF_HOME}/datasets"
  set_default HF_MODULES_CACHE "${HF_HOME}/modules"
  set_default TRANSFORMERS_CACHE "${HF_HUB_CACHE}"
  set_default TMPDIR "${EXTERNAL_DISK}/tmp"
  set_default VENV_DIR "${HOME}/maxdiffusion_venv"

  for item in "${WAN_OVERRIDES[@]}"; do
    IFS='|' read -r key var value <<< "${item}"
    set_default "${var}" "${value}"
  done
  set_default GCS_VIDEO_DIR "${RUN_NAME}/videos"
  GCS_VIDEO_DIR="${GCS_VIDEO_DIR#/}"
  GCS_VIDEO_DIR="${GCS_VIDEO_DIR%/}"
  [[ "${GCS_VIDEO_DIR}" == "${RUN_NAME}/videos" ]] || die "GCS_VIDEO_DIR must be ${RUN_NAME}/videos for TPU generation; pass RUN_NAME to choose the output directory."

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -n "${MAXDIFFUSION_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${MAXDIFFUSION_ROOT}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
}

absolute_file() {
  local path="$1" dir base
  [[ -f "${path}" ]] || die "File not found: ${path}"
  dir="$(cd "$(dirname "${path}")" && pwd)"
  base="$(basename "${path}")"
  printf '%s/%s\n' "${dir}" "${base}"
}

print_config() {
  echo "=========================================================================="
  echo "Starting VBench video generation on TPU VM"
  echo "  Run Name:    ${RUN_NAME}"
  echo "  GCS Bucket:  gs://${GCS_BUCKET}"
  echo "  Video Target: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "  Prompt File: ${PROMPT_FILE}"
  echo "  Config File: ${CONFIG_FILE}"
  echo "  Seed:        12345"
  echo "  Repo Root:   ${REPO_ROOT}"
  echo "  HF Cache:    ${HF_CACHE_ROOT}"
  echo "  HF Hub:      ${HF_HUB_CACHE}"
  echo "  HF Xet:      ${HF_XET_CACHE}"
  echo "  Temp Dir:    ${TMPDIR}"
  echo "=========================================================================="
}

emit_remote_arg() {
  local var="$1"
  printf '  %s\n' "$(quote "${var}=${!var-}")"
}

emit_remote_arg_if_explicit() {
  local var="$1"
  if arg_was_explicit "${var}"; then
    emit_remote_arg "${var}"
  fi
}

emit_remote_args() {
  local item key var value
  for var in GCS_BUCKET GCS_VIDEO_DIR CONFIG_FILE EXTERNAL_DISK HF_CACHE_ROOT HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE TRANSFORMERS_CACHE TMPDIR; do
    emit_remote_arg "${var}"
  done
  emit_remote_arg_if_explicit VENV_DIR
  for item in "${WAN_OVERRIDES[@]}"; do
    IFS='|' read -r key var value <<< "${item}"
    emit_remote_arg "${var}"
  done
}

run_over_ssh() {
  local source_script remote_script remote_dir_label remote_dir_assignment remote_command
  local gcloud_args=()
  [[ -n "${TPU_ZONE}" ]] && gcloud_args+=("--zone=${TPU_ZONE}")
  [[ -n "${TPU_PROJECT}" ]] && gcloud_args+=("--project=${TPU_PROJECT}")

  source_script="$(absolute_file "${BASH_SOURCE[0]}")"
  remote_script="/tmp/run_tpu_generation_${USER:-user}_$$.sh"
  remote_dir_label="${REMOTE_DIR:-\$HOME/maxdiffusion}"
  if [[ -n "${REMOTE_DIR:-}" ]]; then
    remote_dir_assignment="remote_dir=$(quote "${REMOTE_DIR}")"
  else
    remote_dir_assignment='remote_dir="${HOME}/maxdiffusion"'
  fi

  echo "=========================================================================="
  echo "Executing VBench generation remotely"
  echo "  TPU Name:    ${TPU_NAME}"
  echo "  TPU Zone:    ${TPU_ZONE:-<default>}"
  echo "  TPU Project: ${TPU_PROJECT:-<default>}"
  echo "  Remote Dir:  ${remote_dir_label}"
  echo "  GCS Bucket:  gs://${GCS_BUCKET}"
  echo "  Video Target: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "=========================================================================="

  gcloud compute tpus tpu-vm scp "${source_script}" "${TPU_NAME}:${remote_script}" "${gcloud_args[@]}"

  remote_command=$(cat <<EOF
set -euo pipefail
${remote_dir_assignment}
git_repo=$(quote "${GIT_REPO}")
git_branch=$(quote "${GIT_BRANCH}")

echo "==> [TPU VM] Syncing MaxDiffusion at \${remote_dir}..."
if [[ ! -d "\${remote_dir}/.git" ]]; then
  mkdir -p "\$(dirname "\${remote_dir}")"
  git clone "\${git_repo}" "\${remote_dir}"
fi
cd "\${remote_dir}"
git fetch origin "\${git_branch}" || true
git checkout "\${git_branch}" || true
git pull origin "\${git_branch}" || true

args=(
$(emit_remote_args)
  "MAXDIFFUSION_ROOT=\${remote_dir}"
)
bash $(quote "${remote_script}") "\${args[@]}"
EOF
)

  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" "${gcloud_args[@]}" --command="${remote_command}"
}

configure_cache_dirs() {
  local cache_var external_device root_device

  [[ -d "${EXTERNAL_DISK}" ]] || die "External disk ${EXTERNAL_DISK} is not mounted. Mount it or pass EXTERNAL_DISK=<mounted-disk-path>."
  [[ -w "${EXTERNAL_DISK}" ]] || die "External disk ${EXTERNAL_DISK} is not writable."

  root_device="$(df -P / | awk 'NR == 2 {print $1}')"
  external_device="$(df -P "${EXTERNAL_DISK}" | awk 'NR == 2 {print $1}')"
  if [[ "${external_device}" == "${root_device}" ]]; then
    local free_kb free_gb
    free_kb="$(df -P / | awk 'NR == 2 {print $4}')"
    free_gb=$(( free_kb / 1024 / 1024 ))
    if (( free_gb < 100 )); then
      die "${EXTERNAL_DISK} is on root filesystem with only ${free_gb}GB free. Need at least 100GB."
    else
      echo "WARNING: HF cache is on root filesystem (${free_gb}GB free). Consider mounting a dedicated disk for large runs."
    fi
  fi

  for cache_var in HF_CACHE_ROOT HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE TRANSFORMERS_CACHE TMPDIR; do
    case "${!cache_var%/}" in
      "${EXTERNAL_DISK}" | "${EXTERNAL_DISK}/"*) ;;
      *) die "${cache_var}=${!cache_var} must be under EXTERNAL_DISK=${EXTERNAL_DISK}." ;;
    esac
  done

  export HF_CACHE_ROOT HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE TRANSFORMERS_CACHE TMPDIR
  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${HF_ASSETS_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"
}

python_supports_wan() {
  "$1" -c 'import sys; assert sys.version_info >= (3, 12)' >/dev/null 2>&1
}

create_python_env() {
  export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
  if [[ -x "${VENV_DIR}/bin/python3" ]] && python_supports_wan "${VENV_DIR}/bin/python3"; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    return
  fi
  # Always create and use the virtual environment to avoid PEP 668 issues and system package conflicts.

  step "Creating Python 3.12 virtualenv..."
  have uv || python3 -m pip install --user --upgrade uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [[ ! -d "${VENV_DIR}" ]] || rm -rf "${VENV_DIR}"
  python3 -m uv venv "${VENV_DIR}" --python 3.12 --seed || uv venv "${VENV_DIR}" --python 3.12 --seed
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

install_dependencies() {
  step "Installing MaxDiffusion TPU dependencies..."
  bash setup.sh MODE=stable DEVICE=tpu
  python3 -m uv pip install -e . || uv pip install -e . || python3 -m pip install -e .
}

run_generation() {
  local item key var value
  local -a args
  step "Running WAN 2.2 27B inference with seed: 12345..."
  args=(python3 src/maxdiffusion/generate_wan.py "${CONFIG_FILE}")
  for item in "${WAN_OVERRIDES[@]}"; do
    IFS='|' read -r key var value <<< "${item}"
    args+=("${key}=${!var}")
  done
  args+=("seed=12345" "base_output_directory=gs://${GCS_BUCKET}")
  "${args[@]}"
}

sync_metadata() {
  step "Syncing benchmark metadata to GCS..."
  python3 - <<PY
import glob
from maxdiffusion.max_utils import upload_file_to_gcs

for json_file in sorted(glob.glob("./benchmarks/vbench/VBench_full_info_sub*.json")):
  upload_file_to_gcs("gs://${GCS_BUCKET}/${RUN_NAME}", json_file)
PY
}

run_local() {
  print_config
  cd "${REPO_ROOT}"
  configure_cache_dirs
  create_python_env
  install_dependencies
  run_generation
  sync_metadata

  echo ""
  echo "=========================================================================="
  echo "TPU video generation complete!"
  echo "Generated videos saved to: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/"
  echo ""
  echo "Next Step: run benchmarks/vbench/run_gpu_eval.sh on your GPU VM:"
  if [[ "${GCS_VIDEO_DIR}" != "${RUN_NAME}/videos" ]]; then
    echo "  bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=${GCS_BUCKET} RUN_NAME=${RUN_NAME} GCS_VIDEO_DIR=${GCS_VIDEO_DIR}"
  else
    echo "  bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=${GCS_BUCKET} RUN_NAME=${RUN_NAME}"
  fi
  echo "=========================================================================="
}

parse_args "$@"
normalize_config

if [[ "${SSH_MODE}" == "true" ]]; then
  run_over_ssh
else
  run_local
fi
