#!/bin/bash
set -e

# Activate virtual environment
if [ -f "/mnt/workspace/maxdiffusion_venv/bin/activate" ]; then
  source /mnt/workspace/maxdiffusion_venv/bin/activate
elif [ -f "/mnt/data/mdiff_venv/bin/activate" ]; then
  source /mnt/data/mdiff_venv/bin/activate
fi

# Detect CDK tpu_python wrapper
if [ -f "/home/ameypasarkar_google_com/cloud-devkit/tpu_python" ]; then
  PYTHON_BIN="/home/ameypasarkar_google_com/cloud-devkit/tpu_python"
elif [ -f "/home/ameypasarkar/cloud-devkit/tpu_python" ]; then
  PYTHON_BIN="/home/ameypasarkar/cloud-devkit/tpu_python"
else
  PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Point to workspace HF cache containing FLUX.2-Klein 4B weights
export HF_HOME="${HF_HOME:-/mnt/workspace/hf_cache}"

echo "================================================================================"
echo "🚀 Launching FLUX.2-Klein Multi-Image Editing E2E Parity Test on TPU v7 via CDK"
echo "   Python Runner: $PYTHON_BIN"
echo "   HF_HOME:       $HF_HOME"
echo "================================================================================"

$PYTHON_BIN -m pytest src/maxdiffusion/tests/edit_flux2klein_e2e_test.py -k test_e2e_image_edit_parity_vs_diffusers -s -v "$@"
