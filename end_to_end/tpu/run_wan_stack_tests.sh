#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs the tests of the Wan fast-inference kernels and serving path on a TPU VM
# (validated on v6e-8 and tpu7x-8), including the Pallas kernel grids and
# multi-device ring/Ulysses tests that GitHub Actions skips (they check
# GITHUB_ACTIONS=true) to keep CI short.
#
# Usage, from the repo root of an installed checkout:
#   bash end_to_end/tpu/run_wan_stack_tests.sh [extra pytest args, e.g. -x --durations=20]
set -euo pipefail
cd "$(dirname "$0")/../.."
unset GITHUB_ACTIONS
T=src/maxdiffusion/tests
TESTS=(
  # Fixed-m custom splash kernel and ring (feat/fixed-m-kernel)
  "$T/custom_splash_fixed_m_test.py"
  "$T/ring_fixed_m_test.py"
  # Ulysses x Ring attention (feat/ring-attention)
  "$T/attention_config_guards_test.py"
  "$T/custom_splash_unpadded_test.py"
  "$T/dot_fallback_layout_test.py"
  "$T/fused_producers_test.py"
  "$T/tile_size_grid_search_test.py"
  # Wan fast serving / AOT cache (feat/wan-fast-serving)
  "$T/aot_cache_test.py"
  "$T/converted_weights_cache_test.py"
  "$T/wan/wan_transformer_test.py"
  "$T/wan/wan_warmup_coverage_test.py"
  # Fused RMSNorm+RoPE producer, Wan switches in config (feat/wan-custom-kernels)
  "$T/fused_rmsnorm_rope_pallas_test.py"
  "$T/wan_runtime_options_test.py"
)
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python3}" -m pytest -q -rs "${TESTS[@]}" "$@"
