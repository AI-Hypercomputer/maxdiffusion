#!/bin/bash
# Copyright 2025 Google LLC
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
# Ideogram 4 text-to-image inference.
#
#   bash end_to_end/tpu/run_ideogram_inference.sh                 # 50 steps, 1024x1024
#   bash end_to_end/tpu/run_ideogram_inference.sh STEPS=4         # quick smoke run
#   bash end_to_end/tpu/run_ideogram_inference.sh RES=1536 ATTN=flash
#
# Any extra `key=value` pairs are passed straight through to pyconfig, e.g.
#   bash end_to_end/tpu/run_ideogram_inference.sh prompt="a red bicycle"
#
# The reported `generation_time` is steady state: the denoise step is compiled
# per step, so the 2-step warmup compiles exactly what the timed run reuses. The
# run logs a WARNING if it recompiles anyway -- if you see that, the number
# includes compile and should not be quoted as latency.

set -euo pipefail

# The knobs below are read from the environment, but the documented usage puts
# them on the command line (`... STEPS=4`). Consume any leading knob=value args
# so both forms work; anything else falls through to pyconfig via "$@".
while [ $# -gt 0 ]; do
  case "$1" in
    STEPS=*|RES=*|ATTN=*|PD=*|OUT_DIR=*|HF_HOME=*) export "${1?}"; shift ;;
    *) break ;;
  esac
done

STEPS="${STEPS:-50}"
RES="${RES:-1024}"
# Measured on 8x TPU7x, bf16, 34 layers: dense wins at 1024 (339 vs 356 ms/step),
# flash wins at 1536 (906 vs 968). See base_ideogram.yml.
ATTN="${ATTN:-}"

# Keep every cache off the boot disk -- it is ~85% full and the checkpoint alone
# is 27.5 GB. /mnt/pd is the 1.5 TB persistent disk.
PD="${PD:-/mnt/pd/elisatsai-data}"
export HF_HOME="${HF_HOME:-$PD}"
export HF_HUB_ENABLE_HF_TRANSFER=1
JAX_CACHE="$PD/jax_cache"
OUT_DIR="${OUT_DIR:-$PD/ideogram_out}"
mkdir -p "$JAX_CACHE" "$OUT_DIR"

REPO="ideogram-ai/ideogram-4-fp8"

# This repo is gated. Without an authorized token the download fails partway
# through with a 403 that is easy to mistake for a code error, so check first.
python3 - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

repo = sys.argv[1]
api = HfApi()
try:
    who = api.whoami()
    name = who.get("name", "?")
    role = who.get("auth", {}).get("accessToken", {}).get("role")
except Exception as e:
    print(f"[FAIL] HF_TOKEN is not usable: {type(e).__name__}: {str(e)[:160]}")
    sys.exit(1)

try:
    # auth_check reports the *reason* for denial, unlike hf_hub_download, which
    # collapses a 403 into LocalEntryNotFoundError ("check your connection").
    api.auth_check(repo)
    print(f"[ok] HF user '{name}' can read {repo}")
except GatedRepoError:
    print(f"[FAIL] '{name}' has not accepted the license for {repo}.")
    print(f"       Sign in as '{name}' and click agree at https://huggingface.co/{repo}")
    print("       (gated=auto, so approval is immediate).")
    sys.exit(1)
except HfHubHTTPError as e:
    msg = str(e)
    if "fine-grained" in msg or "public gated repositories" in msg:
        print(f"[FAIL] the token for '{name}' (role={role}) lacks gated-repo permission.")
        print("       https://huggingface.co/settings/tokens -> edit the token -> Repositories permissions")
        print("       -> tick 'Read access to contents of all public gated repos you can access'.")
        print("       A classic 'Read' token also works.")
    else:
        print(f"[FAIL] cannot read {repo}: {msg[:220]}")
    sys.exit(1)
PY

echo "steps=$STEPS  resolution=${RES}x${RES}  attention=${ATTN:-<from yml>}"
echo "HF_HOME=$HF_HOME  jax_cache=$JAX_CACHE  out=$OUT_DIR"

ARGS=(
  src/maxdiffusion/configs/base_ideogram.yml
  num_inference_steps="$STEPS"
  height="$RES"
  width="$RES"
  jax_cache_dir="$JAX_CACHE"
  output_dir="$OUT_DIR"
  run_name="ideogram_${RES}_${STEPS}step"
)
[ -n "$ATTN" ] && ARGS+=(attention="$ATTN")

cd "$(git rev-parse --show-toplevel)"
time python3 src/maxdiffusion/generate_ideogram.py "${ARGS[@]}" "$@"
