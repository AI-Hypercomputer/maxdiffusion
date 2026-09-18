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

# WAN T2V fast-serving example: AOT executable cache + converted-weights
# cache + zero-exec warmup, with a tuned 2D-ring attention recipe.
#
# The XLA flag set, attention tile and text-encoder options differ per TPU
# generation, so the platform is auto-detected from the GCE metadata server
# and the matching recipe is selected (see "TPU platform detection" below).
# Supported profiles: v6e (Trillium) and v7. Anything else falls back to a
# conservative generic profile.
#
# First run per (model, shape) pays one-time conversion + compile and
# populates the caches; every later process start is ~25s to ready.
#
# Usage:
#   ./run_wan_fast_inference.sh [21|22] [steps] ["prompt..."]
# Env overrides:
#   WAN_CACHE_ROOT   cache root (default ~/.cache/maxdiffusion_wan)
#   OUTPUT_DIR       video/metrics output (default /tmp/wan_out)
#   COMPILE_TE=true  torch.compile the text encoder (adds ~30s to load,
#                    saves ~10s/encode; worth it for long-lived processes)
#   EXTRA_LIBTPU     extra libtpu flags, appended to the tuned set
#   TPU_PROFILE      force a platform profile (v6e|v7|generic), skipping
#                    autodetection
#   ACCEL_TYPE       force the raw accelerator type (e.g. v6e-8)
#   ATTENTION / ULYSSES_SHARDS / BQ / BKV / BQ_DKV / VMEM_LIMIT_BYTES
#                    override the per-platform attention recipe
#   DP / CP          override the derived ici_data/context_parallelism
set -u
MODEL=${1:-22}
STEPS=${2:-40}
PROMPT=${3:-""}
shift $(($# > 3 ? 3 : $#))

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# Without these the JAX persistent cache silently skips most entries, so the
# "warm" start still recompiles a large part of the graph.
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

CACHE_ROOT=${WAN_CACHE_ROOT:-$HOME/.cache/maxdiffusion_wan}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/maxdiffusion_wan_output}
export TMPDIR=${TMPDIR:-$CACHE_ROOT/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torch_compile}
mkdir -p "$CACHE_ROOT/jax" "$CACHE_ROOT/aot_wan$MODEL" "$CACHE_ROOT/converted" \
         "$OUTPUT_DIR" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

# ---------------------------------------------------------------------------
# TPU platform detection
# ---------------------------------------------------------------------------
# Preference order: explicit override -> TPU_ACCELERATOR_TYPE env (set by some
# runtimes) -> GCE metadata "accelerator-type" (e.g. "v6e-8") -> the
# ACCELERATOR_TYPE line inside the "tpu-env" metadata blob. Detection is pure
# metadata/env: it must not initialise the TPU, or it would take the device
# before the real process starts.
_tpu_metadata() {
  curl -s -m 2 -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2> /dev/null
}

_detect_accel_type() {
  local t="${TPU_ACCELERATOR_TYPE:-}"
  [ -z "$t" ] && t="$(_tpu_metadata accelerator-type)"
  [ -z "$t" ] && t="$(_tpu_metadata tpu-env | sed -n "s/^ACCELERATOR_TYPE: *'\([^']*\)'.*/\1/p")"
  # Guard against metadata returning an HTML error page. Real names seen in the
  # wild: "v6e-8", "v5litepod-8", "tpu7x-8" (v7 reports as tpu7x, not v7x).
  case "$t" in
    v[0-9]* | tpu[0-9]*) printf '%s' "$t" ;;
    *) printf '' ;;
  esac
}

ACCEL_TYPE=${ACCEL_TYPE:-$(_detect_accel_type)}
if [ -n "$ACCEL_TYPE" ]; then
  TPU_GEN="${ACCEL_TYPE%%-*}"  # v6e-8      -> v6e
  TPU_CHIPS="${ACCEL_TYPE##*-}" # v6e-8      -> 8
else
  TPU_GEN=""
  TPU_CHIPS=""
fi
case "$TPU_CHIPS" in
  '' | *[!0-9]*) TPU_CHIPS="" ;;
esac

if [ -z "${TPU_PROFILE:-}" ]; then
  case "$TPU_GEN" in
    v6e | v5litepod | v5e) TPU_PROFILE=v6e ;;
    v7 | v7x | v7p | v7e | tpu7 | tpu7x | tpu7p | tpu7e) TPU_PROFILE=v7 ;;
    *) TPU_PROFILE=generic ;;
  esac
fi

# ---------------------------------------------------------------------------
# Per-platform XLA flags
# ---------------------------------------------------------------------------
# One line per variable: libtpu tokenises LIBTPU_INIT_ARGS on whitespace and
# stops parsing at the first literal backslash, so a multi-line continuation
# inside a *single-quoted* string silently drops every flag after the first
# line break. Keep these as single-line strings.
COMMON_LIBTPU="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_memory_bound_loop_optimizer_options=enabled:true --xla_tpu_enable_dot_strength_reduction=true --xla_enable_async_collective_permute=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_async_all_to_all=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_tpu_enable_ici_ag_pipelining=true --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_sparse_core_all_gather_latency_multiplier=1 --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_assign_all_reduce_scatter_layout=true"

# v6e (Trillium). xla_tpu_enable_async_collective_fusion MUST stay false here:
# continuation fusion for AllGather is a Viperlite-only path and aborts backend
# init on v6e with
#   FAILED_PRECONDITION: Continuation fusion for AllGather is enabled on
#   platform ghostlite ... Please disable it with
#   xla_tpu_enable_async_collective_fusion=false
# The two sub-flags are inert while the parent is false; they are kept so the
# set stays a one-line edit away from a platform that does support it.
# Note xla_tpu_memory_bound_loop_optimizer_options is proto-valued, so
# "enabled:true" is its correct syntax -- unlike the bare enum flags above,
# which only accept true/enabled/ENABLED.
V6E_LIBTPU="--xla_tpu_enable_async_collective_fusion=false"

# v7 has two TensorCores per chip, so megacore fusion applies; v6e has one and
# ignores these.
V7_LIBTPU="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_megacore_fusion=true --xla_tpu_megacore_fusion_allow_ags=true"

# ---------------------------------------------------------------------------
# Attention recipe: measurements behind the per-platform defaults
# ---------------------------------------------------------------------------
# The kernel, Ulysses degree and tile are set per platform in the profile case
# below:
#   - v6e profile: ulysses_custom_fixed_m_per_q_block, U=4 (R=1 on CP=4), BQ=9472, BKV=1024
#   - v7 profile:  ulysses_ring_custom_fixed_m_per_q_block, U=2 (R=2 on CP=4), BQ=6400, BKV=2048
#
# Cloud TPU v6e-8 (Wan 2.2 T2V-A14B, 720p, 81f, 40 steps, CP=4, DP=2):
#   this script, single prompt, end to end:               138.7s (denoise 136.0s, VAE 0.8s)
#   origin/main @ a4ff3aec, single prompt:                156.4s (denoise 153.3s, VAE 2.9s)
#   Denoise improvement: -17.3s (-11.3%); VAE decode: 3.6x faster (0.8s vs 2.9s).
#   Outputs are not bit-identical to the baseline (fixed-m reorders float
#   accumulation): mean pairwise PSNR +3.25 dB vs exact online softmax, all VBench dimensions within 0.25%.
#
# Cloud TPU v7 (tpu7x-8, Wan 2.2 T2V-A14B, 720p, 81f, 40 steps, CP=4, DP=2):
#   this script (profile v7, U=2, R=2, BQ=6400, BKV=2048): 115.2s (denoise 114.4s, VAE 0.8s)
#   origin/main @ 1bc54811, single prompt:                 119.8s (denoise 119.0s, VAE 0.8s)
#   Total inference beats origin/main by -4.6s (-3.9%); xprof collective-permute-done: 2405ms -> 47ms.
#   persistent AOT: compile 94.7s cold -> 15.0s warm; 100% bit-identical video output vs unpadded reference.
#   vae_decode_chunk: 1 -> 0.8s, -1 (unchunked) -> 3.0s, so keep chunking on

# ---------------------------------------------------------------------------
# Platform profile: attention recipe, XLA flags, VMEM budget, text encoder
# ---------------------------------------------------------------------------
# Each arm carries the best measured configuration for that generation. Every
# value is overridable via the matching env var (see the header).
case "$TPU_PROFILE" in
  v6e)
    PLATFORM_LIBTPU="$V6E_LIBTPU"
    DEFAULT_ATTENTION="ulysses_custom_fixed_m_per_q_block"
    DEFAULT_U=4
    DEFAULT_BQ=9472
    DEFAULT_BKV=1024
    DEFAULT_BKV_COMPUTE=512
    DEFAULT_BKV_COMPUTE_IN=512
    # v6e carries 128MiB of VMEM per core; the larger budget lets the 9472-wide
    # Q tile stay resident. Measured identical to the 64MiB budget on this
    # shape, so lower it if a different resolution spills.
    DEFAULT_VMEM=127506841
    DEFAULT_BQ_DKV=$DEFAULT_BQ
    DEFAULT_COMPILE_TE=true
    DEFAULT_BATCHED_TE=true
    DEFAULT_VAE_CHUNK=1
    DEFAULT_VAE_SPATIAL=8
    ;;
  v7)
    PLATFORM_LIBTPU="$V7_LIBTPU"
    DEFAULT_ATTENTION="ulysses_ring_custom_fixed_m"
    DEFAULT_U=2
    DEFAULT_BQ=6400
    DEFAULT_BKV=2048
    DEFAULT_BKV_COMPUTE=2048
    DEFAULT_BKV_COMPUTE_IN=2048
    DEFAULT_VMEM=67108864
    DEFAULT_BQ_DKV=$DEFAULT_BQ
    DEFAULT_COMPILE_TE=true
    DEFAULT_BATCHED_TE=true
    DEFAULT_VAE_CHUNK=1
    DEFAULT_VAE_SPATIAL=8
    ;;
  *)
    echo "== warning: unrecognised accelerator '${ACCEL_TYPE:-unknown}';" \
      "using the generic profile. Set TPU_PROFILE=v6e|v7 to override." >&2
    PLATFORM_LIBTPU=""
    DEFAULT_ATTENTION="ulysses_custom_fixed_m_per_q_block"
    DEFAULT_U=4
    DEFAULT_BQ=9472
    DEFAULT_BKV=1024
    DEFAULT_BKV_COMPUTE=512
    DEFAULT_BKV_COMPUTE_IN=512
    DEFAULT_VMEM=67108864
    DEFAULT_BQ_DKV=$DEFAULT_BQ
    DEFAULT_COMPILE_TE=true
    DEFAULT_BATCHED_TE=true
    DEFAULT_VAE_CHUNK=1
    DEFAULT_VAE_SPATIAL=8
    ;;
esac

export LIBTPU_INIT_ARGS="${COMMON_LIBTPU} ${PLATFORM_LIBTPU} ${EXTRA_LIBTPU:-}"
# A literal backslash truncates libtpu's flag parsing; fail loudly rather than
# running with silently-dropped flags.
case "$LIBTPU_INIT_ARGS" in
  *\\*)
    echo "ERROR: LIBTPU_INIT_ARGS contains a literal backslash; libtpu would" \
      "stop parsing there and drop the remaining flags." >&2
    exit 1
    ;;
esac

ATTENTION=${ATTENTION:-$DEFAULT_ATTENTION}
ULYSSES_SHARDS=${ULYSSES_SHARDS:-$DEFAULT_U}
BQ=${BQ:-$DEFAULT_BQ}
BKV=${BKV:-$DEFAULT_BKV}
BKV_COMPUTE=${BKV_COMPUTE:-$DEFAULT_BKV_COMPUTE}
BKV_COMPUTE_IN=${BKV_COMPUTE_IN:-$DEFAULT_BKV_COMPUTE_IN}
BQ_DKV=${BQ_DKV:-$DEFAULT_BQ_DKV}
VMEM_LIMIT_BYTES=${VMEM_LIMIT_BYTES:-$DEFAULT_VMEM}
COMPILE_TE=${COMPILE_TE:-$DEFAULT_COMPILE_TE}
USE_BATCHED_TE=${USE_BATCHED_TE:-$DEFAULT_BATCHED_TE}
VAE_SPATIAL=${VAE_SPATIAL:-$DEFAULT_VAE_SPATIAL}
VAE_DECODE_CHUNK=${VAE_DECODE_CHUNK:-$DEFAULT_VAE_CHUNK}

# Mesh: context parallelism carries the Ulysses shards, data parallelism takes
# whatever chips remain. Defaults to CP=4 / DP=2 on an 8-chip slice. On a slice
# smaller than CP, clamp rather than emit a mesh larger than the hardware.
CP=${CP:-4}
if [ -n "$TPU_CHIPS" ] && [ "$TPU_CHIPS" -lt "$CP" ]; then
  echo "== note: $TPU_CHIPS-chip slice is smaller than CP=$CP; clamping CP to $TPU_CHIPS" >&2
  CP=$TPU_CHIPS
fi
if [ "$ULYSSES_SHARDS" -gt "$CP" ]; then
  echo "== note: clamping ulysses_shards $ULYSSES_SHARDS -> $CP (must divide CP)" >&2
  ULYSSES_SHARDS=$CP
fi
if [ -z "${DP:-}" ]; then
  if [ -n "$TPU_CHIPS" ] && [ "$TPU_CHIPS" -ge "$CP" ]; then
    DP=$((TPU_CHIPS / CP))
  else
    DP=2
  fi
fi
NUM_CHIPS=$((DP * CP))
# One global video per step: per-device batch is 1/num_chips.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-$(awk -v c="$NUM_CHIPS" 'BEGIN { printf "%.6g", 1.0 / c }')}

if [ "$MODEL" = "21" ]; then
  CONFIG=src/maxdiffusion/configs/base_wan_14b.yml
  GUIDANCE_ARGS=()
else
  CONFIG=src/maxdiffusion/configs/base_wan_27b.yml
  GUIDANCE_ARGS=(guidance_scale_low=3.0 guidance_scale_high=4.0)
fi

PROMPT_ARG=()
[ -n "$PROMPT" ] && PROMPT_ARG=("prompt=$PROMPT")
RUN_NAME="wan${MODEL}_fast_$(date +%m%d-%H%M%S)"
echo "== platform ${ACCEL_TYPE:-unknown} -> profile ${TPU_PROFILE} | mesh DP=${DP} CP=${CP}"
echo "== ${ATTENTION} | U=${ULYSSES_SHARDS} | tile ${BQ}/${BKV} (compute=${BKV_COMPUTE}, in=${BKV_COMPUTE_IN}) | ${STEPS} steps"

FLASH_BLOCK_SIZES="{\"block_q\":$BQ,\"block_kv\":$BKV,\"block_kv_compute\":$BKV_COMPUTE,\"block_kv_compute_in\":$BKV_COMPUTE_IN,\"heads_per_tile\":1,\"vmem_limit_bytes\":$VMEM_LIMIT_BYTES,\"block_q_dkv\":$BQ_DKV,\"block_kv_dkv\":$BKV,\"block_kv_dkv_compute\":$BKV,\"block_q_dq\":$BQ_DKV,\"block_kv_dq\":$BKV}"

# libtpu's XLA:CPU AOT feature-mismatch log is cosmetic and ignores every
# log-level env var; filter just that message from stderr.
python src/maxdiffusion/generate_wan.py "$CONFIG" \
  run_name="$RUN_NAME" \
  output_dir="$OUTPUT_DIR" \
  jax_cache_dir="$CACHE_ROOT/jax" \
  aot_cache_dir="$CACHE_ROOT/aot_wan$MODEL" \
  converted_weights_dir="$CACHE_ROOT/converted" \
  attention="$ATTENTION" \
  ulysses_shards="$ULYSSES_SHARDS" \
  ici_data_parallelism="$DP" ici_fsdp_parallelism=1 \
  ici_context_parallelism="$CP" ici_tensor_parallelism=1 \
  per_device_batch_size="$PER_DEVICE_BATCH" \
  num_inference_steps="$STEPS" num_frames=81 width=1280 height=720 \
  weights_dtype=bfloat16 activations_dtype=bfloat16 \
  vae_spatial="$VAE_SPATIAL" vae_decode_chunk="$VAE_DECODE_CHUNK" \
  vae_weights_dtype=bfloat16 vae_dtype=bfloat16 \
  text_encoder_dtype=bfloat16 compile_text_encoder="$COMPILE_TE" use_batched_text_encoder="$USE_BATCHED_TE" \
  use_kv_cache=true use_base2_exp=true use_experimental_scheduler=true \
  fps=16 "${GUIDANCE_ARGS[@]}" \
  seed="${SEED:-12345}" \
  flash_block_sizes="$FLASH_BLOCK_SIZES" \
  "${PROMPT_ARG[@]}" \
  "$@" \
  2> >(grep -vE --line-buffered 'cpu_aot_loader|machine type for execution' >&2)

mp4=$(ls -t "$OUTPUT_DIR"/${RUN_NAME}*.mp4 "$OUTPUT_DIR"/wan_output_*.mp4 2> /dev/null | head -1)
if [ -n "$mp4" ]; then
  echo ""
  echo "=== video saved: $mp4 ==="
fi
