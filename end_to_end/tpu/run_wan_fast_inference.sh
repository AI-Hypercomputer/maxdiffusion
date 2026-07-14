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
# cache + zero-exec warmup, with a tuned v7 2D-ring attention recipe.
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
set -u
MODEL=${1:-22}
STEPS=${2:-40}
PROMPT=${3:-""}

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

CACHE_ROOT=${WAN_CACHE_ROOT:-$HOME/.cache/maxdiffusion_wan}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/wan_out}
mkdir -p "$CACHE_ROOT/jax" "$CACHE_ROOT/aot_wan$MODEL" "$CACHE_ROOT/converted" "$OUTPUT_DIR"

# Tuned collective/scheduler flag set for v7 (from the PR #430 2D-ring
# baseline). One line: libtpu stops parsing at a literal backslash.
export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_dot_strength_reduction=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_enable_async_collective_permute=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_async_all_to_all=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_tpu_enable_ici_ag_pipelining=true --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_megacore_fusion=true --xla_tpu_megacore_fusion_allow_ags=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_sparse_core_all_gather_latency_multiplier=1 --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_assign_all_reduce_scatter_layout=true"

if [ "$MODEL" = "21" ]; then
  CONFIG=src/maxdiffusion/configs/base_wan_14b.yml
  GUIDANCE_ARGS=""
else
  CONFIG=src/maxdiffusion/configs/base_wan_27b.yml
  GUIDANCE_ARGS="guidance_scale_low=3.0 guidance_scale_high=4.0"
fi

PROMPT_ARG=()
[ -n "$PROMPT" ] && PROMPT_ARG=("prompt=$PROMPT")
RUN_NAME="wan${MODEL}_fast_$(date +%m%d-%H%M%S)"

# libtpu's XLA:CPU AOT feature-mismatch log is cosmetic and ignores every
# log-level env var; filter just that message from stderr.
python src/maxdiffusion/generate_wan.py "$CONFIG" \
  run_name="$RUN_NAME" \
  output_dir="$OUTPUT_DIR" \
  jax_cache_dir="$CACHE_ROOT/jax" \
  aot_cache_dir="$CACHE_ROOT/aot_wan$MODEL" \
  converted_weights_dir="$CACHE_ROOT/converted" \
  attention=ulysses_ring_custom \
  ulysses_shards=2 \
  ici_data_parallelism=2 ici_fsdp_parallelism=1 \
  ici_context_parallelism=4 ici_tensor_parallelism=1 \
  per_device_batch_size=0.125 \
  num_inference_steps="$STEPS" num_frames=81 width=1280 height=720 \
  weights_dtype=bfloat16 activations_dtype=bfloat16 \
  vae_spatial=4 vae_decode_chunk=-1 \
  vae_weights_dtype=bfloat16 vae_dtype=bfloat16 \
  text_encoder_dtype=bfloat16 compile_text_encoder="${COMPILE_TE:-false}" use_batched_text_encoder=false \
  use_base2_exp=true use_experimental_scheduler=true \
  fps=16 $GUIDANCE_ARGS \
  flash_block_sizes='{"block_q":9472,"block_kv":1024,"block_kv_compute":1024,"block_kv_compute_in":1024,"heads_per_tile":1,"vmem_limit_bytes":67108864,"block_q_dkv":9472,"block_kv_dkv":1024,"block_kv_dkv_compute":1024}' \
  "${PROMPT_ARG[@]}" \
  2> >(grep -vE --line-buffered 'cpu_aot_loader|machine type for execution' >&2)

mp4=$(ls -t wan_output_*.mp4 2>/dev/null | head -1)
if [ -n "$mp4" ]; then
  mv "$mp4" "$OUTPUT_DIR/${RUN_NAME}.mp4"
  echo ""
  echo "=== video saved: $OUTPUT_DIR/${RUN_NAME}.mp4 ==="
fi
