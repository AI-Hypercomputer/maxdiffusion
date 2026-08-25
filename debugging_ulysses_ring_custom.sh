export TPU_VISIBLE_CHIPS=0
#!/bin/bash
set -e

# Activate environment
if [ -f "/mnt/workspace/maxdiffusion_venv/bin/activate" ]; then
  source /mnt/workspace/maxdiffusion_venv/bin/activate
fi

export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_dot_strength_reduction=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_enable_async_collective_permute=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_async_all_to_all=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_tpu_enable_ici_ag_pipelining=true --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_megacore_fusion=true --xla_tpu_megacore_fusion_allow_ags=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_sparse_core_all_gather_latency_multiplier=1 --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_llo_profiling=true --xla_enable_mxu_trace=true --xla_enable_transpose_trace=true --xla_enable_local_dma_trace=true --xla_tpu_emit_tracing_vwaits=true --xla_jf_debug_level=1'

if [ -f "/home/ameypasarkar_google_com/cloud-devkit/tpu_python" ]; then
  PYTHON_BIN="/home/ameypasarkar_google_com/cloud-devkit/tpu_python"
elif [ -f "/home/ameypasarkar/cloud-devkit/tpu_python" ]; then
  PYTHON_BIN="/home/ameypasarkar/cloud-devkit/tpu_python"
else
  PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TPU_VISIBLE_DEVICES=0 $PYTHON_BIN src/maxdiffusion/generate_flux2klein.py \
    src/maxdiffusion/configs/base_flux2klein.yml \
    output_name=flux2klein_ring_custom.png \
    attention=ulysses_ring_custom \
    ulysses_attention_chunks=1 \
    text_encoder_attention=dot_product \
    ici_context_parallelism=2 \
    ulysses_shards=2 \
    per_device_batch_size=0.5 \
    height=1024 \
    width=1024 \
    num_inference_steps=4 \
    prompt='a dog running in a field' \
    num_reps=5 \
    mask_padding_tokens=False \
    flash_block_sizes='{"block_q": 4608, "block_kv": 1024, "block_kv_compute": 1024}'
