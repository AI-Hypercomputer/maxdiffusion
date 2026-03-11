export WORKLOAD_NAME="wan-2.2-6ve-runs"
export OUTPUT_DIR=gs://rish_wan_benchmarking/outputs/${WORKLOAD_NAME}
export IMAGE_NAME="us-central1-docker.pkg.dev/tpu-prod-env-one-vm/rishabhmanoj/maxdiffusion-base:v1.0"

export PER_DEVICE_BATCH_SIZE=0.125
export GLOBAL_BATCH_SIZE=1
export ICI_DATA_PARALLELISM=2
export ICI_CONTEXT_PARALLELISM=4
export ICI_TENSOR_PARALLELISM=1
export TPU_TYPE="v6e-8"

# Generate a UUID using nanosecond precision to ensure high uniqueness
export UUID=$(date +%s%N)

# Construct the RUN_NAME with descriptive parameters and the unique UUID
export RUN_NAME="wan2_2-benchmark-${TPU_TYPE}-${GLOBAL_BATCH_SIZE}-${ICI_DATA_PARALLELISM}x${ICI_CONTEXT_PARALLELISM}x${ICI_TENSOR_PARALLELISM}-${UUID}"

echo "Generated UUID: ${UUID}"
echo "Generated RUN_NAME: ${RUN_NAME}"

# Optimized command string
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 && \
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16 && \
export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_enable_dot_strength_reduction=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_tpu_megacore_fusion_allow_ags=false \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=81920 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_enable_megacore=true \
--xla_tpu_assign_all_reduce_scatter_layout=true' && \
export HF_HUB_CACHE=/dev/shm/maxdiffusion_hf_cache/ && \
export HF_HUB_ENABLE_HF_TRANSFER=1 && \
HF_HUB_CACHE=/dev/shm python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_27b_v6e.yml \
  run_name=${RUN_NAME} \
  output_dir=${OUTPUT_DIR} \
  jax_cache_dir=${OUTPUT_DIR}/${RUN_NAME}/jax_cache/ \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  global_batch_size=${GLOBAL_BATCH_SIZE} \
  ici_data_parallelism=${ICI_DATA_PARALLELISM} \
  ici_context_parallelism=${ICI_CONTEXT_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  num_inference_steps=40 \
  attention='tokamax_flash'