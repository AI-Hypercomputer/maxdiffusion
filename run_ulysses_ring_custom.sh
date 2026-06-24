#!/bin/bash
# End-to-end Wan2.2 27B T2V generation using the new ulysses_ring_custom
# (Ulysses + Ring 2D context parallel) attention kernel.
set -uo pipefail
cd /mnt/data_500g/maxdiffusion

export GLOG_minloglevel=3
export JAX_CPP_MIN_LOG_LEVEL=3
export TPU_LOGS="/mnt/data_500g/sagar_tmp/tpu_logs/"
mkdir -p "$TPU_LOGS"

# Complete local Wan2.2-T2V-A14B HF cache (weights) on this device.
export HF_HUB_CACHE="/mnt/data_500g/maxdiffusion_runtime/wan_hf_cache/"
export HF_HUB_ENABLE_HF_TRANSFER=1

export JAX_COMPILATION_CACHE_DIR="/mnt/data_500g/sagar_jax_cache_wan22"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16

export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_enable_dot_strength_reduction=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=8192 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_memory_bound_loop_optimizer_options=enabled:true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_assign_all_reduce_scatter_layout=true'

export PYTHONPATH=src

.venv/bin/python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_27b.yml \
  run_name=wan-ulysses-ring-custom \
  pretrained_orbax_dir=/mnt/data_500g/sagar_orbax_22_27b_v2 \
  seed=12345 \
  attention="ulysses_ring_custom" \
  ulysses_shards=2 \
  flash_min_seq_length=0 \
  num_inference_steps=40 num_frames=81 width=1280 height=720 \
  per_device_batch_size=.125 \
  vae_spatial=4 vae_decode_chunk=-1 vae_weights_dtype='bfloat16' vae_dtype='bfloat16' \
  text_encoder_dtype='bfloat16' compile_text_encoder=true \
  ici_data_parallelism=2 ici_context_parallelism=4 fps=16 \
  use_kv_cache=true use_base2_exp=true use_experimental_scheduler=true use_batched_text_encoder=true \
  flash_block_sizes='{"block_q" : 6400, "block_kv" : 1024, "block_kv_compute" : 1024, "block_kv_compute_in" : 1024, "heads_per_tile" : 1, "vmem_limit_bytes" : 67108864}'
