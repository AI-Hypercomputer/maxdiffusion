OUT_DIR='gs://mlperf-exp/qinwen/stablediffusion/v5p-8/perf-debug/0918'
#ici_fsdp_parallelism=4 \
#export LIBTPU_INIT_ARGS='--xla_tpu_enable_megacore_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_async_collective_fusion_fuse_multiple_collectives=false --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'
export JAX_TRACEBACK_FILTERING=off
#export LIBTPU_INIT_ARGS='--xla_tpu_enable_megacore_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true '
#LIBTPU_INIT_ARGS+='--xla_jf_auto_cross_replica_sharding=true xla_tpu_spmd_rng_bit_generator_unsafe=1 '

#--xla_jf_crs_combiner_threshold_in_bytes=0
#--xla_enable_async_all_reduce=true
#LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true '
#gsutil cp gs://test-example-123/libtpu/2024-08-12-17:49:01-libtpu.so  /home/qinwen/.local/lib/python3.10/site-packages/libtpu/libtpu.so

#gsutil -m cp gs://test-example-123/libtpu/2024-08-16-20:20:12-libtpu.so /home/qinwen/.local/lib/python3.10/site-packages/libtpu/libtpu.so
#gsutil -m cp gs://test-example-123/libtpu/2024-09-03-17:37:03-libtpu.so /home/qinwen/.local/lib/python3.10/site-packages/libtpu/libtpu.so

#gsutil -m cp gs://test-example-123/libtpu/2024-09-11-23:13:31-libtpu.so /home/qinwen/.local/lib/python3.10/site-packages/libtpu/libtpu.so

export LIBTPU_INIT_ARGS='--xla_tpu_enable_megacore_fusion=false --xla_tpu_megacore_fusion_allow_ags=false  --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true'

#--xla_latency_hiding_scheduler_enable_selective_resources=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_mosaic_fusion=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_enable_async_all_gather=true'

#LIBTPU_INIT_ARGS+=' --xla_enable_async_reduce_scatter_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true'
# LIBTPU_INIT_ARGS+=' --xla_tpu_spmd_threshold_for_allgather_cse=1000000 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000'
LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_enable_async_all_reduce=true --xla_latency_hiding_scheduler_enable_selective_resources=true --xla_tpu_scavenge_vmem_for_fusions=true --xla_tpu_allocate_scoped_vmem_at_same_offset=false'
#LIBTPU_INIT_ARGS+=' --xla_tpu_decompose_einsum_reduce_scatter=true' 
#numactl --cpunodebind=1 

#LIBTPU_INIT_ARGS+=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"

DATA_DIR=gs://jfacevedo-maxdiffusion/laion400m/raw_data/tf_records_512_encoder_state_fp32

TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_fdsp_other --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd|sharding" python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=v5p-8 base_output_directory=${OUT_DIR}  \
train_data_dir=${DATA_DIR} per_device_batch_size=64 \
split_head_dim=True  attention=flash  train_new_unet=True norm_num_groups=16 \
start_step_to_checkpoint=5120000 max_grad_norm=1.0 \
write_metrics=true \
learning_rate=1e-4 \
ici_fsdp_parallelism=4 \
enable_profiler=True \
skip_first_n_steps_for_profiler=290 \
reuse_example_batch=false max_train_steps=300  2>&1 | tee /tmp/SD_test_log
