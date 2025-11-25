#!/bin/bash
bash docker_build_dependency_image.sh
docker tag maxdiffusion_base_image:latest gcr.io/cloud-tpu-multipod-dev/sanbao/maxdiffusion_base_image:latest
docker push gcr.io/cloud-tpu-multipod-dev/sanbao/maxdiffusion_base_image:latest
CLUSTER_NAME=bodaborg-tpu7x-128
DEVICE_TYPE=tpu7x-128 # can change to any size <= tpu7x-256
PROJECT=cloud-tpu-multipod-dev
ZONE=us-central1

# Please change the RUN_NAME and OUTPUT_DIR to your own GCS bucket path.
export RUN_NAME=sanbao-wan-v7x-20k-${RANDOM}
OUTPUT_DIR=gs://sanbao-bucket/wan/${RUN_NAME}
# OUTPUT_DIR=gs://sanbao-bucket/wan/sanbao-wan-train-test
DATASET_DIR=gs://sanbao-bucket/wan_tfr_dataset_pusa_v1/train/
EVAL_DATA_DIR=gs://sanbao-bucket/wan_tfr_dataset_pusa_v1/eval_timesteps/
SAVE_DATASET_DIR=gs://sanbao-bucket/wan_tfr_dataset_pusa_v1/save/
RANDOM=123456789
IMAGE_DIR=gcr.io/cloud-tpu-multipod-dev/sanbao/maxdiffusion_base_image:latest
# IMAGE_DIR=gcr.io/tpu-prod-env-multipod/maxdiffusion_jax_stable_stack_nightly@sha256:fd27d49a3be7f743f08e3b6b03e5ae00196794944310e3fee2a7795b99d81195
LIBTPU_VERSION=libtpu-0.0.25.dev20251013+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl

xpk workload create \
--cluster=$CLUSTER_NAME \
--project=$PROJECT \
--zone=$ZONE \
--device-type=$DEVICE_TYPE \
--num-slices=1 \
--command=" \
pip install . && \
gsutil cp gs://libtpu-tpu7x-releases/wheels/libtpu/${LIBTPU_VERSION} . && \
python -m pip install ${LIBTPU_VERSION} && \
export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_enable_async_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_max_concurrent_async_all_gathers=4 \
--xla_tpu_enable_async_all_to_all=true \
--xla_latency_hiding_scheduler_rerun=5 \
--xla_tpu_rwb_fusion=false \
--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false \
--xla_tpu_impure_enable_packed_bf16_math_ops=false \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_all_gather_offload_tracing=true \
--xla_tpu_use_tc_device_shape_on_sc=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true \
--xla_enable_transpose_trace=false' && \
echo 'Starting WAN training ...' && \
HF_HUB_CACHE=/dev/shm python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  attention='flash' \
  weights_dtype=bfloat16 \
  activations_dtype=bfloat16 \
  guidance_scale=5.0 \
  flow_shift=5.0 \
  fps=16 \
  skip_jax_distributed_system=False \
  run_name='test-wan-training-new' \
  output_dir=${OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR} \
  load_tfrecord_cached=True \
  height=1280 \
  width=720 \
  num_frames=81 \
  num_inference_steps=50 \
  prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
  jax_cache_dir=${OUTPUT_DIR}/jax_cache/ \
  enable_profiler=True \
  dataset_save_location=${SAVE_DATASET_DIR} \
  remat_policy='HIDDEN_STATE_WITH_OFFLOAD' \
  flash_min_seq_length=0 \
  seed=$RANDOM \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
  per_device_batch_size=0.5 \
  ici_data_parallelism=64 \
  ici_fsdp_parallelism=2 \
  ici_tensor_parallelism=1 \
  allow_split_physical_axes=True \
  max_train_steps=150 \
  scan_layers=true \
  flash_block_sizes='{\"block_q\":2048,\"block_kv_compute\":512,\"block_kv\":2048,\"block_q_dkv\":2048,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":512,\"use_fused_bwd_kernel\":true}' \
  " \
--base-docker-image=${IMAGE_DIR} \
--enable-debug-logs \
--workload=${RUN_NAME} \
--priority=medium \
--max-restarts=0
