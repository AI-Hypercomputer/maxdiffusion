#!/bin/bash
# export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
# export MODEL_NAME=llama3.1-405b

export MODEL_NAME=sdxl

# Common parameters
export CLUSTER_NAME=a3plus-benchmark

# export ZONE=us-central1-b
export ZONE=australia-southeast1
export PROJECT=supercomputer-testing

export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/maxdiffusion-1023

export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxdiffusion_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH
export RUN_NAME=$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

# # export JAX_ENABLE_PGLE=false
# export JAX_ENABLE_PGLE=false
# export JAX_PGLE_AGGREGATION_PERCENTILE=50
# export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
# export JAX_PGLE_PROFILING_RUNS=3
# export STRICT_CHECKER=false
# # JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# # JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# # JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# # TF_CPP_VMODULE=profile_guided_latency_estimator=10
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# # CUDA_DEVICE_MAX_CONNECTIONS=1

# cat <<EOF > env.txt
# TF_CPP_MIN_LOG_LEVEL=0
# TF_CPP_VMODULE=gpu_executable=2
# NCCL_DEBUG=INFO
# NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
# NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
# JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
# JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
# JAX_DEBUG_LOG_MODULES=compiler
# XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
# --xla_gpu_enable_triton_gemm=false \
# --xla_gpu_enable_highest_priority_async_stream=true \
# --xla_gpu_all_reduce_combine_threshold_bytes=536870912 \
# --xla_gpu_all_gather_combine_threshold_bytes=536870912 \
# --xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 \
# --xla_gpu_enable_pipelined_all_gather=true \
# --xla_gpu_enable_pipelined_reduce_scatter=true \
# --xla_gpu_enable_pipelined_all_reduce=true \
# --xla_gpu_enable_while_loop_double_buffering=true \
# --xla_disable_hlo_passes=rematerialization \
# --xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \
# --xla_gpu_enable_triton_softmax_fusion=false \
# --xla_gpu_enable_all_gather_combine_by_dim=false \
# --xla_gpu_enable_reduce_scatter_combine_by_dim=false
# EOF

export WORKLOAD_NAME=$USER-$RUN_NAME-${RANDOM:0:2}
export NUM_NODES=2

COMMAND="pwd && ls && pip install . && python -m src.maxdiffusion.train_sdxl src/maxdiffusion/configs/base_xl.yml
        hardware=gpu run_name=$RUN_NAME output_dir=$OUTPUT_PATH "

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

xpk workload delete --project $PROJECT --zone $ZONE --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 

xpk workload create --project $PROJECT --zone $ZONE --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto ;
