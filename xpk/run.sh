#!/bin/bash
#!/bin/bash

# export PLATFORM=GKE

# set -ex

# echo "Adjust Network settings and apply non cache copy"

# # Install ip.
# # Disable slow start after idle
# sysctl net.ipv4.tcp_slow_start_after_idle=0

# # Disable metrics cache
# sysctl net.ipv4.tcp_no_metrics_save=1

# # Address rto_min issue with two default routing entries: screen/7RGgkiXkGXSeYF2
# route=$(ip route show | sed -n 1p)
# second_route=$(ip route show | sed -n 2p)
# if [[ "${second_route}" =~ ^default.* ]]; then
#   modified_route=${route//" lock"/}
#   ip route delete "${modified_route}"
# fi
# route=$(ip route show | sed -n 1p)
# echo "route rto before change: $route"
# if [[ "${route}" =~ .*lock.*5ms.* ]]; then
#   echo "${route}"
# else
#   # shellcheck disable=SC2086
#   ip route change $route rto_min 5ms
# fi
# route=$(ip route show | sed -n 1p)
# echo "route rto after change: $route"

# # Disable Cubic Hystart Ack-Train
# echo 2 > /sys/module/tcp_cubic/parameters/hystart_detect

# # Improve handling SYN burst
# echo 4096 > /proc/sys/net/core/somaxconn
# echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog

# # Disable MTU Discovery
# echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing

# # Increase TCP Zerocopy control memory
# sysctl -w net.core.optmem_max=131072

# # Printing output of `ip route show`
# echo -e "\nPrinting output of 'ip route show':"
# ip route show

# first_line_res=$(ip route show | head -n 1)
# dev_name=$(echo "$first_line_res" | awk -F'[[:space:]]' '{ print $5 }')
# echo "dev_name=${dev_name}"
# ethtool -K "${dev_name}" tx-nocache-copy on

# echo "rto_setup.sh finished"

export JAX_TRACEBACK_FILTERING=off
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_spmd_threshold_for_allgather_cse=1000000 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000'
LIBTPU_INIT_ARGS+=' --xla_sc_disable_megacore_partitioning=true --xla_tpu_use_tc_device_shape_on_sc=true --tpu_use_continuations=true --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false --2a886c8_chip_config_name=megachip_tccontrol --xla_jf_crs_combiner_threshold_count=10 --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true'
# export TPU_STDERR_LOG_LEVEL=0
# export TPU_MIN_LOG_LEVEL=0
# export TF_CPP_MIN_LOG_LEVEL=0

# git clone -b mlperf_4.1 https://github.com/google/maxdiffusion maxdiffusion
# cd maxdiffusion

# pip install .

# checkpoint interval for num of pics consumed
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-512000}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-10}
# NUM_DEVICES=${NUM_DEVICES:-64}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5000}
LR=${LR:-1.75e-4}
WARM_UP=${WARM_UP:-0.1}
METRICS_INTERVAL=${METRICS_INTERVAL:-100}
# different random seed for each run
SEED=${SEED:-0}

EVAL_OUT_DIR=/tmp/outputs
mkdir -p $EVAL_OUT_DIR
# training
RUN_NAME=${RUN_NAME:-"mlperf_e2e"}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-gs://raymondzou-multipod-prod/v6e/$RUN_NAME}
DATA_DIR=gs://mlperf-llm-public2/laion400m/raw_data/tf_records_512_encoder_state_fp32
bucket_name=raymondzou-multipod-prod/v6e

python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name="$RUN_NAME" base_output_directory="$OUTPUT_DIRECTORY" train_data_dir="${DATA_DIR}" \
per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" split_head_dim=True  attention=flash  norm_num_groups=16 \
eval_at_checkpoint=False \
train_new_unet=True \
warmup_steps_fraction="${WARM_UP}" learning_rate="${LR}" \
noise_offset=-1.0 input_peturbation=-1.0 prediction_type='v_prediction' snr_gamma=-1.0 \
learning_rate_scheduler='linear' \
timestep_spacing='trailing' \
upload_images=False \
caption_coco_file="gs://mlperf-exp/sd-copy/cocodata/val2014_30k_padded.tsv" \
images_directory="$EVAL_OUT_DIR/" \
stat_output_directory="output/" \
stat_output_file="output/stats.npz" \
stat_coco_file="gs://mlperf-exp/sd-copy/cocodata/val2014_30k_stats.npz" \
clip_cache_dir="clip_cache_dir" \
start_step_to_checkpoint=5120000 \
skip_first_n_steps_for_profiler=10 \
profiler_steps=3 \
enable_profiler=True \
seed="$SEED" \
checkpoint_every="$CHECKPOINT_EVERY" max_train_steps="$MAX_TRAIN_STEPS" metrics_period="$METRICS_INTERVAL" 2>&1 | tee /tmp/log

sleep 60


if [[ $(grep "MLLOG" /tmp/log | wc -l) -gt 0 ]];then
  python src/maxdiffusion/report_end.py --metrics-path="${OUTPUT_DIRECTORY}"/"${RUN_NAME}"/eval_metrics.csv --mllog-path=/tmp/log 2>&1 | tee -a /tmp/log
  gsutil cp /tmp/log "${OUTPUT_DIRECTORY}"/"${RUN_NAME}"/log_"${MEGASCALE_SLICE_ID}"_"${TPU_WORKER_ID}"
fi