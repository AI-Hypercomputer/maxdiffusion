#!/bin/bash

export PLATFORM=GKE

set -e

echo "Adjust Network settings and apply non cache copy"

# Install ip.
# Disable slow start after idle
sysctl net.ipv4.tcp_slow_start_after_idle=0

# Disable metrics cache
sysctl net.ipv4.tcp_no_metrics_save=1

# Address rto_min issue with two default routing entries: screen/7RGgkiXkGXSeYF2
route=$(ip route show | sed -n 1p)
second_route=$(ip route show | sed -n 2p)
if [[ "${second_route}" =~ ^default.* ]]; then
  modified_route=${route//" lock"/}
  ip route delete ${modified_route}
fi
route=$(ip route show | sed -n 1p)
echo "route rto before change: $route"
if [[ "${route}" =~ .*lock.*5ms.* ]]; then
  echo "${route}"
else
  # shellcheck disable=SC2086
  ip route change $route rto_min 5ms
fi
route=$(ip route show | sed -n 1p)
echo "route rto after change: $route"

# Disable Cubic Hystart Ack-Train
echo 2 > /sys/module/tcp_cubic/parameters/hystart_detect

# Improve handling SYN burst
echo 4096 > /proc/sys/net/core/somaxconn
echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog

# Disable MTU Discovery
echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing

# Increase TCP Zerocopy control memory
sysctl -w net.core.optmem_max=131072

# Printing output of `ip route show`
echo -e "\nPrinting output of 'ip route show':"
ip route show

first_line_res=$(ip route show | head -n 1)
dev_name=$(echo "$first_line_res" | awk -F'[[:space:]]' '{ print $5 }')
echo "dev_name=${dev_name}"
ethtool -K "${dev_name}" tx-nocache-copy on

echo "rto_setup.sh finished"

OUT_DIR=$1

export JAX_TRACEBACK_FILTERING=off
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'


#cd /maxdiffusion
git clone -b  qinwen/eval  https://github.com/google/maxdiffusion maxdiffusion 
cd maxdiffusion

pip install .
mkdir generated_images
mkdir output

TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 python -m src.maxdiffusion.eval src/maxdiffusion/configs/base_2_base.yml run_name=v5p-128-eval per_device_batch_size=16 \
pretrained_model_name_or_path="gs://jfacevedo-maxdiffusion-v5p/training_results/v5p-32-xpk-moments-wsf-0.1-512-clipping-lr-4e-4/app/maxdiffusion/jfacevedo-maxdiffusion-v5p/training_results/v5p-32-xpk-moments-wsf-0.1-512-clipping-lr-4e-4/checkpoints/1024000/" \
caption_coco_file="/app/datasets/coco2014/val2014_30k.tsv" \
images_directory="/app/maxdiffusion/generated_images/" \
stat_output_directory="output/" \
stat_output_file="output/stats.npz" \
stat_coco_file="/app/datasets/coco2014/val2014_30k_stats.npz" \
clip_cache_dir="clip_cache_dir" \
base_output_directory=${OUT_DIR}

#gsutil cp -r generated_images ${OUT_DIR}/output/