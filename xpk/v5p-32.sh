#!/bin/bash

export PLATFORM=GKE

set -e

echo "Adjust Network settings and apply non cache copy"

# Install ip.
apt-get update
yes | apt-get install net-tools
yes | apt-get install iproute2
yes | apt-get install procps
yes | apt-get install ethtool

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

export JAX_TRACEBACK_FILTERING=off
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'
rm -rf maxdiffusion
git clone https://github.com/google/maxdiffusion.git
cd maxdiffusion
git checkout mlperf_4
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
pip install .
python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=v5p-32-xpk base_output_directory=gs://jfacevedo-maxdiffusion/training_results  train_data_dir=gs://jfacevedo-maxdiffusion-v5p/laion400m/processed/laion400m_tfrec per_device_batch_size=16 split_head_dim=True  attention=flash train_new_unet=True norm_num_groups=16 enable_profiler=True