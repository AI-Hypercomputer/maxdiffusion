#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# bash setup.sh MODE={stable,nightly} DEVICE={tpu,gpu}

# You need to specify a MODE, default value stable.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.33
# Enable "exit immediately if any command fails" option
set -e
export DEBIAN_FRONTEND=noninteractive

(sudo bash || bash) <<'EOF'
apt update && \
apt install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool && \
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# Set environment variables from command line arguments
for ARGUMENT in "$@"; do
  IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
  export "$KEY"="$VALUE"
done

# Default device is TPU
if [[ -z "$DEVICE" ]]; then
  export DEVICE="tpu"
fi

# Unset JAX_VERSION if set to "NONE"
if [[ $JAX_VERSION == NONE ]]; then
  unset JAX_VERSION
fi

# Validate JAX_VERSION is only used with stable mode
if [[ -n $JAX_VERSION && ! ($MODE == "stable" || -z $MODE) ]]; then
  echo -e "\n\nError: You can only specify a JAX_VERSION with stable mode.\n\n"
  exit 1
fi

# Install dependencies from requirements.txt first
pip3 install -U -r requirements.txt || echo "Failed to install dependencies in the requirements" >&2

# Install JAX and JAXlib based on the specified mode
if [[ "$MODE" == "stable" || ! -v MODE ]]; then
  # Stable mode
  if [[ $DEVICE == "tpu" ]]; then
    echo "Installing stable jax, jaxlib for tpu"
    if [[ -n "$JAX_VERSION" ]]; then
      echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
      pip3 install "jax[tpu]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    else
      echo "Installing stable jax, jaxlib, libtpu
  for tpu"
      pip3 install 'jax[tpu]>0.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    fi
  elif [[ $DEVICE == "gpu" ]]; then
      echo "Installing stable jax, jaxlib for NVIDIA gpu"
    if [[ -n "$JAX_VERSION" ]]; then
        echo "Installing stable jax, jaxlib ${JAX_VERSION}"
        pip3 install -U "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Installing stable jax, jaxlib, libtpu for NVIDIA gpu"
        pip3 install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi
    export NVTE_FRAMEWORK=jax
    pip3 install transformer_engine[jax]==2.1.0
  fi

elif [[ $MODE == "nightly" ]]; then
  # Nightly mode
  if [[ $DEVICE == "gpu" ]]; then
      echo "Installing jax-nightly, jaxlib-nightly"
      # Install jax-nightly
      pip install -U --pre jax jaxlib jax-cuda12-plugin[with_cuda] jax-cuda12-pjrt -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
      # Install Transformer Engine
      export NVTE_FRAMEWORK=jax
      pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@stable
  elif [[ $DEVICE == "tpu" ]]; then
    echo "Installing jax-nightly,jaxlib-nightly"
    # Install jax-nightly
    pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    # Install jaxlib-nightly
    pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    # Install libtpu-nightly
    pip3 install --pre -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  fi
  echo "Installing nightly tensorboard plugin profile"
  pip3 install tbp-nightly --upgrade
else
  echo -e "\n\nError: You can only set MODE to [stable,nightly].\n\n"
  exit 1
fi

# Install maxdiffusion
pip3 install -U . || echo "Failed to install maxdiffusion" >&2
