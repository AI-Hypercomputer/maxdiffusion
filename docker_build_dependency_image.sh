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

# This scripts takes a docker image that already contains the MaxDiffusion dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_maxdiffusion_image_upload.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# bash docker_build_dependency_image.sh MODE=stable_stack BASEIMAGE={{JAX_STABLE_STACK BASEIMAGE FROM ARTIFACT REGISTRY}}
# bash docker_build_dependency_image.sh MODE=nightly
# bash docker_build_dependency_image.sh MODE=stable JAX_VERSION=0.4.13
# bash docker_build_dependency_image.sh MODE=stable

set -e

export LOCAL_IMAGE_NAME=maxdiffusion_base_image

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ -z ${MODE} ]]; then
  export MODE=stable
  echo "Default MODE=${MODE}"
fi

if [[ -z ${DEVICE} ]]; then
  export DEVICE=tpu
  echo "Default DEVICE=${DEVICE}"
fi
echo "DEVICE=${DEVICE}"

if [[ -z ${JAX_VERSION+x} ]] ; then
  export JAX_VERSION=NONE
  echo "Default JAX_VERSION=${JAX_VERSION}"
fi

COMMIT_HASH=$(git rev-parse --short HEAD)

echo "Building MaxDiffusion with MODE=${MODE} at commit hash ${COMMIT_HASH} . . ."

if [[ ${DEVICE} == "gpu" ]]; then
  if [[ ${MODE} == "pinned" ]]; then
    export BASEIMAGE=ghcr.io/nvidia/jax:base-2024-10-17
  else
    export BASEIMAGE=ghcr.io/nvidia/jax:base
  fi
  docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION --build-arg DEVICE=$DEVICE --build-arg BASEIMAGE=$BASEIMAGE -f ./maxdiffusion_gpu_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
else 
  if [[ "${MODE}" == "stable_stack" ]]; then
    if [[ ! -v BASEIMAGE ]]; then
      echo "Erroring out because BASEIMAGE is unset, please set it!"
      exit 1
    fi
    docker build --no-cache \
      --build-arg JAX_STABLE_STACK_BASEIMAGE=${BASEIMAGE} \
      --build-arg COMMIT_HASH=${COMMIT_HASH} \
      --network=host \
      -t ${LOCAL_IMAGE_NAME} \
      -f maxdiffusion_jax_stable_stack_tpu.Dockerfile .
  else
    docker build --no-cache \
      --network=host \
      --build-arg MODE=${MODE} \
      --build-arg JAX_VERSION=${JAX_VERSION} \
      -t ${LOCAL_IMAGE_NAME} \
      -f maxdiffusion_dependencies.Dockerfile .
  fi
fi