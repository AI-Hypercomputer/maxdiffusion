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

# Example command:
# bash docker_maxdiffusion_image_upload.sh MODE=stable PROJECT_ID=tpu-prod-env-multipod BASEIMAGE=us-docker.pkg.dev/tpu-prod-env-multipod/jax-stable-stack/tpu:jax0.4.30-rev1 CLOUD_IMAGE_NAME=maxdiffusion-jax-stable-stack IMAGE_TAG=latest MAXDIFFUSION_REQUIREMENTS_FILE=requirements_with_jax_stable_stack.txt

# You need to specify a MODE {stable|nightly}, default value stable.

set -e

export LOCAL_IMAGE_NAME=maxdiffusion_base_image

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ ! -v PROJECT_ID ]]; then
  echo "Erroring out because PROJECT_ID is unset, please set it!"
  exit 1
fi

if [[ ! -v CLOUD_IMAGE_NAME ]]; then
  echo "Erroring out because CLOUD_IMAGE_NAME is unset, please set it!"
  exit 1
fi

if [[ ! -v IMAGE_TAG ]]; then
  echo "Erroring out because IMAGE_TAG is unset, please set it!"
  exit 1
fi

if [[ ! -v MAXDIFFUSION_REQUIREMENTS_FILE ]]; then
  echo "Erroring out because MAXDIFFUSION_REQUIREMENTS_FILE is unset, please set it!"
  exit 1
fi

if [[ -z MODE ]]; then
  export MODE=stable
  echo "Default MODE=${MODE}"
fi

# Default: Don't delete local image
DELETE_LOCAL_IMAGE="${DELETE_LOCAL_IMAGE:-false}"

gcloud auth configure-docker us-docker.pkg.dev --quiet

COMMIT_HASH=$(git rev-parse --short HEAD)


IMAGE_DATE=$(date +%Y-%m-%d)

IMAGE=us-docker.pkg.dev/${PROJECT_ID}/${CLOUD_IMAGE_NAME}/tpu:${IMAGE_TAG}-${IMAGE_DATE}

if [[ "${MODE}" == "nightly" ]]; then
  echo "Building MaxDiffusion with JAX and JAXLIB nightly at commit hash ${COMMIT_HASH} . . ."  
  docker build --no-cache \
    --build-arg MAXDIFFUSION_REQUIREMENTS_FILE=${MAXDIFFUSION_REQUIREMENTS_FILE} \
    --network=host \
    -t ${IMAGE} \
    -f maxdiffusion_tpu.Dockerfile .
else
  echo "Building JAX Stable Stack MaxDiffusion at commit hash ${COMMIT_HASH} . . ."  
  if [[ ! -v BASEIMAGE ]]; then
    echo "Erroring out because BASEIMAGE is unset, please set it!"
    exit 1
  fi
  docker build --no-cache \
    --build-arg JAX_STABLE_STACK_BASEIMAGE=${BASEIMAGE} \
    --build-arg COMMIT_HASH=${COMMIT_HASH} \
    --build-arg MAXDIFFUSION_REQUIREMENTS_FILE=${MAXDIFFUSION_REQUIREMENTS_FILE} \
    --network=host \
    -t ${IMAGE} \
    -f maxdiffusion_jax_stable_stack_tpu.Dockerfile .
fi

docker push ${IMAGE}

echo "All done, check out your artifacts at: ${IMAGE}"

if [ "$DELETE_LOCAL_IMAGE" == "true" ]; then
  docker rmi ${IMAGE}
  echo "Local image deleted."
fi