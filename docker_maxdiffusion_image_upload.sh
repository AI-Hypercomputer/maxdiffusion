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

# This scripts takes a docker image that already contains the MaxText dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_maxdiffusion_image_upload.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Example command:
# bash docker_maxdiffusion_image_upload.sh BASEIMAGE=<<Base Image name>> CLOUD_IMAGE_NAME=${USER}_maxdiffusion

set -e

export PROJECT=$(gcloud config get-value project)

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ ! -v CLOUD_IMAGE_NAME ]]; then
  echo "Erroring out because CLOUD_IMAGE_NAME is unset, please set it!"
  exit 1
fi

echo "Building JAX SS MaxDiffusion . . ."
  
docker build --build-arg JAX_SS_BASEIMAGE=${BASEIMAGE} --network host -f ./maxdiffusion_jax_ss_tpu.Dockerfile -t ${CLOUD_IMAGE_NAME} .

docker tag ${CLOUD_IMAGE_NAME} gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest

docker push gcr.io/${PROJECT}/${CLOUD_IMAGE_NAME}:latest

echo "All done, check out your artifacts at: gcr.io/${PROJECT}/${CLOUD_IMAGE_NAME}:latest"
