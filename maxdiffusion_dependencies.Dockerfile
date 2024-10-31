# Use Python 3.10-slim-bullseye as the base image
FROM python:3.10-slim-bullseye

# Environment variable for no-cache-dir and pip root user warning
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PYTHON_VERSION=3.10
ENV CLOUD_SDK_VERSION=latest

# Set DEBIAN_FRONTEND to noninteractive to avoid frontend errors
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip --no-warn-script-location

# Install system dependencies
RUN apt-get update && apt-get install -y apt-utils git curl gnupg procps iproute2 ethtool && rm -rf /var/lib/apt/lists/*

# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*

# Install cloud-accelerator-diagnostics
RUN pip install cloud-accelerator-diagnostics

# Install cloud-tpu-diagnostics
RUN pip install cloud-tpu-diagnostics

# Install gcsfs
RUN pip install gcsfs

# Install google-cloud-storage
RUN pip install google-cloud-storage

# Args
ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .

RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION"

RUN --mount=type=cache,target=/root/.cache/pip bash setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION}

# Cleanup
RUN rm -rf /root/.cache/pip