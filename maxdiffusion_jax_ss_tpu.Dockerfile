ARG JAX_SS_BASEIMAGE

# JAX Stable Stack Base Image
From $JAX_SS_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

ARG USE_MAXTEXT_REQUIREMENTS_FILE

# Install Maxdiffusion requirements
RUN if [ "${USE_MAXDIFFUSION_REQUIREMENTS_FILE}" = "true" ]; then \
        echo "Using MaxDiffusion requirements: /deps/requirements.txt" && \
        pip install -r /deps/requirements.txt; \
    else \
        echo "Not using MaxDiffusion requirements: /deps/requirements.txt"; \
    fi

# Install MaxDiffusion
RUN pip install .

# Run the script  available in JAX-SS base image to generate the manifest file
RUN bash /generate_manifest.sh PREFIX=maxdiffusion COMMIT_HASH=$COMMIT_HASH