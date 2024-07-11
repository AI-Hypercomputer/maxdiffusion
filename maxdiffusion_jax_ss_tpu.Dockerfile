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

ARG MAXDIFFUSION_REQUIREMENTS_FILE

# Install Maxdiffusion requirements
RUN if [ ! -z "${MAXDIFFUSION_REQUIREMENTS_FILE}" ]; then \
        echo "Using MaxDiffusion requirements: ${MAXDIFFUSION_REQUIREMENTS_FILE}" && \
        pip install -r /deps/${MAXDIFFUSION_REQUIREMENTS_FILE}; \
    fi

# Install MaxDiffusion
RUN pip install .

# Run the script available in JAX-SS base image to generate the manifest file
RUN bash /generate_manifest.sh PREFIX=maxdiffusion COMMIT_HASH=$COMMIT_HASH