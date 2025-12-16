ARG JAX_AI_IMAGE_BASEIMAGE

# JAX AI Base Image
FROM $JAX_AI_IMAGE_BASEIMAGE

ARG JAX_AI_IMAGE_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .

# Install Maxdiffusion Jax AI Image requirements
RUN pip install -r /deps/requirements_with_jax_ai_image.txt

# TODO: Remove the flax pin and fsspec overrides once flax stable version releases
RUN if echo "$JAX_AI_IMAGE_BASEIMAGE" | grep -q "nightly"; then \
        echo "Nightly build detected: Installing specific Flax commit and fsspec." && \
        pip install --upgrade --force-reinstall git+https://github.com/google/flax.git@ef78d6584623511746be4824965cdef42b464583 && \
        pip install "fsspec==2025.10.0"; \
    fi

# Run the script available in JAX-AI-Image base image to generate the manifest file
RUN bash /jax-ai-image/generate_manifest.sh PREFIX=maxdiffusion COMMIT_HASH=$COMMIT_HASH