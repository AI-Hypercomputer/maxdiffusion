ARG BASEIMAGE=maxdiffusion_base_image
FROM $BASEIMAGE

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .

WORKDIR /deps