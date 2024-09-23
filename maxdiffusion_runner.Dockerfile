ARG BASEIMAGE=maxdiffusion_base_image
FROM $BASEIMAGE

#FROM maxdiffusion_base_image

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .

WORKDIR /deps