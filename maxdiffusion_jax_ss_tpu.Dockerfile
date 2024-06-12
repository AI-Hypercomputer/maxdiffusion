ARG JAX_SS_BASEIMAGE

# JAX Stable Stack Base Image
From $JAX_SS_BASEIMAGE

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

# Install Python packages from requirements.txt
RUN pip install -r /deps/requirements.txt

# Install MaxDiffusion
RUN pip install .