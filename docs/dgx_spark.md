# MaxDiffusion on Nvidia DGX Spark GPU: A complete User Guide

This guide provides a detailed step-by-step walkthrough for setting up and running the maxdiffusion library within a custom Docker environment on an ARM-based machine with NVIDIA GPU support. We will cover everything from building the optimized Docker image to generating your first image and retrieving it successfully.

## Prerequisites

Before you begin, ensure you have the following:

- Access to [Nvidia DGX Spark Box](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).
- The maxdiffusion source code cloned onto the machine.
  - Branch: dgx_spark
- An internet connection for the initial Docker build and for downloading models (if not cached).

## Part 1: Building the Optimized Docker Image

The foundation of a smooth workflow is a well-built Docker image. The following Dockerfile is optimized for build speed by caching dependencies, ensuring that code changes don't require a full reinstall of all libraries.

### Step1: Create the Dockerfile

In the root directory of your maxdiffusion project, create a file named box.Dockerfile and paste the following content into it.

```docker
# Nvidia Base image for ARM64 with CUDA support
# As JAX AI Image as it currently doesn't support ARM builds.
FROM nvcr.io/nvidia/cuda-dl-base@sha256:3631d968c12ef22b1dfe604de63dbc71a55f3ffcc23a085677a6d539d98884a4

# Set environment variables (these rarely change)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies (these change very infrequently)
RUN apt-get update && apt-get install -y python3 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# --- Dependency Installation Layer ---
# First, copy only the requirements file to leverage caching
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install other major Python libraries in separate layers for better caching
RUN pip install "jax[cuda13-local]==0.7.2"

# --- Application Code Layer ---
# Now, copy your application source code. This layer is rebuilt only when your code changes.
COPY . .

# Install the maxdiffusion package from the copied source
RUN pip install .

# Set a default command to keep the container running for interactive use
CMD ["/bin/bash"]
```

### Step2: Build the Image

Open your terminal on DGX Spark, navigate to the root directory of the maxdiffusion project, and run the build command:

```bash
docker build -f box.Dockerfile -t maxdiffusion-arm-gpu .
```

This command will execute the steps in your Dockerfile, download the necessary layers, install all dependencies, and create a local Docker image named `maxdiffusion-arm-gpu`. The first build may take some time. Subsequent builds will be much faster if you only change the source code.

## Part 2: Running the Container for Image Generation

To run the image generator effectively, we need to connect our local machine's folders to the container. This prevents re-downloading models and makes it easy to retrieve the output images.

### Step 1: Create a Local Output Directory

On your DGX Spark, create a directory to store the generated images.

```bash
mkdir -p ~/maxdiffusion_output
```

### Step 2a: Launch the Container with Volume Mounts

Run the following command to start an interactive session inside your container. This command links your Hugging Face cache (to avoid re-downloading models) and the output directory you just created.

```bash
docker run -it --gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-v ~/maxdiffusion_output:/tmp \
maxdiffusion-arm-gpu
```
Your terminal prompt will change, indicating you are now inside the running container.

#### Step 2b: Log in to Hugging Face (First-Time Setup)

You must do this once to download the required model weights.

```bash
#  [Inside  the  Docker  Container]
huggingface-cli  login
```

You will be prompted to paste a Hugging Face User Access Token.

1.  Go to[  huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) in your web browser.

2.  Copy your token (or create a new one with write permissions).

3.  Paste the token into the terminal and press Enter.


## Part 3: Generating Your First Image

Now that you are inside the container's interactive shell, you can execute the image generation script. Run the following command:

```bash
NVTE_FRAMEWORK=JAX NVTE_FUSED_ATTN=1 HF_HUB_ENABLE_HF_TRANSFER=1 python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt='A cute corgi lives in a house made out of sushi, anime' num_inference_steps=28 split_head_dim=True per_device_batch_size=1 attention="cudnn_flash_te" hardware=gpu
```
The script will initialize, use the models from your mounted cache, and begin the generation process.

## Part 4: Accessing Your Generated Image

The generation script saves the final image to its working directory (/app) inside the container. Here is the complete workflow to get that image onto your Laptop.

### Step 1: Copy the Image from Container to DGX Spark

Open a new terminal window. Do not close the terminal where the container is running.
First, find your container's ID:

```bash
docker ps
```

Look for the container with the image maxdiffusion-arm-gpu and note its ID (e.g., 9049895399fc).
Now, copy the image from the container to a temporary location on DGX Spark and fix its permissions.

```bash
# Copy the file to the /tmp/ directory on DGX Spark
docker cp 9049895399fc:/app/flux_0.png /tmp/flux_0.png

# Change the file's owner to your user to avoid permission errors
sudo chown username:username /tmp/flux_0.png
```

### Step 2: Copy the Image from DGX Spark to Your Laptop

Now, open the Terminal app on your Laptop and use the scp (secure copy) command to download the file from DGX Spark.

```bash
scp username@spark:/tmp/flux_0.png .
```

This command will download flux_0.png to the current directory on your Laptop. You can now view your generated image!

## Troubleshooting and Common Pitfalls

Here are solutions to common issues you might encounter:
- Error: `pip: command not found` during Docker build.
  - **Cause**: The base Docker image doesn't have pip in the system's default PATH.
  - **Solution**: The provided Dockerfile fixes this by explicitly installing python3-pip and using update-alternatives to create the necessary symbolic links.
- Error: `externally-managed-environment` during `pip install`.
  - **Cause**: Newer versions of Debian/Ubuntu protect system Python packages from being modified by pip.
  - **Solution**: The `ENV PIP_BREAK_SYSTEM_PACKAGES=1` line in the `Dockerfile` safely bypasses this protection within the container's isolated environment.
- Error: `OSError: ...is not a local folder and is not a valid model identifier`
  - **Cause**:  The script is trying to download models from the Hugging Face Hub because it cannot find them locally.
  - **Solution**: This is solved by launching the container with the `-v ~/.cache/huggingface:/root/.cache/huggingface` flag, which gives the container access to your local model cache.
- Error: `open ... permission denied` when trying to access a copied file.
  - **Cause**: Files copied from a Docker container with docker cp are owned by the root user by default.
  - **Solution**: After copying the file to the DGX Spark, immediately run `sudo chown your_user:your_user /path/to/file` to take ownership before trying to access or transfer it.
- Can't find the generated image.
  - **Cause**: The script may not be saving the image to the directory specified by the output_dir argument.
  - **Solution**: Always check the script's source code to confirm the final save location. As we discovered, generate_flux.py saves to the current working directory (/app), not /tmp. Knowing this allows you to copy the file from the correct location.
- If a process requires more memory than the available RAM, your system will crash with an "Out-of-Memory" (OOM) error.
  - `Swap memory is your safety net.` It's a designated space on your hard drive that the operating system uses as a "virtual" extension of your RAM. When RAM is full, the system moves less active data to the slower swap space, freeing up RAM for the immediate task. While it's slower than RAM, it's infinitely better than a system crash, ensuring your long-running training or generation jobs can complete successfully. For a machine with 119GB of RAM, adding 64GB of swap provides a robust buffer for memory-intensive operations.
  - Step 1: Create a 64GB Swap File
    -  Run these commands on your DGX Spark to create, format, and enable a permanent 64GB swap file.

    ```bash
    # Instantly allocate a 64GB file
    sudo fallocate -l 64G /swapfile
    # Set secure permissions (only root can access)
    sudo chmod 600 /swapfile
    # Format the file as swap space
    sudo mkswap /swapfile
    # Enable the swap file for the current session
    sudo swapon /swapfile
    # Add the swap file to the system's startup configuration to make it permanent
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    ```

  - Step 2: Verify Swap is Active
    - Check that the swap space is correctly configured.

    ```bash
    free -h
    # The output should now show a 64GB total for Swap.
    ```
