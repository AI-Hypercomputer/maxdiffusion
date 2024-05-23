<!--
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/google/maxdiffusion/actions/workflows/UnitTests.yml)

# Overview 

MaxDiffusion is a collection of reference implementations of various latent diffusion models written in pure Python/Jax that run on XLA devices including Cloud TPUs and GPUs. MaxDiffusion aims to be a launching off point for ambitious Diffusion projects both in research and production. We encourage you to start by experimenting with MaxDiffusion out of the box and then fork and modify MaxDiffusion to meet your needs.

The goal of this project is to provide reference implementations for latent diffusion models that help developers get started with training, tuning, and serving solutions on XLA devices including Cloud TPUs and GPUs. We started with Stable Diffusion inference on TPUs, but welcome code contributions to grow.

MaxDiffusion supports 
* Stable Diffusion 2 base (training and inference)
* Stable Diffusion 2.1 (training and inference) 
* Stable Diffusion XL (training and inference).

**WARNING: The training code is purely experimental and is under development.**

# Table of Contents

* [Getting Started](#getting-started)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)

# Getting Started

We recommend starting with a single TPU host and then moving to multihost.

Minimum requirements: Ubuntu Version 22.04, Python 3.10 and Tensorflow >= 2.12.0.

## Getting Started: Local Development for single host
Local development is a convenient way to run MaxDiffusion on a single host. 

1. [Create and SSH to a single-host TPU (v4-8). ](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud)
2. Clone MaxDiffusion in your TPU VM.
3. Within the root directory of the MaxDiffusion `git` repo, install dependencies by running:
```bash
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install .
```
4. After installation completes, run the training script.

- Stable Diffusion 2 base

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name="my_run" base_output_directory="gs://your-bucket/"
  ```

  To generate images with a trained checkpoint, run:

  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run" pretrained_model_name_or_path=<your_saved_checkpoint_path> from_pt=False attention=dot_product
  ```

- Stable Diffusion XL

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_xl_run" base_output_directory="gs://your-bucket/" per_device_batch_size=1
  ```

  To generate images with a trained checkpoint, add `pretrained_model_name_or_path=<your_saved_checkpoint_path>` to the commands below.

5. To generate images, run the following command:
 
- Stable Diffusion 2 base
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run"

- Stable Diffusion 2.1
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base21.yml run_name="my_run"
  ```

- Stable Diffusion XL Lightning

  Multi host inference is supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run" lightning_repo="ByteDance/SDXL-Lightning" lightning_ckpt="sdxl_lightning_4step_unet.safetensors"
  ```
- Stable Diffusion XL

  Multi host inference is supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run"
  ```

  Single host pmap version:
  
  ```bash
  python -m src.maxdiffusion.generate_sdxl_replicated
  ```

## Getting Started: Multihost development
Multihost training for Stable Diffusion 2 base can be run using the following command:
```bash
TPU_NAME=<your-tpu-name>
ZONE=<your-zone>
PROJECT_ID=<your-project-id>
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project $PROJECT_ID --worker=all --command="
git clone https://github.com/google/maxdiffusion
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install .
python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=my_run base_output_directory=gs://your-bucket/"
```

# Comparison to Alternatives

MaxDiffusion started as a fork of [Diffusers](https://github.com/huggingface/diffusers), a Hugging Face diffusion library written in Python, Pytorch and Jax. MaxDiffusion is compatible with Hugging Face Jax models. MaxDiffusion is more complex and was designed to run distributed across TPU Pods. 

# Development

Whether you are forking MaxDiffusion for your own needs or intending to contribute back to the community, a full suite of tests can be found in `tests` and `src/maxdiffusion/tests`.

To run unit tests and lint, simply run:
```
python -m pytest
ruff check --fix .
```

The full suite of -end-to end tests is in `tests` and `src/maxdiffusion/tests`. We run them with a nightly cadance.
