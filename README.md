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

# What's new?
- **`2025/04/17`**: Flux Finetuning.
- **`2025/02/12`**: Flux LoRA for inference.
- **`2025/02/08`**: Flux schnell & dev inference.
- **`2024/12/12`**: Load multiple LoRAs for inference.
- **`2024/10/22`**: LoRA support for Hyper SDXL.
- **`2024/8/1`**: Orbax is the new default checkpointer. You can still use `pipeline.save_pretrained` after training to save in diffusers format.
- **`2024/7/20`**: Dreambooth training for Stable Diffusion 1.x,2.x is now supported.

# Overview

MaxDiffusion is a collection of reference implementations of various latent diffusion models written in pure Python/Jax that run on XLA devices including Cloud TPUs and GPUs. MaxDiffusion aims to be a launching off point for ambitious Diffusion projects both in research and production. We encourage you to start by experimenting with MaxDiffusion out of the box and then fork and modify MaxDiffusion to meet your needs.

The goal of this project is to provide reference implementations for latent diffusion models that help developers get started with training, tuning, and serving solutions on XLA devices including Cloud TPUs and GPUs. We started with Stable Diffusion inference on TPUs, but welcome code contributions to grow.

MaxDiffusion supports
* Stable Diffusion 2 base (training and inference)
* Stable Diffusion 2.1 (training and inference)
* Stable Diffusion XL (training and inference).
* Stable Diffusion Lightning (inference).
* Hyper-SD XL LoRA loading (inference).
* Load Multiple LoRA (SDXL inference).
* ControlNet inference (Stable Diffusion 1.4 & SDXL).
* Dreambooth training support for Stable Diffusion 1.x,2.x.

**WARNING: The training code is purely experimental and is under development.**

# Table of Contents

- [What's new?](#whats-new)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Getting Started:](#getting-started-1)
  - [Training](#training)
  - [Dreambooth](#dreambooth)
  - [Inference](#inference)
  - [Flux](#flux)
    - [Fused Attention for GPU:](#fused-attention-for-gpu)
  - [Hyper SDXL LoRA](#hyper-sdxl-lora)
  - [Load Multiple LoRA](#load-multiple-lora)
  - [SDXL Lightning](#sdxl-lightning)
  - [ControlNet](#controlnet)
  - [Getting Started: Multihost development](#getting-started-multihost-development)
- [Comparison to Alternatives](#comparison-to-alternatives)
- [Development](#development)

# Getting Started

We recommend starting with a single TPU host and then moving to multihost.

Minimum requirements: Ubuntu Version 22.04, Python 3.10 and Tensorflow >= 2.12.0.

## Getting Started:

For your first time running Maxdiffusion, we provide specific [instructions](docs/getting_started/first_run.md).

## Training

After installation completes, run the training script.

- **Flux**

  Expected results on 1024 x 1024 images with flash attention and bfloat16:

  | Model | Accelerator | Sharding Strategy | Per Device Batch Size | Global Batch Size | Step Time (secs) |
  | --- | --- | --- | --- | --- | --- |
  | Flux-dev | v5p-8 | DDP | 1 | 4 | 1.31 |

  Flux finetuning has only been tested on TPU v5p.

  ```bash
  python src/maxdiffusion/train_flux.py src/maxdiffusion/configs/base_flux_dev.yml run_name="test-flux-train" output_dir="gs://<your-gcs-bucket>/" save_final_checkpoint=True  jax_cache_dir="/tmp/jax_cache"
  ```

  To generate images with a finetuned checkpoint, run:

  ```bash
  python src/maxdiffusion/generate_flux_pipeline.py src/maxdiffusion/configs/base_flux_dev.yml  run_name="test-flux-train" output_dir="gs://<your-gcs-bucket>/" jax_cache_dir="/tmp/jax_cache"
  ```

- **Stable Diffusion XL**

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_xl_run" output_dir="gs://your-bucket/" per_device_batch_size=1
  ```

  On GPUS with Fused Attention:

  First install Transformer Engine by following the [instructions here](#fused-attention-for-gpu).

  ```bash
  NVTE_FUSED_ATTN=1 python -m src.maxdiffusion.train_sdxl src/maxdiffusion/configs/base_xl.yml hardware=gpu run_name='test-sdxl-train' output_dir=/tmp/ train_new_unet=true train_text_encoder=false cache_latents_text_encoder_outputs=true max_train_steps=200 weights_dtype=bfloat16 resolution=512 per_device_batch_size=1 attention="cudnn_flash_te" jit_initializers=False
  ```

  To generate images with a trained checkpoint, run:

  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_xl.yml run_name="my_run" pretrained_model_name_or_path=<your_saved_checkpoint_path> from_pt=False attention=dot_product
  ```

- **Stable Diffusion 2 base**

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train src/maxdiffusion/configs/base_2_base.yml run_name="my_run" jax_cache_dir=gs://your-bucket/cache_dir activations_dtype=float32 weights_dtype=float32 per_device_batch_size=2 precision=DEFAULT dataset_save_location=/tmp/my_dataset/ output_dir=gs://your-bucket/ attention=flash
  ```

- **Stable Diffusion 1.4**

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train src/maxdiffusion/configs/base14.yml run_name="my_run" jax_cache_dir=gs://your-bucket/cache_dir activations_dtype=float32 weights_dtype=float32 per_device_batch_size=2 precision=DEFAULT dataset_save_location=/tmp/my_dataset/ output_dir=gs://your-bucket/ attention=flash
  ```

  To generate images with a trained checkpoint, run:

  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run" output_dir=gs://your-bucket/ from_pt=False attention=dot_product
  ```

  ## Dreambooth

  **Stable Diffusion 1.x,2.x**

  ```bash
  python src/maxdiffusion/dreambooth/train_dreambooth.py src/maxdiffusion/configs/base14.yml class_data_dir=<your-class-dir> instance_data_dir=<your-instance-dir> instance_prompt="a photo of ohwx dog" class_prompt="photo of a dog" max_train_steps=150 jax_cache_dir=<your-cache-dir> class_prompt="a photo of a dog" activations_dtype=bfloat16 weights_dtype=float32 per_device_batch_size=1 enable_profiler=False precision=DEFAULT cache_dreambooth_dataset=False learning_rate=4e-6 num_class_images=100 run_name=<your-run-name> output_dir=gs://<your-bucket-name>
  ```

## Inference

To generate images, run the following command:
- **Stable Diffusion XL**

  Single and Multi host inference is supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run"
  ```

  Single host pmap version:

  ```bash
  python -m src.maxdiffusion.generate_sdxl_replicated
  ```

- **Stable Diffusion 2 base**
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run"

- **Stable Diffusion 2.1**
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base21.yml run_name="my_run"
  ```
  ## Flux

  First make sure you have permissions to access the Flux repos in Huggingface.

  Expected results on 1024 x 1024 images with flash attention and bfloat16:

  | Model | Accelerator | Sharding Strategy | Batch Size | Steps | time (secs) |
  | --- | --- | --- | --- | --- | --- |
  | Flux-dev | v4-8 | DDP | 4 | 28 | 23 |
  | Flux-schnell | v4-8 | DDP | 4 | 4 | 2.2 |
  | Flux-dev | v6e-4 | DDP | 4 | 28 | 5.5 |
  | Flux-schnell | v6e-4 | DDP | 4 | 4 | 0.8 |
  | Flux-schnell | v6e-4 | FSDP | 4 | 4 | 1.2 |

  Schnell:

  ```bash
  python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_schnell.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt="photograph of an electronics chip in the shape of a race car with trillium written on its side" per_device_batch_size=1
  ```

  Dev:

  ```bash
  python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt="photograph of an electronics chip in the shape of a race car with trillium written on its side" per_device_batch_size=1
  ```

  If you are using a TPU v6e (Trillium), you can use optimized flash block sizes for faster inference. Uncomment Flux-dev [config](src/maxdiffusion/configs/base_flux_dev.yml#60) and Flux-schnell [config](src/maxdiffusion/configs/base_flux_schnell.yml#68)

  To keep text encoders, vae and transformer on HBM memory at all times, the following command shards the model across devices. 

  ```bash
  python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_schnell.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt="photograph of an electronics chip in the shape of a race car with trillium written on its side" per_device_batch_size=1 ici_data_parallelism=1 ici_fsdp_parallelism=-1 offload_encoders=False
  ```

    ## Fused Attention for GPU:
    Fused Attention for GPU is supported via TransformerEngine. Installation instructions:

    ```bash
    cd maxdiffusion
    pip install -U "jax[cuda12]"
    pip install -r requirements.txt
    pip install --upgrade torch torchvision
    pip install "transformer_engine[jax]
    pip install .
    ```

    Now run the command:

    ```bash
    NVTE_FUSED_ATTN=1 HF_HUB_ENABLE_HF_TRANSFER=1 python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt='A cute corgi lives in a house made out of sushi, anime' num_inference_steps=28 split_head_dim=True per_device_batch_size=1 attention="cudnn_flash_te" hardware=gpu
    ```

    ## Flux LoRA

    Disclaimer: not all LoRA formats have been tested. If there is a specific LoRA that doesn't load, please let us know.

    Tested with [Amateur Photography](https://civitai.com/models/652699/amateur-photography-flux-dev) and [XLabs-AI](https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main) LoRA collection.

    First download the LoRA file to a local directory, for example, `/home/jfacevedo/anime_lora.safetensors`. Then run as follows:

    ```bash
    python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt='A cute corgi lives in a house made out of sushi, anime' num_inference_steps=28 ici_data_parallelism=1 ici_fsdp_parallelism=-1 split_head_dim=True lora_config='{"lora_model_name_or_path" : ["/home/jfacevedo/anime_lora.safetensors"], "weight_name" : ["anime_lora.safetensors"], "adapter_name" : ["anime"], "scale": [0.8], "from_pt": ["true"]}'
    ```

    Loading multiple LoRAs is supported as follows:

    ```bash
    python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml jax_cache_dir=/tmp/cache_dir run_name=flux_test output_dir=/tmp/ prompt='A cute corgi lives in a house made out of sushi, anime' num_inference_steps=28 ici_data_parallelism=1 ici_fsdp_parallelism=-1 split_head_dim=True lora_config='{"lora_model_name_or_path" : ["/home/jfacevedo/anime_lora.safetensors", "/home/jfacevedo/amateurphoto-v6-forcu.safetensors"], "weight_name" : ["anime_lora.safetensors","amateurphoto-v6-forcu.safetensors"], "adapter_name" : ["anime","realistic"], "scale": [0.6, 0.6], "from_pt": ["true","true"]}'
    ```

  ## Hyper SDXL LoRA

  Supports Hyper-SDXL models from [ByteDance](https://huggingface.co/ByteDance/Hyper-SD)

  ```bash
  python src/maxdiffusion/generate_sdxl.py src/maxdiffusion/configs/base_xl.yml run_name="test-lora" output_dir=/tmp/ jax_cache_dir=/tmp/cache_dir/ num_inference_steps=2 do_classifier_free_guidance=False prompt="a photograph of a cat wearing a hat riding a skateboard in a park." per_device_batch_size=1 pretrained_model_name_or_path="Lykon/AAM_XL_AnimeMix" from_pt=True revision=main diffusion_scheduler_config='{"_class_name" : "FlaxDDIMScheduler", "timestep_spacing" : "trailing"}' lora_config='{"lora_model_name_or_path" : ["ByteDance/Hyper-SD"], "weight_name" : ["Hyper-SDXL-2steps-lora.safetensors"], "adapter_name" : ["hyper-sdxl"], "scale": [0.7], "from_pt": ["true"]}'
  ```

  ## Load Multiple LoRA

    Supports loading multiple LoRAs for inference. Both from local or from HuggingFace hub.

    ```bash
    python src/maxdiffusion/generate_sdxl.py src/maxdiffusion/configs/base_xl.yml run_name="test-lora" output_dir=/tmp/tmp/ jax_cache_dir=/tmp/cache_dir/ num_inference_steps=30 do_classifier_free_guidance=True prompt="ultra detailed diagram blueprint of a papercut Sitting MaineCoon cat, wide canvas, ampereart, electrical diagram, bl3uprint, papercut" per_device_batch_size=1 diffusion_scheduler_config='{"_class_name" : "FlaxDDIMScheduler", "timestep_spacing" : "trailing"}' lora_config='{"lora_model_name_or_path" : ["/home/jfacevedo/blueprintify-sd-xl-10.safetensors","TheLastBen/Papercut_SDXL"], "weight_name" : ["/home/jfacevedo/blueprintify-sd-xl-10.safetensors","papercut.safetensors"], "adapter_name" : ["blueprint","papercut"], "scale": [0.8, 0.7], "from_pt": ["true", "true"]}'
    ```

  ## SDXL Lightning

  Single and Multi host inference is supported with sharding annotations:

    ```bash
    python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl_lightning.yml run_name="my_run" lightning_repo="ByteDance/SDXL-Lightning" lightning_ckpt="sdxl_lightning_4step_unet.safetensors"
    ```

  ## ControlNet

  Might require installing extra libraries for opencv: `apt-get update && apt-get install ffmpeg libsm6 libxext6  -y`

  - Stable Diffusion 1.4

    ```bash
    python src/maxdiffusion/controlnet/generate_controlnet_replicated.py
    ```

  - Stable Diffusion XL

    ```bash
    python src/maxdiffusion/controlnet/generate_controlnet_sdxl_replicated.py
    ```


## Getting Started: Multihost development
Multihost training for Stable Diffusion 2 base can be run using the following command:
```bash
TPU_NAME=<your-tpu-name>
ZONE=<your-zone>
PROJECT_ID=<your-project-id>
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project $PROJECT_ID --worker=all --command="
export LIBTPU_INIT_ARGS=""
git clone https://github.com/google/maxdiffusion
cd maxdiffusion
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install .
python -m src.maxdiffusion.train src/maxdiffusion/configs/base_2_base.yml run_name=my_run output_dir=gs://your-bucket/"
```

# Comparison to Alternatives

MaxDiffusion started as a fork of [Diffusers](https://github.com/huggingface/diffusers), a Hugging Face diffusion library written in Python, Pytorch and Jax. MaxDiffusion is compatible with Hugging Face Jax models. MaxDiffusion is more complex and was designed to run distributed across TPU Pods.

# Development

Whether you are forking MaxDiffusion for your own needs or intending to contribute back to the community, a full suite of tests can be found in `tests` and `src/maxdiffusion/tests`.

To run unit tests simply run:
```
python -m pytest
```

This project uses `pylint` and `pyink` to enforce code style. Before submitting a pull request, please ensure your code passes these checks by running:

```
bash code_style.sh
```

This script will automatically format your code with `pyink` and help you identify any remaining style issues.


The full suite of -end-to end tests is in `tests` and `src/maxdiffusion/tests`. We run them with a nightly cadance.
