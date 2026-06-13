# Model Configs

This directory contains model configuration for different Stable Diffusion models.

## Stable Diffusion 1.5

base15.yml - used for training and inference using [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
The upstream checkpoint ships PyTorch weights only, so this config sets `from_pt: True`; point
`pretrained_model_name_or_path` at a local diffusers snapshot for offline runs. It defaults to the
checkpoint's PNDM scheduler (epsilon prediction) to match the reference inference path.

## Stable Diffusion 2.1

base21.yml - used for training and inference using [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

## Stable Diffusion 2 Base

base_2_base.yml - used for training and inference using [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)

## Stable Diffusion XL & SDXL Lightning

base_xl.yml - used to run inference using [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

base_xl_lightning.yml - used to run inference using [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)

## Flux

base_flux_dev.yml - used for training and inference using [Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

base_flux_schnell.yml - used for training and inference using [Flux Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)