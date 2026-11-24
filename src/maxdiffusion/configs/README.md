# Model Configs

This directory contains the base YAML configuration for every model family supported by MaxDiffusion
(Stable Diffusion, SDXL, Flux, Flux.2-Klein, Z-Image, Wan and LTX video models). Pass a config as the first
positional argument to a `train_*.py` / `generate_*.py` script and override any key on the command line,
e.g. `python src/maxdiffusion/generate_wan.py src/maxdiffusion/configs/base_wan_14b.yml run_name=my_run`.

## Stable Diffusion 1.4 / 1.5

base14.yml - used for training (including Dreambooth) and inference using [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).

base15.yml - used for training and inference using [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
The upstream checkpoint ships PyTorch weights only, so this config sets `from_pt: True`; point
`pretrained_model_name_or_path` at a local diffusers snapshot for offline runs. It defaults to the
checkpoint's PNDM scheduler (epsilon prediction) to match the reference inference path.

## Stable Diffusion 2.1

base21.yml - used for training and inference using [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

## Stable Diffusion 2 Base

base_2_base.yml - used for training and inference using [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)

## Stable Diffusion XL & SDXL Lightning

base_xl.yml - used for training and inference using [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

base_xl_lightning.yml - used to run inference using [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)

## Flux

base_flux_dev.yml - used for training and inference using [Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

base_flux_dev_multi_res.yml - used with `generate_flux_multi_res.py` for multi-resolution inference using [Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

base_flux_schnell.yml - used for training and inference using [Flux Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Flux.2-Klein

base_flux2klein.yml - used with `generate_flux2klein.py` for text-to-image and image editing using [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

base_flux2klein_9B.yml - same as above for [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)

## Z-Image

base_zimage.yml - used with `generate_zimage.py` for inference using [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)

base_zimage_turbo.yml - used with `generate_zimage.py` for inference using [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

## Wan 2.1

base_wan_1_3b.yml - used for text-to-video inference using [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)

base_wan_14b.yml - used for text-to-video training (`train_wan.py`) and inference (`generate_wan.py`) using [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)

base_wan_i2v_14b.yml - used for image-to-video inference using [Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers)

## Wan 2.2

base_wan_27b.yml - used for dual-expert text-to-video training (`train_wan.py`) and inference (`generate_wan.py`) using [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)

base_wan_i2v_27b.yml - used for image-to-video inference using [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)

base_wan_animate.yml - used with `generate_wan_animate.py` for inference using [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers)

## LTX-Video

ltx_video.yml - used with `generate_ltx_video.py` for inference using [LTX-Video](https://huggingface.co/Lightricks/LTX-Video)

ltx2_video.yml - used with `generate_ltx2.py` for inference using [LTX-2](https://huggingface.co/Lightricks/LTX-2)

ltx2_3_video.yml - used with `generate_ltx2.py` for inference using [LTX-2.3](https://huggingface.co/dg845/LTX-2.3-Diffusers)
