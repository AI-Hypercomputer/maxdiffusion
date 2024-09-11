# Common Learning Guide

# Overview

Please refer to the [README](../README.md#getting-started) of this repository beforehand and set up the environment.

The following is explained:

1. [Training Scripts](#training-scripts).
1. [Configs](#configs).
    * [Changing The Base Model](#changing-the-base-model).
    * [Changing The Sharding Strategy](#changing-the-sharding-strategy).
    * [Checkpointing](#checkpointing).

## Training Scripts

MaxDiffusion provides training scripts:

 * [train.py](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/train.py) : supports training sd1.x, sd 2 base and sd2.1.
 * [train_dreambooth.py](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/dreambooth/train_dreambooth.py) : supports training dreambooth sd1.x, sd 2 base, and sd2.1.
 * [train_sdxl.py](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/train_sdxl.py) : supports sdxl training.

## Configs

The maxdiffusion repo is based on [configuration files](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion/configs) with the idea that few to no code changes will be required to run a training or inference job and config settings are modified instead. 

In this session, we'll explain some of the core config parameters and how they affect training. Let's start with configuration to model mappings:

| config | model | supports |
| ------ | ----- | -------- |
| [base14.yml](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base14.yml) | [stable-diffusion-v1-4](CompVis/stable-diffusion-v1-4) | training / inference
| [base_2_base.yml](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base_2_base.yml) | [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) | training / inference
| [base21.yml](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base21.yml) | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | training / inference
| [base_xl.yml](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base_xl.yml) | [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | training / inference
| [base_xl_lightning.yml](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base_xl_lightning.yml) | [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) & [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | inference

Changes to a config can be applied by changing the yml file directly or by passing those parameters in cli when creating a job. The only required parameters to pass to a job are `run_name` and `output_dir`. 

Let's start with a simple example. After setting up your environment, create a training job as follows:

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base14.yml run_name="my_run" jax_cache_dir=gs://your-bucket/cache_dir activations_dtype=float32 weights_dtype=float32 per_device_batch_size=2 precision=DEFAULT dataset_save_location=/tmp/my_dataset/ output_dir=gs://your-bucket/ attention=flash
  ```

The job will use the predefined parameters in base14.yml and will overwrite any parameters that as passed into the cli.

### Changing The Base Model

MaxDiffusion configs come with predefined models, mostly based on the base models created by [StabilityAI](https://stability.ai/) and [RunwayAI](https://runwayml.com/). The base model can be changed by setting `pretrained_model_name_or_path` to a different model, the only requirement is that the model is in diffusers format (full checkpoints will be supported in the future). 

To load Pytorch weights, set `from_pt=True` and set `revision=main`. Let's look at an example. Here we'll load [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) from a Pytorch checkpoint.

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base14.yml run_name="my_run" output_dir="gs://your-bucket/" pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 from_pt=True revision=main
  ```

After training, a new folder structure with weights and metrics has been created under the `output_dir` folder:

  ```bash
  ├── output_dir
  │   ├── run_name
  │       ├── checkpoints
  │       ├── metrics
  │       ├── tensorboard
  ```

It is recommended to use a Google Cloud Storage bucket as the `output_dir`. This will ensure all your work persists across VM creations. You can also use a local directory.

To use the trained checkpoint, then run:

  ```bash
  python src/maxdiffusion/generate.py src/maxdiffusion/configs/base14.yml output_dir="gs://your-bucket/" run_name="my_run"
  ```


### Changing The Sharding Strategy

MaxDiffusion models use logical axis annotations, which allows users to explore different sharding layouts without making changes to the model code. To learn more about distributed arrays and Flax partitioning, checkout JAX's [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) and then FLAX's [Scale up Flax Modules on multiple devices](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#flax-and-jax-jit-scaled-up)

The main [config values](https://github.com/google/maxdiffusion/blob/main/src/maxdiffusion/configs/base14.yml#L74) for these are:

- mesh_axes
- logical_axis_rules
- data_sharding
- dcn_data_parallelism
- dcn_fsdp_parallelism
- dcn_tensor_parallelism
- ici_data_parallelism
- ici_fsdp_parallelism
- ici_tensor_parallelism

**Out of the box, all maxdiffusion configs use data parallelism.**

`mesh_axes` supports 3 mesh axes: data, fsdp and tensor.

`logical_axis_rules` are used to define which weights and activations should be sharded across a mesh axes. 

`data_sharding` defines the data sharding strategy.

`dcn_*` stands for data center network parallelism and define parallelism strategies for [TPU multi-slice](https://cloud.google.com/tpu/docs/multislice-introduction).

`ici_*` stands for interchip interconnect parallelism and define parallelism strategies for [TPU single-slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

See [Multislice vs single slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#multislice).

**Note: maxdiffusion does not yet support multi-slice**.

Let's look at how these settings work to implement data parallelism. Let's assume we're using a TPUv4-8 and define the ici parallelism strategy:

  ```bash
  mesh_axes: ['data', 'fsdp', 'tensor']
  ici_data_parallelism: -1
  ici_fsdp_parallelism: 1  
  ici_tensor_parallelism: 1
  ```

Recall that in a TPUv4-8 configuration, the number of chips is 4 (each TPU v4 chip contains two TensorCores). Passing a -1 to an axis tells maxdiffusion to set all devices to that given axis, thus our mesh is created as `Mesh('data': 4, 'fsdp': 1, 'tensor': 1)`.

Now let's change the configuration as follows:

  ```bash
  mesh_axes: ['data', 'fsdp', 'tensor']
  ici_data_parallelism: 2
  ici_fsdp_parallelism: 2  
  ici_tensor_parallelism: 1
  ```

Then our mesh will look like `Mesh('data': 2, 'fsdp': 2, 'tensor': 1)`.

The `logical_axis_rules` specifies the sharding across the mesh. You are encouranged to add or remove rules and find what best works for you. 

### Checkpointing

Checkpointing can be enabled by using `checkpoint_every`. It is based on the number of samples (per_device_batch_size * jax.device_count()).

Orbax is used to save checkpoints, however, orbax does not currently store tokenizers. Instead the tokenizer model name or path is stored inside of the checkpoint and then loaded during inference. 