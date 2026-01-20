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

[![Unit Tests](https://github.com/AI-Hypercomputer/maxdiffusion/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxdiffusion/actions/workflows/UnitTests.yml)

# What's new?
- **`2026/01/29`**: Wan LoRA for inference is now supported
- **`2026/01/15`**: Wan2.1 and Wan2.2 Img2vid generation is now supported
- **`2025/11/11`**: Wan2.2 txt2vid generation is now supported
- **`2025/10/10`**: Wan2.1 txt2vid training and generation is now supported.
- **`2025/10/14`**: NVIDIA DGX Spark Flux support.
- **`2025/08/14`**: LTX-Video img2vid generation is now supported.
- **`2025/07/29`**: LTX-Video text2vid generation is now supported.
- **`2025/04/17`**: Flux Finetuning.
- **`2025/02/12`**: Flux LoRA for inference.
- **`2025/02/08`**: Flux schnell & dev inference.
- **`2024/12/12`**: Load multiple LoRAs for inference.
- **`2024/10/22`**: LoRA support for Hyper SDXL.
- **`2024/08/01`**: Orbax is the new default checkpointer. You can still use `pipeline.save_pretrained` after training to save in diffusers format.
- **`2024/07/20`**: Dreambooth training for Stable Diffusion 1.x,2.x is now supported.

# Overview

MaxDiffusion is a collection of reference implementations of various latent diffusion models written in pure Python/Jax that run on XLA devices including Cloud TPUs and GPUs. MaxDiffusion aims to be a launching off point for ambitious Diffusion projects both in research and production. We encourage you to start by experimenting with MaxDiffusion out of the box and then fork and modify MaxDiffusion to meet your needs.

The goal of this project is to provide reference implementations for latent diffusion models that help developers get started with training, tuning, and serving solutions on XLA devices including Cloud TPUs and GPUs. We started with Stable Diffusion inference on TPUs, but welcome code contributions to grow.

MaxDiffusion supports
* Stable Diffusion 2 base (inference)
* Stable Diffusion 2.1 (training and inference)
* Stable Diffusion XL (training and inference).
* Flux Dev and Schnell (Training and inference).
* Stable Diffusion Lightning (inference).
* Hyper-SD XL LoRA loading (inference).
* Load Multiple LoRA (SDXL inference).
* ControlNet inference (Stable Diffusion 1.4 & SDXL).
* Dreambooth training support for Stable Diffusion 1.x,2.x.
* LTX-Video text2vid, img2vid (inference).
* Wan2.1 text2vid (training and inference).
* Wan2.2 text2vid (inference).

**Note on GPU Support:** GPU support is not actively maintained, but contributions are welcome


# Table of Contents

- [What's new?](#whats-new)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Getting Started:](#getting-started-1)
  - [NVIDIA DGX Spark](#nvidia-dgx-spark)
  - [Training](#training)
    - [Wan2.1](#wan-21-training)
    - [Flux](#flux-training)
    - [SDXL](#stable-diffusion-xl-training)
    - [SD 2 base](#stable-diffusion-2-base-training)
    - [SD 1.4](#stable-diffusion-14-training)
    - [Dreambooth](#dreambooth)
  - [Inference](#inference)
    - [Wan](#wan-models)
    - [LTX-Video](#ltx-video)
    - [Flux](#flux)
      - [Fused Attention for GPU](#fused-attention-for-gpu)
    - [SDXL](#stable-diffusion-xl)
    - [SD 2 base](#stable-diffusion-2-base)
    - [SD 2.1](#stable-diffusion-21)
    - [Wan LoRA](#wan-lora)
    - [Flux LoRA](#flux-lora)
    - [Hyper SDXL LoRA](#hyper-sdxl-lora)
    - [Load Multiple LoRA](#load-multiple-lora)
    - [SDXL Lightning](#sdxl-lightning)
    - [ControlNet](#controlnet)
  - [Getting Started: Multihost development](#getting-started-multihost-development)
- [Comparison to Alternatives](#comparison-to-alternatives)
- [Development](#development)

# Getting Started

We recommend starting with a single TPU host and then moving to multihost.

Minimum requirements: Ubuntu Version 22.04, Python 3.12 and Tensorflow >= 2.12.0.

## Getting Started:

For your first time running Maxdiffusion, we provide specific [instructions](docs/getting_started/first_run.md).

## NVIDIA DGX Spark

Try out MaxDiffusion on NVIDIA's DGX Spark. We provide specific [instructions](docs/dgx_spark.md).

## Training

After installation completes, run the training script.

  ## Wan 2.1 Training

  in the first part, we'll run on a single host VM to get familiar with the workflow, then run on xpk for large scale training.

  Although not required, attaching an external disk is recommended as weights take up a lot of disk space. [Follow these instructions if you would like to attach an external disk](https://cloud.google.com/tpu/docs/attach-durable-block-storage).

  This workflow was tested using v5p-8 with a 500GB disk attached.

  ### Dataset Preparation

  For this example, we'll be using the [PusaV1 dataset](https://huggingface.co/datasets/RaphaelLiu/PusaV1_training).

  First, download the dataset.

  ```bash
  export HF_DATASET_DIR=/mnt/disks/external_disk/PusaV1_training/
  export TFRECORDS_DATASET_DIR=/mnt/disks/external_disk/wan_tfr_dataset_pusa_v1
  huggingface-cli download RaphaelLiu/PusaV1_training --repo-type dataset --local-dir $HF_DATASET_DIR
  ```

  Next run the TFRecords conversion script. This step prepares training and eval datasets. Validation is done as described in  [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206). More details [here](https://github.com/mlcommons/training/tree/master/text_to_image#5-quality)

  Training dataset.

  ```bash
  python src/maxdiffusion/data_preprocessing/wan_pusav1_to_tfrecords.py src/maxdiffusion/configs/base_wan_14b.yml train_data_dir=$HF_DATASET_DIR tfrecords_dir=$TFRECORDS_DATASET_DIR/train no_records_per_shard=10 enable_eval_timesteps=False
  ```

  The script will not have an output, but you can check the progress using:

  ```bash
  ls -ll $TFRECORDS_DATASET_DIR/train
  ```

  Evaluation dataset.

  ```bash
  python src/maxdiffusion/data_preprocessing/wan_pusav1_to_tfrecords.py src/maxdiffusion/configs/base_wan_14b.yml train_data_dir=$HF_DATASET_DIR tfrecords_dir=$TFRECORDS_DATASET_DIR/eval no_records_per_shard=10 enable_eval_timesteps=True
  ```

  The evaluation dataset creation takes the first 420 samples of the dataset and adds a timestep field. We then need to manually delete the first 420 samples from the `train` folder so they are not used in training.


  ```bash
  printf "%s\n" $TFRECORDS_DATASET_DIR/train/file_*-*.tfrec | awk -F '[-.]' '$2+0 <= 420' | xargs -d '\n' rm
  ```

  And verify that they do not exist.

  ```bash
  printf "%s\n" $TFRECORDS_DATASET_DIR/train/file_*-*.tfrec | awk -F '[-.]' '$2+0 <= 420' | xargs -d '\n' echo
  ```

  After the script is done running, you should see the following directory structure inside `$TFRECORDS_DATASET_DIR`

  ```
  train
  eval_timesteps
  ```

  In some instances an empty file `file_42-430.tfrec` is created inside `eval_timesteps`, for sanity check, let's run a delete command.

  ```bash
  rm $TFRECORDS_DATASET_DIR/eval_timesteps/file_42-430.tfrec
  ```
  
  ### Training on a Single VM

  Loading the data is supported both locally from the disk created above, or from `gcs`. In this guide, we'll be using a gcs bucket to train. First copy the data to the GCS bucket.

  ```bash
  BUCKET_NAME=my-bucket
  gsutil -m cp -r $TFRECORDS_DATASET_DIR gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}
  ```

  Now run the training command:

  ```bash
  RUN_NAME=jfacevedo-wan-v5p-8-${RANDOM}
  OUTPUT_DIR=gs://$BUCKET_NAME/wan/
  DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/train/
  EVAL_DATA_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/eval_timesteps/
  SAVE_DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/save/
  ```

  ```bash
  export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_megacore_fusion_allow_ags=false \
  --xla_enable_async_collective_permute=true \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_async_all_to_all=true \
  --xla_tpu_enable_all_experimental_scheduler_features=true \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_tpu_host_transfer_overlap_limit=24 \
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED \
  --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
  --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
  --xla_should_add_loop_invariant_op_in_chain=ENABLED \
  --xla_max_concurrent_host_send_recv=100 \
  --xla_tpu_scheduler_percent_shared_memory_limit=100 \
  --xla_latency_hiding_scheduler_rerun=2 \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
  --xla_tpu_assign_all_reduce_scatter_layout=true'
  ```

  ```bash
  HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/ python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  attention='flash' \
  weights_dtype=bfloat16 \
  activations_dtype=bfloat16 \
  guidance_scale=5.0 \
  flow_shift=5.0 \
  fps=16 \
  skip_jax_distributed_system=False \
  run_name=${RUN_NAME} \
  output_dir=${OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR} \
  load_tfrecord_cached=True \
  height=1280 \
  width=720 \
  num_frames=81 \
  num_inference_steps=50 \
  jax_cache_dir=${OUTPUT_DIR}/jax_cache/ \
  max_train_steps=1000 \
  enable_profiler=True \
  dataset_save_location=${SAVE_DATASET_DIR} \
  remat_policy='HIDDEN_STATE_WITH_OFFLOAD' \
  flash_min_seq_length=0 \
  seed=$RANDOM \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
  per_device_batch_size=0.25 \
  ici_data_parallelism=1 \
  ici_fsdp_parallelism=4 \
  ici_tensor_parallelism=1
  ```

  It is important to note a couple of things:
  - per_device_batch_size can be a fractional, but must be a whole number when multiplied by number of devices. In this example, 0.25 * 4 (devices) = effective global batch size = 1.
  - The step time in v5p-8 with global batch size = 1 is large due to using `FULL` remat. On larger number of chips we can run larger batch sizes greatly increasing MFU, as we will see in the next session of deploying with xpk.
  - To enable eval during training set `eval_every` to a value > 0.
  - In Wan2.1, the ici_fsdp_parallelism axis is used for sequence parallelism, the ici_tensor_parallelism axis is used for head parallelism. 
    - You can enable both, keeping in mind that Wan2.1 has 40 heads and 40 must be evenly divisible by ici_tensor_parallelism.
    - For Sequence parallelism, the code pads the sequence length to evenly divide the sequence. Try out different ici_fsdp_parallelism numbers, but we find 2 and 4 to be the best right now.
  - For use on GPU it is recommended to enable the cudnn_te_flash attention kernel for optimal performance.
    - Best performance is achieved with the use of batch parallelism, which can be enabled by using the ici_fsdp_batch_parallelism axis. Note that this parallelism strategy does not support fractional batch sizes.
    - ici_fsdp_batch_parallelism and ici_fsdp_parallelism can be combined to allow for fractional batch sizes. However, padding is not currently supported for the cudnn_te_flash attention kernel and it is therefore required that the sequence length is divisible by the number of devices in the ici_fsdp_parallelism axis.
  - For benchmarking training performance on multiple data dimension input without downloading/re-processing the dataset, the synthetic data iterator is supported.
    - Set dataset_type='synthetic' and synthetic_num_samples=null to enable the synthetic data iterator.
    - The following overrides on data dimensions are supported:
      - synthetic_override_height: 720
      - synthetic_override_width: 1280
      - synthetic_override_num_frames: 85
      - synthetic_override_max_sequence_length: 512
      - synthetic_override_text_embed_dim: 4096
      - synthetic_override_num_channels_latents: 16
      - synthetic_override_vae_scale_factor_spatial: 8
      - synthetic_override_vae_scale_factor_temporal: 4

  You should eventually see a training run as:

  ```bash
  ***** Running training *****
  Instantaneous batch size per device = 0.25
  Total train batch size (w. parallel & distributed) = 1
  Total optimization steps = 1000
  Calculated TFLOPs per pass: 4893.2719
  Warning, batch dimension should be shardable among the devices in data and fsdp axis, batch dimension: 1, devices_in_data_fsdp: 4
  Warning, batch dimension should be shardable among the devices in data and fsdp axis, batch dimension: 1, devices_in_data_fsdp: 4
  Warning, batch dimension should be shardable among the devices in data and fsdp axis, batch dimension: 1, devices_in_data_fsdp: 4
  Warning, batch dimension should be shardable among the devices in data and fsdp axis, batch dimension: 1, devices_in_data_fsdp: 4
  completed step: 0, seconds: 142.395, TFLOP/s/device: 34.364, loss: 0.270
  To see full metrics 'tensorboard --logdir=gs://jfacevedo-maxdiffusion-v5p/wan/jfacevedo-wan-v5p-8-17263/tensorboard/'
  completed step: 1, seconds: 137.207, TFLOP/s/device: 35.664, loss: 0.144
  completed step: 2, seconds: 36.014, TFLOP/s/device: 135.871, loss: 0.210
  completed step: 3, seconds: 36.016, TFLOP/s/device: 135.864, loss: 0.120
  completed step: 4, seconds: 36.008, TFLOP/s/device: 135.894, loss: 0.107
  completed step: 5, seconds: 36.008, TFLOP/s/device: 135.895, loss: 0.346
  completed step: 6, seconds: 36.006, TFLOP/s/device: 135.900, loss: 0.169
  ```

  ### Deploying with XPK

  This assumes the user has already created an xpk cluster, installed all dependencies and the also created the dataset from the step above. For getting started with MaxDiffusion and xpk see [this guide](docs/getting_started/run_maxdiffusion_via_xpk.md).
  
  Using v5p-256 Then the command to run on xpk is as follows:

  ```bash
  RUN_NAME=jfacevedo-wan-v5p-8-${RANDOM}
  OUTPUT_DIR=gs://$BUCKET_NAME/wan/
  DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/train/
  EVAL_DATA_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/eval_timesteps/
  SAVE_DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/save/
  ```

  ```bash
  LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_megacore_fusion_allow_ags=false \
  --xla_enable_async_collective_permute=true \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_async_all_to_all=true \
  --xla_tpu_enable_all_experimental_scheduler_features=true \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_tpu_host_transfer_overlap_limit=24 \
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED \
  --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
  --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
  --xla_should_add_loop_invariant_op_in_chain=ENABLED \
  --xla_max_concurrent_host_send_recv=100 \
  --xla_tpu_scheduler_percent_shared_memory_limit=100 \
  --xla_latency_hiding_scheduler_rerun=2 \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
  --xla_tpu_assign_all_reduce_scatter_layout=true'
  ```

  ```bash
  python3 ~/xpk/xpk.py workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT \
  --zone=$ZONE \
  --device-type=$DEVICE_TYPE \
  --num-slices=1 \
  --command=" \
  HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/ python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  attention='flash' \
  weights_dtype=bfloat16 \
  activations_dtype=bfloat16 \
  guidance_scale=5.0 \
  flow_shift=5.0 \
  fps=16 \
  skip_jax_distributed_system=False \
  run_name=${RUN_NAME} \
  output_dir=${OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR} \
  load_tfrecord_cached=True \
  height=1280 \
  width=720 \
  num_frames=81 \
  num_inference_steps=50 \
  jax_cache_dir=${OUTPUT_DIR}/jax_cache/ \
  enable_profiler=True \
  dataset_save_location=${SAVE_DATASET_DIR} \
  remat_policy='HIDDEN_STATE_WITH_OFFLOAD' \
  flash_min_seq_length=0 \
  seed=$RANDOM \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
  per_device_batch_size=0.25 \
  ici_data_parallelism=32 \
  ici_fsdp_parallelism=4 \
  ici_tensor_parallelism=1 \
  max_train_steps=5000 \
  eval_every=100 \
  eval_data_dir=${EVAL_DATA_DIR} \
  enable_generate_video_for_eval=True" \
  --base-docker-image=${IMAGE_DIR} \
  --enable-debug-logs \
  --workload=${RUN_NAME} \
  --priority=medium \
  --max-restarts=0
  ```

  ## Flux Training

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

  ## Stable Diffusion XL Training

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

  ## Stable Diffusion 2 base Training

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train src/maxdiffusion/configs/base_2_base.yml run_name="my_run" jax_cache_dir=gs://your-bucket/cache_dir activations_dtype=float32 weights_dtype=float32 per_device_batch_size=2 precision=DEFAULT dataset_save_location=/tmp/my_dataset/ output_dir=gs://your-bucket/ attention=flash
  ```

  ## Stable Diffusion 1.4 Training

  ```bash
  export LIBTPU_INIT_ARGS=""
  python -m src.maxdiffusion.train src/maxdiffusion/configs/base14.yml run_name="my_run" jax_cache_dir=gs://your-bucket/cache_dir activations_dtype=float32 weights_dtype=float32 per_device_batch_size=2 precision=DEFAULT dataset_save_location=/tmp/my_dataset/ output_dir=gs://your-bucket/ attention=flash
  ```

  To generate images with a trained checkpoint, run:

  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run" output_dir=gs://your-bucket/ from_pt=False attention=dot_product
  ```

  ## Dreambooth

  Supported models are **Stable Diffusion 1.x,2.x**

  ```bash
  python src/maxdiffusion/dreambooth/train_dreambooth.py src/maxdiffusion/configs/base14.yml class_data_dir=<your-class-dir> instance_data_dir=<your-instance-dir> instance_prompt="a photo of ohwx dog" class_prompt="photo of a dog" max_train_steps=150 jax_cache_dir=<your-cache-dir> class_prompt="a photo of a dog" activations_dtype=bfloat16 weights_dtype=float32 per_device_batch_size=1 enable_profiler=False precision=DEFAULT cache_dreambooth_dataset=False learning_rate=4e-6 num_class_images=100 run_name=<your-run-name> output_dir=gs://<your-bucket-name>
  ```

## Inference

To generate images, run the following command:
  ## Stable Diffusion XL

  Single and Multi host inference is supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run"
  ```

  Single host pmap version:

  ```bash
  python -m src.maxdiffusion.generate_sdxl_replicated
  ```

  ## Stable Diffusion 2 base
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_2_base.yml run_name="my_run"
  ```

  ## Stable Diffusion 2.1
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base21.yml run_name="my_run"
  ```

  ## LTX-Video
  In the folder src/maxdiffusion/models/ltx_video/utils, run:

  ```bash
  python convert_torch_weights_to_jax.py --ckpt_path [LOCAL DIRECTORY FOR WEIGHTS] --transformer_config_path ../ltxv-13B.json
  ```

  In the repo folder, run:
  ```bash
  python src/maxdiffusion/generate_ltx_video.py src/maxdiffusion/configs/ltx_video.yml output_dir="[SAME DIRECTORY]" config_path="src/maxdiffusion/models/ltx_video/ltxv-13B.json"
  ```
  Img2video Generation: 
  
  Add conditioning image path as conditioning_media_paths in the form of ["IMAGE_PATH"] along with other generation parameters in the ltx_video.yml file. Then follow same instruction as above.

  ## Wan Models

  Although not required, attaching an external disk is recommended as weights take up a lot of disk space. [Follow these instructions if you would like to attach an external disk](https://cloud.google.com/tpu/docs/attach-durable-block-storage).

  Supports both Text2Vid and Img2Vid pipelines.

  The following command will run Wan2.1 T2V:

  ```bash
  HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/ \
  LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_reduce=true" \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  attention="flash" \
  num_inference_steps=50 \
  num_frames=81 \
  width=1280 \
  height=720 \
  jax_cache_dir=gs://jfacevedo-maxdiffusion/jax_cache/ \
  per_device_batch_size=.125 \
  ici_data_parallelism=2 \
  ici_context_parallelism=2 \
  flow_shift=5.0 \
  enable_profiler=True \
  run_name=wan-inference-testing-720p \
  output_dir=gs:/jfacevedo-maxdiffusion \
  fps=16 \
  flash_min_seq_length=0 \
  flash_block_sizes='{"block_q" : 3024, "block_kv_compute" : 1024, "block_kv" : 2048, "block_q_dkv": 3024, "block_kv_dkv" : 2048, "block_kv_dkv_compute" : 2048, "block_q_dq" : 3024, "block_kv_dq" : 2048 }' \
  seed=118445
  ```

  To run other Wan model inference pipelines, change the config file in the command above:

  * For Wan2.1 I2V, use `base_wan_i2v_14b.yml`.
  * For Wan2.2 T2V, use `base_wan_27b.yml`.
  * For Wan2.2 I2V, use `base_wan_i2v_27b.yml`.

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
  ## Wan LoRA

  Disclaimer: not all LoRA formats have been tested. Currently supports ComfyUI and AI Toolkit formats. If there is a specific LoRA that doesn't load, please let us know.

  First create a copy of the relevant config file eg: `src/maxdiffusion/configs/base_wan_{*}.yml`. Update the prompt and LoRA details in the config. Make sure to set `enable_lora: True`. Then run the following command:

  ```bash
  HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/ \
  LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_reduce=true" \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_i2v_14b.yml \   # --> Change to your copy
  jax_cache_dir=gs://jfacevedo-maxdiffusion/jax_cache/ \
  per_device_batch_size=.125 \
  ici_data_parallelism=2 \
  ici_context_parallelism=2 \
  run_name=wan-lora-inference-testing-720p \
  output_dir=gs:/jfacevedo-maxdiffusion \
  seed=118445 \
  enable_lora=True \
  ```

  Loading multiple LoRAs is supported as well.

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

  ### Stable Diffusion 1.4

  ```bash
  python src/maxdiffusion/controlnet/generate_controlnet_replicated.py
  ```

  ### Stable Diffusion XL

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


