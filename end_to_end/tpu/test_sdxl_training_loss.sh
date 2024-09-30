#!/bin/bash
set -ex

echo "Running test_sdxl_training_loss.sh"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

TRAIN_CMD="python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml \
        pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0 \
        revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 metrics_file=metrics.txt write_metrics=True \
        dataset_name=gs://jfacevedo-maxdiffusion-v5p/pokemon-datasets/pokemon-gpt4-captions_xl resolution=1024 per_device_batch_size=1 \
        jax_cache_dir=gs://jfacevedo-maxdiffusion/cache_dir/ max_train_steps=$STEPS attention=flash run_name=sdxl-fsdp-v5p-64-ddp enable_profiler=True  \
        run_name=$RUN_NAME \
        output_dir=$OUTPUT_DIR "

# Train
export LIBTPU_INIT_ARGS=""
$TRAIN_CMD

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD