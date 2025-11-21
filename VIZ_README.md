# WAN 2.1 Visualization System

A visualization system for debugging and understanding the WAN 2.1 diffusion denoising process.

## Quick Start

1. **Enable visualization** in the config:
```yaml
# In src/maxdiffusion/configs/base_wan_14b.yml
visualize_frame_debug: true
visualization_output_dir: "wan_visualization_output"
# optional:
save_tensor_stats: True      # Save tensor statistics to JSON files
```

2. **Run inference**:
```bash
export RUN_NAME=wan21-8tpu
export LIBTPU_VERSION=libtpu-0.0.25.dev20251013+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl
export YOUR_GCS_BUCKET=gs://${USR_NAME}-wan-maxdiffusion

export OUTPUT_DIR=${YOUR_GCS_BUCKET}/wan/${RUN_NAME}
export DATASET_DIR=${YOUR_GCS_BUCKET}/wan_tfr_dataset_pusa_v1/train/
export EVAL_DATA_DIR=${YOUR_GCS_BUCKET}/wan_tfr_dataset_pusa_v1/eval_timesteps/
export SAVE_DATASET_DIR=${YOUR_GCS_BUCKET}/wan_tfr_dataset_pusa_v1/save/

export RANDOM=123456789
export IMAGE_DIR=gcr.io/tpu-prod-env-multipod/maxdiffusion_jax_stable_stack_nightly:2025-10-27
export LIBTPU_VERSION=libtpu-0.0.25.dev20251013+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl

export HUGGINGFACE_HUB_CACHE=/dev/shm

echo 'Starting WAN inference ...' && \
python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  enable_jax_named_scopes=False \
  attention='flash' \
  weights_dtype=bfloat16 \
  activations_dtype=bfloat16 \
  guidance_scale=5.0 \
  flow_shift=3.0 \
  fps=24 \
  skip_jax_distributed_system=True \
  run_name='test-wan-training-new' \
  output_dir=${OUTPUT_DIR} \
  load_tfrecord_cached=True \
  height=720 \
  width=1280 \
  num_frames=81 \
  num_inference_steps=50 \
  prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
  negative_prompt='low quality, over exposure.' \
  jax_cache_dir=${OUTPUT_DIR}/jax_cache/ \
  max_train_steps=20000 \
  enable_profiler=True \
  dataset_save_location=${SAVE_DATASET_DIR} \
  remat_policy='FULL' \
  flash_min_seq_length=0 \
  seed=$RANDOM \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
  per_device_batch_size=0.125 \
  allow_split_physical_axes=True \
  ici_data_parallelism=2 \
  ici_fsdp_parallelism=2 \
  ici_tensor_parallelism=2

echo 'WAN inference completed. Output saved to '${OUTPUT_DIR}
```

3. **View outputs** in `wan_visualization_output/frame_debug/`:
```
wan_visualization_output/
├── frame_debug/
│   ├── noise_t999_frame0.png              # Latent channel 0 at each timestep
│   ├── current_image_t999_frame0.png      # VAE-decoded image at each timestep
│   ├── ...                                # One pair per denoising step
│   ├── wan_visualization_denoising_process.mp4   # Video: noise → final image
│   └── wan_visualization_noise_evolution.mp4     # Video: latent evolution
```

## Output Examples

The system automatically generates two videos showing the complete denoising process:

```bash
$ ls wan_visualization_output/*.mp4
wan_visualization_output/wan_visualization_denoising_process.mp4
wan_visualization_output/wan_visualization_noise_evolution.mp4
```

**Videos** (4 fps, 0.25s per frame):
- `denoising_process.mp4`: Shows VAE-decoded images evolving from noise to final result
- `noise_evolution.mp4`: Shows raw latent space (channel 0) evolution during denoising

**Individual Frames**: 50+ timestamped PNG files showing step-by-step progression

## What We Built

### 1. **VisualizationMixin Architecture** 
- Reusable base class in `src/maxdiffusion/visualization/base_mixin.py`
- Common utilities: file I/O, plotting, statistics
- Used by `WanPipeline(VisualizationMixin)`

### 2. **Step-by-Step Visualization**
- **Modified**: `src/maxdiffusion/pipelines/wan/wan_pipeline.py`
- **Added**: `visualize_frame()` method with automatic calls during inference
- **Captures**: Both latent space and VAE-decoded representations at each timestep

### 3. **Automatic Video Generation**
- **Added**: `src/maxdiffusion/visualization/video_utils.py` 
- **Uses**: Same `imageio` method as WAN 2.1's video export
- **Creates**: Two videos automatically after inference completes

### 4. **Configuration-Driven**
- **Control**: `visualize_frame_debug: true/false` in config files
- **Output**: Configurable directory via `visualization_output_dir`

## Key Features

- **Zero overhead** when disabled (config-controlled)
- **Consistent sizing** (fixed matplotlib dimensions prevent video corruption)
- **Complete timeline** (50 timesteps from t=999 → t=57)
- **Automatic integration** (no separate scripts needed)
- **WAN-compatible** (uses same video export method as WAN 2.1)

## Technical Details

- **Latent visualization**: Shows channel 0 (10x8 matplotlib figure)
- **Image visualization**: VAE-decoded RGB frames (10x10 matplotlib figure)  
- **Video format**: MP4, 4 fps, imageio with quality=8
- **File naming**: `{type}_t{timestep}_frame{frame_idx}.png`
- **Statistics**: JSON files with tensor stats (shape, dtype, mean, std, etc.)

The system provides complete visibility into WAN's denoising process, from initial Gaussian noise to final coherent video frames.