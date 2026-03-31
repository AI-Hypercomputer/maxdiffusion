# Wan Animate Setup Runbook (TPU v6e)

This is the exact setup we used so you can reproduce it later quickly.

## 1) Environment and venv

- Python venv in repo: `./.venv` (Python 3.11)
- Install deps with uv (using large disk cache):

```bash
UV_CACHE_DIR=/mnt/data/maxdiffusion/uv-cache \
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match --upgrade -r requirements.txt

UV_CACHE_DIR=/mnt/data/maxdiffusion/uv-cache \
uv pip install --python .venv/bin/python -e .
```

- Install TPU JAX runtime in venv:

```bash
UV_CACHE_DIR=/mnt/data/maxdiffusion/uv-cache \
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match --upgrade "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## 2) Where things are stored

We used the large disk at `/mnt/data`:

- HF home/cache: `/mnt/data/maxdiffusion/hf-home`
- UV cache: `/mnt/data/maxdiffusion/uv-cache`
- Wan assets cache: `/mnt/data/maxdiffusion/wan-assets`
- Wan2.2 repo clone: `/mnt/data/maxdiffusion/wan2.2`
- Preprocess checkpoints: `/mnt/data/maxdiffusion/checkpoints/Wan2.2-Animate-14B/process_checkpoint`

Create these dirs:

```bash
mkdir -p /mnt/data/maxdiffusion/{checkpoints,hf-home,uv-cache,wan-assets,wan2.2}
```

## 3) Preprocessing script and local outputs

Script added:

- `scripts/prepare_wan_animate_assets.sh`

What it does:

1. Downloads official sample from HF Space `Wan-AI/Wan2.2-Animate` (`mix/1` or `mov/1`)
2. Stages raw files to:
   - `examples/wan_animate/sample_inputs/video.mp4`
   - `examples/wan_animate/sample_inputs/image.jpeg`
3. Optionally downloads preprocess checkpoint (`--download-process-ckpt`)
4. Optionally runs official preprocess (`--run-preprocess`)

Run (download + preprocess):

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
export HF_HOME=/mnt/data/maxdiffusion/hf-home
export WAN_ASSETS_DIR=/mnt/data/maxdiffusion/wan-assets

./scripts/prepare_wan_animate_assets.sh \
  --sample mix \
  --mode animate \
  --download-process-ckpt \
  --run-preprocess \
  --wan22-root /mnt/data/maxdiffusion/wan2.2 \
  --ckpt-path /mnt/data/maxdiffusion/checkpoints/Wan2.2-Animate-14B/process_checkpoint
```

Preprocess outputs end up in:

- `examples/wan_animate/sample_inputs/process_results/src_pose.mp4`
- `examples/wan_animate/sample_inputs/process_results/src_face.mp4`

Notes:

- `--use_flux` is optional and disabled by default in our script.
- FLUX requires additional `FLUX.1-Kontext-dev` files under checkpoint path.

## 4) Wan2.2 repo and preprocess deps

Clone:

```bash
git clone https://github.com/Wan-Video/Wan2.2 /mnt/data/maxdiffusion/wan2.2
```

Extra preprocess deps we installed:

```bash
UV_CACHE_DIR=/mnt/data/maxdiffusion/uv-cache \
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  decord peft onnxruntime pandas matplotlib loguru sentencepiece moviepy
```

SAM2 note:

- Upstream SAM2 default build expects CUDA extension.
- For CPU-safe import, we installed from a local patched copy with CUDA extension disabled.

## 5) TPU verification and libtpu lock recovery

Check metadata (on TPU VM):

```bash
curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type
```

Verify JAX devices (use writable TPU logs dir):

```bash
mkdir -p .tpu_logs
TPU_LOG_DIR=$(pwd)/.tpu_logs .venv/bin/python -c "import jax; print(jax.__version__); print(jax.devices())"
```

If you see lock errors (`/tmp/libtpu_lockfile`) or “TPU already in use by pid …”:

```bash
fuser -v /tmp/libtpu_lockfile
kill <pid>
rm -f /tmp/libtpu_lockfile
```

Then re-run `jax.devices()`.

## 6) Config/code changes made

- Updated `src/maxdiffusion/generate_wan_animate.py`:
  - Supports local input paths: `reference_image_path`, `pose_video_path`, `face_video_path`
  - Supports `mode`, plus replace-mode paths (`background_video_path`, `mask_video_path`)
  - Keeps dummy fallback if pose/face paths are missing
  - Forces `scan_layers=False` for WAN animate runs as a temporary workaround for animate checkpoint loading

- Updated `src/maxdiffusion/configs/base_wan_27b.yml`:
  - Added keys:
    - `guidance_scale`
    - `mode`
    - `reference_image_path`
    - `pose_video_path`
    - `face_video_path`
    - `background_video_path`
    - `mask_video_path`

- Updated `src/maxdiffusion/models/wan/wan_utils.py`
  - Fixed Wan Animate checkpoint key remapping for:
    - `face_adapter_<n>` -> `face_adapter, <n>`
    - `motion_network_<n>` -> `motion_network, <n>`
  - Preserves raw `weight` params for animate motion-encoder linear layers instead of renaming them to `kernel`

- Updated `src/maxdiffusion/pipelines/wan/wan_pipeline.py`
  - Loads CLIP `image_processor` and `image_encoder` for WAN animate / i2v-style paths on Wan 2.2 as well

- Added `scripts/prepare_wan_animate_assets.sh`
  - Supports `/mnt/data` storage and optional checkpoint download/preprocess.

## 7) Important runtime gotchas seen

- `skip_jax_distributed_system=True` may be needed for local/single-host runs.
- Some configs in `base_wan_27b.yml` are tuned for TPU distributed runs; use reduced settings for smoke tests.
- Model support checks in `pyconfig` may reject some model IDs depending on Wan version routing.
- A stale `/tmp/libtpu_lockfile` can block TPU startup even when the code is fine. If `jax.devices()` fails, check the lock owner with `fuser -v /tmp/libtpu_lockfile`, kill the stale process, and remove the lockfile.
- For a true 1-sample smoke test on an 8-core TPU, use `per_device_batch_size=0.125`. Using `per_device_batch_size=1` becomes a global batch of 8 and makes compile/runtime much heavier.
- WAN animate currently does not load correctly with `scan_layers=True`. The animate checkpoint loader mismatches scanned-layer parameter structure, so runs should force `scan_layers=False` until the scanned loader path is fixed properly.
- Wan Animate on `Wan-AI/Wan2.2-Animate-14B-Diffusers` needed extra checkpoint-key normalization beyond base WAN:
  - `face_adapter_<n>` had to map to the nnx list structure `face_adapter, <n>`
  - `motion_encoder.motion_network_<n>` had to map to `motion_encoder.motion_network, <n>`
  - motion-network linear weights had to stay as `weight` instead of being generically remapped to `kernel`
- WAN animate uses CLIP image embeddings like the i2v path, even on Wan 2.2. If `image_processor` / `image_encoder` are not loaded, the run fails at `encode_image(...)` with `TypeError: 'NoneType' object is not callable`.
- The current smoke command reached real pipeline execution after model load:
  - loaded VAE
  - loaded animate transformer
  - loaded preprocessed `src_pose.mp4` and `src_face.mp4`
  - entered the pipeline call on TPU
- A runtime warning from `subprocess`/`os.fork()` appears during video loading after JAX has started multithreading. It is worth watching in the next session if we hit hangs during media IO or export.
