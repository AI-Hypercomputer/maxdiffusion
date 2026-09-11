<!--
 Copyright 2026 Google LLC

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

# VBench

`benchmarks/vbench/` contains the MaxDiffusion integration for
[VBench](https://github.com/Vchitect/VBench), a video-generation benchmark suite
introduced in the
[original VBench paper](https://arxiv.org/pdf/2311.17982). The current workflow
is intentionally scoped to Wan text-to-video evaluation. Other MaxDiffusion
video models can be added here later by adding model-specific generation
defaults and documenting their output contract.

The current benchmark data is a 110-prompt downsampled subset of the full VBench
prompt set. This is a compressed workflow: it generates and evaluates exactly
one video per prompt.

* `prompts_110.txt`: prompts passed to Wan generation
* `VBench_full_info_sub110.json`: VBench metadata for the same prompts and
  dimensions

The prompt file and JSON metadata must stay aligned by order. Prompt `N` in
`prompts_110.txt` should match entry `N` in `VBench_full_info_sub110.json`.

## Files

* `run_tpu_generation.sh`: generates Wan videos on a TPU VM, or locally on a
  TPU host, and writes videos to GCS.
* `run_gpu_eval.sh`: downloads generated videos, prepares VBench-compatible
  filenames, runs VBench evaluation on a GPU VM or local GPU host, and uploads
  results to GCS by default.
* `gpu_eval_utils.py`: helper utilities for extracting VBench dimensions,
  patching the upstream VBench checkout for compatibility, and preparing the
  video manifest layout expected by VBench.
* `prompts_110.txt`: default Wan prompt file for this downsampled evaluation.
* `VBench_full_info_sub110.json`: default VBench metadata used by evaluation.

## Workflow

The scripts use a two-stage workflow:

1. Generate videos with MaxDiffusion Wan on TPU.
2. Evaluate the generated videos with VBench on GPU.

Use the same `GCS_BUCKET` and `RUN_NAME` for both stages. The generation script
stores videos at:

```text
gs://<bucket>/<run-name>/videos
```

The evaluation script reads from that location and writes results to:

```text
gs://<bucket>/<run-name>/vbench_results
```

## Generate Wan Videos

Run on a TPU VM over SSH:

```bash
bash benchmarks/vbench/run_tpu_generation.sh --ssh GCS_BUCKET=<bucket> TPU_NAME=<tpu-vm> RUN_NAME=<run-name>
```

Or run directly on a TPU host:

```bash
bash benchmarks/vbench/run_tpu_generation.sh GCS_BUCKET=<bucket> RUN_NAME=<run-name>
```

Common options:

* `PROMPT_FILE`: prompt file path. Defaults to
  `./benchmarks/vbench/prompts_110.txt`.
* `CONFIG_FILE`: Wan config. Defaults to `src/maxdiffusion/configs/base_wan_27b.yml`.
* `EXTERNAL_DISK`: mounted TPU disk root for large local files. Defaults to
  `/mnt/disks/external_disk`.
* `HF_CACHE_ROOT`: Hugging Face cache root. Defaults to
  `$EXTERNAL_DISK/hf_cache`.

The generation script keeps `GCS_VIDEO_DIR` fixed to `${RUN_NAME}/videos`,
because MaxDiffusion writes generated MP4s using
`base_output_directory=gs://${GCS_BUCKET}` and `run_name=${RUN_NAME}`. It uses
the fixed seed `12345`.

## Run VBench Evaluation

Run on a GPU VM over SSH:

```bash
bash benchmarks/vbench/run_gpu_eval.sh --ssh GCS_BUCKET=<bucket> GPU_NAME=<gpu-vm> RUN_NAME=<run-name>
```

Or run directly on a GPU host:

```bash
bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=<bucket> RUN_NAME=<run-name>
```

This is a compressed one-sample-per-prompt workflow. Evaluation rejects a
missing or duplicate video for any prompt.

Common options:

* `BENCHMARK_JSON`: VBench metadata file name. Defaults to
  `VBench_full_info_sub110.json`.
* `DIMENSIONS`: space-separated VBench dimensions. Defaults to reading all
  dimensions from `BENCHMARK_JSON`.
* `GCS_VIDEO_DIR`: generated video prefix. Defaults to `${RUN_NAME}/videos`.
* `GCS_RESULTS_DIR`: result prefix. Defaults to `${RUN_NAME}/vbench_results`.
* `UPLOAD_RESULTS`: whether to upload results to GCS. Defaults to `true`.
* `WORK_DIR`: working directory on the GPU host. Defaults to
  `$HOME/vbench_evaluation`.

In SSH mode, evaluation results are copied back from the GPU VM to the caller
machine and uploaded from the caller's credentials. This avoids relying on GPU
VM service-account scopes for GCS result uploads.

## Extending Support

Today this directory supports Wan VBench evals only. When adding another model,
keep the model-specific generation defaults explicit, document the command, and
ensure its output filenames can be mapped to the VBench prompt order.

When adding another VBench prompt subset, keep the prompt text file and JSON
metadata together, use matching names, and validate that their entries are in
the same order.
