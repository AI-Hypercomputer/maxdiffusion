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

# Benchmarks

The `benchmarks/` directory contains runnable evaluation workflows and benchmark
metadata for measuring MaxDiffusion model outputs. Each subdirectory should own
one benchmark suite or evaluation family.

Current support:

* `vbench/`: [VBench](https://github.com/Vchitect/VBench)
  video-generation evaluation support, based on the
  [original VBench paper](https://arxiv.org/pdf/2311.17982). This currently
  covers Wan text-to-video generation and VBench evaluation over a 110-prompt
  downsampled subset.

## Adding benchmark suites

Add a new subdirectory under `benchmarks/` for each new benchmark or evaluation
family. Keep suite-specific scripts, prompt files, metadata, adapters, and
README instructions inside that subdirectory.

When adding a new suite, document:

* what the benchmark measures
* which MaxDiffusion models are supported
* required input files and generated outputs
* how to run generation and evaluation
* where results are written locally or in GCS

## Current VBench flow

For the current Wan VBench workflow, generate videos first and then evaluate
them. This compressed evaluation generates and evaluates exactly one video per
prompt:

```bash
bash benchmarks/vbench/run_tpu_generation.sh --ssh GCS_BUCKET=<bucket> TPU_NAME=<tpu-vm> RUN_NAME=<run-name>
bash benchmarks/vbench/run_gpu_eval.sh --ssh GCS_BUCKET=<bucket> GPU_NAME=<gpu-vm> RUN_NAME=<run-name>
```

See `benchmarks/vbench/README.md` for the full script options and data layout.
