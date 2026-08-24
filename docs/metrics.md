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

# Metrics Collection and Monitoring with Google Cloud ML Diagnostics

This guide describes how to capture, monitor, and visualize training, system, and performance metrics in **MaxDiffusion** using the **Google Cloud ML Diagnostics SDK** (`google-cloud-mldiagnostics`).

---

## 1. Overview

MaxDiffusion integrates with Google Cloud ML Diagnostics to provide real-time telemetry during training runs on Cloud TPUs:
- **Workload Metrics**: In multi-host JAX jobs, step-level metrics (loss, step time, learning rate, gradient norm, parameter weights, custom activations) are buffered and dispatched from the master node (process index 0) to prevent duplicate logs.
- **System & Accelerator Metrics**: The SDK automatically runs background daemon threads on all worker hosts to capture hardware utilization (`tpu_duty_cycle`, `hbm_utilization`, `host_cpu_utilization`, `host_memory_utilization`).
- **Cloud Logging Sink**: Metrics are written to Google Cloud Logging.
- **Control Plane UI**: The Diagnostics Console automatically discovers and renders standard and custom metric plots.

---

## 2. Metric Types

### Predefined Metrics

MaxDiffusion automatically translates internal scalar keys to canonical `MetricType` enums expected by the Control Plane UI:

- **Loss** (`loss`): Training loss value per step (mapped from `learning/loss`).
- **Learning Rate** (`learning_rate`): Current optimizer learning rate (mapped from `learning/current_learning_rate`).
- **Gradient Norm** (`gradient_norm`): Global L2 norm of model gradients (mapped from `learning/grad_norm`).
- **Total Weights** (`total_weights`): Total trainable model parameter count (mapped from `learning/total_weights`).
- **Step Time** (`step_time`): Duration of each training step in seconds (mapped from `perf/step_time_seconds`).
- **TFLOPS** (`tflops`): Hardware compute throughput per accelerator in TFLOP/s (mapped from `perf/per_device_tflops_per_sec`).

### Custom Metrics

Any key in `metrics["scalar"]` that is not part of `_METRICS_TO_MANAGED` is treated as a **Custom Metric**:
- Retains its raw string name (e.g., `"custom/latents_mean"`, `"snr_loss_weight"`, `"cross_attn_entropy"`).
- Are dynamically discovered by the Control Plane UI and rendered in dedicated chart cards (`Over Time` and `Over Steps`).

### Automated System & Accelerator Metrics

When `enable_ml_diagnostics=True` is enabled, the SDK automatically captures:
- `tpu_duty_cycle`: Core accelerator compute utilization percentage.
- `hbm_utilization`: High Bandwidth Memory consumed percentage.
- `host_cpu_utilization`: Host CPU usage percentage.
- `host_memory_utilization`: Host system RAM usage percentage.

---

## 3. Integration Guide for Training Scripts

Metric mapping and dispatch are centralized in `train_utils.py` and `max_utils.py`. Authors of training scripts can integrate metrics using two steps:

### Step 1: Initialize MachineLearningRun

Initialize the run at the start of training:

```python
from maxdiffusion import max_utils

max_utils.ensure_machinelearning_job_runs(config)
```

### Step 2: Record Scalar Metrics in the Training Loop

Inside the trainer's `training_loop()`:

```python
from maxdiffusion import train_utils

# Record standard step metrics (and any custom metrics in train_metric["scalar"]):
train_utils.record_scalar_metrics(
    train_metric,
    step_time_delta,
    self.per_device_tflops,
    learning_rate_scheduler(step),
)

if self.config.write_metrics:
  train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
```

---

## 4. Configuration

Enable ML Diagnostics via YAML configuration files (using `enable_ml_diagnostics: True`) or command-line flags:

```yaml
# src/maxdiffusion/configs/base_2_base.yml
run_name: "my-training-run"
enable_ml_diagnostics: True
write_metrics: True
log_period: 10
```

> [!NOTE]
> To enable automated profiling and on-demand XProf traces alongside metrics, see the [ML Diagnostics Profiling Guide](profiling.md).

Run command:

```bash
python -m src.maxdiffusion.train src/maxdiffusion/configs/base_2_base.yml \
  run_name=my-training-run \
  output_dir=gs://my-bucket/output \
  enable_ml_diagnostics=True \
  write_metrics=True
```

---

## 5. Verification

### Google Cloud Logging

Inspect metric logs directly using `gcloud`:

```bash
# Query loss metrics
gcloud logging read 'logName="projects/<PROJECT_ID>/logs/ml_diagnostics_metric" AND resource.labels.namespace="loss"' \
  --limit=5 \
  --format="json"

# Query custom metrics
gcloud logging read 'logName="projects/<PROJECT_ID>/logs/ml_diagnostics_metric" AND resource.labels.namespace="custom/latents_mean"' \
  --limit=5 \
  --format="json"

# Query hardware metrics
gcloud logging read 'logName="projects/<PROJECT_ID>/logs/ml_diagnostics_metric" AND resource.labels.namespace="hbm_utilization"' \
  --limit=5 \
  --format="json"
```

### Google Cloud Console

1. Open Google Cloud Console and navigate to **Hypercompute Clusters** → **Diagnostics**.
2. Select your cluster and active `MachineLearningRun`.
3. Inspect:
   - **Model Metrics**: View predefined plots for `loss`, `learning_rate`, `gradient_norm`, and `total_weights`.
   - **Custom Metrics**: View dynamically generated charts for all `custom/*` metrics over time and steps.
   - **Performance**: View `step_time`, `tflops`, `tpu_duty_cycle`, and `hbm_utilization`.
