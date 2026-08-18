"""
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
"""

import datetime
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from maxdiffusion import max_utils, train_utils


class MockConfig:

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def get_keys(self):
    return self.__dict__


class MetricsTest(unittest.TestCase):

  def setUp(self):
    max_utils._ml_run = None
    train_utils._buffered_step = None
    train_utils._buffered_metrics = None

  def test_ml_diagnostics_enabled(self):
    config_enabled = MockConfig(enable_ml_diagnostics=True)
    config_disabled = MockConfig(enable_ml_diagnostics=False)
    config_missing = MockConfig()

    self.assertTrue(max_utils.ml_diagnostics_enabled(config_enabled))
    self.assertFalse(max_utils.ml_diagnostics_enabled(config_disabled))
    self.assertFalse(max_utils.ml_diagnostics_enabled(config_missing))

  def test_clean_config_dict(self):
    config = MockConfig(
        run_name="test_run",
        learning_rate=0.001,
        infinity_val=float("inf"),
        nan_val=float("nan"),
        batch_size=16,
    )
    cleaned = max_utils._clean_config_dict(config)
    self.assertEqual(cleaned["run_name"], "test_run")
    self.assertEqual(cleaned["learning_rate"], 0.001)
    self.assertEqual(cleaned["batch_size"], 16)
    self.assertNotIn("infinity_val", cleaned)
    self.assertNotIn("nan_val", cleaned)

  @patch("maxdiffusion.max_utils.machinelearning_run", None)
  def test_ensure_machinelearning_job_runs_raises_import_error(self):
    config = MockConfig(enable_ml_diagnostics=True)
    with self.assertRaises(ImportError) as ctx:
      max_utils.ensure_machinelearning_job_runs(config)
    self.assertIn("enable_ml_diagnostics is True", str(ctx.exception))

  @patch("maxdiffusion.max_utils.machinelearning_run")
  def test_ensure_machinelearning_job_runs_success(self, mock_ml_run):
    config = MockConfig(
        enable_ml_diagnostics=True,
        run_name="my_run",
        profiler_gcs_path="gs://my-bucket/profiler",
        enable_ondemand_xprof=True,
    )
    max_utils.ensure_machinelearning_job_runs(config)
    mock_ml_run.assert_called_once_with(
        name="my_run",
        gcs_path="gs://my-bucket/profiler",
        configs=max_utils._clean_config_dict(config),
        on_demand_xprof=True,
        log_system_metrics=True,
        region=None,
    )

  def test_record_scalar_metrics(self):
    metrics = {"scalar": {}}
    step_delta = datetime.timedelta(seconds=2.5)
    train_utils.record_scalar_metrics(
        metrics=metrics,
        step_time_delta=step_delta,
        per_device_tflops=100.0,
        lr=0.0001,
        total_weights=1500000000,
    )
    scalars = metrics["scalar"]
    self.assertEqual(scalars["perf/step_time_seconds"], 2.5)
    self.assertEqual(scalars["perf/per_device_tflops"], 100.0)
    self.assertEqual(scalars["perf/per_device_tflops_per_sec"], 40.0)
    self.assertEqual(scalars["learning/current_learning_rate"], 0.0001)
    self.assertEqual(scalars["learning/total_weights"], 1500000000.0)

  @patch("maxdiffusion.train_utils.mld_metrics")
  @patch("jax.process_index", return_value=0)
  def test_write_metrics_mld_dispatch_master(self, mock_process_index, mock_mld_metrics):
    config = MockConfig(
        enable_ml_diagnostics=True,
        metrics_file=False,
        gcs_metrics=False,
        log_period=10,
        tensorboard_dir="/tmp/tensorboard",
    )
    mock_writer = MagicMock()

    # Step 0 (buffers metric)
    metrics_step_0 = {
        "scalar": {
            "learning/loss": np.array(0.42),
            "custom/accuracy": 0.95,
        }
    }
    train_utils.record_scalar_metrics(
        metrics=metrics_step_0,
        step_time_delta=datetime.timedelta(seconds=1.0),
        per_device_tflops=50.0,
        lr=0.0001,
        total_weights=1000000,
    )
    train_utils.write_metrics(mock_writer, None, None, metrics_step_0, 0, config)
    mock_mld_metrics.record_metrics.assert_not_called()

    # Step 1 (flushes buffered step 0 metrics)
    metrics_step_1 = {"scalar": {"learning/loss": np.array(0.38)}}
    train_utils.record_scalar_metrics(
        metrics=metrics_step_1,
        step_time_delta=datetime.timedelta(seconds=1.0),
        per_device_tflops=50.0,
        lr=0.0001,
    )
    train_utils.write_metrics(mock_writer, None, None, metrics_step_1, 1, config)

    mock_mld_metrics.record_metrics.assert_called_once()
    records = mock_mld_metrics.record_metrics.call_args[0][0]

    # Verify records contain translated names and float values
    record_dict = {r["metric_name"]: r["value"] for r in records}
    self.assertAlmostEqual(record_dict[train_utils._METRICS_TO_MANAGED["learning/loss"]], 0.42, places=4)
    self.assertAlmostEqual(record_dict[train_utils._METRICS_TO_MANAGED["learning/current_learning_rate"]], 0.0001, places=6)
    self.assertAlmostEqual(record_dict[train_utils._METRICS_TO_MANAGED["learning/total_weights"]], 1000000.0, places=1)
    self.assertAlmostEqual(record_dict["custom/accuracy"], 0.95, places=4)

  @patch("maxdiffusion.train_utils.mld_metrics")
  @patch("jax.process_index", return_value=1)
  def test_write_metrics_mld_skipped_on_worker(self, mock_process_index, mock_mld_metrics):
    config = MockConfig(
        enable_ml_diagnostics=True,
        metrics_file=False,
        gcs_metrics=False,
        log_period=10,
    )
    mock_writer = MagicMock()

    metrics_0 = {"scalar": {"learning/loss": 0.5}}
    train_utils.write_metrics(mock_writer, None, None, metrics_0, 0, config)
    train_utils.write_metrics(mock_writer, None, None, metrics_0, 1, config)

    mock_mld_metrics.record_metrics.assert_not_called()


if __name__ == "__main__":
  unittest.main()
