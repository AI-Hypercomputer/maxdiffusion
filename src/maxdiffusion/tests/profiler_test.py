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

import unittest
from unittest.mock import patch
from maxdiffusion import max_utils


# A simple mock configuration object to simulate pyconfig.config
class MockConfig:

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def get_keys(self):
    return self.__dict__


class ProfilerTest(unittest.TestCase):

  def setUp(self):
    # Reset the global state before each test
    max_utils._ml_run = None

  @patch("maxdiffusion.max_utils.machinelearning_run")
  @patch("maxdiffusion.max_utils.xprof")
  @patch("jax.process_index", return_value=0)
  def test_ml_diagnostics_profiler(self, mock_process_index, mock_xprof, mock_ml_run):
    """Tests that ML Diagnostics starts and stops correctly without hitting the real API."""
    config = MockConfig(
        enable_ml_diagnostics=True,
        profiler_gcs_path="gs://fake-bucket/profiler",
        enable_ondemand_xprof=True,
        run_name="test_run",
        enable_profiler=False,  # JAX profiler off
        tensorboard_dir="/tmp/fake_tensorboard",
    )

    # 1. Test manual initialization
    max_utils.ensure_machinelearning_job_runs(config)
    mock_ml_run.assert_called_once()

    # 2. Test Context Manager starts and stops xprof
    with max_utils.Profiler(config, session_name="test_session"):
      mock_xprof.return_value.start.assert_called_once_with("test_session")

    mock_xprof.return_value.stop.assert_called_once()

  @patch("jax.profiler.start_trace")
  @patch("jax.profiler.stop_trace")
  @patch("jax.process_index", return_value=0)
  def test_jax_profiler(self, mock_process_index, mock_stop_trace, mock_start_trace):
    """Tests that the standard JAX profiler starts and stops correctly."""
    config = MockConfig(enable_ml_diagnostics=False, enable_profiler=True, tensorboard_dir="/tmp/fake_tensorboard")

    with max_utils.Profiler(config):
      mock_start_trace.assert_called_once()

    mock_stop_trace.assert_called_once()

  @patch("maxdiffusion.max_utils.machinelearning_run")
  def test_profiler_disabled(self, mock_ml_run):
    """Tests that nothing runs if configs are set to False."""
    config = MockConfig(enable_ml_diagnostics=False, enable_profiler=False)

    max_utils.ensure_machinelearning_job_runs(config)
    mock_ml_run.assert_not_called()


if __name__ == "__main__":
  unittest.main()
