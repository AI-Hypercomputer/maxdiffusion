"""
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
 """

"""
Example to run
python end_to_end/tpu/eval_assert.py avg_tflops metrics.txt 100
python end_to_end/tpu/eval_assert.py avg_step_time metrics.txt 0.5 100
python end_to_end/tpu/eval_assert.py avg_step_time metrics.txt 0.5 100
"""



# pylint: skip-file
"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
import json


def get_last_n_data(metrics_file, target, n=10):
  last_n_data = []
  with open(metrics_file, "r", encoding="utf8") as file:
    lines = file.readlines()
    for line in lines[::-1]:
      metrics = json.loads(line)
      if target in metrics:
        last_n_data.append(metrics[target])
        if len(last_n_data) >= n:
          break
  return last_n_data


def test_final_loss(metrics_file, target_loss, num_samples_str="10"):
  target_loss = float(target_loss)
  num_samples = int(num_samples_str)
  with open(metrics_file, "r", encoding="utf8") as _:
    last_n_data = get_last_n_data(metrics_file, "learning/loss",num_samples)
    avg_last_n_data = sum(last_n_data) / len(last_n_data)
    print(f"Mean of last {len(last_n_data)} losses is {avg_last_n_data}")
    print(f"Target loss is {target_loss}")
    assert avg_last_n_data < target_loss
    print("Final loss test passed.")


def test_avg_step_time(metrics_file, max_avg_step_time_str, num_samples_str="10"):
  """Tests if the average of the last N step times is below a maximum threshold."""
  max_avg_step_time = float(max_avg_step_time_str)
  num_samples = int(num_samples_str)
  metric_key = "perf/step_time_seconds"
  last_n_step_times = get_last_n_data(metrics_file, metric_key, num_samples)

  if not last_n_step_times:
    raise ValueError(f"Metric '{metric_key}' not found or no data points in {metrics_file}.")

  avg_last_n_step_time = sum(last_n_step_times) / len(last_n_step_times)

  print(f"Found {len(last_n_step_times)} data points for '{metric_key}'.")
  print(f"Mean of last {len(last_n_step_times)} step times is {avg_last_n_step_time:.4f} s")

  assert (
      avg_last_n_step_time < max_avg_step_time
  ), f"Average step time {avg_last_n_step_time:.4f}s is not less than target {max_avg_step_time}s."
  print("Average step time test passed.")


def test_avg_tflops(metrics_file, min_avg_tflops_str, num_samples_str="10"):
  """Tests if the average of the last N TFLOPs/sec values is above a minimum threshold."""
  min_avg_tflops = float(min_avg_tflops_str)
  num_samples = int(num_samples_str)
  metric_key = "perf/per_device_tflops_per_sec"

  last_n_tflops = get_last_n_data(metrics_file, metric_key, num_samples)

  if not last_n_tflops:
    raise ValueError(f"Metric '{metric_key}' not found or no data points in {metrics_file}.")

  avg_last_n_tflops = sum(last_n_tflops) / len(last_n_tflops)

  print(f"Found {len(last_n_tflops)} data points for '{metric_key}'.")
  print(f"Mean of last {len(last_n_tflops)} steps TFLOPs/sec is {avg_last_n_tflops:.2f}")

  assert (
      avg_last_n_tflops > min_avg_tflops
  ), f"Average TFLOPs/sec {avg_last_n_tflops:.2f} is not greater than target {min_avg_tflops}."
  print("Average TFLOPs/sec test passed.")


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    print("Usage: python script.py <test_scenario> [test_vars...]")
    print("Available scenarios: final_loss, avg_step_time, avg_tflops")
    raise ValueError("Test scenario not specified.")

  _, test_scenario, *test_vars = argv

  if test_scenario == "final_loss":
    if len(test_vars) < 2:
      raise ValueError("Usage: final_loss <metrics_file> <target_loss> [num_samples]")
    metrics_file, target_loss, *num_samples_opt = test_vars
    num_samples = num_samples_opt[0] if num_samples_opt else "10"
    test_final_loss(metrics_file, target_loss, num_samples)
  elif test_scenario == "avg_step_time":
    if len(test_vars) < 2:
      raise ValueError("Usage: avg_step_time <metrics_file> <max_avg_step_time> [num_samples]")
    metrics_file, max_avg_step_time, *num_samples_opt = test_vars
    num_samples = num_samples_opt[0] if num_samples_opt else "10"
    test_avg_step_time(metrics_file, max_avg_step_time, num_samples)
  elif test_scenario == "avg_tflops":
    if len(test_vars) < 2:
      raise ValueError("Usage: avg_tflops <metrics_file> <min_avg_tflops> [num_samples]")
    metrics_file, min_avg_tflops, *num_samples_opt = test_vars
    num_samples = num_samples_opt[0] if num_samples_opt else "10"
    test_avg_tflops(metrics_file, min_avg_tflops, num_samples)
  else:
    raise ValueError(f"Unrecognized test_scenario '{test_scenario}'. Available: final_loss, avg_step_time, avg_tflops")


if __name__ == "__main__":
  app.run(main)
