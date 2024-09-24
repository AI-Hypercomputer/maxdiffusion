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

# pylint: skip-file
"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
import json

def get_last_n_data(metrics_file, target, n=10):
  last_n_data = []
  with open(metrics_file, 'r', encoding='utf8') as file:
    lines = file.readlines()
    for line in lines[::-1]:
      metrics = json.loads(line)
      if target in metrics:
        last_n_data.append(metrics[target])
        if len(last_n_data) >= n:
          break
  return last_n_data


def test_final_loss(metrics_file, target_loss):
  target_loss = float(target_loss)
  with open(metrics_file, 'r', encoding='utf8') as _:
    use_last_n_data = 10
    last_n_data = get_last_n_data(metrics_file, 'learning/loss', use_last_n_data)
    avg_last_n_data = sum(last_n_data) / len(last_n_data)
    print(f"Mean of last {len(last_n_data)} losses is {avg_last_n_data}")
    print(f"Target loss is {target_loss}")
    assert avg_last_n_data < target_loss
    print('Final loss test passed.')


def main(argv: Sequence[str]) -> None:

  _, test_scenario, *test_vars = argv

  if test_scenario == 'final_loss':
    test_final_loss(*test_vars)
  else:
     raise ValueError(f"Unrecognized test_scenario {test_scenario}")


if __name__ == "__main__":
  app.run(main)
