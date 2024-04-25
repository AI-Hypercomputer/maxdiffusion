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
"""Find and output the convergence timestamp and step

Usage:

python src/maxdiffusion/report_end.py \
    --metrics-path=/path/to/eval_metrics.csv \
    --mllog-path=/path/to/mllog.txt
"""


import pandas as pd
import argparse
import tensorflow as tf
import json
import mllog_utils


TARGET_FID = 90.0
TARGET_CLIP = 0.15
MLLOG_PREFIX=":::MLLOG"


def parse_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--target-fid", type=float, default=TARGET_FID)
  parser.add_argument("--target-clip", type=float, default=TARGET_CLIP)
  parser.add_argument("--metrics-path", type=str, required=True)
  parser.add_argument("--mllog-path", type=str, required=True)
  args = parser.parse_args()
  return args

def create_step_num_to_timestamp(mllog_path: str):
  step_num_to_timestamp = {}
  with tf.io.gfile.GFile(mllog_path, 'r') as f:
    for line in f:
      if line.startswith(":::MLLOG") and "checkpoint" in line:
        log_dict = json.loads(line[len(":::MLLOG"):])
        step_num, timestamp = log_dict["metadata"]["step_num"], log_dict['time_ms']
        step_num_to_timestamp[step_num] = timestamp

  return step_num_to_timestamp

def main():
  args = parse_command_line_args()
  with tf.io.gfile.GFile(args.metrics_path, 'r') as f:
    df = pd.read_csv(f)

  df = df.sort_values(by=['step_num'])
  step_num_to_timestamp = create_step_num_to_timestamp(args.mllog_path)

  success_timestamp = None
  success_step_num = None
  success_samples_count = None
  for row in df.itertuples():
    if row.fid <= args.target_fid and row.clip >= args.target_clip:
      success_step_num = row.step_num
      success_timestamp = step_num_to_timestamp[success_step_num]
      success_samples_count = row.samples_count
      mllog_utils.timestamp_fid(row.fid, success_timestamp, row.step_num, row.samples_count)
      mllog_utils.timestamp_clip(row.clip, success_timestamp, row.step_num, row.samples_count)
      print(f"Found checkpoint matching targets at step_num={row.step_num} samples_count={row.samples_count}: {args.target_fid=} {args.target_clip=}")
      break
  else:
    print(f"Could not find checkpoint matching targets: {args.target_fid=} {args.target_clip=}")

  if success_timestamp is not None:
    mllog_utils.timestamp_run_stop_success(success_timestamp, success_step_num, success_samples_count)
  else:
    mllog_utils.timestamp_run_stop_abort()


if __name__ == "__main__":
  main()