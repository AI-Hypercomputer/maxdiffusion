#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small helpers for run_gpu_eval.sh."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import shutil


def extract_dimensions(args: argparse.Namespace) -> None:
  with open(args.json_file, encoding="utf-8") as f:
    data = json.load(f)

  dimensions = dict.fromkeys(
      dimension
      for item in data
      for dimension in item.get("dimension", [])
  )
  print(" ".join(dimensions))


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
  if not path.exists():
    return

  text = path.read_text(encoding="utf-8")
  for old, new in replacements:
    if old not in text:
      print(f"Warning: patch target {old!r} not found in {path}. Upstream may have changed.")
    text = text.replace(old, new)
  path.write_text(text, encoding="utf-8")


def patch_vbench(args: argparse.Namespace) -> None:
  root = Path(args.vbench_dir)
  patch_file(
      root / "setup.py",
      [("def check_torch_version():", "def check_torch_version():\n    return")],
  )
  patch_file(
      root / "vbench/distributed.py",
      [
          (
              "backend = 'gloo' if os.name == 'nt' else 'nccl'",
              "backend = 'gloo' if (os.name == 'nt' or not torch.cuda.is_available()) else 'nccl'",
          ),
          (
              "torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))",
              "if torch.cuda.is_available():\n        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))",
          ),
      ],
  )
  patch_file(
      root / "evaluate.py",
      [
          (
              'device = torch.device("cuda")',
              'assert torch.cuda.is_available(), "CUDA is not available, but is required for VBench evaluation."\n'
              '    device = torch.device(f"cuda:{int(os.environ.get(\'LOCAL_RANK\', \'0\'))}")',
          ),
      ],
  )
  patch_file(
      root / "vbench/__init__.py",
      [
          ("import os\n", "import os\nimport re\n"),
          (
              (
                  "            for prompt_dict in full_info_list:\n"
                  "                # if the prompt belongs to any dimension we want to evaluate\n"
                  '                if set(dimension_list) & set(prompt_dict["dimension"]): \n'
                  "                    prompt = prompt_dict['prompt_en']\n"
                  "                    prompt_dict['video_list'] = []\n"
                  "                    for i in range(5): # video index for the same prompt\n"
                  "                        intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'"
              ),
              (
                  "            for idx, prompt_dict in enumerate(full_info_list):\n"
                  "                # if the prompt belongs to any dimension we want to evaluate\n"
                  '                if set(dimension_list) & set(prompt_dict["dimension"]): \n'
                  "                    prompt = prompt_dict['prompt_en']\n"
                  "                    safe_prompt = re.sub(r'[^\\w\\s-]', '_', prompt).strip()[:120]\n"
                  "                    prompt_dict['video_list'] = []\n"
                  "                    for i in range(5): # video index for the same prompt\n"
                  "                        intended_video_name = f'{safe_prompt}_{idx}-{str(i)}{postfix}'\n"
                  "                        if intended_video_name not in video_names:\n"
                  "                            intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'"
              ),
          ),
      ],
  )


def prepare_videos(args: argparse.Namespace) -> None:
  with open(args.json_file, encoding="utf-8") as f:
    bench_data = json.load(f)

  downloaded = sorted(glob.glob(os.path.join(args.download_dir, "*.mp4")))
  print(f"Downloaded {len(downloaded)} videos from GCS.")

  prompt_video_map: dict[int, list[str]] = {}
  for video_path in downloaded:
    match = re.search(r"_(\d+)\.mp4$", os.path.basename(video_path))
    if match:
      prompt_video_map.setdefault(int(match.group(1)), []).append(video_path)

  prepared_videos: list[tuple[str, int, str]] = []
  for idx, item in enumerate(bench_data):
    prompt = item["prompt_en"]
    safe_prompt = re.sub(r"[^\w\s-]", "_", prompt).strip()[:120]
    candidates = prompt_video_map.get(idx, [])
    if len(candidates) != 1:
      raise ValueError(
          f"Expected exactly one video for prompt index {idx} "
          f"({prompt[:40]!r}), but found {len(candidates)}."
      )
    prepared_videos.append((safe_prompt, idx, candidates[0]))

  shutil.rmtree(args.vbench_dir, ignore_errors=True)
  os.makedirs(args.vbench_dir, exist_ok=True)
  for safe_prompt, idx, source_path in prepared_videos:
    target_path = os.path.join(args.vbench_dir, f"{safe_prompt}_{idx}-0.mp4")
    try:
      os.symlink(os.path.abspath(source_path), target_path)
    except OSError:
      shutil.copy2(source_path, target_path)

  print(f"Prepared {len(prepared_videos)} VBench video entries.")


def main() -> None:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(required=True)

  dimensions = subparsers.add_parser("dimensions")
  dimensions.add_argument("json_file")
  dimensions.set_defaults(func=extract_dimensions)

  patch = subparsers.add_parser("patch-vbench")
  patch.add_argument("vbench_dir")
  patch.set_defaults(func=patch_vbench)

  prepare = subparsers.add_parser("prepare-videos")
  prepare.add_argument("json_file")
  prepare.add_argument("download_dir")
  prepare.add_argument("vbench_dir")
  prepare.set_defaults(func=prepare_videos)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
