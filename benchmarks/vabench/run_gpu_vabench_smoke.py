#!/usr/bin/env python3
"""GPU VABench Smoke-5 Preparation & Evaluation Script for jalwaniya-a100-8-80gb."""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VABENCH_DIR = Path("/home/jalwaniya_google_com/VABench")
CACHE_DIR = Path("/home/jalwaniya_google_com/vabench_cache")
WORK_DIR = Path("/home/jalwaniya_google_com/vabench_smoke_eval")
VENV_PYTHON = Path("/home/jalwaniya_google_com/vabench_venv/bin/python")
GCS_BASE = "gs://jalwaniya-vbench-videos/vabench"

CASES = {
    "dense": {
        "label": "Case 1 (Dense)",
        "run_name": "ltx2-vabench-dense-smoke5",
        "gpus": "0,1",
    },
    "svg": {
        "label": "Case 2 (SVG)",
        "run_name": "ltx2-vabench-svg-smoke5",
        "gpus": "2,3",
    },
}


def download_gcs_prefix(gcs_prefix: str, local_dir: Path):
  """Downloads files from GCS using gcloud storage or gsutil."""
  local_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run(
      ["gcloud", "storage", "cp", f"{gcs_prefix}/*", str(local_dir) + "/"],
      check=True,
  )


def prepare_case_data(case: dict) -> Path:
  run_name = case["run_name"]
  case_dir = WORK_DIR / run_name
  raw_dir = case_dir / "raw_videos"
  data_dir = case_dir / "data_dir"
  video_sub = data_dir / "video" / "smoke5"
  audio_sub = data_dir / "audio" / "smoke5"
  json_sub = data_dir / "json"

  for d in [raw_dir, video_sub, audio_sub, json_sub]:
    d.mkdir(parents=True, exist_ok=True)

  print(f"[{run_name}] Downloading videos and metadata from {GCS_BASE}/{run_name}...")
  subprocess.run(
      ["gcloud", "storage", "cp", f"{GCS_BASE}/{run_name}/vabench_smoke5.json", str(case_dir / "vabench_smoke5.json")],
      check=True,
  )
  subprocess.run(
      ["gcloud", "storage", "cp", f"{GCS_BASE}/{run_name}/videos/*.mp4", str(raw_dir) + "/"],
      check=True,
  )

  prompts_meta = json.loads((case_dir / "vabench_smoke5.json").read_text(encoding="utf-8"))
  (json_sub / "smoke5.json").write_text(json.dumps(prompts_meta, indent=2, ensure_ascii=False), encoding="utf-8")

  for i, item in enumerate(prompts_meta):
    idx_name = item["vabench_idx"]
    candidates = sorted(raw_dir.glob(f"*_{i}.mp4"))
    if not candidates:
      raise FileNotFoundError(f"[{run_name}] Missing video for prompt index {i} in {raw_dir}")
    src_mp4 = candidates[0]
    dst_mp4 = video_sub / f"{idx_name}.mp4"
    dst_wav = audio_sub / f"{idx_name}.wav"

    shutil.copy2(src_mp4, dst_mp4)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(dst_mp4),
            "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            str(dst_wav),
        ],
        check=True,
    )
    print(f"[{run_name}] Prepared [{i}] '{idx_name}.mp4' + '{idx_name}.wav' ({dst_wav.stat().st_size} bytes)")

  return case_dir


def run_eval_case(case: dict):
  run_name = case["run_name"]
  case_dir = prepare_case_data(case)
  data_dir = case_dir / "data_dir"
  out_dir = case_dir / "vabench_results"
  out_dir.mkdir(parents=True, exist_ok=True)
  log_path = case_dir / "evaluate.log"

  env = os.environ.copy()
  env["PATH"] = f"{VENV_PYTHON.parent}:" + env.get("PATH", "")
  env["CUDA_VISIBLE_DEVICES"] = case["gpus"]
  env["VABENCH_CACHE_DIR"] = str(CACHE_DIR)
  env["PYTHONPATH"] = (
      f"{VABENCH_DIR}:{VABENCH_DIR}/third_party:{VABENCH_DIR}/third_party/ViCLIP:"
      f"{VABENCH_DIR}/third_party/syncnet_eval:" + env.get("PYTHONPATH", "")
  )

  cmd = [
      str(VENV_PYTHON),
      str(VABENCH_DIR / "evaluate.py"),
      "--data_dir", str(data_dir),
      "--output_dir", str(out_dir),
  ]
  print(f"[{run_name}] Running evaluate.py on GPUs {case['gpus']} (log: {log_path})...")
  with open(log_path, "w", encoding="utf-8") as log_f:
    ret = subprocess.call(cmd, cwd=str(VABENCH_DIR), env=env, stdout=log_f, stderr=subprocess.STDOUT)
  print(f"[{run_name}] evaluate.py exited with code {ret}")
  if ret != 0:
    print(f"--- Tail of {run_name} evaluate.log ---")
    print(log_path.read_text(encoding="utf-8")[-4000:])
    sys.exit(ret)

  subprocess.run(
      ["gcloud", "storage", "cp", "-r", str(out_dir / "smoke5") + "/*", f"{GCS_BASE}/{run_name}/vabench_results/"],
      check=False,
  )


def collate():
  summary = {}
  for key, c in CASES.items():
    run_name = c["run_name"]
    out_dir = WORK_DIR / run_name / "vabench_results" / "smoke5"
    if not out_dir.exists():
      continue
    eval_jsons = sorted(out_dir.glob("*_eval_results.json"))
    if not eval_jsons:
      continue
    res = json.loads(eval_jsons[-1].read_text(encoding="utf-8"))
    dim_means = {}
    for dim, dim_obj in res.items():
      recs = dim_obj.get("video_results", [])
      vals = [
          float(r["video_results"])
          for r in recs
          if isinstance(r, dict) and isinstance(r.get("video_results"), (int, float))
      ]
      dim_means[dim] = sum(vals) / len(vals) if vals else None
    summary[run_name] = dim_means

  (WORK_DIR / "smoke5_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print("=== SMOKE-5 VABENCH SUMMARY ===")
  print(json.dumps(summary, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=["dense", "svg", "all", "collate"], default="all")
  args = parser.parse_args()

  mapping_link = VABENCH_DIR / "mapping_text"
  if not mapping_link.exists():
    mapping_link.symlink_to(VABENCH_DIR / "mapping")

  if args.case == "dense":
    run_eval_case(CASES["dense"])
  elif args.case == "svg":
    run_eval_case(CASES["svg"])
  elif args.case == "all":
    run_eval_case(CASES["dense"])
    run_eval_case(CASES["svg"])
  collate()


if __name__ == "__main__":
  main()
