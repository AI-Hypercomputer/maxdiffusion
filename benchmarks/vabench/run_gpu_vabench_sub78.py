#!/usr/bin/env python3
"""GPU 4-Worker Sharded VABench Evaluation Script for Mini-VABench-78 on jalwaniya-a100-8-80gb."""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VABENCH_DIR = Path("/home/jalwaniya_google_com/VABench")
CACHE_DIR = Path("/home/jalwaniya_google_com/vabench_cache")
WORK_DIR = Path("/home/jalwaniya_google_com/vabench_sub78_eval")
VENV_PYTHON = Path("/home/jalwaniya_google_com/vabench_venv/bin/python")
GCS_BASE = "gs://jalwaniya-vbench-videos/vabench"

CASES = {
    "dense": {
        "label": "Case 1 (Dense Baseline)",
        "run_name": "ltx2-vabench-dense-241f-sub78",
    },
    "svg": {
        "label": "Case 2 (SVG Ulysses)",
        "run_name": "ltx2-vabench-svg-241f-sub78",
    },
}

WORKER_GPUS = ["0,1", "2,3", "4,5", "6,7"]

BUNDLE_TO_CATEGORY = {
    "easy_Immediate_Event_Synchronization": "Synchronous Physical Sounds",
    "easy_human_nonverbal": "Human Sounds",
    "easy_human_language": "Human Sounds",
    "easy_Physical_Interaction_and_Acoustic_Phenomena": "Synchronous Physical Sounds",
    "easy_Indoor_Environment_and_Urban_Atmosphere": "Environmental Sounds",
    "easy_bioacoustics": "Animals",
    "easy_music": "Music",
    "easy_natural_environment": "Environmental Sounds",
    "easy_subjective_feeling": "Complex Scenes",
    "easy_strong_synchronization_and_rhythmic_movements": "Synchronous Physical Sounds",
    "hard_Virtual_Environment": "Virtual Worlds",
    "hard_complex_acoustic_scene": "Complex Scenes",
    "hard_sound_symbolism_and_metaphor": "Complex Scenes",
    "hard_Physics_knowledge_and_world_knowledge": "Synchronous Physical Sounds",
    "hard_human_language": "Human Sounds",
    "hard_invisible_sound_source": "Environmental Sounds",
    "hard_Exact_Time_Synchronization": "Synchronous Physical Sounds",
    "hard_nonverbal_human": "Human Sounds",
    "hard_Physical_Interaction_and_Acoustic_Phenomena": "Synchronous Physical Sounds",
    "hard_subjective_feeling": "Complex Scenes",
    "hard_strong_synchronization_and_rhythmic_movements": "Synchronous Physical Sounds",
    "hard_Indoor_Environment_and_Urban_Atmosphere": "Environmental Sounds",
    "hard_music": "Music",
    "hard_natural_environment": "Environmental Sounds",
}


def prepare_sharded_case_data(case: dict) -> Path:
  run_name = case["run_name"]
  case_dir = WORK_DIR / run_name
  raw_dir = case_dir / "raw_videos"
  raw_dir.mkdir(parents=True, exist_ok=True)

  print(f"[{run_name}] Downloading 78 videos and metadata from {GCS_BASE}/{run_name}...")
  subprocess.run(
      ["gcloud", "storage", "cp", f"{GCS_BASE}/{run_name}/vabench_sub78.json", str(case_dir / "vabench_sub78.json")],
      check=True,
  )
  subprocess.run(
      ["gcloud", "storage", "cp", f"{GCS_BASE}/{run_name}/videos/*.mp4", str(raw_dir) + "/"],
      check=True,
  )

  prompts_meta = json.loads((case_dir / "vabench_sub78.json").read_text(encoding="utf-8"))
  assert len(prompts_meta) == 78, f"Expected 78 prompts, got {len(prompts_meta)}"

  # Partition into 4 balanced shards (20, 20, 19, 19)
  num_workers = len(WORKER_GPUS)
  shards = [prompts_meta[w::num_workers] for w in range(num_workers)]

  for w, shard_items in enumerate(shards):
    shard_dir = case_dir / f"worker_{w}" / "data_dir"
    video_sub = shard_dir / "video" / f"shard_{w}"
    audio_sub = shard_dir / "audio" / f"shard_{w}"
    json_sub = shard_dir / "json"
    for d in [video_sub, audio_sub, json_sub]:
      d.mkdir(parents=True, exist_ok=True)

    (json_sub / f"shard_{w}.json").write_text(
        json.dumps(shard_items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

  # Demux audio and copy each video into its assigned worker shard
  for i, item in enumerate(prompts_meta):
    w = i % num_workers
    shard_dir = case_dir / f"worker_{w}" / "data_dir"
    video_sub = shard_dir / "video" / f"shard_{w}"
    audio_sub = shard_dir / "audio" / f"shard_{w}"

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
  print(f"[{run_name}] Prepared all 78 videos across 4 worker shards.")
  return case_dir


def run_eval_case(case: dict):
  run_name = case["run_name"]
  case_dir = prepare_sharded_case_data(case)

  procs = []
  for w, gpus in enumerate(WORKER_GPUS):
    w_dir = case_dir / f"worker_{w}"
    data_dir = w_dir / "data_dir"
    out_dir = w_dir / "vabench_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = w_dir / "evaluate.log"

    env = os.environ.copy()
    env["PATH"] = f"{VENV_PYTHON.parent}:" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = gpus
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
    print(f"[{run_name}] Launching Worker {w} on GPUs {gpus} (log: {log_path})...")
    log_f = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=str(VABENCH_DIR), env=env, stdout=log_f, stderr=subprocess.STDOUT)
    procs.append((w, gpus, p, log_f, log_path))

  for w, gpus, p, log_f, log_path in procs:
    ret = p.wait()
    log_f.close()
    print(f"[{run_name}] Worker {w} (GPUs {gpus}) exited with code {ret}")
    if ret != 0:
      print(f"--- Tail of Worker {w} evaluate.log ---")
      print(log_path.read_text(encoding="utf-8")[-4000:])
      sys.exit(ret)

  # Merge all 4 shards into a unified merged_eval_results.json
  merged_results = defaultdict(lambda: {"video_results": []})
  for w in range(len(WORKER_GPUS)):
    shard_out = case_dir / f"worker_{w}" / "vabench_results" / f"shard_{w}"
    eval_jsons = sorted(shard_out.glob("*_eval_results.json"))
    if not eval_jsons:
      raise FileNotFoundError(f"No *_eval_results.json found in {shard_out}")
    shard_res = json.loads(eval_jsons[-1].read_text(encoding="utf-8"))
    for dim, dim_obj in shard_res.items():
      merged_results[dim]["video_results"].extend(dim_obj.get("video_results", []))

  merged_dir = case_dir / "merged_results"
  merged_dir.mkdir(parents=True, exist_ok=True)
  merged_path = merged_dir / f"{run_name}_eval_results.json"
  merged_path.write_text(json.dumps(merged_results, indent=2), encoding="utf-8")
  print(f"[{run_name}] Saved merged 78-prompt results to {merged_path}")


def collate():
  summary = {}
  for key, c in CASES.items():
    run_name = c["run_name"]
    case_dir = WORK_DIR / run_name
    merged_path = case_dir / "merged_results" / f"{run_name}_eval_results.json"
    meta_path = case_dir / "vabench_sub78.json"
    if not merged_path.exists() or not meta_path.exists():
      continue

    res = json.loads(merged_path.read_text(encoding="utf-8"))
    prompts_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stem_to_meta = {it["vabench_idx"]: it for it in prompts_meta}

    dim_means = {}
    difficulty_scores = {"Easy": defaultdict(list), "Hard": defaultdict(list)}
    category_scores = defaultdict(lambda: defaultdict(list))

    for dim, dim_obj in res.items():
      recs = dim_obj.get("video_results", [])
      vals = []
      for r in recs:
        if not isinstance(r, dict):
          continue
        score = r.get("video_results")
        if not isinstance(score, (int, float)):
          continue
        vals.append(float(score))
        path_str = r.get("video_path") or r.get("audio_path") or ""
        stem = Path(path_str).stem
        meta = stem_to_meta.get(stem)
        if meta:
          bundle = meta.get("vabench_bundle", "")
          diff = "Easy" if bundle.startswith("easy_") else "Hard"
          cat = BUNDLE_TO_CATEGORY.get(bundle, "Other")
          difficulty_scores[diff][dim].append(float(score))
          category_scores[cat][dim].append(float(score))

      dim_means[dim] = sum(vals) / len(vals) if vals else None

    summary[run_name] = {
        "overall_15_dims": dim_means,
        "by_difficulty": {
            d: {dim: sum(v) / len(v) for dim, v in dims.items() if v}
            for d, dims in difficulty_scores.items()
        },
        "by_category": {
            cat: {dim: sum(v) / len(v) for dim, v in dims.items() if v}
            for cat, dims in category_scores.items()
        },
    }

  out_file = WORK_DIR / "sub78_comparison_summary.json"
  out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print("=== MINI-VABENCH-78 SUMMARY ===")
  print(json.dumps(summary, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=["dense", "svg", "all", "collate"], default="all")
  args = parser.parse_args()

  WORK_DIR.mkdir(parents=True, exist_ok=True)
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
