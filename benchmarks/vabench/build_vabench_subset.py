#!/usr/bin/env python3
"""Deterministic VABench 10% Stratified + k-Medoids Subsampling (778 -> 78 prompts) & Smoke-5 Suite."""

import base64
import csv
import io
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

STEPS_DIR = Path("/usr/local/google/home/jalwaniya/.gemini/jetski/brain/80790fd1-eb8c-454a-956d-79a8fba2af9a/.system_generated/steps")
OUT_DIR = Path("/google/src/cloud/jalwaniya/jetski_v6e/google3/experimental/users/jalwaniya/jetski_managed/ravimax/benchmarks/vabench")


def load_step_json(step_num: int) -> dict:
  p = STEPS_DIR / str(step_num) / "content.md"
  text = p.read_text(encoding="utf-8").split("---\n", 1)[1].strip()
  return json.loads(text)


def tokenize(text: str) -> Counter:
  words = [
      w.strip(".,!?;:\"'()[]{}-").lower()
      for w in text.split()
  ]
  words = [w for w in words if len(w) > 2]
  tokens = list(words) + [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
  return Counter(tokens)


def cosine_sim(c1: Counter, c2: Counter) -> float:
  common = set(c1.keys()) & set(c2.keys())
  if not common:
    return 0.0
  dot = sum(c1[k] * c2[k] for k in common)
  n1 = math.sqrt(sum(v * v for v in c1.values()))
  n2 = math.sqrt(sum(v * v for v in c2.values()))
  if n1 == 0 or n2 == 0:
    return 0.0
  return dot / (n1 * n2)


def select_k_medoids(items: list[dict], k: int, seed: int = 42) -> list[dict]:
  """Selects k diverse items using farthest-point / k-medoids refinement."""
  if k >= len(items):
    return list(items)
  vecs = [tokenize(it.get("prompt_en", "") + " " + it.get("prompt_audio", "")) for it in items]
  n = len(items)
  dist = [[1.0 - cosine_sim(vecs[i], vecs[j]) for j in range(n)] for i in range(n)]

  # Start with the medoid closest to the bundle center
  avg_dist = [sum(dist[i]) / n for i in range(n)]
  first = min(range(n), key=lambda i: avg_dist[i])
  selected = [first]

  while len(selected) < k:
    # Pick candidate maximizing minimum distance to already-selected items
    best_idx = max(
        (i for i in range(n) if i not in selected),
        key=lambda i: min(dist[i][s] for s in selected),
    )
    selected.append(best_idx)

  selected.sort()
  return [items[i] for i in selected]


def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  # 1. Load canonical final_idx_to_prompt.csv
  csv_blob = load_step_json(63)
  csv_text = base64.b64decode(csv_blob["content"]).decode("utf-8")
  (OUT_DIR / "final_idx_to_prompt.csv").write_text(csv_text, encoding="utf-8")

  mapping = {}
  for row in csv.DictReader(io.StringIO(csv_text), delimiter="\t"):
    p = json.loads('"' + row["prompt"] + '"')
    mapping[p] = row["idx"]

  # 2. Load all 24 JSON files
  tree = load_step_json(35)["tree"]
  sha_to_path = {
      item["sha"]: item["path"]
      for item in tree
      if item["path"].startswith("json_files/json_text/")
  }
  step_nums = [64] + list(range(105, 113)) + list(range(114, 122)) + list(range(123, 130))

  bundle_data = {}
  for s in step_nums:
    blob = load_step_json(s)
    bundle_name = os.path.basename(sha_to_path[blob["sha"]]).replace(".json", "")
    raw_items = json.loads(base64.b64decode(blob["content"]).decode("utf-8"))
    matched_items = []
    for it in raw_items:
      if it["prompt_en"] in mapping:
        it_copy = dict(it)
        it_copy["vabench_idx"] = mapping[it["prompt_en"]]
        it_copy["vabench_bundle"] = bundle_name
        matched_items.append(it_copy)
    bundle_data[bundle_name] = matched_items

  bundles_sorted = sorted(bundle_data.keys())
  counts = [len(bundle_data[b]) for b in bundles_sorted]
  target = 78
  quotas = [c * target / 778.0 for c in counts]
  floors = [max(1, int(q)) for q in quotas]
  diff = target - sum(floors)
  remainders = sorted(
      range(len(bundles_sorted)),
      key=lambda i: quotas[i] - floors[i],
      reverse=(diff > 0),
  )
  for idx in remainders[:abs(diff)]:
    floors[idx] += 1 if diff > 0 else -1

  sampled_78 = []
  for b, k in zip(bundles_sorted, floors):
    chosen = select_k_medoids(bundle_data[b], k, seed=42)
    sampled_78.extend(chosen)

  assert len(sampled_78) == 78, f"Expected 78 prompts, got {len(sampled_78)}"

  # 3. Select 5 Smoke Test prompts across the 5 target domains:
  # Physical Sync, Bioacoustics, Human Speech/Lip-Sync, Music, Natural Environment
  smoke_bundles = [
      "easy_Immediate_Event_Synchronization",
      "easy_bioacoustics",
      "easy_human_language",
      "easy_music",
      "easy_natural_environment",
  ]
  smoke_5 = []
  for sb in smoke_bundles:
    # Pick the first sampled prompt from this bundle
    cand = [it for it in sampled_78 if it["vabench_bundle"] == sb][0]
    smoke_5.append(cand)

  # Write Smoke-5 files
  (OUT_DIR / "prompts_smoke5.txt").write_text(
      "\n".join(it["prompt_en"].strip() for it in smoke_5) + "\n",
      encoding="utf-8",
  )
  (OUT_DIR / "vabench_smoke5.json").write_text(
      json.dumps(smoke_5, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )

  # Write Mini-VABench-78 files
  (OUT_DIR / "prompts_78.txt").write_text(
      "\n".join(it["prompt_en"].strip() for it in sampled_78) + "\n",
      encoding="utf-8",
  )
  (OUT_DIR / "vabench_sub78.json").write_text(
      json.dumps(sampled_78, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )

  print(f"Successfully generated Smoke-5 ({len(smoke_5)} prompts) and Mini-VABench-78 ({len(sampled_78)} prompts) in {OUT_DIR}")
  for i, it in enumerate(smoke_5):
    print(f"  [Smoke {i}] idx={it['vabench_idx']} ({it['vabench_bundle']}): {it['prompt_en'][:90]}...")


if __name__ == "__main__":
  main()
