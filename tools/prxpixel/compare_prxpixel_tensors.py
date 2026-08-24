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

"""Utility for recursively comparing two safetensors files or dictionary of tensors."""

import argparse
import sys
import numpy as np
import safetensors.numpy as st_np
from typing import Dict, Any


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
  """Computes error metrics between two numpy arrays."""
  a_f = a.astype(np.float64).flatten()
  b_f = b.astype(np.float64).flatten()

  diff = np.abs(a_f - b_f)
  max_abs = float(np.max(diff))
  mean_abs = float(np.mean(diff))
  rmse = float(np.sqrt(np.mean(diff**2)))

  norm_b = float(np.linalg.norm(b_f))
  rel_l2 = float(np.linalg.norm(diff) / (norm_b + 1e-12))

  norm_a = float(np.linalg.norm(a_f))
  if norm_a > 1e-12 and norm_b > 1e-12:
    cos_sim = float(np.dot(a_f, b_f) / (norm_a * norm_b))
  else:
    cos_sim = 1.0 if max_abs < 1e-6 else 0.0

  return {
      "max_abs": max_abs,
      "mean_abs": mean_abs,
      "rmse": rmse,
      "rel_l2": rel_l2,
      "cos_sim": cos_sim,
  }


def compare_tensor_dicts(
    actual: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray],
    prefix: str = "",
    verbose: bool = True,
) -> bool:
  """Compares two dictionaries of tensors and prints a formatted report table."""
  all_keys = sorted(list(set(actual.keys()).union(set(expected.keys()))))
  results = []
  missing_in_actual = []
  missing_in_expected = []

  for k in all_keys:
    if k not in actual:
      missing_in_actual.append(k)
      continue
    if k not in expected:
      missing_in_expected.append(k)
      continue

    arr_act = np.asarray(actual[k])
    arr_exp = np.asarray(expected[k])

    if arr_act.shape != arr_exp.shape:
      results.append({
          "key": k,
          "shape_act": str(arr_act.shape),
          "shape_exp": str(arr_exp.shape),
          "error": f"SHAPE MISMATCH: {arr_act.shape} vs {arr_exp.shape}",
          "max_abs": float("inf"),
          "mean_abs": float("inf"),
          "rmse": float("inf"),
          "rel_l2": float("inf"),
          "cos_sim": -1.0,
      })
      continue

    m = compute_metrics(arr_act, arr_exp)
    m["key"] = k
    m["shape_act"] = str(arr_act.shape)
    m["shape_exp"] = str(arr_exp.shape)
    m["dtype_act"] = str(arr_act.dtype)
    m["dtype_exp"] = str(arr_exp.dtype)
    results.append(m)

  if verbose:
    header = (
        f"{'Tensor Name':<45} | {'Shape':<18} | {'Max Abs':<11} | {'Mean Abs':<11} | {'RMSE':<11} | {'Rel L2':<11} | {'Cos Sim':<9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
      if "error" in r:
        print(f"{r['key']:<45} | {r['error']}")
      else:
        print(
            f"{r['key']:<45} | {r['shape_act']:<18} | {r['max_abs']:<11.4e} | {r['mean_abs']:<11.4e} | {r['rmse']:<11.4e} | {r['rel_l2']:<11.4e} | {r['cos_sim']:<9.6f}"
        )

    if missing_in_actual:
      print(f"\n⚠️ Missing in actual: {missing_in_actual}")
    if missing_in_expected:
      print(f"\n⚠️ Missing in expected: {missing_in_expected}")

    # Top largest discrepancies
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
      sorted_by_err = sorted(valid_results, key=lambda x: x["max_abs"], reverse=True)
      print("\n" + "-" * 80)
      print("🔍 TOP 5 LARGEST DISCREPANCIES:")
      for top in sorted_by_err[:5]:
        print(f"  • {top['key']}: MaxAbs={top['max_abs']:.4e}, RelL2={top['rel_l2']:.4e}, CosSim={top['cos_sim']:.6f}")
      print("-" * 80)

  has_errors = any("error" in r or r["max_abs"] > 1e-3 for r in results) or bool(missing_in_actual) or bool(missing_in_expected)
  return not has_errors


def main():
  parser = argparse.ArgumentParser(description="Compare two safetensors files.")
  parser.add_argument("actual_file", type=str, help="Path to actual safetensors file")
  parser.add_argument("expected_file", type=str, help="Path to expected/golden safetensors file")
  args = parser.parse_args()

  actual = st_np.load_file(args.actual_file)
  expected = st_np.load_file(args.expected_file)
  success = compare_tensor_dicts(actual, expected)
  sys.exit(0 if success else 1)


if __name__ == "__main__":
  main()
