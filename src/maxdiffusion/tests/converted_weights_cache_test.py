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

"""Tests for the memoized torch->flax converted-weights cache."""

import os
import tempfile
import unittest

import ml_dtypes
import numpy as np

import json
from unittest import mock

from maxdiffusion.models.wan import wan_utils
from maxdiffusion.models.wan.wan_utils import save_converted_weights, try_load_converted_weights

_FP = "fp_test"


def _flat_tree():
  return {
      ("blocks", "attn1", "kernel"): np.arange(24, dtype=np.float32).reshape(2, 3, 4),
      ("proj_out", 0, "bias"): np.ones(5, dtype=np.float16),
      # bf16 exercises the uint bit-view path (npy mmap cannot resolve
      # ml_dtypes descriptors directly).
      ("blocks", "ffn", "kernel"): np.arange(12, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(3, 4),
  }


def _eval_shapes(flat):
  shapes = {}
  for key, value in flat.items():
    node = shapes
    for part in key[:-1]:
      node = node.setdefault(part, {})
    node[key[-1]] = value  # only keys/structure are validated
  return shapes


class ConvertedWeightsCacheTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.cache_dir = os.path.join(self._tmp.name, "cache")
    self.flat = _flat_tree()
    self.eval_shapes = _eval_shapes(self.flat)

  def tearDown(self):
    self._tmp.cleanup()

  def test_round_trip(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP)
    self.assertIsNotNone(loaded)
    np.testing.assert_array_equal(loaded["blocks"]["attn1"]["kernel"], self.flat[("blocks", "attn1", "kernel")])
    np.testing.assert_array_equal(loaded["proj_out"][0]["bias"], self.flat[("proj_out", 0, "bias")])
    bf16 = loaded["blocks"]["ffn"]["kernel"]
    self.assertEqual(bf16.dtype, np.dtype(ml_dtypes.bfloat16))
    np.testing.assert_array_equal(bf16.view(np.uint16), self.flat[("blocks", "ffn", "kernel")].view(np.uint16))

  def test_missing_cache_returns_none(self):
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP))

  def test_dtype_policy_change_invalidates(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, lambda key: np.dtype(np.float64), _FP)
    self.assertIsNone(loaded)

  def test_key_set_change_invalidates(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    bigger = dict(self.flat)
    bigger[("new_param", "kernel")] = np.zeros(2, dtype=np.float32)
    loaded = try_load_converted_weights(self.cache_dir, _eval_shapes(bigger), None, _FP)
    self.assertIsNone(loaded)

  def test_dtypes_preserved(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP)
    for key, value in self.flat.items():
      node = loaded
      for part in key:
        node = node[part]
      self.assertEqual(node.dtype, value.dtype)

  def test_resave_replaces_invalidated_cache(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, lambda key: np.dtype(np.float16), _FP))
    updated = {k: v.astype(np.float16) for k, v in self.flat.items()}
    save_converted_weights(self.cache_dir, updated, _FP)
    loaded = try_load_converted_weights(self.cache_dir, _eval_shapes(updated), lambda key: np.dtype(np.float16), _FP)
    self.assertIsNotNone(loaded)
    self.assertEqual(loaded["blocks"]["attn1"]["kernel"].dtype, np.float16)

  def test_source_fingerprint_mismatch_invalidates(self):
    save_converted_weights(self.cache_dir, self.flat, source_fingerprint="fp_v1")
    self.assertIsNotNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, source_fingerprint="fp_v1"))
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, source_fingerprint="fp_v2"))

  def _rewrite_meta(self, meta):
    path = os.path.join(self.cache_dir, "manifest.json")
    with open(path) as f:
      manifest = json.load(f)
    if meta is None:
      manifest.pop("__meta__")
    else:
      manifest["__meta__"] = meta
    with open(path, "w") as f:
      json.dump(manifest, f)

  def test_missing_or_null_fingerprint_is_rejected(self):
    """Fail-closed: a manifest that cannot prove its source is a miss."""
    save_converted_weights(self.cache_dir, self.flat, _FP)
    self._rewrite_meta({"format_version": wan_utils._CONVERTED_WEIGHTS_FORMAT_VERSION, "source_fingerprint": None})
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP))
    self._rewrite_meta({"format_version": wan_utils._CONVERTED_WEIGHTS_FORMAT_VERSION})
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP))
    self._rewrite_meta(None)  # pre-versioned manifest
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP))
    # A caller without a fingerprint cannot verify anything either.
    save_converted_weights(self.cache_dir, self.flat, _FP)
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, None))
    with self.assertRaises(ValueError):
      save_converted_weights(self.cache_dir, self.flat, None)

  def test_old_format_version_is_rejected(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    self._rewrite_meta({"format_version": 1, "source_fingerprint": _FP})
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP))

  def test_fingerprint_does_not_depend_on_mount_path(self):
    """Same repo/revision/index under two different HF_HOMEs -> same fingerprint; other revision -> different."""

    def make_index(root, revision, content=b'{"weight_map": {"a": "s1.safetensors"}}'):
      d = os.path.join(self._tmp.name, root, "hub", "models--org--repo", "snapshots", revision, "transformer")
      os.makedirs(d)
      path = os.path.join(d, "index.json")
      with open(path, "wb") as f:
        f.write(content)
      return path

    fp = wan_utils._compute_source_checkpoint_fingerprint
    a = fp(make_index("home_a", "rev1"), "org/repo")
    b = fp(make_index("snapshots/mnt/home_b", "rev1"), "org/repo")
    c = fp(make_index("home_c", "rev2"), "org/repo")
    d = fp(make_index("home_d", "rev1", b"{}"), "org/repo")
    self.assertEqual(a, b)
    self.assertNotEqual(a, c)
    self.assertNotEqual(a, d)
    self.assertNotEqual(a, fp(make_index("home_e", "rev1"), "org/other"))

  def _hf_snapshot(self, revision, subfolders, content=b'{"weight_map": {"a": "s1.safetensors"}}'):
    """Builds an HF-cache layout: snapshots/<rev>/<sub>/index.json symlinked to one shared blob."""
    repo = os.path.join(self._tmp.name, "hub", "models--org--repo")
    blob = os.path.join(repo, "blobs", f"blob-{revision}")
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    with open(blob, "wb") as f:
      f.write(content)
    paths = {}
    for sub in subfolders:
      d = os.path.join(repo, "snapshots", revision, sub)
      os.makedirs(d)
      paths[sub] = os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")
      os.symlink(blob, paths[sub])
    return paths

  def test_fingerprint_reads_revision_through_hf_symlinks_and_includes_subfolder(self):
    """HF snapshot entries are symlinks into blobs/: the revision must still be captured, and two
    subfolders with byte-identical index files (Wan 2.2 transformer / transformer_2) must differ."""
    fp = wan_utils._compute_source_checkpoint_fingerprint
    rev1 = self._hf_snapshot("rev1", ("transformer", "transformer_2"))
    rev2 = self._hf_snapshot("rev2", ("transformer",))
    self.assertEqual(wan_utils._snapshot_revision(rev1["transformer"]), "rev1")
    self.assertNotEqual(
        fp(rev1["transformer"], "org/repo", "transformer"), fp(rev1["transformer_2"], "org/repo", "transformer_2")
    )
    # Same index bytes, different snapshot revision -> different fingerprint.
    self.assertNotEqual(
        fp(rev1["transformer"], "org/repo", "transformer"), fp(rev2["transformer"], "org/repo", "transformer")
    )
    # The v2 formula lost both (this is the bug being fixed).
    legacy = wan_utils._legacy_v2_source_fingerprint
    self.assertEqual(legacy(rev1["transformer"], "org/repo"), legacy(rev1["transformer_2"], "org/repo"))
    self.assertEqual(legacy(rev1["transformer"], "org/repo"), legacy(rev2["transformer"], "org/repo"))

  def test_v2_manifest_with_matching_legacy_fingerprint_is_migrated_once(self):
    paths = self._hf_snapshot("rev1", ("transformer",))
    new_fp = wan_utils._compute_source_checkpoint_fingerprint(paths["transformer"], "org/repo", "transformer")
    old_fp = wan_utils._legacy_v2_source_fingerprint(paths["transformer"], "org/repo")
    save_converted_weights(self.cache_dir, self.flat, new_fp)
    self._rewrite_meta({"format_version": 2, "source_fingerprint": old_fp})
    # Without the legacy fingerprint a v2 manifest is a miss.
    self.assertIsNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, new_fp))
    loaded = try_load_converted_weights(self.cache_dir, self.eval_shapes, None, new_fp, legacy_source_fingerprint=old_fp)
    self.assertIsNotNone(loaded)
    with open(os.path.join(self.cache_dir, "manifest.json")) as f:
      meta = json.load(f)["__meta__"]
    self.assertEqual(meta["format_version"], wan_utils._CONVERTED_WEIGHTS_FORMAT_VERSION)
    self.assertEqual(meta["source_fingerprint"], new_fp)
    # Now current-format: loads with the new fingerprint alone.
    self.assertIsNotNone(try_load_converted_weights(self.cache_dir, self.eval_shapes, None, new_fp))

  def test_v2_manifest_with_other_fingerprint_is_rejected(self):
    save_converted_weights(self.cache_dir, self.flat, _FP)
    self._rewrite_meta({"format_version": 2, "source_fingerprint": "someone_else"})
    self.assertIsNone(
        try_load_converted_weights(self.cache_dir, self.eval_shapes, None, _FP, legacy_source_fingerprint="legacy")
    )
    with open(os.path.join(self.cache_dir, "manifest.json")) as f:
      self.assertEqual(json.load(f)["__meta__"]["format_version"], 2)  # untouched

  def test_save_is_skipped_when_disk_is_too_full(self):
    with mock.patch.object(wan_utils, "_free_bytes", return_value=1024):
      self.assertFalse(save_converted_weights(self.cache_dir, self.flat, _FP))
    self.assertFalse(os.path.exists(self.cache_dir))
    self.assertTrue(save_converted_weights(self.cache_dir, self.flat, _FP))

  def test_shard_download_refuses_to_fill_the_disk(self):
    import huggingface_hub

    index = {"metadata": {"total_size": 50 * 1024**3}, "weight_map": {}}
    with (
        mock.patch.object(huggingface_hub, "try_to_load_from_cache", return_value=None),
        mock.patch.object(wan_utils, "_free_bytes", return_value=10 * 1024**3),
    ):
      with self.assertRaisesRegex(OSError, "Refusing to download"):
        wan_utils._check_disk_for_shard_download("org/repo", "transformer", ["s1", "s2"], index)
    with mock.patch.object(huggingface_hub, "try_to_load_from_cache", return_value="/cached/path"):
      wan_utils._check_disk_for_shard_download("org/repo", "transformer", ["s1", "s2"], index)  # all cached: no-op

  def test_warm_start_checks_cache_before_the_network(self):
    """A valid converted cache for the locally cached index is used without a networked hf_hub_download."""
    snap = os.path.join(self._tmp.name, "hub", "models--org--repo", "snapshots", "rev1", "transformer")
    os.makedirs(snap)
    index_path = os.path.join(snap, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path, "w") as f:
      json.dump({"weight_map": {}}, f)
    fingerprint = wan_utils._compute_source_checkpoint_fingerprint(index_path, "org/repo", "transformer")
    save_converted_weights(self.cache_dir, self.flat, fingerprint)

    def fake_download(repo_id, subfolder=None, filename=None, local_files_only=False):
      del repo_id, subfolder, filename
      if not local_files_only:
        raise AssertionError("networked hf_hub_download called despite a valid converted cache")
      return index_path

    with mock.patch.object(wan_utils, "hf_hub_download", side_effect=fake_download):
      loaded = wan_utils.load_base_wan_transformer(
          "org/repo", self.eval_shapes, "cpu", subfolder="transformer", converted_cache_dir=self.cache_dir
      )
    np.testing.assert_array_equal(loaded["blocks"]["attn1"]["kernel"], self.flat[("blocks", "attn1", "kernel")])


if __name__ == "__main__":
  unittest.main()
