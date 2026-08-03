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

"""Tests for LTX2 AOT cache identity and invalidation."""

from types import SimpleNamespace
import unittest
from unittest import mock

from maxdiffusion import aot_cache
from maxdiffusion import generate_ltx2


class LTX2AotMetadataTest(unittest.TestCase):

  def setUp(self):
    self.config = SimpleNamespace(
        pretrained_model_name_or_path="test-model",
        use_base2_exp=False,
        use_experimental_scheduler=False,
    )
    self.pipeline = SimpleNamespace(
        transformer=SimpleNamespace(config={}),
        mesh=SimpleNamespace(shape={"data": 1}, axis_names=("data",)),
    )

  def _fingerprint(self, revision):
    metadata = generate_ltx2.ltx2_aot_metadata(self.config, self.pipeline, source_revision=revision)
    return metadata, aot_cache._metadata_fingerprint(metadata)

  def test_source_revision_changes_cache_fingerprint(self):
    metadata_a, fingerprint_a = self._fingerprint("revision-a")
    metadata_b, fingerprint_b = self._fingerprint("revision-b")

    self.assertEqual(metadata_a["source_revision"], "revision-a")
    self.assertEqual(metadata_b["source_revision"], "revision-b")
    self.assertNotEqual(fingerprint_a, fingerprint_b)

  def test_unavailable_revision_never_reuses_a_cache_identity(self):
    metadata_a, fingerprint_a = self._fingerprint(None)
    metadata_b, fingerprint_b = self._fingerprint(None)

    self.assertTrue(metadata_a["source_revision"].startswith("unversioned:"))
    self.assertTrue(metadata_b["source_revision"].startswith("unversioned:"))
    self.assertNotEqual(fingerprint_a, fingerprint_b)

  def test_build_revision_is_used_for_programmatic_run(self):
    self.config.aot_build_revision = "build-123"
    self.assertEqual(generate_ltx2._resolve_ltx2_aot_source_revision(self.config), "build-123")
    self.assertEqual(
        generate_ltx2._resolve_ltx2_aot_source_revision(self.config, commit_hash="commit-456"),
        "commit-456",
    )

  def test_attention_math_flags_change_cache_fingerprint(self):
    _, fingerprint_a = self._fingerprint("same-revision")
    self.config.use_base2_exp = True
    _, fingerprint_b = self._fingerprint("same-revision")
    self.config.use_experimental_scheduler = True
    _, fingerprint_c = self._fingerprint("same-revision")

    self.assertNotEqual(fingerprint_a, fingerprint_b)
    self.assertNotEqual(fingerprint_b, fingerprint_c)

  def test_codegen_and_remat_inputs_change_cache_fingerprint(self):
    _, fingerprint_a = self._fingerprint("same-revision")
    self.config.precision = "highest"
    _, fingerprint_b = self._fingerprint("same-revision")
    self.config.names_which_can_be_saved = ("attn",)
    _, fingerprint_c = self._fingerprint("same-revision")
    self.config.names_which_can_be_offloaded = ("mlp",)
    _, fingerprint_d = self._fingerprint("same-revision")

    self.assertNotEqual(fingerprint_a, fingerprint_b)
    self.assertNotEqual(fingerprint_b, fingerprint_c)
    self.assertNotEqual(fingerprint_c, fingerprint_d)

  def test_tile_search_rejects_a_prebuilt_pipeline(self):
    config = SimpleNamespace(get_keys=lambda: {"enable_tile_search": True})
    with self.assertRaisesRegex(ValueError, "cannot be used with a prebuilt pipeline"):
      generate_ltx2.run(config, pipeline=object())

  def test_tile_search_forwards_config_and_applies_winner(self):
    keys = {
        "enable_tile_search": True,
        "tile_search_mode": "full",
        "tile_search_iters": 3,
        "tile_search_out": "results",
        "tile_search_vmem_limit_bytes": 123456,
    }
    config = SimpleNamespace(
        get_keys=lambda: keys,
        mesh_axes=("data",),
        flash_block_sizes={"block_q": 64},
        attention="flash",
    )
    best = SimpleNamespace(bq=128, bkv=256, bkv_compute=256, mean_ms=1.5)
    benchmark = SimpleNamespace(label="test-benchmark")

    with (
        mock.patch.object(generate_ltx2.max_utils, "create_device_mesh", return_value=["device"]),
        mock.patch.object(generate_ltx2.jax.sharding, "Mesh", return_value="mesh"),
        mock.patch(
            "maxdiffusion.utils.ltx2_block_benchmark.LTX2BlockBenchmark.from_config",
            return_value=benchmark,
        ) as benchmark_from_config,
        mock.patch(
            "maxdiffusion.utils.tile_size_grid_search.grid_search",
            return_value=SimpleNamespace(best=best),
        ) as grid_search,
        mock.patch.object(
            generate_ltx2.max_utils,
            "flash_block_sizes_for_candidate",
            return_value={"block_q": 128},
        ) as apply_candidate,
        mock.patch.object(
            generate_ltx2.max_utils,
            "get_flash_block_sizes",
            return_value=SimpleNamespace(block_q=128),
        ),
    ):
      generate_ltx2.maybe_tune_block_sizes(config)

    grid_search.assert_called_once_with(
        benchmark,
        mode="full",
        iters=3,
        out_dir="results",
        log=generate_ltx2.max_logging.log,
    )
    benchmark_from_config.assert_called_once_with(config, "mesh", vmem_limit_bytes=123456)
    apply_candidate.assert_called_once_with(
        config.flash_block_sizes,
        "flash",
        128,
        256,
        256,
        vmem_limit_bytes=123456,
    )
    self.assertEqual(keys["flash_block_sizes"], {"block_q": 128})

  @mock.patch.object(generate_ltx2.max_logging, "log")
  @mock.patch.object(
      generate_ltx2.subprocess,
      "check_output",
      side_effect=[
          b"0123456789abcdef\n",
          b" M src/maxdiffusion/generate_ltx2.py\n",
          b"",
          b"0123456789abcdef\n",
          b" M src/maxdiffusion/generate_ltx2.py\n",
          b"",
      ],
  )
  def test_dirty_tree_does_not_reuse_cache_identity(self, _check_output, _log):
    revision_a = generate_ltx2.get_git_commit_hash()
    revision_b = generate_ltx2.get_git_commit_hash()
    self.assertEqual(revision_a, "dirty:0123456789abcdef")
    self.assertEqual(revision_b, revision_a)
    self.assertFalse(generate_ltx2._is_reusable_aot_revision(revision_a))


if __name__ == "__main__":
  unittest.main()
