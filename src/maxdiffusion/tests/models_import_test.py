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

"""Smoke test for MaxDiffusion lazy module imports."""

import unittest


class ModelsImportTest(unittest.TestCase):
  """Smoke tests verifying _LazyModule import resolution for models and utilities."""

  def test_import_flax_models(self):
    from maxdiffusion.models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
    self.assertIsNotNone(FlaxAutoencoderKL)
    self.assertIsNotNone(FlaxUNet2DConditionModel)

  def test_import_checkpointer(self):
    from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import BaseStableDiffusionCheckpointer
    self.assertIsNotNone(BaseStableDiffusionCheckpointer)

  def test_import_root_utilities(self):
    from maxdiffusion import max_logging, max_utils, pyconfig, maxdiffusion_utils
    self.assertIsNotNone(max_logging)
    self.assertIsNotNone(max_utils)
    self.assertIsNotNone(pyconfig)
    self.assertIsNotNone(maxdiffusion_utils)


if __name__ == "__main__":
  unittest.main()
