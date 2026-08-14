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

"""Exhaustive smoke test for MaxDiffusion lazy module imports and ML Diagnostics integration."""

import unittest


class ModelsImportTest(unittest.TestCase):
  """Exhaustive smoke tests verifying _LazyModule import resolution for models, pipelines, schedulers, utilities, and ML Diagnostics."""

  def test_import_flax_models(self):
    from maxdiffusion.models import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxControlNetModel
    self.assertIsNotNone(FlaxAutoencoderKL)
    self.assertIsNotNone(FlaxUNet2DConditionModel)
    self.assertIsNotNone(FlaxControlNetModel)

  def test_import_pipelines_and_checkpointers(self):
    from maxdiffusion.pipelines import FlaxStableDiffusionPipeline, FlaxStableDiffusionXLPipeline
    self.assertIsNotNone(FlaxStableDiffusionPipeline)
    self.assertIsNotNone(FlaxStableDiffusionXLPipeline)

    from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import BaseStableDiffusionCheckpointer
    self.assertIsNotNone(BaseStableDiffusionCheckpointer)

  def test_import_schedulers(self):
    from maxdiffusion.schedulers import FlaxDDIMScheduler, FlaxDDPMScheduler, FlaxDPMSolverMultistepScheduler
    self.assertIsNotNone(FlaxDDIMScheduler)
    self.assertIsNotNone(FlaxDDPMScheduler)
    self.assertIsNotNone(FlaxDPMSolverMultistepScheduler)

  def test_import_all_root_utilities(self):
    from maxdiffusion import (
        aot_cache,
        checkpointing,
        common_types,
        configuration_utils,
        max_logging,
        max_utils,
        maxdiffusion_google,
        maxdiffusion_google_hub,
        maxdiffusion_utils,
        multihost_dataloading,
        pyconfig,
        tpu_utils,
        train_utils,
    )
    self.assertIsNotNone(aot_cache)
    self.assertIsNotNone(checkpointing)
    self.assertIsNotNone(common_types)
    self.assertIsNotNone(configuration_utils)
    self.assertIsNotNone(max_logging)
    self.assertIsNotNone(max_utils)
    self.assertIsNotNone(maxdiffusion_google)
    self.assertIsNotNone(maxdiffusion_google_hub)
    self.assertIsNotNone(maxdiffusion_utils)
    self.assertIsNotNone(multihost_dataloading)
    self.assertIsNotNone(pyconfig)
    self.assertIsNotNone(tpu_utils)
    self.assertIsNotNone(train_utils)

  def test_mldiagnostics_import_and_usage(self):
    try:
      from google_cloud_mldiagnostics import machinelearning_run, xprof
      self.assertIsNotNone(machinelearning_run)
    except ImportError:
      pass

    from maxdiffusion.max_utils import ensure_machinelearning_job_runs, profiler_enabled
    self.assertIsNotNone(ensure_machinelearning_job_runs)
    self.assertIsNotNone(profiler_enabled)


if __name__ == "__main__":
  unittest.main()
