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

import os
import time
import unittest
import jax

from maxdiffusion import pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1

try:
  jax.distributed.initialize()
except Exception:
  pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanSmokeTest(unittest.TestCase):
  """End-to-end smoke test for Wan."""

  @classmethod
  def setUpClass(cls):
    # Initialize config with the Wan video config file
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_14b.yml"),
            "num_inference_steps=2",  # Small number of steps for fast test
            "height=256",  # Small resolution (using what we used for cache tests)
            "width=256",
            "num_frames=9",  # Small number of frames
            "seed=0",
            "attention=flash",
            "ici_fsdp_parallelism=1",
            "ici_data_parallelism=1",
            "ici_context_parallelism=1",
            "ici_tensor_parallelism=-1",
        ],
        unittest=True,
    )
    cls.config = pyconfig.config
    checkpoint_loader = WanCheckpointer2_1(config=cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint()

    cls.prompt = [cls.config.prompt]
    cls.negative_prompt = [cls.config.negative_prompt]

  def test_wan_inference(self):
    """Test that Wan pipeline can run inference and produce output."""
    t0 = time.perf_counter()
    videos = self.pipeline(
        prompt=self.prompt,
        negative_prompt=self.negative_prompt,
        height=self.config.height,
        width=self.config.width,
        num_frames=self.config.num_frames,
        num_inference_steps=self.config.num_inference_steps,
        guidance_scale=self.config.guidance_scale,
    )
    t1 = time.perf_counter()

    print(f"Wan Inference took: {t1 - t0:.2f}s")

    self.assertIsNotNone(videos)
    # Check that we got frames
    self.assertGreater(len(videos), 0)

  @classmethod
  def tearDownClass(cls):
    del cls.pipeline
    import gc

    gc.collect()


if __name__ == "__main__":
  unittest.main()
