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
import jax.numpy as jnp
import gc

from maxdiffusion import pyconfig
from maxdiffusion.checkpointing.ltx2_checkpointer import LTX2Checkpointer

try:
  jax.distributed.initialize()
except Exception:  # pylint: disable=broad-exception-caught
  pass

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class LTX2SmokeTest(unittest.TestCase):
  """End-to-end smoke test for LTX2."""

  @classmethod
  def setUpClass(cls):
    gc.collect()
    jax.clear_caches()

    # Initialize config with the LTX2 video config file
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "ltx2_video.yml"),
            "num_inference_steps=2",  # Small number of steps for fast test
            "height=256",  # Small resolution
            "width=256",
            "num_frames=9",  # Small number of frames
            "max_sequence_length=256",  # Highly optimized sequence length to prevent VMEM OOM
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

    # Since this is just a smoke test to ensure the diffusion loop logic and inference code
    # work end-to-end, we don't actually need the real 4B parameter TorchAX text encoder running.
    # We patch the loader to prevent it from consuming the limited TPU HBM memory.
    cls.patcher1 = unittest.mock.patch(
        "maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.load_text_encoder", return_value=None
    )
    cls.patcher2 = unittest.mock.patch(
        "maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2Pipeline.load_tokenizer", return_value=None
    )
    cls.patcher1.start()
    cls.patcher2.start()

    checkpoint_loader = LTX2Checkpointer(config=cls.config)
    cls.pipeline, _, _ = checkpoint_loader.load_checkpoint(load_upsampler=False)

    cls.prompt = [cls.config.prompt]
    cls.negative_prompt = [cls.config.negative_prompt]

  def test_ltx2_inference(self):
    """Test that LTX2 pipeline can run inference and produce output."""
    generator = jax.random.key(self.config.seed)

    batch_size = len(self.prompt)
    # Create dummy embeddings to bypass the text encoder
    # LTX2AudioVideoGemmaTextEncoder expects a list of 49 tensors of shape (B, 1024, 3840)
    prompt_embeds = [jnp.zeros((batch_size, 1024, 3840), dtype=jnp.bfloat16) for _ in range(49)]
    prompt_attention_mask = jnp.ones((batch_size, 1024), dtype=jnp.bool_)
    negative_prompt_embeds = [jnp.zeros((batch_size, 1024, 3840), dtype=jnp.bfloat16) for _ in range(49)]
    negative_prompt_attention_mask = jnp.ones((batch_size, 1024), dtype=jnp.bool_)

    t0 = time.perf_counter()
    out = self.pipeline(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        height=self.config.height,
        width=self.config.width,
        num_frames=self.config.num_frames,
        num_inference_steps=self.config.num_inference_steps,
        guidance_scale=self.config.guidance_scale,
        generator=generator,
        dtype=jnp.bfloat16,
        max_sequence_length=self.config.max_sequence_length,
    )
    t1 = time.perf_counter()

    print(f"LTX2 Inference took: {t1 - t0:.2f}s")

    videos = out.frames if hasattr(out, "frames") else out[0]
    audios = out.audio if hasattr(out, "audio") else None

    self.assertIsNotNone(videos)
    # Check that we got frames
    self.assertGreater(len(videos), 0)

    if audios is not None:
      print(f"Audio produced with shape: {audios[0].shape}")
      self.assertGreater(len(audios), 0)

  @classmethod
  def tearDownClass(cls):
    del cls.pipeline
    cls.patcher1.stop()
    cls.patcher2.stop()
    gc.collect()


if __name__ == "__main__":
  unittest.main()
