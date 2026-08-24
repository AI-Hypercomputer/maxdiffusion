"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

End-to-end pipeline test with synthetic weights.

Besides checking that `generate` produces an image, this asserts the property
that makes the reported latency meaningful: a short warmup must compile the same
executable the long run uses, so a longer run must trigger no recompilation.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from maxdiffusion import max_logging
from maxdiffusion.models.ideogram import Ideogram4Transformer, Ideogram4Config, AutoEncoder, AutoEncoderParams
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline, _denoise_step


class _FakeTokenizer:
  """Deterministic stand-in so the test needs no checkpoint download."""

  def __init__(self, num_tokens=300):
    self.num_tokens = num_tokens

  def __call__(self, prompt, return_tensors="np", add_special_tokens=True):
    return {"input_ids": np.arange(self.num_tokens, dtype=np.int32)[None, :]}


def _build_pipeline(config):
  rngs = nnx.Rngs(0)
  transformer = Ideogram4Transformer(rngs, config)
  ae_params = AutoEncoderParams(
      resolution=256, in_channels=3, ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, z_channels=32
  )
  autoencoder = AutoEncoder(rngs, ae_params)

  def mock_text_encoder(token_ids, _attention_mask, _pos_2d):
    batch_size, seq_len = token_ids.shape
    return jnp.zeros((batch_size, seq_len, config.llm_features_dim), dtype=jnp.float32)

  return IdeogramPipeline(
      conditional_transformer=transformer,
      unconditional_transformer=transformer,
      autoencoder=autoencoder,
      text_encoder=mock_text_encoder,
      tokenizer=_FakeTokenizer(),
  )


def test_end_to_end():
  max_logging.log("Initializing components...")
  config = Ideogram4Config(emb_dim=128, num_heads=2, in_channels=128, llm_features_dim=128, adanln_dim=128, num_layers=2)
  pipeline = _build_pipeline(config)

  max_logging.log("Warmup (2 steps)...")
  t0 = time.perf_counter()
  warm, _ = pipeline.generate(prompts=["a cute dog"], height=256, width=256, num_steps=2, guidance_scale=7.0, seed=42)
  jax.block_until_ready(warm)
  warmup_time = time.perf_counter() - t0

  # The denoise step is compiled per step, so a longer run must reuse the same
  # executable. If this count moves, the timed number below includes compile.
  cache_before = _denoise_step._cache_size()

  steps = 8
  t0 = time.perf_counter()
  images, trace = pipeline.generate(
      prompts=["a cute dog"], height=256, width=256, num_steps=steps, guidance_scale=7.0, seed=42
  )
  jax.block_until_ready(images)
  run_time = time.perf_counter() - t0

  assert _denoise_step._cache_size() == cache_before, (
      f"a {steps}-step run recompiled after a 2-step warmup "
      f"(jit cache {cache_before} -> {_denoise_step._cache_size()}); warmup is not valid"
  )

  max_logging.log(f"Generation complete! Output shape: {images.shape}")
  max_logging.log(f"warmup (2 steps, incl. compile): {warmup_time:.2f}s")
  max_logging.log(f"{steps} steps: {run_time:.2f}s  ({run_time / steps * 1e3:.1f} ms/step)")
  max_logging.log(
      f"  breakdown: conditioning {trace['conditioning']:.2f}s "
      f"(text encode {trace['text_encode']:.2f}s), "
      f"denoise {trace['denoise_total']:.2f}s ({trace['denoise_per_step'] * 1e3:.1f} ms/step steady), "
      f"vae decode {trace['vae_decode']:.2f}s"
  )

  image_np = (np.array(images[0]) * 255).astype(np.uint8)
  out_path = "ideogram_end_to_end_test.png"
  Image.fromarray(image_np).save(out_path)
  max_logging.log(f"Saved test image to {out_path}")


def test_text_token_bucketing_avoids_recompiles():
  """Prompts of different lengths inside one bucket must share an executable."""
  config = Ideogram4Config(emb_dim=128, num_heads=2, in_channels=128, llm_features_dim=128, adanln_dim=128, num_layers=2)
  pipeline = _build_pipeline(config)

  pipeline.tokenizer = _FakeTokenizer(300)
  jax.block_until_ready(pipeline.generate(prompts=["a"], height=256, width=256, num_steps=1)[0])
  cache_before = _denoise_step._cache_size()

  # 300 and 400 tokens both round up to the same 512-token bucket.
  pipeline.tokenizer = _FakeTokenizer(400)
  jax.block_until_ready(pipeline.generate(prompts=["a"], height=256, width=256, num_steps=1)[0])
  assert _denoise_step._cache_size() == cache_before, "same bucket should not recompile"
  max_logging.log("Bucketing holds: 300- and 400-token prompts share one executable.")


if __name__ == "__main__":
  test_end_to_end()
  test_text_token_bucketing_avoids_recompiles()
