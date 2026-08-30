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

import unittest
from unittest.mock import MagicMock
import jax
import jax.numpy as jnp

from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2


class WanI2VPrepareLatentsTest(unittest.TestCase):

  def _create_mock_pipeline(self, pipeline_cls):
    """Creates a mock pipeline instance with required VAE attributes."""
    pipeline = object.__new__(pipeline_cls)
    pipeline.vae = MagicMock(z_dim=16)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8

    def mock_prepare_latents_i2v_base(image, num_frames, dtype, last_image=None, trace=None):
      num_latent_frames = (num_frames - 1) // pipeline.vae_scale_factor_temporal + 1
      latent_height = 32 // pipeline.vae_scale_factor_spatial
      latent_width = 32 // pipeline.vae_scale_factor_spatial
      latent_condition = jnp.zeros(
          (image.shape[0], num_latent_frames, latent_height, latent_width, pipeline.vae.z_dim),
          dtype=dtype,
      )
      return latent_condition, None

    pipeline.prepare_latents_i2v_base = MagicMock(side_effect=mock_prepare_latents_i2v_base)
    return pipeline

  def test_single_image_repetition(self):
    """Verifies that a single conditioning image is repeated when batch_size > 1."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((1, 3, 32, 32))
        batch_size = 4
        latents, condition, _ = pipeline.prepare_latents(
            image=image,
            batch_size=batch_size,
            height=32,
            width=32,
            num_frames=5,
            dtype=jnp.float32,
            rng=rng,
        )
        self.assertEqual(latents.shape[0], batch_size)
        self.assertEqual(condition.shape[0], batch_size)
        call_image = pipeline.prepare_latents_i2v_base.call_args[0][0]
        self.assertEqual(call_image.shape[0], batch_size)

  def test_batched_image_repetition(self):
    """Verifies that multiple conditioning images are repeated correctly when divisible."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((2, 3, 32, 32))
        batch_size = 4
        latents, condition, _ = pipeline.prepare_latents(
            image=image,
            batch_size=batch_size,
            height=32,
            width=32,
            num_frames=5,
            dtype=jnp.float32,
            rng=rng,
        )
        self.assertEqual(latents.shape[0], batch_size)
        self.assertEqual(condition.shape[0], batch_size)
        call_image = pipeline.prepare_latents_i2v_base.call_args[0][0]
        self.assertEqual(call_image.shape[0], batch_size)

  def test_with_last_image(self):
    """Verifies that both start and last images are repeated when provided."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((1, 3, 32, 32))
        last_image = jnp.zeros((1, 3, 32, 32))
        batch_size = 3
        latents, condition, _ = pipeline.prepare_latents(
            image=image,
            batch_size=batch_size,
            height=32,
            width=32,
            num_frames=5,
            dtype=jnp.float32,
            rng=rng,
            last_image=last_image,
        )
        self.assertEqual(latents.shape[0], batch_size)
        self.assertEqual(condition.shape[0], batch_size)
        call_image = pipeline.prepare_latents_i2v_base.call_args[0][0]
        call_last_image = pipeline.prepare_latents_i2v_base.call_args[0][3]
        self.assertEqual(call_image.shape[0], batch_size)
        self.assertEqual(call_last_image.shape[0], batch_size)

  def test_indivisible_image_batch_size_raises(self):
    """Verifies ValueError when batch_size is not divisible by image batch size."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((2, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "divisible by image batch size"):
          pipeline.prepare_latents(
              image=image,
              batch_size=3,
              height=32,
              width=32,
              num_frames=5,
              dtype=jnp.float32,
              rng=rng,
          )

  def test_indivisible_last_image_batch_size_raises(self):
    """Verifies ValueError when batch_size is not divisible by last_image batch size."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((1, 3, 32, 32))
        last_image = jnp.zeros((2, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "divisible by last_image batch size"):
          pipeline.prepare_latents(
              image=image,
              batch_size=3,
              height=32,
              width=32,
              num_frames=5,
              dtype=jnp.float32,
              rng=rng,
              last_image=last_image,
          )

  def test_mismatched_image_and_last_image_batch_sizes_raises(self):
    """Verifies ValueError when image and last_image have conflicting batch sizes > 1."""
    rng = jax.random.key(0)
    for pipeline_cls in (WanPipelineI2V_2_1, WanPipelineI2V_2_2):
      with self.subTest(pipeline=pipeline_cls.__name__):
        pipeline = self._create_mock_pipeline(pipeline_cls)
        image = jnp.zeros((2, 3, 32, 32))
        last_image = jnp.zeros((3, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "must match when both are greater than 1"):
          pipeline.prepare_latents(
              image=image,
              batch_size=6,
              height=32,
              width=32,
              num_frames=5,
              dtype=jnp.float32,
              rng=rng,
              last_image=last_image,
          )


if __name__ == "__main__":
  unittest.main()
