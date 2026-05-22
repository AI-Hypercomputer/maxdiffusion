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
import unittest
from unittest.mock import MagicMock, patch

import flax
import jax

import jax.numpy as jnp
import numpy as np

from maxdiffusion import pyconfig
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
from maxdiffusion.schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanKvCacheTest(unittest.TestCase):

  def setUp(self):
    # Initialize pyconfig with base_wan_1_3b.yml and overrides some parameters for speed
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "..", "configs", "base_wan_1_3b.yml"),
            "pretrained_model_name_or_path=Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "num_inference_steps=2",  # Reduced steps for speed
            "height=240",  # Reduced resolution for speed (divisible by 16)
            "width=416",  # Reduced resolution for speed (divisible by 16)
            "num_frames=9",  # Reduced num_frames for speed
            "attention=flash",
            "scan_layers=False",
            "jit_initializers=False",
            "skip_jax_distributed_system=True",
        ],
        unittest=True,
    )
    self.config = pyconfig.config

  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanModel.load_config")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.AutoencoderKLWan.load_config")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.load_wan_transformer")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.load_wan_vae")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_tokenizer")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_text_encoder")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_scheduler")
  def test_wan_2_1_kv_cache(
      self,
      mock_load_scheduler_fn,
      mock_load_text_encoder_fn,
      mock_load_tokenizer_fn,
      mock_load_wan_vae_fn,
      mock_load_wan_transformer_fn,
      mock_vae_load_config_fn,
      mock_transformer_load_config_fn,
  ):
    # Mock transformer config
    def mock_transformer_load_config(pretrained_model_name_or_path, return_unused_kwargs=False, **kwargs):
      config_dict = {
          "added_kv_proj_dim": None,
          "attention_head_dim": 128,
          "cross_attn_norm": True,
          "eps": 1e-06,
          "ffn_dim": 8960,
          "freq_dim": 256,
          "image_dim": None,
          "in_channels": 16,
          "num_attention_heads": 12,
          "num_layers": 2,
          "out_channels": 16,
          "patch_size": [1, 2, 2],
          "pos_embed_seq_len": None,
          "qk_norm": "rms_norm_across_heads",
          "rope_max_seq_len": 1024,
          "text_dim": 4096,
      }
      if return_unused_kwargs:
        return config_dict, kwargs
      return config_dict

    mock_transformer_load_config_fn.side_effect = mock_transformer_load_config

    # Mock VAE config
    def mock_vae_load_config(pretrained_model_name_or_path, return_unused_kwargs=False, **kwargs):
      config_dict = {
          "attn_scales": [],
          "base_dim": 96,
          "dim_mult": [1, 2, 4, 4],
          "dropout": 0.0,
          "latents_mean": [0.0] * 16,
          "latents_std": [1.0] * 16,
          "num_res_blocks": 2,
          "temperal_downsample": [False, True, True],
          "z_dim": 16,
      }
      if return_unused_kwargs:
        return config_dict, kwargs
      return config_dict

    mock_vae_load_config_fn.side_effect = mock_vae_load_config

    # Mock weight loaders
    def mock_load_wan_transformer(pretrained_model_name_or_path, eval_shapes, *args, **kwargs):
      cpu = jax.local_devices(backend="cpu")[0]
      flat_shapes = flax.traverse_util.flatten_dict(eval_shapes)
      flat_params = {}
      key = jax.random.key(42)
      for k, shape_struct in flat_shapes.items():
        dtype = shape_struct.dtype
        shape = shape_struct.shape
        key, subkey = jax.random.split(key)
        val = jax.random.normal(subkey, shape, dtype=dtype)
        flat_params[k] = jax.device_put(val, device=cpu)
      return flax.traverse_util.unflatten_dict(flat_params)

    mock_load_wan_transformer_fn.side_effect = mock_load_wan_transformer

    def mock_load_wan_vae(pretrained_model_name_or_path, eval_shapes, *args, **kwargs):
      cpu = jax.local_devices(backend="cpu")[0]
      flat_shapes = flax.traverse_util.flatten_dict(eval_shapes)
      flat_params = {}
      key = jax.random.key(42)
      for k, shape_struct in flat_shapes.items():
        dtype = shape_struct.dtype
        shape = shape_struct.shape
        key, subkey = jax.random.split(key)
        val = jax.random.normal(subkey, shape, dtype=dtype)
        flat_params[k] = jax.device_put(val, device=cpu)
      return flax.traverse_util.unflatten_dict(flat_params)

    mock_load_wan_vae_fn.side_effect = mock_load_wan_vae

    # Mock scheduler
    def mock_load_scheduler(config):
      scheduler = FlaxUniPCMultistepScheduler.from_config({
          "beta_end": 0.02,
          "beta_schedule": "linear",
          "beta_start": 0.0001,
          "flow_shift": config.flow_shift,
          "num_train_timesteps": 1000,
          "prediction_type": "flow_prediction",
          "timestep_spacing": "linspace",
          "use_flow_sigmas": True,
      })
      state = scheduler.create_state()
      return scheduler, state

    mock_load_scheduler_fn.side_effect = mock_load_scheduler

    mock_load_tokenizer_fn.return_value = MagicMock()
    mock_load_text_encoder_fn.return_value = MagicMock()

    pipeline = WanPipeline2_1.from_pretrained(self.config)

    batch_size = 1
    height = self.config.height
    width = self.config.width
    num_frames = self.config.num_frames

    prompt_embeds = jnp.zeros((batch_size, 512, 4096), dtype=self.config.weights_dtype)
    negative_prompt_embeds = jnp.zeros((batch_size, 512, 4096), dtype=self.config.weights_dtype)

    # Run without cache
    video_no_cache, _ = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt=None,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=self.config.num_inference_steps,
        use_kv_cache=False,
    )

    # Run with cache
    video_with_cache, _ = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt=None,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=self.config.num_inference_steps,
        use_kv_cache=True,
    )

    self.assertEqual(len(video_no_cache), batch_size)
    self.assertEqual(video_no_cache[0].shape, (num_frames, height, width, 3))

    self.assertEqual(len(video_with_cache), batch_size)
    self.assertEqual(video_with_cache[0].shape, (num_frames, height, width, 3))

    # Compare outputs
    np.testing.assert_allclose(video_no_cache, video_with_cache, rtol=1e-1, atol=0.7)


if __name__ == "__main__":
  unittest.main()
