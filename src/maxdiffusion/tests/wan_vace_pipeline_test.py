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
import PIL.Image

# Enable JAX's CPU interpreter mode for Pallas custom kernels.
# Needed because the physical CPU backend does not support Pallas/Splash attention compilation.
os.environ["PALLAS_INTERPRET"] = "1"
import unittest
from unittest.mock import MagicMock, patch

import flax
import jax
import jax._src.config as jax_config

# Force the CPU Pallas interpreter globally. JAX 0.10.0+ uses this internal config manager
# and ignores the standard environment variable during eager pipeline execution.
jax_config.pallas_tpu_interpret_mode_context_manager.set_global(True)
import jax.numpy as jnp

from maxdiffusion import pyconfig

from maxdiffusion.pipelines.wan.wan_vace_pipeline_2_1 import VaceWanPipeline2_1
from maxdiffusion.schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class WanVacePipelineTest(unittest.TestCase):

  def setUp(self):
    # Initialize pyconfig with base_wan_1_3b.yml and overrides some parameters for speed
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_wan_1_3b.yml"),
            # For completeness, all configs and weights are mocked in this test
            "pretrained_model_name_or_path=Wan-AI/Wan2.1-VACE-1.3B-Diffusers",
            "num_inference_steps=2",  # Reduced steps for speed
            "height=240",  # Reduced resolution for speed (divisible by 16)
            "width=416",  # Reduced resolution for speed (divisible by 16)
            "num_frames=9",  # Reduced num_frames for speed
            "attention=flash",
            "scan_layers=False",  # Explicitly disable scan for VACE
            "jit_initializers=False",  # Disable JIT for faster setup & better CPU debugging
            "skip_jax_distributed_system=True",
        ],
        unittest=True,
    )
    self.config = pyconfig.config

  @patch("maxdiffusion.pipelines.wan.wan_vace_pipeline_2_1.WanVACEModel.load_config")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.AutoencoderKLWan.load_config")
  @patch("maxdiffusion.pipelines.wan.wan_vace_pipeline_2_1.load_wan_transformer")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.load_wan_vae")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_tokenizer")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_text_encoder")
  @patch("maxdiffusion.pipelines.wan.wan_pipeline.WanPipeline.load_scheduler")
  # pylint: disable=too-many-positional-arguments
  def test_pipeline_load_and_inference(
      self,
      mock_load_scheduler_fn,
      mock_load_text_encoder_fn,
      mock_load_tokenizer_fn,
      mock_load_wan_vae_fn,
      mock_load_wan_transformer_fn,
      mock_vae_load_config_fn,
      mock_transformer_load_config_fn,
  ):
    # Mock configs to represent a 1.3B model but with only 2 layers
    # Reference real config: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers/blob/main/transformer/config.json
    # pylint: disable=unused-argument
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
          "num_layers": 2,  # Overridden to 2 layers for speed
          "out_channels": 16,
          "patch_size": [1, 2, 2],
          "pos_embed_seq_len": None,
          "qk_norm": "rms_norm_across_heads",
          "rope_max_seq_len": 1024,
          "text_dim": 4096,
          "vace_in_channels": 96,
          "vace_layers": [0, 1],  # VACE conditioning on both layers
      }
      if return_unused_kwargs:
        return config_dict, kwargs
      return config_dict

    mock_transformer_load_config_fn.side_effect = mock_transformer_load_config

    # Full-size VAE config
    # Reference real config: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers/blob/main/vae/config.json
    # pylint: disable=unused-argument
    def mock_vae_load_config(pretrained_model_name_or_path, return_unused_kwargs=False, **kwargs):
      config_dict = {
          "attn_scales": [],
          "base_dim": 96,
          "dim_mult": [1, 2, 4, 4],
          "dropout": 0.0,
          "latents_mean": [
              -0.7571,
              -0.7089,
              -0.9113,
              0.1075,
              -0.1745,
              0.9653,
              -0.1517,
              1.5508,
              0.4134,
              -0.0715,
              0.5517,
              -0.3632,
              -0.1922,
              -0.9497,
              0.2503,
              -0.2921,
          ],
          "latents_std": [
              2.8184,
              1.4541,
              2.3275,
              2.6558,
              1.2196,
              1.7708,
              2.6052,
              2.0743,
              3.2687,
              2.1526,
              2.8652,
              1.5579,
              1.6382,
              1.1253,
              2.8251,
              1.916,
          ],
          "num_res_blocks": 2,
          "temperal_downsample": [False, True, True],
          "z_dim": 16,
      }
      if return_unused_kwargs:
        return config_dict, kwargs
      return config_dict

    mock_vae_load_config_fn.side_effect = mock_vae_load_config

    # Mock weight loaders to generate random weights in memory
    # pylint: disable=unused-argument
    def mock_load_wan_transformer(pretrained_model_name_or_path, eval_shapes, *args, **kwargs):
      cpu = jax.local_devices(backend="cpu")[0]
      flat_shapes = flax.traverse_util.flatten_dict(eval_shapes)
      flat_params = {}
      # Use a static seed to ensure deterministic random weights
      key = jax.random.key(42)
      for k, shape_struct in flat_shapes.items():
        dtype = shape_struct.dtype
        shape = shape_struct.shape
        key, subkey = jax.random.split(key)
        val = jax.random.normal(subkey, shape, dtype=dtype)
        flat_params[k] = jax.device_put(val, device=cpu)
      return flax.traverse_util.unflatten_dict(flat_params)

    mock_load_wan_transformer_fn.side_effect = mock_load_wan_transformer

    # pylint: disable=unused-argument
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

    # Mock scheduler to load from local config dictionary
    # Reference real config: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers/blob/main/scheduler/scheduler_config.json  # pylint: disable=line-too-long
    def mock_load_scheduler(config):
      scheduler = FlaxUniPCMultistepScheduler.from_config({
          "beta_end": 0.02,
          "beta_schedule": "linear",
          "beta_start": 0.0001,
          "disable_corrector": [],
          "dynamic_thresholding_ratio": 0.995,
          "final_sigmas_type": "zero",
          "flow_shift": config.flow_shift,
          "lower_order_final": True,
          "num_train_timesteps": 1000,
          "predict_x0": True,
          "prediction_type": "flow_prediction",
          "rescale_zero_terminal_snr": False,
          "sample_max_value": 1.0,
          "solver_order": 2,
          "solver_p": None,
          "solver_type": "bh2",
          "steps_offset": 0,
          "thresholding": False,
          "timestep_spacing": "linspace",
          "trained_betas": None,
          "use_beta_sigmas": False,
          "use_exponential_sigmas": False,
          "use_flow_sigmas": True,
          "use_karras_sigmas": False,
      })
      state = scheduler.create_state()
      return scheduler, state

    mock_load_scheduler_fn.side_effect = mock_load_scheduler

    # Mock tokenizer and text encoder to avoid Hugging Face downloads
    mock_load_tokenizer_fn.return_value = MagicMock()
    mock_load_text_encoder_fn.return_value = MagicMock()

    pipeline = VaceWanPipeline2_1.from_pretrained(self.config)

    # Prepare dummy inputs
    batch_size = 1

    height = self.config.height
    width = self.config.width
    num_frames = self.config.num_frames

    # Pre-computed dummy text embeddings matching T5 text dimension (4096)
    # Bypasses the actual text encoder
    prompt_embeds = jnp.zeros((batch_size, 512, 4096), dtype=self.config.weights_dtype)
    negative_prompt_embeds = jnp.zeros((batch_size, 512, 4096), dtype=self.config.weights_dtype)

    # Create dummy PIL images for conditioning inputs
    dummy_image_rgb = PIL.Image.new("RGB", (width, height), color="white")
    dummy_image_l = PIL.Image.new("L", (width, height), color="white")

    video_input = [dummy_image_rgb] * num_frames
    mask_input = [dummy_image_l] * num_frames
    ref_images_input = [dummy_image_rgb]

    video = pipeline(
        video=video_input,
        mask=mask_input,
        reference_images=ref_images_input,
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt=None,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=self.config.num_inference_steps,
    )

    self.assertEqual(len(video), batch_size)
    self.assertEqual(video[0].shape, (num_frames, height, width, 3))


if __name__ == "__main__":
  unittest.main()
