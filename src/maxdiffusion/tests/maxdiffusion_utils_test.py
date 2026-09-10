"""
Copyright 2024 Google LLC

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
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from jax.sharding import Mesh

from .. import pyconfig
from maxdiffusion.max_utils import (
    chunk_and_pad,
    create_device_mesh,
    get_flash_block_sizes,
    get_gcs_output_path,
    load_prompts,
)
from maxdiffusion import (FlaxStableDiffusionXLPipeline, FlaxDDIMScheduler, FlaxDDPMScheduler, maxdiffusion_utils)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class MaxDiffusionUtilsTest(unittest.TestCase):
  """Test maxdiffusion_utils.py functions"""

  def setUp(self):
    MaxDiffusionUtilsTest.dummy_data = {}

  def test_load_prompts_raises_when_no_prompt_source_exists(self):
    with self.assertRaisesRegex(ValueError, "No prompts found"):
      load_prompts("", "")
    with self.assertRaisesRegex(ValueError, "No prompts found"):
      load_prompts("   ", "")

  def test_load_prompts_fallback_to_default_prompt(self):
    default_prompt = "A test default prompt"
    self.assertEqual(load_prompts("", default_prompt=default_prompt), [default_prompt])
    self.assertEqual(load_prompts("   ", default_prompt=default_prompt), [default_prompt])

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
      f.write("   \n\n\t  \n")
      f_path = f.name
    try:
      self.assertEqual(load_prompts(f_path, default_prompt=default_prompt), [default_prompt])
    finally:
      os.unlink(f_path)

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
      f.write("# comment line 1\n  # comment line 2\n")
      f_path = f.name
    try:
      self.assertEqual(load_prompts(f_path, default_prompt=default_prompt), [default_prompt])
    finally:
      os.unlink(f_path)

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
      f.write("   \n\n")
      f_path = f.name
    try:
      with self.assertRaisesRegex(ValueError, "contains no valid non-empty prompts"):
        load_prompts(f_path, default_prompt="")
    finally:
      os.unlink(f_path)

  def test_load_prompts_from_file_with_comments_and_whitespace(self):
    content = (
        "# Benchmark Prompts\n"
        "\n"
        "  A cat playing piano in space  \n"
        "# Another comment\n"
        "   # Indented comment\n"
        "A robot walking through a neon city\n"
        "\n"
        "A peaceful sunrise over mountains\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
      f.write(content)
      f_path = f.name
    try:
      prompts = load_prompts(f_path, default_prompt="fallback")
      expected = [
          "A cat playing piano in space",
          "A robot walking through a neon city",
          "A peaceful sunrise over mountains",
      ]
      self.assertEqual(prompts, expected)
    finally:
      os.unlink(f_path)

  def test_load_prompts_file_not_found(self):
    with self.assertRaises(FileNotFoundError):
      load_prompts("non_existent_prompt_file_xyz.txt", default_prompt="fallback")

  def test_get_gcs_output_path(self):
    # output_dir starting with gs://
    config = SimpleNamespace(output_dir="gs://bucket/test", run_name="exp1")
    self.assertEqual(get_gcs_output_path(config), "gs://bucket/test/exp1")

    # output_dir starting with gs:// without run_name
    config = SimpleNamespace(output_dir="gs://bucket/test", run_name="")
    self.assertEqual(get_gcs_output_path(config), "gs://bucket/test")

    # output_dir starting with gs:// with run_name as None
    config = SimpleNamespace(output_dir="gs://bucket/test", run_name=None)
    self.assertEqual(get_gcs_output_path(config), "gs://bucket/test")

    # output_dir local, base_output_directory starting with gs://
    config = SimpleNamespace(output_dir="/tmp/test", base_output_directory="gs://bucket/base", run_name="exp2")
    self.assertEqual(get_gcs_output_path(config), "gs://bucket/base/exp2")

    # output_dir local, base_output_directory starting with gs://, no run_name
    config = SimpleNamespace(output_dir="/tmp/test", base_output_directory="gs://bucket/base")
    self.assertEqual(get_gcs_output_path(config), "gs://bucket/base")

    # Non-GCS outputs
    config = SimpleNamespace(output_dir="/tmp/test", base_output_directory="/tmp/base", run_name="exp3")
    self.assertEqual(get_gcs_output_path(config), "")

    # Defensive handling against None / missing attributes
    config = SimpleNamespace(output_dir=None, base_output_directory=None, run_name=None)
    self.assertEqual(get_gcs_output_path(config), "")
    config = SimpleNamespace()
    self.assertEqual(get_gcs_output_path(config), "")

  def test_chunk_and_pad(self):
    # Exact multiple
    chunks = list(chunk_and_pad(["a", "b", "c", "d"], batch_size=2))
    self.assertEqual(chunks, [(0, ["a", "b"], 2), (2, ["c", "d"], 2)])

    # Tail batch requiring padding
    chunks = list(chunk_and_pad(["a", "b", "c"], batch_size=2))
    self.assertEqual(chunks, [(0, ["a", "b"], 2), (2, ["c", "c"], 1)])

    # Single item with larger batch size
    chunks = list(chunk_and_pad(["a"], batch_size=3))
    self.assertEqual(chunks, [(0, ["a", "a", "a"], 1)])

    # Empty items
    chunks = list(chunk_and_pad([], batch_size=2))
    self.assertEqual(chunks, [])

    # Invalid batch_size
    with self.assertRaises(ValueError):
      list(chunk_and_pad(["a"], batch_size=0))
    with self.assertRaises(ValueError):
      list(chunk_and_pad(["a"], batch_size=-1))

  def test_get_dummy_wan_inputs_generates_latents_without_pipeline_prepare_latents(self):
    config = SimpleNamespace(height=64, width=80, num_frames=9, seed=0)
    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=16)),
        vae_scale_factor_temporal=4,
        vae_scale_factor_spatial=8,
        prepare_latents=Mock(side_effect=AssertionError("prepare_latents should not be called")),
    )

    latents, prompt_embeds, timesteps = maxdiffusion_utils.get_dummy_wan_inputs(config, pipeline, batch_size=2)

    pipeline.prepare_latents.assert_not_called()
    self.assertEqual(latents.shape, (2, 16, 3, 8, 10))
    self.assertEqual(prompt_embeds.shape, (2, 512, 4096))
    self.assertEqual(timesteps.shape, (2,))

  def test_get_dummy_wan_inputs_supports_two_expert_pipeline(self):
    config = SimpleNamespace(height=64, width=80, num_frames=9, seed=0)
    pipeline = SimpleNamespace(
        low_noise_transformer=SimpleNamespace(config=SimpleNamespace(in_channels=48)),
        high_noise_transformer=SimpleNamespace(config=SimpleNamespace(in_channels=48)),
        vae_scale_factor_temporal=4,
        vae_scale_factor_spatial=8,
        prepare_latents=Mock(side_effect=AssertionError("prepare_latents should not be called")),
    )

    latents, prompt_embeds, timesteps = maxdiffusion_utils.get_dummy_wan_inputs(config, pipeline, batch_size=2)

    pipeline.prepare_latents.assert_not_called()
    self.assertEqual(latents.shape, (2, 48, 3, 8, 10))
    self.assertEqual(prompt_embeds.shape, (2, 512, 4096))
    self.assertEqual(timesteps.shape, (2,))

  def test_create_scheduler(self):
    """Test create scheduler with different schedulers"""
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config

    # Setup Mesh
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    flash_block_sizes = get_flash_block_sizes(config)

    pipeline, _ = FlaxStableDiffusionXLPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        revision=config.revision,
        dtype=config.activations_dtype,
        split_head_dim=config.split_head_dim,
        norm_num_groups=config.norm_num_groups,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        text_encoder_2=None,
        unet=None,
    )
    scheduler_config = pipeline.scheduler.config

    assert scheduler_config["prediction_type"] == "epsilon"
    assert not scheduler_config.get("rescale_zero_terminal_snr", False)
    assert scheduler_config["timestep_spacing"] == "leading"

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"

    # Test class name override without Flax Name.
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"_class_name" : "DDIMScheduler", "prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config
    scheduler_config = scheduler.config

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"
    assert type(scheduler) is FlaxDDIMScheduler

    # Test class name override with Flax Name.
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "revision=refs/pr/95",
            "activations_dtype=bfloat16",
            'diffusion_scheduler_config={"_class_name" : "FlaxDDPMScheduler", "prediction_type" : "v_prediction", '
            '"rescale_zero_terminal_snr" : true, "timestep_spacing" : "trailing"}',
        ],
        unittest=True,
    )

    config = pyconfig.config
    scheduler_config = scheduler.config

    scheduler, _ = maxdiffusion_utils.create_scheduler(scheduler_config, config)

    assert scheduler.config["prediction_type"] == "v_prediction"
    assert scheduler.config["rescale_zero_terminal_snr"]
    assert scheduler.config["timestep_spacing"] == "trailing"
    assert type(scheduler) is FlaxDDPMScheduler
