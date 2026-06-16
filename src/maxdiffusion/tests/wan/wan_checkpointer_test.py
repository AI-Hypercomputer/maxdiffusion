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
"""

import unittest
from unittest.mock import patch, MagicMock
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion.pipelines.wan.wan_pipeline import _select_restored_transformer_state
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2


class WanPretrainedCacheTest(unittest.TestCase):
  """Tests for the shared WAN pretrained Orbax cache helper."""

  def setUp(self):
    self.config = MagicMock()
    self.config.pretrained_orbax_dir = "/tmp/wan_pretrained_cache"
    self.pipeline_cls = MagicMock()
    self.state_sources = (("wan_state", "transformer"),)
    self.config_transformer_attr = "transformer"

  @patch.object(WanCheckpointer, "_restore_pretrained_checkpoint")
  def test_loads_from_pretrained_cache_hit(self, mock_restore):
    restored_checkpoint = MagicMock()
    mock_restore.return_value = restored_checkpoint
    pipeline = MagicMock()
    self.pipeline_cls.from_checkpoint.return_value = pipeline

    result = WanCheckpointer.load_pretrained_pipeline_or_diffusers(
        self.config, self.pipeline_cls, self.state_sources, self.config_transformer_attr
    )

    mock_restore.assert_called_once_with(self.config.pretrained_orbax_dir, ("wan_state",))
    self.pipeline_cls.from_checkpoint.assert_called_once_with(self.config, restored_checkpoint)
    self.pipeline_cls.from_pretrained.assert_not_called()
    self.assertEqual(result, pipeline)

  @patch.object(WanCheckpointer, "_save_pretrained_checkpoint")
  @patch.object(WanCheckpointer, "_restore_pretrained_checkpoint")
  def test_loads_from_diffusers_and_saves_on_cache_miss(self, mock_restore, mock_save):
    mock_restore.return_value = None
    pipeline = MagicMock()
    self.pipeline_cls.from_pretrained.return_value = pipeline

    result = WanCheckpointer.load_pretrained_pipeline_or_diffusers(
        self.config, self.pipeline_cls, self.state_sources, self.config_transformer_attr
    )

    self.pipeline_cls.from_pretrained.assert_called_once_with(self.config)
    mock_save.assert_called_once_with(
        self.config.pretrained_orbax_dir, pipeline, self.state_sources, self.config_transformer_attr
    )
    self.assertEqual(result, pipeline)

  @patch.object(WanCheckpointer, "_save_pretrained_checkpoint")
  @patch.object(WanCheckpointer, "_restore_pretrained_checkpoint")
  def test_empty_pretrained_dir_uses_diffusers_without_cache(self, mock_restore, mock_save):
    self.config.pretrained_orbax_dir = ""
    pipeline = MagicMock()
    self.pipeline_cls.from_pretrained.return_value = pipeline

    result = WanCheckpointer.load_pretrained_pipeline_or_diffusers(
        self.config, self.pipeline_cls, self.state_sources, self.config_transformer_attr
    )

    mock_restore.assert_not_called()
    mock_save.assert_not_called()
    self.pipeline_cls.from_pretrained.assert_called_once_with(self.config)
    self.assertEqual(result, pipeline)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.nnx.split")
  def test_pretrained_save_items_uses_explicit_transformer_config(self, mock_split):
    pipeline = MagicMock()
    low_noise_transformer = MagicMock()
    high_noise_transformer = MagicMock()
    low_noise_transformer.to_json_string.return_value = '{"model_type": "wan"}'
    pipeline.low_noise_transformer = low_noise_transformer
    pipeline.high_noise_transformer = high_noise_transformer
    low_noise_state = MagicMock()
    high_noise_state = MagicMock()
    low_noise_state.to_pure_dict.return_value = {"low": "state"}
    high_noise_state.to_pure_dict.return_value = {"high": "state"}
    mock_split.side_effect = [
        (None, low_noise_state, None),
        (None, high_noise_state, None),
    ]

    items = WanCheckpointer._pretrained_save_items(
        pipeline,
        (
            ("low_noise_transformer_state", "low_noise_transformer"),
            ("high_noise_transformer_state", "high_noise_transformer"),
        ),
        "low_noise_transformer",
    )

    low_noise_transformer.to_json_string.assert_called_once()
    high_noise_transformer.to_json_string.assert_not_called()
    mock_split.assert_any_call(low_noise_transformer, unittest.mock.ANY, ...)
    mock_split.assert_any_call(high_noise_transformer, unittest.mock.ANY, ...)
    self.assertIn("low_noise_transformer_state", items)
    self.assertIn("high_noise_transformer_state", items)
    self.assertIn("wan_config", items)

  def test_pretrained_save_items_requires_transformer_source(self):
    with self.assertRaisesRegex(ValueError, "at least one transformer source"):
      WanCheckpointer._pretrained_save_items(MagicMock(), (), "transformer")


class WanRestoredTransformerStateTest(unittest.TestCase):
  """Tests for strict WAN checkpoint state selection."""

  def test_selects_single_wan_state(self):
    restored_checkpoint = {"wan_state": {"params": {}}}

    self.assertEqual(_select_restored_transformer_state(restored_checkpoint, ""), {"params": {}})

  def test_selects_wan_2_2_low_noise_state(self):
    restored_checkpoint = {"low_noise_transformer_state": {"low": "state"}}

    self.assertEqual(
        _select_restored_transformer_state(restored_checkpoint, "transformer_2"),
        {"low": "state"},
    )

  def test_selects_wan_2_2_high_noise_state(self):
    restored_checkpoint = {"high_noise_transformer_state": {"high": "state"}}

    self.assertEqual(
        _select_restored_transformer_state(restored_checkpoint, "transformer"),
        {"high": "state"},
    )

  def test_rejects_mismatched_wan_2_2_state(self):
    restored_checkpoint = {"low_noise_transformer_state": {"low": "state"}}

    with self.assertRaisesRegex(ValueError, "high_noise_transformer_state"):
      _select_restored_transformer_state(restored_checkpoint, "transformer")

  def test_rejects_unknown_subfolder(self):
    restored_checkpoint = {"low_noise_transformer_state": {}, "high_noise_transformer_state": {}}

    with self.assertRaisesRegex(ValueError, "Unsupported WAN checkpoint transformer subfolder"):
      _select_restored_transformer_state(restored_checkpoint, "unexpected")


class WanCheckpointer2_1Test(unittest.TestCase):
  """Tests for WAN 2.1 checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_1, "from_pretrained", autospec=True)
  def test_load_from_diffusers(self, mock_from_pretrained, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_from_pretrained.assert_called_once_with(
        self.config,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_1, "from_checkpoint", autospec=True)
  def test_load_checkpoint_no_optimizer(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(step=1, args=unittest.mock.ANY)
    mock_from_checkpoint.assert_called_with(
        self.config,
        mock_manager.restore.return_value,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_1, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(step=1, args=unittest.mock.ANY)
    mock_from_checkpoint.assert_called_with(
        self.config,
        mock_manager.restore.return_value,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)


class WanCheckpointer2_2Test(unittest.TestCase):
  """Tests for WAN 2.2 checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_2_2_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_2, "from_pretrained", autospec=True)
  def test_load_from_diffusers(self, mock_from_pretrained, mock_create_manager):
    """Test loading from pretrained when no checkpoint exists."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_from_pretrained.assert_called_once_with(
        self.config,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_no_optimizer(self, mock_from_checkpoint, mock_create_manager):
    """Test loading checkpoint without optimizer state."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(step=1, args=unittest.mock.ANY)
    mock_from_checkpoint.assert_called_with(
        self.config,
        mock_manager.restore.return_value,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer_in_low_noise(self, mock_from_checkpoint, mock_create_manager):
    """Test loading checkpoint with optimizer state in low_noise_transformer."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(step=1, args=unittest.mock.ANY)
    mock_from_checkpoint.assert_called_with(
        self.config,
        mock_manager.restore.return_value,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer_in_high_noise(self, mock_from_checkpoint, mock_create_manager):
    """Test loading checkpoint with optimizer state in high_noise_transformer."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.002}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(step=1, args=unittest.mock.ANY)
    mock_from_checkpoint.assert_called_with(
        self.config,
        mock_manager.restore.return_value,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.002)
    self.assertEqual(step, 1)


class WanCheckpointerI2V_2_1Test(unittest.TestCase):
  """Tests for WAN 2.1 I2V checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_i2v_checkpoint_test"
    self.config.dataset_type = "test_dataset"
    self.config.model_type = "I2V"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_1, "from_pretrained", autospec=True)
  def test_load_from_diffusers(self, mock_from_pretrained, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_from_pretrained.assert_called_once_with(
        self.config,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_1, "from_checkpoint", autospec=True)
  def test_load_checkpoint_no_optimizer(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once()
    mock_from_checkpoint.assert_called_once_with(
        self.config,
        restored_mock,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_1, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once()
    mock_from_checkpoint.assert_called_once_with(
        self.config,
        restored_mock,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)


class WanCheckpointerI2V_2_2Test(unittest.TestCase):
  """Tests for WAN 2.2 I2V checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_i2v_2_2_checkpoint_test"
    self.config.dataset_type = "test_dataset"
    self.config.model_type = "I2V"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_2, "from_pretrained", autospec=True)
  def test_load_from_diffusers(self, mock_from_pretrained, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_from_pretrained.assert_called_once_with(
        self.config,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_no_optimizer(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once()
    mock_from_checkpoint.assert_called_once_with(
        self.config,
        restored_mock,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer_in_low_noise(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once()
    mock_from_checkpoint.assert_called_once_with(
        self.config,
        restored_mock,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipelineI2V_2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_optimizer_in_high_noise(self, mock_from_checkpoint, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.002}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointerI2V_2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once()
    mock_from_checkpoint.assert_called_once_with(
        self.config,
        restored_mock,
        vae_only=False,
        load_vae=None,
        load_text_encoder=None,
        load_transformer=None,
        load_scheduler=None,
    )
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.002)
    self.assertEqual(step, 1)


class WanCheckpointerEdgeCasesTest(unittest.TestCase):
  """Tests for edge cases and error handling."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_edge_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_1, "from_checkpoint", autospec=True)
  def test_load_checkpoint_with_explicit_none_step(self, mock_from_checkpoint, mock_create_manager):
    """Test loading checkpoint with explicit None step falls back to latest."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 5
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}}
    restored_mock.wan_config = {}
    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    self.assertEqual(step, 5)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch.object(WanPipeline2_2, "from_checkpoint", autospec=True)
  def test_load_checkpoint_both_optimizers_present(self, mock_from_checkpoint, mock_create_manager):
    """Test loading checkpoint when both transformers have optimizer state (prioritize low_noise)."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.high_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.002}}
    restored_mock.wan_config = {}
    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(config=self.config)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    # Should prioritize low_noise_transformer's optimizer state
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)


if __name__ == "__main__":
  unittest.main(verbosity=2)
