"""Copyright 2026 Google LLC

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
from unittest.mock import MagicMock, patch
from maxdiffusion.trainers.stable_diffusion_trainer import StableDiffusionTrainer


class MockConfig:

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


class StableDiffusionTrainerTest(unittest.TestCase):

  @patch("maxdiffusion.trainers.stable_diffusion_trainer.train_utils")
  @patch("maxdiffusion.trainers.stable_diffusion_trainer.max_utils")
  @patch("maxdiffusion.trainers.stable_diffusion_trainer.jax")
  @patch("maxdiffusion.trainers.stable_diffusion_trainer.os")
  def test_training_loop_total_weights(self, mock_os, mock_jax, mock_max_utils, mock_train_utils):
    # Setup mocks
    mock_jax.process_index.return_value = 0
    mock_jax.random.split.return_value = ("dummy1", "dummy2")
    mock_os.environ = {"LIBTPU_INIT_ARGS": ""}

    mock_max_utils.profiler_enabled.return_value = False

    def fake_calc_params(pytree):
      if pytree == "unet_params":
        return 1000
      elif pytree == "text_encoder_params":
        return 500
      return 0

    mock_max_utils.calculate_num_params_from_pytree.side_effect = fake_calc_params

    # We want the loop to hit exactly 1 step then exit.
    mock_train_utils.get_first_step.return_value = 0

    unet_state = MagicMock()
    unet_state.params = "unet_params"

    vae_state = MagicMock()

    text_encoder_state = MagicMock()
    text_encoder_state.params = "text_encoder_params"

    train_states = {
        "unet_state": unet_state,
        "vae_state": vae_state,
        "text_encoder_state": text_encoder_state,
    }

    p_train_step = MagicMock()
    # p_train_step returns: unet_state, text_encoder_state, train_metric, train_rngs
    p_train_step.return_value = (unet_state, text_encoder_state, {}, "rngs")

    data_iterator = MagicMock()
    lr_scheduler = MagicMock()
    lr_scheduler.return_value = 0.001

    # Instance of trainer
    with patch("maxdiffusion.trainers.stable_diffusion_trainer.BaseStableDiffusionTrainer.__init__") as mock_init:
      mock_init.return_value = None

      # Test train_text_encoder = False
      config_false = MockConfig(
          train_text_encoder=False,
          max_train_steps=1,
          per_device_batch_size=1,
          checkpoint_every=-1,
          write_metrics=False,
          metrics_file=None,
          gcs_metrics=None,
          skip_first_n_steps_for_profiler=999,
          profiler_steps=10,
      )
      trainer = StableDiffusionTrainer(config_false)
      trainer.config = config_false
      trainer.total_train_batch_size = 1
      trainer.per_device_tflops = 1.0
      trainer.rng = "rng"
      trainer.checkpoint_manager = MagicMock()
      trainer.checkpoint_manager.reached_preemption.return_value = False
      trainer.save_checkpoint = MagicMock()

      trainer.training_loop(p_train_step, None, None, train_states, data_iterator, lr_scheduler)

      # Verify total_weights recorded
      mock_train_utils.record_scalar_metrics.assert_called()
      kwargs = mock_train_utils.record_scalar_metrics.call_args.kwargs
      self.assertEqual(kwargs.get("total_weights"), 1000)

      # Reset mocks
      mock_train_utils.record_scalar_metrics.reset_mock()

      # Test train_text_encoder = True
      config_true = MockConfig(
          train_text_encoder=True,
          max_train_steps=1,
          per_device_batch_size=1,
          checkpoint_every=-1,
          write_metrics=False,
          metrics_file=None,
          gcs_metrics=None,
          skip_first_n_steps_for_profiler=999,
          profiler_steps=10,
      )
      trainer_true = StableDiffusionTrainer(config_true)
      trainer_true.config = config_true
      trainer_true.total_train_batch_size = 1
      trainer_true.per_device_tflops = 1.0
      trainer_true.rng = "rng"
      trainer_true.checkpoint_manager = MagicMock()
      trainer_true.checkpoint_manager.reached_preemption.return_value = False
      trainer_true.save_checkpoint = MagicMock()

      trainer_true.training_loop(p_train_step, None, None, train_states, data_iterator, lr_scheduler)

      mock_train_utils.record_scalar_metrics.assert_called()
      kwargs = mock_train_utils.record_scalar_metrics.call_args.kwargs
      self.assertEqual(kwargs.get("total_weights"), 1500)


if __name__ == "__main__":
  unittest.main()
