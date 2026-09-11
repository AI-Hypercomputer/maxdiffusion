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
from maxdiffusion.trainers.dreambooth_trainer import DreamboothTrainer

UNET_PARAMS = 1000
TEXT_ENCODER_PARAMS = 500


class MockConfig:

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


class DreamboothTrainerTest(unittest.TestCase):

  @patch("maxdiffusion.trainers.dreambooth_trainer.train_utils")
  @patch("maxdiffusion.trainers.dreambooth_trainer.max_utils")
  @patch("maxdiffusion.trainers.dreambooth_trainer.jax")
  @patch("maxdiffusion.trainers.dreambooth_trainer.os")
  def test_training_loop_total_weights(self, mock_os, mock_jax, mock_max_utils, mock_train_utils):
    """total_weights includes the text encoder only when it is trained."""
    mock_jax.process_index.return_value = 0
    mock_jax.random.split.return_value = ("dummy1", "dummy2")
    mock_os.environ = {"LIBTPU_INIT_ARGS": ""}
    mock_max_utils.profiler_enabled.return_value = False
    mock_max_utils.calculate_num_params_from_pytree.side_effect = lambda params: {
        "unet_params": UNET_PARAMS,
        "text_encoder_params": TEXT_ENCODER_PARAMS,
    }[params]
    mock_train_utils.get_first_step.return_value = 0

    unet_state = MagicMock()
    unet_state.params = "unet_params"
    text_encoder_state = MagicMock()
    text_encoder_state.params = "text_encoder_params"
    train_states = {"unet_state": unet_state, "text_encoder_state": text_encoder_state}

    p_train_step = MagicMock()
    p_train_step.return_value = (unet_state, text_encoder_state, {}, "rngs")

    for train_text_encoder, expected_total_weights in (
        (False, UNET_PARAMS),
        (True, UNET_PARAMS + TEXT_ENCODER_PARAMS),
    ):
      with self.subTest(train_text_encoder=train_text_encoder):
        mock_train_utils.record_scalar_metrics.reset_mock()
        config = MockConfig(
            train_text_encoder=train_text_encoder,
            max_train_steps=1,
            per_device_batch_size=1,
            checkpoint_every=-1,
            write_metrics=False,
            metrics_file=None,
            gcs_metrics=None,
            skip_first_n_steps_for_profiler=999,
            profiler_steps=10,
        )

        with patch("maxdiffusion.trainers.dreambooth_trainer.BaseStableDiffusionTrainer.__init__", return_value=None):
          trainer = DreamboothTrainer(config)
        trainer.config = config
        trainer.total_train_batch_size = 1
        trainer.per_device_tflops = 1.0
        trainer.rng = "rng"
        trainer.checkpoint_manager = MagicMock()
        trainer.save_checkpoint = MagicMock()

        trainer.training_loop(p_train_step, None, None, train_states, MagicMock(), MagicMock())

        kwargs = mock_train_utils.record_scalar_metrics.call_args.kwargs
        self.assertEqual(kwargs.get("total_weights"), expected_total_weights)


if __name__ == "__main__":
  unittest.main()
