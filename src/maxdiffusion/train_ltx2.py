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

"""CLI entrypoint for LTX-2 / LTX-2.3 bucketed joint audio-video training."""

from typing import Sequence

from absl import app
import flax
import jax

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.common_types import LTX2_3, LTX2_VIDEO
from maxdiffusion.train_utils import transformer_engine_context, validate_train_config
from maxdiffusion.trainers.ltx2_trainer import LTX2Trainer


_ALLOWED_LTX2_TRAINING_MODELS = {LTX2_VIDEO, LTX2_3}


def _validate_ltx2_training_model_name(model_name: str | None) -> None:
  if model_name not in _ALLOWED_LTX2_TRAINING_MODELS:
    raise ValueError(
        f"Invalid config.model_name {model_name!r} for train_ltx2.py. "
        f"Allowed values: {sorted(_ALLOWED_LTX2_TRAINING_MODELS)}."
    )


def train(config):
  _validate_ltx2_training_model_name(getattr(config, "model_name", None))
  trainer = LTX2Trainer(config)
  return trainer.start_training()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  config = pyconfig.config
  _validate_ltx2_training_model_name(getattr(config, "model_name", None))
  max_utils.ensure_machinelearning_job_runs(config)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  train(config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
