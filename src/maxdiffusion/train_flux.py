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

from typing import Sequence

import jax
from absl import app
from maxdiffusion import (
    max_logging,
    pyconfig,
    mllog_utils,
)

from maxdiffusion.train_utils import (
    validate_train_config,
)


def train(config):
  from maxdiffusion.trainers.flux_trainer import FluxTrainer

  trainer = FluxTrainer(config)
  trainer.start_training()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  config = pyconfig.config
  mllog_utils.train_init_start(config)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  train(config)


if __name__ == "__main__":
  app.run(main)
