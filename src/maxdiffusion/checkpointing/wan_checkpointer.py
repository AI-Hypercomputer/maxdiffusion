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

from abc import ABC
from flax import nnx
from maxdiffusion.checkpointing.checkpointing_utils import (create_orbax_checkpoint_manager)
from ..pipelines.wan.wan_pipeline import WanPipeline
from .. import max_logging, max_utils

WAN_CHECKPOINT = "WAN_CHECKPOINT"


class WanCheckpointer(ABC):

  def __init__(self, config, checkpoint_type):
    self.config = config
    self.checkpoint_type = checkpoint_type

    self.checkpoint_manager = create_orbax_checkpoint_manager(
        self.config.checkpoint_dir,
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=config.dataset_type,
    )

  def _create_optimizer(self, model, config, learning_rate):
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return nnx.Optimizer(model, tx), learning_rate_scheduler

  def load_wan_configs_from_orbax(self, step):
    max_logging.log("Restoring stable diffusion configs")
    if step is None:
      step = self.checkpoint_manager.latest_step()
      if step is None:
        return None

  def load_diffusers_checkpoint(self):
    pipeline = WanPipeline.from_pretrained(self.config)
    return pipeline

  def load_checkpoint(self, step=None):
    model_configs = self.load_wan_configs_from_orbax(step)

    if model_configs:
      raise NotImplementedError("model configs should not exist in orbax")
    else:
      pipeline = self.load_diffusers_checkpoint()

    return pipeline
