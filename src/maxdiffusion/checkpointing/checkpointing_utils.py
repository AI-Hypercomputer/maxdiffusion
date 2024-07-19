# ruff: noqa
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

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

from typing import Optional, Union
import json
import jax
import numpy as np
from maxdiffusion import max_logging
from etils import epath
from flax.training import train_state
import orbax
import orbax.checkpoint as ocp
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint import type_handlers
from orbax.checkpoint.checkpoint_manager import Checkpointer, CheckpointManager, CheckpointManagerOptions

STABLE_DIFFUSION_CHECKPOINT = "STABLE_DIFFUSION_CHECKPOINT"
STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"

def create_orbax_checkpoint_manager(
  checkpoint_dir: str,
  enable_checkpointing: bool,
  save_interval_steps,
  checkpoint_type: str,
  use_async: bool = True,
  orbax_logger: Optional[abstract_logger.AbstractLogger] = None
):
  """
  Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled.
  checkpoint_type: Options are sd or sdxl.
  """
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  
  max_logging.log("Creating checkpoing manager...")
  max_logging.log(f"checkpoint dir: {checkpoint_dir}")
  p = epath.Path(checkpoint_dir)

  item_names = (
    "unet_config",
    "vae_config",
    "text_encoder_config",
    "text_encoder_2_config",
    "scheduler_config",
    "unet_state",
    "vae_state",
    "text_encoder_state",
    "tokenizer_config"
  )
  if checkpoint_type == STABLE_DIFFUSION_XL_CHECKPOINT:
    item_names + ("text_encoder_2_state", "text_encoder_2_config")
  
  mngr = CheckpointManager(
    p,
    item_names=item_names,
    options=CheckpointManagerOptions(
      create=True,
      save_interval_steps=save_interval_steps,
      enable_async_checkpointing=use_async
    ),
    logger=orbax_logger
  )

  max_logging.log("Checkpoint manager created!")
  return mngr

def load_stable_diffusion_configs(
  checkpoint_manager: CheckpointManager,
  checkpoint_type: str,
  step: Optional[int] = None,
):
  f"""
  Loads Orbax configurations for different stable diffusion models
  
  Args:
  checkpoint_manager (`orbax.checkpoint.checkpoint_manager`)
  checkpoint_type (`str`) : use sd or sdxl
  step (int) : step to restore, if None is passed, defaults to latest.
  """
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      return None
  
  restore_args = {
    "unet_config" : orbax.checkpoint.args.JsonRestore(),
    "vae_config" : orbax.checkpoint.args.JsonRestore(),
    "text_encoder_config" : orbax.checkpoint.args.JsonRestore(),
    "scheduler_config" : orbax.checkpoint.args.JsonRestore(),
  }

  if checkpoint_type == STABLE_DIFFUSION_XL_CHECKPOINT:
    restore_args["text_encoder_2_config"] = orbax.checkpoint.args.JsonRestore()
  
  return (
      checkpoint_manager.restore(step,
        args=orbax.checkpoint.args.Composite(**restore_args)
      ),None)

def load_state_if_possible(
  checkpoint_manager: CheckpointManager,
  abstract_unboxed_pre_state: train_state.TrainState
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  if checkpoint_manager is None:
    max_logging.log("no checkpoint manager, not restoring checkpoint")
    return None
  latest_step = checkpoint_manager.latest_step()
  if latest_step is None:
    return None
  else:
    max_logging.log(
      f"restoring from this run's directory latest step {latest_step}"
    )
    return checkpoint_manager.restore(
      latest_step,
      args=orbax.checkpoint.args.Composite(
        unet_state=orbax.checkpoint.args.StandardRestore(item=abstract_unboxed_pre_state)
      )
    )