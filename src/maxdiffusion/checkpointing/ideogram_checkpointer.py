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

import json
import jax
import numpy as np
from typing import Optional, Tuple
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline
from maxdiffusion import max_logging
from maxdiffusion.checkpointing.checkpointing_utils import create_orbax_checkpoint_manager
import orbax.checkpoint as ocp
from etils import epath

IDEOGRAM_CHECKPOINT = "IDEOGRAM_CHECKPOINT"


class IdeogramCheckpointer:

  def __init__(self, config, checkpoint_type: str = IDEOGRAM_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None

    self.checkpoint_manager: ocp.CheckpointManager = create_orbax_checkpoint_manager(
        getattr(self.config, "checkpoint_dir", ""),
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=getattr(config, "dataset_type", None),
    )

  def load_ideogram_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if self.checkpoint_manager is None:
      max_logging.log("No checkpoint manager configured, skipping Orbax load.")
      return None, None

    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest Ideogram checkpoint step: {step}")
      if step is None:
        max_logging.log("No Ideogram checkpoint found.")
        return None, None
    max_logging.log(f"Loading Ideogram checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)
    transformer_metadata = metadatas.ideogram_state
    abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, transformer_metadata)
    params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_params,
        )
    )

    max_logging.log("Restoring Ideogram checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            ideogram_state=params_restore,
            ideogram_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(f"restored checkpoint ideogram_state {restored_checkpoint.ideogram_state.keys()}")
    max_logging.log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.ideogram_state.keys()}")
    return restored_checkpoint, step

  def load_checkpoint(
      self, step=None, vae_only=False, load_transformer=True
  ) -> Tuple[IdeogramPipeline, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_ideogram_configs_from_orbax(step)
    opt_state = None

    if restored_checkpoint:
      max_logging.log("Loading Ideogram pipeline from checkpoint")
      pipeline = IdeogramPipeline.from_checkpoint(self.config, restored_checkpoint, vae_only, load_transformer)
      if "opt_state" in restored_checkpoint.ideogram_state.keys():
        opt_state = restored_checkpoint.ideogram_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading pipeline from pretrained hub")
      pipeline = IdeogramPipeline.from_pretrained(self.config, vae_only, load_transformer)

    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: IdeogramPipeline, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "ideogram_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
    }

    items["ideogram_state"] = ocp.args.PyTreeSave(train_states)

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
