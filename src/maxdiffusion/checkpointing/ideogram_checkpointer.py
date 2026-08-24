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

import dataclasses
import jax
import numpy as np
from typing import Optional, Tuple
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline
from maxdiffusion import max_logging
from maxdiffusion.checkpointing.checkpointing_utils import create_orbax_checkpoint_manager
import orbax.checkpoint as ocp
from etils import epath

IDEOGRAM_CHECKPOINT = "IDEOGRAM_CHECKPOINT"
# The unconditional branch is a separate checkpoint (asymmetric CFG), so it
# needs its own item. IdeogramPipeline.from_checkpoint reads this key.
UNCONDITIONAL_STATE_KEY = "unconditional_ideogram_state"


def _config_to_json(config) -> dict:
  """Ideogram4Config as a JSON-safe dict (dtype fields are not serializable)."""
  return {
      k: (str(v) if not isinstance(v, (int, float, str, bool, list, type(None))) else v)
      for k, v in dataclasses.asdict(config).items()
  }


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

    def _params_restore(metadata):
      abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, metadata)
      return ocp.args.PyTreeRestore(
          restore_args=jax.tree.map(
              lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
              abstract_tree_structure_params,
          )
      )

    max_logging.log("Restoring Ideogram checkpoint")
    # Ideogram uses asymmetric CFG: the conditional and unconditional branches
    # are genuinely different checkpoints, so both must round-trip.
    restore_items = {
        "ideogram_state": _params_restore(metadatas.ideogram_state),
        "ideogram_config": ocp.args.JsonRestore(),
    }
    uncond_metadata = getattr(metadatas, UNCONDITIONAL_STATE_KEY, None)
    if uncond_metadata is not None:
      restore_items[UNCONDITIONAL_STATE_KEY] = _params_restore(uncond_metadata)

    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(**restore_items),
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

  def save_checkpoint(self, train_step, pipeline: IdeogramPipeline, train_states: dict, unconditional_states: dict = None):
    """Saves the training state and model configurations.

    ``train_states`` is the conditional branch. ``unconditional_states``, if
    given, is saved under ``unconditional_ideogram_state`` -- without it a
    restored pipeline has no unconditional weights and ``from_checkpoint``
    raises.
    """
    max_logging.log(f"Saving checkpoint for step {train_step}")
    # IdeogramPipeline holds `conditional_transformer`/`unconditional_transformer`;
    # there is no `pipeline.transformer`. Persist the dataclass config instead.
    items = {
        "ideogram_config": ocp.args.JsonSave(_config_to_json(pipeline.conditional_transformer.config)),
        "ideogram_state": ocp.args.PyTreeSave(train_states),
    }
    if unconditional_states is not None:
      items[UNCONDITIONAL_STATE_KEY] = ocp.args.PyTreeSave(unconditional_states)

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
