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
from ..pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from .. import max_logging
import orbax.checkpoint as ocp
from etils import epath
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer


class WanCheckpointer2_2(WanCheckpointer):

  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest WAN checkpoint step: {step}")
      if step is None:
        max_logging.log("No WAN checkpoint found.")
        return None, None
    max_logging.log(f"Loading WAN checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)

    # Handle low_noise_transformer
    low_noise_transformer_metadata = metadatas.low_noise_transformer_state
    abstract_tree_structure_low_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, low_noise_transformer_metadata
    )
    low_params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_low_params,
        )
    )

    # Handle high_noise_transformer
    high_noise_transformer_metadata = metadatas.high_noise_transformer_state
    abstract_tree_structure_high_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, high_noise_transformer_metadata
    )
    high_params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_high_params,
        )
    )

    max_logging.log("Restoring WAN 2.2 checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            low_noise_transformer_state=low_params_restore,
            high_noise_transformer_state=high_params_restore,
            wan_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(
        f"restored checkpoint low_noise_transformer_state {restored_checkpoint.low_noise_transformer_state.keys()}"
    )
    max_logging.log(
        f"restored checkpoint high_noise_transformer_state {restored_checkpoint.high_noise_transformer_state.keys()}"
    )
    max_logging.log(
        f"optimizer found in low_noise checkpoint {'opt_state' in restored_checkpoint.low_noise_transformer_state.keys()}"
    )
    max_logging.log(
        f"optimizer found in high_noise checkpoint {'opt_state' in restored_checkpoint.high_noise_transformer_state.keys()}"
    )
    max_logging.log(f"optimizer state saved in attribute self.opt_state {self.opt_state}")
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    pipeline = WanPipeline2_2.from_pretrained(self.config)
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_2, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      max_logging.log("Loading WAN pipeline from checkpoint")
      pipeline = WanPipeline2_2.from_checkpoint(self.config, restored_checkpoint)
      # Check for optimizer state in either transformer
      if "opt_state" in restored_checkpoint.low_noise_transformer_state.keys():
        opt_state = restored_checkpoint.low_noise_transformer_state["opt_state"]
      elif "opt_state" in restored_checkpoint.high_noise_transformer_state.keys():
        opt_state = restored_checkpoint.high_noise_transformer_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default pipeline.")
      pipeline = self.load_diffusers_checkpoint()

    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_2, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.low_noise_transformer)),
    }

    items["low_noise_transformer_state"] = ocp.args.PyTreeSave(train_states["low_noise_transformer"])
    items["high_noise_transformer_state"] = ocp.args.PyTreeSave(train_states["high_noise_transformer"])

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
