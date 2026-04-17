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
from typing import Optional, Tuple
import jax
from maxdiffusion.checkpointing.checkpointing_utils import add_sharding_to_struct, get_cpu_mesh_and_sharding
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
import orbax.checkpoint as ocp
from .. import max_logging
from ..pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1


class WanCheckpointerI2V_2_1(WanCheckpointer):

  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest WAN checkpoint step: {step}")
      if step is None:
        max_logging.log("No WAN checkpoint found.")
        return None, None
    max_logging.log(f"Loading WAN checkpoint from step {step}")

    mesh, replicated_sharding = get_cpu_mesh_and_sharding()
    metadatas = self.checkpoint_manager.item_metadata(step)
    state = metadatas.wan_state

    target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, state)

    with mesh:
      abstract_train_state_with_sharding = jax.tree_util.tree_map(add_sharding_to_struct, state, target_shardings)

    max_logging.log("Restoring WAN checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        step=step,
        args=ocp.args.Composite(
            wan_config=ocp.args.JsonRestore(),
            wan_state=ocp.args.StandardRestore(abstract_train_state_with_sharding),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(f"restored checkpoint wan_state {restored_checkpoint.wan_state.keys()}")
    max_logging.log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.wan_state.keys()}")
    max_logging.log(f"optimizer state saved in attribute self.opt_state {self.opt_state}")
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    pipeline = WanPipelineI2V_2_1.from_pretrained(self.config)
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipelineI2V_2_1, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      max_logging.log("Loading WAN pipeline from checkpoint")
      pipeline = WanPipelineI2V_2_1.from_checkpoint(self.config, restored_checkpoint)
      if "opt_state" in restored_checkpoint.wan_state.keys():
        opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default pipeline.")
      pipeline = self.load_diffusers_checkpoint()

    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipelineI2V_2_1, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
    }

    items["wan_state"] = ocp.args.StandardSave(train_states)

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
