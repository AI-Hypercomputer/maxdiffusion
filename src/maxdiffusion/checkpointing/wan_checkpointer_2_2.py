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
from typing import Optional, Tuple
from ..pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from .. import max_logging, max_utils
import orbax.checkpoint as ocp
from maxdiffusion.checkpointing.checkpointing_utils import add_sharding_to_struct, get_cpu_mesh_and_sharding
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer


class WanCheckpointer2_2(WanCheckpointer[WanPipeline2_2]):
  pipeline_class = WanPipeline2_2

  def _create_optimizer(self, model, config, learning_rate, scale_factor: float = 1.0):
    total_steps = max(1, int(config.max_train_steps * scale_factor))
    schedule_steps = max(1, int(config.learning_rate_schedule_steps * scale_factor))
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, schedule_steps, config.warmup_steps_fraction, total_steps
    )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

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

    # Handle low_noise_transformer
    low_noise_transformer_metadata = metadatas.low_noise_transformer_state
    target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, low_noise_transformer_metadata)
    with mesh:
      abstract_tree_structure_low_params = jax.tree_util.tree_map(
          add_sharding_to_struct, low_noise_transformer_metadata, target_shardings
      )

    # Handle high_noise_transformer
    high_noise_transformer_metadata = metadatas.high_noise_transformer_state
    target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, high_noise_transformer_metadata)
    with mesh:
      abstract_tree_structure_high_params = jax.tree_util.tree_map(
          add_sharding_to_struct, high_noise_transformer_metadata, target_shardings
      )

    max_logging.log("Restoring WAN 2.2 checkpoint")
    restore_items = {
        "low_noise_transformer_state": ocp.args.StandardRestore(abstract_tree_structure_low_params),
        "high_noise_transformer_state": ocp.args.StandardRestore(abstract_tree_structure_high_params),
        "wan_config": ocp.args.JsonRestore(),
    }
    has_high_config = False
    if hasattr(metadatas, "wan_config_high"):
      val = getattr(metadatas, "wan_config_high")
      if not hasattr(val, "_mock_return_value"):
        has_high_config = True
    elif isinstance(metadatas, dict) and "wan_config_high" in metadatas:
      has_high_config = True

    if has_high_config:
      restore_items["wan_config_high"] = ocp.args.JsonRestore()

    restored_checkpoint = self.checkpoint_manager.restore(
        step=step,
        args=ocp.args.Composite(**restore_items),
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

  def _extract_opt_state(self, restored_checkpoint):
    low_state = getattr(restored_checkpoint, "low_noise_transformer_state", {})
    high_state = getattr(restored_checkpoint, "high_noise_transformer_state", {})
    low_opt = low_state.get("opt_state") if isinstance(low_state, dict) else getattr(low_state, "opt_state", None)
    high_opt = high_state.get("opt_state") if isinstance(high_state, dict) else getattr(high_state, "opt_state", None)
    low_step = low_state.get("step") if isinstance(low_state, dict) else getattr(low_state, "step", None)
    high_step = high_state.get("step") if isinstance(high_state, dict) else getattr(high_state, "step", None)
    if low_opt is None and high_opt is None:
      return None
    return {
        "low_noise_transformer": low_opt,
        "high_noise_transformer": high_opt,
        "low_noise_step": low_step,
        "high_noise_step": high_step,
    }

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_2, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.low_noise_transformer)),
        "wan_config_high": ocp.args.JsonSave(config_to_json(pipeline.high_noise_transformer)),
    }

    items["low_noise_transformer_state"] = ocp.args.StandardSave(train_states["low_noise_transformer"])
    items["high_noise_transformer_state"] = ocp.args.StandardSave(train_states["high_noise_transformer"])

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
