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
import optax
from typing import Optional, Tuple
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
from maxdiffusion import max_logging, max_utils
from maxdiffusion.checkpointing.checkpointing_utils import LTX2_CHECKPOINT, create_orbax_checkpoint_manager
import orbax.checkpoint as ocp
from etils import epath


class LTX2Checkpointer:

  def __init__(self, config, checkpoint_type: str = LTX2_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None

    checkpoint_dir = getattr(self.config, "checkpoint_dir", "") or ""
    raw_use_async = getattr(self.config, "enable_async_checkpointing", True)
    use_async = raw_use_async if isinstance(raw_use_async, bool) else True

    self.checkpoint_manager: Optional[ocp.CheckpointManager] = create_orbax_checkpoint_manager(
        checkpoint_dir,
        enable_checkpointing=bool(checkpoint_dir),
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=getattr(config, "dataset_type", None),
        use_async=use_async,
    )

  def _create_optimizer(self, config, learning_rate: float):
    schedule_type = getattr(config, "lr_schedule_type", "constant")
    total_steps = int(getattr(config, "learning_rate_schedule_steps", -1))
    if total_steps <= 0:
      total_steps = int(config.max_train_steps)
    warmup_steps = int(total_steps * float(getattr(config, "warmup_steps_fraction", 0.0)))

    if schedule_type == "cosine" and total_steps > warmup_steps and total_steps > 0:
      learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=learning_rate,
          warmup_steps=warmup_steps,
          decay_steps=max(total_steps, 1),
          end_value=float(getattr(config, "lr_end_value", 0.0)),
      )
    else:
      learning_rate_scheduler = max_utils.create_learning_rate_schedule(
          learning_rate,
          total_steps,
          float(getattr(config, "warmup_steps_fraction", 0.0)),
          int(config.max_train_steps),
      )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  def load_ltx2_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if self.checkpoint_manager is None:
      max_logging.log("No checkpoint manager configured, skipping Orbax load.")
      return None, None

    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest LTX2 checkpoint step: {step}")
      if step is None:
        max_logging.log("No LTX2 checkpoint found.")
        return None, None
    max_logging.log(f"Loading LTX2 checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)
    transformer_metadata = metadatas.ltx2_state
    abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, transformer_metadata)
    params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_params,
        )
    )

    max_logging.log("Restoring LTX2 checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            ltx2_state=params_restore,
            ltx2_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(f"restored checkpoint ltx2_state {restored_checkpoint.ltx2_state.keys()}")
    max_logging.log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.ltx2_state.keys()}")
    return restored_checkpoint, step

  def load_checkpoint(
      self, step=None, vae_only=False, load_transformer=True, load_upsampler=False
  ) -> Tuple[LTX2Pipeline, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_ltx2_configs_from_orbax(step)
    opt_state = None

    if restored_checkpoint:
      max_logging.log("Loading LTX2 pipeline from checkpoint")
      pipeline = LTX2Pipeline.from_checkpoint(self.config, restored_checkpoint, vae_only, load_transformer, load_upsampler)
      if "opt_state" in restored_checkpoint.ltx2_state.keys():
        opt_state = restored_checkpoint.ltx2_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading pipeline from pretrained hub")
      pipeline = LTX2Pipeline.from_pretrained(self.config, vae_only, load_transformer, load_upsampler)

    return pipeline, opt_state, step

  def save_checkpoint(
      self,
      train_step,
      pipeline: LTX2Pipeline,
      train_states: dict,
      transformer_config_dict: Optional[dict] = None,
  ):
    """Saves the training state and model configurations."""
    if self.checkpoint_manager is None:
      max_logging.log("No checkpoint manager configured, skipping checkpoint save.")
      return

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    cfg_payload = transformer_config_dict if transformer_config_dict is not None else config_to_json(pipeline.transformer)
    items = {
        "ltx2_config": ocp.args.JsonSave(cfg_payload),
        "ltx2_state": ocp.args.PyTreeSave(train_states),
    }

    # Save the checkpoint
    self.checkpoint_manager.save(int(train_step), args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
