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

from abc import ABC, abstractmethod
import json

import jax
import numpy as np
from typing import Optional, Tuple, Type
from maxdiffusion.checkpointing.checkpointing_utils import (create_orbax_checkpoint_manager)
from ..pipelines.wan.wan_pipeline import WanPipeline2_1, WanPipeline2_2
from .. import max_logging, max_utils
import orbax.checkpoint as ocp
from etils import epath


WAN_CHECKPOINT = "WAN_CHECKPOINT"


class WanCheckpointer(ABC):
  _SUBCLASS_MAP: dict[str, Type['WanCheckpointer']] = {}

  def __new__(cls, model_key: str, config, checkpoint_type: str = WAN_CHECKPOINT):
    if cls is WanCheckpointer:
      subclass = cls._SUBCLASS_MAP.get(model_key)
      if subclass is None:
          raise ValueError(
              f"Unknown model_key: '{model_key}'. "
              f"Supported keys are: {list(cls._SUBCLASS_MAP.keys())}"
          )
      return super().__new__(subclass)
    else:
      return super().__new__(cls)

  def __init__(self, model_key, config, checkpoint_type: str = WAN_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None

    self.checkpoint_manager: ocp.CheckpointManager = (
        create_orbax_checkpoint_manager(
            self.config.checkpoint_dir,
            enable_checkpointing=True,
            save_interval_steps=1,
            checkpoint_type=checkpoint_type,
            dataset_type=config.dataset_type,
        )
    )

  def _create_optimizer(self, model, config, learning_rate):
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  @abstractmethod
  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    raise NotImplementedError

  @abstractmethod
  def load_diffusers_checkpoint(self):
    raise NotImplementedError

  @abstractmethod
  def load_checkpoint(self, step=None) -> Tuple[Optional[WanPipeline2_1 | WanPipeline2_2], Optional[dict], Optional[int]]:
    raise NotImplementedError

  @abstractmethod
  def save_checkpoint(self, train_step, pipeline, train_states: dict):
    raise NotImplementedError


class WanCheckpointer2_1(WanCheckpointer):

  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest WAN checkpoint step: {step}")
      if step is None:
        max_logging.log("No WAN checkpoint found.")
        return None, None
    max_logging.log(f"Loading WAN checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)
    transformer_metadata = metadatas.wan_state
    abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, transformer_metadata)
    params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_params,
        )
    )

    max_logging.log("Restoring WAN checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            wan_state=params_restore,
            wan_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(f"restored checkpoint wan_state {restored_checkpoint.wan_state.keys()}")
    max_logging.log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.wan_state.keys()}")
    max_logging.log(f"optimizer state saved in attribute self.opt_state {self.opt_state}")
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    pipeline = WanPipeline2_1.from_pretrained(self.config)
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_1, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      max_logging.log("Loading WAN pipeline from checkpoint")
      pipeline = WanPipeline2_1.from_checkpoint(self.config, restored_checkpoint)
      if "opt_state" in restored_checkpoint.wan_state.keys():
        opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default pipeline.")
      pipeline = self.load_diffusers_checkpoint()

    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_1, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
    }

    items["wan_state"] = ocp.args.PyTreeSave(train_states)

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")


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
    abstract_tree_structure_low_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, low_noise_transformer_metadata)
    low_params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_low_params,
        )
    )

    # Handle high_noise_transformer
    high_noise_transformer_metadata = metadatas.high_noise_transformer_state
    abstract_tree_structure_high_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, high_noise_transformer_metadata)
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
    max_logging.log(f"restored checkpoint low_noise_transformer_state {restored_checkpoint.low_noise_transformer_state.keys()}")
    max_logging.log(f"restored checkpoint high_noise_transformer_state {restored_checkpoint.high_noise_transformer_state.keys()}")
    max_logging.log(f"optimizer found in low_noise checkpoint {'opt_state' in restored_checkpoint.low_noise_transformer_state.keys()}")
    max_logging.log(f"optimizer found in high_noise checkpoint {'opt_state' in restored_checkpoint.high_noise_transformer_state.keys()}")
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

WanCheckpointer._SUBCLASS_MAP["wan2.1"] = WanCheckpointer2_1
WanCheckpointer._SUBCLASS_MAP["wan2.2"] = WanCheckpointer2_2

def save_checkpoint_orig(self, train_step, pipeline, train_states: dict):
  """Saves the training state and model configurations."""

  def config_to_json(model_or_config):
    """
    only save the config that is needed and can be serialized to JSON.
    """
    if not hasattr(model_or_config, "config"):
      return None
    source_config = dict(model_or_config.config)

    # 1. configs that can be serialized to JSON
    SAFE_KEYS = [
        "_class_name",
        "_diffusers_version",
        "model_type",
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "qk_norm",
        "eps",
        "image_dim",
        "added_kv_proj_dim",
        "rope_max_seq_len",
        "pos_embed_seq_len",
        "flash_min_seq_length",
        "flash_block_sizes",
        "attention",
        "_use_default_values",
    ]

    # 2. save the config that are in the SAFE_KEYS list
    clean_config = {}
    for key in SAFE_KEYS:
      if key in source_config:
        clean_config[key] = source_config[key]

    # 3. deal with special data type and precision
    if "dtype" in source_config and hasattr(source_config["dtype"], "name"):
      clean_config["dtype"] = source_config["dtype"].name  # e.g 'bfloat16'

    if "weights_dtype" in source_config and hasattr(source_config["weights_dtype"], "name"):
      clean_config["weights_dtype"] = source_config["weights_dtype"].name

    if "precision" in source_config and isinstance(source_config["precision"]):
      clean_config["precision"] = source_config["precision"].name  # e.g. 'HIGHEST'

    return clean_config

  items_to_save = {
      "transformer_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
  }

  items_to_save["transformer_states"] = ocp.args.PyTreeSave(train_states)

  # Create CompositeArgs for Orbax
  save_args = ocp.args.Composite(**items_to_save)

  # Save the checkpoint
  self.checkpoint_manager.save(train_step, args=save_args)
  max_logging.log(f"Checkpoint for step {train_step} saved.")
