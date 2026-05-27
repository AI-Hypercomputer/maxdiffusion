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
from typing import Optional, Tuple
import jax
from flax import nnx
from maxdiffusion.checkpointing.checkpointing_utils import (
    add_sharding_to_struct,
    create_orbax_checkpoint_manager,
    get_cpu_mesh_and_sharding,
)
from ..pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
from ..pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
from ..pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
from ..pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2
from .. import max_logging, max_utils
import orbax.checkpoint as ocp


WAN_CHECKPOINT = "WAN_CHECKPOINT"


class WanCheckpointer(ABC):

  def __init__(self, config, checkpoint_type: str = WAN_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None

    self.checkpoint_manager: ocp.CheckpointManager = create_orbax_checkpoint_manager(
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
    return tx, learning_rate_scheduler

  @classmethod
  def load_pretrained_pipeline_or_diffusers(
      cls, config, pipeline_cls, pretrained_state_sources, pretrained_config_transformer_attr
  ):
    """Load a WAN pipeline from the pretrained Orbax cache, or seed it from diffusers.

    This helper is used only for inference when no training checkpoint exists.
    `pretrained_config_transformer_attr` is explicit because WAN 2.2 has separate
    transformer states but still saves one `wan_config`, matching the existing
    training checkpoint format.
    """
    pretrained_dir = getattr(config, "pretrained_orbax_dir", "")
    if pretrained_dir:
      restored_checkpoint = cls._restore_pretrained_checkpoint(
          pretrained_dir, tuple(state_item_name for state_item_name, _ in pretrained_state_sources)
      )
      if restored_checkpoint is not None:
        max_logging.log(f"Loading WAN pipeline from pretrained orbax checkpoint at {pretrained_dir}")
        return pipeline_cls.from_checkpoint(config, restored_checkpoint)

    max_logging.log("No checkpoint found, loading default pipeline.")
    pipeline = pipeline_cls.from_pretrained(config)
    if pretrained_dir:
      cls._save_pretrained_checkpoint(pretrained_dir, pipeline, pretrained_state_sources, pretrained_config_transformer_attr)
    return pipeline

  @classmethod
  def _restore_pretrained_checkpoint(cls, pretrained_dir: str, state_item_names: Tuple[str, ...]):
    """Restore pretrained WAN transformer states and config from an Orbax cache."""
    try:
      checkpoint_manager = create_orbax_checkpoint_manager(
          pretrained_dir,
          enable_checkpointing=True,
          save_interval_steps=1,
          checkpoint_type=WAN_CHECKPOINT,
          use_async=False,
      )
      step = checkpoint_manager.latest_step()
      if step is None:
        max_logging.log(f"No pretrained orbax checkpoint found in {pretrained_dir}")
        return None

      max_logging.log(f"Found pretrained orbax checkpoint step {step} in {pretrained_dir}")
      metadatas = checkpoint_manager.item_metadata(step)
      mesh, replicated_sharding = get_cpu_mesh_and_sharding()
      restore_items = {"wan_config": ocp.args.JsonRestore()}
      for state_item_name in state_item_names:
        restore_items[state_item_name] = cls._standard_restore_arg(
            getattr(metadatas, state_item_name), mesh, replicated_sharding
        )
      return checkpoint_manager.restore(step=step, args=ocp.args.Composite(**restore_items))
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Failed to load pretrained orbax checkpoint from {pretrained_dir}: {e}")
      return None

  @staticmethod
  def _standard_restore_arg(metadata, mesh, replicated_sharding):
    target_shardings = jax.tree_util.tree_map(lambda _: replicated_sharding, metadata)
    with mesh:
      abstract_state = jax.tree_util.tree_map(add_sharding_to_struct, metadata, target_shardings)
    return ocp.args.StandardRestore(abstract_state)

  @classmethod
  def _save_pretrained_checkpoint(
      cls, pretrained_dir: str, pipeline, pretrained_state_sources, pretrained_config_transformer_attr
  ):
    """Save pretrained WAN transformer states to the inference-only Orbax cache."""
    try:
      max_logging.log(f"Saving pretrained WAN weights to orbax at {pretrained_dir}")
      checkpoint_manager = create_orbax_checkpoint_manager(
          pretrained_dir,
          enable_checkpointing=True,
          save_interval_steps=1,
          checkpoint_type=WAN_CHECKPOINT,
          use_async=False,
      )
      save_items = cls._pretrained_save_items(pipeline, pretrained_state_sources, pretrained_config_transformer_attr)
      checkpoint_manager.save(0, args=ocp.args.Composite(**save_items))
      checkpoint_manager.wait_until_finished()
      max_logging.log(f"Pretrained weights saved to {pretrained_dir}")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Failed to save pretrained orbax checkpoint to {pretrained_dir}: {e}")

  @staticmethod
  def _pretrained_save_items(pipeline, pretrained_state_sources, pretrained_config_transformer_attr):
    """Build Orbax save args for pretrained WAN transformer states.

    `pretrained_state_sources` contains `(orbax_item_name, pipeline_attribute)` pairs.
    `pretrained_config_transformer_attr` names the transformer whose config should be
    serialized as `wan_config`.
    """
    pretrained_state_sources = tuple(pretrained_state_sources)
    if not pretrained_state_sources:
      raise ValueError("pretrained_state_sources must contain at least one transformer source.")

    try:
      config_transformer = getattr(pipeline, pretrained_config_transformer_attr)
    except AttributeError as e:
      raise ValueError(
          f"Pipeline does not have pretrained config transformer attribute `{pretrained_config_transformer_attr}`."
      ) from e

    items = {}
    for state_item_name, transformer_attr in pretrained_state_sources:
      try:
        transformer = getattr(pipeline, transformer_attr)
      except AttributeError as e:
        raise ValueError(f"Pipeline does not have pretrained transformer attribute `{transformer_attr}`.") from e

      _, state, _ = nnx.split(transformer, nnx.Param, ...)
      items[state_item_name] = ocp.args.StandardSave(state.to_pure_dict())

    items["wan_config"] = ocp.args.JsonSave(json.loads(config_transformer.to_json_string()))
    return items

  @abstractmethod
  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    raise NotImplementedError

  @abstractmethod
  def load_diffusers_checkpoint(self):
    raise NotImplementedError

  @abstractmethod
  def load_checkpoint(
      self, step=None
  ) -> Tuple[
      Optional[WanPipeline2_1 | WanPipeline2_2 | WanPipelineI2V_2_1 | WanPipelineI2V_2_2], Optional[dict], Optional[int]
  ]:
    raise NotImplementedError

  @abstractmethod
  def save_checkpoint(self, train_step, pipeline, train_states: dict):
    raise NotImplementedError


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
