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
    "unet_params",
    "vae_state",
    "vae_params",
    "text_encoder_state",
    "text_encoder_params",
    "text_encoder_2_state",
    "text_encoder_2_params"
  )
  if checkpoint_type == "sdxl":
    item_names + ("text_encoder_2_params", "text_encoder_2_state", "text_encoder_2_config")
  
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

def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)

def load_state_if_possible(
  checkpoint_manager: CheckpointManager,
  abstract_unboxed_pre_state: train_state.TrainState,
  enable_single_replica_ckpt_restoring: Optional[bool] = False,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    first_checkpoint_path: if there is no checkpoint in the checkpoint manager,
      return the Params from the first_checkpoint_path if they exist. This
      enables loading just the parameters and is intended for finetuning.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    mesh: a physical TPU mesh
    state_mesh_annotation: a PyTree of sharding rules, matching
      abstract_unboxed_pre_state.

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  if checkpoint_manager is None:
    max_logging.log("no checkpoint manager, not restoring checkpoint")
    return None, None
  
  latest_step = checkpoint_manager.latest_step()
  if latest_step is not None:
    max_logging.log(
      f"restoring from this run's directory latest step {latest_step}"
    )

    def map_to_pspec(data):
      pspec = data.sharding.spec
      mesh = data.sharding.mesh
      if not enable_single_replica_ckpt_restoring:
        return orbax.checkpoint.type_handlers.ArrayRestoreArgs(
          mesh=mesh, mesh_axes = pspec
        )
      replicate_axis_index = 0
      replica_devices = _replica_devices(mesh.devices, replicate_axis_index)
      replicate_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
      single_replica_sharding = jax.sharding.NamedSharding(
        replicate_mesh, pspec
      )

      array_handler = (
        orbax.checkpoint.type_handlers.SingleReplicaArrayHandler(
          replica_axis_index=0,
          broadcast_memory_limit_bytes=1024 * 1024 * 1000  # 1000 MB limit
        )
      )
      orbax.checkpoint.type_handler.register_type_handler(
        jax.Array,
        array_handler,
        override=True
      )

      return orbax.checkpoint.type_handlers.SingleReplicaArrayRestoreArgs(
        sharding=jax.sharding.NamedSharding(mesh, pspec),
        single_replica_sharding=single_replica_sharding,
        global_shape=data.shape,
        dtype=data.dtype
      )
    
    restore_args = jax.tree_util.tree_map(
      map_to_pspec,
      abstract_unboxed_pre_state
    ) 

    return (
      checkpoint_manager.restore(
        latest_step,
        args=orbax.checkpoint.args.Composite(
          unet_state=orbax.checkpoint.args.PyTreeRestore(item=abstract_unboxed_pre_state, restore_args=restore_args)
        )
      ),
      None,
    )

def validate_checkpoint(
  pipeline,
  params,
  unet_state,
  vae_state,
  text_encoder_state,
  text_encoder_2_state,
):
  
  assert unet_state is not None or params["unet"] is not None, \
    'At least unet_state or params["unet"] must not be none.'
  assert vae_state is not None or params["vae"] is not None, \
    'At least vae_state or params["vae"] must not be none.'
  assert text_encoder_state is not None or params["text_encoder"] is not None, \
    'At least text_encoder_state or params["text_encoder"] must not be none.'

  if hasattr(pipeline, "text_encoder_2"):
    assert text_encoder_2_state is not None or params["text_encoder_2"] is not None, \
      'At least text_encoder_2_state or params["text_encoder_2"] must not be none.'


  

def save_checkpoint(
  checkpoint_manager,
  step,
  pipeline,
  params,
  unet_state = None,
  vae_state = None,
  text_encoder_state = None,
  text_encoder_2_state = None,
):
  validate_checkpoint(pipeline, params, unet_state, vae_state, text_encoder_state, text_encoder_2_state)

  def config_to_json(model_or_config):
    return json.loads(model_or_config.to_json_string())
  items = {
    'unet_config' : ocp.args.JsonSave(config_to_json(pipeline.unet)),
    'vae_config' : ocp.args.JsonSave(config_to_json(pipeline.vae)),
    'text_encoder_config' : ocp.args.JsonSave(config_to_json(pipeline.text_encoder.config)),
    "scheduler_config" : ocp.args.JsonSave(config_to_json(pipeline.scheduler))
  }
  
  if unet_state:
    items['unet_state'] = ocp.args.StandardSave(unet_state)
  else:
    items['unet_params'] = ocp.args.StandardSave(params["unet"])

  if vae_state:
    items['vae_state'] = ocp.args.StandardSave(vae_state)
  else:
    items['vae_params'] = ocp.args.StandardSave(params["vae"])
  
  if text_encoder_state:
    items["text_encoder_state"] = ocp.args.StandardSave(text_encoder_state)
  else:
    items["text_encoder_params"] = ocp.args.StandardSave(params["text_encoder"])

  if hasattr(pipeline, "text_encoder_2"):
    if text_encoder_2_state:
      items["text_encoder_state_2"] = ocp.args.StandardSave(text_encoder_2_state)
    else:
      items["text_encoder_2_params"] = ocp.args.StandardSave(params["text_encoder_2"])
    items["text_encoder_2_config"] = ocp.args.JsonSave(config_to_json(pipeline.text_encoder_2.config))

  checkpoint_manager.save(step, args=ocp.args.Composite(
    **items
  ))