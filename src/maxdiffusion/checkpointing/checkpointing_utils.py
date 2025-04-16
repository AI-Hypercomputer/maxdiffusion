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

from typing import Optional, Any
import jax
import numpy as np
import os

import orbax.checkpoint
from maxdiffusion import max_logging
from etils import epath
from flax.training import train_state
import orbax
import orbax.checkpoint as ocp
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions

STABLE_DIFFUSION_CHECKPOINT = "STABLE_DIFFUSION_CHECKPOINT"
STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"
FLUX_CHECKPOINT = "FLUX_CHECKPOINT"


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    save_interval_steps,
    checkpoint_type: str,
    dataset_type: str = "tf",
    use_async: bool = True,
    orbax_logger: Optional[abstract_logger.AbstractLogger] = None,
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

  if checkpoint_type == FLUX_CHECKPOINT:
    item_names = ("flux_state", "flux_config", "vae_state", "vae_config", "scheduler", "scheduler_config")
  else:
    item_names = (
        "unet_config",
        "vae_config",
        "text_encoder_config",
        "scheduler_config",
        "unet_state",
        "vae_state",
        "text_encoder_state",
        "tokenizer_config",
    )
  if checkpoint_type == STABLE_DIFFUSION_XL_CHECKPOINT or checkpoint_type == FLUX_CHECKPOINT:
    item_names += (
        "text_encoder_2_state",
        "text_encoder_2_config",
    )
  if dataset_type == "grain":
    item_names += ("iter",)

  print("item_names: ", item_names)

  mngr = CheckpointManager(
      p,
      item_names=item_names,
      options=CheckpointManagerOptions(
          create=True, save_interval_steps=save_interval_steps, enable_async_checkpointing=use_async
      ),
      logger=orbax_logger,
  )

  max_logging.log("Checkpoint manager created!")
  return mngr


def load_stable_diffusion_configs(
    config: dict,
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
  max_logging.log("Restoring stable diffusion configs")
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      return None

  restore_args = {
      "unet_config": orbax.checkpoint.args.JsonRestore(),
      "vae_config": orbax.checkpoint.args.JsonRestore(),
      "text_encoder_config": orbax.checkpoint.args.JsonRestore(),
      "scheduler_config": orbax.checkpoint.args.JsonRestore(),
      "tokenizer_config": orbax.checkpoint.args.JsonRestore(),
  }

  if checkpoint_type == STABLE_DIFFUSION_XL_CHECKPOINT or checkpoint_type == FLUX_CHECKPOINT:
    restore_args["text_encoder_2_config"] = orbax.checkpoint.args.JsonRestore()

  return (checkpoint_manager.restore(step, args=orbax.checkpoint.args.Composite(**restore_args)), None)


def load_params_from_path(
    config,
    checkpoint_manager: CheckpointManager,
    unboxed_abstract_params,
    checkpoint_item: str,
    step: Optional[int] = None,
):
  ckptr = ocp.PyTreeCheckpointer()

  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      return None

  ckpt_path = os.path.join(config.checkpoint_dir, str(step), checkpoint_item)
  ckpt_path = epath.Path(ckpt_path)
  ckpt_path = os.path.abspath(ckpt_path)

  restore_args = ocp.checkpoint_utils.construct_restore_args(unboxed_abstract_params)
  restored = ckptr.restore(
      ckpt_path, item={"params": unboxed_abstract_params}, transforms={}, restore_args={"params": restore_args}
  )
  return restored["params"]


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
    checkpoint_item: str,
    enable_single_replica_ckpt_restoring: bool,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    checkpoint_item: the name of the checkpoint item that is being loaded. Ex: vae_state
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitng
      with SingleReplicaArrayHandler

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  max_logging.log(f"loading state for {checkpoint_item}")
  if checkpoint_manager is None:
    max_logging.log("no checkpoint manager, not restoring checkpoint")
    return None
  latest_step = checkpoint_manager.latest_step()
  if latest_step is None:
    return None
  else:
    max_logging.log(f"restoring from this run's directory latest step {latest_step}")
    try:
      if not enable_single_replica_ckpt_restoring:
        item = {checkpoint_item: orbax.checkpoint.args.PyTreeRestore(item=abstract_unboxed_pre_state)}
        return checkpoint_manager.restore(latest_step, args=orbax.checkpoint.args.Composite(**item))

      def map_to_pspec(data):
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
          replica_axis_index=0,
          broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
      )
      ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

      restore_args = jax.tree_util.tree_map(
          map_to_pspec,
          abstract_unboxed_pre_state,
      )
      item = {checkpoint_item: ocp.args.PyTreeRestore(item=abstract_unboxed_pre_state, restore_args=restore_args)}
      return checkpoint_manager.restore(latest_step, args=orbax.checkpoint.args.Composite(**item))
    except:
      max_logging.log(f"could not load {checkpoint_item} from orbax")
      return None
