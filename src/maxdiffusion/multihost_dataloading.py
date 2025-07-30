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

# pylint: disable=unused-import
"""SPMD Multihost Dataloading Utilities.

Adapted from Sholto's:
https://github.com/sholtodouglas/multihost_dataloading
"""
from functools import partial  # pylint: disable=g-importing-member
from typing import Union
from collections.abc import Iterator, Iterable
import tensorflow as tf  # pylint: disable=g-import-not-at-top
import time
import numpy as np

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh

from maxdiffusion import max_logging


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh, global_batch_size: int = 0
) -> tuple[tuple[int, ...], NamedSharding]:
  #Handle sharding for setting a gbs < jax.device_count
  if global_batch_size > 0:
    sharding = NamedSharding(global_mesh, PartitionSpec(*global_mesh.axis_names))
  else:
    sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh, global_batch_size: int = 0, split_axis_index: int = 0) -> jax.Array:
  """Put local sharded array into local devices"""
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh, global_batch_size)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=split_axis_index)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
        f"at {jtu.keystr(path)}"
    ) from array_split_error

  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def get_next_batch_sharded(local_dataset: Iterator, global_mesh: Mesh, global_batch_size: int = 0, split_axis_index: int = 0) -> jax.Array:
  """Splits the host loaded data equally over all devices."""

  SLEEP_TIME = 10
  MAX_DATA_LOAD_ATTEMPTS = 30

  data_load_attempts = 0
  loaded_data_success = False
  while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
    data_load_attempts += 1
    try:
      local_data = next(local_dataset)
      loaded_data_success = True
    except tf.errors.FailedPreconditionError:
      max_logging.log("Failed to get next data batch, retrying")
      time.sleep(SLEEP_TIME)

  # Try one last time, if this fails we will see the full stack trace.
  if not loaded_data_success:
    local_data = local_dataset.next()

  input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=global_mesh, global_batch_size=global_batch_size, split_axis_index=split_axis_index), local_data)

  return input_gdas


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(self, dataloader: Union[tf.data.Dataset, Iterable], global_mesh: Mesh, global_batch_size: int = 0):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    # Handles sharding for when gbs < number of devices
    self.global_batch_size = global_batch_size
    # Use the correct axis for splitting the data across when using global_batch_size
    split_axis_name = max(global_mesh.shape, key=global_mesh.shape.get)
    split_axis_index = 0
    if global_batch_size > 0:
      max_logging.log(f"global_batch_size was set to {global_batch_size}, splitting data across {split_axis_name}.")
      if split_axis_name == "data":
        split_axis_index = 0
      elif split_axis_name == "fsdp":
        split_axis_index = 1
      elif split_axis_name == "tensor":
        split_axis_index = 2
      else:
        raise ValueError(f"Could not find {split_axis_name} to split data over.") 
    self.split_axis_index = split_axis_index
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")

  def reset(self):
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or grain.DataLoader.")

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return get_next_batch_sharded(self.local_iterator, self.global_mesh, self.global_batch_size, self.split_axis_index)
