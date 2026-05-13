"""
Copyright 2026 Google LLC

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

import jax
from enum import Enum


def print_device_memory_info(devices):
  """Prints information about JAX devices.

  Args:
    devices: The JAX device or list of jax.Device to print information about.
  """
  if not isinstance(devices, list):
    devices = [devices]

  for device in devices:
    id = device.id
    device_kind = device.device_kind
    platform = device.platform
    memory_stats = device.memory_stats()
    jax.debug.print("**** device_id: {x}", x=id)
    jax.debug.print("\tdevice_kind: {x}", x=device_kind)
    jax.debug.print("\tplatform: {x}", x=platform)
    for key in memory_stats.keys():
      jax.debug.print("\t{x} : {y}", x=key, y=memory_stats[key])


def print_array_info(array, name):
  """Prints information about a JAX array.

  Note: This function is intended for use with concrete JAX arrays outside of
  JIT-compiled contexts. Calling this function inside a `@jax.jit` (or similar)
  context will fail because `array` will be a Tracer object, which does not
  have attributes like `is_fully_replicated` or `device_buffers`.

  Args:
    array: The JAX array (jax.Array) to print information about.
    name: (str) A name to identify the array in the output.
  """
  print("**** name: ", name)
  jax.debug.print("dtype: {x}", x=array.dtype)
  jax.debug.print("shape: {x}", x=array.shape)
  jax.debug.print("is fully replicated: {x}", x=array.is_fully_replicated)
  for device_idx, buffer in enumerate(array.device_buffers):
    jax.debug.print("shape on device {x} : {y}", x=device_idx, y=buffer.shape)
    jax.debug.print("size on device {x} : {y}", x=device_idx, y=buffer.size / array.size)


class TpuType(Enum):
  TPU_V6_LITE = "v6e"
  TPU_7X = "v7x"
  UNKNOWN = "unknown"


def get_tpu_type() -> TpuType:
  """Detects the current TPU hardware generation."""
  try:
    device_kind = jax.devices()[0].device_kind
    if "7x" in device_kind:
      return TpuType.TPU_7X
    elif "v6 lite" in device_kind:
      return TpuType.TPU_V6_LITE
    else:
      return TpuType.UNKNOWN
  except Exception:
    return TpuType.UNKNOWN
