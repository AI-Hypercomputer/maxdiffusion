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

import jax


def print_device_memory_info(devices):
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
  print("**** name: ", name)
  jax.debug.print("dtype: {x}", x=array.dtype)
  jax.debug.print("shape: {x}", x=array.shape)
  jax.debug.print("is fully replicated: {x}", x=array.is_fully_replicated)
  num_devices = jax.device_count()
  for device_idx in num_devices:
    jax.debug.print("shape on device {x} : {y}", x=device_idx, y=array.device_buffers[0].shape)
    jax.debug.print("size on device {x} : {y}", x=device_idx, y=array.device_buffers[device_idx].size / array.size)
