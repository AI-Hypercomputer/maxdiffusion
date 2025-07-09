# Copyright 2025 Lightricks Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Lightricks/LTX-Video/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the Torch version available at:
# https://github.com/Lightricks/LTX-Video/tree/main
from dataclasses import field
from typing import Any, Callable, Dict, List, Tuple, Optional

import jax
from flax import linen as nn
from flax.linen import partitioning


class RepeatableCarryBlock(nn.Module):
  """
  Integrates an input module in a jax carry format

  ergo, the module assumes the role of a building block
  and returns both input and output across all blocks
  """

  module: Callable[[Any], nn.Module]
  module_init_args: List[Any]
  module_init_kwargs: Dict[str, Any]

  @nn.compact
  def __call__(self, *args) -> Tuple[jax.Array, None]:
    """
    jax carry-op format of block
    assumes the input contains an input tensor to the block along with kwargs that might be send to the block
    kwargs are assumed to have static role, while the input changes between cycles

    Returns:
        Tuple[jax.Array, None]: Output tensor from the block
    """
    mod = self.module(*self.module_init_args, **self.module_init_kwargs)
    output = mod(*args)
    return output, None


class RepeatableLayer(nn.Module):
  """
  RepeatableLayer will assume a similar role to torch.nn.ModuleList
  with the condition that each block has the same graph, and only the parameters differ

  The compilation in RepeatableLayer will happen only once, in contrast to repeat-graph compilation
  """

  module: Callable[[Any], nn.Module]
  """
    A Callable function for single block construction
    """

  num_layers: int
  """
    The amount of blocks to build
    """

  module_init_args: List[Any] = field(default_factory=list)
  """
    args passed to RepeatableLayer.module callable, to support block construction
    """

  module_init_kwargs: Dict[str, Any] = field(default_factory=dict)
  """
    kwargs passed to RepeatableLayer.module callable, to support block construction
    """

  pspec_name: Optional[str] = None
  """
    Partition spec metadata
    """

  param_scan_axis: int = 0
  """
    The axis that the "layers" will be aggragated on
    eg: if a kernel is shaped (8, 16)
    N layers will be (N, 8, 16) if param_scan_axis=0
    and (8, N, 16) if param_scan_axis=1
    """

  @nn.compact
  def __call__(self, *args):

    scan_kwargs = {}
    if self.pspec_name is not None:
      scan_kwargs["metadata_params"] = {nn.PARTITION_NAME: self.pspec_name}

    initializing = self.is_mutable_collection("params")
    params_spec = self.param_scan_axis if initializing else partitioning.ScanIn(self.param_scan_axis)
    scan_fn = nn.scan(
        RepeatableCarryBlock,
        variable_axes={
            "params": params_spec,
            "cache": 0,
            "intermediates": 0,
            "aqt": 0,
            "_overwrite_with_gradient": 0,
        },  # Separate params per timestep
        split_rngs={"params": True},
        in_axes=(nn.broadcast,) * (len(args) - 1),
        length=self.num_layers,
        **scan_kwargs,
    )
    wrapped_function = scan_fn(self.module, self.module_init_args, self.module_init_kwargs)
    x, _ = wrapped_function(*args)
    return x
