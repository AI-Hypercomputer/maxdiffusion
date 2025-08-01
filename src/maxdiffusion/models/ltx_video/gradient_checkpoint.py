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
from enum import Enum, auto
from typing import Optional

import jax
from flax import linen as nn

SKIP_GRADIENT_CHECKPOINT_KEY = "skip"


class GradientCheckpointType(Enum):
  """
  Defines the type of the gradient checkpoint we will have

  NONE - means no gradient checkpoint
  FULL - means full gradient checkpoint, wherever possible (minimum memory usage)
  MATMUL_WITHOUT_BATCH - means gradient checkpoint for every linear/matmul operation,
                          except for ones that involve batch dimension - that means that all attention and projection
                          layers will have gradient checkpoint, but not the backward with respect to the parameters
  """

  NONE = auto()
  FULL = auto()
  MATMUL_WITHOUT_BATCH = auto()

  @classmethod
  def from_str(cls, s: Optional[str] = None) -> "GradientCheckpointType":
    """
    Constructs the gradient checkpoint type from a string

    Args:
        s (Optional[str], optional): The name of the gradient checkpointing policy. Defaults to None.

    Returns:
        GradientCheckpointType: The policy that corresponds to the string
    """
    if s is None:
      s = "none"
    return GradientCheckpointType[s.upper()]

  def to_jax_policy(self):
    """
    Converts the gradient checkpoint type to a jax policy
    """
    match self:
      case GradientCheckpointType.NONE:
        return SKIP_GRADIENT_CHECKPOINT_KEY
      case GradientCheckpointType.FULL:
        return None
      case GradientCheckpointType.MATMUL_WITHOUT_BATCH:
        return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

  def apply(self, module: nn.Module) -> nn.Module:
    """
    Applies a gradient checkpoint policy to a module
    if no policy is needed, it will return the module as is

    Args:
        module (nn.Module): the module to apply the policy to

    Returns:
        nn.Module: the module with the policy applied
    """
    policy = self.to_jax_policy()
    if policy == SKIP_GRADIENT_CHECKPOINT_KEY:
      return module
    return nn.remat(  # pylint: disable=invalid-name
        module,
        prevent_cse=False,
        policy=policy,
    )
