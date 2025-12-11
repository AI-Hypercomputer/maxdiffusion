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

from typing import Union, Tuple, Optional
import jax.numpy as jnp
from flax import nnx

class BaseLoRALayer(nnx.Module):
  """
  Base LoRA layer class for all NNX LoRA layer implementations.
  Stores common LoRA attributes.
  """
  def __init__(self, rank: int, scale: float = 1.0, verbose: bool = False):
    self.rank = rank
    self.scale = scale
    self.verbose = verbose

  def scaling(self) -> float:
    return self.scale


class LoRALinear(BaseLoRALayer):
  """
  Implements LoRA for a Linear layer in NNX.
  Wraps an existing nnx.Linear layer in-place.
  """
  def __init__(
      self,
      base_layer: nnx.Linear,
      rank: int,
      rngs: nnx.Rngs,
      scale: float = 1.0,
      dtype=jnp.float32
  ):
    super().__init__(rank, scale)
    self.base_layer = base_layer  # Keep reference to frozen base layer
    
    # Infer dimensions from the base layer
    # nnx.Linear stores weights in 'kernel' usually shaped (in_features, out_features)
    in_features, out_features = base_layer.kernel.shape

    # 1. Down Projection (A): Random Initialization
    # Projects inputs down to rank 'r'
    self.A = nnx.Param(
        nnx.initializers.kaiming_uniform()(rngs.params(), (in_features, rank), dtype)
    )

    # 2. Up Projection (B): Zero Initialization
    # Projects back up to output dimension.
    # Must be zero so the layer starts as Identity (no change to base model).
    self.B = nnx.Param(
        jnp.zeros((rank, out_features), dtype=dtype)
    )

  def __call__(self, x):
    # 1. Compute frozen base output
    base_out = self.base_layer(x)

    # 2. Compute trainable LoRA path
    # Equation: (x @ A @ B) * scaling
    lora_out = (x @ self.A.value @ self.B.value) * self.scaling()

    # 3. Sum
    return base_out + lora_out


class LoRAConv(BaseLoRALayer):
  """
  Implements LoRA for a Conv layer in NNX.
  Wraps an existing nnx.Conv layer in-place.
  """
  def __init__(
      self,
      base_layer: nnx.Conv,
      rank: int,
      rngs: nnx.Rngs,
      scale: float = 1.0,
      dtype=jnp.float32
  ):
    super().__init__(rank, scale)
    self.base_layer = base_layer

    # Extract configuration from base layer to ensure compatibility
    kernel_size = base_layer.kernel_size
    strides = base_layer.strides
    padding = base_layer.padding
    input_dilation = base_layer.input_dilation
    kernel_dilation = base_layer.kernel_dilation
    feature_group_count = base_layer.feature_group_count
    
    in_features = base_layer.features_in
    out_features = base_layer.features_out

    # 1. Down Projection: Standard Convolution (reduces channels to rank)
    self.down = nnx.Conv(
        in_features=in_features,
        features=rank,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        input_dilation=input_dilation,
        kernel_dilation=kernel_dilation,
        feature_group_count=feature_group_count,
        use_bias=False,
        kernel_init=nnx.initializers.kaiming_uniform(),
        rngs=rngs,
        dtype=dtype
    )

    # 2. Up Projection: 1x1 Convolution (restores channels to out_features)
    # Must be zero-initialized
    self.up = nnx.Conv(
        in_features=rank,
        features=out_features,
        kernel_size=(1, 1), # 1x1 conv preserves spatial dims
        strides=(1, 1),
        use_bias=False,
        kernel_init=nnx.initializers.zeros, # Zero init is crucial
        rngs=rngs,
        dtype=dtype
    )

  def __call__(self, x):
    # 1. Base Output
    base_out = self.base_layer(x)

    # 2. LoRA Output
    lora_out = self.up(self.down(x)) * self.scaling()

    return base_out + lora_out


# -----------------------------------------------------------------------------
# Helper: The "Discovery" Logic (Graph Transformation)
# -----------------------------------------------------------------------------

def inject_lora(
    model: nnx.Module, 
    rank: int, 
    rngs: nnx.Rngs, 
    scale: float = 1.0,
    target_linear: bool = True, 
    target_conv: bool = False
):
    """
    Traverses the NNX model graph and replaces target layers with LoRA wrappers.
    This modifies the 'model' object in-place.
    """
    for path, module in nnx.iter_graph(model):
        parent = path[-2]
        attr_name = path[-1]

        # Handle Linear Layers
        if target_linear and isinstance(module, nnx.Linear):
            # Do not wrap if it's already wrapped (sanity check)
            if isinstance(parent, BaseLoRALayer): 
                continue

            print(f"Injecting LoRA (Linear) at {'.'.join([str(p) for p in path])}")
            wrapper = LoRALinear(base_layer=module, rank=rank, scale=scale, rngs=rngs)
            setattr(parent, attr_name, wrapper)

        # Handle Conv Layers
        elif target_conv and isinstance(module, nnx.Conv):
            if isinstance(parent, BaseLoRALayer):
                continue

            print(f"Injecting LoRA (Conv) at {'.'.join([str(p) for p in path])}")
            wrapper = LoRAConv(base_layer=module, rank=rank, scale=scale, rngs=rngs)
            setattr(parent, attr_name, wrapper)

    return model