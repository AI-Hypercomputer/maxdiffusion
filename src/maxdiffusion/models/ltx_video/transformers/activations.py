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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal

from diffusers.utils.deprecation_utils import deprecate

from maxdiffusion.models.ltx_video.linear import DenseGeneral, KernelInitializer


ACTIVATION_FUNCTIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    # Mish is not in JAX by default
    "mish": lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
}


@jax.jit
def approximate_gelu(x: jax.Array) -> jax.Array:
  """
  Computes Gaussian Error Linear Unit (GELU) activation function

  Args:
      x (jax.Array): The input tensor

  jax.Array: The output tensor
  """
  # The error function (erf) in GELU asymptotically approaches -1 for very large negative inputs
  # sometimes it results in jnp.nan in jax on TPU's, this prevents this behavior
  if x.dtype in (jax.numpy.float64,):
    x = x.clip(-10, None)
  return jax.nn.gelu(x, approximate=True)


def get_activation(act_fn: str):
  """Returns the activation function from string."""
  act_fn = act_fn.lower()
  if act_fn in ACTIVATION_FUNCTIONS:
    return ACTIVATION_FUNCTIONS[act_fn]
  raise ValueError(f"Unsupported activation function: {act_fn}")


class GELU(nn.Module):
  r"""
  GELU activation function with tanh approximation support with `approximate="tanh"`.

  Parameters:
      dim_in (`int`): The number of channels in the input.
      dim_out (`int`): The number of channels in the output.
      approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
      bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
  """

  dim_in: int
  dim_out: int
  approximate: str = "none"
  bias: bool = True

  kernel_axes: Tuple[Optional[str], ...] = ()
  kernel_init: KernelInitializer = lecun_normal()

  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  def gelu(self, gate: jax.Array) -> jax.Array:
    approximate_to_tanh = self.approximate == "tanh"
    if approximate_to_tanh:
      return approximate_gelu(gate)
    else:
      return jax.nn.gelu(gate, approximate=False)

  @nn.compact
  def __call__(self, hidden_states):
    if self.approximate not in ("none", "tanh"):
      raise ValueError(f"approximate must be 'none' or 'tanh', got {self.approximate}")
    proj = DenseGeneral(
        features=self.dim_out,
        use_bias=self.bias,
        kernel_axes=self.kernel_axes,
        kernel_init=self.kernel_init,
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="proj",
    )
    hidden_states = proj(hidden_states)
    hidden_states = self.gelu(hidden_states)
    return hidden_states


class GEGLU(nn.Module):
  r"""
  A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

  Parameters:
      dim_in (`int`): The number of channels in the input.
      dim_out (`int`): The number of channels in the output.
      bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
  """

  dim_in: int
  dim_out: int
  bias: bool = True

  kernel_axes: Tuple[Optional[str], ...] = ()
  kernel_init: KernelInitializer = lecun_normal()

  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  @nn.compact
  def __call__(self, hidden_states, *args, **kwargs):
    if len(args) > 0 or kwargs.get("scale", None) is not None:
      deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
      deprecate("scale", "1.0.0", deprecation_message)

    proj = DenseGeneral(
        features=self.dim_out * 2,
        use_bias=self.bias,
        kernel_axes=self.kernel_axes,
        kernel_init=self.kernel_init,
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="proj",
    )

    hidden_states = proj(hidden_states)
    hidden_states, gate = jnp.split(hidden_states, 2, axis=-1)
    return hidden_states * jax.nn.gelu(gate, approximate=False)


class ApproximateGELU(nn.Module):
  r"""
  The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).

  Parameters:
      dim_in (`int`): The number of channels in the input.
      dim_out (`int`): The number of channels in the output.
      bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
  """

  dim_in: int
  dim_out: int
  bias: bool = True

  kernel_axes: Tuple[Optional[str], ...] = ()
  kernel_init: KernelInitializer = lecun_normal()

  dtype: jnp.dtype = jnp.float32
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"

  @nn.compact
  def __call__(self, x):
    proj = DenseGeneral(
        features=self.dim_out,
        use_bias=self.bias,
        kernel_axes=self.kernel_axes,
        kernel_init=self.kernel_init,
        matmul_precision=self.matmul_precision,
        weight_dtype=self.weight_dtype,
        dtype=self.dtype,
        name="proj",
    )
    x = proj(x)
    return x * jax.nn.sigmoid(1.702 * x)
