from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from maxdiffusion import common_types
from . import initializers
from . import quantizations
import numpy as np

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

Quant = quantizations.AqtQuantization


def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init()
  kernel_axes: Tuple[str, ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    def compute_dot_general(inputs, kernel, axis, contract_ind):
      """Computes a dot_general operation that may be quantized."""
      dot_general = lax.dot_general
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls()
        dot_general = dot_general_cls()
      return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      kernel = jnp.zeros(kernel_shape)
    else:
      kernel = self.param(
          "kernel",
          nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    output = compute_dot_general(inputs, kernel, axis, contract_ind)

    if self.use_bias:
      bias_axes, bias_shape = self.kernel_axes[-len(features) :], kernel_shape[-len(features) :]
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output