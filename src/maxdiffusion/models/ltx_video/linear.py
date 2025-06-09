from typing import Union, Iterable, Tuple, Optional, Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal


Shape = Tuple[int, ...]
Initializer = Callable[[jax.random.PRNGKey, Shape, jax.numpy.dtype], jax.Array]
InitializerAxis = Union[int, Shape]


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


NdInitializer = Callable[[jax.random.PRNGKey, Shape, jnp.dtype, InitializerAxis, InitializerAxis], jax.Array]
KernelInitializer = Callable[[jax.random.PRNGKey, Shape, jnp.dtype, InitializerAxis, InitializerAxis], jax.Array]


class DenseGeneral(nn.Module):
    """A linear transformation with flexible axes.

    Adapted from https://github.com/AI-Hypercomputer/maxtext/blob/4bf3beaa5e721745427bfed09938427e369c2aaf/MaxText/layers/linears.py#L86

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      use_bias: whether to add bias in linear transformation.
      bias_norm: whether to add normalization before adding bias.
      quant: quantization config, defaults to None implying no quantization.
    """

    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    weight_dtype: jnp.dtype = jnp.float32
    dtype: np.dtype = jnp.float32
    kernel_init: KernelInitializer = lecun_normal()
    kernel_axes: Tuple[Optional[str], ...] = ()
    use_bias: bool = False
    matmul_precision: str = "default"

    bias_init: Initializer = jax.nn.initializers.constant(0.0)

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        def compute_dot_general(inputs, kernel, axis, contract_ind):
            """Computes a dot_general operation that may be quantized."""
            dot_general = jax.lax.dot_general
            matmul_precision = jax.lax.Precision(self.matmul_precision)
            return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)

        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
            kernel_shape,
            self.weight_dtype,
        )
        kernel = jnp.asarray(kernel, self.dtype)

        contract_ind = tuple(range(0, len(axis)))
        output = compute_dot_general(inputs, kernel, axis, contract_ind)

        if self.use_bias:
            bias_axes, bias_shape = (
                self.kernel_axes[-len(features) :],
                kernel_shape[-len(features) :],
            )
            bias = self.param(
                "bias",
                nn.with_logical_partitioning(self.bias_init, bias_axes),
                bias_shape,
                self.weight_dtype,
            )
            bias = jnp.asarray(bias, self.dtype)

            output += bias
        return output
