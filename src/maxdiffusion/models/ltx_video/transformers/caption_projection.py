from flax import linen as nn
import jax.numpy as jnp

from maxdiffusion.models.ltx_video.linear import DenseGeneral
from maxdiffusion.models.ltx_video.transformers.activations import approximate_gelu


class CaptionProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    """

    in_features: int
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    weight_dtype: jnp.dtype = jnp.float32
    matmul_precision: str = "default"

    @nn.compact
    def __call__(self, caption):
        hidden_states = DenseGeneral(
            self.hidden_size,
            use_bias=True,
            kernel_axes=("embed", None),
            matmul_precision=self.matmul_precision,
            weight_dtype=self.weight_dtype,
            dtype=self.dtype,
            name="linear_1",
        )(caption)
        hidden_states = approximate_gelu(hidden_states)
        hidden_states = DenseGeneral(
            self.hidden_size,
            use_bias=True,
            kernel_axes=("embed", None),
            matmul_precision=self.matmul_precision,
            weight_dtype=self.weight_dtype,
            dtype=self.dtype,
            name="linear_2",
        )(hidden_states)
        return hidden_states
