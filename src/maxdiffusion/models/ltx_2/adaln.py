
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Tuple
from maxdiffusion.models.embeddings_flax import NNXPixArtAlphaCombinedTimestepSizeEmbeddings

class AdaLayerNormSingle(nnx.Module):
    """
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    """
    def __init__(self, rngs: nnx.Rngs, embedding_dim: int, embedding_coefficient: int = 6, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32):
        self.emb = NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
            rngs=rngs,
            embedding_dim=embedding_dim,
            size_emb_dim=embedding_dim // 3,
            dtype=dtype,
            weights_dtype=weights_dtype
        )
        self.silu = nnx.silu
        self.linear = nnx.Linear(
            rngs=rngs,
            in_features=embedding_dim,
            out_features=embedding_coefficient * embedding_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros
        )

    def __call__(self, timestep: jax.Array, hidden_dtype: Optional[jnp.dtype] = None) -> Tuple[jax.Array, jax.Array]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
