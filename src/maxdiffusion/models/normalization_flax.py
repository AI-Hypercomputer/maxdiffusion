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
import jax.numpy as jnp
import flax.linen as nn

class AdaLayerNormContinuous(nn.Module):
    embedding_dim: int
    elementwise_affine: bool = True
    eps: float = 1e-5
    bias: bool = True
    norm_type: str = "layer_norm"
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    @nn.compact
    def __call__(self, x, conditioning_embedding):
        assert self.norm_type == 'layer_norm'
        emb = nn.Dense(self.embedding_dim * 2, use_bias=self.bias)(nn.sigmoid(conditioning_embedding))
        scale, shift = jnp.split(emb, 2, axis=1)
        x = nn.LayerNorm(epsilon=self.eps, use_bias=self.elementwise_affine, use_scale=self.elementwise_affine)(x)
        x *= (1 + scale[:, None, :]) + shift[:, None, :]
        return x

class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """
    embedding_dim: int
    norm_type: str = "layer_norm"
    bias: bool = True
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    @nn.compact
    def __call__(self, x, emb):
        emb = nn.silu(emb)
        emb = nn.Dense(
              3 * self.embedding_dim,
              use_bias=self.bias,
              dtype=self.dtype,
              param_dtype=self.weights_dtype,
              precision=self.precision
            )(emb)
        shift_msa, scale_msa, gate_msa = jnp.split(emb, 3, axis=1)
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({self.norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        return x, gate_msa