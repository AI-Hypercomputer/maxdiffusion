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

import jax
import jax.numpy as jnp
from flax import nnx
from .... import common_types, max_logging
from ...modeling_flax_utils import FlaxModelMixin
from ....configuration_utils import ConfigMixin

BlockSizes = common_types.BlockSizes

class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):
  def __init__(
    self,
    rngs: nnx.Rngs,
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2038,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=16,
    num_layers=32,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    flash_min_seq_length: int = 4096,
    flash_block_sizes: BlockSizes = None,
    mesh: jax.sharding.Mesh = None,
    dtype: jnp.dtype = jnp.float32,
    weights_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = None,
    attention: str = "dot_product",
  ):
    self.path_embedding = nnx.Conv(
      in_dim,
      dim,
      kernel_size=patch_size,
      strides=patch_size,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.with_partitioning(
        nnx.initializers.xavier_uniform(),
        ("batch",)
      ),
      rngs=rngs
    )

  def __call__(self, x):
    x = self.path_embedding(x)
    return x
