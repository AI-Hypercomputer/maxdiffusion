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

import math
from dataclasses import dataclass
from einops import rearrange
import jax
import jax.numpy as jnp
from chex import Array
from jax.typing import DTypeLike
import flax.linen as nn
from ...attention_flax import AttentionOp
from .... import common_types

BlockSizes = common_types.BlockSizes

def rope(pos: Array, dim: int, theta: int) -> Array:
  assert dim % 2 == 0
  scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
  omega = 1.0 / (theta ** scale)
  out = jnp.einsum("...n,d->...nd", pos, omega)
  out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
  out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
  return out.astype(jnp.float32)

class QKNorm(nn.Module):
  dtype: DTypeLike = jnp.bfloat16
  weights_dtype: DTypeLike = jnp.bfloat16

  @nn.compact
  def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
    q = nn.RMSNorm(
      dtype=self.dtype,
      param_dtype=self.weights_dtype
    )(q)
    k = nn.RMSNorm(
      dtype=self.dtype,
      param_dtype=self.weights_dtype
    )(k)
    return q, k

class EmbedND(nn.Module):
  dim: int
  theta: int
  axes_dim: list[int]

  def __call__(self, ids: Array):
    n_axes = ids.shape[-1]
    emb = jnp.concatenate(
      [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], axis=-3,
    )

    return jnp.expand_dims(emb, axis=1)

def timestep_embedding(
    t: Array, dim: int, max_period=10000, time_factor: float = 1000.0
) -> Array:
    """
    Generate timestep embeddings.

    Args:
        t: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        time_factor: Tensor of positional embeddings.

    Returns:
        timestep embeddings.
    """
    breakpoint()
    t = time_factor * t
    half = dim // 2

    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    ).astype(dtype=t.dtype)

    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )

    if jnp.issubdtype(t.dtype, jnp.floating):
        embedding = embedding.astype(t.dtype)

    return embedding


class MLPEmbedder(nn.Module):
  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  
  @nn.compact
  def __call__(self, x: Array) -> Array:

    x = nn.Dense(
      self.hidden_dim,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )(x)
    x = nn.silu(x)
    x = nn.Dense(
      self.hidden_dim,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("heads", "embed")
      )
    )(x)

    return x

@dataclass
class ModulationOut:
  shift: Array
  scale: Array
  gate: Array

class Modulation(nn.Module):
  dim: int
  double: bool
  dtype: DTypeLike = jnp.bfloat16
  weights_dtype: DTypeLike = jnp.bfloat16
  precision: jax.lax.Precision = None
  
  @nn.compact
  def __call__(self, vec: Array) -> tuple[ModulationOut, ModulationOut | None]:
    multiplier = 6 if self.double else 3
    lin = nn.Dense(
      multiplier * self.dim,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )(nn.silu(vec))
    out = jnp.split(lin[:, None, :], multiplier, axis=-1)

    return (
      ModulationOut(*out[:3]),
      ModulationOut(*out[3:] if self.double else None)
    )

class DoubleStreamBlock(nn.Module):
  hidden_size: int
  num_heads: int
  mlp_ratio: float
  attention_head_dim: int = 128
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  qkv_bias: bool = False
  attention_kernel: str = "dot_product"

  @nn.compact
  def __call__(self, img: Array, txt: Array, vec: Array, pe: Array) -> tuple[Array, Array]:
    
    mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
    
    img_mod1, img_mod2 = Modulation(
      self.hidden_size,
      double=True,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    )(vec)

    txt_mod1, txt_mod2 = Modulation(
      self.hidden_size,
      double=True,
      dtype=self.dtype,
      weights_dtype=self.weights_dtype,
      precision=self.precision
    )(vec)

    # prepare image for attention
    img_modulated = nn.LayerNorm(
      use_scale=False,
      use_bias=False,
      epsilon=1e-6,
      dtype=self.dtype,
      param_dtype=self.weights_dtype
    )(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = nn.Dense(
      self.hidden_size * 3,
      use_bias=self.qkv_bias,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )(img_modulated)
    img_q, img_k, img_v = rearrange(
      img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
    )
    img_q, img_k = QKNorm(
      dtype=self.dtype,
      weights_dtype=self.weights_dtype
    )(img_q, img_k, img_v)

    # prepare text for attention
    txt_modulated = nn.LayerNorm(
      use_scale=False,
      use_bias=False,
      epsilon=1e-6,
      dtype=self.dtype,
      param_dtype=self.weights_dtype
    )(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = nn.Dense(
      self.hidden_size * 3,
      use_bias=self.qkv_bias,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("embed", "heads")
      )
    )(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(
      txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
    )
    txt_q, txt_k = QKNorm(
      dtype=self.dtype,
      weights_dtype=self.weights_dtype
    )(txt_q, txt_k, txt_v)

    # run actual attention
    q = jnp.concatenate((txt_q, img_q), axis=2)
    k = jnp.concatenate((txt_k, img_k), axis=2)
    v = jnp.concatenate((txt_v, img_v), axis=2)

    attn = AttentionOp(
      mesh=self.mesh,
      attention_kernel=self.attention_kernel,
      scale=self.attention_head_dim**-0.5,
      heads=self.num_heads,
      dim_head=self.attention_head_dim,
      flash_min_seq_length=self.flash_min_seq_length,
      use_memory_efficient_attention=False,
      split_head_dim=True,
      flash_block_sizes=self.flash_block_sizes,
      dtype=self.dtype
    )(q, k, v)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    #calculate the img blocks
    img = img + img_mod1.gate * nn.Dense(
      self.hidden_size,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("heads", "embed")
      ),
    )(img_attn)
    img = img + img_mod2.gate * nn.Sequential(
      [
        nn.Dense(
          mlp_hidden_dim,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("embed", "heads")
          )
        ),
        nn.gelu,
        nn.Dense(
          self.hidden_size,
          use_bias=True,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("heads", "embed")
          )
        )
      ]
    )(
      (1 + img_mod2.scale) * nn.LayerNorm(
        use_scale=False,
        use_bias=False,
        param_dtype=self.weights_dtype
      )(img) + img_mod2.shift
    )
    
    # calculate the txt blocks
    txt = txt + txt_mod1.gate * nn.Dense(
      self.hidden_size,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.weights_dtype,
      precision=self.precision,
      kernel_init=nn.with_logical_partitioning(
        nn.initializers.lecun_normal(),
        ("heads", "embed")
      ),
    )(txt_attn)
    txt = txt + txt_mod2.gate * nn.Sequential(
      [
        nn.Dense(
          mlp_hidden_dim,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("embed", "heads")
          )
        ),
        nn.gelu,
        nn.Dense(
          self.hidden_size,
          use_bias=True,
          param_dtype=self.weights_dtype,
          precision=self.precision,
          kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("heads", "embed")
          )
        )
      ]
    )(
      (1 + txt_mod2.scale) * nn.LayerNorm(
        use_scale=False,
        use_bias=False,
        param_dtype=self.weights_dtype
      )(txt) + txt_mod2.shift
    )

    return img, txt
    
