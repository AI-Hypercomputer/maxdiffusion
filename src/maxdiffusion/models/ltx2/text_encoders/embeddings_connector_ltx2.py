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

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion import common_types
from maxdiffusion.models.ltx2.attention_ltx2 import LTX2Attention
from maxdiffusion.models.attention_flax import NNXSimpleFeedForward

Array = common_types.Array
DType = common_types.DType


class _BasicTransformerBlock1D(nnx.Module):

  def __init__(
      self,
      dim: int,
      heads: int,
      dim_head: int,
      rope_type: str = "interleaved",
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
  ):
    self.attn1 = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rope_type=rope_type,
        bias=True,  # LTX-2 default
        out_bias=True,
        attention_kernel=attention_kernel,
        mesh=mesh,
        rngs=rngs,
    )
    self.ff = NNXSimpleFeedForward(rngs=rngs, dim=dim, dim_out=dim)
    self.norm1 = nnx.RMSNorm(dim, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=True, rngs=rngs)
    self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=True, rngs=rngs)

  def __call__(
      self,
      hidden_states: Array,
      attention_mask: Optional[Array] = None,
      rotary_emb: Optional[Tuple[Array, Array]] = None,
  ) -> Array:
    # 1. Norm -> Attention
    normed = self.norm1(hidden_states)
    attn_output = self.attn1(normed, attention_mask=attention_mask, rotary_emb=rotary_emb)
    hidden_states = hidden_states + attn_output

    # 2. Norm -> FeedForward
    normed = self.norm2(hidden_states)
    ff_output = self.ff(normed)
    hidden_states = hidden_states + ff_output

    return hidden_states


class Embeddings1DConnector(nnx.Module):
  """
  Applies 1D transformer processing with Thinking Tokens (Learnable Registers).
  Uses nnx.scan for efficient JAX-idiomatic layer execution.
  """

  def __init__(
      self,
      input_dim: int,
      heads: int = 30,
      head_dim: int = 128,
      layers: int = 2,
      theta: float = 10000.0,
      num_learnable_registers: int = 128,
      rope_type: str = "interleaved",
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
  ):
    self.dim = input_dim
    self.theta = theta
    self.num_learnable_registers = num_learnable_registers
    self.num_layers = layers

    # 1. Initialize Stacked Layers using vmap
    # This creates a single module where parameters have an extra leading dimension [layers, ...]
    # We need to ensure rngs are split for each layer
    @nnx.split_rngs(splits=layers)
    @nnx.vmap(in_axes=0, out_axes=0, axis_size=layers)
    def create_block(rngs):
      return _BasicTransformerBlock1D(
          dim=input_dim,
          heads=heads,
          dim_head=head_dim,
          rope_type=rope_type,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
      )

    # Call the vmapped constructor
    self.stacked_blocks = create_block(rngs)

    # 2. Thinking Tokens
    if num_learnable_registers > 0:
      key = rngs.params()
      self.learnable_registers = nnx.Param(
          jax.random.uniform(key, (num_learnable_registers, self.dim), dtype=jnp.bfloat16) * 2.0 - 1.0
      )

    self.final_norm = nnx.RMSNorm(
        self.dim, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=True, rngs=rngs
    )

  def _replace_padded_with_learnable_registers(self, hidden_states: Array, attention_mask: Array) -> Tuple[Array, Array]:
    b, t, d = hidden_states.shape
    if t % self.num_learnable_registers != 0:
      raise ValueError(f"Sequence length {t} must be divisible by {self.num_learnable_registers}")

    num_duplications = t // self.num_learnable_registers
    registers = jnp.tile(self.learnable_registers[...], (num_duplications, 1))
    registers = jnp.expand_dims(registers, 0)

    if attention_mask.ndim == 2:
      mask = attention_mask[:, :, None]
    else:
      mask = attention_mask

    output = jnp.where(mask > 0.5, hidden_states, registers)
    new_mask = jnp.ones_like(attention_mask)
    return output, new_mask

  def _compute_1d_rope(self, seq_len: int, dtype: DType) -> Tuple[Array, Array]:
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
    emb = jnp.outer(t, freqs)
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    return cos[None, ...], sin[None, ...]

  def __call__(
      self,
      hidden_states: Array,
      attention_mask: Optional[Array] = None,
  ) -> Array:
    # 1. Thinking Tokens
    if self.num_learnable_registers > 0 and attention_mask is not None:
      hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

    # 2. RoPE
    seq_len = hidden_states.shape[1]
    rotary_emb = self._compute_1d_rope(seq_len, hidden_states.dtype)

    # 3. Transformer Blocks (Scan)

    # Scan function signature: (carry, x) -> (carry, y)
    def block_scan_fn(carry, block_module):
      hidden_states = carry
      # block_module is a sliced view of the vmapped module
      hidden_states = block_module(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)
      return hidden_states, None

    # Execute scan
    hidden_states, _ = nnx.scan(
        block_scan_fn,
        length=self.num_layers,
        in_axes=(nnx.Carry, 0),  # Scan over the layers dimension (0) of block_module
        out_axes=(nnx.Carry, 0),
    )(hidden_states, self.stacked_blocks)

    # 4. Final Norm
    hidden_states = self.final_norm(hidden_states)

    return hidden_states
