"""
Copyright 2026 Google LLC

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
      gated_attn: bool = False,
  ):
    self.attn1 = LTX2Attention(
        query_dim=dim,
        heads=heads,
        dim_head=dim_head,
        rope_type=rope_type,
        bias=True,
        out_bias=True,
        attention_kernel=attention_kernel,
        mesh=mesh,
        rngs=rngs,
        gated_attn=gated_attn,
    )
    self.ff = NNXSimpleFeedForward(rngs=rngs, dim=dim, dim_out=dim, activation_fn="gelu_tanh")
    self.norm1 = nnx.RMSNorm(dim, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=False, rngs=rngs)
    self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, use_scale=False, rngs=rngs)

  def __call__(
      self,
      hidden_states: Array,
      attention_mask: Optional[Array] = None,
      rotary_emb: Optional[Tuple[Array, Array]] = None,
  ) -> Array:
    # 1. Norm -> Attention
    normed = self.norm1(hidden_states).astype(hidden_states.dtype)
    attn_output = self.attn1(normed, attention_mask=attention_mask, rotary_emb=rotary_emb)
    hidden_states = hidden_states + attn_output

    # 2. Norm -> FeedForward
    normed = self.norm2(hidden_states).astype(hidden_states.dtype)
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
      base_seq_len: int = 4096,
      double_precision: bool = True,
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
      gated_attn: bool = False,
  ):
    self.dim = input_dim
    self.heads = heads
    self.head_dim = head_dim
    self.theta = theta
    self.num_learnable_registers = num_learnable_registers
    self.num_layers = layers
    self.rope_type = rope_type
    self.base_seq_len = base_seq_len
    self.double_precision = double_precision

    # 1. Initialize Stacked Layers using vmap
    # This creates a single module where parameters have an extra leading dimension [layers, ...]
    # We need to ensure rngs are split for each layer
    @nnx.split_rngs(splits=layers)
    @nnx.vmap(in_axes=0, out_axes=0, axis_size=layers, transform_metadata={nnx.PARTITION_NAME: "layers"})
    def create_block(rngs):
      return _BasicTransformerBlock1D(
          dim=input_dim,
          heads=heads,
          dim_head=head_dim,
          rope_type=rope_type,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
          gated_attn=gated_attn,
      )

    # Call the vmapped constructor
    self.stacked_blocks = create_block(rngs)

    # 2. Thinking Tokens
    if num_learnable_registers > 0:
      key = rngs.params()
      self.learnable_registers = nnx.Param(
          jax.random.uniform(key, (num_learnable_registers, self.dim), dtype=jnp.bfloat16) * 2.0 - 1.0
      )

    self.final_norm = nnx.RMSNorm(self.dim, epsilon=1e-6, dtype=jnp.float32, use_scale=False, rngs=rngs)

  def _replace_padded_with_learnable_registers(self, hidden_states: Array, attention_mask: Array) -> Tuple[Array, Array]:
    b, t, d = hidden_states.shape
    if t % self.num_learnable_registers != 0:
      raise ValueError(f"Sequence length {t} must be divisible by {self.num_learnable_registers}")

    num_duplications = t // self.num_learnable_registers
    registers = jnp.tile(self.learnable_registers[...], (num_duplications, 1))

    if attention_mask.ndim == 2:
      mask = attention_mask
    else:
      mask = attention_mask.squeeze(-1)  # [B, T]

    # Mask valid tokens as 1 (or True)
    curr_mask = (mask > 0.5).astype(jnp.int32)

    # 1. Shift valid tokens to the left
    num_valid = jnp.sum(curr_mask, axis=1, keepdims=True)
    valid_positions = jnp.cumsum(curr_mask, axis=1) - 1
    invalid_positions = jnp.cumsum(1 - curr_mask, axis=1) - 1 + num_valid
    target_indices = jnp.where(curr_mask == 1, valid_positions, invalid_positions)

    b_idx = jnp.arange(b)[:, None]

    # Shift hidden states
    shifted_hidden_states = jnp.zeros_like(hidden_states)
    shifted_hidden_states = shifted_hidden_states.at[b_idx, target_indices, :].set(hidden_states)

    # Shift mask
    shifted_mask = jnp.zeros_like(curr_mask)
    shifted_mask = shifted_mask.at[b_idx, target_indices].set(curr_mask)

    # 2. Add Learnable Registers
    # Where shifted_mask is 1, keep valid tokens. Where 0, insert registers.
    output = jnp.where(shifted_mask[..., None] == 1, shifted_hidden_states, registers)

    # Padding has been filled with valid register tokens. The entire sequence
    # must now be attended to, so return an all-ones mask (matching diffusers).
    new_mask = jnp.ones((b, t), dtype=jnp.int32)

    return output, new_mask

  def _compute_1d_rope(self, batch_size: int, seq_len: int, dtype: DType) -> Tuple[Array, Array]:
    grid_1d = jnp.arange(seq_len, dtype=jnp.float32)
    grid_1d = grid_1d / self.base_seq_len
    grid = jnp.expand_dims(grid_1d, 0)
    grid = jnp.tile(grid, (batch_size, 1))

    num_rope_elems = 2
    freqs_dtype = jnp.float64 if self.double_precision else jnp.float32
    steps = self.dim // num_rope_elems
    pow_indices = jnp.power(self.theta, jnp.linspace(0.0, 1.0, steps, dtype=freqs_dtype))
    base_freqs = (pow_indices * jnp.pi / 2.0).astype(jnp.float32)

    freqs = (jnp.expand_dims(grid, -1) * 2.0 - 1.0) * base_freqs

    cos_freqs = jnp.cos(freqs)
    sin_freqs = jnp.sin(freqs)

    if self.rope_type == "interleaved":
      cos_freqs = jnp.repeat(cos_freqs, 2, axis=-1)
      sin_freqs = jnp.repeat(sin_freqs, 2, axis=-1)

      if self.dim % num_rope_elems != 0:
        curr_dim = cos_freqs.shape[-1]
        pad_amt = self.dim - curr_dim
        if pad_amt > 0:
          cos_padding = jnp.ones((*cos_freqs.shape[:-1], pad_amt), dtype=cos_freqs.dtype)
          sin_padding = jnp.zeros((*sin_freqs.shape[:-1], pad_amt), dtype=sin_freqs.dtype)
          cos_freqs = jnp.concatenate([cos_padding, cos_freqs], axis=-1)
          sin_freqs = jnp.concatenate([sin_padding, sin_freqs], axis=-1)

    elif self.rope_type == "split":
      expected_freqs = self.dim // 2
      current_freqs = freqs.shape[-1]
      pad_size = expected_freqs - current_freqs

      if pad_size > 0:
        cos_padding = jnp.ones((*cos_freqs.shape[:-1], pad_size), dtype=cos_freqs.dtype)
        sin_padding = jnp.zeros((*sin_freqs.shape[:-1], pad_size), dtype=sin_freqs.dtype)
        cos_freqs = jnp.concatenate([cos_padding, cos_freqs], axis=-1)
        sin_freqs = jnp.concatenate([sin_padding, sin_freqs], axis=-1)

      b = cos_freqs.shape[0]
      t = cos_freqs.shape[1]
      h = self.heads
      cos_freqs = cos_freqs.reshape(b, t, h, -1).transpose(0, 2, 1, 3)
      sin_freqs = sin_freqs.reshape(b, t, h, -1).transpose(0, 2, 1, 3)

    return cos_freqs, sin_freqs

  def __call__(
      self,
      hidden_states: Array,
      attention_mask: Optional[Array] = None,
  ) -> Tuple[Array, Array]:
    # 1. Thinking Tokens
    if self.num_learnable_registers > 0 and attention_mask is not None:
      hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

    # 2. RoPE
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    rotary_emb = self._compute_1d_rope(batch_size, seq_len, hidden_states.dtype)

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

    return hidden_states, attention_mask
