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

from typing import Tuple, Union
import jax.numpy as jnp
from flax import nnx
from maxdiffusion import common_types

Array = common_types.Array
DType = common_types.DType


def _norm_and_concat_padded_batch(
    encoded_text: Array,
    sequence_lengths: Array,
    padding_side: str = "right",
) -> Array:
  """Normalize and flatten multi-layer hidden states, respecting padding.
  Performs per-batch, per-layer normalization using masked mean and range,
  then concatenates across the layer dimension.

  Args:
      encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
      sequence_lengths: Number of valid (non-padded) tokens per batch item.
      padding_side: Whether padding is on "left" or "right".

  Returns:
      Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
      with padded positions zeroed out.
  """
  b, t, d, l = encoded_text.shape

  # Build mask: [B, T] -> [B, T, 1, 1]
  # token_indices: [1, T]
  token_indices = jnp.arange(t)[None, :]

  if padding_side == "right":
    # Valid: indices < lengths
    mask = token_indices < sequence_lengths[:, None]
  elif padding_side == "left":
    # Valid: indices >= (T - lengths)
    start_indices = t - sequence_lengths[:, None]
    mask = token_indices >= start_indices
  else:
    raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

  # [B, T, 1, 1]
  mask = mask[:, :, None, None]

  eps = 1e-6

  # 1. Compute Masked Mean
  # Masked sum: [B, 1, 1, L] (sum over T, D)
  # Using jnp.where to zero-out padding
  masked_text = jnp.where(mask, encoded_text, 0.0)
  sum_vals = jnp.sum(masked_text, axis=(1, 2), keepdims=True)

  # Denom: sequence_length * D
  denom = (sequence_lengths * d).reshape(b, 1, 1, 1)
  mean = sum_vals / (denom + eps)

  # 2. Compute Masked Min/Max for Range
  # Use jnp.inf / -jnp.inf for padding to ignore them in min/max
  safe_text_min = jnp.where(mask, encoded_text, jnp.inf)
  safe_text_max = jnp.where(mask, encoded_text, -jnp.inf)

  x_min = jnp.min(safe_text_min, axis=(1, 2), keepdims=True)
  x_max = jnp.max(safe_text_max, axis=(1, 2), keepdims=True)

  range_val = x_max - x_min

  # 3. Normalize
  # Only valid tokens are normalized. Padding will be garbage but masked out later.
  normed = 8.0 * (encoded_text - mean) / (range_val + eps)

  # 4. Concatenate/Flatten Layers
  # [B, T, D, L] -> [B, T, D * L]
  normed = normed.reshape(b, t, -1)

  # 5. Apply Mask to Output
  # Ensure padding positions are exactly 0.0
  # mask: [B, T, 1, 1] -> [B, T, 1]
  output_mask = mask.squeeze(-1).squeeze(-1)[:, :, None]
  normed = jnp.where(output_mask, normed, 0.0)

  return normed


class LTX2GemmaFeatureExtractor(nnx.Module):
  """
  Feature extractor module for Gemma models in LTX-2.
  Applies mean-centered scaling and a linear projection.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      dtype: DType = jnp.float32,
      rngs: nnx.Rngs = None,
  ):
    """
    Args:
        input_dim: Dimension of flattened hidden states (Gemma dim * Num layers).
        output_dim: Target dimension for diffusion conditioning.
    """
    # LTX-2 uses bias=False for the projection
    self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, dtype=dtype, rngs=rngs)

  def __call__(
      self, hidden_states: Union[Tuple[Array, ...], Array], attention_mask: Array, padding_side: str = "right"
  ) -> Array:
    """
    Args:
        hidden_states: Tuple of arrays from Gemma, each [B, T, D].
                       Or pre-stacked array [B, T, D, L].
        attention_mask: Mask [B, T] (1 for valid, 0 for padding).
        padding_side: "right" or "left".

    Returns:
        Projected features [B, T, OutputDim].
    """

    # 1. Stack Hidden States if needed
    if isinstance(hidden_states, (tuple, list)):
      # [B, T, D, L]
      x = jnp.stack(hidden_states, axis=-1)
    else:
      x = hidden_states

    # 2. Calculate Sequence Lengths
    sequence_lengths = jnp.sum(attention_mask, axis=-1)

    # 3. Norm and Concat
    x_norm = _norm_and_concat_padded_batch(x, sequence_lengths, padding_side=padding_side)

    # 4. Projection
    return self.linear(x_norm)
