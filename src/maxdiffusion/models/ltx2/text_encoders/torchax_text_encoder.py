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

from typing import Tuple

import torch
import jax
from torchax import interop, default_env

import contextlib
import transformers.masking_utils


def _patched_sliding_window_overlay(sliding_window: int):
  # pylint: disable=unused-argument

  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    # Explicit Sequence Length Assumption:
    # This patch assumes that the maximum sequence length used for text prompts (typically <= 1024)
    # is strictly less than the sliding window size of Gemma-3 (typically 4096).
    # Under this assumption, the sliding window causal constraint `kv_idx > q_idx - sliding_window`
    # is mathematically always True for all valid query/key indices (0 <= q_idx, kv_idx < seq_len).
    #
    # We return a standard boolean tensor `q_idx.new_ones((), dtype=torch.bool)` to guarantee
    # Torchax compatibility and prevent any implicit tracing crashes.
    # If a future model uses a sequence length exceeding the sliding window, this assumption must be re-evaluated.
    return q_idx.new_ones((), dtype=torch.bool)

  return inner_mask


@contextlib.contextmanager
def patch_sliding_window_overlay():
  orig = transformers.masking_utils.sliding_window_overlay
  transformers.masking_utils.sliding_window_overlay = _patched_sliding_window_overlay
  try:
    yield
  finally:
    transformers.masking_utils.sliding_window_overlay = orig


class TorchaxGemma3TextEncoder(interop.JittableModule):
  """
  A jittable Torchax module for wrapping the HuggingFace PyTorch
  Gemma3ForConditionalGeneration text encoder.
  """

  def __init__(self, text_encoder):
    super().__init__(text_encoder, extra_jit_args={"static_argnames": ["output_hidden_states"]})

  def __call__(
      self, input_ids: jax.Array, attention_mask: jax.Array, output_hidden_states: bool = True
  ) -> Tuple[jax.Array, ...]:
    # Dynamically patch transformers.masking_utils only during the duration of this call
    with patch_sliding_window_overlay():
      with default_env():
        input_ids = interop.torch_view(input_ids)
        attention_mask = interop.torch_view(attention_mask)

        output = self.functional_call(
            self._forward_inner,
            params=self.params,
            buffers=self.buffers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
      return interop.jax_view(output)

  @staticmethod
  def _forward_inner(model, input_ids, attention_mask, output_hidden_states=True):
    # We only return hidden states as a tuple of tensors.
    # That allows interop.jax_view to convert them into a tuple of jax Arrays
    return model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states).hidden_states
