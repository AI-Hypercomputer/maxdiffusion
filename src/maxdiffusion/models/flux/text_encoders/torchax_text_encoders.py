# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLUX text encoders (CLIP-L, T5-XXL) running under JAX through Torchax.

transformers dropped its Flax implementations in v5, so FLUX loads the
PyTorch encoders and traces them into JAX here, the same way LTX2 wraps
Gemma 3 and WAN wraps UMT5. Both encoders are frozen: training only calls
them to precompute prompt embeddings before the transformer sees a batch.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from torchax import default_env, interop
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

CLIP_TOKENIZER_MAX_LENGTH = 77

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _torch_dtype(jax_dtype) -> torch.dtype:
  name = jax_dtype.name if hasattr(jax_dtype, "name") else str(jax_dtype)
  if name not in TORCH_DTYPE_MAP:
    raise ValueError(f"Unsupported text encoder dtype: {name}. Supported values are: {list(TORCH_DTYPE_MAP.keys())}")
  return TORCH_DTYPE_MAP[name]


class _TorchaxTextEncoder(interop.JittableModule):
  """Common Torchax plumbing for the frozen FLUX text encoders."""

  def __init__(self, model: torch.nn.Module, jax_dtype: jnp.dtype):
    super().__init__(model)
    self.jax_dtype = jax_dtype

  @classmethod
  def from_torch(cls, model: torch.nn.Module, jax_dtype: jnp.dtype) -> "_TorchaxTextEncoder":
    model.eval()
    with default_env():
      return cls(model.to("jax"), jax_dtype)

  def place_params(self, device_or_sharding) -> None:
    """Move the frozen weights onto `device_or_sharding`.

    Torchax holds the weights as torch views over JAX arrays, so the placement
    goes through `jax_view`/`torch_view`. A `Sharding` target is built with
    `make_array_from_callback` (as `max_utils.device_put_replicated` does),
    because each host only holds its own full copy of these weights and
    `device_put` cannot span non-addressable devices.
    """

    def place(leaf):
      if isinstance(device_or_sharding, jax.sharding.Sharding):
        return jax.make_array_from_callback(leaf.shape, device_or_sharding, lambda index: leaf[index])
      return jax.device_put(leaf, device_or_sharding)

    self.params = interop.torch_view(jax.tree_util.tree_map(place, interop.jax_view(self.params)))

  def offload_params(self) -> None:
    """Park the weights in host memory until the next encode call."""
    self.place_params(jax.devices("cpu")[0])


class TorchaxCLIPTextEncoder(_TorchaxTextEncoder):
  """Wraps `transformers.CLIPTextModel`, returning FLUX's pooled embedding."""

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    with default_env():
      pooler_output = self.functional_call(
          self._pooler_output,
          params=self.params,
          buffers=self.buffers,
          input_ids=interop.torch_view(input_ids),
      )
    return interop.jax_view(pooler_output).astype(self.jax_dtype)

  @staticmethod
  def _pooler_output(model, input_ids):
    # Returning the tensor rather than the output dataclass keeps the result a
    # plain pytree that `interop.jax_view` can convert.
    return model(input_ids=input_ids).pooler_output


class TorchaxT5TextEncoder(_TorchaxTextEncoder):
  """Wraps `transformers.T5EncoderModel`, returning the last hidden state."""

  def __call__(self, input_ids: jax.Array, attention_mask: Optional[jax.Array] = None) -> jax.Array:
    with default_env():
      last_hidden_state = self.functional_call(
          self._last_hidden_state,
          params=self.params,
          buffers=self.buffers,
          input_ids=interop.torch_view(input_ids),
          attention_mask=None if attention_mask is None else interop.torch_view(attention_mask),
      )
    return interop.jax_view(last_hidden_state).astype(self.jax_dtype)

  @staticmethod
  def _last_hidden_state(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _optional_subfolder(config, key) -> str:
  """Read a subfolder key that a config predating the PyTorch weights may not carry.

  pyconfig raises ValueError for a key it does not hold, which getattr's default
  cannot absorb, so an older config would fail here rather than fall back to the
  repository root.
  """
  try:
    return getattr(config, key) or ""
  except (AttributeError, KeyError, ValueError):
    return ""


def load_clip_encoder_and_tokenizer(config) -> Tuple[TorchaxCLIPTextEncoder, CLIPTokenizer]:
  """Load FLUX's CLIP-L text encoder and its tokenizer from PyTorch weights."""
  subfolder = _optional_subfolder(config, "clip_model_subfolder")
  # `dtype` rather than the deprecated `torch_dtype`: transformers 5 documents
  # only the former. Eager attention keeps the graph traceable under Torchax.
  encoder = CLIPTextModel.from_pretrained(
      config.clip_model_name_or_path,
      subfolder=subfolder,
      dtype=_torch_dtype(config.weights_dtype),
      attn_implementation="eager",
  )
  tokenizer = CLIPTokenizer.from_pretrained(
      config.clip_model_name_or_path,
      subfolder=_optional_subfolder(config, "clip_tokenizer_subfolder"),
      model_max_length=CLIP_TOKENIZER_MAX_LENGTH,
  )
  return TorchaxCLIPTextEncoder.from_torch(encoder, config.weights_dtype), tokenizer


def load_t5_encoder_and_tokenizer(config) -> Tuple[TorchaxT5TextEncoder, AutoTokenizer]:
  """Load FLUX's T5-XXL text encoder and its tokenizer from PyTorch weights."""
  subfolder = _optional_subfolder(config, "t5xxl_model_subfolder")
  encoder = T5EncoderModel.from_pretrained(
      config.t5xxl_model_name_or_path,
      subfolder=subfolder,
      dtype=_torch_dtype(config.weights_dtype),
      attn_implementation="eager",
  )
  tokenizer = AutoTokenizer.from_pretrained(
      config.t5xxl_model_name_or_path,
      subfolder=_optional_subfolder(config, "t5xxl_tokenizer_subfolder"),
      model_max_length=config.max_sequence_length,
      use_fast=True,
  )
  return TorchaxT5TextEncoder.from_torch(encoder, config.weights_dtype), tokenizer
