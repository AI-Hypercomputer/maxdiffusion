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

from typing import Optional, Tuple, Union, List
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion import common_types

from .feature_extractor_ltx2 import LTX2GemmaFeatureExtractor
from .embeddings_connector_ltx2 import Embeddings1DConnector

Array = common_types.Array
DType = common_types.DType


class LTX2VideoGemmaTextEncoder(nnx.Module):
  """
  Encoder for Video-only tasks.
  Pipeline: Gemma Hidden States -> Feature Extractor -> Video Connector -> Output
  """

  def __init__(
      self,
      # Feature Extractor Config
      gemma_dim: int = 3840,  # Gemma-3-12b
      gemma_layers: int = 49,  # Gemma-3 has 48 layers + 1 embedding layer output = 49 hidden states
      projection_dim: int = 3840,  # LTX-2 conditioning dim
      # Connector Config
      connector_heads: int = 32,
      connector_head_dim: int = 128,
      connector_layers: int = 2,
      num_thinking_tokens: int = 128,
      dtype: DType = jnp.float32,
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
  ):
    input_dim = gemma_dim * gemma_layers

    self.feature_extractor = LTX2GemmaFeatureExtractor(
        input_dim=input_dim,
        output_dim=projection_dim,
        dtype=dtype,
        rngs=rngs,
    )

    self.embeddings_connector = Embeddings1DConnector(
        input_dim=projection_dim,
        heads=connector_heads,
        head_dim=connector_head_dim,
        layers=connector_layers,
        num_learnable_registers=num_thinking_tokens,
        rope_type="interleaved",
        attention_kernel=attention_kernel,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Union[Tuple[Array, ...], List[Array]],
      attention_mask: Array,
  ) -> Array:
    """
    Args:
        hidden_states: From Gemma output.hidden_states (Tuple of [B, T, D])
        attention_mask: [B, T]
    """
    # 1. Feature Extraction (Stack -> Norm -> Project)
    features = self.feature_extractor(hidden_states, attention_mask)

    # 2. Connection (Refine + Thinking Tokens)
    video_embeds = self.embeddings_connector(features, attention_mask)

    return video_embeds


class LTX2AudioVideoGemmaTextEncoder(nnx.Module):
  """
  Encoder for Audio-Video tasks.
  Pipeline: Gemma Hidden States -> Feature Extractor -> [Video Connector, Audio Connector]
  """

  def __init__(
      self,
      # Feature Extractor Config (Shared)
      gemma_dim: int = 3840,  # Gemma-3-12b
      gemma_layers: int = 49,  # Gemma-3 has 48 layers + 1 embedding layer output = 49 hidden states
      projection_dim: int = 3840,
      # Connector Config
      connector_heads: int = 32,
      connector_head_dim: int = 128,
      connector_layers: int = 2,
      num_thinking_tokens: int = 128,
      dtype: DType = jnp.float32,
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
  ):
    input_dim = gemma_dim * gemma_layers

    self.feature_extractor = LTX2GemmaFeatureExtractor(
        input_dim=input_dim,
        output_dim=projection_dim,
        dtype=dtype,
        rngs=rngs,
    )

    # Two independent connectors
    self.video_embeddings_connector = Embeddings1DConnector(
        input_dim=projection_dim,
        heads=connector_heads,
        head_dim=connector_head_dim,
        layers=connector_layers,
        num_learnable_registers=num_thinking_tokens,
        rope_type="interleaved",
        attention_kernel=attention_kernel,
        mesh=mesh,
        rngs=rngs,
    )

    self.audio_embeddings_connector = Embeddings1DConnector(
        input_dim=projection_dim,
        heads=connector_heads,
        head_dim=connector_head_dim,
        layers=connector_layers,
        num_learnable_registers=num_thinking_tokens,
        rope_type="interleaved",
        attention_kernel=attention_kernel,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Union[Tuple[Array, ...], List[Array]],
      attention_mask: Array,
  ) -> Tuple[Array, Array]:
    """
    Returns:
        (video_embeds, audio_embeds)
    """
    # 1. Shared Feature Extraction
    features = self.feature_extractor(hidden_states, attention_mask)

    # 2. Parallel Connection
    video_embeds = self.video_embeddings_connector(features, attention_mask)
    audio_embeds = self.audio_embeddings_connector(features, attention_mask)

    return video_embeds, audio_embeds
