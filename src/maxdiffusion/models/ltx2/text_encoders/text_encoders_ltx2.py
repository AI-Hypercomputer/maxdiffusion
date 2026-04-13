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

from typing import Optional, Tuple, Union, List
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion import common_types

from .feature_extractor_ltx2 import LTX2GemmaFeatureExtractor
from .embeddings_connector_ltx2 import Embeddings1DConnector
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.models.modeling_flax_utils import FlaxModelMixin

Array = common_types.Array
DType = common_types.DType


class LTX2AudioVideoGemmaTextEncoder(nnx.Module, FlaxModelMixin, ConfigMixin):
  """
  Encoder for Audio-Video tasks.
  Pipeline: Gemma Hidden States -> Feature Extractor -> [Video Connector, Audio Connector]
  """

  @register_to_config
  def __init__(
      self,
      caption_channels: int = 3840,
      video_caption_channels: Optional[int] = None,
      audio_caption_channels: Optional[int] = None,
      text_proj_in_factor: int = 49,
      video_connector_attention_head_dim: int = 128,
      video_connector_num_attention_heads: int = 30,
      video_connector_num_layers: int = 2,
      video_connector_num_learnable_registers: int = 128,
      audio_connector_attention_head_dim: int = 128,
      audio_connector_num_attention_heads: int = 30,
      audio_connector_num_layers: int = 2,
      audio_connector_num_learnable_registers: int = 128,
      connector_rope_base_seq_len: int = 4096,
      rope_double_precision: bool = True,
      rope_theta: float = 10000.0,
      rope_type: str = "split",
      causal_temporal_positioning: bool = False,
      dtype: DType = jnp.float32,
      attention_kernel: str = "flash",
      mesh: jax.sharding.Mesh = None,
      rngs: nnx.Rngs = None,
      per_modality_projections: bool = False,
      proj_bias: bool = False,
      video_gated_attn: bool = False,
      audio_gated_attn: bool = False,
      **kwargs,
  ):
    gemma_dim = 3840 if video_caption_channels is not None else caption_channels
    input_dim = gemma_dim * text_proj_in_factor

    v_dim = video_caption_channels if video_caption_channels is not None else caption_channels
    a_dim = audio_caption_channels if audio_caption_channels is not None else caption_channels

    self.per_modality_projections = per_modality_projections

    if per_modality_projections:
      self.video_text_proj_in = nnx.Linear(
          in_features=input_dim, out_features=v_dim, use_bias=proj_bias, rngs=rngs
      )
      self.audio_text_proj_in = nnx.Linear(
          in_features=input_dim, out_features=a_dim, use_bias=proj_bias, rngs=rngs
      )

      self.video_connector = Embeddings1DConnector(
          input_dim=v_dim,
          heads=video_connector_num_attention_heads,
          head_dim=video_connector_attention_head_dim,
          layers=video_connector_num_layers,
          num_learnable_registers=video_connector_num_learnable_registers,
          rope_type=rope_type,
          theta=rope_theta,
          base_seq_len=connector_rope_base_seq_len,
          double_precision=rope_double_precision,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
          gated_attn=video_gated_attn,
      )
      self.audio_connector = Embeddings1DConnector(
          input_dim=a_dim,
          heads=audio_connector_num_attention_heads,
          head_dim=audio_connector_attention_head_dim,
          layers=audio_connector_num_layers,
          num_learnable_registers=audio_connector_num_learnable_registers,
          rope_type=rope_type,
          theta=rope_theta,
          base_seq_len=connector_rope_base_seq_len,
          double_precision=rope_double_precision,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
          gated_attn=audio_gated_attn,
      )
    else:
      self.feature_extractor = LTX2GemmaFeatureExtractor(
          input_dim=input_dim,
          output_dim=caption_channels,
          dtype=dtype,
          rngs=rngs,
          per_modality_projections=per_modality_projections,
          use_bias=proj_bias,
          video_output_dim=v_dim,
          audio_output_dim=a_dim,
      )

      # Two independent connectors
      self.video_embeddings_connector = Embeddings1DConnector(
          input_dim=v_dim,
          heads=video_connector_num_attention_heads,
          head_dim=video_connector_attention_head_dim,
          layers=video_connector_num_layers,
          num_learnable_registers=video_connector_num_learnable_registers,
          rope_type=rope_type,
          theta=rope_theta,
          base_seq_len=connector_rope_base_seq_len,
          double_precision=rope_double_precision,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
          gated_attn=video_gated_attn,
      )
      self.audio_embeddings_connector = Embeddings1DConnector(
          input_dim=a_dim,
          heads=audio_connector_num_attention_heads,
          head_dim=audio_connector_attention_head_dim,
          layers=audio_connector_num_layers,
          num_learnable_registers=audio_connector_num_learnable_registers,
          rope_type=rope_type,
          theta=rope_theta,
          base_seq_len=connector_rope_base_seq_len,
          double_precision=rope_double_precision,
          attention_kernel=attention_kernel,
          mesh=mesh,
          rngs=rngs,
          gated_attn=audio_gated_attn,
      )

  def __call__(
      self,
      hidden_states: Union[Tuple[Array, ...], List[Array]],
      attention_mask: Array,
  ) -> Tuple[Array, Array, Array]:
    """
    Returns:
        (video_embeds, audio_embeds, new_attention_mask)
    """
    with jax.named_scope("Text Encoder Forward"):
      if self.per_modality_projections:
        # 1. Stack Hidden States if needed
        if isinstance(hidden_states, (tuple, list)):
          x = jnp.stack(hidden_states, axis=-1)
        else:
          x = hidden_states

        b, l, d, k = x.shape
        
        # 2. Per-token RMS norm
        variance = jnp.mean(x**2, axis=2, keepdims=True)
        
        # Debug prints
        print(f"DEBUG - x shape: {x.shape}")
        jax.debug.print("DEBUG - x min: {min}, max: {max}, mean: {mean}", min=jnp.min(x), max=jnp.max(x), mean=jnp.mean(x))
        jax.debug.print("DEBUG - variance min: {min}, max: {max}", min=jnp.min(variance), max=jnp.max(variance))

        norm_text_encoder_hidden_states = x * jax.lax.rsqrt(variance + 1e-6)

        norm_text_encoder_hidden_states = norm_text_encoder_hidden_states.reshape(b, l, -1)

        bool_mask = (attention_mask > 0.5).astype(jnp.float32)[..., None]
        norm_text_encoder_hidden_states = norm_text_encoder_hidden_states * bool_mask

        # 3. Rescale norms
        # Using self.caption_channels if available, or fallback to config or 3840
        cap_channels = getattr(self, "caption_channels", getattr(self.config, "caption_channels", 3840))
        
        video_scale_factor = jnp.sqrt(self.video_connector.dim / cap_channels)
        video_norm_text_emb = norm_text_encoder_hidden_states * video_scale_factor
        audio_scale_factor = jnp.sqrt(self.audio_connector.dim / cap_channels)
        audio_norm_text_emb = norm_text_encoder_hidden_states * audio_scale_factor

        video_text_emb_proj = self.video_text_proj_in(video_norm_text_emb)
        audio_text_emb_proj = self.audio_text_proj_in(audio_norm_text_emb)

        video_embeds, new_attention_mask = self.video_connector(video_text_emb_proj, attention_mask)
        audio_embeds, _ = self.audio_connector(audio_text_emb_proj, attention_mask)
      else:
        # 1. Shared Feature Extraction
        features = self.feature_extractor(hidden_states, attention_mask)

        # 2. Parallel Connection
        video_embeds, new_attention_mask = self.video_embeddings_connector(features, attention_mask)
        audio_embeds, _ = self.audio_embeddings_connector(features, attention_mask)

      return video_embeds, audio_embeds, new_attention_mask
