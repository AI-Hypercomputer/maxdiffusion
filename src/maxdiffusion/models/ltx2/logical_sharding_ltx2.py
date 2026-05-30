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

from dataclasses import dataclass
from typing import Any
from maxdiffusion.tpu_utils import get_tpu_type, TpuType
from maxdiffusion import max_logging


# --- Discrete Specs ---
@dataclass
class LTX2DiTShardingSpecs:
  """Sharding specs for the LTX2 Diffusion Transformer."""

  # --- Attention Layers (LTX2Attention) ---
  qkv_kernel: tuple = ("embed", "heads")
  out_kernel: tuple = ("heads", "embed")
  out_bias: tuple = ("embed",)
  qkv_bias: tuple = ("heads",)
  gate_logits_kernel: tuple = ("embed", "heads")
  gate_logits_bias: tuple = ("heads",)

  # --- Feed Forward Network (NNXSimpleFeedForward) ---
  net_0_kernel: tuple = ("embed", "mlp")
  net_0_bias: tuple = ("mlp",)
  net_2_kernel: tuple = ("mlp", "embed")
  net_2_bias: tuple = ("embed",)

  # --- Input/Output Projections and Tables ---
  embed_kernel: tuple = (None, "embed")
  embed_bias: tuple = ("embed",)
  adaln_kernel: tuple = (None, "embed")
  adaln_bias: tuple = ("embed",)
  scale_shift_table: tuple = (None, "embed")
  out_embed_kernel: tuple = ("embed", None)
  out_embed_bias: tuple = (None,)

  # --- Shared Embeddings (NNXTimestepEmbedding, NNXPixArtAlphaTextProjection) ---
  emb_linear_1_kernel: tuple = ("embed", "mlp")
  emb_linear_1_bias: tuple = ("mlp",)
  emb_linear_2_kernel: tuple = ("mlp", "embed")
  emb_linear_2_bias: tuple = ("embed",)

  # --- Normalization ---
  norm_scale: tuple = ("norm",)


@dataclass
class TextConnectorShardingSpecs:
  """Specs for the Text Connector execution."""

  # --- MLP Specs (NNXSimpleFeedForward) ---
  net_0_kernel: tuple = ("embed", "mlp")
  net_0_bias: tuple = ("mlp",)
  net_2_kernel: tuple = ("mlp", "embed")
  net_2_bias: tuple = ("embed",)

  # --- Attention Specs (LTX2Attention) ---
  qkv_kernel: tuple = ("embed", "heads")
  out_kernel: tuple = ("heads", "embed")
  out_bias: tuple = ("embed",)
  qkv_bias: tuple = ("heads",)
  gate_logits_kernel: tuple = ("embed", "heads")
  gate_logits_bias: tuple = ("heads",)
  norm_scale: tuple = ("norm",)

  # --- Projection Specs (feature_extractor / per_modality_projections) ---
  proj_kernel: tuple = (None, None)
  proj_bias: tuple = (None,)


@dataclass
class VAEShardingSpecs:
  """Sharding specs for the VAE."""

  # --- Shared Embeddings Specs (NNXPixArtAlphaCombinedTimestepSizeEmbeddings) ---
  emb_linear_1_kernel: tuple = ("embed", "mlp")
  emb_linear_1_bias: tuple = ("mlp",)
  emb_linear_2_kernel: tuple = ("mlp", "embed")
  emb_linear_2_bias: tuple = ("embed",)

  # --- ResNet Block Specs ---
  scale_shift_table: tuple = (None, None)
  per_channel_scale: tuple = (None,)


# --- Unified Registry for LTX2 ---
STRATEGIES = {
    "ironwood": {
        "ltx2_dit": LTX2DiTShardingSpecs(
            qkv_kernel=(None, "heads"),
            qkv_bias=("heads",),
            out_kernel=("heads", None),
            out_bias=(None,),
            gate_logits_kernel=(None, "heads"),
            gate_logits_bias=("heads",),
            net_0_kernel=(None, "mlp"),
            net_2_kernel=("mlp", None),
            net_2_bias=(None,),
            embed_kernel=(None, None),
            embed_bias=(None,),
            adaln_kernel=(None, None),
            adaln_bias=(None,),
            scale_shift_table=(None, None),
            out_embed_kernel=(None, None),
            out_embed_bias=(None,),
            emb_linear_1_kernel=(None, "mlp"),
            emb_linear_2_kernel=("mlp", None),
            emb_linear_2_bias=(None,),
            norm_scale=(None,),
        ),
        "text_connector": TextConnectorShardingSpecs(
            qkv_kernel=(None, "heads"),
            qkv_bias=("heads",),
            out_kernel=("heads", None),
            out_bias=(None,),
            gate_logits_kernel=(None, "heads"),
            gate_logits_bias=("heads",),
            net_0_kernel=(None, "mlp"),
            net_2_kernel=("mlp", None),
            net_2_bias=(None,),
            norm_scale=(None,),
        ),
        "vae": VAEShardingSpecs(
            emb_linear_1_kernel=(None, "mlp"),
            emb_linear_2_kernel=("mlp", None),
            emb_linear_2_bias=(None,),
        ),
    },
    "trillium": {
        "ltx2_dit": LTX2DiTShardingSpecs(),
        "text_connector": TextConnectorShardingSpecs(),
        "vae": VAEShardingSpecs(),
    },
}


def get_sharding_specs(strategy_name: str, component_name: str) -> Any:
  """Unified factory to get specs for any component.

  If strategy_name is 'default', it auto-detects the hardware.
  """
  if strategy_name == "default":
    tpu_type = get_tpu_type()
    if tpu_type == TpuType.TPU_7X:
      strategy_name = "ironwood"
    else:
      strategy_name = "trillium"

  if strategy_name not in STRATEGIES:
    max_logging.log(
        f"Warning: Strategy '{strategy_name}' is not recognized in the LTX2 registry. "
        f"Falling back to the highly compatible 'trillium' strategy profile."
    )
    strategy_name = "trillium"

  hardware_profile = STRATEGIES[strategy_name]
  specs = hardware_profile.get(component_name)
  if specs is None:
    raise ValueError(f"Component {component_name} not found in strategy {strategy_name}")
  return specs
