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
class ZImageDiTShardingSpecs:
  """Sharding specs for the Z-Image Diffusion Transformer."""

  # --- Attention Layers (ZImageAttention) ---
  qkv_kernel: tuple = ("embed", "heads")
  qkv_bias: tuple = ("heads",)
  out_kernel: tuple = ("heads", "embed")
  out_bias: tuple = ("embed",)

  # --- Feed Forward Network (ZImageFeedForward) ---
  net_0_kernel: tuple = ("embed", "mlp")
  net_0_bias: tuple = ("mlp",)
  net_2_kernel: tuple = ("mlp", "embed")
  net_2_bias: tuple = ("embed",)

  # --- Input/Output Projections and Embeddings ---
  embed_kernel: tuple = ("embed", "heads")
  embed_bias: tuple = ("heads",)
  out_embed_kernel: tuple = ("embed", "out_channels")
  out_embed_bias: tuple = ("out_channels",)

  # --- Shared Embeddings (ZImageTimestepEmbedder) ---
  emb_linear_1_kernel: tuple = ("embed", "mlp")
  emb_linear_1_bias: tuple = ("mlp",)
  emb_linear_2_kernel: tuple = ("mlp", "embed")
  emb_linear_2_bias: tuple = ("embed",)

  # --- Modulation ---
  adaln_kernel: tuple = ("embed", "mlp")
  adaln_bias: tuple = ("mlp",)

  # --- Normalization ---
  norm_scale: tuple = ("norm",)


# --- Unified Registry for Z-Image ---
STRATEGIES = {
    "ironwood": {
        "z_image_dit": ZImageDiTShardingSpecs(),
    },
    "trillium": {
        "z_image_dit": ZImageDiTShardingSpecs(),
    },
}


def get_sharding_specs(strategy_name: str, component_name: str) -> Any:
  """Unified factory to get specs for Z-Image components.

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
        f"Warning: Strategy '{strategy_name}' is not recognized in the Z-Image registry. "
        f"Falling back to the highly compatible 'trillium' strategy profile."
    )
    strategy_name = "trillium"

  hardware_profile = STRATEGIES[strategy_name]
  specs = hardware_profile.get(component_name)
  if specs is None:
    raise ValueError(f"Component {component_name} not found in strategy {strategy_name}")
  return specs
