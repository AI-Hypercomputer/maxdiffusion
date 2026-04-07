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

"""NNX-based LoRA loader for LTX2 models."""

from flax import nnx
from .lora_base import LoRABaseMixin
from .lora_pipeline import StableDiffusionLoraLoaderMixin
from ..models import lora_nnx
from .. import max_logging
from . import lora_conversion_utils


class LTX2NNXLoraLoader(LoRABaseMixin):
  """
  Handles loading LoRA weights into NNX-based LTX2 model.
  Assumes LTX2 pipeline contains 'transformer'
  attributes that are NNX Modules.
  """

  def load_lora_weights(
      self,
      pipeline: nnx.Module,
      lora_model_path: str,
      transformer_weight_name: str,
      rank: int,
      scale: float = 1.0,
      scan_layers: bool = False,
      dtype: str = "float32",
      **kwargs,
  ):
    """
    Merges LoRA weights into the pipeline from a checkpoint.
    """
    lora_loader = StableDiffusionLoraLoaderMixin()

    merge_fn = lora_nnx.merge_lora_for_scanned if scan_layers else lora_nnx.merge_lora

    def translate_fn(nnx_path_str):
      return lora_conversion_utils.translate_ltx2_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=scan_layers)

    h_state_dict = None
    if hasattr(pipeline, "transformer") and transformer_weight_name:
      max_logging.log(f"Merging LoRA into transformer with rank={rank}")
      h_state_dict, _ = lora_loader.lora_state_dict(lora_model_path, weight_name=transformer_weight_name, **kwargs)
      # Filter state dict for transformer keys to avoid confusing warnings
      transformer_state_dict = {k: v for k, v in h_state_dict.items() if k.startswith("diffusion_model")}
      merge_fn(pipeline.transformer, transformer_state_dict, rank, scale, translate_fn, dtype=dtype)
    else:
      max_logging.log("transformer not found or no weight name provided for LoRA.")

    if hasattr(pipeline, "connectors"):
      max_logging.log(f"Merging LoRA into connectors with rank={rank}")
      if h_state_dict is None and transformer_weight_name:
        h_state_dict, _ = lora_loader.lora_state_dict(lora_model_path, weight_name=transformer_weight_name, **kwargs)

      if h_state_dict is not None:
        # Filter state dict for connector keys to avoid confusing warnings
        connector_state_dict = {k: v for k, v in h_state_dict.items() if k.startswith("text_embedding_projection")}
        merge_fn(pipeline.connectors, connector_state_dict, rank, scale, translate_fn, dtype=dtype)
      else:
        max_logging.log("Could not load LoRA state dict for connectors.")

    return pipeline
