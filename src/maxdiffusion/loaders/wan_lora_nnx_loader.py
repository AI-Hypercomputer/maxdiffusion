# Copyright 2025 Google LLC
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

"""NNX-based LoRA loader for WAN models."""

from flax import nnx
from .lora_base import LoRABaseMixin
from .lora_pipeline import StableDiffusionLoraLoaderMixin
from ..models import lora_nnx
from .. import max_logging
from . import lora_conversion_utils


class Wan2_1NNXLoraLoader(LoRABaseMixin):
  """
  Handles loading LoRA weights into NNX-based WAN 2.1 model.
  Assumes WAN pipeline contains 'transformer'
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
      return lora_conversion_utils.translate_wan_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=scan_layers)

    if hasattr(pipeline, "transformer") and transformer_weight_name:
      max_logging.log(f"Merging LoRA into transformer with rank={rank}")
      h_state_dict, _ = lora_loader.lora_state_dict(lora_model_path, weight_name=transformer_weight_name, **kwargs)
      h_state_dict = lora_conversion_utils.preprocess_wan_lora_dict(h_state_dict)
      merge_fn(pipeline.transformer, h_state_dict, rank, scale, translate_fn, dtype=dtype)
    else:
      max_logging.log("transformer not found or no weight name provided for LoRA.")

    return pipeline


class Wan2_2NNXLoraLoader(LoRABaseMixin):
  """
  Handles loading LoRA weights into NNX-based WAN 2.2 model.
  Assumes WAN pipeline contains 'high_noise_transformer' and 'low_noise_transformer'
  attributes that are NNX Modules.
  """

  def load_lora_weights(
      self,
      pipeline: nnx.Module,
      lora_model_path: str,
      high_noise_weight_name: str,
      low_noise_weight_name: str,
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

    def translate_fn(nnx_path_str: str):
      return lora_conversion_utils.translate_wan_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=scan_layers)

    # Handle high noise model
    if hasattr(pipeline, "high_noise_transformer") and high_noise_weight_name:
      max_logging.log(f"Merging LoRA into high_noise_transformer with rank={rank}")
      h_state_dict, _ = lora_loader.lora_state_dict(lora_model_path, weight_name=high_noise_weight_name, **kwargs)
      h_state_dict = lora_conversion_utils.preprocess_wan_lora_dict(h_state_dict)
      merge_fn(pipeline.high_noise_transformer, h_state_dict, rank, scale, translate_fn, dtype=dtype)
    else:
      max_logging.log("high_noise_transformer not found or no weight name provided for LoRA.")

    # Handle low noise model
    if hasattr(pipeline, "low_noise_transformer") and low_noise_weight_name:
      max_logging.log(f"Merging LoRA into low_noise_transformer with rank={rank}")
      l_state_dict, _ = lora_loader.lora_state_dict(lora_model_path, weight_name=low_noise_weight_name, **kwargs)
      l_state_dict = lora_conversion_utils.preprocess_wan_lora_dict(l_state_dict)
      merge_fn(pipeline.low_noise_transformer, l_state_dict, rank, scale, translate_fn, dtype=dtype)
    else:
      max_logging.log("low_noise_transformer not found or no weight name provided for LoRA.")

    return pipeline
