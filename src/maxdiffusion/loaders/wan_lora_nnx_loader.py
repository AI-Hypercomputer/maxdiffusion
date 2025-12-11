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

import re
from flax import nnx
import jax
import jax.numpy as jnp
from .lora_base import LoRABaseMixin
from .lora_pipeline import StableDiffusionLoraLoaderMixin
from ..models import lora_nnx
from .. import max_logging

class WanNnxLoraLoader(LoRABaseMixin):
  """
  Handles loading LoRA weights into NNX-based WAN models.
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
      rng: jax.Array = None,
      **kwargs,
  ):
    """
    Injects LoRA layers into the pipeline and loads weights
    from a checkpoint.
    """
    if rng is None:
      rng = jax.random.key(0)

    lora_loader = StableDiffusionLoraLoaderMixin()

    # Handle high noise model
    if hasattr(pipeline, "high_noise_transformer") and high_noise_weight_name:
        max_logging.log(f"Injecting LoRA into high_noise_transformer with rank={rank}")
        lora_nnx.inject_lora(
            pipeline.high_noise_transformer, rank=rank, scale=scale, rngs=nnx.Rngs(rng), target_linear=True, target_conv=True
        )
        h_state_dict, h_alphas = lora_loader.lora_state_dict(
            lora_model_path, weight_name=high_noise_weight_name, **kwargs
        )
        self._assign_weights_to_nnx_model(pipeline.high_noise_transformer, h_state_dict, h_alphas if h_alphas else {})
    else:
        max_logging.warning("high_noise_transformer not found or no weight name provided for LoRA.")

    # Handle low noise model
    if hasattr(pipeline, "low_noise_transformer") and low_noise_weight_name:
        max_logging.log(f"Injecting LoRA into low_noise_transformer with rank={rank}")
        lora_nnx.inject_lora(
            pipeline.low_noise_transformer, rank=rank, scale=scale, rngs=nnx.Rngs(rng), target_linear=True, target_conv=True
        )
        l_state_dict, l_alphas = lora_loader.lora_state_dict(
            lora_model_path, weight_name=low_noise_weight_name, **kwargs
        )
        self._assign_weights_to_nnx_model(pipeline.low_noise_transformer, l_state_dict, l_alphas if l_alphas else {})
    else:
        max_logging.warning("low_noise_transformer not found or no weight name provided for LoRA.")

    return pipeline

  def _assign_weights_to_nnx_model(self, model: nnx.Module, state_dict: dict, network_alphas: dict):
    """
    Assigns weights from a Diffusers-formatted state dict to
    injected LoRALinear/LoRAConv layers in an NNX model.
    """
    lora_params = {}
    for k, v in state_dict.items():
      m = re.match(r"^(.*?)_lora\.(down|up)\.weight$", k)
      if not m:
        m = re.match(r"^(.*?)\.lora\.(down|up)\.weight$", k)

      if m:
        module_path_str, weight_type = m.group(1), m.group(2)
        if module_path_str not in lora_params:
          lora_params[module_path_str] = {}
        lora_params[module_path_str][weight_type] = jnp.array(v)
      else:
        max_logging.warning(f"Could not parse LoRA key: {k}")

    assigned_count = 0
    for path, submodule in nnx.iter_graph(model):
        if isinstance(submodule, (lora_nnx.LoRALinear, lora_nnx.LoRAConv)):
            nnx_path_str = ".".join(map(str, path))
            
            matched_key = None
            if nnx_path_str in lora_params:
              matched_key = nnx_path_str
            else:
              # Fallback: check if any param key matches end of nnx path
              for k in lora_params:
                if nnx_path_str.endswith(k) or k.endswith(nnx_path_str):
                   matched_key = k
                   break
            
            if matched_key and matched_key in lora_params:
              weights = lora_params[matched_key]
              if "down" in weights and "up" in weights:
                  if isinstance(submodule, lora_nnx.LoRALinear):
                      submodule.A.value = weights["down"].T
                      submodule.B.value = weights["up"].T
                      assigned_count +=1
                  elif isinstance(submodule, lora_nnx.LoRAConv):
                      submodule.down.kernel.value = weights["down"]
                      submodule.up.kernel.value = weights["up"]
                      assigned_count += 1
                  
                  pass
              else:
                  max_logging.warning(f"LoRA weights for {matched_key} incomplete.")
    max_logging.log(f"Assigned weights to {assigned_count} LoRA layers in {type(model)}.")
