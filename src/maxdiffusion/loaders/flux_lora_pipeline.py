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

from typing import Union, Dict
from .lora_base import LoRABaseMixin
from ..models.lora import LoRALinearLayer, BaseLoRALayer
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze
from ..models.modeling_flax_pytorch_utils import convert_flux_lora_pytorch_state_dict_to_flax
from huggingface_hub.utils import validate_hf_hub_args
from maxdiffusion.models.modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor)
class FluxLoraLoaderMixin(LoRABaseMixin):

  _lora_lodable_modules = ["transformer", "text_encoder"]
  
  def load_lora_weights(
      self,
      config,
      pretrained_model_name_or_path_or_dict: Union[str, Dict[str, jnp.ndarray]],
      params,
      adapter_name=None,
      **kwargs
  ):
    state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

    params, rank, network_alphas = self.load_lora(
      config, 
      state_dict,
      params=params,
      adapter_name=adapter_name,
    )

    return params, rank, network_alphas

  def rename_for_interceptor(params_keys, network_alphas, adapter_name):
    new_params_keys = []
    new_network_alphas = {}
    lora_name = f"lora-{adapter_name}"
    for layer_lora in params_keys:
      if lora_name in layer_lora:
        new_layer_lora = layer_lora[: layer_lora.index(lora_name)]
        if new_layer_lora not in new_params_keys:
          new_params_keys.append(new_layer_lora)
          network_alpha = network_alphas[layer_lora]
          new_network_alphas[new_layer_lora] = network_alpha
    return new_params_keys, new_network_alphas

  @classmethod
  def make_lora_interceptor(cls, params, rank, network_alphas, adapter_name):
    network_alphas_for_interceptor = {}

    transformer_keys = flatten_dict(params["transformer"]).keys()
    lora_keys, transformer_alphas = cls.rename_for_interceptor(transformer_keys, network_alphas, adapter_name)
    network_alphas_for_interceptor.update(transformer_alphas)
  
    def _intercept(next_fn, args, kwargs, context):
      mod = context.module
      while mod is not None:
        if isinstance(mod, BaseLoRALayer):
          return next_fn(*args, **kwargs)
        mod = mod.parent
      h = next_fn(*args, **kwargs)
      if context.method_name == "__call__":
        module_path = context.module.path
        if module_path in lora_keys:
          lora_layer = cls._get_lora_layer(module_path, context.module, rank, network_alphas_for_interceptor, adapter_name)
          return lora_layer(h, *args, **kwargs)
      return h

    return _intercept

  @classmethod
  def _get_lora_layer(cls, module_path, module, rank, network_alphas, adapter_name):
    network_alpha = network_alphas.get(module_path, None)
    lora_module = LoRALinearLayer(
        out_features=module.features,
        rank=rank,
        network_alpha=network_alpha,
        dtype=module.dtype,
        weights_dtype=module.param_dtype,
        precision=module.precision,
        name=f"lora-{adapter_name}",
    )
    return lora_module

  @classmethod
  @validate_hf_hub_args
  def lora_state_dict(cls, pretrained_model_name_or_path: str, **kwargs):

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    unet_config = kwargs.pop("unet_config", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    resume_download = kwargs.pop("resume_download", False)

    allow_pickle = False
    if use_safetensors is None:
      use_safetensors = True
      allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    state_dict = cls._fetch_state_dict(
        pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path,
        weight_name=weight_name,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        use_auth_token=use_auth_token,
        revision=revision,
        subfolder=subfolder,
        user_agent=user_agent,
        allow_pickle=allow_pickle,
    )

    return state_dict
  
  @classmethod
  def load_lora(cls, config, state_dict, params, adapter_name=None):
    params, rank, network_alphas = convert_flux_lora_pytorch_state_dict_to_flax(config, state_dict, params, adapter_name)
    return params, rank, network_alphas