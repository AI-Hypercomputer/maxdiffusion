# Copyright 2023 Google LLC
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
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze
from .lora_base import LoRABaseMixin
from ..models.lora import LoRALinearLayer, LoRAConv2DLayer
from .lora_conversion_utils import (
  _convert_non_diffusers_lora_to_diffusers,
  _maybe_map_sgm_blocks_to_diffusers,  
)
from ..models.modeling_flax_pytorch_utils import convert_lora_pytorch_state_dict_to_flax
from huggingface_hub.utils import validate_hf_hub_args

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

class StableDiffusionLoraLoaderMixin(LoRABaseMixin):
  r"""
  Load LoRA layers into Stable Diffusion [`UNet2DConditionModel`] and
  [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
  """

  _lora_lodable_modules = ["unet", "text_encoder"]
  unet_name = UNET_NAME
  text_encoder_name = TEXT_ENCODER_NAME

  def load_lora_weights(
      self,
      pretrained_model_name_or_path_or_dict: Union[str, Dict[str, jnp.ndarray]],
      params,
      adapter_name=None,
      **kwargs):
    """
    Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
    `self.text_encoder`.

    All kwargs are forwarded to `self.lora_state_dict`.

    See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is
    loaded.

    See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is
    loaded into `self.unet`.

    See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state
    dict is loaded into `self.text_encoder`.

    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
            See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        kwargs (`dict`, *optional*):
            See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        adapter_name (`str`, *optional*):
            Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
            `default_{i}` where i is the total number of adapters being loaded.
    """
    # if a dict is passed, copy it instead of modifying it inplace
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
      pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

    state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

    is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
    if not is_correct_format:
      raise ValueError("Invalid LoRA checkpoint.")
    
    unet_lora_params, rank = self.load_lora_into_unet(
      state_dict,
      network_alphas=network_alphas,
      unet=getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet,
      params=params,
      adapter_name=adapter_name,
      _pipeline=self,
    )
    return unfreeze(unet_lora_params), rank

  @classmethod
  def _get_lora_layer(cls, module_path, module, rank):
    is_conv = any('conv' in str_ for str_ in module_path)
    if is_conv:
      lora_module = LoRAConv2DLayer(
        out_features=module.features,
        rank=rank,
        kernel_size=module.kernel_size,
        strides=module.strides,
        padding=module.padding,
        input_dilation=module.input_dilation,
        kernel_dilation=module.kernel_dilation,
        feature_group_count=module.feature_group_count,
        dtype=module.dtype,
        weights_dtype=module.param_dtype,
        precision=module.precision,
        name="lora"
        )
    else:
      lora_module = LoRALinearLayer(
        out_features=module.features,
        rank=rank,
        dtype=module.dtype,
        weights_dtype=module.param_dtype,
        precision=module.precision,
        name="lora"
      )
    return lora_module

  @classmethod
  def make_lora_interceptor(
    cls,
    params_keys,
    rank
  ):
    tmp = []
    for layer_lora in params_keys:
      if 'lora' in layer_lora:
        new_layer_lora = layer_lora[:layer_lora.index('lora')]
        if new_layer_lora not in tmp:
          tmp.append(new_layer_lora)
    params_keys = tmp
    def _intercept(next_fn, args, kwargs, context):
      h = next_fn(*args, **kwargs)
      if context.method_name == '__call__':
        module_path = context.module.path
        if module_path in params_keys:
          lora_layer = cls._get_lora_layer(module_path, context.module, rank)
          return lora_layer(h, *args, **kwargs)
      return h
    return _intercept
  
  @classmethod
  @validate_hf_hub_args
  def lora_state_dict(
    cls,
    pretrained_model_name_or_path: str,
    **kwargs
  ):
    r"""
    Return state dict for lora weights and the network alphas.

    <Tip warning={true}>

    We support loading A1111 formatted LoRA checkpoints in a limited capacity.

    This function is experimental and might change in the future.

    </Tip>

    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
            Can be either:

                - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                  the Hub.
                - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                  with [`ModelMixin.save_pretrained`].
                - A [torch state
                  dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.

        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        subfolder (`str`, *optional*, defaults to `""`):
            The subfolder location of a model file within a larger model repository on the Hub or locally.
        weight_name (`str`, *optional*, defaults to None):
            Name of the serialized state dict file.
    """
    # Load the main state dict first which has the LoRA layers for either of
    # UNet and text encoder or both.
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

    network_alphas = None
    if all(
      (
        k.startswith("lora_te_")
        or k.startswith("lora_unet_")
        or k.startswith("lora_te1_")
        or k.startswith("lora_te2_")
      )
      for k in state_dict.keys()
    ):
      # Map SDXL blocks correctly.
      if unet_config is not None:
        # use unet config to remap block numbers
        state_dict = _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)
      state_dict, network_alphas = _convert_non_diffusers_lora_to_diffusers(state_dict)
    
    return state_dict, network_alphas

  @classmethod
  def load_lora_into_unet(
    cls,
    state_dict,
    network_alphas,
    unet,
    params,
    adapter_name=None,
    _pipeline=None
    ):
    """
    This will load the LoRA layers specified in `state_dict` into `unet`.

    Parameters:
        state_dict (`dict`):
            A standard state dict containing the lora layer parameters. The keys can either be indexed directly
            into the unet or prefixed with an additional `unet` which can be used to distinguish between text
            encoder lora layers.
        network_alphas (`Dict[str, float]`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the
            same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
            link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
        unet (`UNet2DConditionModel`):
            The UNet model to load the LoRA layers into.
        adapter_name (`str`, *optional*):
            Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
            `default_{i}` where i is the total number of adapters being loaded.
    """
    keys = list(state_dict.keys())
    only_text_encoder = all(key.startswith(cls.text_encoder_name) for key in keys)
    if not only_text_encoder:
      # Load the layers corresponding to Unet.
      unet_lora_params, rank = convert_lora_pytorch_state_dict_to_flax(state_dict, params["unet"])
    return unet_lora_params, rank