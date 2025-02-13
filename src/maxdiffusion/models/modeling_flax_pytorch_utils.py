# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch - Flax general utilities."""
import re

import jax
import jax.numpy as jnp
from flax.linen import Partitioned
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze
from jax.random import PRNGKey

from ..utils import logging


logger = logging.get_logger(__name__)


def rename_key(key):
  regex = r"\w+[.]\d+"
  pats = re.findall(regex, key)
  for pat in pats:
    key = key.replace(pat, "_".join(pat.split(".")))
  return key


#####################
# PyTorch => Flax #
#####################


# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
  """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
  # conv norm or layer norm
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)

  # rename attention layers
  if len(pt_tuple_key) > 1:
    for rename_from, rename_to in (
        ("to_out_0", "proj_attn"),
        ("to_k", "key"),
        ("to_v", "value"),
        ("to_q", "query"),
        ("txt_attn_proj", "txt_attn_proj"),
        ("img_attn_proj", "img_attn_proj"),
        ("txt_attn_qkv", "txt_attn_qkv"),
        ("img_attn_qkv", "img_attn_qkv"),
    ):
      if pt_tuple_key[-2] == rename_from:
        weight_name = pt_tuple_key[-1]
        weight_name = "kernel" if weight_name == "weight" else weight_name
        renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)
        if renamed_pt_tuple_key in random_flax_state_dict:
          if isinstance(random_flax_state_dict[renamed_pt_tuple_key], Partitioned):
            assert random_flax_state_dict[renamed_pt_tuple_key].value.shape == pt_tensor.T.shape
          else:
            assert random_flax_state_dict[renamed_pt_tuple_key].shape == pt_tensor.T.shape
          return renamed_pt_tuple_key, pt_tensor.T

  if (
      any("norm" in str_ for str_ in pt_tuple_key)
      and (pt_tuple_key[-1] == "bias")
      and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
      and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
  ):
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor
  elif pt_tuple_key[-1] in ["weight", "gamma"] and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict:
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor

  # embedding
  if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
    pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    return renamed_pt_tuple_key, pt_tensor

  # conv layer
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
    return renamed_pt_tuple_key, pt_tensor

  # linear layer
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight":
    pt_tensor = pt_tensor.T
    return renamed_pt_tuple_key, pt_tensor

  # old PyTorch layer norm weight
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
  if pt_tuple_key[-1] == "gamma":
    return renamed_pt_tuple_key, pt_tensor

  # old PyTorch layer norm bias
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
  if pt_tuple_key[-1] == "beta":
    return renamed_pt_tuple_key, pt_tensor

  return pt_tuple_key, pt_tensor


def get_network_alpha_value(pt_key, network_alphas):
  network_alpha_value = -1
  network_alpha_key = tuple(pt_key.split("."))
  for item in network_alpha_key:
    # alpha names for LoRA follow different convention for qkv values.
    # Ex:
    # conv layer - unet.down_blocks.0.downsamplers.0.conv.alpha
    # to_k_lora - unet.down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor.to_k_lora.down.weight.alpha
    if "lora" == item:
      network_alpha_key = network_alpha_key[: network_alpha_key.index(item)] + ("alpha",)
      break
    elif "lora" in item:
      network_alpha_key = network_alpha_key + ("alpha",)
      break
  network_alpha_key = ".".join(network_alpha_key)
  if network_alpha_key in network_alphas:
    network_alpha_value = network_alphas[network_alpha_key]
  return network_alpha_value


def create_flax_params_from_pytorch_state(
    pt_state_dict,
    unet_state_dict,
    text_encoder_state_dict,
    text_encoder_2_state_dict,
    network_alphas,
    adapter_name,
    is_lora=False,
):
  rank = None
  renamed_network_alphas = {}
  # Need to change some parameters name to match Flax names
  for pt_key, pt_tensor in pt_state_dict.items():
    network_alpha_value = get_network_alpha_value(pt_key, network_alphas)

    # rename text encoders fc1 lora layers.
    pt_key = pt_key.replace("lora_linear_layer", "lora")

    # only rename the unet keys, text encoders are already correct.
    if "unet" in pt_key:
      renamed_pt_key = rename_key(pt_key)
    else:
      renamed_pt_key = pt_key
    pt_tuple_key = tuple(renamed_pt_key.split("."))
    # conv
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
      pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
      pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
      flax_key_list = [*pt_tuple_key]
      flax_tensor = pt_tensor
      if "lora" in flax_key_list:
        flax_key_list[flax_key_list.index("lora")] = f"lora-{adapter_name}"
    else:
      flax_key_list = [*pt_tuple_key]
      if "text_encoder" in pt_tuple_key or "text_encoder_2" in pt_tuple_key:
        rename_from_to = (
            ("to_k_lora", ("k_proj", f"lora-{adapter_name}")),
            ("to_q_lora", ("q_proj", f"lora-{adapter_name}")),
            ("to_v_lora", ("v_proj", f"lora-{adapter_name}")),
            ("to_out_lora", ("out_proj", f"lora-{adapter_name}")),
            ("lora", f"lora-{adapter_name}"),
            ("weight", "kernel"),
        )
      # the unet
      else:
        rename_from_to = (
            ("to_k_lora", ("to_k", f"lora-{adapter_name}")),
            ("to_q_lora", ("to_q", f"lora-{adapter_name}")),
            ("to_v_lora", ("to_v", f"lora-{adapter_name}")),
            ("to_out_lora", ("to_out_0", f"lora-{adapter_name}")),
            ("lora", f"lora-{adapter_name}"),
            ("weight", "kernel"),
        )
      for rename_from, rename_to in rename_from_to:
        tmp = []
        for s in flax_key_list:
          if s == rename_from:
            if type(rename_to) is tuple:
              for s_in_tuple in rename_to:
                tmp.append(s_in_tuple)
            else:
              tmp.append(rename_to)
          else:
            tmp.append(s)
        flax_key_list = tmp

      flax_tensor = pt_tensor.T

    if is_lora:
      if "lora.up" in renamed_pt_key:
        rank = pt_tensor.shape[1]
    if "processor" in flax_key_list:
      flax_key_list.remove("processor")
    if "unet" in flax_key_list:
      flax_key_list.remove("unet")
      unet_state_dict[tuple(flax_key_list)] = jnp.asarray(flax_tensor)

    if "text_encoder" in flax_key_list:
      flax_key_list.remove("text_encoder")
      text_encoder_state_dict[tuple(flax_key_list)] = jnp.asarray(flax_tensor)

    if "text_encoder_2" in flax_key_list:
      flax_key_list.remove("text_encoder_2")
      text_encoder_2_state_dict[tuple(flax_key_list)] = jnp.asarray(flax_tensor)

    if network_alpha_value >= 0:
      renamed_network_alphas[tuple(flax_key_list)] = network_alpha_value
  return unet_state_dict, text_encoder_state_dict, text_encoder_2_state_dict, rank, renamed_network_alphas


def convert_flux_lora_pytorch_state_dict_to_flax(config, pt_state_dict, params, adapter_name):
  pt_state_dict = {k: v.float().numpy() for k, v in pt_state_dict.items()}
  transformer_params = flatten_dict(unfreeze(params["transformer"]))
  network_alphas = {}
  rank = None
  for pt_key, tensor in pt_state_dict.items():
    renamed_pt_key = rename_key(pt_key)
    renamed_pt_key = renamed_pt_key.replace("lora_unet_", "")
    renamed_pt_key = renamed_pt_key.replace("lora_down", f"lora-{adapter_name}.down")
    renamed_pt_key = renamed_pt_key.replace("lora_up", f"lora-{adapter_name}.up")

    if "double_blocks" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("double_blocks.", "double_blocks_")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora1.down", f"attn.i_proj.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora1.up", f"attn.i_proj.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora2.down", f"attn.e_proj.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora2.up", f"attn.e_proj.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora1.down", f"attn.i_qkv.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora1.up", f"attn.i_qkv.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora2.down", f"attn.e_qkv.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora2.up", f"attn.e_qkv.lora-{adapter_name}.up")

      renamed_pt_key = renamed_pt_key.replace("_img_attn_proj", ".attn.i_proj")
      renamed_pt_key = renamed_pt_key.replace("_img_attn_qkv", ".attn.i_qkv")
      renamed_pt_key = renamed_pt_key.replace("_img_mlp_0", ".img_mlp.layers_0")
      renamed_pt_key = renamed_pt_key.replace("_img_mlp_2", ".img_mlp.layers_2")
      renamed_pt_key = renamed_pt_key.replace("_img_mod_lin", ".img_norm1.lin")
      renamed_pt_key = renamed_pt_key.replace("_txt_attn_proj", ".attn.e_proj")
      renamed_pt_key = renamed_pt_key.replace("_txt_attn_qkv", ".attn.e_qkv")
      renamed_pt_key = renamed_pt_key.replace("_txt_mlp_0", ".txt_mlp.layers_0")
      renamed_pt_key = renamed_pt_key.replace("_txt_mlp_2", ".txt_mlp.layers_2")
      renamed_pt_key = renamed_pt_key.replace("_txt_mod_lin", ".txt_norm1.lin")
    elif "single_blocks" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("_linear1", ".linear1")
      renamed_pt_key = renamed_pt_key.replace("_linear2", ".linear2")
      renamed_pt_key = renamed_pt_key.replace("_modulation_lin", ".norm.lin")

    renamed_pt_key = renamed_pt_key.replace("weight", "kernel")

    pt_tuple_key = tuple(renamed_pt_key.split("."))
    if "alpha" in pt_tuple_key:
      pt_tuple_key = pt_tuple_key[:-1] + (f"lora-{adapter_name}", "down", "kernel")
      network_alphas[tuple([*pt_tuple_key])] = tensor.item()  # noqa: C409
      pt_tuple_key = pt_tuple_key[:-1] + (f"lora-{adapter_name}", "up", "kernel")
      network_alphas[tuple([*pt_tuple_key])] = tensor.item()  # noqa: C409
    else:
      if pt_tuple_key[-2] == "up":
        rank = tensor.shape[1]
      transformer_params[tuple([*pt_tuple_key])] = jnp.asarray(tensor.T, dtype=config.weights_dtype)  # noqa: C409

  params["transformer"] = unflatten_dict(transformer_params)

  return params, rank, network_alphas


def convert_flux_lora_pytorch_state_dict_to_flax(config, pt_state_dict, params, adapter_name):
  pt_state_dict = {k: v.float().numpy() for k, v in pt_state_dict.items()}
  transformer_params = flatten_dict(unfreeze(params["transformer"]))
  network_alphas = {}
  rank = None
  for pt_key, tensor in pt_state_dict.items():
    renamed_pt_key = rename_key(pt_key)
    renamed_pt_key = renamed_pt_key.replace("lora_unet_", "")
    renamed_pt_key = renamed_pt_key.replace("lora_down", f"lora-{adapter_name}.down")
    renamed_pt_key = renamed_pt_key.replace("lora_up", f"lora-{adapter_name}.up")

    if "double_blocks" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("double_blocks.", "double_blocks_")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora1.down", f"attn.i_proj.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora1.up", f"attn.i_proj.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora2.down", f"attn.e_proj.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.proj_lora2.up", f"attn.e_proj.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora1.down", f"attn.i_qkv.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora1.up", f"attn.i_qkv.lora-{adapter_name}.up")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora2.down", f"attn.e_qkv.lora-{adapter_name}.down")
      renamed_pt_key = renamed_pt_key.replace("processor.qkv_lora2.up", f"attn.e_qkv.lora-{adapter_name}.up")

      renamed_pt_key = renamed_pt_key.replace("_img_attn_proj", ".attn.i_proj")
      renamed_pt_key = renamed_pt_key.replace("_img_attn_qkv", ".attn.i_qkv")
      renamed_pt_key = renamed_pt_key.replace("_img_mlp_0", ".img_mlp.layers_0")
      renamed_pt_key = renamed_pt_key.replace("_img_mlp_2", ".img_mlp.layers_2")
      renamed_pt_key = renamed_pt_key.replace("_img_mod_lin", ".img_norm1.lin")
      renamed_pt_key = renamed_pt_key.replace("_txt_attn_proj", ".attn.e_proj")
      renamed_pt_key = renamed_pt_key.replace("_txt_attn_qkv", ".attn.e_qkv")
      renamed_pt_key = renamed_pt_key.replace("_txt_mlp_0", ".txt_mlp.layers_0")
      renamed_pt_key = renamed_pt_key.replace("_txt_mlp_2", ".txt_mlp.layers_2")
      renamed_pt_key = renamed_pt_key.replace("_txt_mod_lin", ".txt_norm1.lin")
    elif "single_blocks" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("_linear1", ".linear1")
      renamed_pt_key = renamed_pt_key.replace("_linear2", ".linear2")
      renamed_pt_key = renamed_pt_key.replace("_modulation_lin", ".norm.lin")

    renamed_pt_key = renamed_pt_key.replace("weight", "kernel")

    pt_tuple_key = tuple(renamed_pt_key.split("."))
    if "alpha" in pt_tuple_key:
      pt_tuple_key = pt_tuple_key[:-1] + (f"lora-{adapter_name}", "down", "kernel")
      network_alphas[tuple([*pt_tuple_key])] = tensor.item()  # noqa: C409
      pt_tuple_key = pt_tuple_key[:-1] + (f"lora-{adapter_name}", "up", "kernel")
      network_alphas[tuple([*pt_tuple_key])] = tensor.item()  # noqa: C409
    else:
      if pt_tuple_key[-2] == "up":
        rank = tensor.shape[1]
      transformer_params[tuple([*pt_tuple_key])] = jnp.asarray(tensor.T, dtype=config.weights_dtype)  # noqa: C409

  params["transformer"] = unflatten_dict(transformer_params)

  return params, rank, network_alphas


def convert_lora_pytorch_state_dict_to_flax(pt_state_dict, params, network_alphas, adapter_name):
  # Step 1: Convert pytorch tensor to numpy
  # sometimes we load weights in bf16 and numpy doesn't support it
  pt_state_dict = {k: v.float().numpy() for k, v in pt_state_dict.items()}

  unet_params = flatten_dict(unfreeze(params["unet"]))
  text_encoder_params = flatten_dict(unfreeze(params["text_encoder"]))
  if "text_encoder_2" in params.keys():
    text_encoder_2_params = flatten_dict(unfreeze(params["text_encoder_2"]))
  else:
    text_encoder_2_params = None
  (unet_state_dict, text_encoder_state_dict, text_encoder_2_state_dict, rank, network_alphas) = (
      create_flax_params_from_pytorch_state(
          pt_state_dict, unet_params, text_encoder_params, text_encoder_2_params, network_alphas, adapter_name, is_lora=True
      )
  )
  params["unet"] = unflatten_dict(unet_state_dict)
  params["text_encoder"] = unflatten_dict(text_encoder_state_dict)
  if text_encoder_2_state_dict is not None:
    params["text_encoder_2"] = unflatten_dict(text_encoder_2_state_dict)

  return params, rank, network_alphas


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42):
  # Step 1: Convert pytorch tensor to numpy
  pt_state_dict = {k: v.float().numpy() for k, v in pt_state_dict.items()}

  # Step 2: Since the model is stateless, run eval_shape to get the pytree structure
  random_flax_params = flax_model.init_weights(PRNGKey(init_key), eval_only=True)

  random_flax_state_dict = flatten_dict(random_flax_params)
  flax_state_dict = {}
  # Keep in cpu. Will get moved to host later.
  cpu = jax.local_devices(backend="cpu")[0]
  # Need to change some parameters name to match Flax names
  for pt_key, pt_tensor in pt_state_dict.items():
    renamed_pt_key = rename_key(pt_key)
    pt_tuple_key = tuple(renamed_pt_key.split("."))

    # Correctly rename weight parameters
    flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict)

    # also add unexpected weight so that warning is thrown
    flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

  return unflatten_dict(flax_state_dict)
