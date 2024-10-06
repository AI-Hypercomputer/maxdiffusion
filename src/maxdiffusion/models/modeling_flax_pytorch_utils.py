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
from flax.core.frozen_dict import unfreeze, freeze
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
        ("to_k_lora", "to_k_lora"),
        ("to_k_lora", "to_k_lora"),
        ("to_q_lora", "to_q_lora"),
        ("to_v_lora", "to_v_lora"),
        ("to_out_lora", "to_out_lora")
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

def create_flax_params_from_pytorch_state(pt_state_dict, flax_state_dict, is_lora=False):
    rank = None
    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, flax_state_dict)
        
        if is_lora:
            if "lora.up" in renamed_pt_key:
                rank = pt_tensor.shape[1]
            
            flax_key_list = list(flax_key)
            flax_key_list.remove("processor")
            flax_key_list.remove("unet")
            flax_key = tuple(flax_key_list)


        if flax_key in flax_state_dict:
            if flax_tensor.shape != flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
    
    return flax_state_dict, rank

def convert_lora_pytorch_state_dict_to_flax(pt_state_dict, unet_params):
    # Step 1: Convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    unet_params = flatten_dict(unfreeze(unet_params))
    flax_state_dict, rank = create_flax_params_from_pytorch_state(pt_state_dict, unet_params,is_lora=True)

    return freeze(unflatten_dict(flax_state_dict)), rank

def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42):
  # Step 1: Convert pytorch tensor to numpy
  pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

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
