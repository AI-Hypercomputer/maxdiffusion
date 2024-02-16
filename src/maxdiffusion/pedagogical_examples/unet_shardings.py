#!/usr/bin/python3

"""
 Copyright 2024 Google LLC

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

'''This script is used an example of how to shard the UNET on TPU.'''

import os
from absl import app
from typing import Sequence

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.compilation_cache import compilation_cache as cc
from maxdiffusion.models import FlaxUNet2DConditionModel

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_dtype,
  get_abstract_state,
  setup_initial_state
)

from flax import traverse_util

cc.initialize_cache(os.path.expanduser("~/jax_cache"))

def run(config):
  rng = jax.random.PRNGKey(config.seed)

  # Creates mesh using number of devices available
  # and ici/dcn parallelism rules
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  weight_dtype = get_dtype(config)

  # Load the UNET from the checkpoint
  unet, params = FlaxUNet2DConditionModel.from_pretrained(
    config.pretrained_model_name_or_path,
    revision=config.revision,
    dtype=weight_dtype,
    subfolder="unet",
    split_head_dim=True
  )
  params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)

  # Initialize the model in order to "activate" the PartitionSpecs
  unet_variables = jax.jit(unet.init_weights)(rng)
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(unet, None, config, mesh, unet_variables, training=False)

  # get_abstract_state performs:
  # 1. shape inference (https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html)
  # 2. turns logical annotations to mesh annotations.
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(unet, None, config, mesh, unet_variables, training=False)

  # We will use the pretrained weights for inference, so this can be deleted.
  del unet_variables

  # Validate state_mesh_annotations based on logical_axis_rules
  conv_sharding = PartitionSpec(None, None, None, 'fsdp')
  qkv_sharding = PartitionSpec(None, None)
  to_out_sharding = PartitionSpec(None, None)
  time_emb_proj_sharding = PartitionSpec()

  assert state_mesh_annotations.params['down_blocks_0']['resnets_0']['time_emb_proj']['kernel'] == time_emb_proj_sharding
  assert state_mesh_annotations.params['down_blocks_0']['downsamplers_0']['conv']['kernel'] == conv_sharding
  assert state_mesh_annotations.params['down_blocks_0']['resnets_0']['conv1']['kernel'] == conv_sharding
  assert state_mesh_annotations.params['down_blocks_0']['resnets_0']['conv2']['kernel'] == conv_sharding
  assert state_mesh_annotations.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_q']['kernel'] == qkv_sharding
  assert state_mesh_annotations.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_k']['kernel'] == qkv_sharding
  assert state_mesh_annotations.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_v']['kernel'] == qkv_sharding
  assert state_mesh_annotations.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_out_0']['kernel'] == to_out_sharding

  # initializes state with state_mesh_shardings.
  state, state_mesh_shardings = setup_initial_state(
    unet,
    None,
    config,
    mesh,
    params,
    unboxed_abstract_state,
    state_mesh_annotations,
    training=False
  )

  # Validate named shardings.
  conv_named_sharding = NamedSharding(mesh, conv_sharding)
  qkv_named_sharding = NamedSharding(mesh, qkv_sharding)
  to_out_named_sharding = NamedSharding(mesh, to_out_sharding)
  time_emb_proj_named_sharding = NamedSharding(mesh, time_emb_proj_sharding)

  assert state_mesh_shardings.params['down_blocks_0']['resnets_0']['time_emb_proj']['kernel'] == time_emb_proj_named_sharding
  assert state_mesh_shardings.params['down_blocks_0']['downsamplers_0']['conv']['kernel'] == conv_named_sharding
  assert state_mesh_shardings.params['down_blocks_0']['resnets_0']['conv1']['kernel'] == conv_named_sharding
  assert state_mesh_shardings.params['down_blocks_0']['resnets_0']['conv2']['kernel'] == conv_named_sharding
  assert state_mesh_shardings.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_q']['kernel'] == qkv_named_sharding
  assert state_mesh_shardings.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_k']['kernel'] == qkv_named_sharding
  assert state_mesh_shardings.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_v']['kernel'] == qkv_named_sharding
  assert state_mesh_shardings.params['down_blocks_1']['attentions_1']['transformer_blocks_0']['attn1']['to_out_0']['kernel'] == to_out_named_sharding

  # Validate weights are sharded and distributed across devices.
  flat_params = traverse_util.flatten_dict(state.params, sep='/')
  sized_params = jax.tree_util.tree_map(lambda x: x.device_buffers[0].size/x.size, flat_params)
  assert sized_params['down_blocks_0/resnets_0/time_emb_proj/kernel'] == 1.
  assert sized_params['down_blocks_0/downsamplers_0/conv/kernel'] == 1.
  assert sized_params['down_blocks_0/resnets_0/conv1/kernel'] == 1.
  assert sized_params['down_blocks_0/resnets_0/conv2/kernel'] == 1.
  assert sized_params['down_blocks_1/attentions_1/transformer_blocks_0/attn1/to_k/kernel'] == 1.
  assert sized_params['down_blocks_1/attentions_1/transformer_blocks_0/attn1/to_q/kernel'] == 1.
  assert sized_params['down_blocks_1/attentions_1/transformer_blocks_0/attn1/to_v/kernel'] == 1.
  assert sized_params['down_blocks_1/attentions_1/transformer_blocks_0/attn1/to_out_0/kernel'] == 1.

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

# Run via:
# python src/maxdiffusion/pedagogical_examples/unet_shardings.py src/diffusers/configs/base_xl.yml
if __name__ == "__main__":
  app.run(main)
