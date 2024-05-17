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

""" Smoke test """

"""
{"_class_name": "UNet2DConditionModel", "_diffusers_version": "0.19.0.dev0", 
"act_fn": "silu", 
"addition_embed_type": "text_time", 
"addition_embed_type_num_heads": 64, 
"addition_time_embed_dim": 256, 
"attention_head_dim": [5, 10, 20], 
"block_out_channels": [320, 640, 1280], 
"center_input_sample": false, 
"class_embed_type": null, 
"class_embeddings_concat": false, 
"conv_in_kernel": 3, "conv_out_kernel": 3, 
"cross_attention_dim": 2048, 
"cross_attention_norm": null, 
"down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], 
"downsample_padding": 1, 
"dual_cross_attention": false, 
"encoder_hid_dim": null, 
"encoder_hid_dim_type": null, 
"flip_sin_to_cos": true, 
"freq_shift": 0, 
"in_channels": 4, 
"layers_per_block": 2, 
"mid_block_only_cross_attention": null, 
"mid_block_scale_factor": 1, 
"mid_block_type": "UNetMidBlock2DCrossAttn", 
"norm_eps": 1e-05, "norm_num_groups": 32, 
"num_attention_heads": null, 
"num_class_embeds": null, 
"only_cross_attention": false, 
"out_channels": 4, 
"projection_class_embeddings_input_dim": 2816, 
"resnet_out_scale_factor": 1.0, 
"resnet_skip_time_act": false, 
"resnet_time_scale_shift": "default", 
"sample_size": 128, "time_cond_proj_dim": null, 
"time_embedding_act_fn": null, 
"time_embedding_dim": null, 
"time_embedding_type": "positional", 
"timestep_post_act": null, 
"transformer_layers_per_block": [1, 2, 10], 
"up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"], 
"upcast_attention": null, 
"use_linear_projection": true}

"""

import os
import unittest
from absl.testing import absltest

import jax
import jax.numpy as jnp
from ..import max_utils
from ..import pyconfig
from maxdiffusion import FlaxUNet2DConditionModel
from flax.training import train_state
import optax
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from flax import traverse_util
from jax.sharding import PartitionSpec as P
from ..max_utils import get_dtype

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def init_fn(params, model, optimizer):
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer)
  return state


class UnetTest(unittest.TestCase):
  """Test Unet sharding"""
  def setUp(self):
    UnetTest.dummy_data = {}

  def test_unet_config_params(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "norm_num_groups=32"])

    config = pyconfig.config
    unet, _ = FlaxUNet2DConditionModel.from_pretrained(
      config.pretrained_model_name_or_path,
      revision="refs/pr/95",
      subfolder="unet",
      dtype=jnp.bfloat16,
      from_pt=config.from_pt,
      norm_num_groups=config.norm_num_groups
    )

    assert unet.config.norm_num_groups == config.norm_num_groups

  def test_unetxl_sharding_test(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
      "revision=refs/pr/95","dtype=bfloat16","resolution=1024"])
    config = pyconfig.config
    unet, params = FlaxUNet2DConditionModel.from_pretrained(
      config.pretrained_model_name_or_path, revision=config.revision, subfolder="unet", dtype=jnp.bfloat16, from_pt=config.from_pt
    )

    weight_dtype = get_dtype(config)

    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)
    devices_array = max_utils.create_device_mesh(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    k = jax.random.key(0)
    variables = unet.init_weights(k, eval_only=True)
    unboxed_abstract_state, state_mesh_annotations = max_utils.get_abstract_state(unet, None, config, mesh, variables, training=False)
    del variables
    # breakpoint()
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

    state, state_mesh_shardings = max_utils.setup_initial_state(unet,None,config,mesh,params,unboxed_abstract_state,state_mesh_annotations, training=False)


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

  
if __name__ == '__main__':
  absltest.main()
