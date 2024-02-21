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

  def test_unet21_sharding_test(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","resolution=768"])
    config = pyconfig.config
    unet, params = FlaxUNet2DConditionModel.from_pretrained(
      config.pretrained_model_name_or_path, revision=config.revision, subfolder="unet", dtype=jnp.bfloat16, from_pt=config.from_pt
    )
    devices_array = max_utils.create_device_mesh(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    k = jax.random.key(0)
    tx = optax.adam(learning_rate=0.001)
    latents = jnp.ones((4, 4,96,96), dtype=jnp.float32)
    timesteps = jnp.ones((4,))
    encoder_hidden_states = jnp.ones((4, 77, 1024))

    variables = jax.jit(unet.init)(k, latents, timesteps, encoder_hidden_states)
    unboxed_abstract_state, state_mesh_annotations = max_utils.get_abstract_state(unet, tx, config, mesh, variables['params'])
    del variables
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

    state, state_mesh_shardings = max_utils.setup_initial_state(
      unet,
      tx,config,
      mesh,
      params,
      unboxed_abstract_state,
      state_mesh_annotations
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
if __name__ == '__main__':
  absltest.main()
