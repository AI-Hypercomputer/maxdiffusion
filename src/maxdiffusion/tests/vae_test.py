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
from maxdiffusion import FlaxAutoencoderKL
from flax.training import train_state
import optax
from jax.sharding import Mesh, PartitionSpec, NamedSharding


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def init_fn(params, model, optimizer):
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer)
  return state

class VaeTest(unittest.TestCase):
  """Test Unet sharding"""
  def setUp(self):
    VaeTest.dummy_data = {}

  def test_vae21_sharding_test(self):
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","resolution=768"])
    config = pyconfig.config
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae", dtype=jnp.bfloat16, from_pt=config.from_pt
    )
    devices_array = max_utils.create_device_mesh(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    k = jax.random.key(0)
    tx = optax.adam(learning_rate=0.001)
    latents = jnp.ones((4,4,96,96), dtype=jnp.float32)

    variables = jax.jit(vae.init)(k, latents)
    unboxed_abstract_state, state_mesh_annotations = max_utils.get_abstract_state(vae, tx, config, mesh, variables['params'])
    del variables
    qkv_sharding =PartitionSpec(None, None)
    conv_sharding = PartitionSpec(None, None, None, 'fsdp')
    assert state_mesh_annotations.params['decoder']['mid_block']['resnets_0']['conv1']['kernel'] == conv_sharding
    assert state_mesh_annotations.params['decoder']['mid_block']['resnets_0']['conv2']['kernel'] == conv_sharding
    assert state_mesh_annotations.params['decoder']['mid_block']['attentions_0']['key']['kernel'] == qkv_sharding
    assert state_mesh_annotations.params['decoder']['mid_block']['attentions_0']['query']['kernel'] == qkv_sharding
    assert state_mesh_annotations.params['decoder']['mid_block']['attentions_0']['value']['kernel'] == qkv_sharding

    vae_state, vae_state_state_mesh_shardings = max_utils.setup_initial_state(
      vae,
      tx,config,
      mesh,
      vae_params,
      unboxed_abstract_state,
      state_mesh_annotations
    )

    qkv_named_sharding = NamedSharding(mesh, qkv_sharding)
    conv_named_sharding = NamedSharding(mesh, conv_sharding)
    assert vae_state_state_mesh_shardings.params['decoder']['mid_block']['resnets_0']['conv1']['kernel'] == conv_named_sharding
    assert vae_state_state_mesh_shardings.params['decoder']['mid_block']['resnets_0']['conv2']['kernel'] == conv_named_sharding
    assert vae_state_state_mesh_shardings.params['decoder']['mid_block']['attentions_0']['key']['kernel'] == qkv_named_sharding
    assert vae_state_state_mesh_shardings.params['decoder']['mid_block']['attentions_0']['query']['kernel'] == qkv_named_sharding
    assert vae_state_state_mesh_shardings.params['decoder']['mid_block']['attentions_0']['value']['kernel'] == qkv_named_sharding

if __name__ == '__main__':
  absltest.main()
