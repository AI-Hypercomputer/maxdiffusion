# Copyright 2025 Google LLC
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

from math import e
from types import NoneType
from typing import Any, Dict
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from flax.linen import partitioning as nn_partitioning

from ...pyconfig import HyperParameters
from ...max_utils import (
    create_device_mesh,
    setup_initial_state,
    get_memory_allocations
)
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import os
import json
import functools
import orbax.checkpoint as ocp

def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids):
    print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)
    print("fractional_coords.shape: ", fractional_coords.shape, fractional_coords.dtype)
    print("latents.shape: ", latents.shape, latents.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    print("segment_ids.shape: ", segment_ids.shape, segment_ids.dtype)
    print("encoder_attention_segment_ids.shape: ", encoder_attention_segment_ids.shape, encoder_attention_segment_ids.dtype)
class LTXVideoPipeline:
  def __init__(
    self,
    transformer: Transformer3DModel,
    devices_array: np.array,
    mesh: Mesh,
    config: HyperParameters,
    states: Dict[Any, Any] = None,
    state_shardings: Dict[Any, Any] = NoneType
  ):
    self.transformer = transformer
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.p_run_inference = None
    self.states = states    ## check is it okay to keep this as a paramter?
    self.state_shardings = state_shardings


  @classmethod
  def from_pretrained(cls, config: HyperParameters):
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    base_dir = os.path.dirname(__file__)

    ##load in model config
    config_path = os.path.join(base_dir, "../../models/ltx_video/xora_v1.2-13B-balanced-128.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)
    relative_ckpt_path = model_config["ckpt_path"]

    ignored_keys = ["_class_name", "_diffusers_version", "_name_or_path", "causal_temporal_positioning", "in_channels", "ckpt_path"]
    in_channels = model_config["in_channels"]
    for name in ignored_keys:
        if name in model_config:
            del model_config[name]
    transformer = Transformer3DModel(**model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh=mesh)
    transformer_param_shapes = transformer.init_weights(in_channels, model_config['caption_channels'], eval_only = True)
    weights_init_fn = functools.partial(
        transformer.init_weights,
        in_channels,
        model_config['caption_channels'],
        eval_only = True
    )

    absolute_ckpt_path = os.path.abspath(relative_ckpt_path)

    checkpoint_manager = ocp.CheckpointManager(absolute_ckpt_path)
    transformer_state, transformer_state_shardings = setup_initial_state(
        model=transformer,
        tx=None,
        config=config,
        mesh=mesh,
        weights_init_fn=weights_init_fn,
        checkpoint_manager=checkpoint_manager,
        checkpoint_item=" ",
        model_params=None,
        training=False,
    )
    transformer_state = jax.device_put(transformer_state, transformer_state_shardings)
    get_memory_allocations()

    states = {}
    state_shardings = {}
    state_shardings["transformer"] = transformer_state_shardings
    states["transformer"] = transformer_state

    return LTXVideoPipeline(
      transformer=transformer,
      devices_array=devices_array,
      mesh=mesh,
      config=config,
      states=states,
      state_shardings=state_shardings
    )


  ##change the paramters of these, currently pass in dummy inputs
  def __call__(
    self,
    example_inputs
  ):
    data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
    latents = jax.device_put(example_inputs["latents"], data_sharding)
    prompt_embeds = jax.device_put(example_inputs["prompt_embeds"], data_sharding)
    fractional_coords = jax.device_put(example_inputs["fractional_coords"], data_sharding)
    noise_cond = jax.device_put(example_inputs["timestep"], data_sharding)
    segment_ids = jax.device_put(example_inputs["segment_ids"], data_sharding)
    encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"], data_sharding)
    validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids)
    p_run_inference = jax.jit(
      functools.partial(
          run_inference,
          transformer=self.transformer,
          config=self.config,
          mesh=self.mesh,
          latents=latents,
          fractional_cords=fractional_coords,
          prompt_embeds=prompt_embeds,
          timestep = noise_cond,
          segment_ids=segment_ids,
          encoder_attention_segment_ids=encoder_attention_segment_ids
      ),
      in_shardings=(self.state_shardings,),
      out_shardings=None,
    )
    with self.mesh:
      noise_pred = p_run_inference(self.states).block_until_ready()

    return noise_pred


def transformer_forward_pass(
  step,
  args,
  transformer,
  fractional_cords,
  prompt_embeds,
  segment_ids,
  encoder_attention_segment_ids):
  import pdb; pdb.set_trace()

  latents, state, noise_cond = args
  noise_pred = transformer.apply(
      {"params": state.params},
      hidden_states=latents,
      indices_grid=fractional_cords,
      encoder_hidden_states=prompt_embeds,
      timestep=noise_cond,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids
  )  #need .param here?
  return noise_pred, state, noise_cond


def run_inference(
  states, transformer, config, mesh, latents, fractional_cords, prompt_embeds, timestep, segment_ids, encoder_attention_segment_ids
  ):
    transformer_state = states["transformer"]
    transformer_forward_pass_p = functools.partial(
      transformer_forward_pass,
      transformer=transformer,
      fractional_cords=fractional_cords,
      prompt_embeds=prompt_embeds,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids
    )
      ##handle whether there's guidance here?
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        latents, transformer_state, timestep = jax.lax.fori_loop(0, 1, transformer_forward_pass_p, (latents, transformer_state, timestep)) #TODO: change 1 to num_inference_step
    return latents
