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
import inspect

from regex import F
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from typing import Optional, Union, List
import torch
from maxdiffusion.checkpointing import checkpointing_utils
from flax.linen import partitioning as nn_partitioning

from ...pyconfig import HyperParameters
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState
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
import pickle
import orbax.checkpoint as ocp
import os

class PickleCheckpointHandler(ocp.CheckpointHandler):
    def save(self, directory: str, item, args=None):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(item, f)

    def restore(self, directory: str, args=None):
        with open(os.path.join(directory, 'checkpoint.pkl'), 'rb') as f:
            return pickle.load(f)

    def structure(self, directory: str):
        return {}  # not needed for simple pickle-based handling
def save_tensor_dict(tensor_dict, timestep):
    base_dir = os.path.dirname(__file__)
    local_path = os.path.join(base_dir, f"schedulerTest{timestep}")
    
    try:
        torch.save(tensor_dict, local_path)
        print(f"Dictionary of tensors saved to: {local_path}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")
        raise

def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids):
    print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)
    print("fractional_coords.shape: ", fractional_coords.shape, fractional_coords.dtype)
    print("latents.shape: ", latents.shape, latents.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    # print("segment_ids.shape: ", segment_ids.shape, segment_ids.dtype)
    print("encoder_attention_segment_ids.shape: ", encoder_attention_segment_ids.shape, encoder_attention_segment_ids.dtype)
    
def get_scheduler_config():  #got parameters from maxdiffusion scheduler test, change later
    config = {
      "_class_name": 'FlaxEulerDiscreteScheduler',
      "_diffusers_version": "0.33.0.dev0",
      "beta_end": 0.02,
      "beta_schedule": "linear",
      "beta_start": 0.0001,
      "disable_corrector": [],
      "dynamic_thresholding_ratio": 0.995,
      "final_sigmas_type": "zero",
      "flow_shift": 3.0,
      "lower_order_final": True,
      "num_train_timesteps": 1000,
      "predict_x0": True,
      "prediction_type": "epsilon",
      "rescale_zero_terminal_snr": False,
      "rescale_betas_zero_snr": False,
      "sample_max_value": 1.0,
      "solver_order": 2,
      "solver_p": None,
      "solver_type": "bh2",
      "steps_offset": 0,
      "thresholding": False,
      "timestep_spacing": "trailing",
      "trained_betas": None,
      "use_beta_sigmas": False,
      "use_exponential_sigmas": False,
      "use_flow_sigmas": True,
      "use_karras_sigmas": False
    }
    return config
 
def prepare_extra_step_kwargs(generator):
    extra_step_kwargs = {}
    extra_step_kwargs["generator"] = generator
    return extra_step_kwargs  

# def retrieve_timesteps(
#     scheduler,
#     scheduler_state,
#     latents_shape,
#     num_inference_steps: Optional[int] = None,
#     timesteps: Optional[List[int]] = None,
#     skip_initial_inference_steps: int = 0,
#     skip_final_inference_steps: int = 0,
#     **kwargs,
# ):
#   # if timesteps is not None:       #this part currently doesn't work, cause scheduler doesn't support custom timestep schedulers
#   #   accepts_timesteps = "timesteps" in set(
#   #     inspect.signature(scheduler.set_timesteps).parameters.keys()
#   #   )
#   #   if not accepts_timesteps:
#   #     raise ValueError(
#   #       f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
#   #       f" timestep schedules. Please check whether you are using the correct scheduler."
#   #     )
#   #   scheduler_state = scheduler.set_timesteps(
#   #     scheduler_state, num_inference_steps=num_inference_steps, shape=latents_shape
#   #   )
#   #   timesteps = scheduler_state.timesteps
#   #   num_inference_steps = len(timesteps)
#   # else:
#   #   scheduler_state = scheduler.set_timesteps(
#   #     scheduler_state, num_inference_steps=num_inference_steps, shape=latents_shape
#   #   )
#   #   timesteps = scheduler_state.timesteps

#   #   if (
#   #       skip_initial_inference_steps < 0
#   #       or skip_final_inference_steps < 0
#   #       or skip_initial_inference_steps + skip_final_inference_steps
#   #       >= num_inference_steps
#   #   ):
#   #       raise ValueError(
#   #           "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
#   #       )

#   #   timesteps = timesteps[
#   #       skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
#   #   ]
#   #   scheduler_state = scheduler.set_timesteps(
#   #     scheduler_state, num_inference_steps=num_inference_steps, shape=latents_shape
#   #   )
#   #   num_inference_steps = len(timesteps)
#     scheduler_state = scheduler.set_timesteps(
#       scheduler_state, num_inference_steps=num_inference_steps, shape=latents_shape
#     )

#     return num_inference_steps, scheduler_state



  


class LTXVideoPipeline:
  def __init__(
    self,
    transformer: Transformer3DModel,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state: UniPCMultistepSchedulerState,
    devices_array: np.array,
    mesh: Mesh,
    config: HyperParameters,
    states: Dict[Any, Any] = None,
    state_shardings: Dict[Any, Any] = NoneType,
  ):
    self.transformer = transformer
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.p_run_inference = None
    self.states = states    ## check is it okay to keep this as a paramter?
    self.state_shardings = state_shardings
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state


  
  
  @classmethod
  def load_scheduler(cls):

    # scheduler_config = get_scheduler_config()
    # scheduler = FlaxUniPCMultistepScheduler(**scheduler_config)
    # scheduler_state = scheduler.create_state()
    scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_pretrained(
       "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        subfolder="scheduler",
        flow_shift=3.0  # 5.0 for 720p, 3.0 for 480p
    )
    return scheduler, scheduler_state


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
    transformer = Transformer3DModel(**model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh = mesh) #change this sharding back
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


    #initialize scheduler

    
    scheduler, scheduler_state = cls.load_scheduler()
    scheduler_checkpointer = ocp.Checkpointer(PickleCheckpointHandler())
    base_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(base_dir, "scheduler_ckpt")
    #scheduler_checkpointer.save(ckpt_path, scheduler_state)
    #scheduler_state = scheduler_checkpointer.restore(ckpt_path)
    return LTXVideoPipeline(
      transformer=transformer,
      scheduler=scheduler,
      scheduler_state=scheduler_state,
      devices_array=devices_array,
      mesh=mesh,
      config=config,
      states=states,
      state_shardings=state_shardings
    )
  
  
  
   

  ##change the paramters of these, currently pass in dummy inputs
  def __call__(
    self,
    example_inputs,
    num_inference_steps: int = 50,
    # guidance_scale: Union[float, List[float]] = 4.5,
  ):
    
    # if not isinstance(guidance_scale, List):
    #   guidance_scale = [guidance_scale] * len(self.scheduler_state.timesteps)
    # guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]
    # num_conds = 1
    # if do_classifier_free_guidance:
    #   num_conds += 1
    # data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
    # latents = jax.device_put(example_inputs["latents"], data_sharding)
    # prompt_embeds = jax.device_put(example_inputs["prompt_embeds"], data_sharding)
    # fractional_coords = jax.device_put(example_inputs["fractional_coords"], data_sharding)
    # noise_cond = jax.device_put(example_inputs["timestep"], data_sharding)
    # segment_ids = jax.device_put(example_inputs["segment_ids"], data_sharding)
    # encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"], data_sharding)
    # validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids)
    noise_cond = jnp.ones(  #initialize first round with this!
      (1, 1)
    )
    
    # noise_cond = None
    saved_tensor_path = "/home/serenagu_google_com/LTX-Video/ltx_video/pipelines/schedulerTest1.0"
    tensor_dict = torch.load(saved_tensor_path)

    for key, value in tensor_dict.items():
      if value is not None:
        tensor_dict[key] = jnp.array(value.to(torch.float32).cpu().numpy())
    example_inputs = tensor_dict
    latents = jax.device_put(example_inputs["latent_model_input"])
    prompt_embeds = jax.device_put(example_inputs["encoder_hidden_states"])
    fractional_coords = jax.device_put(example_inputs["indices_grid"])
    encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"])
    segment_ids = None
    # validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids)
    
    #only run this for the first time!
    scheduler_state = self.scheduler.set_timesteps(state=self.scheduler_state, shape=latents.shape, num_inference_steps=num_inference_steps)
    extra_step_kwargs = prepare_extra_step_kwargs(generator = jax.random.PRNGKey(0)) #check if this value needs to be changed, for unipc eta is not taken
    # scheduler_state = self.scheduler_state
    # num_warmup_steps = max(len(self.scheduler_state.timesteps) - num_inference_steps * self.scheduler.order, 0) #no paramter order here
    # p_run_inference = jax.jit(
    #   functools.partial(
    #       run_inference,
    #       transformer=self.transformer,
    #       config=self.config,
    #       mesh=self.mesh,
    #       fractional_cords=fractional_coords,
    #       prompt_embeds = prompt_embeds,
    #       segment_ids=segment_ids,
    
    
    #       encoder_attention_segment_ids=encoder_attention_segment_ids,
    #       num_inference_steps=num_inference_steps,
    #       scheduler=self.scheduler,
    #   ),
    #   in_shardings=(self.state_shardings, data_sharding, data_sharding, None),   #not sure if this sharding is correct
    #   out_shardings=None,
    # )
    
     # num_warmup_steps = max(len(self.scheduler_state.timesteps) - num_inference_steps * self.scheduler.order, 0) #no paramter order here
    p_run_inference = functools.partial(
          run_inference,
          transformer=self.transformer,
          config=self.config,
          mesh=self.mesh,
          fractional_cords=fractional_coords,
          prompt_embeds = prompt_embeds,
          segment_ids=segment_ids,
          encoder_attention_segment_ids=encoder_attention_segment_ids,
          num_inference_steps=num_inference_steps,
          scheduler=self.scheduler,
          # guidance_scale=guidance_scale
      )

    with self.mesh:
      latents, scheduler_state = p_run_inference(states=self.states, latents=
                                latents, timestep=noise_cond, scheduler_state=scheduler_state) #add scheduler state back in
    dict_to_save = {}
    dict_to_save["latents"] = torch.from_numpy(np.array(latents))
 
    save_tensor_dict(dict_to_save, 2)
    scheduler_checkpointer = ocp.Checkpointer(PickleCheckpointHandler())
    base_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(base_dir, "scheduler_ckpt_next")
    scheduler_checkpointer.save(ckpt_path, scheduler_state)
    return latents, scheduler_state
  #save states here


def transformer_forward_pass(   #need to jit this? wan didnt
                             
  latents,
  state,
  noise_cond,
  transformer,
  fractional_cords,
  prompt_embeds,
  segment_ids,
  encoder_attention_segment_ids
  ):
  

  noise_pred = transformer.apply(
      {"params": state.params},
      hidden_states=latents,
      indices_grid=fractional_cords,
      encoder_hidden_states=prompt_embeds,
      timestep=noise_cond,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids
  )  #need .param here?
  return noise_pred, state


def run_inference(
  states, transformer, config, mesh, latents, fractional_cords, prompt_embeds, timestep, num_inference_steps, scheduler, segment_ids, encoder_attention_segment_ids, scheduler_state
  ):
  # do_classifier_free_guidance = guidance_scale > 1.0
  transformer_state = states["transformer"]
  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    timestep = jnp.broadcast_to(t, timestep.shape) #(4, 256)
    # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules): #error out with this line
    
    noise_pred, transformer_state = transformer_forward_pass(latents, transformer_state, timestep/1000, transformer, fractional_cords, prompt_embeds, segment_ids, encoder_attention_segment_ids)
    #ValueError: One of pjit outputs with pytree key path result was given the sharding of NamedSharding(mesh=Mesh('data': 4, 'fsdp': 1, 'tensor': 1, 'fsdp_transpose': 1, 'expert': 1, 'tensor_transpose': 1, 'tensor_sequence': 1, 'sequence': 1, axis_types=(Auto, Auto, Auto, Auto, Auto, Auto, Auto, Auto)), spec=PartitionSpec(('data', 'fsdp'), None, None), memory_kind=device), which implies that the global size of its dimension 0 should be divisible by 4, but it is equal to 1 (full shape: (1, 1, 128))
    
    # # latents = self.denoising
    #   latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
       # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    #noise_pred, transformer_state = transformer_forward_pass(latents, transformer_state, timestep, transformer, fractional_cords, prompt_embeds, segment_ids, encoder_attention_segment_ids) #need to check if transformer_state is successfully updated
    # if do_classifier_free_guidance:
    #   noise
    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    
  return latents, scheduler_state
    
  
  
  
  
  
  
  
  
  
    
    #   ##handle whether there's guidance here?
    # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    #     latents, transformer_state, timestep = jax.lax.fori_loop(0, 1, transformer_forward_pass_p, (latents, transformer_state, timestep)) #TODO: change 1 to num_inference_step
    # return latents
