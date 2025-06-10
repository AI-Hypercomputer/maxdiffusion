"""
 Copyright 2025 Google LLC

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

import functools
import numpy as np
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from flax import nnx
from ..schedulers import FlaxEulerDiscreteScheduler
from .. import max_utils
from .. import max_logging
from ..checkpointing.wan_checkpointer import (
  WanCheckpointer,
  WAN_CHECKPOINT
)
from multihost_dataloading import _form_global_array

class WanTrainer(WanCheckpointer):
  def __init__(self, config):
    WanCheckpointer.__init__(self, config, WAN_CHECKPOINT)
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def create_scheduler(self, pipeline, params):
    # TODO - set right scheduler
    noise_scheduler, noise_scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=self.config.pretrained_model_name_or_path, subfolder="scheduler", dtype=jnp.float32
    )
    noise_scheduler_state = noise_scheduler.set_timesteps(
        state=noise_scheduler_state, num_inference_steps=self.config.num_inference_steps, timestep_spacing="flux"
    )
    return noise_scheduler, noise_scheduler_state

  def calculate_tflops(self, pipeline):
    pass

  def load_dataset(self, pipeline):
    # Stages of training as described in the Wan 2.1 paper - https://arxiv.org/pdf/2503.20314
    # Image pre-training - txt2img 256px
    # Image-video joint training - stage 1. 256 px images and 192px 5 sec videos at fps=16
    # Image-video joint training - stage 2. 480px images and 480px 5 sec videos at fps=16
    # Image-video joint training - stage final. 720px images and 720px 5 sec videos at fps=16
    # prompt embeds shape: (1, 512, 4096)
    # For now, we will pass the same latents over and over
    # TODO - create a dataset
    global_batch_size = self.config.per_device_batch_size * jax.device_count()
    prompt_embeds = jax.random.normal(jax.random.key(self.config.seed), (global_batch_size, 512, 4096))
    latents = pipeline.prepare_latents(
      global_batch_size,
      vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
      vae_scale_factor_spatial=pipeline.vae_scale_factor_spatial,
      height=self.config.height,
      width=self.config.width,
      num_frames=self.config.num_frames,
      num_channels_latents=pipeline.transformer.config.in_channels
    )
    return (latents, prompt_embeds)

  def start_training(self):

    pipeline = self.load_checkpoint()
    mesh = pipeline.mesh

    optimizer, learning_rate_scheduler = self._create_optimizer(pipeline.transformer, self.config, self.config.learning_rate)

    # @nnx.jit
    # def create_transformer_state(transformer):
    #   optimizer = self._create_optimizer(transformer, self.config, self.config.learning_rate)
    #   breakpoint()
    #   _, state = nnx.split((transformer, optimizer))
    
    # with mesh:
    #   create_transformer_state(pipeline.transformer)

    #graphdef, state = nnx.plit((pipeline.transformer, optimizer))
    dummy_inputs = self.load_dataset(pipeline)
    dummy_inputs = tuple([jtu.tree_map_with_path(functools.partial(_form_global_array, global_mesh=mesh), input) for input in dummy_inputs])

    self.training_loop(pipeline, optimizer, learning_rate_scheduler, dummy_inputs)
  
  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, data):
    
    graphdef, state = nnx.split((pipeline.transformer, optimizer))
    state = state.to_pure_dict()
    p_train_step = jax.jit(
      train_step,
      donate_argnums=(1,),
    )
    rng = jax.random.key(self.config.seed)
    start_step = 0
    for step in np.arange(start_step, self.config.max_train_steps):
      with pipeline.mesh:
        loss, state, rng = p_train_step(graphdef, state, data, rng)
        max_logging.log(f"loss: {loss}")

def train_step(graphdef, state, data, rng):
  return step_optimizer(graphdef, state, data, rng)

def step_optimizer(graphdef, state, data, rng):
  _, new_rng = jax.random.split(rng)
  def loss_fn(model):
    latents, prompt_embeds = data
    bsz = latents.shape[0]
    timesteps = jnp.array([0] * bsz, dtype=jnp.int32)

    noise = jax.random.normal(
      key=new_rng,
      shape=latents.shape,
      dtype=latents.dtype
    )

    # TODO - add noise here

    model_pred = model(
      hidden_states=noise,
      timestep=timesteps,
      encoder_hidden_states=prompt_embeds,
      is_uncond=jnp.array(False, dtype=jnp.bool_),
      slg_mask=jnp.zeros(1, dtype=jnp.bool_)
    )
    target = noise - latents
    loss = (target - model_pred) ** 2
    loss = jnp.mean(loss)
    #breakpoint()
    return loss
  model, optimizer = nnx.merge(graphdef, state)
  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  state = nnx.state((model, optimizer))
  state = state.to_pure_dict()
  return loss, state, new_rng