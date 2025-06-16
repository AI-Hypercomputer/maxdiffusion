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

import os
import datetime
import functools
import numpy as np
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from ..schedulers import FlaxEulerDiscreteScheduler
from .. import max_utils, max_logging, train_utils, maxdiffusion_utils
from ..checkpointing.wan_checkpointer import (WanCheckpointer, WAN_CHECKPOINT)
from multihost_dataloading import _form_global_array


class WanTrainer(WanCheckpointer):

  def __init__(self, config):
    WanCheckpointer.__init__(self, config, WAN_CHECKPOINT)
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

    self.global_batch_size = self.config.per_device_batch_size * jax.device_count()

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
    max_logging.log("WARNING : Calculting tflops is not implemented in Wan 2.1. Returning 0...")
    return 0

  def load_dataset(self, pipeline):
    # Stages of training as described in the Wan 2.1 paper - https://arxiv.org/pdf/2503.20314
    # Image pre-training - txt2img 256px
    # Image-video joint training - stage 1. 256 px images and 192px 5 sec videos at fps=16
    # Image-video joint training - stage 2. 480px images and 480px 5 sec videos at fps=16
    # Image-video joint training - stage final. 720px images and 720px 5 sec videos at fps=16
    # prompt embeds shape: (1, 512, 4096)
    # For now, we will pass the same latents over and over
    # TODO - create a dataset
    return maxdiffusion_utils.get_dummy_wan_inputs(self.config, pipeline, self.global_batch_size)

  def start_training(self):

    pipeline = self.load_checkpoint()
    del pipeline.vae
    dummy_inputs = self.load_dataset(pipeline)
    mesh = pipeline.mesh
    optimizer, learning_rate_scheduler = self._create_optimizer(pipeline.transformer, self.config, 1e-5)
    dummy_inputs = tuple(
        [jtu.tree_map_with_path(functools.partial(_form_global_array, global_mesh=mesh), input) for input in dummy_inputs]
    )
    self.training_loop(pipeline, optimizer, learning_rate_scheduler, dummy_inputs)

  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, data):

    graphdef, state = nnx.split((pipeline.transformer, optimizer))
    writer = max_utils.initialize_summary_writer(self.config)
    num_model_parameters = max_utils.calculate_num_params_from_pytree(state[0])
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Total train batch size (w. parallel & distributed) = {self.global_batch_size}")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    state = state.to_pure_dict()
    p_train_step = jax.jit(
        train_step,
        donate_argnums=(0,),
    )
    rng = jax.random.key(self.config.seed)
    start_step = 0
    last_step_completion = datetime.datetime.now()
    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None
    first_profiling_step = self.config.skip_first_n_steps_for_profiler
    if self.config.enable_profiler and first_profiling_step >= self.config.max_train_steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(
        first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.max_train_steps - 1
    )
    # TODO - 0 needs to be changed to last step if continuing from an orbax checkpoint.
    start_step = 0
    per_device_tflops = self.calculate_tflops(pipeline)

    for step in np.arange(start_step, self.config.max_train_steps):
      if self.config.enable_profiler and step == first_profiling_step:
        max_utils.activate_profiler(self.config)
      with jax.profiler.StepTraceAnnotation("train", step_num=step), pipeline.mesh, nn_partitioning.axis_rules(
          self.config.logical_axis_rules
      ):
        state, train_metric, rng = p_train_step(state, graphdef, data, rng)

      new_time = datetime.datetime.now()

      if self.config.enable_profiler and step == last_profiling_step:
        max_utils.deactivate_profiler(self.config)

      train_utils.record_scalar_metrics(
          train_metric, new_time - last_step_completion, per_device_tflops, learning_rate_scheduler(step)
      )
      if self.config.write_metrics:
        train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
      last_step_completion = new_time


def train_step(state, graphdef, data, rng):
  return step_optimizer(graphdef, state, data, rng)


def step_optimizer(graphdef, state, data, rng):
  _, new_rng = jax.random.split(rng)

  def loss_fn(model):
    latents, prompt_embeds, timesteps = data

    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)

    # TODO - add noise here

    model_pred = model(
        hidden_states=noise,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        is_uncond=jnp.array(False, dtype=jnp.bool_),
        slg_mask=jnp.zeros(1, dtype=jnp.bool_),
    )
    target = noise - latents
    loss = (target - model_pred) ** 2
    loss = jnp.mean(loss)
    # breakpoint()
    return loss

  model, optimizer = nnx.merge(graphdef, state)
  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  state = nnx.state((model, optimizer))
  state = state.to_pure_dict()
  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}
  return state, metrics, new_rng
