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
import threading
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from flax import nnx  
from ..schedulers import FlaxFlowMatchScheduler
from flax.linen import partitioning as nn_partitioning
from ..schedulers import FlaxEulerDiscreteScheduler
from .. import max_utils, max_logging, train_utils, maxdiffusion_utils
from ..checkpointing.wan_checkpointer import (WanCheckpointer, WAN_CHECKPOINT)
from maxdiffusion.multihost_dataloading import _form_global_array
from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.train_utils import (
  _tensorboard_writer_worker,
  load_next_batch,
  _metrics_queue
)

def generate_sample(config, pipeline, filename_prefix):
  """
  Generates a video to validate training did not corrupt the model
  """
  generate_wan(config, pipeline, filename_prefix)


class WanTrainer(WanCheckpointer):

  def __init__(self, config):
    WanCheckpointer.__init__(self, config, WAN_CHECKPOINT)
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

    self.global_batch_size = self.config.per_device_batch_size * jax.device_count()

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def create_scheduler(self):
    """Creates and initializes the Flow Match scheduler for training."""
    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)
    return noise_scheduler, noise_scheduler_state

  def calculate_tflops(self, pipeline):
    max_logging.log("WARNING : Calculting tflops is not implemented in Wan 2.1. Returning 0...")
    return 0

  def load_dataset(self, mesh):
    # Stages of training as described in the Wan 2.1 paper - https://arxiv.org/pdf/2503.20314
    # Image pre-training - txt2img 256px
    # Image-video joint training - stage 1. 256 px images and 192px 5 sec videos at fps=16
    # Image-video joint training - stage 2. 480px images and 480px 5 sec videos at fps=16
    # Image-video joint training - stage final. 720px images and 720px 5 sec videos at fps=16
    # prompt embeds shape: (1, 512, 4096)
    # For now, we will pass the same latents over and over
    # TODO - create a dataset
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError("Wan 2.1 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True")

    feature_description = {
      "latents" : tf.io.FixedLenFeature([], tf.string),
      "encoder_hidden_states" : tf.io.FixedLenFeature([], tf.string),
    }

    def prepare_sample(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      return {"latents" : latents, "encoder_hidden_states" : encoder_hidden_states}
    
    data_iterator = make_data_iterator(
      config,
      jax.process_index(),
      jax.process_count(),
      mesh,
      self.global_batch_size,
      feature_description=feature_description,
      prepare_sample_fn=prepare_sample
    )
    return data_iterator

  def start_training(self):

    pipeline = self.load_checkpoint()
    #del pipeline.vae

    # Generate a sample before training to compare against generated sample after training.
    generate_sample(self.config, pipeline, filename_prefix='pre-training-')
    mesh = pipeline.mesh
    data_iterator = self.load_dataset(mesh)
    
    # Load FlowMatch scheduler
    scheduler, scheduler_state = self.create_scheduler()
    pipeline.scheduler = scheduler
    pipeline.scheduler_state = scheduler_state

    optimizer, learning_rate_scheduler = self._create_optimizer(pipeline.transformer, self.config, 1e-5)
    self.training_loop(pipeline, optimizer, learning_rate_scheduler, data_iterator)

  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, data_iterator):

    graphdef, state = nnx.split((pipeline.transformer, optimizer))
    
    writer = max_utils.initialize_summary_writer(self.config)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
    writer_thread.start()
    
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
        functools.partial(train_step, scheduler=pipeline.scheduler),
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

    scheduler_state = pipeline.scheduler_state
    example_batch = load_next_batch(data_iterator, None, self.config)
    with ThreadPoolExecutor(max_workers=1) as executor:
      for step in np.arange(start_step, self.config.max_train_steps):
        if self.config.enable_profiler and step == first_profiling_step:
          max_utils.activate_profiler(self.config)
        start_step_time = datetime.datetime.now()
        next_batch_future = executor.submit(load_next_batch, data_iterator, example_batch, self.config)
        with jax.profiler.StepTraceAnnotation("train", step_num=step), pipeline.mesh, nn_partitioning.axis_rules(
            self.config.logical_axis_rules
        ):
          state, scheduler_state, train_metric, rng = p_train_step(state, graphdef, scheduler_state, example_batch, rng)
          train_metric["scalar"]["learning/loss"].block_until_ready()
        last_step_completion = datetime.datetime.now()

        if self.config.enable_profiler and step == last_profiling_step:
          max_utils.deactivate_profiler(self.config)

        train_utils.record_scalar_metrics(
            train_metric, last_step_completion - start_step_time, per_device_tflops, learning_rate_scheduler(step)
        )
        if self.config.write_metrics:
          train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
        example_batch = next_batch_future.result()

      _metrics_queue.put(None)
      writer_thread.join()
      if writer:
        writer.flush()
      
      # load new state for trained tranformer
      graphdef, _, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)
      pipeline.transformer = nnx.merge(graphdef, state[0], rest_of_state)

      generate_sample(self.config, pipeline, filename_prefix='post-training-')


def train_step(state, graphdef, scheduler_state, data, rng, scheduler):
  return step_optimizer(graphdef, state, scheduler, scheduler_state, data, rng)


def step_optimizer(graphdef, state, scheduler, scheduler_state, data, rng):
  _, new_rng, timestep_rng = jax.random.split(rng, num=3)

  def loss_fn(model):
    latents = data['latents']
    encoder_hidden_states = data['encoder_hidden_states']
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
      timestep_rng,
      (bsz,),
      0,
      scheduler.config.num_train_timesteps,
    )
    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents = scheduler.add_noise(scheduler_state, latents, noise, timesteps)

    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        is_uncond=jnp.array(False, dtype=jnp.bool_),
        slg_mask=jnp.zeros(1, dtype=jnp.bool_),
    )
    
    training_target = scheduler.training_target(latents, noise, timesteps)
    training_weight = jnp.expand_dims(scheduler.training_weight(scheduler_state, timesteps), axis=(1, 2, 3, 4))
    loss = ((training_target - model_pred) ** 2)
    loss = loss * training_weight
    loss = jnp.mean(loss)

    return loss

  model, optimizer = nnx.merge(graphdef, state)
  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  state = nnx.state((model, optimizer))
  state = state.to_pure_dict()
  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}
  return state, scheduler_state, metrics, new_rng
