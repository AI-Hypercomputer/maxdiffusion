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
from jax.sharding import PartitionSpec as P
from flax import nnx
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from flax.linen import partitioning as nn_partitioning
from maxdiffusion import max_utils, max_logging, train_utils
from maxdiffusion.checkpointing.wan_checkpointer import (WanCheckpointer, WAN_CHECKPOINT)
from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.train_utils import (_tensorboard_writer_worker, load_next_batch, _metrics_queue)
from maxdiffusion.video_processor import VideoProcessor
from maxdiffusion.utils import load_video
from skimage.metrics import structural_similarity as ssim
from flax.training import train_state

class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef
  rest_of_state: nnx.State

def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x

def generate_sample(config, pipeline, filename_prefix):
  """
  Generates a video to validate training did not corrupt the model
  """
  return generate_wan(config, pipeline, filename_prefix)


def print_ssim(pretrained_video_path, posttrained_video_path):
  video_processor = VideoProcessor()
  pretrained_video = load_video(pretrained_video_path[0])
  pretrained_video = video_processor.preprocess_video(pretrained_video)
  pretrained_video = np.array(pretrained_video)
  pretrained_video = np.transpose(pretrained_video, (0, 2, 3, 4, 1))
  pretrained_video = np.uint8((pretrained_video + 1) * 255 / 2)

  posttrained_video = load_video(posttrained_video_path[0])
  posttrained_video = video_processor.preprocess_video(posttrained_video)
  posttrained_video = np.array(posttrained_video)
  posttrained_video = np.transpose(posttrained_video, (0, 2, 3, 4, 1))
  posttrained_video = np.uint8((posttrained_video + 1) * 255 / 2)

  ssim_compare = ssim(pretrained_video[0], posttrained_video[0], multichannel=True, channel_axis=-1, data_range=255)

  max_logging.log(f"SSIM score after training is {ssim_compare}")


class WanTrainer(WanCheckpointer):

  def __init__(self, config):
    WanCheckpointer.__init__(self, config, WAN_CHECKPOINT)
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

    #self.global_batch_size = self.config.per_device_batch_size * jax.device_count()
    self.global_batch_size = config.global_batch_size if config.global_batch_size > 0 else config.per_device_batch_size * jax.device_count()

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
  
  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {
      "latents" : data_sharding,
      "encoder_hidden_states" : data_sharding
    }
    return data_sharding

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
      raise ValueError(
          "Wan 2.1 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True"
      )

    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    }

    def prepare_sample(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      return {"latents": latents, "encoder_hidden_states": encoder_hidden_states}

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        self.global_batch_size,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample,
    )
    return data_iterator

  def start_training(self):

    pipeline = self.load_checkpoint()
    # del pipeline.vae

    # Generate a sample before training to compare against generated sample after training.
    pretrained_video_path = generate_sample(self.config, pipeline, filename_prefix="pre-training-")
    mesh = pipeline.mesh
    data_iterator = self.load_dataset(mesh)

    # Load FlowMatch scheduler
    scheduler, scheduler_state = self.create_scheduler()
    pipeline.scheduler = scheduler
    pipeline.scheduler_state = scheduler_state
    optimizer, learning_rate_scheduler = self._create_optimizer(pipeline.transformer, self.config, 1e-5)
    # Returns pipeline with trained transformer state
    pipeline = self.training_loop(pipeline, optimizer, learning_rate_scheduler, data_iterator)

    posttrained_video_path = generate_sample(self.config, pipeline, filename_prefix="post-training-")
    print_ssim(pretrained_video_path, posttrained_video_path)

  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, data_iterator):
    mesh = pipeline.mesh
    graphdef, params, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      state = TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optimizer,
        graphdef=graphdef,
        rest_of_state=rest_of_state
      )
      state = jax.tree.map(_to_array, state)
      state_spec = nnx.get_partition_spec(state)
      state = jax.lax.with_sharding_constraint(state, state_spec)
      state_shardings = nnx.get_named_sharding(state, mesh)
    data_shardings = self.get_data_shardings(mesh)

    writer = max_utils.initialize_summary_writer(self.config)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
    writer_thread.start()

    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Total train batch size (w. parallel & distributed) = {self.global_batch_size}")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    p_train_step = jax.jit(
        functools.partial(train_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings = (state_shardings, data_shardings, None, None),
        out_shardings = (state_shardings, None, None, None),
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
          state, scheduler_state, train_metric, rng = p_train_step(state, example_batch, rng, scheduler_state)
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
      pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
      return pipeline


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng = jax.random.split(rng, num=3)

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
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
    )

    training_target = scheduler.training_target(latents, noise, timesteps)
    training_weight = jnp.expand_dims(scheduler.training_weight(scheduler_state, timesteps), axis=(1, 2, 3, 4))
    loss = (training_target - model_pred) ** 2
    loss = loss * training_weight
    loss = jnp.mean(loss)

    return loss
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}
  return new_state, scheduler_state, metrics, new_rng
