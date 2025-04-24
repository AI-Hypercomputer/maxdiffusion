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

from abc import abstractmethod
import time
from typing import Any, Callable
import jax
from maxdiffusion import (max_utils, maxdiffusion_utils, max_logging)

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (BaseStableDiffusionCheckpointer)

# Define a filename for logging


def _log_to_file(message: str, log_file: str = ""):
  """Appends a message to the global log file with a timestamp."""
  timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
  full_message = f"[{timestamp}] {message}\n"
  if log_file:
    with open(log_file, "a") as f:
      f.write(full_message)
  max_logging.log(full_message.strip())


class BaseStableDiffusionTrainer(BaseStableDiffusionCheckpointer):

  def __init__(self, config, checkpoint_type):
    BaseStableDiffusionCheckpointer.__init__(self, config, checkpoint_type)

    # sharding
    self.data_sharding = None

    self.per_device_tflops = None

    self.writer = max_utils.initialize_summary_writer(config)

    self.p_train_step = None

  @abstractmethod
  def get_shaped_batch(self, config, pipeline):
    pass

  @abstractmethod
  def compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings):
    pass

  @abstractmethod
  def pre_training_steps(self):
    pass

  @abstractmethod
  def post_training_steps(self, pipeline, params, train_states):
    pass

  @abstractmethod
  def load_dataset(self, pipeline, params, train_states):
    pass

  @abstractmethod
  def training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler):
    pass

  @abstractmethod
  def get_data_shardings(self):
    pass

  @abstractmethod
  def create_scheduler(self, pipeline, params):
    pass

  def _time_and_log_call(
      self, func_obj: Callable[..., Any], *func_args: Any, description: str = "", **func_kwargs: Any
  ) -> Any:
    """
    Times a function call, logs its duration, and returns its result.
    """
    if not description:
      if hasattr(func_obj, "__name__"):
        description = func_obj.__name__
      elif hasattr(func_obj, "__call__") and hasattr(type(func_obj), "__name__"):
        description = type(func_obj).__name__
    log_file = ""

    if self.config.write_timing_metrics and self.config.timing_metrics_file:
      log_file = self.config.get.timing_metrics_file
    _log_to_file(f"Starting: {description}...", log_file=log_file)
    start_time = time.perf_counter()  # Use perf_counter for more precise duration measurement
    result = func_obj(*func_args, **func_kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    _log_to_file(f"Finished: {description} - Duration: {duration:.4f} seconds", log_file=log_file)
    return result

  def calculate_tflops(self, pipeline, params):
    per_device_tflops = maxdiffusion_utils.calculate_unet_tflops(
        self.config, pipeline, (self.config.per_device_batch_size * jax.local_device_count()), self.rng, train=True
    )
    max_logging.log(f"UNET per device TFLOPS: {per_device_tflops}")
    return per_device_tflops

  def start_training(self):
    # Hook
    self.pre_training_steps()
    # Load checkpoint - will load or create states
    pipeline, params = self._time_and_log_call(self.load_checkpoint)
    # create train states
    train_states = {}
    state_shardings = {}
    vae_state, vae_state_mesh_shardings = self._time_and_log_call(
        self.create_vae_state,
        # Arguments for create_vae_state
        pipeline=pipeline,
        params=params,
        checkpoint_item_name="vae_state",
        is_training=False,
    )

    train_states["vae_state"] = vae_state
    state_shardings["vae_state_shardings"] = vae_state_mesh_shardings

    text_encoder_state, text_encoder_state_mesh_shardings = self._time_and_log_call(
        self.create_text_encoder_state,
        # Arguments for create_text_encoder_state
        pipeline=pipeline,
        params=params,
        checkpoint_item_name="text_encoder_state",
        is_training=self.config.train_text_encoder,
    )
    train_states["text_encoder_state"] = text_encoder_state
    state_shardings["text_encoder_state_shardings"] = text_encoder_state_mesh_shardings
    if hasattr(pipeline, "text_encoder_2"):
      text_encoder_2_state, text_encoder_2_state_mesh_shardings = self._time_and_log_call(
          self.create_text_encoder_2_state,
          # Arguments for create_text_encoder_2_state
          pipeline=pipeline,
          params=params,
          checkpoint_item_name="text_encoder_2_state",
          is_training=self.config.train_text_encoder,
      )
      train_states["text_encoder_2_state"] = text_encoder_2_state
      state_shardings["text_encoder_2_state_shardings"] = text_encoder_2_state_mesh_shardings

    # Create scheduler
    noise_scheduler, noise_scheduler_state = self.create_scheduler(pipeline, params)
    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    # Calculate tflops
    per_device_tflops = self.calculate_tflops(pipeline, params)
    self.per_device_tflops = per_device_tflops

    # Load dataset
    data_iterator = self._time_and_log_call(self.load_dataset, pipeline, params, train_states)
    if self.config.dataset_type == "grain":
      data_iterator = self._time_and_log_call(self.restore_data_iterator_state, data_iterator=data_iterator)

    unet_state, unet_state_mesh_shardings, unet_learning_rate_scheduler = self._time_and_log_call(
        self.create_unet_state,
        # ambiguous here, but if self.params.get("unet") doesn't exist
        # Then its 1 of 2 scenarios:
        # 1. unet state will be loaded directly from orbax
        # 2. a new unet is being trained from scratch.
        pipeline=pipeline,
        params=params,
        checkpoint_item_name="unet_state",
        is_training=True,
    )
    train_states["unet_state"] = unet_state
    state_shardings["unet_state_shardings"] = unet_state_mesh_shardings

    data_shardings = self.get_data_shardings()
    # Compile train_step
    p_train_step = self._time_and_log_call(
        self.compile_train_step, pipeline, params, train_states, state_shardings, data_shardings
    )
    # Start training
    train_states = self._time_and_log_call(
        self.training_loop, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler
    )
    # 6. save final checkpoint
    # Hook
    self._time_and_log_call(self.post_training_steps, pipeline, params, train_states)
