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

# DISCLAIMER: This is a JAX/Flax conversion of a PyTorch implementation.
# The original PyTorch code was provided by the user.

from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

# Assuming these are part of your project structure, similar to the UniPC example
from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
)


@flax.struct.dataclass
class FlowMatchSchedulerState:
  """
  Data class to hold the mutable state of the FlaxFlowMatchScheduler.
  """

  sigmas: jnp.ndarray
  timesteps: jnp.ndarray
  linear_timesteps_weights: Optional[jnp.ndarray]
  training: bool
  num_inference_steps: int  # Store for training weight calculation

  @classmethod
  def create(cls):
    return cls(
        sigmas=None,
        timesteps=None,
        linear_timesteps_weights=None,
        training=False,
        num_inference_steps=0,
    )


@flax.struct.dataclass(frozen=False)
class FlaxFlowMatchSchedulerOutput(FlaxSchedulerOutput):
  """
  Output class for the JAX FlowMatchScheduler's step function.

  Attributes:
      prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
          The computed sample at the previous timestep.
      state (`FlowMatchSchedulerState`):
          The updated scheduler state.
  """

  state: FlowMatchSchedulerState


class FlaxFlowMatchScheduler(FlaxSchedulerMixin, ConfigMixin):
  """
  FlaxFlowMatchScheduler is a JAX/Flax conversion of a scheduler used for training video generation models like
  WAN 2.1. It operates based on a "flow matching" paradigm.

  This scheduler directly calculates sigmas for a continuous-time diffusion process, which can be beneficial for
  certain types of models and training schemes.
  """

  dtype: jnp.dtype

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      shift: float = 3.0,
      sigma_max: float = 1.0,
      sigma_min: float = 0.003 / 1.002,
      inverse_timesteps: bool = False,
      extra_one_step: bool = False,
      reverse_sigmas: bool = False,
      dtype: jnp.dtype = jnp.float32,
      base_image_seq_len: int = 1024,
      base_shift: float = 0.95,
      invert_sigmas: bool = False,
      max_image_seq_len: int = 4096,
      max_shift: float = 2.05,
      shift_terminal: float = 0.1,
      stochastic_sampling: bool = False,
      time_shift_type: str = "exponential",
      use_beta_sigmas: bool = False,
      use_dynamic_shifting: bool = True,
      use_exponential_sigmas: bool = False,
      use_karras_sigmas: bool = False,
      **kwargs,
  ):
    self.dtype = dtype

  def create_state(self) -> FlowMatchSchedulerState:
    """Creates the initial state for the scheduler."""

    return FlowMatchSchedulerState.create()

  def set_timesteps(
      self,
      state: FlowMatchSchedulerState,
      num_inference_steps: int = 100,
      shape: Tuple = None,  # Not used but part of the standard API
      denoising_strength: float = 1.0,
      training: bool = False,
      shift: Optional[float] = None,
      timesteps: Optional[jnp.ndarray] = None,
      sigmas: Optional[jnp.ndarray] = None,
  ) -> FlowMatchSchedulerState:
    """
    Sets the discrete timesteps used for the diffusion chain.

    Args:
        state (`FlowMatchSchedulerState`):
            The current scheduler state.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
        shape (`Tuple`):
            The shape of the samples.
        denoising_strength (`float`):
            The strength of the denoising process.
        training (`bool`):
            Whether the scheduler is being used for training.
        shift (`Optional[float]`):
            An optional shift value to override the one in the config.
        timesteps (`Optional[jnp.ndarray]`):
            Custom timesteps to use for the denoising process.
        sigmas (`Optional[jnp.ndarray]`):
            Custom sigmas to use for the denoising process.

    Returns:
        `FlowMatchSchedulerState`: The updated scheduler state.
    """
    current_shift = shift if shift is not None else self.config.shift

    if timesteps is not None and sigmas is not None:
      pass
    elif timesteps is not None:
      sigmas = timesteps / self.config.num_train_timesteps
    elif sigmas is not None:
      timesteps = sigmas * self.config.num_train_timesteps
    else:
      sigma_start = self.config.sigma_min + (self.config.sigma_max - self.config.sigma_min) * denoising_strength

      if self.config.extra_one_step:
        sigmas = jnp.linspace(sigma_start, self.config.sigma_min, num_inference_steps + 1, dtype=self.dtype)[:-1]
      else:
        sigmas = jnp.linspace(sigma_start, self.config.sigma_min, num_inference_steps, dtype=self.dtype)

      if self.config.inverse_timesteps:
        sigmas = jnp.flip(sigmas, dims=[0])

      sigmas = current_shift * sigmas / (1 + (current_shift - 1) * sigmas)

      if self.config.reverse_sigmas:
        sigmas = 1 - sigmas

      timesteps = sigmas * self.config.num_train_timesteps

    if timesteps is not None:
      num_inference_steps = len(timesteps)

    linear_timesteps_weights = None
    if training:
      x = timesteps
      y = jnp.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
      y_shifted = y - jnp.min(y)
      bsmntw_weighing = y_shifted * (num_inference_steps / jnp.sum(y_shifted))
      linear_timesteps_weights = bsmntw_weighing

    return state.replace(
        sigmas=sigmas,
        timesteps=timesteps,
        linear_timesteps_weights=linear_timesteps_weights,
        training=training,
        num_inference_steps=num_inference_steps,
    )

  def _find_timestep_id(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray) -> jnp.ndarray:
    """Finds the index of the closest timestep in the scheduler's `timesteps` array."""
    timestep = jnp.asarray(timestep, dtype=state.timesteps.dtype)
    if timestep.ndim == 0:
      return jnp.argmin(jnp.abs(state.timesteps - timestep))
    else:
      diffs = jnp.abs(state.timesteps[None, :] - timestep[:, None])
      return jnp.argmin(diffs, axis=1)

  def step(
      self,
      state: FlowMatchSchedulerState,
      model_output: jnp.ndarray,
      timestep: jnp.ndarray,
      sample: jnp.ndarray,
      to_final: bool = False,
      return_dict: bool = True,
  ) -> Union[FlaxFlowMatchSchedulerOutput, Tuple]:
    """
    Propagates the sample with the flow matching scheduler.

    Args:
        state (`FlowMatchSchedulerState`):
            The current scheduler state.
        model_output (`jnp.ndarray`):
            The direct output from the learned diffusion model.
        timestep (`jnp.ndarray`):
            The current timestep in the diffusion chain.
        sample (`jnp.ndarray`):
            The current sample (e.g. noisy latents).
        to_final (`bool`):
            Whether this is the final step.
        return_dict (`bool`):
            Whether to return a `FlaxFlowMatchSchedulerOutput` object.

    Returns:
        `FlaxFlowMatchSchedulerOutput` or `tuple`: A tuple (`prev_sample`, `state`) or a
        `FlaxFlowMatchSchedulerOutput` object containing the previous sample and the updated state.
    """
    timestep_id = self._find_timestep_id(state, timestep)
    sigma = state.sigmas[timestep_id]

    def get_next_sigma():
      return state.sigmas[timestep_id + 1]

    def get_final_sigma():
      return jnp.array(1.0 if (self.config.inverse_timesteps or self.config.reverse_sigmas) else 0.0, dtype=sigma.dtype)

    is_final_step = to_final or jnp.all(timestep_id + 1 >= state.timesteps.shape[0])
    sigma_next = jax.lax.cond(is_final_step, get_final_sigma, get_next_sigma)

    if jnp.ndim(timestep) != 0:
      broadcast_shape = (-1,) + (1,) * (sample.ndim - 1)
      sigma = sigma.reshape(broadcast_shape)
      sigma_next = sigma_next.reshape(broadcast_shape)

    prev_sample = sample + model_output * (sigma_next - sigma)

    if not return_dict:
      return (prev_sample, state)

    return FlaxFlowMatchSchedulerOutput(prev_sample=prev_sample, state=state)

  def return_to_timestep(
      self, state: FlowMatchSchedulerState, timestep: jnp.ndarray, sample: jnp.ndarray, sample_stablized: jnp.ndarray
  ) -> jnp.ndarray:
    """Calculates the model output required to go from a stabilized sample back to the original sample."""
    timestep_id = self._find_timestep_id(state, timestep)
    sigma = state.sigmas[timestep_id]

    if jnp.ndim(timestep) != 0:
      sigma = sigma.reshape((-1,) + (1,) * (sample.ndim - 1))

    model_output = (sample - sample_stablized) / sigma
    return model_output

  def add_noise(
      self,
      state: FlowMatchSchedulerState,
      original_samples: jnp.ndarray,
      noise: jnp.ndarray,
      timesteps: jnp.ndarray,
  ) -> jnp.ndarray:
    """
    Adds noise to the original samples according to the flow matching schedule.

    Args:
        state (`FlowMatchSchedulerState`):
            The current scheduler state.
        original_samples (`jnp.ndarray`):
            The original clean samples.
        noise (`jnp.ndarray`):
            The noise to add to the samples.
        timesteps (`jnp.ndarray`):
            The timesteps that correspond to the noise levels.

    Returns:
        `jnp.ndarray`: The noisy samples.
    """
    if state.sigmas is None or state.timesteps is None:
      raise ValueError("Scheduler's `sigmas` and `timesteps` are not set. Please call `set_timesteps` before `add_noise`.")

    timestep_ids = self._find_timestep_id(state, timesteps)
    sigmas = state.sigmas[timestep_ids]

    broadcast_shape = (-1,) + (1,) * (original_samples.ndim - 1)
    sigmas = sigmas.reshape(broadcast_shape)

    noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
    return noisy_samples

  def training_target(self, sample: jnp.ndarray, noise: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    """
    Calculates the training target. For flow matching, this is typically the velocity, `x_1 - x_0`,
    which is equivalent to `noise - sample` under this scheduler's `add_noise` definition.
    """
    target = noise - sample
    return target

  def training_weight(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray) -> jnp.ndarray:
    """Calculates the training weight for a given timestep."""
    timestep_ids = self._find_timestep_id(state, timestep)
    weights = state.linear_timesteps_weights[timestep_ids]
    return weights

  def __len__(self) -> int:
    return self.config.num_train_timesteps
