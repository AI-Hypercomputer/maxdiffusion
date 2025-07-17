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


from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import os
from safetensors import safe_open

import flax
import jax
import jax.numpy as jnp
import json
from maxdiffusion.configuration_utils import ConfigMixin, register_to_config
from maxdiffusion.schedulers.scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
)


def linear_quadratic_schedule_jax(
    num_steps: int, threshold_noise: float = 0.025, linear_steps: Optional[int] = None
) -> jnp.ndarray:
  if num_steps == 1:
    return jnp.array([1.0], dtype=jnp.float32)
  if linear_steps is None:
    linear_steps = num_steps // 2

  linear_sigma_schedule = jnp.arange(linear_steps) * threshold_noise / linear_steps

  threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
  quadratic_steps = num_steps - linear_steps
  quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
  linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
  const = quadratic_coef * (linear_steps**2)
  quadratic_indices = jnp.arange(linear_steps, num_steps)
  quadratic_sigma_schedule = quadratic_coef * (quadratic_indices**2) + linear_coef * quadratic_indices + const

  sigma_schedule = jnp.concatenate([linear_sigma_schedule, quadratic_sigma_schedule])
  sigma_schedule = jnp.concatenate([sigma_schedule, jnp.array([1.0])])
  sigma_schedule = 1.0 - sigma_schedule
  return sigma_schedule[:-1].astype(jnp.float32)


def time_shift_jax(mu: float, sigma: float, t: jnp.ndarray) -> jnp.ndarray:
  mu_f = jnp.array(mu, dtype=jnp.float32)
  sigma_f = jnp.array(sigma, dtype=jnp.float32)
  return jnp.exp(mu_f) / (jnp.exp(mu_f) + (1 / t - 1) ** sigma_f)


def _prod_jax(iterable):
  return jnp.prod(jnp.array(iterable, dtype=jnp.float32))


def get_normal_shift_jax(
    n_tokens: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
  m = (max_shift - min_shift) / (max_tokens - min_tokens)
  b = min_shift - m * min_tokens
  return m * n_tokens + b


def append_dims_jax(x: jnp.ndarray, target_dims: int) -> jnp.ndarray:
  """Appends singleton dimensions to the end of a tensor until it reaches `target_dims`."""
  return x[(...,) + (None,) * (target_dims - x.ndim)]


def strech_shifts_to_terminal_jax(shifts: jnp.ndarray, terminal: float = 0.1) -> jnp.ndarray:
  if shifts.size == 0:
    raise ValueError("The 'shifts' tensor must not be empty.")
  if terminal <= 0 or terminal >= 1:
    raise ValueError("The terminal value must be between 0 and 1 (exclusive).")

  one_minus_z = 1.0 - shifts
  # Using shifts[-1] for the last element
  scale_factor = one_minus_z[-1] / (1.0 - terminal)
  stretched_shifts = 1.0 - (one_minus_z / scale_factor)

  return stretched_shifts


def sd3_resolution_dependent_timestep_shift_jax(
    samples_shape: Tuple[int, ...],
    timesteps: jnp.ndarray,
    target_shift_terminal: Optional[float] = None,
) -> jnp.ndarray:
  if len(samples_shape) == 3:
    _, m, _ = samples_shape
  elif len(samples_shape) in [4, 5]:
    m = _prod_jax(samples_shape[2:])
  else:
    raise ValueError("Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)")

  shift = get_normal_shift_jax(int(m))
  time_shifts = time_shift_jax(shift, 1.0, timesteps)

  if target_shift_terminal is not None:
    time_shifts = strech_shifts_to_terminal_jax(time_shifts, target_shift_terminal)
  return time_shifts


def simple_diffusion_resolution_dependent_timestep_shift_jax(
    samples_shape: Tuple[int, ...],
    timesteps: jnp.ndarray,
    n: int = 32 * 32,
) -> jnp.ndarray:
  if len(samples_shape) == 3:
    _, m, _ = samples_shape
  elif len(samples_shape) in [4, 5]:
    m = _prod_jax(samples_shape[2:])
  else:
    raise ValueError("Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)")
  # Ensure m and n are float32 for calculations
  m_f = jnp.array(m, dtype=jnp.float32)
  n_f = jnp.array(n, dtype=jnp.float32)

  snr = (timesteps / (1 - timesteps)) ** 2  # Add epsilon for numerical stability
  shift_snr = jnp.log(snr) + 2 * jnp.log(m_f / n_f)  # Add epsilon for numerical stability
  shifted_timesteps = jax.nn.sigmoid(0.5 * shift_snr)

  return shifted_timesteps


@flax.struct.dataclass
class RectifiedFlowSchedulerState:
  """
  Data class to hold the mutable state of the RectifiedFlowScheduler.
  """

  common: CommonSchedulerState
  init_noise_sigma: float
  num_inference_steps: Optional[int] = None
  timesteps: Optional[jnp.ndarray] = None
  sigmas: Optional[jnp.ndarray] = None

  @classmethod
  def create(cls, common_state: CommonSchedulerState, init_noise_sigma: float):
    return cls(
        common=common_state,
        init_noise_sigma=init_noise_sigma,
        num_inference_steps=None,
        timesteps=None,
        sigmas=None,
    )


@dataclass
class FlaxRectifiedFlowSchedulerOutput(FlaxSchedulerOutput):
  state: RectifiedFlowSchedulerState


class FlaxRectifiedFlowMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):

  dtype: jnp.dtype
  order = 1

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(
      self,
      num_train_timesteps=1000,
      trained_betas: Optional[Union[jnp.ndarray, List[float]]] = None,
      beta_schedule: str = "linear",
      rescale_zero_terminal_snr: bool = False,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      shifting: Optional[str] = None,
      base_resolution: int = 32**2,
      target_shift_terminal: Optional[float] = None,
      sampler: Optional[str] = "Uniform",
      shift: Optional[float] = None,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.dtype = dtype

  def create_state(self, common: Optional[CommonSchedulerState] = None) -> RectifiedFlowSchedulerState:
    if common is None:
      common = CommonSchedulerState.create(self)
    init_noise_sigma = 1.0
    return RectifiedFlowSchedulerState.create(common_state=common, init_noise_sigma=init_noise_sigma)

  def get_initial_timesteps_jax(self, num_timesteps: int, shift: Optional[float] = None) -> jnp.ndarray:
    if self.config.sampler == "Uniform":
      return jnp.linspace(1.0, 1.0 / num_timesteps, num_timesteps, dtype=self.dtype)
    elif self.config.sampler == "LinearQuadratic":
      return linear_quadratic_schedule_jax(num_timesteps).astype(self.dtype)
    elif self.config.sampler == "Constant":
      assert shift is not None, "Shift must be provided for constant time shift sampler."
      return time_shift_jax(shift, 1.0, jnp.linspace(1.0, 1.0 / num_timesteps, num_timesteps, dtype=self.dtype)).astype(
          self.dtype
      )
    else:
      raise ValueError(f"Sampler {self.config.sampler} is not supported.")

  def shift_timesteps_jax(self, samples_shape: Tuple[int, ...], timesteps: jnp.ndarray) -> jnp.ndarray:
    if self.config.shifting == "SD3":
      return sd3_resolution_dependent_timestep_shift_jax(samples_shape, timesteps, self.config.target_shift_terminal)
    elif self.config.shifting == "SimpleDiffusion":
      return simple_diffusion_resolution_dependent_timestep_shift_jax(samples_shape, timesteps, self.config.base_resolution)
    return timesteps

  def from_pretrained_jax(pretrained_model_path: Union[str, os.PathLike]):
    pretrained_model_path = Path(pretrained_model_path)
    config = None
    if pretrained_model_path.is_file():
      with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
      configs = json.loads(metadata["config"])
      config = configs["scheduler"]

    elif pretrained_model_path.is_dir():
      diffusers_noise_scheduler_config_path = pretrained_model_path / "scheduler" / "scheduler_config.json"

      if not diffusers_noise_scheduler_config_path.is_file():
        raise FileNotFoundError(f"Scheduler config not found at {diffusers_noise_scheduler_config_path}")

      with open(diffusers_noise_scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)
      config = scheduler_config
    return FlaxRectifiedFlowMultistepScheduler.from_config(config)

  def set_timesteps(
      self,
      state: RectifiedFlowSchedulerState,
      num_inference_steps: Optional[int] = None,
      samples_shape: Optional[Tuple[int, ...]] = None,
      timesteps: Optional[jnp.ndarray] = None,
      device: Optional[str] = None,
  ) -> RectifiedFlowSchedulerState:
    if timesteps is not None and num_inference_steps is not None:
      raise ValueError("You cannot provide both `timesteps` and `num_inference_steps`.")

    # Determine the number of inference steps if not provided
    if num_inference_steps is None and timesteps is None:
      raise ValueError("Either `num_inference_steps` or `timesteps` must be provided.")

    if timesteps is None:
      num_inference_steps = jnp.minimum(self.config.num_train_timesteps, num_inference_steps)
      timesteps = self.get_initial_timesteps_jax(num_inference_steps, shift=self.config.shift).astype(self.dtype)

      # Apply shifting if samples_shape is provided and shifting is configured
      if samples_shape is not None:
        timesteps = self.shift_timesteps_jax(samples_shape, timesteps)
    else:
      timesteps = jnp.asarray(timesteps, dtype=self.dtype)
      num_inference_steps = len(timesteps)

    return state.replace(
        timesteps=timesteps,
        num_inference_steps=num_inference_steps,
        sigmas=timesteps,  # sigmas are the same as timesteps in RF
    )

  def scale_model_input(
      self, state: RectifiedFlowSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
  ) -> jnp.ndarray:
    # Rectified Flow scheduler typically doesn't scale model input, returns as is.
    return sample

  def step(
      self,
      state: RectifiedFlowSchedulerState,
      model_output: jnp.ndarray,
      timestep: jnp.ndarray,
      sample: jnp.ndarray,
      return_dict: bool = True,
      stochastic_sampling: bool = False,
      generator: Optional[jax.random.PRNGKey] = None,
  ) -> Union[FlaxRectifiedFlowSchedulerOutput, Tuple[jnp.ndarray, RectifiedFlowSchedulerState]]:
    if state.num_inference_steps is None:
      raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

    t_eps = 1e-6  # Small epsilon for numerical issues

    timesteps_padded = jnp.concatenate([state.timesteps, jnp.array([0.0], dtype=self.dtype)])

    if timestep.ndim == 0:
      idx = jnp.searchsorted(timesteps_padded, timestep - t_eps, side="right")  #noqa: F841
      current_t_idx = jnp.where(state.timesteps == timestep, size=1, fill_value=len(state.timesteps))[0][0]
      lower_timestep = jnp.where(current_t_idx + 1 < len(timesteps_padded), timesteps_padded[current_t_idx + 1], 0.0)
      dt = timestep - lower_timestep
    else:
      current_t_indices = jnp.searchsorted(state.timesteps, timestep, side="right")  # timesteps is decreasing
      current_t_indices = jnp.where(current_t_indices > 0, current_t_indices - 1, 0)  # adjust for right side search
      lower_timestep_indices = jnp.minimum(current_t_indices + 1, len(timesteps_padded) - 1)
      lower_timestep = timesteps_padded[lower_timestep_indices]
      dt = timestep - lower_timestep
      dt = append_dims_jax(dt, sample.ndim)

    # Compute previous sample
    if stochastic_sampling:
      if generator is None:
        raise ValueError("`generator` PRNGKey must be provided for stochastic sampling.")
      broadcastable_timestep = append_dims_jax(timestep, sample.ndim)

      x0 = sample - broadcastable_timestep * model_output
      next_timestep = timestep - dt.squeeze((1,) * (dt.ndim - timestep.ndim))  # Remove extra dims from dt to match timestep

      noise = jax.random.normal(generator, sample.shape, dtype=self.dtype)
      prev_sample = self.add_noise(state.common, x0, noise, next_timestep)
    else:
      prev_sample = sample - dt * model_output

    if not return_dict:
      return (prev_sample, state)

    return FlaxRectifiedFlowSchedulerOutput(prev_sample=prev_sample, state=state)
