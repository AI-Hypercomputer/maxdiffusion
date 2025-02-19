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
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import math
import flax
import numpy as np
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import is_scipy_available
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
)

if is_scipy_available():
  import scipy.stats

class FlaxFlowMatchEulerDiscreteScheduler:
  pass

@flax.struct.dataclass
class FlowMatchEulerDiscreteSchedulerState:

  # setable values
  timesteps: jnp.ndarray
  sigmas: jnp.ndarray
  sigma_min: float = None
  sigma_max: float = None
  step_index: int = None
  begin_index: int = None
  num_inference_steps: Optional[int] = None

  @classmethod
  def create(cls, step_index: int, begin_index: int, sigma_min: float, sigma_max:float, timesteps: jnp.ndarray, sigmas: jnp.ndarray):
    return cls(step_index=step_index, begin_index=begin_index, sigma_min=sigma_min, sigma_max=sigma_max, timesteps=timesteps, sigmas=sigmas)

@dataclass
class FlaxFlowMatchEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
  state: FlowMatchEulerDiscreteSchedulerState

class FlaxFlowMatchEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):

  _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]
  dtype: jnp.dtype

  @property
  def has_state(self):
    return True
  
  @register_to_config
  def __init__(
    self,
    num_train_timesteps: int = 1000,
    shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    base_shift: Optional[float] = 0.5,
    max_shift: Optional[float] = 1.15,
    base_image_seq_len: Optional[int] = 256,
    max_image_seq_len: Optional[int] = 4096,
    invert_sigmas: bool = False,
    shift_terminal: Optional[float] = None,
    use_karras_sigmas: Optional[bool] = False,
    use_exponential_sigmas: Optional[bool] = False,
    use_beta_sigmas: Optional[bool] = False,
    time_shift_type: str = "exponential",
    dtype: jnp.dtype = jnp.float32
  ):
    self.dtype = dtype
    if self.config.use_beta_sigmas and not is_scipy_available():
      raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
    if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
        raise ValueError(
            "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
        )
    if time_shift_type not in {"exponential", "linear"}:
        raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")
  
  def create_state(self) -> FlowMatchEulerDiscreteSchedulerState:
    
    timesteps = jnp.arange(1, self.config.num_train_timesteps).round()[::-1]
    sigmas = timesteps / self.config.num_train_timesteps
    if not self.config.use_dynamic_shifting:
      sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
    
    self.timesteps = sigmas * self.config.num_train_timesteps

    return FlowMatchEulerDiscreteSchedulerState.create(
      step_index=None,
      begin_index=None,
      timesteps=timesteps,
      sigma_min=sigmas[-1],
      sigma_max=sigmas[0],
      sigmas=sigmas
    )

  def _sigma_to_t(self, sigma):
    return sigma * self.config.num_train_timesteps

  def time_shift(self, mu: float, sigma: float, t: jnp.ndarray) -> jnp.ndarray:
    if self.config.time_shift_type == "exponential":
      return self._time_shift_exponential(mu, sigma, t)
    elif self.config.time_shift_type == "linear":
      return self._time_shift_linear(mu, sigma, t)
  
  def set_timesteps(
      self,
      state: FlowMatchEulerDiscreteSchedulerState,
      num_inference_steps: Optional[int] = None,
      mu: Optional[float] = None,
      sigmas: Optional[List[float]] = None,
      timesteps: Optional[List[float]] = None
  ):
    if self.config.use_dynamic_shifting and mu is None:
      raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to `True`")
    if sigmas is not None and timesteps is not None:
      if len(sigmas) != len(timesteps):
        raise ValueError("`sigmas` and `timesteps` should have the same length")
    
    if num_inference_steps is not None:
      if (sigmas is not None and len(sigmas) != num_inference_steps) or (
        timesteps is not None and len(timesteps) != num_inference_steps
      ):
        raise ValueError(
          "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
        )
    else:
      num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)
    
    # 1. Prepare default sigmas
    is_timesteps_provided = timesteps is not None

    if is_timesteps_provided:
      timesteps = jnp.array(timesteps, dtype=jnp.float32)
    
    if sigmas is None:
      if timesteps is None:
        timesteps = jnp.linspace(
          self._sigma_to_t(state.sigma_max), self._sigma_to_t(state.sigma_min), num_inference_steps
        )
      sigmas = timesteps / self.config.num_train_timesteps
    else:
      sigmas = jnp.array(sigmas, dtype=jnp.float32)
      num_inference_steps = len(sigmas)
    
    # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting
    # of "exponential" or "linear" type is applied
    if self.config.use_dynamic_shifting:
      sigmas = self.time_shift(mu, 1.0, sigmas)
    else:
      sigmas = self.config.shift * sigmas / (1+ (self.config.shift - 1) * sigmas)
    
    # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value.
    if self.config.shift_terminal:
      sigmas = self.stretch_shift_to_terminal(sigmas)
    
    # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
    if self.config.use_karras_sigmas:
      sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    elif self.config.use_exponential_sigmas:
      sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    elif self.config.use_beta_sigmas:
      sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    
    # 5. Verify sigmas and timesteps are jnp arrays
    if not is_timesteps_provided:
      timesteps = sigmas * self.config.num_train_timesteps

    # 6. Append the terminal sigma value.
    # If a model requires inverted sigma schedule for denoising but timesteps without invertion, the
    # `invert_sigmas` flag can bet set to `True`. This case is only required in Mochi.
    if self.config.invert_sigmas:
      sigmas = 1.0 - sigmas
      timesteps = sigmas * self.config.num_train_timesteps
      sigmas = jnp.concatenate((sigmas, jnp.ones((1))))
    else:
      sigmas = jnp.concatenate((sigmas, jnp.zeros((1))))

    assert sigmas.dtype == jnp.float32
    assert timesteps.dtype == jnp.float32

    return state.replace(
      sigmas=sigmas,
      step_index=None,
      begin_index=None,
      timesteps=timesteps,
      num_inference_steps=num_inference_steps
    )
  
  
  def index_for_timestep(self, timestep, schedule_timesteps=None):
    breakpoint()
    if schedule_timesteps is None:
      schedule_timesteps = self.timesteps
    mask = schedule_timesteps == timestep
    indices = jnp.where(mask)[0]

    # The sigma index that is taken for the **very** first `step`
    # is always the second index (or the last index if there is only 1)
    # This way we can ensure we don't accidentally skip a sigma in
    # case we start in the middle of the denoising schedule (e.g. for image-to-image)
    pos = jnp.where(indices.size > 1, 1, 0)

    return indices[pos]

  def step(
    self,
    state: FlowMatchEulerDiscreteSchedulerState,
    model_output: jnp.ndarray,
    timestep: Union[float, jnp.ndarray],
    sample: jnp.ndarray,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    return_dict: bool = True
  ) -> Union[FlaxFlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_flow_match_euler_discrete_flax.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_flow_match_euler_discrete_flax.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_flow_match_euler_discrete_flax.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor.
    """
    if state.step_index is None:
      if state.begin_index is None:
        step_index = timestep#self.index_for_timestep(timestep)
      else:
        step_index = state.begin_index
    
    sample = jnp.array(sample, dtype=jnp.float32)

    sigma = state.sigmas[step_index]
    sigma_next = state.sigmas[step_index + 1]

    prev_sample = sample + (sigma_next - sigma) * model_output

    # Cast sample back to model compatible dtype
    prev_sample = jnp.array(prev_sample, dtype=model_output.dtype)

    # upon completion increase step index by one
    step_index +=1
    
    state.replace(
      step_index=step_index
    )

    if not return_dict:
      return (prev_sample, state)

    return FlaxFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample, state=state)

  def _get_sigma_min_max(self, state: FlowMatchEulerDiscreteSchedulerState, in_sigmas: jnp.ndarray):
    if hasattr(state, "sigma_min"):
      sigma_min = state.sigma_min
    else:
      sigma_min = None
    
    if hasattr(state, "sigma_max"):
      sigma_max = state.sigma_max
    else:
      sigma_max = None
    
    sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1]
    sigma_max = sigma_max if sigma_max is not None else in_sigmas[0]

    return sigma_min, sigma_max


  def _convert_to_karras(self, state: FlowMatchEulerDiscreteSchedulerState, in_sigmas: jnp.ndarray, num_inference_steps: int) -> jnp.ndarray:
    """Constructs the noise schedule of Karras et al. (2022)."""

    sigma_min, sigma_max = self._get_sigma_min_max(state, in_sigmas)

    rho = 7.0 # 7.0 is the value used in the paper
    ramp = jnp.arange(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

  def _convert_to_exponential(self, state: FlowMatchEulerDiscreteSchedulerState, in_sigmas: jnp.ndarray, num_inference_steps: int) -> jnp.ndarray:
    """Constructs an exponential noise schedule."""
    
    sigma_min, sigma_max = self._get_sigma_min_max(state, in_sigmas)

    sigmas = jnp.exp(jnp.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
    return sigmas
  
  def _convert_to_beta(self, state: FlowMatchEulerDiscreteSchedulerState, in_sigmas: jnp.ndarray, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6) -> jnp.ndarray:
    sigma_min, sigma_max = self._get_sigma_min_max(state, in_sigmas)
    sigmas = np.array(
      [
        sigma_min + (ppf * (sigma_max - sigma_min))
        for ppf in [
          scipy.stats.beta.ppf(timestep, alpha, beta) for timestep in 1 - np.linspace(0, 1, num_inference_steps)
        ]
      ]
    )
    return jnp.array(sigmas, dtype=jnp.float32)

  def _time_shift_exponential(self, mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t-1) ** sigma)
  
  def _time_shift_linear(self, mu, sigma, t):
    return mu / (mu + (1 / t - 1) ** sigma)