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

# DISCLAIMER: reference pytorch implementation: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_unipc_multistep.py

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import flax
import jax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import is_scipy_available
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    add_noise_common,
)


@flax.struct.dataclass
class UniPCMultistepSchedulerState:
  """
  Data class to hold the mutable state of the FlaxUniPCMultistepScheduler.
  """

  common: CommonSchedulerState

  # Core schedule parameters (derived from CommonSchedulerState in create_state)
  sigmas: jnp.ndarray
  alpha_t: jnp.ndarray
  sigma_t: jnp.ndarray
  lambda_t: jnp.ndarray
  init_noise_sigma: float

  # History buffers for multi-step solver
  # `model_outputs` stores previous converted model outputs (e.g., predicted x0 or epsilon)
  timesteps: jnp.ndarray = None
  model_outputs: jnp.ndarray = None
  timestep_list: jnp.ndarray = None  # Stores corresponding timesteps for `model_outputs`

  # State variables for tracking progress and solver order
  lower_order_nums: int = 0
  last_sample: Optional[jnp.ndarray] = None  # Sample from the previous predictor step
  step_index: Optional[int] = None
  begin_index: Optional[int] = None  # Used for img2img/inpaing
  this_order: int = 0  # Current effective order of the UniPC solver for this step

  @classmethod
  def create(
      cls,
      common_state: CommonSchedulerState,
      alpha_t: jnp.ndarray,
      sigma_t: jnp.ndarray,
      lambda_t: jnp.ndarray,
      sigmas: jnp.ndarray,
      init_noise_sigma: jnp.ndarray,
  ):
    return cls(
        common=common_state,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        lambda_t=lambda_t,
        sigmas=sigmas,
        init_noise_sigma=init_noise_sigma,
        lower_order_nums=0,
        last_sample=None,
        step_index=None,
        begin_index=None,
        this_order=0,
    )


@dataclass
class FlaxUniPCMultistepSchedulerOutput(FlaxSchedulerOutput):
  state: UniPCMultistepSchedulerState


class FlaxUniPCMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):
  """
  `FlaxUniPCMultistepScheduler` is a JAX/Flax training-free framework designed for the fast sampling of diffusion models.
  It implements the UniPC (Unified Predictor-Corrector) algorithm for efficient diffusion model sampling.
  """

  dtype: jnp.dtype

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
      trained_betas: Optional[Union[jnp.ndarray, List[float]]] = None,
      solver_order: int = 2,
      prediction_type: str = "epsilon",
      thresholding: bool = False,
      dynamic_thresholding_ratio: float = 0.995,
      sample_max_value: float = 1.0,
      predict_x0: bool = True,
      solver_type: str = "bh2",
      lower_order_final: bool = True,
      disable_corrector: List[int] = [],
      solver_p: Optional[FlaxSchedulerMixin] = None,
      use_karras_sigmas: Optional[bool] = False,
      use_exponential_sigmas: Optional[bool] = False,
      use_beta_sigmas: Optional[bool] = False,
      use_flow_sigmas: Optional[bool] = False,
      flow_shift: Optional[float] = 1.0,
      timestep_spacing: str = "linspace",
      steps_offset: int = 0,
      final_sigmas_type: Optional[str] = "zero",
      rescale_zero_terminal_snr: bool = False,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.dtype = dtype

    # Validation checks from original __init__
    if self.config.use_beta_sigmas and not is_scipy_available():
      raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
    if (
        sum(
            [
                self.config.use_beta_sigmas,
                self.config.use_exponential_sigmas,
                self.config.use_karras_sigmas,
            ]
        )
        > 1
    ):
      raise ValueError(
          "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
      )
    if self.config.solver_type not in ["bh1", "bh2"]:
      raise NotImplementedError(f"{self.config.solver_type} is not implemented for {self.__class__}")

  def create_state(self, common: Optional[CommonSchedulerState] = None) -> UniPCMultistepSchedulerState:
    if common is None:
      common = CommonSchedulerState.create(self)

    if self.config.get("rescale_zero_terminal_snr", False):
      # Close to 0 without being 0 so first sigma is not inf
      # FP16 smallest positive subnormal works well here
      alphas_cumprod = common.alphas_cumprod
      alphas_cumprod = alphas_cumprod.at[-1].set(2**-24)
      common = common.replace(alphas_cumprod=alphas_cumprod)

    # Currently we only support VP-type noise schedule
    alpha_t = jnp.sqrt(common.alphas_cumprod)
    sigma_t = jnp.sqrt(1 - common.alphas_cumprod)
    lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
    sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5

    # standard deviation of the initial noise distribution
    init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

    if self.config.solver_type not in ["bh1", "bh2"]:
      if self.config.solver_type in ["midpoint", "heun", "logrho"]:
        self.config.solver_type = "bh2"
      else:
        raise NotImplementedError(f"{self.config.solver_type} is not implemented for {self.__class__}")

    return UniPCMultistepSchedulerState.create(
        common_state=common,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        lambda_t=lambda_t,
        sigmas=sigmas,
        init_noise_sigma=init_noise_sigma,
    )

  def set_begin_index(self, state: UniPCMultistepSchedulerState, begin_index: int = 0) -> UniPCMultistepSchedulerState:
    """
    Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
    """
    return state.replace(begin_index=begin_index)

  def set_timesteps(
      self,
      state: UniPCMultistepSchedulerState,
      num_inference_steps: int,
      shape: Tuple,
  ) -> UniPCMultistepSchedulerState:
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).
    """
    #### Copied from scheduling_dmpsolver_multistep_flax
    last_timestep = self.config.num_train_timesteps
    if self.config.timestep_spacing == "linspace":
      timesteps = jnp.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].astype(jnp.int32)
    elif self.config.timestep_spacing == "leading":
      step_ratio = last_timestep // (num_inference_steps + 1)
      # creates integer timesteps by multiplying by ratio
      # casting to int to avoid issues when num_inference_step is power of 3
      timesteps = (jnp.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(jnp.int32)
      timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == "trailing":
      step_ratio = self.config.num_train_timesteps / num_inference_steps
      # creates integer timesteps by multiplying by ratio
      # casting to int to avoid issues when num_inference_step is power of 3
      timesteps = jnp.arange(last_timestep, 0, -step_ratio).round().copy().astype(jnp.int32)
      timesteps -= 1
    else:
      raise ValueError(
          f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
      )

    # initial running values
    sigmas = state.sigmas

    # TODO
    # # Apply Karras/Exponential/Beta/Flow Sigmas if configured
    if self.config.use_karras_sigmas:
      # sigmas = _convert_to_karras_jax(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
      # timesteps = jnp.array([_sigma_to_t_jax(s, log_sigmas_full) for s in sigmas]).round().astype(jnp.int64)
      raise NotImplementedError("`use_karras_sigmas` is not implemented in JAX version yet.")
    elif self.config.use_exponential_sigmas:
      # sigmas = _convert_to_exponential_jax(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
      # timesteps = jnp.array([_sigma_to_t_jax(s, log_sigmas_full) for s in sigmas]).round().astype(jnp.int64)
      raise NotImplementedError("`use_exponential_sigmas` is not implemented in JAX version yet.")
    elif self.config.use_beta_sigmas:
      # sigmas = _convert_to_beta_jax(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
      # timesteps = jnp.array([_sigma_to_t_jax(s, log_sigmas_full) for s in sigmas]).round().astype(jnp.int64)
      raise NotImplementedError("`use_beta_sigmas` is not implemented in JAX version yet.")
    if self.config.use_flow_sigmas:
      alphas = jnp.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
      sigmas = 1.0 - alphas
      sigmas = jnp.flip(self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas))[:-1].copy()
      timesteps = (sigmas * self.config.num_train_timesteps).copy().astype(jnp.int64)
      if self.config.final_sigmas_type == "sigma_min":
        sigma_last = sigmas[-1]
      elif self.config.final_sigmas_type == "zero":
        sigma_last = 0
      else:
        raise ValueError(
            f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
        )
      sigmas = jnp.concatenate([sigmas, jnp.array([sigma_last])]).astype(jnp.float32)
    else:  # Default case if none of the specialized sigmas are used
      sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
      if self.config.final_sigmas_type == "sigma_min":
        sigma_last = ((1 - state.common.alphas_cumprod[0]) / state.common.alphas_cumprod[0]) ** 0.5
      elif self.config.final_sigmas_type == "zero":
        sigma_last = 0
      else:
        raise ValueError(
            f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
        )
      sigmas = jnp.concatenate([sigmas, jnp.array([sigma_last])]).astype(jnp.float32)

    model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=self.dtype)
    timestep_list = jnp.zeros((self.config.solver_order,), dtype=jnp.int32)  # Timesteps are integers
    # Update the state with the new schedule and re-initialized history
    return state.replace(
        timesteps=timesteps,
        sigmas=sigmas,
        model_outputs=model_outputs,
        timestep_list=timestep_list,
        lower_order_nums=0,  # Reset counters for a new inference run
        step_index=None,
        begin_index=None,
        last_sample=None,
        this_order=0,
    )

  def convert_model_output(
      self,
      state: UniPCMultistepSchedulerState,
      model_output: jnp.ndarray,
      sample: jnp.ndarray,
  ) -> jnp.ndarray:
    """
    Converts the model output based on the prediction type and current state.
    """
    sigma = state.sigmas[state.step_index]  # Current sigma

    # Ensure sigma is a JAX array for _sigma_to_alpha_sigma_t
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

    if self.config.predict_x0:
      if self.config.prediction_type == "epsilon":
        x0_pred = (sample - sigma_t * model_output) / alpha_t
      elif self.config.prediction_type == "sample":
        x0_pred = model_output
      elif self.config.prediction_type == "v_prediction":
        x0_pred = alpha_t * sample - sigma_t * model_output
      elif self.config.prediction_type == "flow_prediction":
        # Original code has `sigma_t = self.sigmas[self.step_index]`.
        # This implies current sigma `sigma` is used as sigma_t for flow.
        x0_pred = sample - sigma * model_output
      else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
            "`v_prediction`, or `flow_prediction` for the UniPCMultistepScheduler."
        )

      if self.config.thresholding:
        raise NotImplementedError("Dynamic thresholding isn't implemented.")
        # x0_pred = self._threshold_sample(x0_pred)
      return x0_pred
    else:  # self.config.predict_x0 is False
      if self.config.prediction_type == "epsilon":
        return model_output
      elif self.config.prediction_type == "sample":
        epsilon = (sample - alpha_t * model_output) / sigma_t
        return epsilon
      elif self.config.prediction_type == "v_prediction":
        epsilon = alpha_t * model_output + sigma_t * sample
        return epsilon
      else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction` for the UniPCMultistepScheduler."
        )

  def multistep_uni_p_bh_update(
      self,
      state: UniPCMultistepSchedulerState,
      model_output: jnp.ndarray,  # Original model output from the diffusion model
      sample: jnp.ndarray,
      order: int,
  ) -> jnp.ndarray:
    """
    One step for the UniP (B(h) version) - the Predictor.
    """
    if self.config.solver_p:
      raise NotImplementedError("Nested `solver_p` is not implemented in JAX version yet.")

    m0 = state.model_outputs[self.config.solver_order - 1]  # Most recent stored converted model output
    x = sample

    sigma_t_val, sigma_s0_val = (
        state.sigmas[state.step_index + 1],
        state.sigmas[state.step_index],
    )

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_val)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_val)

    lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
    lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)

    h = lambda_t - lambda_s0

    rks_list = []
    D1s_list = []

    for i in range(1, order):
      history_idx = self.config.solver_order - 1 - i  # Correct index for history array

      mi = state.model_outputs[history_idx]
      si_val = state.timestep_list[history_idx]  # This is the actual timestep value

      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(state.sigmas[self.index_for_timestep(state, si_val)])
      lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si + 1e-10)

      rk = (lambda_si - lambda_s0) / h
      rks_list.append(rk)
      D1s_list.append((mi - m0) / rk)

    rks_list.append(1.0)  # Append the last 1.0 for r_order
    rks = jnp.stack(rks_list)  # Shape (order,)

    R_list = []
    b_list = []

    hh = -h if self.config.predict_x0 else h
    h_phi_1 = jnp.expm1(hh)

    current_h_phi_k = h_phi_1 / hh - 1.0
    factorial_val = 1.0  # factorial(1) is 1. For `factorial_i *= i + 1`

    if self.config.solver_type == "bh1":
      B_h = hh
    elif self.config.solver_type == "bh2":
      B_h = jnp.expm1(hh)
    else:
      raise NotImplementedError()

    for i in range(1, order + 1):  # Loop from i=1 to order
      R_list.append(jnp.power(rks, i - 1))
      b_list.append(current_h_phi_k * factorial_val / B_h)

      if i < order:  # Update for next iteration (i+1)
        factorial_val *= i + 1
        current_h_phi_k = current_h_phi_k / hh - 1.0 / factorial_val  # Update for next i

    R = jnp.stack(R_list)  # Shape (order, order)
    b = jnp.stack(b_list)  # Shape (order,)

    D1s = None
    if len(D1s_list) > 0:
      D1s = jnp.stack(D1s_list, axis=1)  # Resulting shape (B, K, C, H, W)

    if order == 2:  # Special case for order 2 from original
      rhos_p = jnp.array([0.5], dtype=x.dtype)
    else:  # General case, solve linear system

      rhos_p = jnp.linalg.solve(R[:-1, :-1], b[:-1]).astype(x.dtype)

    if self.config.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      if D1s is not None:
        # einsum `k,bkc...->bc...` where k is rhos_p dim, b is batch, c is channel, ...
        pred_res = jnp.einsum("k,bkc...->bc...", rhos_p, D1s)
      else:
        pred_res = 0.0
      x_t = x_t_ - alpha_t * B_h * pred_res
    else:  # Predict epsilon
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      if D1s is not None:
        pred_res = jnp.einsum("k,bkc...->bc...", rhos_p, D1s)
      else:
        pred_res = 0.0
      x_t = x_t_ - sigma_t * B_h * pred_res

    return x_t.astype(x.dtype)

  def multistep_uni_c_bh_update(
      self,
      state: UniPCMultistepSchedulerState,
      this_model_output: jnp.ndarray,
      last_sample: jnp.ndarray,  # Sample after predictor `x_{t-1}`
      this_sample: jnp.ndarray,  # Sample before corrector `x_t` (after predictor step)
      order: int,
  ) -> jnp.ndarray:
    """
    One step for the UniC (B(h) version) - the Corrector.
    """
    model_output_list = state.model_outputs  # History buffer
    m0 = model_output_list[self.config.solver_order - 1]  # Most recent model output from history

    x = last_sample  # Sample after predictor (`x_{t-1}`)
    x_t = this_sample  # Sample after predictor (`x_t`)
    model_t = this_model_output  # The new model output evaluated at `x_t`

    sigma_t_val = state.sigmas[state.step_index]
    sigma_s0_val = state.sigmas[state.step_index - 1]  # This is the sigma corresponding to `x` (last_sample)

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_val)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_val)

    lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
    lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)

    h = lambda_t - lambda_s0

    rks_list = []
    D1s_list = []

    for i in range(1, order):
      history_idx = self.config.solver_order - (i + 1)  # Index in the fixed-size history array

      mi = state.model_outputs[history_idx]
      si_val = state.timestep_list[history_idx]  # This is the actual timestep value

      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(state.sigmas[self.index_for_timestep(state, si_val)])
      lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si + 1e-10)

      rk = (lambda_si - lambda_s0) / h
      rks_list.append(rk)
      D1s_list.append((mi - m0) / rk)

    rks_list.append(1.0)
    rks = jnp.stack(rks_list)

    R_list = []
    b_list = []

    hh = -h if self.config.predict_x0 else h
    h_phi_1 = jnp.expm1(hh)

    # Calculate h_phi_k values for coefficients
    current_h_phi_k = h_phi_1 / hh - 1.0  # Initial value for i=1
    factorial_val = 1.0

    if self.config.solver_type == "bh1":
      B_h = hh
    elif self.config.solver_type == "bh2":
      B_h = jnp.expm1(hh)
    else:
      raise NotImplementedError()

    for i in range(1, order + 1):
      R_list.append(jnp.power(rks, i - 1))
      b_list.append(current_h_phi_k * factorial_val / B_h)

      if i < order:
        factorial_val *= i + 1
        current_h_phi_k = current_h_phi_k / hh - 1.0 / factorial_val

    R = jnp.stack(R_list)
    b = jnp.stack(b_list)

    D1s = None
    if len(D1s_list) > 0:
      D1s = jnp.stack(D1s_list, axis=1)  # (B, K, C, H, W)

    if order == 1:
      rhos_c = jnp.array([0.5], dtype=x.dtype)
    else:
      rhos_c = jnp.linalg.solve(R, b).astype(x.dtype)  # Use all of R and b for corrector

    if self.config.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      if D1s is not None:
        # einsum `k,bkc...->bc...` where k is rhos_c[:-1] dim
        corr_res = jnp.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
      else:
        corr_res = 0.0
      D1_t = model_t - m0
      x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    else:
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      if D1s is not None:
        corr_res = jnp.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
      else:
        corr_res = 0.0
      D1_t = model_t - m0
      x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

    return x_t.astype(x.dtype)

  def index_for_timestep(
      self,
      state: UniPCMultistepSchedulerState,
      timestep: Union[int, jnp.ndarray],
      schedule_timesteps: Optional[jnp.ndarray] = None,
  ) -> int:
    """ "Gets the step_index for timestep."""
    if schedule_timesteps is None:
      schedule_timesteps = state.timesteps

    timestep_val = timestep.item() if isinstance(timestep, jnp.ndarray) and timestep.ndim == 0 else timestep

    index_candidates = jnp.where(schedule_timesteps == timestep_val, size=1, fill_value=-1)[0]

    if index_candidates[0] == -1:  # No match found
      step_index = len(schedule_timesteps) - 1  # Default to last index
    elif len(index_candidates) > 1:
      step_index = index_candidates[1].item()  # Take the second match (diffusers behavior)
    else:
      step_index = index_candidates[0].item()  # Take the first (and only) match
    return step_index

  def _init_step_index(
      self, state: UniPCMultistepSchedulerState, timestep: Union[int, jnp.ndarray]
  ) -> UniPCMultistepSchedulerState:
    """Initializes the step_index counter for the scheduler."""
    if state.begin_index is None:
      step_index_val = self.index_for_timestep(state, timestep)
      return state.replace(step_index=step_index_val)
    else:
      return state.replace(step_index=state.begin_index)

  def step(
      self,
      state: UniPCMultistepSchedulerState,
      model_output: jnp.ndarray,  # This is the direct output from the diffusion model (e.g., noise prediction)
      timestep: Union[int, jnp.ndarray],  # Current discrete timestep from the scheduler's sequence
      sample: jnp.ndarray,  # Current noisy sample (latent)
      return_dict: bool = True,
      generator: Optional[jax.random.PRNGKey] = None,  # JAX random key
  ) -> Union[FlaxUniPCMultistepSchedulerOutput, Tuple[jnp.ndarray]]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
    the multistep UniPC.
    """
    if state.timesteps is None:
      raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
    # Ensure timestep is a scalar for indexing/comparison
    timestep_scalar = (
        timestep.item() if isinstance(timestep, jnp.ndarray) and timestep.ndim == 0 else int(timestep)
    )  # Ensure int type

    # Initialize step_index if it's the first step
    if state.step_index is None:
      state = self._init_step_index(state, timestep_scalar)

    # Determine if corrector should be used
    use_corrector = (
        state.step_index > 0
        and state.step_index - 1 not in self.config.disable_corrector
        and state.last_sample is not None  # `last_sample` stores output of previous predictor
    )

    # Convert model_output (noise/v_pred) to x0_pred or epsilon_pred, based on prediction_type
    model_output_for_history = self.convert_model_output(state, model_output, sample)

    # Apply corrector if applicable
    if use_corrector:
      corrected_sample = self.multistep_uni_c_bh_update(
          state=state,
          this_model_output=model_output_for_history,
          last_sample=state.last_sample,
          this_sample=sample,
          order=state.this_order,
      )
      sample = corrected_sample

    # Update history buffers (model_outputs and timestep_list)
    # Shift existing elements to the left and add new one at the end.
    # `state.model_outputs` and `state.timestep_list` are fixed-size arrays.
    # Example:
    # t0:[None,...,model_output0]
    # t1:[None,..model_output0,model_output1]
    # ...
    # tn:[model_output0,model_output1,...,model_output_n]
    if state.step_index == 0:
      updated_model_outputs_history = state.model_outputs.at[-1].set(model_output_for_history)
      updated_timestep_list_history = state.timestep_list.at[-1].set(timestep_scalar)
    else:
      updated_model_outputs_history = jnp.roll(state.model_outputs, shift=-1, axis=0)
      updated_model_outputs_history = updated_model_outputs_history.at[-1].set(model_output_for_history)

      updated_timestep_list_history = jnp.roll(state.timestep_list, shift=-1)
      updated_timestep_list_history = updated_timestep_list_history.at[-1].set(timestep_scalar)

    state = state.replace(
        model_outputs=updated_model_outputs_history,
        timestep_list=updated_timestep_list_history,
    )

    # Determine the order for the current step (warmup phase logic)
    if self.config.lower_order_final:
      this_order = jnp.minimum(self.config.solver_order, len(state.timesteps) - state.step_index)
    else:
      this_order = self.config.solver_order

    # Warmup for multistep: `this_order` can't exceed `lower_order_nums + 1`
    new_this_order = jnp.minimum(this_order, state.lower_order_nums + 1)
    state = state.replace(this_order=new_this_order)

    # Ensure `this_order` is positive, should always be.
    assert new_this_order > 0, "Solver order must be positive."

    # Store current sample as `last_sample` for the *next* step's corrector
    state = state.replace(last_sample=sample)

    # UniP predictor step
    prev_sample = self.multistep_uni_p_bh_update(
        state=state,
        model_output=model_output,
        sample=sample,
        order=state.this_order,
    )

    # Update lower_order_nums for warmup
    if state.lower_order_nums < self.config.solver_order:
      state = state.replace(lower_order_nums=state.lower_order_nums + 1)

    # Upon completion, increase step index by one
    state = state.replace(step_index=state.step_index + 1)

    # Return the updated sample and state
    if not return_dict:
      return (prev_sample, state)

    return FlaxUniPCMultistepSchedulerOutput(prev_sample=prev_sample, state=state)

  def scale_model_input(self, state: UniPCMultistepSchedulerState, sample: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    """
    UniPC does not scale model input, so it returns the sample unchanged.
    """
    return sample

  def add_noise(
      self,
      state: UniPCMultistepSchedulerState,
      original_samples: jnp.ndarray,
      noise: jnp.ndarray,
      timesteps: jnp.ndarray,
  ) -> jnp.ndarray:
    return add_noise_common(state.common, original_samples, noise, timesteps)

  def _sigma_to_alpha_sigma_t(self, sigma):
    if self.config.use_flow_sigmas:
      alpha_t = 1 - sigma
      sigma_t = sigma
    else:
      alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
      sigma_t = sigma * alpha_t

    return alpha_t, sigma_t

  def __len__(self) -> int:
    return self.config.num_train_timesteps
