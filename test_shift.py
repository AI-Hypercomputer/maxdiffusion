import jax
import jax.numpy as jnp
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from diffusers import FlowMatchEulerDiscreteScheduler
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler

def test_schedulers():
    pt_scheduler = FlowMatchEulerDiscreteScheduler(
        shift=3.0,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )
    fx_scheduler = FlaxFlowMatchScheduler(
        shift=3.0,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )

    num_steps = 40
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps).tolist()
    mu = 1.25
    
    pt_scheduler.set_timesteps(num_inference_steps=num_steps, sigmas=sigmas, mu=mu)
    pt_sigmas = pt_scheduler.sigmas.numpy()

    state = fx_scheduler.create_state()
    state = fx_scheduler.set_timesteps_ltx2(
        state,
        num_inference_steps=num_steps,
        sigmas=jnp.array(sigmas),
        shift=mu
    )
    fx_sigmas = np.array(state.sigmas)
    
    # PT adds 0 at the end. FX doesn't inside set_timesteps
    print(f"PT sigmas length: {len(pt_sigmas)}")
    print(f"FX sigmas length: {len(fx_sigmas)}")
    
    # Check if FX up to the last matches PT (excluding the terminal 0.0)
    diff = np.max(np.abs(pt_sigmas[:-1] - fx_sigmas))
    print(f"Max Diff (first 40 steps): {diff}")
    print(pt_sigmas[:5])
    print(fx_sigmas[:5])

if __name__ == "__main__":
    test_schedulers()
