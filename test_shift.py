import torch
import jax.numpy as jnp
from diffusers import FlowMatchEulerDiscreteScheduler
import numpy as np

def pytorch_shifting():
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=3.0,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )
    # LTX2 passes these
    num_inference_steps = 40
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()
    
    mu = 1.25 # hypothetical calculated shift
    
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        sigmas=sigmas,
        mu=mu
    )
    return scheduler.sigmas.numpy()

def flax_shifting():
    num_inference_steps = 40
    sigmas = jnp.array(np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
    
    current_shift = 1.25
    
    # Flax dynamic shifting logic candidate
    shifted_sigmas = np.exp(current_shift) / (np.exp(current_shift) + (1 / sigmas - 1)**1.0)
    
    # Diffusers terminal handling:
    # sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    
    return np.concatenate([shifted_sigmas, np.zeros(1)])

pt = pytorch_shifting()
fx = flax_shifting()

print(f"PT max diff to FX: {np.max(np.abs(pt - fx))}")
print(f"PT sigmas: {pt[:5]}")
print(f"FX sigmas: {fx[:5]}")
