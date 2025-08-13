import jax.numpy as jnp
from maxdiffusion.schedulers.scheduling_rectified_flow import FlaxRectifiedFlowMultistepScheduler
import os
from maxdiffusion import max_logging
import torch
import unittest
from absl.testing import absltest
import numpy as np



class rfTest(unittest.TestCase):

    def test_rf_steps(self):
        # --- Simulation Parameters ---
        latent_tensor_shape = (1, 256, 128) # Example latent tensor shape (Batch, Channels, Height, Width)
        inference_steps_count = 5     # Number of steps for the denoising process

        # --- Run the Simulation ---
        max_logging.log("\n--- Simulating RectifiedFlowMultistepScheduler ---")

        seed = 42
        device = 'cpu'
        max_logging.log(f"Sample shape: {latent_tensor_shape}, Inference steps: {inference_steps_count}, Seed: {seed}")

        generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Instantiate the scheduler
        config = {'_class_name': 'RectifiedFlowScheduler', '_diffusers_version': '0.25.1', 'num_train_timesteps': 1000, 'shifting': None, 'base_resolution': None, 'sampler': 'LinearQuadratic'}
        flax_scheduler = FlaxRectifiedFlowMultistepScheduler.from_config(config)

        # 2. Create and set initial state for the scheduler
        flax_state = flax_scheduler.create_state()
        flax_state = flax_scheduler.set_timesteps(flax_state, inference_steps_count, latent_tensor_shape)
        max_logging.log("\nScheduler initialized.")
        max_logging.log(f"  flax_state timesteps shape: {flax_state.timesteps.shape}")

        # 3. Prepare the initial noisy latent sample
        # In a real scenario, this would typically be pure random noise (e.g., N(0,1))
        # For simulation, we'll generate it.

        sample = jnp.array(torch.randn(latent_tensor_shape, generator=generator, dtype=torch.float32).to(device).numpy())
        max_logging.log(f"\nInitial sample shape: {sample.shape}, dtype: {sample.dtype}")

        # 4. Simulate the denoising loop
        max_logging.log("\nStarting denoising loop:")
        for i, t in enumerate(flax_state.timesteps):
            max_logging.log(f"  Step {i+1}/{inference_steps_count}, Timestep: {t.item()}")

            # Simulate model_output (e.g., noise prediction from a UNet)
            model_output = jnp.array(torch.randn(latent_tensor_shape, generator=generator, dtype=torch.float32).to(device).numpy())

            # Call the scheduler's step function
            scheduler_output = flax_scheduler.step(
                state=flax_state,
                model_output=model_output,
                timestep=t, # Pass the current timestep from the scheduler's sequence
                sample=sample,
                return_dict=True # Return a SchedulerOutput dataclass
            )

            sample = scheduler_output.prev_sample # Update the sample for the next step
            flax_state = scheduler_output.state # Update the state for the next step

            # Compare with pytorch implementation
            base_dir = os.path.dirname(__file__)
            ref_dir = os.path.join(base_dir, "rf_scheduler_test_ref")
            ref_filename = os.path.join(ref_dir, f"step_{i+1:02d}.npy")
            if os.path.exists(ref_filename):
                pt_sample = np.load(ref_filename)
                torch.testing.assert_close(np.array(sample), pt_sample)
            else:
                max_logging.log(f"Warning: Reference file not found: {ref_filename}")


        max_logging.log("\nDenoising loop completed.")
        max_logging.log(f"Final sample shape: {sample.shape}, dtype: {sample.dtype}")
        max_logging.log(f"Final sample min: {sample.min().item():.4f}, max: {sample.max().item():.4f}")

        max_logging.log("\nSimulation of RectifiedMultistepScheduler usage complete.")


if __name__ == "__main__":
  absltest.main()
