import jax.numpy as jnp
from maxdiffusion.schedulers.scheduling_rectified_flow import FlaxRectifiedFlowMultistepScheduler
import os
from huggingface_hub import hf_hub_download
import torch
import unittest
from absl.testing import absltest
from absl import flags # Import absl.flags
import numpy as np
import torch

# Define a command-line flag for models_dir
FLAGS = flags.FLAGS
flags.DEFINE_string('models_dir', None, 'Directory to load scheduler config.')
flags.mark_flag_as_required('models_dir')



class rfTest(unittest.TestCase):

    def test_rf_steps(self):
        # --- Configuration Parameters for the Scheduler ---
        # You can modify these parameters to test different scheduler behaviors

        # --- Simulation Parameters ---
        latent_tensor_shape = (1, 256, 128) # Example latent tensor shape (Batch, Channels, Height, Width)
        inference_steps_count = 5     # Number of steps for the denoising process

        # --- Run the Simulation ---
        # Use the value from the command-line flag
        models_dir = FLAGS.models_dir
        
        # Ensure the directory exists before downloading
        os.makedirs(models_dir, exist_ok=True)

        ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename="ltxv-13b-0.9.7-dev.safetensors",
            local_dir=models_dir,
            repo_type="model",
        )
        print(f"\n--- Simulating RectifiedFlowMultistepScheduler ---")

        seed = 42
        device = 'cpu'
        print(f"Sample shape: {latent_tensor_shape}, Inference steps: {inference_steps_count}, Seed: {seed}")

        generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Instantiate the scheduler
        flax_scheduler = FlaxRectifiedFlowMultistepScheduler.from_pretrained_jax(ltxv_model_path)

        # 2. Create and set initial state for the scheduler
        flax_state = flax_scheduler.create_state()
        flax_state = flax_scheduler.set_timesteps(flax_state, inference_steps_count, latent_tensor_shape)
        print("\nScheduler initialized.")
        print(f"  flax_state timesteps shape: {flax_state.timesteps.shape}")

        # 3. Prepare the initial noisy latent sample
        # In a real scenario, this would typically be pure random noise (e.g., N(0,1))
        # For simulation, we'll generate it.

        sample = jnp.array(torch.randn(latent_tensor_shape, generator=generator, dtype=torch.float32).to(device).numpy())
        print(f"\nInitial sample shape: {sample.shape}, dtype: {sample.dtype}")

        # 4. Simulate the denoising loop
        print("\nStarting denoising loop:")
        for i, t in enumerate(flax_state.timesteps):
            print(f"  Step {i+1}/{inference_steps_count}, Timestep: {t.item()}")

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
            # Ensure the reference directory exists for tests that might write to it,
            # or handle its absence if it's meant to be pre-existing.
            # For this example, assuming 'rf_scheduler_test_ref' exists with pre-saved .npy files.
            if os.path.exists(ref_filename):
                pt_sample = np.load(ref_filename)
                torch.testing.assert_close(np.array(sample), pt_sample)
            else:
                print(f"Warning: Reference file not found: {ref_filename}")


        print("\nDenoising loop completed.")
        print(f"Final sample shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"Final sample min: {sample.min().item():.4f}, max: {sample.max().item():.4f}")

        print("\nSimulation of RectifiedMultistepScheduler usage complete.")


if __name__ == "__main__":
  # absltest.main() automatically parses flags defined by absl.flags
  absltest.main()