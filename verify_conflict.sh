print("--- PyTorch vs. JAX Conflict Test ---")

print("\nStep 1: Attempting to import torch...")
try:
    import torch
    print(f"Successfully imported torch version: {torch.__version__}")
    # This check will confirm you have the CPU-only version
    print(f"Is PyTorch using CUDA? -> {torch.cuda.is_available()}")
except Exception as e:
    print(f"Failed to import torch: {e}")


print("\nStep 2: Now, attempting to initialize JAX...")
try:
    import jax
    devices = jax.devices()
    print("\n--- RESULT: SUCCESS ---")
    print(f"JAX initialized correctly and found devices: {devices}")
except Exception as e:
    print("\n--- RESULT: FAILURE ---")
    print("JAX failed to initialize after PyTorch was imported.")
    print(f"JAX Error: {e}")