"""
This is a test file used for ensuring numerical parity between pytorch and jax implementation of LTX2.
This is to be ignored and will not be pushed when commiting to main branch.
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add maxdiffusion/src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from builtins import Exception
from flax import traverse_util


def load_and_convert_pytorch_weights(pth_path, maxdiffusion_model):
  """
  Loads PyTorch weights and converts them to the Flax/NNX format,
  handling stacked parameters from nnx.scan/vmap.
  """
  import torch
  print(f"Loading PyTorch state dict from {pth_path}...")
  pytorch_state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

  print("Converting weights to MaxDiffusion format for Version 1 (nnx.scan)...")
  _, state_graph = nnx.split(maxdiffusion_model)
  params = state_graph.filter(nnx.Param)
  flax_state_dict = params.to_pure_dict()

  flat_flax_params = traverse_util.flatten_dict(flax_state_dict, sep='.')
  mapped_weights = {}
  missing_keys = []
  processed_flax_keys = set()

  # layer counts for each block from the model instance
  encoder_layers = maxdiffusion_model.encoder.layers_per_block
  decoder_layers = maxdiffusion_model.decoder.layers_per_block

  block_layer_counts = {
      "encoder.down_blocks.0": encoder_layers[0],
      "encoder.down_blocks.1": encoder_layers[1],
      "encoder.down_blocks.2": encoder_layers[2],
      "encoder.down_blocks.3": encoder_layers[3],
      "encoder.mid_block": encoder_layers[4],
      "decoder.mid_block": decoder_layers[0],
      "decoder.up_blocks.0": decoder_layers[1],
      "decoder.up_blocks.1": decoder_layers[2],
      "decoder.up_blocks.2": decoder_layers[3],
  }

  for flax_key_str, flax_tensor in flat_flax_params.items():
    if "rngs" in flax_key_str or "count" in flax_key_str or "key" in flax_key_str:
      continue

    if flax_key_str in processed_flax_keys:
      continue

    pt_key_template = flax_key_str
    if pt_key_template.endswith(".kernel"):
      pt_key_template = pt_key_template.replace(".kernel", ".weight")
    elif pt_key_template.endswith(".scale"):
        pt_key_template = pt_key_template.replace(".scale", ".weight")
    elif pt_key_template.endswith(".embedding"):
        pt_key_template = pt_key_template.replace(".embedding", ".weight")


    is_stacked = False
    num_layers = 1
    block_prefix = ""
    sub_path = ""

    for prefix, count in block_layer_counts.items():
      if flax_key_str.startswith(prefix + ".resnets."):
        is_stacked = True
        num_layers = count
        block_prefix = prefix
        sub_path = flax_key_str.split(prefix + ".resnets.")[1]
        pt_key_template = sub_path
        if pt_key_template.endswith(".kernel"):
            pt_key_template = pt_key_template.replace(".kernel", ".weight")
        elif pt_key_template.endswith(".scale"):
            pt_key_template = pt_key_template.replace(".scale", ".weight")
        break

    if is_stacked:
      stacked_tensors = []
      valid_stack = True
      for i in range(num_layers):
        pt_key = f"{block_prefix}.resnets.{i}.{pt_key_template}"

        if pt_key not in pytorch_state_dict:
          if pt_key not in missing_keys: missing_keys.append(pt_key)
          valid_stack = False
          break

        pt_tensor = pytorch_state_dict[pt_key]
        if pt_tensor.dtype == torch.bfloat16:
          pt_tensor = pt_tensor.float()
        np_array = pt_tensor.numpy()

        if "conv" in pt_key and "weight" in pt_key and len(np_array.shape) == 5:
          np_array = np_array.transpose(2, 3, 4, 1, 0)
        if (("norm" in pt_key) or ("per_channel_scale" in pt_key)) and len(np_array.shape) == 1:
            pass  # No transpose for 1D arrays

        stacked_tensors.append(np_array)

      if valid_stack and stacked_tensors:
        try:
          stacked_np_array = np.stack(stacked_tensors, axis=0)
          flax_key_tuple = tuple(flax_key_str.split('.'))
          if stacked_np_array.shape == flax_tensor.shape:
            mapped_weights[flax_key_tuple] = jnp.array(stacked_np_array)
            processed_flax_keys.add(flax_key_str)
          else:
             print(f"Warning: Stacked shape mismatch for {flax_key_str} - Expected {flax_tensor.shape}, Got {stacked_np_array.shape}")
        except ValueError as e:
          print(f"Error stacking {flax_key_str}: {e}")
          for i, t in enumerate(stacked_tensors):
              print(f"  Layer {i} shape: {t.shape}")

    else:
      # Handle non-stacked parameters
      pt_key = pt_key_template
      if pt_key not in pytorch_state_dict:
        if pt_key not in missing_keys: missing_keys.append(pt_key)
        continue

      pt_tensor = pytorch_state_dict[pt_key]
      if pt_tensor.dtype == torch.bfloat16:
        pt_tensor = pt_tensor.float()
      np_array = pt_tensor.numpy()

      if "conv" in pt_key and "weight" in pt_key and len(np_array.shape) == 5:
          np_array = np_array.transpose(2, 3, 4, 1, 0)

      flax_key_tuple = tuple(flax_key_str.split('.'))
      if np_array.shape != flax_tensor.shape:
          print(f"Warning: Shape mismatch for {pt_key} - PT {np_array.shape} != JAX {flax_tensor.shape}")
          continue

      mapped_weights[flax_key_tuple] = jnp.array(np_array)
      processed_flax_keys.add(flax_key_str)

  for k in sorted(missing_keys):
    print(f"Warning: {k} not found in PyTorch state dict.")

  print(f"Mapped {len(mapped_weights)} parameters out of {len(flat_flax_params)} Flax keys.")
  params_nested = traverse_util.unflatten_dict(mapped_weights, sep='.')
  return params_nested


def main():
  data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ltx2_parity_data"))

  print("Initializing MaxDiffusion LTX-2 VAE...")
  model = LTX2VideoAutoencoderKL(
      in_channels=3,
      out_channels=3,
      latent_channels=128,
      block_out_channels=(256, 512, 1024, 2048),
      decoder_block_out_channels=(256, 512, 1024),
      layers_per_block=(4, 6, 6, 2, 2),
      decoder_layers_per_block=(5, 5, 5, 5),
      rngs=nnx.Rngs(0),
  )

  # Load converted weights
  pth_path = os.path.join(data_dir, "pytorch_model.bin")
  if not os.path.exists(pth_path):
    raise FileNotFoundError(f"PyTorch weights not found at {pth_path}. Run diffusers script first.")

  state_graph = load_and_convert_pytorch_weights(pth_path, model)
  nnx.update(model, state_graph)

  # Load inputs
  print("Loading PyTorch input...")
  pt_input = np.load(os.path.join(data_dir, "input.npy"))
  # PT Shape: (B, C, T, H, W) -> JAX Shape: (B, T, H, W, C)
  jax_input = jnp.transpose(pt_input, (0, 2, 3, 4, 1))

  print(f"\n--- Input ---")
  print(f"JAX Shape: {jax_input.shape}")
  print(f"Mean: {jax_input.mean():.6f}, Std: {jax_input.std():.6f}")

  print("\nRunning Encoder...")
  posterior = model.encode(jax_input).latent_dist
  jax_latents = posterior.mode()

  print(f"\n--- Encoder Latents ---")
  print(f"JAX Shape: {jax_latents.shape}")
  print(f"Mean: {jax_latents.mean():.6f}, Std: {jax_latents.std():.6f}")

  print("\nRunning Decoder...")
  # VAE decode output gives FlaxDecoderOutput with .sample
  jax_recon = model.decode(jax_latents).sample

  print(f"\n--- Decoder Output ---")
  print(f"JAX Shape: {jax_recon.shape}")
  print(f"Mean: {jax_recon.mean():.6f}, Std: {jax_recon.std():.6f}")

  # Compare with stored Diffusers outputs
  print("\n--- Parity Check ---")
  pt_latents = np.load(os.path.join(data_dir, "latents.npy"))
  pt_latents_transposed = np.transpose(pt_latents, (0, 2, 3, 4, 1))

  pt_recon = np.load(os.path.join(data_dir, "reconstruction.npy"))
  pt_recon_transposed = np.transpose(pt_recon, (0, 2, 3, 4, 1))

  latent_diff = np.abs(jax_latents - pt_latents_transposed)
  print(f"Max Latent Absolute Difference: {latent_diff.max():.8f}")

  recon_diff = np.abs(jax_recon - pt_recon_transposed)
  print(f"Max Reconstruction Absolute Difference: {recon_diff.max():.8f}")

  # --- Tiled Passes ---
  print("\nRunning Tiled Encoder/Decoder Passes...")
  # Load the 256x256 input specifically saved for spatial tiling
  pt_input_spatial = np.load(os.path.join(data_dir, "input_spatial.npy"))
  jax_input_spatial = jnp.transpose(pt_input_spatial, (0, 2, 3, 4, 1))

  model.tile_sample_min_height = 192
  model.tile_sample_min_width = 192
  model.tile_sample_stride_height = 128
  model.tile_sample_stride_width = 128
  model.tile_latent_min_height = 3
  model.tile_latent_min_width = 3
  model.tile_latent_stride_height = 2
  model.tile_latent_stride_width = 2
  model.enable_tiling()

  jax_latents_tiled = model.encode(jax_input_spatial).latent_dist.mode()
  jax_recon_tiled = model.decode(jax_latents_tiled).sample

  print(f"\n--- Tiled Encoder Latents ---")
  print(f"JAX Shape: {jax_latents_tiled.shape}")
  print(f"Mean: {jax_latents_tiled.mean():.6f}, Std: {jax_latents_tiled.std():.6f}")

  print(f"\n--- Tiled Decoder Output ---")
  print(f"JAX Shape: {jax_recon_tiled.shape}")
  print(f"Mean: {jax_recon_tiled.mean():.6f}, Std: {jax_recon_tiled.std():.6f}")

  pt_latents_tiled = np.load(os.path.join(data_dir, "latents_tiled.npy"))
  pt_latents_tiled_transposed = np.transpose(pt_latents_tiled, (0, 2, 3, 4, 1))
  pt_recon_tiled = np.load(os.path.join(data_dir, "reconstruction_tiled.npy"))
  pt_recon_tiled_transposed = np.transpose(pt_recon_tiled, (0, 2, 3, 4, 1))

  latent_diff_tiled = np.abs(jax_latents_tiled - pt_latents_tiled_transposed)
  print(f"Max Tiled Latent Absolute Difference: {latent_diff_tiled.max():.8f}")
  recon_diff_tiled = np.abs(jax_recon_tiled - pt_recon_tiled_transposed)
  print(f"Max Tiled Reconstruction Absolute Difference: {recon_diff_tiled.max():.8f}")

  model.use_tiling = False

  # --- Temporal Tiled Passes ---
  print("\nRunning Temporal Tiled Encoder/Decoder Passes...")
  pt_input_temporal = np.load(os.path.join(data_dir, "input_temporal.npy"))
  jax_input_temporal = jnp.transpose(pt_input_temporal, (0, 2, 3, 4, 1))

  model.tile_sample_min_num_frames = 32
  model.tile_sample_stride_num_frames = 16
  model.use_framewise_decoding = True

  # Disable implicit spatial tiling
  model.tile_sample_min_height = 10000
  model.tile_sample_min_width = 10000

  jax_latents_temporal_tiled = model.encode(jax_input_temporal).latent_dist.mode()
  jax_recon_temporal_tiled = model.decode(jax_latents_temporal_tiled).sample

  print(f"\n--- Temporal Tiled Encoder Latents ---")
  print(f"JAX Shape: {jax_latents_temporal_tiled.shape}")
  print(f"Mean: {jax_latents_temporal_tiled.mean():.6f}, Std: {jax_latents_temporal_tiled.std():.6f}")

  print(f"\n--- Temporal Tiled Decoder Output ---")
  print(f"JAX Shape: {jax_recon_temporal_tiled.shape}")
  print(f"Mean: {jax_recon_temporal_tiled.mean():.6f}, Std: {jax_recon_temporal_tiled.std():.6f}")

  pt_latents_temporal_tiled = np.load(os.path.join(data_dir, "latents_temporal_tiled.npy"))
  pt_latents_temporal_tiled_transposed = np.transpose(pt_latents_temporal_tiled, (0, 2, 3, 4, 1))
  pt_recon_temporal_tiled = np.load(os.path.join(data_dir, "reconstruction_temporal_tiled.npy"))
  pt_recon_temporal_tiled_transposed = np.transpose(pt_recon_temporal_tiled, (0, 2, 3, 4, 1))

  latent_diff_temporal = np.abs(jax_latents_temporal_tiled - pt_latents_temporal_tiled_transposed)
  print(f"Max Temporal Tiled Latent Absolute Difference: {latent_diff_temporal.max():.8f}")
  recon_diff_temporal = np.abs(jax_recon_temporal_tiled - pt_recon_temporal_tiled_transposed)
  print(f"Max Temporal Tiled Reconstruction Absolute Difference: {recon_diff_temporal.max():.8f}")

  print("Done!")


if __name__ == "__main__":
  main()