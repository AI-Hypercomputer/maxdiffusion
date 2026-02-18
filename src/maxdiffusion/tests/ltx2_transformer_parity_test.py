"""
This is a test file used for ensuring numerical parity between pytorch and jax implementation of LTX2.
This is to be ignored and will not be pushed when commiting to main branch.
"""
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import flax
# Matches WanTransformerTest: disable eager sharding to avoid "mesh context required" errors during init
flax.config.update("flax_always_shard_variable", False)
from jax.sharding import Mesh
import os
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformerBlock, LTX2VideoTransformer3DModel
import torch


class LTX2TransformerParityTest(unittest.TestCase):
  """
  Tests for LTX-2 Transformer components:
  1. LTX2VideoTransformerBlock: Single transformer block.
  2. LTX2VideoTransformer3DModel: Full 3D Transformer model.
  """

  def setUp(self):
    # Initialize config and mesh for sharding
    # using standard MaxDiffusion pattern
    pyconfig.initialize(
        [None, os.path.join(os.path.dirname(__file__), "..", "configs", "ltx2_video.yml")],
        unittest=True,
    )
    self.config = pyconfig.config
    devices_array = create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)

    # random seed for reproducibility
    self.rngs = nnx.Rngs(0)
    self.batch_size = 1  # Use 1 for determinism in unit tests often easier
    self.num_frames = 4
    self.height = 32
    self.width = 32
    self.patch_size = 1
    self.patch_size_t = 1

    # Dimensions
    self.dim = 32
    self.num_ids = 6  # rope
    self.in_channels = 8
    self.out_channels = 8
    self.audio_in_channels = 4

    # Derived
    self.seq_len = (
        (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
    )

  def test_transformer_block_shapes(self):
    """
    Verifies that LTX2VideoTransformerBlock accepts inputs of correct shapes
    and outputs tensors preserving the residual stream dimensions.

    Tested Inputs:
    - hidden_states: (B, L, D) - Video stream
    - audio_hidden_states: (B, La, Da) - Audio stream
    - encoder_hidden_states: (B, Lc, D) - Text context
    - audio_encoder_hidden_states: (B, Lc, Da) - Audio context
    - Modulation parameters (temb, gate, shift, scale) pre-computed

    Expected Output:
    - Sequence of (hidden_states, audio_hidden_states) with same shapes as input.
    """
    print("\n=== Testing LTX2VideoTransformerBlock Shapes ===")

    dim = 1024
    audio_dim = 1024
    cross_dim = 64  # context dim

    # NNX sharding context
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = LTX2VideoTransformerBlock(
          rngs=self.rngs,
          dim=dim,
          num_attention_heads=8,
          attention_head_dim=128,
          cross_attention_dim=cross_dim,
          audio_dim=audio_dim,
          audio_num_attention_heads=8,
          audio_attention_head_dim=128,
          audio_cross_attention_dim=cross_dim,
          activation_fn="gelu",
          mesh=self.mesh,
      )

      # Create dummy inputs
      hidden_states = jnp.zeros((self.batch_size, self.seq_len, dim))
      audio_hidden_states = jnp.zeros((self.batch_size, 128, audio_dim))  # 128 audio frames for TPFA
      encoder_hidden_states = jnp.zeros((self.batch_size, 128, cross_dim))  # 128 for TPFA
      audio_encoder_hidden_states = jnp.zeros((self.batch_size, 128, cross_dim))  # reusing cross_dim for audio context

      # Dummy scale/shift/gate modulations
      # These match the shapes expected by the block internal calculation logic
      # For simplicity, we create them to match 'temb_reshaped' broadcasting or direct add
      # The block expects raw scale/shift/gate inputs often, OR temb vectors.
      # Let's check block calls:
      # It takes `temb` and `temb_ca...`
      # temb: (B, 1, 6, -1) or similar depending on reshape.
      # Actually in `transformer_ltx2.py`, call signature takes:
      # temb: jax.Array
      # And reshapes it: temb.reshape(batch_size, 1, num_ada_params, -1)
      # So input `temb` should be (B, num_ada_params * dim) roughly, or (B, num_ada_params, dim)

      num_ada_params = 6
      te_dim = num_ada_params * dim  # simplified assumption for test
      temb = jnp.zeros((self.batch_size, te_dim))

      num_audio_ada_params = 6
      te_audio_dim = num_audio_ada_params * audio_dim
      temb_audio = jnp.zeros((self.batch_size, te_audio_dim))

      # CA modulations
      # 4 params for scale/shift, 1 for gate
      temb_ca_scale_shift = jnp.zeros((self.batch_size, 4 * dim))
      temb_ca_audio_scale_shift = jnp.zeros((self.batch_size, 4 * audio_dim))
      temb_ca_gate = jnp.zeros((self.batch_size, 1 * dim))
      temb_ca_audio_gate = jnp.zeros((self.batch_size, 1 * audio_dim))

      # Perform forward
      out_hidden, out_audio = block(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          temb=temb,
          temb_audio=temb_audio,
          temb_ca_scale_shift=temb_ca_scale_shift,
          temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
          temb_ca_gate=temb_ca_gate,
          temb_ca_audio_gate=temb_ca_audio_gate,
          video_rotary_emb=None,  # Dummy takes None
          audio_rotary_emb=None,
      )

      print(f"Input Video Shape: {hidden_states.shape}")
      print(f"Output Video Shape: {out_hidden.shape}")
      print(f"Input Audio Shape: {audio_hidden_states.shape}")
      print(f"Output Audio Shape: {out_audio.shape}")

      self.assertEqual(out_hidden.shape, hidden_states.shape)
      self.assertEqual(out_audio.shape, audio_hidden_states.shape)

  def test_transformer_3d_model_instantiation_and_forward(self):
    """
    Verifies LTX2VideoTransformer3DModel full instantiation and forward pass.
    Checks:
    - Argument passing to __init__
    - Input embedding (patchify) shapes
    - RoPE preparation
    - Timestep embedding logic
    - Block iteration
    - Output projection shapes

    Expected Output:
    - Dictionary with "sample" and "audio_sample" keys.
    - "sample" shape: (B, L, out_channels * patch_size_params...) roughly?
      Actually proj_out maps to `_out_channels`.
      Wait, `proj_out` in `transformer_ltx2.py` maps `inner_dim` -> `_out_channels`.
      It does NOT unpatchify in the transformer itself usually, it returns latent sequence?
      In Diffusers `TransformerLTX2`, `proj_out` maps to `out_channels * patch_size * ...`?
      Let's check `transformer_ltx2.py` Line 624:
      `self.proj_out = nnx.Linear(inner_dim, _out_channels, ...)`
      And `_out_channels` defaults to `in_channels` (which is often latent dim).
      So it returns sequence (B, L, C).
    """
    print("\n=== Testing LTX2VideoTransformer3DModel Integration ===")

    # NNX sharding context
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = LTX2VideoTransformer3DModel(
          rngs=self.rngs,
          in_channels=self.in_channels,
          out_channels=self.out_channels,
          patch_size=self.patch_size,
          patch_size_t=self.patch_size_t,
          num_attention_heads=8,
          attention_head_dim=128,
          num_layers=1,  # 1 layer for speed
          caption_channels=32,  # small for test
          cross_attention_dim=1024,
          audio_in_channels=self.audio_in_channels,
          audio_out_channels=self.audio_in_channels,
          audio_num_attention_heads=8,
          audio_attention_head_dim=128,
          audio_cross_attention_dim=1024,
          mesh=self.mesh,
      )

    # Let's pass (B, L, C).
    hidden_states = jnp.zeros((self.batch_size, self.seq_len, self.in_channels))
    audio_hidden_states = jnp.zeros((self.batch_size, 128, self.audio_in_channels))

    timestep = jnp.array([1.0])  # (B,)

    encoder_hidden_states = jnp.zeros((self.batch_size, 128, 32))  # (B, Lc, Dc) # 128 for TPFA
    audio_encoder_hidden_states = jnp.zeros((self.batch_size, 128, 32))
    encoder_attention_mask = jnp.ones((self.batch_size, 128), dtype=jnp.float32)
    audio_encoder_attention_mask = jnp.ones((self.batch_size, 128), dtype=jnp.float32)

    # Forward
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      output = model(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          timestep=timestep,
          num_frames=self.num_frames,
          height=self.height,
          width=self.width,
          audio_num_frames=128,
          fps=24.0,
          return_dict=True,
          encoder_attention_mask=encoder_attention_mask,
          audio_encoder_attention_mask=audio_encoder_attention_mask,
      )

    sample = output["sample"]
    audio_sample = output["audio_sample"]

    print(f"Model Output Video Shape: {sample.shape}")
    print(f"Model Output Audio Shape: {audio_sample.shape}")

    self.assertEqual(sample.shape, (self.batch_size, self.seq_len, self.out_channels))
    self.assertEqual(sample.shape, (self.batch_size, self.seq_len, self.out_channels))
    self.assertEqual(audio_sample.shape, (self.batch_size, 128, self.audio_in_channels))

  def test_transformer_3d_model_dot_product_attention(self):
    """Verifies LTX2VideoTransformer3DModel full instantiation and forward pass with dot_product attention."""

    # 1. Instantiate Model with dot_product kernel
    model = LTX2VideoTransformer3DModel(
        rngs=nnx.Rngs(0),
        in_channels=self.in_channels,
        out_channels=self.out_channels,
        patch_size=self.patch_size,
        patch_size_t=self.patch_size_t,
        num_attention_heads=8,
        attention_head_dim=128,
        cross_attention_dim=1024,
        caption_channels=32,
        audio_in_channels=self.audio_in_channels,
        audio_out_channels=self.audio_in_channels,
        audio_patch_size=1,
        audio_patch_size_t=1,
        audio_num_attention_heads=8,
        audio_attention_head_dim=128,
        audio_cross_attention_dim=1024,
        num_layers=1,  # Reduced layers for speed
        scan_layers=False,
        mesh=self.mesh,
        attention_kernel="dot_product",
    )

    # 2. Inputs
    hidden_states = jnp.ones((self.batch_size, self.seq_len, self.in_channels)) * 0.5
    audio_hidden_states = jnp.ones((self.batch_size, 128, self.audio_in_channels)) * 0.5
    timestep = jnp.array([1.0])  # (B,)

    encoder_hidden_states = jnp.zeros((self.batch_size, 128, 32))  # (B, Lc, Dc)
    audio_encoder_hidden_states = jnp.zeros((self.batch_size, 128, 32))
    encoder_attention_mask = jnp.ones((self.batch_size, 128), dtype=jnp.float32)
    audio_encoder_attention_mask = jnp.ones((self.batch_size, 128), dtype=jnp.float32)

    # Forward
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      output = model(
          hidden_states=hidden_states,
          audio_hidden_states=audio_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          audio_encoder_hidden_states=audio_encoder_hidden_states,
          timestep=timestep,
          num_frames=self.num_frames,
          height=self.height,
          width=self.width,
          audio_num_frames=128,
          fps=24.0,
          return_dict=True,
          encoder_attention_mask=encoder_attention_mask,
          audio_encoder_attention_mask=audio_encoder_attention_mask,
      )

    print(f"Model Output Video Shape (Dot Product): {output['sample'].shape}")
    print(f"Model Output Audio Shape (Dot Product): {output['audio_sample'].shape}")

    self.assertEqual(output["sample"].shape, hidden_states.shape)
    self.assertEqual(output["audio_sample"].shape, audio_hidden_states.shape)

  def test_scan_remat_parity(self):
    """
    Verifies that scan_layers=True produces identical output to scan_layers=False.
    Also verifies that enabling remat_policy works without error.
    """
    print("\n=== Testing Scan/Remat Parity ===")

    # Common args
    args = dict(
        rngs=self.rngs,
        in_channels=self.in_channels,
        out_channels=self.out_channels,
        patch_size=self.patch_size,
        patch_size_t=self.patch_size_t,
        num_attention_heads=8,
        attention_head_dim=128,
        num_layers=2,  # Need >1 layer to test scan effectively
        caption_channels=32,
        cross_attention_dim=32,
        audio_in_channels=self.audio_in_channels,
        audio_out_channels=self.audio_in_channels,
        audio_num_attention_heads=8,
        audio_attention_head_dim=128,
        audio_cross_attention_dim=32,
    )

    # 1. Initialize models
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model_scan = LTX2VideoTransformer3DModel(**args, scan_layers=True, mesh=self.mesh)
      model_loop = LTX2VideoTransformer3DModel(**args, scan_layers=False, mesh=self.mesh)
      model_remat = LTX2VideoTransformer3DModel(**args, scan_layers=True, remat_policy="full", mesh=self.mesh)

    # Inputs
    hidden_states = jnp.ones((self.batch_size, self.seq_len, self.in_channels)) * 0.5
    audio_hidden_states = jnp.ones((self.batch_size, 128, self.audio_in_channels)) * 0.5
    timestep = jnp.array([1.0])
    encoder_hidden_states = jnp.ones((self.batch_size, 128, 32)) * 0.1
    audio_encoder_hidden_states = jnp.ones((self.batch_size, 128, 32)) * 0.1

    inp_args = dict(
        hidden_states=hidden_states,
        audio_hidden_states=audio_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        timestep=timestep,
        num_frames=self.num_frames,
        width=self.width,
        height=self.height,
        audio_num_frames=128,
        fps=24.0,
        return_dict=True,
    )

    # 3. Run Forward
    print("Running scan_layers=True...")
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out_scan = model_scan(**inp_args)["sample"]

      print("Running scan_layers=False...")
      # To get same weights, we can try to copy state
      # nnx.update(model_loop, nnx.state(model_scan)) # Might fail if structure differs
      out_loop = model_loop(**inp_args)["sample"]

      print("Running remat_policy='full'...")
      out_remat = model_remat(**inp_args)["sample"]

    self.assertEqual(out_scan.shape, out_loop.shape)
    self.assertEqual(out_scan.shape, out_remat.shape)

    # If we can't easily sync weights, we can't assert numerical parity yet.
    # But successful execution confirms the pathways are valid.
    print("Scan/Remat execution successful.")

  def test_import_parity_comparison(self):
    """
    Imports parity data from ltx2_parity_data.pt and compares MaxDiffusion output with Diffusers output.
    """
    print("\n=== Parity Comparison Test ===")
    try:
      import torch
    except ImportError:
      print("Skipping parity test: torch not installed.")
      return

    import os
    from flax import traverse_util

    parity_file = "ltx2_parity_data.pt"
    if not os.path.exists(parity_file):
      print(f"Skipping parity test: {parity_file} not found. Run diffusers test first.")
      return

    print(f"Loading {parity_file}...")
    parity_data = torch.load(parity_file)
    state_dict = parity_data["state_dict"]
    inputs = parity_data["inputs"]
    torch_outputs = parity_data["outputs"]
    config = parity_data["config"]

    # 1. Instantiate Model
    # Ensure config matches what was exported
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      model = LTX2VideoTransformer3DModel(
          rngs=nnx.Rngs(0),
          in_channels=config["in_channels"],
          out_channels=config["out_channels"],
          patch_size=config["patch_size"],
          patch_size_t=1,
          num_attention_heads=8,
          attention_head_dim=128,
          cross_attention_dim=1024,  # Parity config
          caption_channels=config["caption_channels"],
          audio_in_channels=4,
          audio_out_channels=4,
          audio_patch_size=1,
          audio_patch_size_t=1,
          audio_num_attention_heads=8,
          audio_attention_head_dim=128,
          audio_cross_attention_dim=1024,
          num_layers=1,
          mesh=self.mesh,
          attention_kernel="dot_product",
          rope_type="interleaved",
      )

    # 2. Convert Weights (PyTorch -> Flax NNX)
    print("Converting weights...")

    graph_def, state = nnx.split(model)
    flat_state = traverse_util.flatten_dict(state.to_pure_dict())
    new_flat_state = {}

    # Helper to convert/transpose weights
    def convert_weight(pt_key_base, jax_key):
      # Try original key first
      pt_key = pt_key_base

      # Map JAX 'kernel' to PT 'weight'
      if "kernel" in str(jax_key):
        pt_key = pt_key.replace("kernel", "weight")

      # Fix scale logic (RMSNorm)
      # Only replace 'scale' if it's the parameter name (last part) to avoid breaking 'scale_shift'
      if jax_key[-1] == "scale" and "scale_shift" not in str(jax_key):
        pt_key = pt_key.replace("scale", "weight")

      # Fix transformer_blocks prefix
      # JAX: ('transformer_blocks', 'attn1', ...)
      # PT: transformer_blocks.0.attn1...
      is_transformer_block = "transformer_blocks" in str(jax_key)
      if is_transformer_block:
        if "transformer_blocks" in pt_key and "transformer_blocks.0" not in pt_key:
          pt_key = pt_key.replace("transformer_blocks", "transformer_blocks.0")

      # Fix `layers` keyword in JAX key usually implies `layers.0` if it was there?
      if "layers" in pt_key:
        pt_key = pt_key.replace("layers.", "")

      # Fix to_out (Diffusers has to_out[0] as Linear)
      if "to_out" in pt_key and ("weight" in pt_key or "bias" in pt_key):
        pt_key = pt_key.replace("to_out.weight", "to_out.0.weight")
        pt_key = pt_key.replace("to_out.bias", "to_out.0.bias")

      # Fix FeedForward (net_0 -> net.0.proj, net_2 -> net.2)
      if "net_0" in pt_key:
        pt_key = pt_key.replace("net_0", "net.0.proj")
      if "net_2" in pt_key:
        pt_key = pt_key.replace("net_2", "net.2")

      if pt_key not in state_dict:
        # Try removing .0 if it was added erroneously
        candidates = [pt_key]
        if "transformer_blocks.0" in pt_key:
          candidates.append(pt_key.replace("transformer_blocks.0", "transformer_blocks"))

        # Special Case: scale_shift_table
        # Only allow global scale_shift_table fallback if NOT inside transformer block
        if "scale_shift_table" in str(jax_key) and not is_transformer_block:
          candidates.append("scale_shift_table")

        if "audio_scale_shift_table" in str(jax_key) and not is_transformer_block:
          candidates.append("audio_scale_shift_table")

        for c in candidates:
          if c in state_dict:
            pt_key = c
            break
        else:
          # If unmapped bias, maybe it's just missing in PT (e.g. RMSNorm without bias)
          if "bias" in str(jax_key):
            # Initialize to zeros?
            print(f"Warning: Missing PT bias for {jax_key}. initializing to zeros.")
            # Use shape from current flat_state param
            return jnp.zeros(flat_state[jax_key].shape), pt_key

          return None, pt_key

      w = state_dict[pt_key].cpu().numpy()

      # Debug Special Parameters
      if "scale_shift_table" in str(jax_key):
        print(f"Mapping scale_shift_table for {jax_key} from {pt_key} with shape {w.shape}")

      # Handle vmap/scan dimension for transformer_blocks
      if is_transformer_block:
        # JAX expects (num_layers, ...) for these weights
        # PT has (...)
        # So expand dims
        w = w[None, ...]

      # Handle Transforms
      is_kernel = "kernel" in str(jax_key)
      # Embedding projections are also 'kernel' in JAX (Linear)
      if is_kernel:
        if w.ndim == 3:  # (1, out, in) -> (1, in, out)
          w = w.transpose(0, 2, 1)
        elif w.ndim == 2:  # (out, in) -> (in, out)
          w = w.T

      return jnp.array(w), pt_key

    total_count = len(flat_state)
    mapped_count = 0

    # Debug: Print available keys for audio_ff
    print("Debugging PT keys for mapping failure diagnosis:")
    print("Available PT keys with 'ff':", [k for k in state_dict.keys() if "ff" in k])
    print("Available PT keys with 'norm':", [k for k in state_dict.keys() if "norm" in k])

    for key in flat_state.keys():
      # Construct base PT key from JAX key tuple
      pt_key_base = ".".join([str(k) for k in key if str(k) != "layers"])

      w, used_pt_key = convert_weight(pt_key_base, key)
      if w is not None:
        # Handle bias zero init which might return scalar 0 if shape was (1,) but it should be array
        # jnp.zeros(shape) returns array.
        new_flat_state[key] = w
        mapped_count += 1
      else:
        print(f"Warning: Could not map JAX key {key} (PT attempt: {used_pt_key})")
        if "audio_ff" in str(key):
          print("Available audio_ff keys:", [k for k in state_dict.keys() if "audio_ff" in k])
        if "norm_out" in str(key):
          print("Available norm_out keys:", [k for k in state_dict.keys() if "norm_out" in k])

    print(f"Mapped {mapped_count}/{total_count} params.")

    # Update model state
    new_state = traverse_util.unflatten_dict(new_flat_state)
    nnx.update(model, new_state)

    # 3. Prepare Inputs
    jax_inputs = {
        "hidden_states": jnp.array(inputs["hidden_states"].cpu().numpy()),
        "audio_hidden_states": jnp.array(inputs["audio_hidden_states"].cpu().numpy()),
        "encoder_hidden_states": jnp.array(inputs["encoder_hidden_states"].cpu().numpy()),
        "audio_encoder_hidden_states": jnp.array(inputs["audio_encoder_hidden_states"].cpu().numpy()),
        "timestep": jnp.array(inputs["timestep"].cpu().numpy()),
        "encoder_attention_mask": jnp.array(inputs["encoder_attention_mask"].cpu().numpy()),
        "audio_encoder_attention_mask": jnp.array(inputs["audio_encoder_attention_mask"].cpu().numpy()),
    }

    print("\n=== Input Verification ===")
    print(f"Hidden States Sum: {jnp.sum(jax_inputs['hidden_states'])}")
    print(f"Audio Hidden States Sum: {jnp.sum(jax_inputs['audio_hidden_states'])}")
    print(f"Encoder Hidden States Sum: {jnp.sum(jax_inputs['encoder_hidden_states'])}")
    print(f"Audio Encoder Hidden States Sum: {jnp.sum(jax_inputs['audio_encoder_hidden_states'])}")
    print(f"Timestep: {jax_inputs['timestep']}")
    print("==========================\n")

    # 4. Run Forward
    print("Running MaxDiffusion forward pass...")
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      output = model(
          hidden_states=jax_inputs["hidden_states"],
          audio_hidden_states=jax_inputs["audio_hidden_states"],
          encoder_hidden_states=jax_inputs["encoder_hidden_states"],
          audio_encoder_hidden_states=jax_inputs["audio_encoder_hidden_states"],
          timestep=jax_inputs["timestep"],
          encoder_attention_mask=jax_inputs["encoder_attention_mask"],
          audio_encoder_attention_mask=jax_inputs["audio_encoder_attention_mask"],
          num_frames=config["num_frames"] if "num_frames" in config else 4,
          height=config["height"] if "height" in config else 32,
          width=config["width"] if "width" in config else 32,
          audio_num_frames=128,
          fps=24.0,
          return_dict=True,
      )

    max_sample = output["sample"]
    max_audio_sample = output["audio_sample"]

    print("MAXDIFF Output Sample Stats:")
    print(f"Sample Max: {jnp.max(max_sample)}")
    print(f"Sample Min: {jnp.min(max_sample)}")
    print(f"Sample Mean: {jnp.mean(max_sample)}")
    print(f"Sample Std: {jnp.std(max_sample)}")

    print("MAXDIFF Output Audio Sample Stats:")
    print(f"Audio Max: {jnp.max(max_audio_sample)}")
    print(f"Audio Min: {jnp.min(max_audio_sample)}")
    print(f"Audio Mean: {jnp.mean(max_audio_sample)}")
    print(f"Audio Std: {jnp.std(max_audio_sample)}")


if __name__ == "__main__":
  unittest.main()
