
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
from maxdiffusion.models.ltx_2.transformer_ltx2 import LTX2VideoTransformerBlock, LTX2VideoTransformer3DModel

class LTX2TransformerTest(unittest.TestCase):
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
        self.batch_size = 1 # Use 1 for determinism in unit tests often easier
        self.num_frames = 4
        self.height = 32
        self.width = 32
        self.patch_size = 1
        self.patch_size_t = 1
        
        # Dimensions
        self.dim = 32
        self.num_ids = 6 # rope
        self.in_channels = 8
        self.out_channels = 8
        self.audio_in_channels = 4
        
        # Derived
        self.seq_len = (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)


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
        
        dim = 32
        audio_dim = 16
        cross_dim = 20 # context dim
        
        # NNX sharding context
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            block = LTX2VideoTransformerBlock(
                rngs=self.rngs,
                dim=dim,
                num_attention_heads=4,
                attention_head_dim=8,
                cross_attention_dim=cross_dim,
                audio_dim=audio_dim,
                audio_num_attention_heads=4,
                audio_attention_head_dim=4,
                audio_cross_attention_dim=cross_dim,
                activation_fn="gelu",
                qk_norm="rms_norm_across_heads",
            )
            
            # Create dummy inputs
            hidden_states = jnp.zeros((self.batch_size, self.seq_len, dim))
            audio_hidden_states = jnp.zeros((self.batch_size, 10, audio_dim)) # 10 audio frames
            encoder_hidden_states = jnp.zeros((self.batch_size, 5, cross_dim))
            audio_encoder_hidden_states = jnp.zeros((self.batch_size, 5, cross_dim)) # reusing cross_dim for audio context 
            
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
            te_dim = num_ada_params * dim # simplified assumption for test
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
                video_rotary_emb=None, # Dummy takes None
                audio_rotary_emb=None
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
                num_attention_heads=2,
                attention_head_dim=8,
                num_layers=1, # 1 layer for speed
                caption_channels=32, # small for test
                cross_attention_dim=32,
                audio_in_channels=self.audio_in_channels,
                audio_out_channels= self.audio_in_channels,
                audio_num_attention_heads=2,
                audio_attention_head_dim=8,
                audio_cross_attention_dim=32
            )
        
        # Inputs
        # hidden_states: (B, F, H, W, C) or (B, L, C)?
        # Diffusers `forward` takes `hidden_states` usually as latents.
        # If it's 3D, it might expect (B, C, F, H, W) or (B, F, C, H, W)?
        # Checking `transformer_ltx2.py` `__call__` Line 680:
        # `hidden_states = self.proj_in(hidden_states)`
        # `proj_in` is nnx.Linear.
        # This implies `hidden_states` input is ALREADY flattened/sequenced or `proj_in` assumes channel-last inputs.
        # If `proj_in` is Linear, input must be compatible with matrix mult.
        # Usually Transformers expect (B, L, D) or (B, N, D).
        # But `prepare_video_coords` logic suggests it handles spatial awareness.
        # The PROMPT usually implies `latents` of shape (B, C, F, H, W).
        # BUT `nnx.Linear` (Dense) applies to the last dimension.
        # If input is (B, C, F, H, W), Linear would act on W. That's wrong.
        # Diffusers LTX usually patchifies EXTERNALLY or has a conv patch embed?
        # In my definition (Line 491): `self.proj_in = nnx.Linear(...)`.
        # This differs from Conv3d.
        # This implies the user MUST pass flattened tokens?
        # Re-checking Diffusers implementation...
        # If `LTX2VideoTransformer3DModel` in Diffusers uses `patch_embed` (Conv), it takes 5D.
        # Verify `transformer_ltx2.py` user edits...
        # Step 426 (Original) had `nnx.Conv`.
        # Step 491 (New) has `nnx.Linear`.
        # This suggests input is EXPECTED to be flattened/patchified already OR raw channel-last (B, ..., C).
        # IMPORTANT: if `proj_in` is Linear, we pass (B, L, C).
        
        # Let's pass (B, L, C).
        hidden_states = jnp.zeros((self.batch_size, self.seq_len, self.in_channels))
        audio_hidden_states = jnp.zeros((self.batch_size, 10, self.audio_in_channels))
        
        timestep = jnp.array([1.0]) # (B,)
        
        encoder_hidden_states = jnp.zeros((self.batch_size, 5, 32)) # (B, Lc, Dc)
        audio_encoder_hidden_states = jnp.zeros((self.batch_size, 5, 32))
        
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
                audio_num_frames=10,
                fps=24.0,
                return_dict=True
            )
        
        sample = output["sample"]
        audio_sample = output["audio_sample"]
        
        print(f"Model Output Video Shape: {sample.shape}")
        print(f"Model Output Audio Shape: {audio_sample.shape}")
        
        self.assertEqual(sample.shape, (self.batch_size, self.seq_len, self.out_channels))
        self.assertEqual(audio_sample.shape, (self.batch_size, 10, self.audio_in_channels))

if __name__ == "__main__":
    unittest.main()
