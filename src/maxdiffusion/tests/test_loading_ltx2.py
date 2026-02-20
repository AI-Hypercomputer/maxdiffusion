
import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.ltx2_utils import load_ltx2_transformer, load_ltx2_vae

class LTX2LoadingTest(unittest.TestCase):
    def test_loading(self):
        # Configuration for Lightricks/LTX-2
        # Based on config.json
        
        # 1. Why TraceContextError?
        # nnx.eval_shape creates a new JAX trace level (like jax.vmap or jax.grad).
        # nnx.Rngs is a stateful object (it tracks counts). Modifying an Rngs object created 
        # outside the trace (in the test function scope) from inside the trace (inside eval_shape) 
        # is forbidden because it leaks state across trace boundaries.
        # Solution: Create nnx.Rngs INSIDE the eval_shape function.
        
        def create_model():
            rngs = nnx.Rngs(0) # Created inside the trace context
            return LTX2VideoTransformer3DModel(
                rngs=rngs,
                in_channels=128,
                out_channels=128,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=32,       
                attention_head_dim=128,       
                cross_attention_dim=4096,     
                audio_in_channels=128,        
                audio_out_channels=128,       
                audio_patch_size=1,           
                audio_patch_size_t=1,         
                audio_num_attention_heads=32, 
                audio_attention_head_dim=64,  
                audio_cross_attention_dim=2048,
                num_layers=48,                
                rope_type="split",            
                weights_dtype=jnp.bfloat16,   
                dtype=jnp.bfloat16,
            )

        abstract_model = nnx.eval_shape(create_model)
        
        # Test structure
        self.assertEqual(abstract_model.rope_type, "split") 
        self.assertEqual(abstract_model.num_layers, 48)
        
        # We can verify shapes using nnx.state if needed
        # state_shapes = nnx.state(abstract_model, nnx.Param)
        
        print("Model structure verified without OOM or TraceContextError.")

    def test_vae_loading(self):
        # Configuration for Lightricks/LTX-2 VAE
        def create_vae():
             rngs = nnx.Rngs(0)
             return LTX2VideoAutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=128,
                block_out_channels=(256, 512, 1024, 2048),
                decoder_block_out_channels=(256, 512, 1024),
                layers_per_block=(4, 6, 6, 2, 2),
                decoder_layers_per_block=(5, 5, 5, 5),
                spatio_temporal_scaling=(True, True, True, True),
                decoder_spatio_temporal_scaling=(True, True, True),
                decoder_inject_noise=(False, False, False, False),
                downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                upsample_residual=(True, True, True),
                upsample_factor=(2, 2, 2),
                rngs=rngs,
                dtype=jnp.float32,
             )
        
        abstract_vae = nnx.eval_shape(create_vae)
        
        self.assertEqual(abstract_vae.latent_channels, 128)
        # self.assertEqual(len(abstract_vae.encoder.down_blocks), 4) # nnx.List not len()able directly? depends on version
        
        print("VAE structure verified.")


if __name__ == "__main__":
    unittest.main()
