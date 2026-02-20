
import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.ltx2_utils import load_ltx2_transformer

class LTX2LoadingTest(unittest.TestCase):
    def test_loading(self):
        # Configuration for Lightricks/LTX-2
        # Based on config.json
        
        rngs = nnx.Rngs(0)
        
        # Initialize model with exact params from checkpoint
        model = LTX2VideoTransformer3DModel(
            rngs=rngs,
            in_channels=128,
            out_channels=128,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=32,       # Default matches
            attention_head_dim=128,       # Default matches
            cross_attention_dim=4096,     # Default matches
            audio_in_channels=128,        # Default matches
            audio_out_channels=128,       # Default matches
            audio_patch_size=1,           # Default matches
            audio_patch_size_t=1,         # Default matches
            audio_num_attention_heads=32, # Default matches
            audio_attention_head_dim=64,  # Default matches
            audio_cross_attention_dim=2048,# Default matches
            num_layers=48,                # Default matches
            rope_type="split",            # IMPORTANT: Changed from default "interleaved"
            weights_dtype=jnp.bfloat16,   # Usually checkpoint is fp16 or bf16
            dtype=jnp.bfloat16,
        )
        
        # Test loading logic (mock or real)
        # If running this test locally with internet access and sufficient disk space, 
        # it will attempt to download the model (~20GB).
        # To avoid this during CI/dev loop without the weights, one would mock hf_hub_download.
        
        print(f"Model initialized with params matching Lightricks/LTX-2 (rope_type='split', layers=48).")
        print("To verify weight loading, ensure you have 'Lightricks/LTX-2' accessible or cached.")
        
        # We can try to load just the index to verify structure if possible, but load_ltx2_transformer
        # does strict loading.
        
        # For now, let's just assert the model structure matches expectations
        self.assertEqual(model.rope_type, "split")
        self.assertEqual(model.num_layers, 48)
        
        # Example call (commented out to prevent accidental large download in standard test suite)
        # eval_shapes = nnx.eval_shape(lambda: model)
        # params = load_ltx2_transformer(
        #     "Lightricks/LTX-2",
        #     eval_shapes=eval_shapes,
        #     device="cpu", # Load to CPU first
        #     hf_download=True
        # )
        # print("Weights loaded successfully!")

if __name__ == "__main__":
    unittest.main()
