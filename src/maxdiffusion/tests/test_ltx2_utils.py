
import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.ltx2_utils import load_transformer_weights, load_vae_weights
from maxdiffusion.models.modeling_flax_pytorch_utils import validate_flax_state_dict
from flax.traverse_util import flatten_dict

class LTX2UtilsTest(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.rngs = nnx.Rngs(42)
        
    def test_load_transformer_weights(self):
        # Configuration matching Lightricks/LTX-2
        # Defaults in LTX2VideoTransformer3DModel match 2.0 version (48 layers)
        
        # Using eval_shape to avoid OOM on test environment if possible, 
        # but load_transformer_weights returns real weights.
        # We'll rely on the fact that if it runs, it loads.
        
        # Note: This test downloads ~20GB if not cached. 
        # In a real CI, we might mock this. But for this specific user request, we run it.
        
        pretrained_model_name_or_path = "Lightricks/LTX-2"
        
        with jax.default_device(jax.devices("cpu")[0]):
             model = LTX2VideoTransformer3DModel(
                rngs=self.rngs,
                # Explicitly setting key params to version 2.0 to be safe
                in_channels=128,
                out_channels=128,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=32,
                attention_head_dim=128,
                cross_attention_dim=4096,
                num_layers=48,
                scan_layers=True 
             )
        
        # Get abstract state (shapes only)
        # We need the PyTree structure of parameters
        # nnx.state(model) gives the State object
        
        # We need meaningful shapes. 
        # model.init is NOT provided by standard nnx.Module the same way as linen? 
        # NNX initializes aggressively in __init__ usually if rngs provided.
        # So `nnx.state(model)` should have the params.
        
        # abstract_state = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nnx.state(model))
        # But wait, validate_flax_state_dict expects a dict of params, usually just the params subtree.
        
        # We can extract params from state
        state = nnx.state(model)
        # Filter for params? 
        # Usually validate_flax_state_dict expects the full PyTree or a specific dict.
        # state is a State object, acts like a Mapping
        
        # In `ltx2_utils.py`, we construct a flax_state_dict that mirrors this.
        # We should pass `state` or `state.to_pure_dict()` as `eval_shapes`.
        
        eval_shapes = state.to_pure_dict()
        
        print("Loading Transformer Weights...")
        loaded_weights = load_transformer_weights(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            eval_shapes=eval_shapes,
            device=self.device,
            hf_download=True,
            num_layers=48,
            scan_layers=True
        )
        
        print("Validating Transformer Weights...")
        validate_flax_state_dict(eval_shapes, loaded_weights)
        print("Transformer Weights Validated Successfully!")

    def test_load_vae_weights(self):
        pretrained_model_name_or_path = "Lightricks/LTX-2"
        
        with jax.default_device(jax.devices("cpu")[0]):
             model = LTX2VideoAutoencoderKL(
                rngs=self.rngs,
                # Defaults:
                in_channels=3,
                out_channels=3,
                latent_channels=128,
                block_out_channels=(256, 512, 1024, 2048),
                layers_per_block=(4, 6, 6, 2, 2), # Matches 2.0
                upsample_factor=(2, 2, 2)
             )
             
        state = nnx.state(model)
        eval_shapes = state.to_pure_dict()
        
        print("Loading VAE Weights...")
        loaded_weights = load_vae_weights(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            eval_shapes=eval_shapes,
            device=self.device,
            hf_download=True
        )
        
        print("Validating VAE Weights...")
        validate_flax_state_dict(eval_shapes, loaded_weights)
        print("VAE Weights Validated Successfully!")

if __name__ == "__main__":
    unittest.main()
