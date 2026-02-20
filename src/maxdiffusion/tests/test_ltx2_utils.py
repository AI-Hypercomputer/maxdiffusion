
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
            self.config = LTX2VideoConfig()
            self.config.audio_attention_head_dim = 128 # Match Checkpoint
            
            self.transformer = LTX2VideoTransformer3DModel(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                patch_size=self.config.patch_size,
                patch_size_t=self.config.patch_size_t,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                cross_attention_dim=self.config.cross_attention_dim,
                audio_in_channels=self.config.audio_in_channels,
                audio_out_channels=self.config.audio_out_channels,
                audio_patch_size=self.config.audio_patch_size,
                audio_patch_size_t=self.config.audio_patch_size_t,
                audio_num_attention_heads=self.config.audio_num_attention_heads,
                audio_attention_head_dim=128, # Match Config/Checkpoint
                audio_cross_attention_dim=self.config.audio_cross_attention_dim,
                num_layers=self.config.num_layers,
                scan_layers=True,
                param_dtype=jnp.bfloat16,
                rngs=nnx.Rngs(0),
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
