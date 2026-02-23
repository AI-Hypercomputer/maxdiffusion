
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

class LTX2VideoConfig:
    def __init__(self):
        self.in_channels = 128
        self.out_channels = 128
        self.patch_size = 1
        self.patch_size_t = 1
        self.num_attention_heads = 32
        self.attention_head_dim = 128
        self.cross_attention_dim = 4096
        self.audio_in_channels = 128
        self.audio_out_channels = 128
        self.audio_patch_size = 1
        self.audio_patch_size_t = 1
        self.audio_num_attention_heads = 32
        self.audio_attention_head_dim = 128 # Default is 64 but we want 128
        self.audio_cross_attention_dim = 2048
        self.num_layers = 48

class LTX2UtilsTest(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.rngs = nnx.Rngs(42)
        
    def test_load_transformer_weights(self):
        
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
                cross_attention_dim=4096, # T5-XXL uses 4096
                audio_in_channels=self.config.audio_in_channels,
                audio_out_channels=self.config.audio_out_channels,
                audio_patch_size=self.config.audio_patch_size,
                audio_patch_size_t=self.config.audio_patch_size_t,
                audio_num_attention_heads=self.config.audio_num_attention_heads,
                audio_attention_head_dim=64, # Match Checkpoint (2048 / 32)
                audio_cross_attention_dim=self.config.audio_cross_attention_dim,
                num_layers=self.config.num_layers,
                scan_layers=True,
                rngs=nnx.Rngs(0),
            )
        state = nnx.state(self.transformer)
        
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
        from flax.traverse_util import flatten_dict
        validate_flax_state_dict(eval_shapes, flatten_dict(loaded_weights))
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
        # Filter out dropout/rngs keys from eval_shapes as they are not expected in weights
        filtered_eval_shapes = {}
        flat_eval_shapes = flatten_dict(eval_shapes)
        for k, v in flat_eval_shapes.items():
            k_str = [str(x) for x in k]
            if "dropout" in k_str or "rngs" in k_str:
                continue
            filtered_eval_shapes[k] = v
            
        validate_flax_state_dict(filtered_eval_shapes, flatten_dict(loaded_weights))
        print("VAE Weights Validated Successfully!")

if __name__ == "__main__":
    unittest.main()
