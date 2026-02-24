import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.ltx2_utils import load_transformer_weights, load_vae_weights
from maxdiffusion.models.modeling_flax_pytorch_utils import validate_flax_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict

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
                decoder_block_out_channels=(256, 512, 1024),
                layers_per_block=(4, 6, 6, 2, 2),
                decoder_layers_per_block=(5, 5, 5, 5),
                spatio_temporal_scaling=(True, True, True, True),
                decoder_spatio_temporal_scaling=(True, True, True),
                decoder_inject_noise=(False, False, False, False),
                downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                upsample_residual=(True, True, True),
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
        filtered_eval_shapes = {}
        flat_eval_shapes = flatten_dict(eval_shapes)
        for k, v in flat_eval_shapes.items():
            k_str = [str(x) for x in k]
            if "dropout" in k_str or "rngs" in k_str:
                continue
            filtered_eval_shapes[k] = v
            
        from flax.traverse_util import unflatten_dict
        validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flatten_dict(loaded_weights))
        print("VAE Weights Validated Successfully!")

    def test_load_vocoder_weights(self):
        from maxdiffusion.models.ltx2.vocoder_ltx2 import LTX2Vocoder
        from maxdiffusion.models.ltx2.ltx2_utils import load_vocoder_weights
        
        pretrained_model_name_or_path = "Lightricks/LTX-2"
        
        config = {
          "hidden_channels": 1024,
          "in_channels": 128,
          "leaky_relu_negative_slope": 0.1,
          "out_channels": 2,
          "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
          "resnet_kernel_sizes": [3, 7, 11],
          "upsample_factors": [6, 5, 2, 2, 2],
          "upsample_kernel_sizes": [16, 15, 8, 4, 4],
          "rngs": nnx.Rngs(0)
        }
        with jax.default_device(jax.devices("cpu")[0]):
            model = LTX2Vocoder(**config)
        state = nnx.state(model)
        eval_shapes = state.to_pure_dict()
        
        print("Loading Vocoder Weights...")
        loaded_weights = load_vocoder_weights(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            eval_shapes=eval_shapes,
            device=self.device,
            hf_download=True
        )
            
        # Validate
        print("Validating Vocoder Weights...")
        validate_flax_state_dict(eval_shapes, flatten_dict(loaded_weights))
        print("Vocoder Weights Validated Successfully!")

    def test_load_connector_weights(self):
        from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
        from maxdiffusion.models.ltx2.ltx2_utils import load_connector_weights
        
        pretrained_model_name_or_path = "Lightricks/LTX-2"
        
        with jax.default_device(jax.devices("cpu")[0]):
            model = LTX2AudioVideoGemmaTextEncoder(rngs=self.rngs)
            
        state = nnx.state(model)
        eval_shapes = state.to_pure_dict()
        
        print("Loading Connector Weights...")
        loaded_weights = load_connector_weights(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            eval_shapes=eval_shapes,
            device=self.device,
            hf_download=True
        )
        
        print("Validating Connector Weights...")
        validate_flax_state_dict(eval_shapes, flatten_dict(loaded_weights))
        print("Connector Weights Validated Successfully!")

    def test_load_audio_vae_weights(self):
        from maxdiffusion.models.ltx2.audio_vae import FlaxAutoencoderKLLTX2Audio
        from maxdiffusion.models.ltx2.ltx2_utils import load_audio_vae_weights
        
        pretrained_model_name_or_path = "Lightricks/LTX-2"
        
        # Audio VAE Config from user request
        config = {
          "base_channels": 128,
          "ch_mult": (1, 2, 4),
          "double_z": True,
          "dropout": 0.0,
          "in_channels": 2,
          "latent_channels": 8,
          "mel_bins": 64,
          "mel_hop_length": 160,
          "mid_block_add_attention": False,
          "norm_type": "pixel",
          "num_res_blocks": 2,
          "output_channels": 2,
          "resolution": 256,
          "sample_rate": 16000,
          "rngs": nnx.Rngs(0)
        }
        
        with jax.default_device(jax.devices("cpu")[0]):
            model = FlaxAutoencoderKLLTX2Audio(**config)
            
        state = nnx.state(model)
        eval_shapes = state.to_pure_dict()
        
        print("Loading Audio VAE Weights...")
        loaded_weights = load_audio_vae_weights(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            eval_shapes=eval_shapes,
            device=self.device,
            hf_download=True
        )
        
        print("Validating Audio VAE Weights...")
        # Filter eval_shapes for validation as load_audio_vae_weights returns filtered weights
        filtered_eval_shapes = {}
        flat_eval = flatten_dict(eval_shapes)
        for k, v in flat_eval.items():
            k_str = [str(x) for x in k]
            is_stat = False
            for ks in k_str:
                if "dropout" in ks or "rngs" in ks:
                    is_stat = True
                    break
            if not is_stat:
                filtered_eval_shapes[k] = v
                
        validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flatten_dict(loaded_weights))
        print("Audio VAE Weights Validated Successfully!")

if __name__ == "__main__":
    unittest.main()
