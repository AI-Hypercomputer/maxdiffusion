# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2VideoGemmaTextEncoder
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, FlowMatchSchedulerState
from maxdiffusion.pyconfig import HyperParameters


class MockTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "</s>"
    
    def __call__(self, text, **kwargs):
        # Return mock input_ids and attention_mask
        class MockBatchEncoding:
            def __init__(self):
                self.input_ids = np.ones((len(text), 128), dtype=np.int32)
                self.attention_mask = np.ones((len(text), 128), dtype=np.int32)
        return MockBatchEncoding()

class LTX2PipelineTest(unittest.TestCase):
    
    def setUp(self):
        # Setup dummy components
        self.rng = nnx.Rngs(0)
        
        # Transformer
        self.transformer = LTX2VideoTransformer3DModel(
            rngs=self.rng,
            in_channels=4,
            out_channels=4,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=16,
            audio_num_attention_heads=2,
            audio_attention_head_dim=2, 
            audio_cross_attention_dim=4,
            num_layers=1,
            caption_channels=16
        )
        
        # VAE
        self.vae = LTX2VideoAutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            layers_per_block=(1,),
            rngs=self.rng
        )
        
        # Text Encoder Connector
        self.text_encoder_connector = LTX2VideoGemmaTextEncoder(
            gemma_dim=16,
            gemma_layers=5,
            projection_dim=16,
            connector_heads=2,
            connector_head_dim=8,
            connector_layers=1,
            rngs=self.rng
        )
        
        # MaxText Gemma3 Feature Extractor
        self.devices_array = np.array(jax.devices())
        self.mesh = jax.sharding.Mesh(self.devices_array.reshape(1, -1), ('data', 'model'))
        
        try:
            from maxdiffusion.pipelines.ltx2.ltx2_pipeline import MaxTextGemma3FeatureExtractor
            from MaxText.configs import types
            from MaxText import pyconfig
            
            # Use real MaxText Config to avoid missing attributes
            raw_config = types.MaxTextConfig(
                vocab_size=32000,
                model_name="gemma3-4b",
                base_emb_dim=16,
                base_num_query_heads=2,
                base_num_kv_heads=1,
                head_dim=8,
                num_decoder_layers=5,
                max_prefill_predict_length=512,
                max_target_length=1024,
                per_device_batch_size=1,
                dtype=jnp.float32,
                weight_dtype=jnp.float32,
                rope_embedding_dims=16,
                # Add other overrides as needed for the test to be lightweight
            )
            config = pyconfig.HyperParameters(raw_config)
                
            self.text_encoder = MaxTextGemma3FeatureExtractor(
                config=config,
                mesh=self.mesh,
                quant=None,
                rngs=self.rng
            )
        except ImportError:
            self.text_encoder = None
            print("MaxText not found, text_encoder set to None")
        
        # Scheduler
        self.scheduler = FlaxFlowMatchScheduler(num_train_timesteps=10)
        self.scheduler_state = self.scheduler.create_state()
        
        # Tokenizer
        self.tokenizer = MockTokenizer()
        
        # Config (Dummy)
        self.config = HyperParameters()
        
        # Test Mesh (Already defined above)


    def test_pipeline_initialization(self):
        pipeline = LTX2Pipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text_encoder_connector=self.text_encoder_connector,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            scheduler_state=self.scheduler_state,
            devices_array=self.devices_array,
            mesh=self.mesh,
            config=self.config
        )
        self.assertIsInstance(pipeline, LTX2Pipeline)

    def test_pipeline_call(self):
        pipeline = LTX2Pipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text_encoder_connector=self.text_encoder_connector,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            scheduler_state=self.scheduler_state,
            devices_array=self.devices_array,
            mesh=self.mesh,
            config=self.config
        )
        
        # Run pipeline with small parameters
        output = pipeline(
            prompt="A cat dancing",
            height=32,
            width=32,
            num_frames=9, # 9 frames -> latents (9-1)//8 + 1 = 2 frames
            num_inference_steps=1,
            num_videos_per_prompt=1
        )
        
        # Verify output shape
        # Output should be (B, C, F, H, W) or formatted similar to Diffusers
        # My pipeline currently returns `video` from `vae.decode`
        # `vae.decode` returns samples.
        # Check shape
        self.assertIsNotNone(output)
        # Expected frames: 9
        # Expected height/width: 32
        # Shape: (1, 3, 9, 32, 32) (if Diffusers style) or (1, 9, 32, 32, 3,...)
        # Let's inspect output shape in actual running test
        print(f"Output shape: {output.shape}")

    def test_pipeline_call_i2v(self):
        pipeline = LTX2Pipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text_encoder_connector=self.text_encoder_connector,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            scheduler_state=self.scheduler_state,
            devices_array=self.devices_array,
            mesh=self.mesh,
            config=self.config
        )
        
        # Create dummy image
        batch = 1
        height = 32
        width = 32
        # Image shape: (B, H, W, C)
        image = np.random.rand(batch, height, width, 3).astype(np.float32)
        
        output = pipeline(
            prompt="A cat dancing",
            image=image,
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=1,
            num_videos_per_prompt=1
        )
        self.assertIsNotNone(output)
        print(f"I2V Output shape: {output.shape}")

    def test_pipeline_call_no_cfg(self):
        pipeline = LTX2Pipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text_encoder_connector=self.text_encoder_connector,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            scheduler_state=self.scheduler_state,
            devices_array=self.devices_array,
            mesh=self.mesh,
            config=self.config
        )
        
        output = pipeline(
            prompt="A cat dancing",
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=1,
            guidance_scale=1.0, # Disable CFG
            num_videos_per_prompt=1
        )
        self.assertIsNotNone(output)
        print(f"No-CFG Output shape: {output.shape}")

    def test_real_gemma3_import(self):
        """
        Verifies that we can import MaxText Gemma3 using the pipeline's strategy.
        """
        gemma3 = None
        common_types = None
        try:
            import maxtext.models.gemma3 as gemma3
            from MaxText import common_types
            print("Successfully imported maxtext.models.gemma3 and MaxText.common_types")
        except ImportError as e:
             print(f"Could not import MaxText components: {e}")
        
        self.assertIsNotNone(gemma3)
        self.assertIsNotNone(common_types)

    def test_gemma3_feature_extractor(self):
        """
        Verifies MaxTextGemma3FeatureExtractor instantiation and forward pass.
        """
        try:
             from maxdiffusion.pipelines.ltx2.ltx2_pipeline import MaxTextGemma3FeatureExtractor
             from MaxText import common_types
        except ImportError:
            print("Skipping test_gemma3_feature_extractor: MaxText not found or pipeline import failed")
            return

        class DummyConfig:
            # Feature Extractor Config
            emb_dim = 64 # Small for test
            num_query_heads = 4
            num_kv_heads = 4
            head_dim = 16
            max_target_length = 128
            max_prefill_predict_length = 128
            attention = "dot_product" 
            dropout_rate = 0.0
            float32_qk_product = False
            float32_logits = False
            prefill_cache_axis_order = "0,1,2,3"
            ar_cache_axis_order = "0,1,2,3"
            compute_axis_order = "0,1,2,3"
            reshape_q = False
            use_mrope = False
            mrope_section = None
            mlp_dim = 128
            mlp_activations = ["gelu"]
            record_internal_nn_metrics = False
            scan_layers = False 
            vocab_size = 1000
            dtype = jnp.float32
            weight_dtype = jnp.float32
            normalization_layer_epsilon = 1e-6
            shard_mode = common_types.ShardMode.AUTO
            debug_sharding = False
            
            # Gemma3 specific
            sliding_window_size = 128
            attn_logits_soft_cap = 50.0
            use_post_attn_norm = True
            use_post_ffw_norm = True
            num_decoder_layers = 2 # Small for test
            model_name = "gemma3-12b" # Required for some logic
            base_emb_dim = 64 # For consistency
            base_num_query_heads = 4
            
            # Quant
            quantization_local_shard_count = 1
            
        config = DummyConfig()
        mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ('data', 'model'))
        rngs = nnx.Rngs(0)
        
        # Instantiate
        feature_extractor = MaxTextGemma3FeatureExtractor(config, mesh, rngs=rngs)
        
        # Run
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        outputs = feature_extractor(input_ids, output_hidden_states=True)
        
        # Verify
        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.hidden_states), 4)
        print("MaxTextGemma3FeatureExtractor test passed!")

if __name__ == '__main__':
    unittest.main()
