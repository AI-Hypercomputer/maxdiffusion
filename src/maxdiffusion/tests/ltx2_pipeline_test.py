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

class MockGemma3(nnx.Module):
    def __init__(self, key=None):
        pass

    def __call__(self, input_ids, attention_mask, output_hidden_states=False):
        # Return mock outputs
        batch, seq_len = input_ids.shape
        hidden_dim = 16 # Small dummy dim
        # Gemma3 output structure likely has 'hidden_states' tuple
        
        class MockOutput:
            def __init__(self):
                # Tuple of hidden states (one per layer)
                # We need enough layers to satisfy FeatureExtractor
                self.hidden_states = tuple([jnp.zeros((batch, seq_len, hidden_dim)) for _ in range(5)])
        
        return MockOutput()

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
            audio_dim=4,
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
            layers_per_block=1,
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
        
        # Mock Gemma3
        self.text_encoder = MockGemma3()
        
        # Scheduler
        self.scheduler = FlaxFlowMatchScheduler(num_train_timesteps=10)
        self.scheduler_state = self.scheduler.create_state()
        
        # Tokenizer
        self.tokenizer = MockTokenizer()
        
        # Config (Dummy)
        self.config = HyperParameters()
        
        # Test Mesh
        devices = jax.devices()
        self.mesh = jax.sharding.Mesh(np.array(devices).reshape(1, -1), ('data', 'model'))
        self.devices_array = np.array(devices)

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

if __name__ == '__main__':
    unittest.main()
