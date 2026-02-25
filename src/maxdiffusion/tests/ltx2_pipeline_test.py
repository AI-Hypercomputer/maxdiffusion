
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from jax.sharding import Mesh
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline, LTX2PipelineOutput
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler

class LTX2PipelineTest(unittest.TestCase):
    def setUp(self):
        # Initialize pyconfig (needed for some defaults, though we override mostly)
        pyconfig.initialize([None, "src/maxdiffusion/configs/ltx2_video.yml"], unittest=True)
        config = pyconfig.config
        devices_array = create_device_mesh(config)
        self.mesh = Mesh(devices_array, config.mesh_axes)
        
        # Mocks
        self.scheduler = Mock(spec=FlaxFlowMatchScheduler)
        self.scheduler.timesteps = jnp.linspace(1, 0, 4)
        self.scheduler.set_timesteps.return_value = None
        
        self.vae = Mock()
        self.vae.spatial_compression_ratio = 8
        self.vae.temporal_compression_ratio = 4
        self.vae.config.scaling_factor = 1.0
        self.vae.latents_mean = [0.0] * 128
        self.vae.latents_std = [1.0] * 128

        self.audio_vae = Mock()
        self.audio_vae.mel_compression_ratio = 4
        self.audio_vae.temporal_compression_ratio = 4
        self.audio_vae.config.sample_rate = 16000
        self.audio_vae.latents_mean = [0.0] * 128
        self.audio_vae.latents_std = [1.0] * 128
        
        self.text_encoder = Mock()
        # Mock text encoder output
        # (B, L, D) = (1, 10, 64)
        # It should return an object with hidden_states attribute which is a list of torch tensors
        self.text_encoder.return_value.hidden_states = [torch.zeros((1, 10, 64)) for _ in range(3)]
        self.text_encoder.device = torch.device("cpu") # Mock device attribute

        self.tokenizer = Mock()
        self.tokenizer.model_max_length = 512
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = None
        self.tokenizer.eos_token = "</s>"
        
        # When tokenizer is called, it returns a dict-like object (BatchEncoding)
        # We need to simulate return_tensors="pt" behavior
        def tokenizer_side_effect(*args, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return MagicMock(
                    input_ids=torch.zeros((1, 10), dtype=torch.long),
                    attention_mask=torch.ones((1, 10), dtype=torch.long)
                )
            else:
                 return MagicMock(
                    input_ids=np.zeros((1, 10), dtype=np.int32),
                    attention_mask=np.ones((1, 10), dtype=np.int32)
                )
        self.tokenizer.side_effect = tokenizer_side_effect
        
        self.vocoder = Mock()
        self.vocoder.return_value = jnp.zeros((1, 16000)) # Dummy waveform

        # Real small NNX models
        rngs = nnx.Rngs(0)
        with self.mesh:
            self.connectors = LTX2AudioVideoGemmaTextEncoder(
                gemma_dim=64,
                gemma_layers=3,
                projection_dim=32,
                connector_heads=1,
                connector_head_dim=32,
                connector_layers=1,
                num_thinking_tokens=8,
                rngs=rngs,
                mesh=self.mesh
            )
            
            self.transformer = LTX2VideoTransformer3DModel(
                rngs=rngs,
                in_channels=128,
                out_channels=128,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=1,
                attention_head_dim=16,
                cross_attention_dim=32,
                audio_in_channels=128,
                audio_out_channels=128,
                audio_num_attention_heads=1,
                audio_attention_head_dim=16,
                audio_cross_attention_dim=32,
                num_layers=1,
                mesh=self.mesh,
                attention_kernel="dot_product"
            )

    def test_pipeline_call(self):
        pipeline = LTX2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer
        )
        
        # Override params to match small models
        pipeline.vae_spatial_compression_ratio = 8
        pipeline.vae_temporal_compression_ratio = 4
        pipeline.transformer_spatial_patch_size = 1
        pipeline.transformer_temporal_patch_size = 1
        
        # Call pipeline
        output = pipeline(
            prompt="test prompt",
            height=32,
            width=32,
            num_frames=8,
            num_inference_steps=4,
            guidance_scale=1.0,
            output_type="latent" # Return latents directly to verify shape
        )
        
        self.assertIsInstance(output, LTX2PipelineOutput)
        self.assertIsInstance(output.frames, jnp.ndarray)
        # Verify shape roughly (packed shape)

    def test_pipeline_call_with_guidance(self):
        pipeline = LTX2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=self.vocoder
        )
         # Override params to match small models
        pipeline.vae_spatial_compression_ratio = 8
        pipeline.vae_temporal_compression_ratio = 4
        pipeline.transformer_spatial_patch_size = 1
        pipeline.transformer_temporal_patch_size = 1
        
        output = pipeline(
            prompt="test prompt",
            height=32,
            width=32,
            num_frames=8,
            num_inference_steps=2,
            guidance_scale=7.5,
            output_type="latent"
        )
        self.assertIsInstance(output, LTX2PipelineOutput)
        self.assertIsInstance(output.frames, jnp.ndarray)
    @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.load_transformer_weights")
    @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.LTX2VideoTransformer3DModel.load_config")
    def test_load_transformer(self, mock_load_config, mock_load_weights):
        # Use real LTX2VideoTransformer3DModel with tiny config
        tiny_config = {
            "in_channels": 4, 
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 1,
            "attention_head_dim": 4,
            "cross_attention_dim": 4,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 1,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 4,
            "num_layers": 1,
            "caption_channels": 4
        }
        mock_load_config.return_value = tiny_config
        
        # Instantiate real model effectively to generate valid weights structure
        rngs = nnx.Rngs(0)
        with self.mesh:
            real_model = LTX2VideoTransformer3DModel(**tiny_config, rngs=rngs)
        
        graphdef, state = nnx.split(real_model)
        flat_state = state.to_flat_dict()
        
        # Create mock weights that match real model structure
        # keys in flat_state are tuples like ('layer', 'kernel')
        # We need to return a dict with same keys but maybe dummy values (or just use the real ones for testing load)
        # load_transformer_weights returns a flat dict of arrays
        
        mock_weights = {}
        for k, v in flat_state.items():
             mock_weights[k] = np.zeros(v.shape, dtype=v.dtype)
             
        mock_load_weights.return_value = mock_weights

        config = pyconfig.config
        
        # Run load_transformer
        pipeline = LTX2Pipeline.load_transformer(
            devices_array=jnp.array(jax.devices()),
            mesh=self.mesh,
            rngs=rngs,
            config=config,
            subfolder="transformer"
        )
        
        # Verify calls
        mock_load_config.assert_called_once()
        mock_load_weights.assert_called_once()
        
        # Verify returned object is LTX2VideoTransformer3DModel
        self.assertIsInstance(pipeline, LTX2VideoTransformer3DModel)

    @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.create_sharded_logical_transformer")
    def test_load_transformer_calls_create(self, mock_create):
        config = pyconfig.config
        rngs = nnx.Rngs(0)
        
        pipeline = LTX2Pipeline.load_transformer(
            devices_array=jnp.array(jax.devices()),
            mesh=self.mesh,
            rngs=rngs,
            config=config,
            subfolder="transformer"
        )
        
        mock_create.assert_called_once()
        self.assertEqual(pipeline, mock_create.return_value)

    def test_check_inputs(self):
        pipeline = LTX2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer
        )
        
        # Test height/width divisibility
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            pipeline.check_inputs("prompt", 100, 100)
            
        # Test prompt vs prompt_embeds
        with self.assertRaisesRegex(ValueError, "Cannot forward both"):
            pipeline.check_inputs("prompt", 128, 128, prompt_embeds=jnp.zeros((1, 10, 64)))
            
        with self.assertRaisesRegex(ValueError, "Provide either"):
            pipeline.check_inputs(None, 128, 128, prompt_embeds=None)
            
        # Test prompt type
        with self.assertRaisesRegex(ValueError, "prompt` has to be of type"):
            pipeline.check_inputs(123, 128, 128)
            
        # Test mask requirements
        with self.assertRaisesRegex(ValueError, "Must provide `prompt_attention_mask`"):
            pipeline.check_inputs(None, 128, 128, prompt_embeds=jnp.zeros((1, 10, 64)))
            
        with self.assertRaisesRegex(ValueError, "Must provide `negative_prompt_attention_mask`"):
             pipeline.check_inputs(None, 128, 128, prompt_embeds=jnp.zeros((1, 10, 64)), prompt_attention_mask=jnp.ones((1, 10)), negative_prompt_embeds=jnp.zeros((1, 10, 64)))

        # Test shape mismatch
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
             pipeline.check_inputs(
                 None, 128, 128, 
                 prompt_embeds=jnp.zeros((1, 10, 64)), 
                 prompt_attention_mask=jnp.ones((1, 10)),
                 negative_prompt_embeds=jnp.zeros((1, 5, 64)), # Mismatch length
                 negative_prompt_attention_mask=jnp.ones((1, 5))
             )
    
    def test_audio_packing_unpacking(self):
        # (Batch, Channels, Length, Mel)
        batch_size = 1
        channels = 128
        length = 32
        mel = 64
        patch_size = 4
        patch_size_t = 1 # Audio typically has patch_size_t=1 in LTX logic, let's test that
        
        latents = jax.random.normal(jax.random.key(0), (batch_size, channels, length, mel))
        
        packed = LTX2Pipeline._pack_audio_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)
        
        # Verify packed shape
        # original logic: (B, T', F', C, p_t, p) -> (B, T' * F', -1)
        # T' = 32 // 1 = 32
        # F' = 64 // 4 = 16
        # shape should be (1, 32 * 16, 128 * 1 * 4) = (1, 512, 512)
        expected_seq_len = (length // patch_size_t) * (mel // patch_size)
        expected_dim = channels * patch_size * patch_size_t
        self.assertEqual(packed.shape, (batch_size, expected_seq_len, expected_dim))
        
        unpacked = LTX2Pipeline._unpack_audio_latents(
            packed, 
            latent_length=length, 
            num_mel_bins=mel, 
            patch_size=patch_size, 
            patch_size_t=patch_size_t
        )
        
        self.assertEqual(unpacked.shape, latents.shape)
        np.testing.assert_allclose(unpacked, latents, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
