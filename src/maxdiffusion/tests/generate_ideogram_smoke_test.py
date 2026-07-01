import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.ideogram.transformer_ideogram import Ideogram4Transformer
from maxdiffusion.models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline

class DummyConfig:
    emb_dim = 128
    num_heads = 4
    in_channels = 16
    llm_features_dim = 128
    adanln_dim = 128
    rope_theta = 10000
    mrope_section = (16, 8, 8)
    intermediate_size = 512
    norm_eps = 1e-6
    num_layers = 2
    patch_size = 2

class TestIdeogram(unittest.TestCase):
    def test_instantiate_transformer(self):
        rngs = nnx.Rngs(0)
        config = DummyConfig()
        model = Ideogram4Transformer(rngs, config)
        self.assertIsNotNone(model)

    def test_instantiate_autoencoder(self):
        rngs = nnx.Rngs(0)
        params = AutoEncoderParams(
            resolution=32,
            in_channels=3,
            ch=32,
            out_ch=3,
            ch_mult=(1, 2),
            num_res_blocks=1,
            z_channels=8
        )
        ae = AutoEncoder(rngs, params)
        self.assertIsNotNone(ae)

if __name__ == '__main__':
    unittest.main()
