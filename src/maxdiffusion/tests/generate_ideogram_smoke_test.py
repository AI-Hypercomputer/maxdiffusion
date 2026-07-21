import unittest

from flax import nnx

from maxdiffusion.models.ideogram.transformer_ideogram import Ideogram4Transformer, Ideogram4Config
from maxdiffusion.models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams


class TestIdeogram(unittest.TestCase):

  def test_instantiate_transformer(self):
    rngs = nnx.Rngs(0)
    config = Ideogram4Config(emb_dim=128, num_heads=2, in_channels=64, llm_features_dim=128, adanln_dim=128, num_layers=2)
    model = Ideogram4Transformer(rngs, config)

    self.assertIsNotNone(model)

  def test_instantiate_autoencoder(self):
    rngs = nnx.Rngs(0)
    params = AutoEncoderParams(resolution=32, in_channels=3, ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, z_channels=8)
    ae = AutoEncoder(rngs, params)
    self.assertIsNotNone(ae)


if __name__ == "__main__":
  unittest.main()
