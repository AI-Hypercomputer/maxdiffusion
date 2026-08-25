# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
