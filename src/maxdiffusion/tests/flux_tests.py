"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

'''This script is used an example of how to shard the UNET on TPU.'''

import os
import unittest
from absl.testing import absltest
import jax
from ..models.normalization_flax import FlaxAdaLayerNormZeroSingle

class FluxTests(unittest.TestCase):
    def test_adalayernormzerosingle(self):
       ada_layer = FlaxAdaLayerNormZeroSingle(embedding_dim=128)
       x = jax.random.normal(jax.random.key(0), (2,128))
       params = ada_layer.init({"params" : jax.random.key(0)}, x, x)["params"]
       x, y = ada_layer.apply({"params" : params["params"]}, x, x)
       


if __name__ == '__main__':
    absltest.main()
