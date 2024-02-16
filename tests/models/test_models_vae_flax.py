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

import unittest

from maxdiffusion import FlaxAutoencoderKL
from maxdiffusion.utils import is_flax_available
from maxdiffusion.utils.testing_utils import require_flax

from .test_modeling_common_flax import FlaxModelTesterMixin


if is_flax_available():
    import jax


@require_flax
class FlaxAutoencoderKLTests(FlaxModelTesterMixin, unittest.TestCase):
    model_class = FlaxAutoencoderKL

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        prng_key = jax.random.PRNGKey(0)
        image = jax.random.uniform(prng_key, ((batch_size, num_channels) + sizes))

        return {"sample": image, "prng_key": prng_key}

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict
