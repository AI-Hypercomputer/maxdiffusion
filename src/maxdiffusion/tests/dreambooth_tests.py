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
import os
import unittest
from absl.testing import absltest
import shutil
from ..import pyconfig
import jax
import tensorflow as tf
from maxdiffusion.dreambooth.train_dreambooth import prepare_w_prior_preservation

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def cleanup(output_dir):
  shutil.rmtree(output_dir)

class DreamBoothTest(unittest.TestCase):
    """Test DreamBooth"""

    def setUp(self):
        DreamBoothTest.dummy_data = {}

    def test_prior_preservation(self):
        """Test prior preservation function generates images."""

        pyconfig.initialize([None, os.path.join(THIS_DIR,'..','configs','base15.yml'),
            "class_data_dir=/tmp/class_data_dir","class_prompt=a photo of a dog",
            "num_class_images=100"])
        config = pyconfig.config
        rng = jax.random.key(config.seed)
        prepare_w_prior_preservation(rng, config)
        image_count = len(tf.io.gfile.glob(f"{config.class_data_dir}/*.jpg"))
        assert image_count == 100

        #cleanup(config.class_data_dir)

if __name__ == '__main__':
  absltest.main()
