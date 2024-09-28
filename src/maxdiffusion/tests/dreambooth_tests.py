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
import numpy as np
from PIL import Image
from .. import pyconfig
from .. import max_utils
from maxdiffusion.generate import run as generate_run
from skimage.metrics import structural_similarity as ssim
import jax
import tensorflow as tf
from maxdiffusion.dreambooth.train_dreambooth import (prepare_w_prior_preservation, train as train_dreambooth)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def cleanup(output_dir):
  shutil.rmtree(output_dir)


class DreamBooth(unittest.TestCase):
  """Test DreamBooth"""

  def setUp(self):
    DreamBooth.dummy_data = {}

  def test_prior_preservation(self):
    """Test prior preservation function generates images."""

    num_class_images = 16
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base14.yml"),
            "class_data_dir=/tmp/class_data_dir",
            "class_prompt=a photo of a dog",
            f"num_class_images={num_class_images}",
        ]
    )

    config = pyconfig.config
    rng = jax.random.key(config.seed)
    prepare_w_prior_preservation(rng, config)
    image_count = len(tf.io.gfile.glob(f"{config.class_data_dir}/*.jpg"))
    assert image_count == num_class_images

    cleanup(config.class_data_dir)

  def test_dreambooth_training(self):
    """Test full dreambooth training"""
    num_class_images = 100
    output_dir = "train-dreambooth-smoke-test"
    run_name = "dreambooth_smoke_test"
    cache_dir = "gs://maxdiffusion-github-runner-test-assets/cache_dir"
    instance_class_gcs_dir = "gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/instance_class"
    class_class_gcs_dir = "gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/class_class"
    local_dir = "/tmp/"

    instance_class_local_dir = max_utils.download_blobs(instance_class_gcs_dir, local_dir)
    class_class_local_dir = max_utils.download_blobs(class_class_gcs_dir, local_dir)

    # calling train_dreambooth directly bypasses setting the cache dir
    # so setting it here.
    jax.config.update("jax_compilation_cache_dir", cache_dir)

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base14.yml"),
            "class_data_dir=test_dreambooth",
            f"instance_data_dir={instance_class_local_dir}",
            f"class_data_dir={class_class_local_dir}",
            "instance_prompt=a photo of ohwx dog",
            "class_prompt=photo of a dog",
            "max_train_steps=150",
            f"cache_dir={cache_dir}",
            "class_prompt=a photo of a dog",
            "activations_dtype=bfloat16",
            "weights_dtype=float32",
            "per_device_batch_size=1",
            "enable_profiler=False",
            "precision=DEFAULT",
            "cache_dreambooth_dataset=False",
            "learning_rate=4e-6",
            f"output_dir={output_dir}",
            f"num_class_images={num_class_images}",
            f"run_name={run_name}",
            "base_output_directory=gs://maxdiffusion-github-runner-test-assets",
        ]
    )

    train_dreambooth(pyconfig.config)

    img_url = os.path.join(THIS_DIR, "images", "dreambooth_test.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base14.yml"),
            "prompt=a photo of a ohwx dog",
            "revision=main",
            f"pretrained_model_name_or_path={output_dir}/{run_name}/checkpoints/final",
        ]
    )

    images = generate_run(pyconfig.config)
    test_image = np.array(images[1]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)
    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.70

    cleanup(output_dir)
    cleanup(instance_class_local_dir)
    cleanup(class_class_local_dir)


if __name__ == "__main__":
  absltest.main()
