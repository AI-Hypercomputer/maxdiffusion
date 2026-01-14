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
from functools import partial
import pathlib
import shutil
import subprocess
import unittest
import pytest
from absl.testing import absltest
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .. import pyconfig
from .. import max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import (
    make_data_iterator,
    make_dreambooth_train_iterator,
)


from skimage.metrics import structural_similarity as ssim
from PIL import Image

from maxdiffusion import (FlaxStableDiffusionPipeline, FlaxStableDiffusionXLPipeline)
from maxdiffusion.models import FlaxAutoencoderKL
from maxdiffusion.maxdiffusion_utils import (encode, tokenize_captions, encode_xl, tokenize_captions_xl)

from maxdiffusion.maxdiffusion_utils import vae_apply, transform_images

from maxdiffusion.dreambooth.dreambooth_constants import (
    INSTANCE_IMAGE_LATENTS,
    INSTANCE_PROMPT_INPUT_IDS,
    CLASS_IMAGE_LATENTS,
    CLASS_PROMPT_INPUT_IDS,
)

HOME_DIR = pathlib.Path.home()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = str(HOME_DIR / ".cache" / "huggingface" / "datasets")


def cleanup(output_dir):
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)


class InputPipelineInterface(unittest.TestCase):
  """Test Unet sharding"""

  def setUp(self):
    InputPipelineInterface.dummy_data = {}

  def test_make_dreambooth_train_iterator(self):

    instance_class_gcs_dir = "gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/instance_class"
    class_class_gcs_dir = "gs://maxdiffusion-github-runner-test-assets/datasets/dreambooth/class_class"
    local_dir = "/tmp/"
    instance_class_local_dir = max_utils.download_blobs(instance_class_gcs_dir, local_dir)
    class_class_local_dir = max_utils.download_blobs(class_class_gcs_dir, local_dir)

    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base14.yml"),
            "cache_latents_text_encoder_outputs=True",
            "dataset_name=my_dreambooth_dataset",
            "transform_images_num_proc=1",
            f"instance_data_dir={instance_class_local_dir}",
            f"class_data_dir={class_class_local_dir}",
            "instance_prompt=photo of ohwx dog",
            "class_prompt=photo of dog",
        ],
        unittest=True,
    )
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        revision=config.revision,
        dtype=config.activations_dtype,
        safety_checker=None,
        feature_extractor=None,
        from_pt=config.from_pt,
    )

    train_iterator = make_dreambooth_train_iterator(
        config, mesh, global_batch_size, pipeline.tokenizer, pipeline.vae, params["vae"]
    )

    data = next(train_iterator)
    device_count = jax.device_count()

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    instance_hidden_states = data[0][INSTANCE_PROMPT_INPUT_IDS]
    class_hidden_states = data[1][CLASS_PROMPT_INPUT_IDS]

    assert instance_hidden_states.shape == (device_count, 77)
    assert class_hidden_states.shape == (device_count, 77)

    assert data[0][INSTANCE_IMAGE_LATENTS].shape == (
        device_count,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )
    assert data[1][CLASS_IMAGE_LATENTS].shape == (
        device_count,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )

    cleanup(instance_class_local_dir)
    cleanup(class_class_local_dir)

  def test_make_pokemon_hf_iterator(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_2_base.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-2-base",
            "dataset_name=diffusers/pokemon-gpt4-captions",
            "from_pt=False",
            "dataset_type=hf",
        ],
        unittest=True,
    )
    config = pyconfig.config

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )
      p_encode = None
      p_vae_apply = None
      rng = None
      tokenize_fn = partial(
          tokenize_captions, caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode
      )
      image_transforms_fn = partial(
          transform_images,
          image_column=config.image_column,
          image_resolution=config.resolution,
          rng=rng,
          global_batch_size=global_batch_size,
          p_vae_apply=p_vae_apply,
      )

      train_iterator = make_data_iterator(
          config, jax.process_index(), jax.process_count(), mesh, global_batch_size, tokenize_fn, image_transforms_fn
      )
      data = next(train_iterator)
      device_count = jax.device_count()

    assert data["input_ids"].shape == (device_count, 77)
    assert data["pixel_values"].shape == (device_count, 3, config.resolution, config.resolution)

  def test_make_pokemon_hf_iterator_sdxl(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "per_device_batch_size=1",
            "dataset_name=diffusers/pokemon-gpt4-captions",
            "dataset_type=hf",
        ],
        unittest=True,
    )
    config = pyconfig.config

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )
      p_encode = None
      p_vae_apply = None
      rng = None
      tokenize_fn = partial(
          tokenize_captions_xl,
          caption_column=config.caption_column,
          tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
          p_encode=p_encode,
      )
      image_transforms_fn = partial(
          transform_images,
          image_column=config.image_column,
          image_resolution=config.resolution,
          rng=rng,
          global_batch_size=global_batch_size,
          p_vae_apply=p_vae_apply,
      )

      train_iterator = make_data_iterator(
          config, jax.process_index(), jax.process_count(), mesh, global_batch_size, tokenize_fn, image_transforms_fn
      )
      data = next(train_iterator)
      device_count = jax.device_count()

    assert data["input_ids"].shape == (device_count, 2, 77)
    assert data["pixel_values"].shape == (device_count, 3, config.resolution, config.resolution)

  def test_make_pokemon_tf_iterator_cache(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_2_base.yml"),
            "cache_latents_text_encoder_outputs=True",
            "dataset_name=diffusers/pokemon-gpt4-captions",
            "dataset_type=tf",
        ],
        unittest=True,
    )
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )
      rng = jax.random.PRNGKey(config.seed)
      p_encode = None
      p_vae_apply = None
      if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(partial(encode, text_encoder=pipeline.text_encoder, text_encoder_params=params["text_encoder"]))
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
      tokenize_fn = partial(
          tokenize_captions, caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode
      )
      image_transforms_fn = partial(
          transform_images,
          image_column=config.image_column,
          image_resolution=config.resolution,
          rng=rng,
          global_batch_size=global_batch_size,
          p_vae_apply=p_vae_apply,
      )

      train_iterator = make_data_iterator(
          config, jax.process_index(), jax.process_count(), mesh, global_batch_size, tokenize_fn, image_transforms_fn
      )
      data = next(train_iterator)
      device_count = jax.device_count()

      vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
      encoder_hidden_states = data["input_ids"]

    assert encoder_hidden_states.shape == (device_count, 77, 1024)
    assert data["pixel_values"].shape == (
        device_count,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )

  def test_make_pokemon_iterator_no_cache(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_2_base.yml"),
            "cache_latents_text_encoder_outputs=False",
            "tokenize_captions_num_proc=1",
            "transform_images_num_proc=1",
            "dataset_name=diffusers/pokemon-gpt4-captions",
            "dataset_type=tf",
        ],
        unittest=True,
    )
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )
      rng = jax.random.PRNGKey(config.seed)
      p_encode = None
      p_vae_apply = None
      if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(partial(encode, text_encoder=pipeline.text_encoder, text_encoder_params=params["text_encoder"]))
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
      tokenize_fn = partial(
          tokenize_captions, caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode
      )
      image_transforms_fn = partial(
          transform_images,
          image_column=config.image_column,
          image_resolution=config.resolution,
          rng=rng,
          global_batch_size=global_batch_size,
          p_vae_apply=p_vae_apply,
      )

      train_iterator = make_data_iterator(
          config, jax.process_index(), jax.process_count(), mesh, global_batch_size, tokenize_fn, image_transforms_fn
      )
      data = next(train_iterator)
      device_count = jax.device_count()

    encoder_hidden_states = data["input_ids"]
    assert encoder_hidden_states.shape == (device_count, 77)
    assert data["pixel_values"].shape == (device_count, 3, config.resolution, config.resolution)

  def test_make_pokemon_iterator_sdxl_cache(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_xl.yml"),
            "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
            "cache_latents_text_encoder_outputs=True",
            "per_device_batch_size=1",
            "dataset_name=diffusers/pokemon-gpt4-captions",
            "dataset_type=tf",
        ],
        unittest=True,
    )
    config = pyconfig.config

    cleanup(config.dataset_save_location)

    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )
      rng = jax.random.PRNGKey(config.seed)
      p_encode = None
      p_vae_apply = None
      if config.cache_latents_text_encoder_outputs:
        p_encode = jax.jit(
            partial(
                encode_xl,
                text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
                text_encoder_params=[params["text_encoder"], params["text_encoder_2"]],
            )
        )
        p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
      tokenize_fn = partial(
          tokenize_captions_xl,
          caption_column=config.caption_column,
          tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
          p_encode=p_encode,
      )
      image_transforms_fn = partial(
          transform_images,
          image_column=config.image_column,
          image_resolution=config.resolution,
          rng=rng,
          global_batch_size=global_batch_size,
          p_vae_apply=p_vae_apply,
      )

      train_iterator = make_data_iterator(
          config, jax.process_index(), jax.process_count(), mesh, global_batch_size, tokenize_fn, image_transforms_fn
      )
      data = next(train_iterator)
      device_count = jax.device_count()

      vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

      prompt_embeds = data["prompt_embeds"]
      text_embeds = data["text_embeds"]
    assert prompt_embeds.shape == (device_count, 77, 2048)
    assert text_embeds.shape == (device_count, 1280)
    assert data["pixel_values"].shape == (
        device_count,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )

  @pytest.mark.skip("This test is deprecated and will be removed in a future version. Reason: stable diffusion 2 base is no longer in HuggingFace")
  def test_make_laion_grain_iterator(self):
    try:
      subprocess.check_output(
          [
              "bash",
              "setup_gcsfuse.sh",
              "DATASET_GCS_BUCKET=maxdiffusion-github-runner-test-assets",
              "MOUNT_PATH=/tmp/gcsfuse",
          ],
          stderr=subprocess.STDOUT,
      )
    except subprocess.CalledProcessError as e:
      raise ValueError(f"setup_gcsfuse failed with error: {e.output}") from e
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_2_base.yml"),
            "grain_train_files=/tmp/gcsfuse/datasets/array-record/laion400m/tf_records_512_encoder_state_fp32/*.arrayrecord",
            "dataset_type=grain",
        ],
        unittest=True,
    )
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, _ = FlaxStableDiffusionPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )

      train_iterator = make_data_iterator(config, jax.process_index(), jax.process_count(), mesh, global_batch_size)
      data = next(train_iterator)
      device_count = jax.device_count()

      vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
      encoder_hidden_states = data["input_ids"]

      # TODO - laion dataset was prepared with an extra dim.
      # need to preprocess the dataset with dim removed.
      if len(encoder_hidden_states.shape) == 4:
        encoder_hidden_states = jnp.squeeze(encoder_hidden_states)

    assert encoder_hidden_states.shape == (device_count, 77, 1024)
    assert data["pixel_values"].shape == (
        config.total_train_batch_size,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
        8,
    )
    
  @pytest.mark.skip("This test is deprecated and will be removed in a future version. Reason: stable diffusion 2 base is no longer in HuggingFace")
  def test_make_laion_tfrecord_iterator(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "base_2_base.yml"),
            "train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/raw_data/tf_records_512_encoder_state_fp32",
            "dataset_type=tfrecord",
        ],
        unittest=True,
    )
    config = pyconfig.config
    global_batch_size = config.per_device_batch_size * jax.device_count()
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    with mesh:
      pipeline, _ = FlaxStableDiffusionPipeline.from_pretrained(
          config.pretrained_model_name_or_path,
          revision=config.revision,
          dtype=config.activations_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=config.from_pt,
      )

      feature_description = {
          "moments": tf.io.FixedLenFeature([], tf.string),
          "clip_embeddings": tf.io.FixedLenFeature([], tf.string),
      }

      def _parse_tfrecord_fn(example):
        return tf.io.parse_single_example(example, feature_description)

      train_iterator = make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          global_batch_size,
          feature_description=feature_description,
          prepare_sample_fn=_parse_tfrecord_fn,
      )
      data = next(train_iterator)
      device_count = jax.device_count()

      vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
      encoder_hidden_states = data["input_ids"]

      # TODO - laion dataset was prepared with an extra dim.
      # need to preprocess the dataset with dim removed.
      if len(encoder_hidden_states.shape) == 4:
        encoder_hidden_states = jnp.squeeze(encoder_hidden_states)

    assert encoder_hidden_states.shape == (device_count, 77, 1024)
    assert data["pixel_values"].shape == (
        config.total_train_batch_size,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
        8,
    )

  def test_tfrecord(self):
    """Validate latents match a deterministic output image"""

    image_feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "hidden_states": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
      # Parse the input tf.train.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, image_feature_description)

    @tf.function()
    def prepare_sample(features):
      latents = tf.io.parse_tensor(tnp.asarray(features["latents"]), out_type=tf.float32)
      hidden_states = tf.io.parse_tensor(tnp.asarray(features["hidden_states"]), out_type=tf.float32)
      return {"pixel_values": latents, "input_ids": hidden_states}

    raw_image_dataset = tf.data.TFRecordDataset("gs://maxdiffusion-github-runner-test-assets/tfrecords/file_00-1000.tfrec")
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function).map(prepare_sample).batch(4)

    iterator = iter(parsed_image_dataset)
    image_features = iterator.get_next()
    latents = image_features["pixel_values"]
    vae, params = FlaxAutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-base",
        subfolder="vae",
        from_pt=True,
    )
    latents = latents.numpy()
    latents = 1 / vae.config.scaling_factor * latents

    image = vae.apply({"params": params}, latents, method=vae.decode).sample

    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    image = np.array(image)
    test_image = image[0]
    test_image = (test_image * 255).round().astype("uint8")

    img_url = os.path.join(THIS_DIR, "images", "latent_test.png")
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image, multichannel=True, channel_axis=-1, data_range=255)

    assert base_image.shape == test_image.shape
    assert ssim_compare >= 0.70


if __name__ == "__main__":
  absltest.main()
